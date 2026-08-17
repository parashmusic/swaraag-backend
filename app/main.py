"""
app/main.py
FastAPI backend for the folk/raga analysis dashboard.

Run:
    uvicorn app.main:app --reload --port 8000

Endpoints:
    GET  /health
    POST /analyze         { "source": "<youtube url>", "borgeet_mode": true }
    POST /analyze/upload   multipart file upload (wav/mp3/etc.)
    GET  /analyze/stream   SSE endpoint for progress-aware analysis

Phase 1 additions:
  - borgeet_mode parameter for Borgeet-specific analysis
  - SSE progress streaming for long-running analysis (Demucs takes 60-90s)
  - Improved temp directory cleanup
"""
import os
import json
import shutil
import tempfile
import traceback
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

from .pipeline.analyze import analyze_track
from .pipeline.raga_db import get_all_ragas
from . import db
from .practice_router import router as practice_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Survesa Folk/Raga Analysis API", version="0.2.0")
app.include_router(practice_router)

# Initialize database on startup
@app.on_event("startup")
def startup_event():
    db.init_db()

# Dev CORS: allow the Vite frontend (default port 5173) and any localhost port.
# Tighten this to your actual frontend origin before deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Absolute path to the backend root
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STATIC_DIR = os.path.join(BACKEND_DIR, "static")

# Ensure static/stems directory exists
os.makedirs(os.path.join(STATIC_DIR, "stems"), exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class AnalyzeRequest(BaseModel):
    source: str                       # YouTube URL or a path already on the server
    is_youtube: bool | None = None
    borgeet_mode: bool = False         # Prioritize Borgeet ragas in matching
    use_separation: bool = True        # Use HTDemucs vocal separation
    tonic_hint: str | None = None      # Optional override for the tonic (e.g., 'D', 'C#')
    time_of_day: str | None = None     # Optional performance time (e.g., 'morning', 'evening', 'night')


class RecalculateRequest(BaseModel):
    tonic_hint: str | None = None
    borgeet_mode: bool = False


class RagaFeedbackRequest(BaseModel):
    actual_raga: str

@app.get("/health")
def health():
    """Health check — also reports whether source separation is available."""
    from .pipeline.analyze import _SEPARATION_AVAILABLE
    return {
        "status": "ok",
        "version": "0.2.0",
        "separation_available": _SEPARATION_AVAILABLE,
    }


@app.get("/ragas")
def get_ragas():
    """Return a list of all supported raga names."""
    ragas = get_all_ragas()
    return {"ragas": sorted(ragas.keys())}


@app.get("/analytics")
def get_analytics():
    """Return aggregated data for the dashboard."""
    return db.get_analytics_data()


@app.get("/dataset")
def get_dataset(
    limit: int = 50, 
    offset: int = 0, 
    predicted_raga: Optional[str] = None, 
    actual_raga: Optional[str] = None, 
    tonic: Optional[str] = None,
    is_correct: Optional[bool] = None
):
    """Return a paginated and filtered list of tracks for the dataset view."""
    return db.get_dataset_paginated(limit, offset, predicted_raga, actual_raga, tonic, is_correct)


@app.get("/projects")
def get_projects(limit: int = 20):
    """Return a list of recent analysis projects."""
    return db.get_recent_projects(limit)


@app.get("/projects/{session_id}")
def get_project(session_id: str):
    """Return the full analysis JSON for a specific project."""
    project = db.get_project(session_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.put("/projects/{session_id}/raga")
def update_project_raga(session_id: str, req: RagaFeedbackRequest):
    """Update the actual_raga field for user feedback."""
    success = db.update_actual_raga(session_id, req.actual_raga)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found or update failed")
    return {"status": "success", "actual_raga": req.actual_raga}


@app.post("/projects/{session_id}/recalculate")
def recalculate_project_tonic(session_id: str, req: RecalculateRequest):
    """Fast recalculation of a project with a new tonic."""
    try:
        from .pipeline.analyze import recalculate_tonic
        
        # Load existing project to ensure it exists
        project = db.get_project(session_id)
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")
            
        # Run fast recalculation using saved pitch arrays
        new_data = recalculate_tonic(session_id, req.tonic_hint, req.borgeet_mode)
        
        # Update project data
        project.update(new_data)
        
        # We must delete 'feedback_given' as it is not part of the schema when saving
        if 'feedback_given' in project:
            del project['feedback_given']
            
        db.save_project(session_id, project)
        
        # Restore feedback_given for the response
        project['feedback_given'] = db.get_project(session_id).get('feedback_given', False)
        return project
        
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Pitch arrays not found for this session. Please run a full re-analysis.")
    except Exception as e:
        logger.error(f"Failed to recalculate: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """
    Analyze a track (YouTube URL or server-side path).
    Returns the full analysis JSON.

    For long-running analysis with progress updates, use /analyze/stream instead.
    """
    try:
        result = analyze_track(
            req.source,
            is_youtube=req.is_youtube,
            borgeet_mode=req.borgeet_mode,
            use_separation=req.use_separation,
            tonic_hint=req.tonic_hint,
            time_of_day=req.time_of_day,
        )
        # We need the session_id to save the project. It's stored in the result itself, or we can extract it from stems url.
        # Actually, let's make sure analyze_track returns a session_id in the result!
        # For now we'll extract it from the stems_dir or generate a new one if it's missing.
        session_id = result.get("session_id")
        if not session_id and result.get("vocals_url"):
            session_id = result["vocals_url"].split("/")[-2]
            
        if session_id:
            # Save asynchronously to not block the response
            background_tasks = BackgroundTasks()
            db.save_project(session_id, result)
            
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/analyze/upload")
async def analyze_upload(
    file: UploadFile = File(...),
    borgeet_mode: bool = Query(False),
    use_separation: bool = Query(True),
    tonic_hint: str | None = Query(None),
    time_of_day: str | None = Query(None),
):
    """
    Upload an audio file and analyze it.
    Supports WAV, MP3, FLAC, OGG, etc.
    """
    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload{suffix}")
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        result = analyze_track(
            tmp_path,
            is_youtube=False,
            borgeet_mode=borgeet_mode,
            use_separation=use_separation,
            tonic_hint=tonic_hint,
            time_of_day=time_of_day,
        )
        
        session_id = result.get("session_id")
        if not session_id and result.get("vocals_url"):
            session_id = result["vocals_url"].split("/")[-2]
            
        if session_id:
            db.save_project(session_id, result)
            
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        # Clean up all temp files (uploaded file + any stem WAVs from separation)
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/analyze/stream")
async def analyze_stream(req: AnalyzeRequest):
    """
    SSE (Server-Sent Events) endpoint for progress-aware analysis.
    Sends progress events during long-running stages (esp. source separation),
    then sends the final result.
    """
    import asyncio
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_progress(stage: str, pct: int):
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", {"stage": stage, "pct": pct}))

    def run_analysis():
        try:
            result = analyze_track(
                req.source,
                is_youtube=req.is_youtube,
                borgeet_mode=req.borgeet_mode,
                use_separation=req.use_separation,
                tonic_hint=req.tonic_hint,
                time_of_day=req.time_of_day,
                progress_callback=on_progress,
            )
            
            # Extract session ID and save project
            session_id = result.get("session_id")
            if not session_id and result.get("vocals_url"):
                session_id = result["vocals_url"].split("/")[-2]
                result["session_id"] = session_id
                
            if session_id:
                db.save_project(session_id, result)
                
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
        except ValueError as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        except Exception as e:
            traceback.print_exc()
            loop.call_soon_threadsafe(queue.put_nowait, ("error", f"Analysis failed: {e}"))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    async def event_generator():
        # Start background thread
        task = asyncio.create_task(asyncio.to_thread(run_analysis))
        
        # Send initial event
        yield f"data: {json.dumps({'type': 'progress', 'stage': 'starting', 'pct': 0})}\n\n"

        while True:
            evt_type, data = await queue.get()
            
            if evt_type == "progress":
                yield f"data: {json.dumps({'type': 'progress', **data})}\n\n"
            elif evt_type == "result":
                yield f"data: {json.dumps({'type': 'result', 'data': data})}\n\n"
            elif evt_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'detail': data})}\n\n"
            elif evt_type == "done":
                break
                
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.post("/analyze/upload/stream")
async def analyze_upload_stream(
    file: UploadFile = File(...),
    borgeet_mode: bool = Query(False),
    use_separation: bool = Query(True),
    tonic_hint: str | None = Query(None),
):
    import asyncio
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    suffix = os.path.splitext(file.filename or "")[1] or ".wav"
    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, f"upload{suffix}")

    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")

    def on_progress(stage: str, pct: int):
        loop.call_soon_threadsafe(queue.put_nowait, ("progress", {"stage": stage, "pct": pct}))

    def run_analysis():
        try:
            result = analyze_track(
                tmp_path,
                is_youtube=False,
                borgeet_mode=borgeet_mode,
                use_separation=use_separation,
                tonic_hint=tonic_hint,
                progress_callback=on_progress,
            )
            
            session_id = result.get("session_id")
            if not session_id and result.get("vocals_url"):
                session_id = result["vocals_url"].split("/")[-2]
                result["session_id"] = session_id
                
            if session_id:
                db.save_project(session_id, result)
                
            loop.call_soon_threadsafe(queue.put_nowait, ("result", result))
        except ValueError as e:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(e)))
        except Exception as e:
            traceback.print_exc()
            loop.call_soon_threadsafe(queue.put_nowait, ("error", f"Analysis failed: {e}"))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            loop.call_soon_threadsafe(queue.put_nowait, ("done", None))

    async def event_generator():
        task = asyncio.create_task(asyncio.to_thread(run_analysis))
        yield f"data: {json.dumps({'type': 'progress', 'stage': 'starting', 'pct': 0})}\n\n"

        while True:
            evt_type, data = await queue.get()
            if evt_type == "progress":
                yield f"data: {json.dumps({'type': 'progress', **data})}\n\n"
            elif evt_type == "result":
                yield f"data: {json.dumps({'type': 'result', 'data': data})}\n\n"
            elif evt_type == "error":
                yield f"data: {json.dumps({'type': 'error', 'detail': data})}\n\n"
            elif evt_type == "done":
                break
                
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@app.delete("/projects/{session_id}")
def delete_project_endpoint(session_id: str):
    """Delete a project from the database and remove its static files."""
    success = db.delete_project(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Project not found or could not be deleted")
        
    # Attempt to clean up static files (stems) if they exist
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    stems_dir = os.path.join(backend_dir, "static", "stems", session_id)
    if os.path.exists(stems_dir):
        try:
            shutil.rmtree(stems_dir, ignore_errors=True)
            logger.info("Deleted static files for project %s", session_id)
        except Exception as e:
            logger.error("Failed to delete static files for %s: %s", session_id, e)
            
    return {"status": "success", "message": "Project deleted"}
