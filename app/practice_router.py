"""
practice_router.py
FastAPI router for the raga learning/practice feature.
Endpoints:
  POST /practice/upload   — same pipeline as /analyze/upload but lightweight (no stems)
  POST /practice/feedback — call Gemini AI tutor on a previous analysis result
"""
import os
import shutil
import tempfile
import logging
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use system env vars

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/practice", tags=["practice"])


# ---- lazy imports (same pattern as main.py) -------------------------
def _get_analyzer():
    from .pipeline.analyze import analyze_track
    return analyze_track


def _get_tutor():
    from .pipeline.ai_tutor import generate_tutor_feedback
    return generate_tutor_feedback


# ---- request / response models ---------------------------------------
class FeedbackRequest(BaseModel):
    session_id: str
    target_raga: str | None = None
    borgeet_mode: bool = True


# ---- endpoints -------------------------------------------------------
@router.post("/upload")
async def practice_upload(
    file: UploadFile = File(...),
    borgeet_mode: bool = Form(True),
    target_raga: str | None = Form(None),
):
    """
    Accepts a WAV/MP3/WebM blob (mic recording or file upload).
    Runs the pitch + raga pipeline WITHOUT source-separation (fast).
    Returns analysis + AI tutor feedback in one response.
    """
    analyze_track = _get_analyzer()
    generate_tutor_feedback = _get_tutor()

    tmp_dir = tempfile.mkdtemp()
    suffix = os.path.splitext(file.filename or "rec.webm")[1] or ".webm"
    tmp_path = os.path.join(tmp_dir, f"practice{suffix}")

    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        result = analyze_track(
            tmp_path,
            is_youtube=False,
            borgeet_mode=borgeet_mode,
            use_separation=False,   # fast — no Demucs for practice
        )
    except Exception as e:
        logger.error("Practice analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Generate AI feedback immediately
    tutor = generate_tutor_feedback(result, target_raga=target_raga, borgeet_mode=borgeet_mode)

    return {
        "analysis": result,
        "tutor": tutor,
    }


@router.post("/feedback")
async def practice_feedback(req: FeedbackRequest):
    """
    Generates (or re-generates) AI tutor feedback for an existing session.
    The frontend passes the full analysis payload and we call Gemini fresh.
    """
    from . import db
    generate_tutor_feedback = _get_tutor()

    project = db.get_project(req.session_id)
    if not project:
        raise HTTPException(status_code=404, detail="Session not found")

    tutor = generate_tutor_feedback(
        project,
        target_raga=req.target_raga,
        borgeet_mode=req.borgeet_mode,
    )
    return tutor
