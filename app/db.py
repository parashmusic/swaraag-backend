import sqlite3
import json
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Absolute path to the backend root
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BACKEND_DIR, "swaraag.db")

def init_db():
    """Create the projects table if it doesn't exist."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS projects (
                    session_id TEXT PRIMARY KEY,
                    title TEXT,
                    artist TEXT,
                    thumbnail TEXT,
                    duration_sec REAL,
                    top_raga TEXT,
                    tonic TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    full_json TEXT,
                    actual_raga TEXT
                )
            ''')
            # Attempt to add actual_raga column if it doesn't exist (for older DBs)
            try:
                cursor.execute('ALTER TABLE projects ADD COLUMN actual_raga TEXT')
            except sqlite3.OperationalError:
                pass # Column already exists
                
            # Attempt to add feedback_given column
            try:
                cursor.execute('ALTER TABLE projects ADD COLUMN feedback_given BOOLEAN DEFAULT 0')
            except sqlite3.OperationalError:
                pass # Column already exists
                
            # Default existing actual_raga to top_raga
            cursor.execute('UPDATE projects SET actual_raga = top_raga WHERE actual_raga IS NULL')
            
            conn.commit()
            logger.info("Database initialized at %s", DB_PATH)
    except Exception as e:
        logger.error("Failed to initialize database: %s", e)

def save_project(session_id: str, result: dict):
    """Save an analysis result to the projects table."""
    try:
        # Extract fields for easy querying/display
        meta = result.get("meta", {})
        title = meta.get("title", "Unknown Track")
        artist = meta.get("uploader", "Unknown Artist")
        thumbnail = meta.get("thumbnail", "")
        duration_sec = result.get("duration_sec", 0.0)
        
        # Get top raga
        top_raga = "Unknown"
        raga_cands = result.get("raga_candidates", [])
        if raga_cands:
            top_raga = raga_cands[0].get("raga", "Unknown")
            
        tonic = result.get("tonic_nearest_note", "Unknown")
        
        full_json = json.dumps(result)
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            
            # Preserve existing actual_raga if it exists, otherwise default to top_raga
            cursor.execute('SELECT actual_raga FROM projects WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            actual_raga = row[0] if (row and row[0] is not None) else top_raga
            
            cursor.execute('''
                INSERT OR REPLACE INTO projects 
                (session_id, title, artist, thumbnail, duration_sec, top_raga, tonic, full_json, actual_raga)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (session_id, title, artist, thumbnail, duration_sec, top_raga, tonic, full_json, actual_raga))
            conn.commit()
    except Exception as e:
        logger.error("Failed to save project to DB: %s", e)

def get_recent_projects(limit: int = 20) -> List[Dict[str, Any]]:
    """Get a list of recent projects (omitting full_json to save bandwidth)."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT session_id, title, artist, thumbnail, duration_sec, top_raga, tonic, timestamp, actual_raga
                FROM projects
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error("Failed to fetch projects: %s", e)
        return []

def get_project(session_id: str) -> Dict[str, Any]:
    """Get the full JSON result for a specific project."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT full_json, feedback_given FROM projects WHERE session_id = ?', (session_id,))
            row = cursor.fetchone()
            if row:
                logger.info("Retrieved project %s", session_id)
                data = json.loads(row[0])
                data['feedback_given'] = bool(row[1])
                return data
    except Exception as e:
        logger.error("Failed to get project: %s", e)
    return None

def delete_project(session_id: str) -> bool:
    """Delete a project from the projects table."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM projects WHERE session_id = ?", (session_id,))
            conn.commit()
            if cursor.rowcount > 0:
                logger.info("Deleted project %s", session_id)
                return True
            else:
                logger.warning("Project %s not found for deletion", session_id)
                return False
    except Exception as e:
        logger.error("Failed to delete project: %s", e)
        return False

def update_actual_raga(session_id: str, actual_raga: str) -> bool:
    """Update the actual_raga field for a specific project."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE projects SET actual_raga = ?, feedback_given = 1 WHERE session_id = ?", (actual_raga, session_id))
            conn.commit()
            if cursor.rowcount > 0:
                logger.info("Updated actual_raga for project %s to %s", session_id, actual_raga)
                return True
            else:
                logger.warning("Project %s not found for raga update", session_id)
                return False
    except Exception as e:
        logger.error("Failed to update actual_raga: %s", e)
        return False

def get_analytics_data() -> Dict[str, Any]:
    """Calculate aggregated analytics for the dashboard."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute('''
                SELECT top_raga, actual_raga, tonic, feedback_given
                FROM projects
            ''')
            rows = cursor.fetchall()
            
            total_tracks = len(rows)
            correct_predictions = 0
            
            raga_distribution = {}
            tonic_distribution = {}
            raga_accuracy = {} # { "Yaman": { "correct": 0, "incorrect": 0 } }
            
            for row in rows:
                top = row["top_raga"]
                actual = row["actual_raga"] or top
                tonic = row["tonic"]
                
                # Raga Dist
                if actual != "Unknown":
                    raga_distribution[actual] = raga_distribution.get(actual, 0) + 1
                
                # Tonic Dist
                if tonic != "Unknown":
                    # Remove the octave number (e.g., C#4 -> C#) for cleaner grouping
                    import re
                    clean_tonic = re.sub(r'\d+', '', tonic)
                    tonic_distribution[clean_tonic] = tonic_distribution.get(clean_tonic, 0) + 1
                    
                # Accuracy tracking
                is_correct = (top == actual)
                if is_correct:
                    correct_predictions += 1
                
                # Accuracy by Raga (using actual raga)
                if actual != "Unknown":
                    if actual not in raga_accuracy:
                        raga_accuracy[actual] = {"correct": 0, "incorrect": 0}
                    if is_correct:
                        raga_accuracy[actual]["correct"] += 1
                    else:
                        raga_accuracy[actual]["incorrect"] += 1
                        
            # Format distributions for Recharts [{ name: "Yaman", value: 10 }]
            raga_dist_arr = [{"name": k, "value": v} for k, v in raga_distribution.items()]
            raga_dist_arr.sort(key=lambda x: x["value"], reverse=True)
            
            tonic_dist_arr = [{"name": k, "value": v} for k, v in tonic_distribution.items()]
            tonic_dist_arr.sort(key=lambda x: x["value"], reverse=True)
            
            # Format accuracy by raga
            raga_acc_arr = [
                {"name": k, "correct": v["correct"], "incorrect": v["incorrect"]}
                for k, v in raga_accuracy.items()
            ]
            raga_acc_arr.sort(key=lambda x: x["correct"] + x["incorrect"], reverse=True)
            
            return {
                "total_tracks": total_tracks,
                "overall_accuracy": (correct_predictions / total_tracks * 100) if total_tracks > 0 else 0,
                "raga_distribution": raga_dist_arr,
                "tonic_distribution": tonic_dist_arr,
                "raga_accuracy": raga_acc_arr
            }
            
    except Exception as e:
        logger.error("Failed to get analytics: %s", e)
        return {
            "total_tracks": 0,
            "overall_accuracy": 0,
            "raga_distribution": [],
            "tonic_distribution": [],
            "raga_accuracy": []
        }

def get_dataset_paginated(limit: int = 50, offset: int = 0, predicted_raga: str = None, actual_raga: str = None, tonic: str = None, is_correct: bool = None) -> Dict[str, Any]:
    """Get paginated and filtered dataset view."""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = "SELECT session_id, title, artist, duration_sec, top_raga as predicted_raga, actual_raga, tonic, timestamp, feedback_given FROM projects WHERE 1=1"
            count_query = "SELECT COUNT(*) FROM projects WHERE 1=1"
            params = []
            
            if predicted_raga:
                query += " AND top_raga = ?"
                count_query += " AND top_raga = ?"
                params.append(predicted_raga)
            if actual_raga:
                query += " AND actual_raga = ?"
                count_query += " AND actual_raga = ?"
                params.append(actual_raga)
            if tonic:
                query += " AND tonic LIKE ?"
                count_query += " AND tonic LIKE ?"
                params.append(f"{tonic}%")
            if is_correct is not None:
                if is_correct:
                    query += " AND top_raga = actual_raga AND actual_raga IS NOT NULL"
                    count_query += " AND top_raga = actual_raga AND actual_raga IS NOT NULL"
                else:
                    query += " AND top_raga != actual_raga AND actual_raga IS NOT NULL"
                    count_query += " AND top_raga != actual_raga AND actual_raga IS NOT NULL"
                    
            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            
            # Get total count first
            cursor.execute(count_query, params)
            total = cursor.fetchone()[0]
            
            # Get data
            query_params = params + [limit, offset]
            cursor.execute(query, query_params)
            rows = cursor.fetchall()
            
            return {
                "total": total,
                "data": [dict(row) for row in rows]
            }
    except Exception as e:
        logger.error("Failed to fetch dataset: %s", e)
        return {"total": 0, "data": []}
