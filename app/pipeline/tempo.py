"""
tempo.py
Basic tempo/beat estimation. This is deliberately simple (BPM + beat grid) --
full taal (rhythmic cycle) identification, i.e. matching a theka pattern like
teentaal/keherwa, is a separate sequence-classification problem that needs
its own labeled dataset. This module gives you the first, honest layer.
"""
import librosa
import numpy as np


def estimate_tempo(y: np.ndarray, sr: int) -> dict:
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
    return {
        "tempo_bpm": round(tempo_val, 1),
        "beat_times": [round(float(t), 2) for t in beat_times[:200]],  # capped for payload size
    }
