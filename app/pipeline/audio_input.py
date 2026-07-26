"""
audio_input.py
Handles getting a mono 44.1kHz WAV from either a YouTube URL or a local file.

NOTE: yt-dlp needs real internet access to youtube.com/googlevideo.com, which
this sandbox does not have. This module is written to run on your own server.
"""
import os
import re
import subprocess
import tempfile
import librosa
import numpy as np
from urllib.parse import urlparse, parse_qs

SR = 44100  # target sample rate for all downstream analysis


def _parse_start_time(url: str) -> float:
    """
    Extract the start-time offset (in seconds) from a YouTube URL.

    Supports:
      - ?t=30          -> 30s
      - ?t=1m30s       -> 90s
      - ?t=1h2m30s     -> 3750s
      - &start=45      -> 45s
    Returns 0.0 if no timestamp is found.
    """
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)

    # Also handle youtu.be fragment params
    if not qs and parsed.fragment:
        qs = parse_qs(parsed.fragment)

    raw = (qs.get("t") or qs.get("start") or [None])[0]
    if raw is None:
        return 0.0

    # Pure number (seconds)
    if raw.isdigit():
        return float(raw)

    # Format like 1h2m30s or 2m30s or 30s
    total = 0.0
    m = re.findall(r'(\d+)([hms])', raw)
    for val, unit in m:
        if unit == 'h':
            total += int(val) * 3600
        elif unit == 'm':
            total += int(val) * 60
        elif unit == 's':
            total += int(val)
    return total if total > 0 else 0.0


def download_youtube_audio(url: str, out_dir: str = None) -> str:
    """
    Downloads best-quality audio from a YouTube URL and converts to WAV.
    Returns path to the WAV file and metadata dict.
    Requires: yt-dlp python package, ffmpeg installed on the host.
    """
    import yt_dlp
    
    out_dir = out_dir or tempfile.mkdtemp()
    out_template = os.path.join(out_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': out_template,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {
                'player_client': ['android', 'web']
            }
        },
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            meta = {
                "title": info.get("title", "Unknown Title"),
                "uploader": info.get("uploader", "Unknown Artist"),
                "thumbnail": info.get("thumbnail"),
                "duration": info.get("duration")
            }
    except Exception as e:
        raise RuntimeError(f"yt-dlp download failed: {e}")

    # find the produced wav
    for f in os.listdir(out_dir):
        if f.endswith(".wav"):
            return os.path.join(out_dir, f), meta
    raise RuntimeError("yt-dlp did not produce a WAV file (is FFmpeg installed?)")


def load_audio(path: str, sr: int = SR, mono: bool = True,
                trim_silence: bool = True) -> tuple[np.ndarray, int]:
    """Loads any audio file into a normalized float32 waveform."""
    y, sr = librosa.load(path, sr=sr, mono=mono)
    if trim_silence:
        y, _ = librosa.effects.trim(y, top_db=30)
    # peak-normalize
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak
    return y, sr


def get_audio_for_track(source: str, is_youtube: bool = None) -> tuple[np.ndarray, int, dict]:
    """
    Unified entry point. `source` is either a YouTube URL or a local file path.
    Auto-detects unless is_youtube is explicitly passed.

    If the YouTube URL contains a ?t= or &start= parameter, the returned
    audio is trimmed to start from that timestamp.
    """
    if is_youtube is None:
        is_youtube = source.startswith("http") and ("youtube.com" in source or "youtu.be" in source)

    meta = {}
    start_sec = 0.0

    if is_youtube:
        start_sec = _parse_start_time(source)
        wav_path, meta = download_youtube_audio(source)
    else:
        wav_path = source
        meta = {
            "title": os.path.basename(source),
            "uploader": "Local File",
            "thumbnail": None,
            "duration": None
        }

    y, sr = load_audio(wav_path)

    # Trim from the requested start time
    if start_sec > 0:
        start_sample = int(start_sec * sr)
        if start_sample < len(y):
            y = y[start_sample:]
            meta["skipped_sec"] = round(start_sec, 1)

    return y, sr, meta
