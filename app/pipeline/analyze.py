"""
analyze.py
Orchestrates the full pipeline and shapes the output to match exactly what
the frontend (SWARAAGDashboard) expects.

Phase 1 upgrade:
  - Source separation (HTDemucs) isolates vocals before pitch extraction
  - Enhanced tonic detection with Sa/Pa harmonic validation + confidence
  - Borgeet mode: prioritizes Borgeet ragas and includes Borgeet metadata

Output JSON:
{
  tonic_hz, tonic_nearest_note, tonic_confidence, tonic_method,
  dominant_notes: [{note, value}, ...],
  implied_scale: [str, ...],
  raga_candidates: [{raga, thaat, similarity, confidence_pct, notes, borgeet?, ...}, ...],
  swara_histogram: [float x12],
  duration_sec, n_voiced_frames,
  tempo_bpm,
  pitch_contour: [{t, cents}, ...],
  separation_used: bool,
  borgeet_mode: bool,
}
"""
import logging
import time
import numpy as np
import librosa
import uuid
import pretty_midi
import os
import subprocess

from .audio_input import get_audio_for_track
from .pitch_extraction import extract_pitch, clean_pitch_track, HOP_LENGTH
from .tonic_detection import estimate_tonic_with_validation, cents_relative_to_tonic
from .note_mapping import build_swara_histogram, dominant_notes as _dominant_notes, implied_scale as _implied_scale, build_transition_matrix
from .raga_matcher import match_by_similarity
from .tempo import estimate_tempo

logger = logging.getLogger(__name__)

# Absolute path to the backend root (where uvicorn runs from)
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MAX_CONTOUR_POINTS = 300

# Try to import source separation (requires torch + demucs)
_SEPARATION_AVAILABLE = False
try:
    from .source_separation import separate_stems, vocals_to_mono
    _SEPARATION_AVAILABLE = True
    logger.info("Source separation (HTDemucs) is available.")
except ImportError as e:
    logger.warning("Source separation not available (install torch + demucs): %s", e)


def _downsample_contour(times: np.ndarray, cents: np.ndarray, max_points: int) -> list[dict]:
    n = len(times)
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points).astype(int)
    return [{"t": round(float(times[i]), 2), "cents": round(float(cents[i]), 1)} for i in idx]


def analyze_track(source: str, is_youtube: bool = None,
                  borgeet_mode: bool = False,
                  use_separation: bool = True,
                  tonic_hint: str = None,
                  time_of_day: str = None,
                  progress_callback=None) -> dict:
    """
    Full analysis pipeline.

    Parameters
    ----------
    source : str
        YouTube URL or path to an audio file.
    is_youtube : bool, optional
        Force YouTube vs local file detection.
    borgeet_mode : bool
        When True, prioritizes Borgeet ragas in matching and includes
        Borgeet-specific metadata.
    use_separation : bool
        When True and HTDemucs is available, separates vocals before
        pitch extraction. Set False to skip (faster, but less accurate
        on ensemble recordings).
    progress_callback : callable, optional
        Called with (stage_name, progress_pct) for SSE progress reporting.
    """
    timings = {}

    def _report(stage: str, pct: int):
        if progress_callback:
            progress_callback(stage, pct)
        logger.info("Pipeline stage: %s (%d%%)", stage, pct)

    # ---- Stage 1: Load audio ----
    _report("loading_audio", 5)
    t0 = time.time()
    y, sr, track_meta = get_audio_for_track(source, is_youtube=is_youtube)
    timings["load_audio"] = round(time.time() - t0, 2)
    
    session_id = str(uuid.uuid4())
    out_dir = os.path.join(BACKEND_DIR, "static", "stems", session_id)
    os.makedirs(out_dir, exist_ok=True)

    # ---- Stage 2: Source separation (if available and requested) ----
    separation_used = False
    y_for_pitch = y  # default: use full mix
    vocals_url = None
    drums_url = None
    bass_url = None
    other_url = None

    if use_separation and _SEPARATION_AVAILABLE:
        _report("separating_stems", 15)
        t0 = time.time()
        try:
            import tempfile
            import soundfile as sf

            # Save the loaded audio to a temp file for Demucs
            # (Demucs expects a file path, not a numpy array)
            tmp_dir = tempfile.mkdtemp()
            tmp_path = f"{tmp_dir}/input.wav"
            sf.write(tmp_path, y, sr)

            stems = separate_stems(tmp_path)
            y_for_pitch = vocals_to_mono(stems["vocals"])
            # Save stems as MP3
            stems_dir = out_dir
            
            stem_names = ["vocals", "drums", "bass", "other"]
            urls = {}
            
            for stem in stem_names:
                wav_path = os.path.join(tmp_dir, f"{stem}.wav")
                mp3_path = os.path.join(stems_dir, f"{stem}.mp3")
                
                # soundfile expects (samples, channels) so we transpose
                sf.write(wav_path, stems[stem].T, stems["sample_rate"])
                
                try:
                    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-b:a", "128k", mp3_path], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    urls[stem] = f"/static/stems/{session_id}/{stem}.mp3"
                except Exception as ffmpeg_err:
                    logger.error(f"Failed to encode {stem} to MP3: %s", ffmpeg_err)
            
            vocals_url = urls.get("vocals")
            drums_url = urls.get("drums")
            bass_url = urls.get("bass")
            other_url = urls.get("other")

            # Resample if needed (Demucs may output at a different rate)
            if stems["sample_rate"] != sr:
                y_for_pitch = librosa.resample(
                    y_for_pitch, orig_sr=stems["sample_rate"], target_sr=sr
                )

            separation_used = True
            logger.info("Vocal separation successful.")

            # Clean up temp file
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        except Exception as e:
            logger.warning("Source separation failed, using full mix: %s", e)
            y_for_pitch = y

        timings["separation"] = round(time.time() - t0, 2)

    # ---- Stage 3: Pitch extraction ----
    _report("extracting_pitch", 50)
    t0 = time.time()
    
    # Extract pitch for vocal melody
    pitch = extract_pitch(y_for_pitch, sr)
    f0_all = pitch["f0"]
    prob = pitch["voiced_prob"]
    times = pitch["times"]

    f0_clean = clean_pitch_track(pitch, prob_threshold=0.5)
    if len(f0_clean) < 50:
        raise ValueError("Not enough voiced pitch content detected — check audio quality/input.")
    timings["pitch_extraction"] = round(time.time() - t0, 2)

    # ---- Stage 4: Tonic detection ----
    # Rule: the most frequently occurring pitch class = Sa.
    _report("detecting_tonic", 65)
    t0 = time.time()

    if tonic_hint:
        import re
        note_str = tonic_hint.strip()
        if not re.search(r'\d$', note_str):
            note_str += "3"  # Default to octave 3 if not specified
        try:
            tonic_hz = float(librosa.note_to_hz(note_str))
            tonic_result = {
                "tonic_hz": tonic_hz,
                "tonic_note": librosa.hz_to_note(tonic_hz),
                "confidence": 1.0,
                "method": "manual_override",
                "candidates": [],
            }
            logger.info("Using manual tonic override: %s (%f Hz)", note_str, tonic_hz)
        except Exception as e:
            logger.warning("Invalid tonic hint %s: %s", tonic_hint, e)
            tonic_hint = None  # Fallback to auto-detection

    if not tonic_hint:
        tonic_result = estimate_tonic_with_validation(f0_clean)
        tonic_hz = tonic_result["tonic_hz"]


    timings["tonic_detection"] = round(time.time() - t0, 2)



    # ---- Stage 5: Swara histogram + raga matching ----
    _report("matching_ragas", 80)
    t0 = time.time()

    # Build initial swara histogram relative to the detected tonic
    cents_rel_clean = cents_relative_to_tonic(f0_clean, tonic_hz)
    swara_hist = build_swara_histogram(cents_rel_clean)

    # THE RULE: the note that appears most on the swara wheel IS the key (Sa).
    # Whatever bin is highest in the relative histogram → shift the tonic so
    # that note lands on bin 0. No conditions, always applied (unless user
    # gave a manual tonic_hint, in which case we trust them).
    if not tonic_hint:
        dominant_bin = int(np.argmax(swara_hist))
        if dominant_bin != 0:
            corrected_hz = tonic_hz * (2 ** (dominant_bin / 12.0))
            # Normalize to vocal octave (80–450 Hz)
            while corrected_hz < 80:
                corrected_hz *= 2
            while corrected_hz > 450:
                corrected_hz /= 2
            logger.info(
                "Tonic shift: %s → %s (swara wheel dominant was %s, bin %d)",
                librosa.hz_to_note(tonic_hz),
                librosa.hz_to_note(corrected_hz),
                ["S","r","R","g","G","m","M","P","d","D","n","N"][dominant_bin],
                dominant_bin,
            )
            tonic_hz = round(float(corrected_hz), 2)
            tonic_result["tonic_hz"] = tonic_hz
            tonic_result["tonic_note"] = librosa.hz_to_note(tonic_hz)
            # Rebuild histogram with corrected tonic
            cents_rel_clean = cents_relative_to_tonic(f0_clean, tonic_hz)
            swara_hist = build_swara_histogram(cents_rel_clean)

    # Phase 2A: Build bigram transition matrix
    trans_matrix = build_transition_matrix(cents_rel_clean)

    dom = [{"note": n, "value": v} for n, v in _dominant_notes(swara_hist, top_n=5)]
    scale = _implied_scale(swara_hist)
    raga_matches = match_by_similarity(swara_hist, transition_matrix=trans_matrix, top_n=5, borgeet_mode=borgeet_mode, time_of_day=time_of_day)
    timings["raga_matching"] = round(time.time() - t0, 2)


    # ---- Stage 6: Pitch contour for visualization ----
    voiced_mask = (~np.isnan(f0_all)) & (prob >= 0.5)
    contour_cents = 1200 * np.log2(f0_all[voiced_mask] / tonic_hz)
    contour_times = times[voiced_mask]
    pitch_contour = _downsample_contour(contour_times, contour_cents, MAX_CONTOUR_POINTS)

    # ---- Stage 6.5: Export MIDI ----
    _report("exporting_midi", 85)
    midi_url = None
    try:
        # Use the full-resolution voiced pitch data (not the downsampled contour)
        voiced_f0 = f0_all[voiced_mask]
        voiced_times = times[voiced_mask]

        if len(voiced_f0) > 10:
            # Convert Hz -> MIDI pitch numbers (float, then round to int)
            raw_midi = librosa.hz_to_midi(voiced_f0)
            raw_midi = np.round(raw_midi).astype(int)
            raw_midi = np.clip(raw_midi, 0, 127)

            # --- Smoothing Step 1: Median filter ---
            # A window of ~7 frames absorbs vibrato and micro-ornaments
            # (typical CREPE hop is 10ms, so 7 frames ≈ 70ms)
            from scipy.ndimage import median_filter
            kernel_size = 7
            smoothed_midi = median_filter(raw_midi, size=kernel_size).astype(int)

            # --- Smoothing Step 2: Build note segments by grouping consecutive same-pitch ---
            segments = []
            seg_start = 0
            for i in range(1, len(smoothed_midi)):
                # Also break on large time gaps (silence / unvoiced gaps in the original)
                time_gap = voiced_times[i] - voiced_times[i - 1]
                if smoothed_midi[i] != smoothed_midi[seg_start] or time_gap > 0.15:
                    segments.append((
                        int(smoothed_midi[seg_start]),
                        float(voiced_times[seg_start]),
                        float(voiced_times[i - 1])
                    ))
                    seg_start = i
            # Final segment
            segments.append((
                int(smoothed_midi[seg_start]),
                float(voiced_times[seg_start]),
                float(voiced_times[-1])
            ))

            # --- Smoothing Step 3: Filter out very short notes (< 120ms) ---
            MIN_NOTE_DUR = 0.12
            filtered = [s for s in segments if (s[2] - s[1]) >= MIN_NOTE_DUR]

            # --- Smoothing Step 4: Merge adjacent notes of the same pitch ---
            merged = []
            for seg in filtered:
                if merged and merged[-1][0] == seg[0] and (seg[1] - merged[-1][2]) < 0.08:
                    # Same pitch, small gap -> extend previous note
                    merged[-1] = (merged[-1][0], merged[-1][1], seg[2])
                else:
                    merged.append(seg)

            # --- Build MIDI ---
            pm = pretty_midi.PrettyMIDI()
            inst = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

            for pitch, start, end in merged:
                # Give each note a tiny tail so it doesn't clip
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start,
                    end=end + 0.02
                )
                inst.notes.append(note)

            pm.instruments.append(inst)
            midi_path = os.path.join(out_dir, "melody.mid")
            os.makedirs(os.path.dirname(midi_path), exist_ok=True)  # safety
            pm.write(midi_path)
            midi_url = f"/static/stems/{session_id}/melody.mid"
            logger.info("MIDI exported: %d notes (from %d raw frames)", len(merged), len(voiced_f0))

    except Exception as e:
        logger.warning(f"Failed to generate MIDI: {e}")

    # ---- Stage 7: Tempo ----
    _report("analyzing_tempo", 90)
    t0 = time.time()
    tempo_info = estimate_tempo(y, sr)
    timings["tempo"] = round(time.time() - t0, 2)

    _report("complete", 100)

    # ---- Save pitch arrays for fast recalculation ----
    try:
        np.save(os.path.join(out_dir, "f0_all.npy"), f0_all)
        np.save(os.path.join(out_dir, "prob.npy"), prob)
        np.save(os.path.join(out_dir, "times.npy"), times)
    except Exception as e:
        logger.warning(f"Could not save pitch arrays for recalculation: {e}")

    return {
        # Tonic (enhanced)
        "tonic_hz": tonic_result["tonic_hz"],
        "tonic_nearest_note": tonic_result["tonic_note"],
        "tonic_confidence": tonic_result["confidence"],
        "tonic_method": tonic_result["method"],

        # Melodic analysis
        "dominant_notes": dom,
        "implied_scale": scale,
        "raga_candidates": raga_matches,
        "swara_histogram": [round(float(v), 4) for v in swara_hist],

        # Timing & structure
        "duration_sec": round(len(y) / sr, 1),
        "n_voiced_frames": int(len(f0_clean)),
        "tempo_bpm": tempo_info["tempo_bpm"],
        "beat_times": tempo_info["beat_times"],

        # Pitch contour
        "pitch_contour": pitch_contour,

        # Pipeline metadata
        "session_id": session_id,
        "separation_used": separation_used,
        "borgeet_mode": borgeet_mode,
        "timings": timings,
        
        # Track metadata
        "meta": track_meta,
        
        # Stems
        "vocals_url": vocals_url,
        "drums_url": drums_url,
        "bass_url": bass_url,
        "other_url": other_url,
        "midi_url": midi_url,
    }


def recalculate_tonic(session_id: str, tonic_hint: str, borgeet_mode: bool) -> dict:
    """
    Fast recalculation of Stage 4 and Stage 5 using cached pitch arrays.
    """
    out_dir = os.path.join(BACKEND_DIR, "static", "stems", session_id)
    f0_all = np.load(os.path.join(out_dir, "f0_all.npy"))
    prob = np.load(os.path.join(out_dir, "prob.npy"))
    times = np.load(os.path.join(out_dir, "times.npy"))
    
    pitch = {"f0": f0_all, "voiced_prob": prob, "times": times}
    f0_clean = clean_pitch_track(pitch, prob_threshold=0.5)
    
    if tonic_hint:
        import re
        note_str = tonic_hint.strip()
        if not re.search(r'\d$', note_str):
            note_str += "3"
        tonic_hz = float(librosa.note_to_hz(note_str))
        tonic_result = {
            "tonic_hz": tonic_hz,
            "tonic_note": librosa.hz_to_note(tonic_hz),
            "confidence": 1.0,
            "method": "manual_override",
        }
    else:
        # Re-run auto detection
        tonic_result = estimate_tonic_with_validation(f0_clean)
        tonic_hz = tonic_result["tonic_hz"]
            
    # Compute initial swara histogram relative to the detected tonic
    cents_rel_clean = cents_relative_to_tonic(f0_clean, tonic_hz)
    swara_hist = build_swara_histogram(cents_rel_clean)

    # THE RULE: the note that appears most on the swara wheel IS the key (Sa).
    if not tonic_hint:
        dominant_bin = int(np.argmax(swara_hist))
        if dominant_bin != 0:
            corrected_hz = tonic_hz * (2 ** (dominant_bin / 12.0))
            # Normalize to vocal octave (80–450 Hz)
            while corrected_hz < 80:
                corrected_hz *= 2
            while corrected_hz > 450:
                corrected_hz /= 2
            logger.info(
                "Recalc Tonic shift: %s → %s (swara wheel dominant was %s, bin %d)",
                librosa.hz_to_note(tonic_hz),
                librosa.hz_to_note(corrected_hz),
                ["S","r","R","g","G","m","M","P","d","D","n","N"][dominant_bin],
                dominant_bin,
            )
            tonic_hz = round(float(corrected_hz), 2)
            tonic_result["tonic_hz"] = tonic_hz
            tonic_result["tonic_note"] = librosa.hz_to_note(tonic_hz)
            # Rebuild histogram with corrected tonic
            cents_rel_clean = cents_relative_to_tonic(f0_clean, tonic_hz)
            swara_hist = build_swara_histogram(cents_rel_clean)
    
    # Phase 2A: Build bigram transition matrix
    trans_matrix = build_transition_matrix(cents_rel_clean)

    dom = [{"note": n, "value": v} for n, v in _dominant_notes(swara_hist, top_n=5)]
    scale = _implied_scale(swara_hist)
    raga_matches = match_by_similarity(swara_hist, transition_matrix=trans_matrix, top_n=5, borgeet_mode=borgeet_mode)
    
    voiced_mask = (~np.isnan(f0_all)) & (prob >= 0.5)
    contour_cents = 1200 * np.log2(f0_all[voiced_mask] / tonic_hz)
    contour_times = times[voiced_mask]
    pitch_contour = _downsample_contour(contour_times, contour_cents, MAX_CONTOUR_POINTS)
    
    return {
        "tonic_hz": tonic_result["tonic_hz"],
        "tonic_nearest_note": tonic_result["tonic_note"],
        "tonic_confidence": tonic_result["confidence"],
        "tonic_method": tonic_result["method"],
        "dominant_notes": dom,
        "implied_scale": scale,
        "raga_candidates": raga_matches,
        "swara_histogram": [round(float(v), 4) for v in swara_hist],
        "pitch_contour": pitch_contour,
    }

