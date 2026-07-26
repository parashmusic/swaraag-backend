"""
source_separation.py
Isolates vocal stem from ensemble Borgeet recordings using Meta's HTDemucs.

Why this matters for Borgeet analysis:
  - Borgeet performances mix vocal melody with khol drum + taal cymbals
  - Pitch extraction on the raw mix confuses drum transients with pitched notes
  - Separating vocals before pitch tracking dramatically improves tonic/swara accuracy

Architecture:
  - vocal stem  → pitch_extraction.py (melodic analysis)
  - drum stem   → reserved for Phase 3 (khol/taal rhythm analysis)
  - bass stem   → may contain harmonium drone (useful for tonic validation)
  - other stem  → taal cymbals, ambient

Optimized for GTX 1650 (4GB VRAM):
  - segment=5 (process 5-second chunks to keep peak VRAM ~2.5GB)
  - automatic CPU fallback on CUDA OOM
  - model loaded once via @lru_cache, not per-request
"""
import logging
import numpy as np
import torch
import torchaudio
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# ---- Configuration ----
DEMUCS_MODEL = "htdemucs"
SEGMENT_LENGTH = 5        # seconds per chunk — keeps VRAM under 4GB on GTX 1650
OVERLAP = 0.25            # overlap ratio between segments for smooth transitions

# HTDemucs stem ordering (fixed by the model architecture)
STEM_NAMES = ["drums", "bass", "other", "vocals"]


@lru_cache(maxsize=1)
def _load_model():
    """Load the HTDemucs model once and cache it for the process lifetime."""
    from demucs.pretrained import get_model
    logger.info("Loading HTDemucs model '%s' (first call only)...", DEMUCS_MODEL)
    model = get_model(name=DEMUCS_MODEL)
    logger.info("HTDemucs model loaded successfully.")
    return model


def _pick_device() -> str:
    """
    Select the best available device for inference.
    For GTX 1650 (4GB): uses CUDA but with segment-based processing.
    Falls back to CPU if CUDA is unavailable.
    """
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            logger.info("CUDA available: %s (%.1f GB VRAM)", props.name, vram_gb)
            return "cuda"
        except Exception as e:
            logger.warning("CUDA check failed, falling back to CPU: %s", e)
    logger.info("CUDA not available, using CPU (separation will be slower).")
    return "cpu"


def separate_stems(audio_path: str, output_dir: str = None,
                   device: str = "auto") -> dict:
    """
    Separates an audio file into stems using HTDemucs.

    Parameters
    ----------
    audio_path : str
        Path to the input audio file (WAV, MP3, etc.)
    output_dir : str, optional
        If provided, saves each stem as a separate WAV file in this directory.
    device : str
        "auto" (default), "cuda", or "cpu".

    Returns
    -------
    dict with keys:
        vocals      : np.ndarray (channels, samples) — the isolated vocal track
        drums       : np.ndarray — khol/percussion
        bass        : np.ndarray — bass/drone
        other       : np.ndarray — cymbals/ambient
        sample_rate : int — sample rate of the output stems
        vocals_path, drums_path, ... : str (only if output_dir was given)
    """
    from demucs.apply import apply_model

    model = _load_model()

    if device == "auto":
        device = _pick_device()

    model = model.to(device)

    # Load audio
    wav, sr = torchaudio.load(audio_path)
    logger.info("Loaded audio: %d channels, %d samples, %d Hz", wav.shape[0], wav.shape[1], sr)

    # Resample to the model's expected sample rate if needed
    if sr != model.samplerate:
        logger.info("Resampling from %d Hz to %d Hz", sr, model.samplerate)
        wav = torchaudio.transforms.Resample(sr, model.samplerate)(wav)
        sr = model.samplerate

    # Ensure stereo (HTDemucs expects 2 channels)
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    # Normalize for numerical stability
    ref = wav.mean(0)
    ref_mean = ref.mean()
    ref_std = ref.std() + 1e-8
    wav_norm = (wav - ref_mean) / ref_std

    # Apply model with segment-based processing for low VRAM GPUs
    try:
        sources = apply_model(
            model, wav_norm[None].to(device),
            segment=SEGMENT_LENGTH,
            overlap=OVERLAP,
            progress=False,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            logger.warning("CUDA OOM — falling back to CPU for separation.")
            torch.cuda.empty_cache()
            model = model.to("cpu")
            sources = apply_model(
                model, wav_norm[None].to("cpu"),
                segment=SEGMENT_LENGTH,
                overlap=OVERLAP,
                progress=False,
            )
        else:
            raise

    # De-normalize
    sources = sources * ref_std + ref_mean

    # sources shape: (1, n_stems, channels, samples)
    sources = sources[0].cpu().numpy()

    result = {
        "sample_rate": sr,
    }

    for i, stem_name in enumerate(STEM_NAMES):
        result[stem_name] = sources[i]

    # Optionally save stems as WAV files
    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for i, stem_name in enumerate(STEM_NAMES):
            stem_path = out / f"{stem_name}.wav"
            stem_tensor = torch.tensor(sources[i])
            torchaudio.save(str(stem_path), stem_tensor, sr)
            result[f"{stem_name}_path"] = str(stem_path)
            logger.info("Saved stem: %s", stem_path)

    # Release GPU memory back to CUDA so CREPE can use the VRAM cleanly
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def vocals_to_mono(vocals: np.ndarray) -> np.ndarray:
    """
    Convert the stereo vocal stem to a mono float32 waveform,
    matching the format expected by pitch_extraction.py.
    """
    if vocals.ndim == 2:
        mono = vocals.mean(axis=0)
    else:
        mono = vocals
    # Peak-normalize (same as audio_input.load_audio)
    peak = np.max(np.abs(mono)) + 1e-9
    return (mono / peak).astype(np.float32)
