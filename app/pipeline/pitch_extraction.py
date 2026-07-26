"""
pitch_extraction.py
Extracts a frame-wise fundamental frequency (f0) contour from audio,
plus voicing confidence.

Primary: torchcrepe (deep learning, GPU-accelerated on CUDA)
  - Significantly more accurate than pYIN for vocal music with ornaments
  - Runs on GPU (GTX 1650) for fast inference
  - Better voiced/unvoiced detection

Fallback: librosa pYIN (DSP heuristic, CPU-only)
  - Used automatically if torchcrepe is not installed
"""
import logging
import numpy as np
import librosa
import torch

logger = logging.getLogger(__name__)

FMIN = librosa.note_to_hz("C2")   # ~65 Hz, covers low male vocal range
FMAX = librosa.note_to_hz("C7")   # ~2093 Hz, covers high female vocal / flute
HOP_LENGTH = 512

# Check torchcrepe availability at import time
_CREPE_AVAILABLE = False
try:
    import torchcrepe
    _CREPE_AVAILABLE = True
    logger.info("torchcrepe is available — using CREPE for pitch extraction.")
except ImportError:
    logger.warning("torchcrepe not installed — falling back to pYIN (less accurate).")


def _pick_device() -> str:
    """Select best device for CREPE inference."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_pitch_crepe(y: np.ndarray, sr: int,
                        hop_length: int = HOP_LENGTH,
                        device: str = "auto",
                        batch_size: int = 256) -> dict:
    """
    GPU-accelerated pitch extraction using CREPE (deep CNN).

    Returns dict with:
      f0         : Hz per frame (nan where unvoiced)
      voiced     : bool per frame
      voiced_prob: confidence per frame [0, 1]
      times      : timestamp per frame (seconds)
    """
    if device == "auto":
        device = _pick_device()

    # Convert to torch tensor (torchcrepe expects (1, n_samples) float32)
    audio_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).to(device)

    # torchcrepe.predict returns (batch, n_frames) tensors
    # Use 'full' model for best accuracy; 'tiny' for speed
    # hop_length in samples → step_size doesn't exist; we use hop_length directly
    pitch = torchcrepe.predict(
        audio_tensor,
        sr,
        hop_length=hop_length,
        fmin=FMIN,
        fmax=FMAX,
        model='full',
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
        pad=True,
    )

    # Unpack: pitch is a tuple of (f0, periodicity) when return_periodicity=True
    f0_tensor, periodicity = pitch

    # Move to CPU numpy
    f0 = f0_tensor.squeeze(0).cpu().numpy()
    confidence = periodicity.squeeze(0).cpu().numpy()

    # Apply Viterbi smoothing for cleaner tracks
    # torchcrepe uses a different periodicity threshold; 0.21 is the default
    # recommended threshold from the CREPE paper
    voiced = confidence > 0.21
    f0_clean = np.where(voiced, f0, np.nan)

    # Generate time axis
    n_frames = len(f0)
    times = np.arange(n_frames) * hop_length / sr

    logger.info("CREPE pitch extraction complete: %d frames, %d voiced (%.0f%%), device=%s",
                n_frames, voiced.sum(), 100 * voiced.mean(), device)

    return {
        "f0": f0_clean,
        "voiced": voiced,
        "voiced_prob": confidence,
        "times": times,
    }


def extract_pitch_pyin(y: np.ndarray, sr: int) -> dict:
    """
    Fallback: librosa pYIN (DSP heuristic, CPU-only).

    Returns dict with:
      f0        : Hz per frame (nan where unvoiced)
      voiced    : bool per frame
      voiced_prob: confidence per frame
      times     : timestamp per frame (seconds)
    """
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=FMIN, fmax=FMAX, sr=sr, hop_length=HOP_LENGTH
    )
    times = librosa.times_like(f0, sr=sr, hop_length=HOP_LENGTH)
    return {
        "f0": f0,
        "voiced": voiced_flag,
        "voiced_prob": voiced_prob,
        "times": times,
    }


def extract_pitch(y: np.ndarray, sr: int, device: str = "auto") -> dict:
    """
    Primary entry point: uses CREPE if available, else pYIN fallback.
    Logs a warning when falling back to pYIN.
    """
    if _CREPE_AVAILABLE:
        try:
            return extract_pitch_crepe(y, sr, device=device, batch_size=256)
        except Exception as e:
            logger.warning("CREPE failed on primary attempt (%s). Retrying after clearing CUDA cache...", e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                # Retry with smaller batch size or on CPU to prevent OOM / allocation errors
                fallback_dev = "cpu" if "allocation" in str(e).lower() or "memory" in str(e).lower() else device
                return extract_pitch_crepe(y, sr, device=fallback_dev, batch_size=128)
            except Exception as e2:
                logger.warning("CREPE retry failed (%s), falling back to pYIN.", e2)

    logger.info("Using pYIN for pitch extraction (fallback).")
    return extract_pitch_pyin(y, sr)


def clean_pitch_track(pitch: dict, prob_threshold: float = 0.5) -> np.ndarray:
    """Returns only confidently-voiced f0 values (Hz), NaNs dropped."""
    f0 = pitch["f0"]
    prob = pitch["voiced_prob"]

    # CREPE uses periodicity (lower threshold ~0.21), pYIN uses probability (~0.5)
    # Use the provided threshold but also respect the voiced flag
    mask = (~np.isnan(f0)) & (prob >= prob_threshold)
    return f0[mask]
