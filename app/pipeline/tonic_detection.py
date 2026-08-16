"""
tonic_detection.py
Estimates the tonic (Sa) pitch of a performance.

Rule: The note that appears the most is the key (Sa).

In Indian classical music the performer anchors every phrase to Sa —
it is the most frequently visited and most sustained pitch in the recording.
So the dominant peak of the octave-folded pitch-class histogram IS Sa.

This replaces the previous multi-criterion scoring (Pa bonus, stability
weight, raga-guided override) which was causing incorrect tonic detection
by promoting acoustically louder but melodically less frequent notes.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def hz_to_cents(f0_hz: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
    """Convert Hz to cents relative to a reference (default A1=55 Hz)."""
    return 1200 * np.log2(f0_hz / ref_hz)


def estimate_tonic_with_validation(f0_values: np.ndarray,
                                   ref_hz: float = 55.0,
                                   bin_resolution_cents: float = 10.0,
                                   drone_f0_values: np.ndarray = None) -> dict:
    """
    Detect the tonic (Sa) as the most frequently occurring pitch class.

    If drone F0 values are supplied and have enough voiced content they are
    preferred over the melody, because the tanpura/harmonium locks onto Sa
    directly.

    Returns
    -------
    dict with keys:
        tonic_hz       : float
        tonic_note     : str  (e.g. "C#3")
        confidence     : float  [0, 1]
        method         : str
        candidates     : list[dict]  (top 5 peaks, for the UI)
    """
    import librosa

    # Prefer drone histogram if we have sufficient voiced frames
    if (drone_f0_values is not None and
            len(drone_f0_values) >= max(500, int(len(f0_values) * 0.15))):
        hist_f0 = drone_f0_values
        method_base = "drone_peak"
    else:
        if drone_f0_values is not None:
            logger.warning(
                "Drone stem has too few voiced frames (%d vs melody %d), "
                "using melody histogram for tonic detection.",
                len(drone_f0_values), len(f0_values)
            )
        hist_f0 = f0_values
        method_base = "melody_peak"

    # ---- Build octave-folded pitch-class histogram ----
    cents = hz_to_cents(hist_f0, ref_hz)
    folded = np.mod(cents, 1200)                     # fold to 0-1200 cents

    n_bins = int(1200 / bin_resolution_cents)        # 120 bins at 10-cent resolution
    hist, edges = np.histogram(folded, bins=n_bins, range=(0, 1200))

    # Light smoothing to suppress single-bin noise spikes
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    kernel /= kernel.sum()
    hist_smooth = np.convolve(hist, kernel, mode="same").astype(float)

    # ---- Find the dominant peak = Sa ----
    peak_bin = int(np.argmax(hist_smooth))
    peak_cents = edges[peak_bin] + bin_resolution_cents / 2.0

    tonic_hz = ref_hz * (2 ** (peak_cents / 1200))
    # Normalize to a sensible vocal octave (80–450 Hz)
    while tonic_hz < 80:
        tonic_hz *= 2
    while tonic_hz > 450:
        tonic_hz /= 2

    # ---- Confidence: dominant peak vs second-strongest peak ----
    # Sort all bins descending and compute the ratio winner / runner-up.
    sorted_heights = np.sort(hist_smooth)[::-1]
    if sorted_heights[0] > 0 and len(sorted_heights) > 1:
        margin = sorted_heights[0] / (sorted_heights[1] + 1e-9)
        confidence = min(0.98, 0.3 + 0.25 * (margin - 1.0))
    else:
        confidence = 0.5

    logger.info(
        "Tonic detected: %.1f Hz (%s) | method=%s | confidence=%.2f",
        tonic_hz, librosa.hz_to_note(tonic_hz), method_base, confidence
    )

    # ---- Build a candidate list for the UI (top 5 peaks) ----
    # Find local maxima for the dropdown display
    candidates = []
    mean_h = hist_smooth.mean()
    for i in range(1, len(hist_smooth) - 1):
        if (hist_smooth[i] > hist_smooth[i - 1] and
                hist_smooth[i] > hist_smooth[i + 1] and
                hist_smooth[i] > mean_h * 0.3):
            c_cents = edges[i] + bin_resolution_cents / 2.0
            c_hz = ref_hz * (2 ** (c_cents / 1200))
            while c_hz < 80:
                c_hz *= 2
            while c_hz > 450:
                c_hz /= 2
            candidates.append({
                "cents": c_cents,
                "tonic_hz": round(float(c_hz), 2),
                "tonic_note": librosa.hz_to_note(c_hz),
                "height": float(hist_smooth[i]),
                "base_score": round(float(hist_smooth[i]) / (sorted_heights[0] + 1e-9), 3),
                "pa_bonus": 0.0,
                "stability": 1.0,
                "total_score": round(float(hist_smooth[i]) / (sorted_heights[0] + 1e-9), 3),
            })

    candidates.sort(key=lambda c: c["height"], reverse=True)

    return {
        "tonic_hz": round(float(tonic_hz), 2),
        "tonic_note": librosa.hz_to_note(tonic_hz),
        "confidence": round(confidence, 3),
        "method": method_base,
        "candidates": candidates[:5],
    }


# Keep for backward compatibility — wraps the main function
def estimate_tonic_from_histogram(f0_values: np.ndarray, ref_hz: float = 55.0,
                                   bin_resolution_cents: float = 10.0) -> float:
    """Legacy entry point. Returns tonic Hz."""
    result = estimate_tonic_with_validation(f0_values, ref_hz, bin_resolution_cents)
    return result["tonic_hz"]


def cents_relative_to_tonic(f0_values: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Re-expresses every pitch as cents above/below the tonic, folded to 0-1200."""
    cents = 1200 * np.log2(f0_values / tonic_hz)
    return np.mod(cents, 1200)
