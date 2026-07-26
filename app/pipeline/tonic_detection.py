"""
tonic_detection.py
Estimates the tonic (Sa) pitch of a performance.

Indian art/folk music is heterophonic around a drone (tanpura/harmonium/ektara)
fixed at Sa. The tonic is almost always the most frequently-occurring and most
sustained pitch class in the recording (or one of Sa/Pa which are the drone's
two strings) -- this is the standard multi-pitch histogram approach used in
CompMusic research. Essentia's TonicIndianArtMusic algorithm is a more
sophisticated version of the same idea (looks at the drone's spectral energy
directly) -- but does not install natively on Windows, so we implement an
enhanced version here with Sa/Pa harmonic validation.

Upgrade from v0.1: The naive "highest histogram peak = Sa" fails whenever a
raga's vadi/samvadi is sung more than Sa itself (common in Borgeet — e.g.
Dhanashri emphasises Pa/Ga, not Sa). The new algorithm validates tonic
candidates against their perfect-fifth (Pa) harmonic, and scores candidates
by both frequency AND stability.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def hz_to_cents(f0_hz: np.ndarray, ref_hz: float = 55.0) -> np.ndarray:
    """Convert Hz to cents relative to a reference (default A1=55Hz, arbitrary anchor)."""
    return 1200 * np.log2(f0_hz / ref_hz)


def _build_pitch_histogram(f0_values: np.ndarray, ref_hz: float = 55.0,
                            bin_resolution_cents: float = 10.0):
    """
    Builds a fine-grained, octave-folded pitch histogram and returns the
    smoothed histogram + bin edges.
    """
    cents = hz_to_cents(f0_values, ref_hz)
    # Fold to a single octave (0-1200 cents) to find the dominant pitch class
    folded = np.mod(cents, 1200)

    n_bins = int(1200 / bin_resolution_cents)
    hist, edges = np.histogram(folded, bins=n_bins, range=(0, 1200))

    # Smooth histogram to avoid picking noise spikes
    kernel = np.array([0.25, 0.5, 1.0, 0.5, 0.25])
    kernel /= kernel.sum()
    hist_smooth = np.convolve(hist, kernel, mode="same")

    return hist_smooth, edges, bin_resolution_cents


def _find_top_peaks(hist_smooth: np.ndarray, edges: np.ndarray,
                     bin_resolution_cents: float, n_peaks: int = 5) -> list[dict]:
    """
    Finds the top N peaks in the histogram.
    Returns list of dicts with 'cents', 'height', 'bin_idx'.
    """
    # Simple peak finding: local maxima above the mean
    mean_height = hist_smooth.mean()
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if (hist_smooth[i] > hist_smooth[i - 1] and
            hist_smooth[i] > hist_smooth[i + 1] and
            hist_smooth[i] > mean_height * 0.3):
            peak_cents = edges[i] + bin_resolution_cents / 2
            peaks.append({
                "cents": peak_cents,
                "height": hist_smooth[i],
                "bin_idx": i,
            })

    # Sort by height descending
    peaks.sort(key=lambda p: p["height"], reverse=True)
    return peaks[:n_peaks]


def _cents_distance_circular(c1: float, c2: float) -> float:
    """Circular distance between two cent values (0-1200 range)."""
    d = abs(c1 - c2)
    return min(d, 1200 - d)


def estimate_tonic_from_histogram(f0_values: np.ndarray, ref_hz: float = 55.0,
                                   bin_resolution_cents: float = 10.0) -> float:
    """
    LEGACY: Builds a fine-grained pitch histogram, finds the strongest peak,
    and treats it as a tonic candidate. Returns tonic in Hz.

    Kept as fallback — use estimate_tonic_with_validation() for better accuracy.
    """
    hist_smooth, edges, _ = _build_pitch_histogram(f0_values, ref_hz, bin_resolution_cents)

    peak_bin = np.argmax(hist_smooth)
    peak_cents = edges[peak_bin] + bin_resolution_cents / 2

    tonic_hz = ref_hz * (2 ** (peak_cents / 1200))
    # Fold into a sensible octave range for display (e.g. 130-260 Hz ~ typical vocal Sa)
    while tonic_hz < 100:
        tonic_hz *= 2
    while tonic_hz > 300:
        tonic_hz /= 2
    return float(tonic_hz)


def estimate_tonic_with_validation(f0_values: np.ndarray, 
                                   ref_hz: float = 55.0,
                                   bin_resolution_cents: float = 10.0,
                                   drone_f0_values: np.ndarray = None) -> dict:
    """
    Tonic detection: the most frequently occurring pitch (the histogram peak)
    is Sa. This holds true for the vast majority of Indian music recordings —
    the tonic is the anchor note that the singer returns to most often.

    The Pa harmonic check (+700 cents) is used as a CONFIDENCE BOOSTER only,
    not to override the dominant peak.

    Priority:
      1. Drone signal peaks (tanpura/harmonium holds Sa directly) — most reliable
      2. Vocal melody peak — reliable for practice/teaching/Borgeet recordings
    """
    import librosa

    # If drone pitches are provided, prefer the drone histogram ONLY if it has sufficient voiced content
    # (at least 500 frames AND at least 15% of the melody voiced frames).
    # Otherwise, it is likely just source separation noise/bleed.
    if drone_f0_values is not None and len(drone_f0_values) >= max(500, int(len(f0_values) * 0.15)):
        hist_f0 = drone_f0_values
        using_drone = True
    else:
        if drone_f0_values is not None:
            logger.warning(
                "Drone stem has too few voiced frames (%d vs melody %d), ignoring drone for tonic detection.",
                len(drone_f0_values), len(f0_values)
            )
        hist_f0 = f0_values
        using_drone = False

    hist_smooth, edges, bin_res = _build_pitch_histogram(hist_f0, ref_hz, bin_resolution_cents)
    peaks = _find_top_peaks(hist_smooth, edges, bin_res, n_peaks=8)

    if not peaks:
        tonic_hz = estimate_tonic_from_histogram(f0_values, ref_hz, bin_resolution_cents)
        return {
            "tonic_hz": tonic_hz,
            "tonic_note": librosa.hz_to_note(tonic_hz),
            "confidence": 0.2,
            "method": "naive_fallback",
            "candidates": [],
        }

    max_height = peaks[0]["height"]

    scored_candidates = []
    for i, peak in enumerate(peaks):
        # Base score: normalized peak height.
        # The tallest peak = most visited pitch = most likely Sa.
        base_score = peak["height"] / max_height

        # Pa CONFIRMATION (+700 cents): if Pa is also a strong peak, confidence rises.
        # This does NOT override the dominant peak — it only affects confidence.
        pa_target = (peak["cents"] + 700) % 1200
        pa_bonus = 0.0
        pa_found = False
        for other_peak in peaks:
            if other_peak is peak:
                continue
            dist = _cents_distance_circular(other_peak["cents"], pa_target)
            if dist < 50:
                pa_bonus = 0.4 * (other_peak["height"] / max_height)
                pa_found = True
                break

        # Ma CONFIRMATION (+500 cents): partial bonus if Ma is present.
        if not pa_found:
            ma_target = (peak["cents"] + 500) % 1200
            for other_peak in peaks:
                if other_peak is peak:
                    continue
                dist = _cents_distance_circular(other_peak["cents"], ma_target)
                if dist < 50:
                    pa_bonus = 0.15 * (other_peak["height"] / max_height)
                    break

        # Peak stability: a broad sustained peak (drone) is more reliable than
        # a narrow spike (transient ornament).
        bin_idx = peak["bin_idx"]
        width = 0
        for offset in range(1, 6):
            left = max(0, bin_idx - offset)
            right = min(len(hist_smooth) - 1, bin_idx + offset)
            if hist_smooth[left] > peak["height"] * 0.3:
                width += 1
            if hist_smooth[right] > peak["height"] * 0.3:
                width += 1
        stability_weight = 1.0 + 0.04 * width

        total_score = base_score * (1.0 + pa_bonus) * stability_weight

        cand_hz = ref_hz * (2 ** (peak["cents"] / 1200))
        while cand_hz < 80:
            cand_hz *= 2
        while cand_hz > 450:
            cand_hz /= 2

        scored_candidates.append({
            "cents": peak["cents"],
            "tonic_hz": round(float(cand_hz), 2),
            "tonic_note": librosa.hz_to_note(cand_hz),
            "height": peak["height"],
            "base_score": round(base_score, 3),
            "pa_bonus": round(pa_bonus, 3),
            "stability": round(stability_weight, 3),
            "total_score": round(total_score, 3),
        })

    scored_candidates.sort(key=lambda c: c["total_score"], reverse=True)
    winner = scored_candidates[0]
    tonic_hz = winner["tonic_hz"]

    logger.info(
        "Tonic candidates (top 5): %s",
        [(c["tonic_note"], c["total_score"]) for c in scored_candidates[:5]]
    )

    # Confidence: how dominant is the winner vs the runner-up
    if len(scored_candidates) >= 2:
        margin = winner["total_score"] / (scored_candidates[1]["total_score"] + 1e-9)
        confidence = min(0.95, 0.3 + 0.3 * (margin - 1))
    else:
        confidence = 0.5

    if winner["pa_bonus"] > 0:
        confidence = min(0.98, confidence + 0.15)

    if using_drone:
        method = "drone_peak_pa_confirmed" if winner["pa_bonus"] > 0 else "drone_peak"
        confidence = min(0.99, confidence + 0.15)
    else:
        method = "melody_peak_pa_confirmed" if winner["pa_bonus"] > 0 else "melody_peak"

    logger.info("Tonic detected: %.1f Hz (%s), confidence=%.2f, method=%s",
                tonic_hz, librosa.hz_to_note(tonic_hz), confidence, method)

    return {
        "tonic_hz": round(float(tonic_hz), 2),
        "tonic_note": librosa.hz_to_note(tonic_hz),
        "confidence": round(confidence, 3),
        "method": method,
        "candidates": scored_candidates[:5],
    }


def cents_relative_to_tonic(f0_values: np.ndarray, tonic_hz: float) -> np.ndarray:
    """Re-expresses every pitch as cents above/below the detected tonic, folded to 0-1200."""
    cents = 1200 * np.log2(f0_values / tonic_hz)
    return np.mod(cents, 1200)
