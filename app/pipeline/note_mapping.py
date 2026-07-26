"""
note_mapping.py
Maps tonic-relative cents onto the 12 semitone swara positions used in
Hindustani/Carnatic theory, and builds the pitch-class ("swara") histogram
that is the core feature for raga matching.

The 12 positions (just-intonation-ish, but we bucket in equal temperament
for simplicity/robustness -- most computational raga-ID papers do the same):

  0    S   (Shadja)
  100  r   (Komal Rishabh)
  200  R   (Shuddha Rishabh)
  300  g   (Komal Gandhar)
  400  G   (Shuddha Gandhar)
  500  m   (Shuddha Madhyam)
  600  M   (Teevra Madhyam)
  700  P   (Pancham)
  800  d   (Komal Dhaivat)
  900  D   (Shuddha Dhaivat)
  1000 n   (Komal Nishad)
  1100 N   (Shuddha Nishad)
"""
import numpy as np

SWARA_LABELS = ["S", "r", "R", "g", "G", "m", "M", "P", "d", "D", "n", "N"]
SWARA_CENTS = np.arange(12) * 100  # 0,100,...,1100


def build_swara_histogram(cents_relative: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Assigns each pitch observation (already folded to 0-1200 cents relative to
    tonic) to its nearest of the 12 semitone bins and returns a 12-length
    histogram (index 0 = S, 1 = r, ... 11 = N).
    """
    # nearest swara bin, wrapping around the octave
    diffs = np.abs(cents_relative[:, None] - SWARA_CENTS[None, :])
    diffs = np.minimum(diffs, 1200 - diffs)  # circular distance
    nearest = np.argmin(diffs, axis=1)

    hist = np.bincount(nearest, minlength=12).astype(float)
    if normalize:
        hist = hist / (hist.sum() + 1e-9)
    return hist


def dominant_notes(hist: np.ndarray, top_n: int = 5) -> list[tuple[str, float]]:
    """Returns the top_n swaras by occurrence, as (label, proportion) pairs."""
    order = np.argsort(hist)[::-1][:top_n]
    return [(SWARA_LABELS[i], float(hist[i])) for i in order if hist[i] > 0]


def implied_scale(hist: np.ndarray, presence_threshold: float = 0.03) -> list[str]:
    """
    Notes whose presence exceeds a small threshold are considered part of the
    performed scale (as opposed to passing/ornamental tones).
    """
    return [SWARA_LABELS[i] for i in range(12) if hist[i] >= presence_threshold]
