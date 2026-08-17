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


def build_transition_matrix(cents_relative: np.ndarray) -> np.ndarray:
    """
    Builds a 12x12 bigram transition matrix of note changes.
    Returns a normalized matrix where M[i][j] is the probability of moving from note i to note j.
    """
    diffs = np.abs(cents_relative[:, None] - SWARA_CENTS[None, :])
    diffs = np.minimum(diffs, 1200 - diffs)
    nearest = np.argmin(diffs, axis=1)

    # Calculate transitions
    transitions = np.zeros((12, 12))
    
    # We only care about when the note actually changes
    prev_note = nearest[0]
    for current_note in nearest[1:]:
        if current_note != prev_note:
            transitions[prev_note, current_note] += 1
            prev_note = current_note

    # Normalize by row (so each row sums to 1)
    row_sums = transitions.sum(axis=1)
    # Avoid division by zero
    transitions[row_sums > 0] = transitions[row_sums > 0] / row_sums[row_sums > 0, np.newaxis]
    
    return transitions


def extract_alapa_segment(f0_clean: np.ndarray, times: np.ndarray, max_duration=15.0) -> np.ndarray:
    """
    Extract the first ~15 seconds as the alapa segment.
    The alapa in Borgeet is pre-composed and emphasizes Vadi/Samvadi.
    """
    mask = times <= max_duration
    # ensure we don't index out of bounds
    limit = min(len(f0_clean), np.sum(mask))
    return f0_clean[:limit]

