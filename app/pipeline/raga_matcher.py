"""
raga_matcher.py
Matches an extracted swara histogram against the raga template database.

Two matching strategies are provided:
  1. match_by_similarity() -- unsupervised, works with zero training data.
     Good starting point / always-available fallback.
  2. train_knn_classifier() / predict_knn() -- supervised, needs a labeled
     dataset (histogram -> raga label) e.g. built from CompMusic Saraga, or
     later, your own Borgeet/Lokgeet corpus. This is the upgrade path once
     you have real folk data: instead of a hand-built template, you extract
     the same 12-dim swara histogram (+ maybe vadi/samvadi + n-gram features
     of the note sequence) for every labeled track, and fit a classifier
     (start with KNN/SVM -- they work fine on small datasets of ~15-30
     tracks per class; a CNN on the pitch-contour "raga-gram" image is the
     next step once you have hundreds of tracks per class).

Phase 1 upgrade: now supports borgeet_mode to prioritize Borgeet ragas
and show Borgeet-specific metadata (category, composers, regional flag).
"""
import numpy as np
from .raga_db import get_matchable_ragas, template_to_vector


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def is_time_match(raga_time: str, user_time: str) -> bool:
    if not raga_time or not user_time or raga_time == "any" or user_time == "any":
        return False
    raga_time = raga_time.lower()
    user_time = user_time.lower()
    if user_time in raga_time or raga_time in user_time:
        return True
    if user_time in ["morning", "dawn", "early_morning"] and raga_time in ["dawn_noon", "early_morning"]:
        return True
    if user_time in ["afternoon", "noon"] and raga_time in ["noon_sunset", "dawn_noon"]:
        return True
    if user_time in ["evening", "dusk"] and raga_time in ["evening", "noon_sunset"]:
        return True
    if user_time in ["night", "midnight", "late_night"] and raga_time in ["night", "late_night", "evening"]:
        return True
    return False


def match_by_similarity(swara_hist: np.ndarray, transition_matrix: np.ndarray = None, top_n: int = 3,
                         borgeet_mode: bool = False, time_of_day: str = None) -> list[dict]:
    """
    Unsupervised template match: cosine similarity between the performed
    swara histogram and each raga's ideal (uniform-over-allowed-notes)
    template. Returns ranked candidates with a similarity score in [0,1].

    In borgeet_mode:
      - Uses Borgeet raga names (e.g., "Kalyana" instead of "Yaman")
      - Includes Borgeet metadata (category, composers, regional flag)
      - Applies a small boost to Borgeet-origin ragas
    """
    from .note_mapping import SWARA_LABELS

    ragas = get_matchable_ragas(borgeet_mode=borgeet_mode)
    scores = []

    # Enharmonic conflict pairs: (komal_idx, shuddha_idx)
    # If template has one but not the other, and performed histogram
    # clearly shows the opposite is stronger, penalize the score.
    CONFLICT_PAIRS = [
        (1, 2),   # r (komal Re) vs R (shuddha Re)
        (3, 4),   # g (komal Ga) vs G (shuddha Ga)
        (5, 6),   # m (shuddha Ma) vs M (teevra Ma)
        (8, 9),   # d (komal Dha) vs D (shuddha Dha)
        (10, 11), # n (komal Ni) vs N (shuddha Ni)
    ]

    for raga_name, template in ragas.items():
        template_vec = template_to_vector(template)
        if template_vec.sum() < 1e-6:
            continue  # skip TBD ragas with no defined notes

        sim = cosine_similarity(swara_hist, template_vec)

        # --- Chromatic Conflict Penalty ---
        # For each komal/shuddha pair, if the template exclusively uses one
        # variant (i.e., the other is NOT in the template) but the performed
        # histogram shows the excluded variant is clearly stronger, penalize.
        conflict_penalty = 0.0
        template_notes = set(template.get("notes", []))
        for komal_idx, shuddha_idx in CONFLICT_PAIRS:
            komal_val = swara_hist[komal_idx]
            shuddha_val = swara_hist[shuddha_idx]
            diff = abs(komal_val - shuddha_val)

            if diff > 0.03:  # Only penalize when the difference is meaningful
                # Template has shuddha but NOT komal, yet komal is stronger
                if shuddha_idx in template_notes and komal_idx not in template_notes:
                    if komal_val > shuddha_val:
                        conflict_penalty += diff * 1.5
                # Template has komal but NOT shuddha, yet shuddha is stronger
                elif komal_idx in template_notes and shuddha_idx not in template_notes:
                    if shuddha_val > komal_val:
                        conflict_penalty += diff * 1.5

        # --- Vivadi (Forbidden Note) Penalty ---
        # For pentatonic or restricted Borgeet ragas (e.g. Gauri, Mallar omitting Ga/Dha),
        # if an explicitly forbidden note appears in the performed histogram with significant energy, penalize.
        vivadi_penalty = 0.0
        vivadi_notes = template.get("vivadi_notes") or [i for i in range(12) if i not in template_notes]
        for vivadi_idx in vivadi_notes:
            vivadi_val = swara_hist[vivadi_idx]
            if vivadi_val > 0.025:  # Threshold: > 2.5% energy on a forbidden note
                vivadi_penalty += (vivadi_val - 0.025) * 2.0

        sim = max(0.0, sim - conflict_penalty - vivadi_penalty)

        # --- Transition Scoring (Phase 2A) ---
        if transition_matrix is not None and "transitions" in template:
            template_transitions = template["transitions"]
            if len(template_transitions) > 0:
                prob_sum = sum(transition_matrix[i, j] for i, j in template_transitions)
                # Average probability of the signature bigrams
                transition_score = prob_sum / len(template_transitions)
                
                # transition_score is a probability, usually 0.1 to 0.4.
                # We scale it to a 0-1 range.
                normalized_transition = min(1.0, transition_score * 3.0)
                
                # Apply as an ADDITIVE BONUS (up to +0.15) rather than a weighted average.
                # This ensures ragas with transitions don't get artificially deflated
                # compared to ragas without transitions.
                sim = min(1.0, sim + (0.15 * normalized_transition))

        # In borgeet_mode, small boost for Borgeet-origin ragas
        borgeet_boost = 0.03 if (borgeet_mode and template.get("borgeet")) else 0.0

        # Time-of-day raga prior boost
        time_boost = 0.0
        if time_of_day and is_time_match(template.get("time_period"), time_of_day):
            time_boost = 0.05

        entry = {
            "raga": template.get("display_name", raga_name),
            "thaat": template.get("thaat", "—"),
            "time_period": template.get("time_period", "—"),
            "similarity": round(sim + borgeet_boost + time_boost, 4),
            "notes": template.get("notes", []),
        }

        # Add Borgeet metadata when available
        if template.get("borgeet"):
            entry["borgeet"] = True
            entry["borgeet_category"] = template.get("borgeet_category", "—")
            entry["composers"] = template.get("composers", [])
            entry["regional"] = template.get("regional", False)
            if template.get("description"):
                entry["description"] = template["description"]

        scores.append(entry)

    scores.sort(key=lambda x: x["similarity"], reverse=True)
    total = sum(s["similarity"] for s in scores) + 1e-9
    for s in scores:
        s["confidence_pct"] = round(100 * s["similarity"] / total, 1)

    return scores[:top_n]


# ---- Supervised upgrade path (use once you have labeled training data) ----

def train_knn_classifier(X: np.ndarray, y: list[str], n_neighbors: int = 3):
    """
    X: (n_samples, 12) swara histograms (or richer feature vectors)
    y: raga labels
    Returns a fitted sklearn KNeighborsClassifier.
    """
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric="cosine")
    clf.fit(X, y)
    return clf


def predict_knn(clf, swara_hist: np.ndarray, top_n: int = 3) -> list[dict]:
    proba = clf.predict_proba(swara_hist.reshape(1, -1))[0]
    labels = clf.classes_
    ranked = sorted(zip(labels, proba), key=lambda t: t[1], reverse=True)[:top_n]
    return [{"raga": r, "confidence_pct": round(100 * p, 1)} for r, p in ranked]
