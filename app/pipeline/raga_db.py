"""
raga_db.py
Raga template database for matching — now includes Borgeet-specific ragas.

Each raga template is expressed as which of the 12 swara positions are
*allowed* (its scale / that+/avarohana note set), plus an optional "vadi"
(most important note) and "samvadi" (second most important note) which
computational raga-ID literature treats as strong priors on the expected
peak of the pitch histogram.

Index mapping matches note_mapping.SWARA_LABELS:
  0=S 1=r 2=R 3=g 4=G 5=m 6=M 7=P 8=d 9=D 10=n 11=N

Borgeet raga sources:
  - atributetosankaradeva.org (Borgeet raga documentation)
  - sangeetgalaxy.co.in (Sankari Sangeet musicology)
  - nilacharai.com (Borgeet raga analysis)

Regional ragas (Bhatiyali, Belovar, Kau) are TBD — marked as placeholders
until verified against labeled recordings.
"""

# ---- Standard Hindustani ragas (original seed set) ----

_HINDUSTANI_RAGAS = {
    "Bhairav": {
        "notes": [0, 1, 4, 5, 7, 8, 11],       # S r G m P d N
        "vadi": "d", "samvadi": "r",
        "thaat": "Bhairav",
        "time_period": "early_morning",
        "borgeet": False,
        "description": "Hindustani Bhairav. Uses Komal Re and Komal Dha.",
    },
    "Bhairavi": {
        "notes": [0, 1, 3, 5, 7, 8, 10],       # S r g m P d n (all komal)
        "vadi": "m", "samvadi": "S",
        "thaat": "Bhairavi",
        "time_period": "early_morning",
        "borgeet": False,
        "description": "Hindustani Bhairavi. Uses all komal swaras.",
    },
    "Dhanashree": {
        "notes": [0, 2, 3, 5, 7, 8, 10],       # S R g m P d n 
        "vadi": "P", "samvadi": "S",
        "thaat": "Asavari",
        "time_period": "afternoon",
        "borgeet": False,
        "description": "Hindustani Dhanashree (Bhimpalasi/Asavari based). Uses Komal Dha, unlike Borgeet Dhanashri which uses Shuddha Dha (Kafi thaat).",
    },
    "Asavari": {
        "notes": [0, 2, 3, 5, 7, 8, 10],       # S R g m P d n
        "vadi": "d", "samvadi": "g",
        "thaat": "Asavari",
        "time_period": "late_morning",
        "borgeet": False,
        "description": "Hindustani Asavari. Anchors heavily on Komal Dha and Komal Ga, unlike Borgeet Asowari which anchors on Pa and Re.",
    },
    "Shree": {
        "notes": [0, 1, 4, 6, 7, 8, 11],       # S r G M P d N
        "vadi": "r", "samvadi": "P",
        "thaat": "Purvi",
        "time_period": "sunset",
        "borgeet": False,
        "description": "Hindustani Shree. Purvi thaat (Komal Re, Komal Dha, Teevra Ma). Drastically different from Borgeet Sri (Kafi thaat).",
    },
    "Gauri": {
        "notes": [0, 1, 4, 5, 7, 8, 11],       # S r G m P d N
        "vadi": "r", "samvadi": "P",
        "thaat": "Bhairav",
        "time_period": "evening",
        "borgeet": False,
        "description": "Hindustani Gauri. Bhairav thaat based, unlike Borgeet Gauri which is pentatonic S R m P n.",
    },
    "Basant": {
        "notes": [0, 1, 4, 5, 6, 7, 8, 11],    # S r G m M P d N (uses both Ma)
        "vadi": "S", "samvadi": "P",
        "thaat": "Purvi",
        "time_period": "spring",
        "borgeet": False,
        "description": "Hindustani Basant. Uses both Shuddha Ma and Teevra Ma, and Komal Dha. Borgeet Basanta skips Shuddha Ma.",
    },
    "Suha": {
        "notes": [0, 2, 3, 5, 7, 9, 10],       # S R g m P D n
        "vadi": "m", "samvadi": "S",
        "thaat": "Kafi",
        "time_period": "night",
        "borgeet": False,
        "description": "Hindustani Suha. Kanada family.",
    },
    "Ahir Bhairav": {
        "notes": [0, 1, 4, 5, 7, 9, 10],       # S r G m P D n
        "vadi": "m", "samvadi": "S",
        "thaat": "Bhairav",
        "time_period": "morning",
        "borgeet": False,
        "description": "Hindustani Ahir Bhairav.",
    },
}

# ---- Borgeet-specific ragas ----
# Composed by Srimanta Sankardev (S) and/or Madhavdev (M) in 15th-16th century
# Categories:
#   "mela"  = open/unrestricted raga, sung WITHOUT fixed tala (free rhythm)
#   "bandha" = restricted raga, performed WITH specific tala structure

_BORGEET_RAGAS = {
    "Kalyan": {
        "notes": [0, 2, 4, 6, 7, 9, 11],       # S R G M P D N (teevra Ma)
        "vadi": "G", "samvadi": "D",
        "thaat": "Kalyan",
        "time_period": "dawn_noon",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "One of the most revered evening ragas in Borgeet and Ankia Naat. "
                       "As documented in the Barpeta Thul notation (Mrs. Khana Das, Shodhganga p.132), "
                       "prominent notes are Ga, Dha, and Re (unlike Hindustani Kalyan where Ni is Samvadi). "
                       "Features signature phrases like P-D-M-G-R and P-D-S' (skipping Ni in ascent).",
    },
    "Dhanashri": {
        "notes": [0, 2, 3, 5, 7, 9, 10],       # S R g m P D n
        "vadi": "P", "samvadi": "g",
        "thaat": "Kafi",
        "time_period": "noon_sunset",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "The most frequently used raga in Borgeet (12 compositions). "
                       "Borgeet Dhanashri uniquely uses Kafi thaat (Shodhganga p.125), "
                       "differentiating it from Hindustani Dhanashree (Bhairavi/Asavari thaats).",
        "transitions": [
            (7, 10),  # P → n (opening: P-n-S-g)
            (10, 0),  # n → S
            (0, 3),   # S → g
            (3, 5),   # g → m  (Ma Pa Dha Pa Ma phrase)
            (5, 7),   # m → P
            (7, 9),   # P → D
            (2, 3),   # R → g  (Re Ga Re Sa descent)
        ],
    },
    "Gauri": {
        "notes": [0, 2, 5, 7, 10],             # S R m P n (Pentatonic / Audav-Audav omitting Ga and Dha)
        "vadi": "P", "samvadi": "n",
        "thaat": "Kafi / Pentatonic (Sattriya)",
        "time_period": "dawn_noon",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "Primary raga of both saints. As documented in Borgeet Mukur (Mahanta, 2009, p.109), "
                       "Sattriya Gauri is a pentatonic (Audav-Audav) raga using S-R-m-P-n (omitting Ga and Dha entirely, "
                       "unlike Hindustani Bhairav-thaat Gauri). Structured in Juroni, Uroni, and Ghuroni with signature "
                       "phrases like P-n-P, n-S'-R', and R-m-R-S.",
    },
    "Ahir_Bhairav": {
        "notes": [0, 1, 4, 5, 7, 9, 10],       # S r G m P D n
        "vadi": "P", "samvadi": "r",
        "thaat": "Marwa-derived / Bhairav",
        "time_period": "dawn_noon",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev"],
        "description": "Known as 'Ahir' or 'Ahira' in Borgeet and Sattriya tradition (Goswami, 1997, p.91). Morning raga "
                       "anchored heavily on Pancham (opening signature P-D-N-D-P) and Rishabh (weaving lower-octave descent "
                       "G-R-S-n and climb n-S-R-G-M-G-R). Distinct from Hindustani Ahir Bhairav phrasing.",
    },
    "Ahir": {
        "notes": [0, 1, 4, 5, 7, 9, 10],       # S r G m P D n
        "vadi": "P", "samvadi": "r",
        "thaat": "Marwa-derived",
        "time_period": "dawn_noon",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev"],
        "description": "Also called Ahira in Shodhganga. Often conflated with Hindustani Ahir Bhairav, "
                       "but Borgeet Ahir uses komal Rishabh and komal Nishad while anchoring heavily on Pancham. "
                       "Expresses morning awakening.",
        "transitions": [
            (7, 9),   # P → D  (opening: Pa - Dha Ni Dha)
            (9, 11),  # D → N
            (11, 9),  # N → D
            (4, 2),   # G → R  (Ga Re Sa Ni descent)
            (2, 0),   # R → S
            (0, 11),  # S → N (lower octave: Ni Sa Re)
            (5, 4),   # m → G (Ma Ga Re approach)
        ],
    },
    "Asowari": {
        "notes": [0, 2, 3, 5, 7, 8, 10],       # S R g m P d n
        "vadi": "P", "samvadi": "R",
        "thaat": "Asavari",
        "time_period": "evening",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "An evening raga of deep devotion and plea. The Borgeet tradition "
                       "anchors Asowari on Pancham (Vadi) and Rishabh (Samvadi), distinct from "
                       "Hindustani Asavari (Vadi Dha, Samvadi Ga).",
        "vivadi_notes": [4, 9, 11],  # shuddha G, shuddha D, shuddha N forbidden
        "transitions": [
            (7, 0),   # P → S  (opening: Pa - Sa)
            (0, 2),   # S → R  (Sa - Re - Sa, core phrase)
            (2, 0),   # R → S
            (8, 7),   # d → P  (Dha approach to Pa)
            (5, 7),   # m → P  (Ma Pa Dha Pa Ma)
            (7, 8),   # P → d
            (10, 8),  # n → d
        ],
    },
    "Kedara": {
        "notes": [0, 2, 4, 5, 7, 9, 11],       # S R G m P D N
        "vadi": "m", "samvadi": "S",
        "thaat": "Kalyan-adjacent",
        "time_period": "noon_sunset",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev"],
        "description": "Evening raga. Uses shuddha Ma (unlike Yaman/Kalyana) "
                       "but shares the bright upper register.",
    },
    "Sri": {
        "notes": [0, 2, 3, 5, 7, 9, 10],       # S R g m P D n (Sattriya Janya of Dhanashri / Kafi thaat family)
        "vadi": "R", "samvadi": "P",
        "thaat": "Kafi (Sattriya Janya of Dhanashri)",
        "time_period": "noon_sunset",
        "borgeet": True,
        "borgeet_category": "mela",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "In the Borgeet and Sattriya tradition, Raga Sri is a solemn (gambhir), purbangavadi raga "
                       "derived from Raga Dhanashri (Kafi thaat family), distinct from Hindustani Poorvi-thaat Shree. "
                       "As documented in Borgeet Mukur (Mahanta, 2009, p.204), it is structured in Juroni, Uroni, and Ghuroni "
                       "with signature phrases like P-n-S' and n-D-P-m-g-R-S emphasizing Rishabh and Pancham.",
    },
    "Basanta": {
        "notes": [0, 1, 4, 6, 7, 9, 11],       # S r G M P D N
        "vadi": "G", "samvadi": "P",
        "thaat": "Purvi",
        "time_period": "noon_sunset",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "Raga of springtime and joy. Unlike Hindustani Basant (which uses both Mas), "
                       "Sattriya Basanta is modeled closer to Purvi thaat (Goswami 1997).",
        "transitions": [
            (0, 2),   # S → R  (pentatonic ascent: S-R-G-P-D)
            (2, 4),   # R → G
            (4, 7),   # G → P  (skip Ma in ascent)
            (7, 9),   # P → D
            (9, 7),   # D → P  (descent)
            (4, 5),   # G → m  (weaving: G-M-G-M-P-M-G-R)
            (5, 4),   # m → G
        ],
    },
    # "Turabasanta": {
    #     "notes": [0, 1, 4, 6, 7, 8, 11],       # S r G M P d N
    #     "vadi": "G", "samvadi": "d",
    #     "thaat": "Purvi-adjacent",
    #     "time_period": "noon_sunset",
    #     "borgeet": True,
    #     "borgeet_category": "mela",
    #     "regional": False,
    #     "composers": ["Sankardev"],
    #     "description": "A variant of Basanta unique to the Borgeet tradition. "
    #                    "Free-rhythm (Mela) performance.",
    # },
    "Puriya Dhanashree": {
        "notes": [0, 1, 4, 6, 7, 8, 11],       # S r G M P d N
        "vadi": "P", "samvadi": "S",
        "thaat": "Purvi",
        "time_period": "evening",
        "borgeet": False,
        "borgeet_category": None,
        "regional": False,
        "composers": [],
        "description": "An evening raga full of yearning and devotion. It has the same notes as Hindustani Shree, but emphasizes Pa heavily.",
    },
    "Mahur": {
        "notes": [0, 2, 3, 5, 7, 8, 10],       # S R g m P d n (Komal Ga, Dha, Ni)
        "vadi": "P", "samvadi": "S",
        "thaat": "Asavari / Bhairavi (Sattriya)",
        "time_period": "noon_sunset",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "As documented in Borgeet Mukur (Mahanta, 2009, p.126), Sattriya Mahur uses komal Ga, "
                       "komal Dha, and komal Ni (unlike Bilaval-adjacent Hindustani interpretations). "
                       "Structured in Jurani, Uroni, and Ghuroni with characteristic phrases like M-P-d-P, M-P-n-P, "
                       "and S-R-M-P-M-g-M-S centered around Pancham and Shadja.",
    },
    # "Nat_Mallar": {
    #     "notes": [0, 2, 5, 7, 10],             # S R m P n (Pentatonic / Audav-Audav omitting Ga and Dha)
    #     "vadi": "R", "samvadi": "m",
    #     "thaat": "Kafi / Pentatonic (Sattriya)",
    #     "time_period": "night",
    #     "borgeet": True,
    #     "borgeet_category": "bandha",
    #     "borgeet_alias": "Mallar",
    #     "regional": False,
    #     "composers": ["Sankardev"],
    #     "description": "A rain/monsoon raga of night/evening. As documented in Borgeet Mukur (Mahanta, 2009, p.194), "
    #                    "Sattriya Mallar (Nata-Mallar) is a pentatonic (Audav-Audav) raga using S-R-m-P-n (omitting Ga and Dha "
    #                    "entirely, unlike Hindustani Nat Malhar which uses shuddha Ga/Dha). Structured in Juroni, Uroni, "
    #                    "and Ghuroni with characteristic meends and phrases like n-S-R~, m~-R, m~-R-m-P~, and P-m-R.",
    # },
    # "Suhai": {
    #     "notes": [0, 2, 3, 5, 7, 9, 10],       # S R g m P D n (Komal Ga, Komal Ni)
    #     "vadi": "P", "samvadi": "S",
    #     "thaat": "Kafi (Sattriya)",
    #     "time_period": "night",
    #     "borgeet": True,
    #     "borgeet_category": "bandha",
    #     "regional": False,
    #     "composers": ["Sankardev"],
    #     "description": "Night raga of deep devotion (Shodhganga p.126-127). As documented in Borgeet Mukur (Mahanta, 2009, p.177), "
    #                    "Suhai uses Kafi-thaat swaras (komal Ga, komal Ni) with a characteristic ascent leaping from Ga to Pa (skipping Ma) "
    #                    "and Dha to upper Sa (skipping Ni). Structured in Juroni, Uroni, and Ghuroni anchored strongly on Pancham.",
    # },

    # ---- Regional / Unique Sattriya ragas ----
    # These do NOT map directly to standard Hindustani ragas.
    # Note sets are TBD — marked as placeholders until verified against
    # the user's labeled Borgeet recordings.

    "Bhatiyali": {
        "notes": None,  # TBD — unique to Sattriya tradition
        "vadi": None, "samvadi": None,
        "thaat": "Regional",
        "time_period": "noon_sunset",
        "borgeet": True,
        "borgeet_category": "mela",
        "regional": True,
        "composers": ["Sankardev"],
        "description": "Distinct Sattriya raga with no direct classical mapping. "
                       "Note set to be verified against labeled recordings.",
        "status": "tbd",
    },
    "Belovar": {
        "notes": None,  # TBD
        "vadi": None, "samvadi": None,
        "thaat": "Regional",
        "time_period": "evening",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": True,
        "composers": ["Madhavdev"],
        "description": "Regional Sattriya raga. Note set to be verified.",
        "status": "tbd",
    },
    "Kau": {
        "notes": None,  # TBD
        "vadi": None, "samvadi": None,
        "thaat": "Regional",
        "time_period": "dawn_noon",
        "borgeet": True,
        "borgeet_category": "mela",
        "regional": True,
        "composers": ["Sankardev"],
        "description": "Regional Sattriya raga. Note set to be verified.",
        "status": "tbd",
    },
}


def get_all_ragas() -> dict:
    """Returns the combined raga database (standard + Borgeet)."""
    combined = {}
    combined.update(_HINDUSTANI_RAGAS)
    combined.update(_BORGEET_RAGAS)
    return combined


def get_borgeet_ragas() -> dict:
    """Returns only the Borgeet-specific ragas (excluding TBD regional ones)."""
    return {
        name: template for name, template in _BORGEET_RAGAS.items()
        if template.get("status") != "tbd"
    }


def get_matchable_ragas(borgeet_mode: bool = False) -> dict:
    """
    Returns ragas that can be used for template matching.

    In borgeet_mode=True, returns ONLY the authentic Borgeet/Sattriya definitions.
    In borgeet_mode=False, returns ONLY the Hindustani definitions.
    """
    result = {}
    target_ragas = _BORGEET_RAGAS if borgeet_mode else _HINDUSTANI_RAGAS

    for name, template in target_ragas.items():
        # Skip ragas with undefined note sets (TBD regional ragas)
        if template.get("notes") is None:
            continue

        display_template = dict(template)
        display_template["display_name"] = name
        result[name] = display_template

    return result


# Kept for backward compatibility with raga_matcher.py
RAGA_TEMPLATES = get_all_ragas()


def template_to_vector(template: dict) -> "np.ndarray":
    """Converts a raga template's note list to a 12-dim probability vector."""
    import numpy as np
    from .note_mapping import SWARA_LABELS
    
    notes = template.get("notes")
    if notes is None:
        return np.zeros(12)
        
    v = np.zeros(12)
    for idx in notes:
        v[idx] = 1.0
        
    # Weight the Vadi (most important) and Samvadi (second most important) notes heavier.
    # Phase 1 upgrade: increased Vadi weight to +1.5 and Samvadi to +0.75 for stronger anchoring.
    vadi = template.get("vadi")
    samvadi = template.get("samvadi")
    
    if vadi in SWARA_LABELS:
        v[SWARA_LABELS.index(vadi)] += 1.5
    if samvadi in SWARA_LABELS:
        v[SWARA_LABELS.index(samvadi)] += 0.75
        
    # Vivadi (forbidden) notes: zero out any note index explicitly forbidden or not in the allowed scale
    vivadi_notes = template.get("vivadi_notes") or [i for i in range(12) if i not in notes]
    for vivadi_idx in vivadi_notes:
        v[vivadi_idx] = 0.0
        
    return v / (np.linalg.norm(v) + 1e-9)

