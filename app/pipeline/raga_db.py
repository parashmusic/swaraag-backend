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

_STANDARD_RAGAS = {
    "Yaman": {
        "notes": [0, 2, 4, 6, 7, 9, 11],       # S R G M' P D N (teevra Ma)
        "vadi": "G", "samvadi": "N",
        "thaat": "Kalyan",
        "borgeet": False,
    },
    "Bhairav": {
        "notes": [0, 1, 4, 5, 7, 8, 11],       # S r G m P d N
        "vadi": "d", "samvadi": "r",
        "thaat": "Bhairav",
        "borgeet": False,
    },
    "Bhupali": {
        "notes": [0, 2, 4, 7, 9],              # S R G P D (pentatonic)
        "vadi": "G", "samvadi": "D",
        "thaat": "Kalyan",
        "borgeet": False,
    },
    # "Kafi": {
    #     "notes": [0, 2, 3, 5, 7, 9, 10],       # S R g m P D n
    #     "vadi": "P", "samvadi": "S",
    #     "thaat": "Kafi",
    #     "borgeet": False,
    # },
    "Bhairavi": {
        "notes": [0, 1, 3, 5, 7, 8, 10],       # S r g m P d n (all komal)
        "vadi": "m", "samvadi": "S",
        "thaat": "Bhairavi",
        "borgeet": False,
    },
    "Desh": {
        "notes": [0, 2, 4, 5, 7, 9, 10, 11],   # S R G m P D n N (mixed nishad)
        "vadi": "R", "samvadi": "n",
        "thaat": "Khamaj",
        "borgeet": False,
    },
    "Malkauns": {
        "notes": [0, 3, 5, 8, 10],             # S g m d n (pentatonic)
        "vadi": "m", "samvadi": "S",
        "thaat": "Bhairavi-derived",
        "borgeet": False,
    },
    "Khamaj": {
        "notes": [0, 2, 4, 5, 7, 9, 10, 11],
        "vadi": "G", "samvadi": "n",
        "thaat": "Khamaj",
        "borgeet": False,
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
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "One of the most frequently used ragas in Borgeet. "
                       "As documented in Sastriya Sankari Kantha Sangeet (Bhuyan, 2003, p.88), Ga and Ni are flat, "
                       "with Ga, Ni, and Pa acting as the prominent anchor notes. Structured in four traditional "
                       "movements (Dhrum, Urran, Sanchar, Ghuran) with signature phrases like P-n-S-g and M-P-M-g-R.",
    },
    "Gauri": {
        "notes": [0, 2, 5, 7, 10],             # S R m P n (Pentatonic / Audav-Audav omitting Ga and Dha)
        "vadi": "P", "samvadi": "n",
        "thaat": "Kafi / Pentatonic (Sattriya)",
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
        "thaat": "Marwa-derived / Bhairav",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev"],
        "description": "Traditional Assamese name for Ahira / Ahir Bhairav in Borgeet (Goswami, 1997, p.91). Morning raga "
                       "anchored heavily on Pancham (opening signature P-D-N-D-P) and Rishabh (weaving lower-octave descent "
                       "G-R-S-n and climb n-S-R-G-M-G-R).",
    },
    "Asowari": {
        "notes": [0, 2, 3, 5, 7, 8, 10],       # S R g m P d n
        "vadi": "P", "samvadi": "R",
        "thaat": "Asavari",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "In the Sattriya and Borgeet tradition, Asowari is characterized by prominent notes Re, Pa, and Dha "
                       "(unlike Hindustani Asavari which centers on komal dha and komal ga). Structured in traditional "
                       "movements (Dhrum, Urran, Sanchar, Ghuran) with characteristic leaps like P-S-R and M-P-D.",
    },
    "Kedara": {
        "notes": [0, 2, 4, 5, 7, 9, 11],       # S R G m P D N
        "vadi": "m", "samvadi": "S",
        "thaat": "Kalyan-adjacent",
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
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "Spring raga with a joyous, celebratory mood. As documented in Bargit : Parampara Aru Parivesan Paddhati "
                       "(Goswami, 1997, p.94), Borgeet Basanta features a characteristic pentatonic ascent omitting Ma/Ni (S-R-G-P-D-S') "
                       "and a weaving descent (D-N-S-N-P-M, G-M-G-M-P-M-G-R) anchored strongly on Gandhar, Pancham, and Sa.",
    },
    "Turabasanta": {
        "notes": [0, 1, 4, 6, 7, 8, 11],       # S r G M P d N
        "vadi": "G", "samvadi": "d",
        "thaat": "Purvi-adjacent",
        "borgeet": True,
        "borgeet_category": "mela",
        "regional": False,
        "composers": ["Sankardev"],
        "description": "A variant of Basanta unique to the Borgeet tradition. "
                       "Free-rhythm (Mela) performance.",
    },
    "Puriya Dhanashree": {
        "notes": [0, 1, 4, 6, 7, 8, 11],       # S r G M P d N
        "vadi": "P", "samvadi": "S",
        "thaat": "Purvi",
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
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev", "Madhavdev"],
        "description": "As documented in Borgeet Mukur (Mahanta, 2009, p.126), Sattriya Mahur uses komal Ga, "
                       "komal Dha, and komal Ni (unlike Bilaval-adjacent Hindustani interpretations). "
                       "Structured in Jurani, Uroni, and Ghuroni with characteristic phrases like M-P-d-P, M-P-n-P, "
                       "and S-R-M-P-M-g-M-S centered around Pancham and Shadja.",
    },
    "Nat_Mallar": {
        "notes": [0, 2, 5, 7, 10],             # S R m P n (Pentatonic / Audav-Audav omitting Ga and Dha)
        "vadi": "R", "samvadi": "m",
        "thaat": "Kafi / Pentatonic (Sattriya)",
        "borgeet": True,
        "borgeet_category": "bandha",
        "borgeet_alias": "Mallar",
        "regional": False,
        "composers": ["Sankardev"],
        "description": "A rain/monsoon raga of night/evening. As documented in Borgeet Mukur (Mahanta, 2009, p.194), "
                       "Sattriya Mallar (Nata-Mallar) is a pentatonic (Audav-Audav) raga using S-R-m-P-n (omitting Ga and Dha "
                       "entirely, unlike Hindustani Nat Malhar which uses shuddha Ga/Dha). Structured in Juroni, Uroni, "
                       "and Ghuroni with characteristic meends and phrases like n-S-R~, m~-R, m~-R-m-P~, and P-m-R.",
    },
    "Suhai": {
        "notes": [0, 2, 3, 5, 7, 9, 10],       # S R g m P D n (Komal Ga, Komal Ni)
        "vadi": "P", "samvadi": "S",
        "thaat": "Kafi (Sattriya)",
        "borgeet": True,
        "borgeet_category": "bandha",
        "regional": False,
        "composers": ["Sankardev"],
        "description": "Night raga of deep devotion (Shodhganga p.126-127). As documented in Borgeet Mukur (Mahanta, 2009, p.177), "
                       "Suhai uses Kafi-thaat swaras (komal Ga, komal Ni) with a characteristic ascent leaping from Ga to Pa (skipping Ma) "
                       "and Dha to upper Sa (skipping Ni). Structured in Juroni, Uroni, and Ghuroni anchored strongly on Pancham.",
    },

    # ---- Regional / Unique Sattriya ragas ----
    # These do NOT map directly to standard Hindustani ragas.
    # Note sets are TBD — marked as placeholders until verified against
    # the user's labeled Borgeet recordings.

    "Bhatiyali": {
        "notes": None,  # TBD — unique to Sattriya tradition
        "vadi": None, "samvadi": None,
        "thaat": "Regional",
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
    combined.update(_STANDARD_RAGAS)
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
    Returns ragas that can be used for template matching
    (i.e., have defined note sets).

    In borgeet_mode, includes Borgeet ragas and uses Borgeet aliases
    (e.g., shows "Kalyana" instead of "Yaman").
    """
    result = {}
    all_ragas = get_all_ragas()

    for name, template in all_ragas.items():
        # Skip ragas with undefined note sets (TBD regional ragas)
        if template.get("notes") is None:
            continue

        # In borgeet_mode, prefer Borgeet names
        if borgeet_mode and "borgeet_alias" in template:
            # Create a copy with the Borgeet name
            alias_name = template["borgeet_alias"]
            alias_template = dict(template)
            alias_template["borgeet"] = True
            alias_template["display_name"] = alias_name
            result[alias_name] = alias_template
        else:
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
    # This embeds the raga's structural hierarchy directly into the cosine similarity calculation,
    # rewarding performances that emphasize the correct notes.
    vadi = template.get("vadi")
    samvadi = template.get("samvadi")
    
    if vadi in SWARA_LABELS:
        v[SWARA_LABELS.index(vadi)] += 0.5
    if samvadi in SWARA_LABELS:
        v[SWARA_LABELS.index(samvadi)] += 0.25
        
    return v / (np.linalg.norm(v) + 1e-9)
