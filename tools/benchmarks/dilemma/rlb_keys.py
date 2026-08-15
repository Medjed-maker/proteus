"""Key functions for the reverse-lookup baseline ladder.

Three levels of abstraction over a Greek surface form, each one the index key
for one rung of the ladder:

    b1_key       accents/breathings/subscripts dropped, case and sigma folded.
                 This is exactly the key space of Dilemma's own spell_index.db,
                 so B1 measures "what does the existing accent-stripped index
                 already give you, if you return the whole bucket".

    b3_key       b1_key plus the fifteen frozen alternations collapsed into
                 equivalence classes. The claim under test is that reverse
                 lookup needs only "these letters are interchangeable", not a
                 continuous distance in phonological feature space.

    b3_variants  a handful of *query-side only* alternants for the alternations
                 that cannot be expressed as a symmetric collapse without
                 destroying morphological contrasts (movable nu, final sigma,
                 iota adscript, gamma-nu).

FROZEN 2026-08-11 before any measurement on the test split. The fifteen
alternations come from dilemma_papygreek_baseline.md section 5; the list is not
to be edited in response to a test-split number.
"""

import re
import unicodedata

# --------------------------------------------------------------------------
# B1: accent / breathing / case / sigma folding
# --------------------------------------------------------------------------


def _fold_sigma(s: str) -> str:
    """Lunate -> standard, medial/final sigma unified the way the index does."""
    s = s.replace("ϲ", "σ").replace("Ϲ", "Σ")
    s = s.replace("ς", "σ")
    if s.endswith("σ"):
        s = s[:-1] + "ς"
    return s


def b1_key(form: str) -> str:
    """Drop every combining mark, lowercase, fold sigma.

    Iota subscript (U+0345) is a combining mark, so it disappears here; rough
    breathing likewise, which is what makes ῥ and ρ the same key.
    """
    nfd = unicodedata.normalize("NFD", form)
    base = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return _fold_sigma(unicodedata.normalize("NFC", base).lower())


# --------------------------------------------------------------------------
# B3: the fifteen frozen alternations
# --------------------------------------------------------------------------
#
# Each is realised either as a collapse into an equivalence class (symmetric,
# applied to both index and query) or as a query-side alternant (asymmetric).
# The number in brackets is the failing-token count from section 5.

ALTERNATIONS = [
    # ---- vowels: collapsed to archiphonemes -------------------------------
    ("1  ει ~ ι                      [187+127]", "collapse", "I"),
    ("2  η ~ ι ~ ει ~ ῃ              [32]",      "collapse", "I"),
    ("3  οι ~ υ ~ ι                  [30]",      "collapse", "I"),
    ("4  ε ~ αι                      [66]",      "collapse", "E"),
    ("5  ο ~ ω                       [62]",      "collapse", "O"),
    ("10 αυ ~ α, ευ ~ ε, ου ~ ο      [18]",      "collapse", "A/E/O"),
    ("11 intervocalic ι ins/del      [28+42]",   "collapse", "I"),
    # ---- consonants: collapsed --------------------------------------------
    ("6  geminate ~ single           [159]",     "collapse", "degeminate"),
    ("7  nasal assimilation ν/μ/γ    [57]",      "collapse", "N"),
    ("8  κ/γ/χ, π/β/φ, τ/δ/θ         [23]",      "collapse", "K/P/T"),
    ("9  σ ins/del, ξ=κσ, ψ=πσ       [26]",      "collapse", "S"),
    ("15 ζ ~ σδ ~ σ",                            "collapse", "S"),
    # ---- query-side alternants --------------------------------------------
    ("12 iota adscript ᾳ/αι, ῳ/ωι",              "variant",  "adscript"),
    ("13 movable ν, final ς",                    "variant",  "final"),
    ("14 γν ~ γιν (γίγνομαι/γίνομαι)",           "variant",  "gamma_nu"),
]

# Applied in order on a b1_key. Digraphs first, then singles.
_VOWEL_MAP = [
    # i-class: itacism. η ι ει οι υ υι all merge on [i] in Koine.
    ("ει", "I"), ("οι", "I"), ("υι", "I"), ("ῃ", "I"),
    # u-class must be taken before bare υ falls into the i-class.
    ("ου", "U"),
    # glide loss: αυ~α, ευ~ε. Taken before bare α/ε.
    ("αυ", "A"), ("ευ", "E"), ("ηυ", "E"),
    # e-class
    ("αι", "E"),
    # singles
    ("η", "I"), ("ι", "I"), ("υ", "I"),
    ("ε", "E"),
    ("ο", "O"), ("ω", "O"),
    ("α", "A"),
]

_CONS_MAP = {
    "κ": "K", "γ": "K", "χ": "K",
    "π": "P", "β": "P", "φ": "P",
    "τ": "T", "δ": "T", "θ": "T",
    "σ": "S", "ς": "S", "ζ": "S",
    "μ": "M", "ν": "N",
    "λ": "L", "ρ": "R",
}

# Nasals neutralise before any consonant (συν+π -> συμπ, συν+κ -> συγκ).
_NASALS = "μνγ"
_CONSONANTS = "βγδζθκλμνξπρσςτφχψ"


# Everything below is str.replace/translate plus two regexes: the collapse runs
# over 4.8M index keys, so a per-character Python loop is not affordable.
_NASAL_RE = re.compile(f"[{_NASALS}](?=[{_CONSONANTS}])")
_GEM_RE = re.compile(r"(.)\1+")
_DIGRAPHS = [(s, d) for s, d in _VOWEL_MAP if len(s) > 1]
_SINGLES = str.maketrans(
    {s: d for s, d in _VOWEL_MAP if len(s) == 1} | _CONS_MAP)


def collapse(key: str) -> str:
    """The alternation collapse, on an already accent-stripped key."""
    s = key.replace("ξ", "κσ").replace("ψ", "πσ")

    # Nasal assimilation is contextual, so it runs on the letters, before the
    # stop series is folded (γ is both a nasal allophone and a velar stop).
    s = _NASAL_RE.sub("N", s)

    # Digraphs before singles: αυ must be taken as a unit or its υ falls into
    # the i-class along with η/ι/οι.
    for src, dst in _DIGRAPHS:
        s = s.replace(src, dst)
    s = s.translate(_SINGLES)

    # Degemination last, so it also absorbs the doubles created by the folds
    # above (κσσ -> κσ, and λλ -> λ).
    return _GEM_RE.sub(r"\1", s)


def b3_key(form: str) -> str:
    """b1_key with the twelve symmetric alternations collapsed.

    Applied to both sides, so two spellings that differ only by the listed
    alternations produce the same string and meet on an O(1) dictionary hit --
    no distance computation anywhere.
    """
    return collapse(b1_key(form))


def b3_variants(form: str) -> list[str]:
    """Query-side alternants for the three asymmetric alternations.

    These are not applied to the index: dropping a final nu on 6M index keys
    would merge λόγον with λόγο and inflate every bucket. Generating them on
    the query side costs nothing and keeps the index honest.
    """
    base = b1_key(form)
    out = {base}

    # 13. movable nu and final sigma. Adding a nu is only plausible after a
    # vowel, which keeps the alternant set to two or three keys per query.
    for v in list(out):
        if v[-1:] in ("ν", "ς"):
            out.add(v[:-1])
        elif v[-1:] in "αεηιουω":
            out.add(v + "ν")

    # 12. iota adscript written out.
    for v in list(out):
        for long_v in ("ω", "η", "α"):
            if v.endswith(long_v + "ι"):
                out.add(v[:-1])

    # 14. γίγνομαι / γίνομαι and friends.
    for v in list(out):
        if "γν" in v:
            out.add(v.replace("γν", "γιν"))
        if "γιν" in v:
            out.add(v.replace("γιν", "γν"))

    return sorted({b3_key(v) for v in out})
