"""Strip Leiden/papyri.info editorial markup from PapyGreek surface forms.

The raw ``form_orig`` mixes three unrelated things: the scribe's actual
spelling, the editor's uncertainty/lacuna marks, and abbreviation brackets.
Only the first is orthographic variation, so the other two are separated out
before anything is scored.
"""

import re
import unicodedata

# |num:1|, |m:2|, |g:...| style inline annotations.
INLINE_ANNOT = re.compile(r"\|[^|]*\|")

# ❨στρα(τηγῷ)❩ -- editorial expansion of a scribal abbreviation.
ABBREV = re.compile(r"[❨❩]")

MARKUP_CHARS = (
    "̣"      # combining dot below: uncertain letter
    "∼∽"  # ∼ ∽ regularisation brackets
    "∤"      # ∤ word split across lines
    "⸢⸣⸤⸥"  # ⸢⸣⸤⸥ half brackets
    "⊢⊣"  # ⊢ ⊣ numeral / symbol marks
    "⋰⋱"  # ⋰ ⋱ vestiges
    "〚〛"  # 〚 〛 deleted by scribe
    "⟦⟧"  # ⟦ ⟧
    "⸌⸍"  # ⸌ ⸍ inserted above the line
    "⸮?!"      # editorial query marks
    "()"          # abbreviation expansion parens
    "[]"          # lacuna
    # Found missing during the H4 explore-zone review (2026-08-12): these six
    # reached run_eval.clean()'s output uncaught, so rlb_ladder searched on the
    # marked-up string itself. Six tokens this way got n_cand=0 for otherwise
    # ordinary words (e.g. ⎴ἔχοντες⎵, the commonest form of ἔχω).
    "⎴⎵"  # ⎴ ⎵ U+23B4/23B5 supplied-text top/bottom brackets
    "⸜⸝"  # ⸜ ⸝ U+2E1C/2E1D low paraphrase brackets
    "⦑⦒"  # ⦑ ⦒ U+2991/2992 angle brackets with dot
    "⧼"      # ⧼ U+29FC left-pointing curved angle bracket
    # Found on 2026-08-14 by running residue() over all 33,615 tokens of
    # dataset.jsonl instead of eyeballing samples. Every one of these is an
    # ASCII or superscript variant of a mark already listed above, which is
    # precisely the blind spot a denylist has -- see residue() below.
    "…"      # U+2026 lost text / abbreviation continues (δραχμ…), 5 tokens
    "⁽⁾"  # U+207D/U+207E superscript parens; same role as "()" above, 3 tokens
    "-"      # U+002D line-split hyphen; same role as "∤" above (οὐ- -> οὐ), 3
    "~"      # U+007E ASCII form of the "∼∽" (U+223C/223D) regularisation marks
    ""     # private-use glyph marker seen in a handful of PapyGreek exports
)
MARKUP_RE = re.compile("[" + re.escape(MARKUP_CHARS) + "]")

# Latin letters that render identically to a Greek letter. These are source
# typos, not editorial marks, so they are repaired rather than stripped:
# κυρίoυ (U+006F LATIN SMALL LETTER O) is the genitive of κύριος misspelt at
# source, and deleting the letter would corrupt the word. Only "o" occurs in
# the current data; the rest are listed so the next one gets fixed rather than
# found. Safe here because build_dataset.py admits only lang="grc" Greek words,
# so a Latin letter inside a token is always an error.
CONFUSABLES = str.maketrans({
    "o": "ο", "c": "ϲ",
    "A": "Α", "B": "Β", "E": "Ε", "Z": "Ζ", "H": "Η", "I": "Ι", "K": "Κ",
    "M": "Μ", "N": "Ν", "O": "Ο", "P": "Ρ", "T": "Τ", "X": "Χ", "Y": "Υ",
})

# What a cleaned form is allowed to contain, measured over all 33,615 tokens of
# dataset.jsonl rather than assumed: Greek and Coptic, Greek Extended, combining
# diacritics, and the three codepoints PapyGreek uses for the elision
# apostrophe. Anything else is markup that MARKUP_CHARS has not learnt yet.
ALLOWED_RE = re.compile(
    # Written as escapes on purpose: half of this class is invisible or
    # combines with the preceding bracket when pasted as literal characters,
    # which is exactly how the marks this module exists to catch get lost.
    r"["
    r"\u0300-\u036f"   # combining diacritics (NFD accents, breathings)
    r"\u0370-\u03ff"   # Greek and Coptic
    r"\u1f00-\u1fff"   # Greek Extended
    r"\u02bc\u2019'"  # elision apostrophe: PapyGreek uses all three
    r"]"
)


# <ε> editor supplies an omitted letter; {ε} editor deletes a superfluous one.
# Both are real scribal orthography, so the *content* is kept and only the
# brackets are dropped -- but which side of the pair keeps the letters differs,
# so they are recorded as a flag rather than silently merged.
ANGLE_RE = re.compile(r"[<>]")
BRACE_RE = re.compile(r"[{}]")


def clean(form: str) -> str:
    s = INLINE_ANNOT.sub("", form)
    s = ABBREV.sub("", s)
    s = MARKUP_RE.sub("", s)
    s = ANGLE_RE.sub("", s)
    s = BRACE_RE.sub("", s)
    s = s.translate(CONFUSABLES)
    s = s.strip()
    return unicodedata.normalize("NFC", s)


def residue(s: str) -> set[str]:
    """Characters of a cleaned form that fall outside the expected repertoire.

    MARKUP_CHARS is a denylist, so every unlisted mark passes through silently
    and is then searched on as if it were part of the word. That is not a
    cosmetic failure: on 2026-08-12 six such characters produced n_cand=0 for
    perfectly ordinary words, and the 2026-08-14 sweep that added the last four
    found them only because this function looks at the complement of the
    repertoire rather than at the list. run_eval.load() turns a non-empty result
    into a hard stop.
    """
    return {ch for ch in s if not ALLOWED_RE.match(ch)}


def has_abbrev(form: str) -> bool:
    return "❨" in form or "❩" in form


def classify(form_orig: str, form_reg: str) -> str:
    """clean | abbrev | variant"""
    if has_abbrev(form_orig):
        return "abbrev"
    if clean(form_orig) == clean(form_reg):
        return "clean"
    return "variant"
