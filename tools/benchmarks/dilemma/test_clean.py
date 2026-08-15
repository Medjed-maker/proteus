"""Regression coverage for clean.py's Leiden markup stripping.

Six characters reached run_eval.clean()'s output uncaught until 2026-08-12:
U+23B4/23B5 (supplied-text brackets), U+2E1C/2E1D (low paraphrase brackets),
U+2991/2992 (angle brackets with dot), U+29FC (curved angle bracket), and a
private-use glyph marker (U+E1C0). Because rlb_ladder.py searches on
clean()'s output directly, the leak was not cosmetic: six dev/test tokens
carrying this markup got n_cand=0 for otherwise ordinary words. This file
pins the fix so the six characters cannot silently regress back into
MARKUP_CHARS' blind spot.
"""

import unicodedata

from clean import classify, clean, has_abbrev, residue

# The exact tokens observed in dump_b3u.jsonl with n_cand=0 or a dropped
# recall@5 hit before the fix (dev split unless noted). Each pair here is the
# marked-up form_orig and the word it must reduce to.
LEAK_CASES = [
    ("⎴ἔχοντες⎵", "ἔχοντες"),          # had n_cand=0 pre-fix
    ("⎴ἐλάσσονος⎵", "ἐλάσσονος"),      # had n_cand=0 pre-fix
    ("⎴εὐχερῶς⎵", "εὐχερῶς"),          # had n_cand=0 pre-fix
    ("⎴ὑγιαίνοντας⎵", "ὑγιαίνοντας"),  # had n_cand=0 pre-fix
    ("⎴ἀσππασώμεθα⎵", "ἀσππασώμεθα"),  # had n_cand=0 pre-fix
    ("⎴οὐδʼ⎵", "οὐδʼ"),
    ("⎴ὑμᾶς⎵", "ὑμᾶς"),
    ("⸜γ⸝", "γ"),
    ("χαίρι⸜ν⸝", "χαίριν"),            # test split; brackets mid-word
    ("⸜Σαραπίωνι⸝", "Σαραπίωνι"),      # test split
    ("⦑τὰς⦒", "τὰς"),                  # test split
]

NEW_CHARS = "⎴⎵⸜⸝⦑⦒⧼"


def test_each_new_markup_char_is_registered():
    from clean import MARKUP_CHARS
    for ch in NEW_CHARS:
        assert ch in MARKUP_CHARS, f"{ch!r} (U+{ord(ch):04X}) not in MARKUP_CHARS"


def test_leak_cases_reduce_to_the_plain_word():
    for marked, plain in LEAK_CASES:
        assert clean(marked) == unicodedata.normalize("NFC", plain)


def test_bracket_pairs_are_fully_removed_not_just_one_side():
    # A regression that only stripped the opening half would leave the
    # closing half attached to the following character in real running text.
    for open_ch, close_ch in [("⎴", "⎵"), ("⸜", "⸝"), ("⦑", "⦒")]:
        s = f"{open_ch}λόγος{close_ch}"
        cleaned = clean(s)
        assert open_ch not in cleaned and close_ch not in cleaned
        assert cleaned == "λόγος"


def test_curved_angle_bracket_and_private_use_glyph():
    assert clean("⧼παρθένος") == "παρθένος"
    assert clean("κύριος\ue1c0") == "κύριος"


def test_output_is_nfc_normalised():
    # NFD input with the new markup interleaved must still come out NFC.
    nfd = unicodedata.normalize("NFD", "⎴ἔχοντες⎵")
    result = clean(nfd)
    assert result == unicodedata.normalize("NFC", result)
    assert result == "ἔχοντες"


# -- non-regression: markup this fix did not touch --------------------------


def test_preexisting_markup_still_stripped():
    assert clean("[λόγος]") == "λόγος"
    assert clean("(λόγος)") == "λόγος"
    assert clean("⸢λόγος⸣") == "λόγος"
    assert clean("⟦λόγος⟧") == "λόγος"
    assert clean("λόγ∼ος") == "λόγος"


def test_angle_and_brace_content_survives_bracket_removal():
    # clean() drops only the < > { } delimiters, not the letters they
    # enclose, for both <ε> (supplied) and {ε} (deleted) -- the deletion
    # semantics of {} are a downstream flag, not a content removal here.
    assert clean("λ<ε>γος") == "λεγος"
    assert clean("λ{ε}γος") == "λεγος"


def test_abbrev_detection_unaffected():
    assert has_abbrev("❨στρα(τηγῷ)❩")
    assert not has_abbrev("λόγος")


def test_classify_unaffected_by_the_new_chars():
    # form_orig carrying only the newly-stripped markup, identical to
    # form_reg once cleaned, must still classify as "clean" (not "variant").
    assert classify("⎴ἔχοντες⎵", "ἔχοντες") == "clean"
    assert classify("⎴ελασσων⎵", "ἐλάσσων") == "variant"


# -- 2026-08-14: the four characters residue() found that eyeballing missed ---


# Every token in dataset.jsonl whose cleaned form still carried markup on
# 2026-08-14, with the word it must reduce to. Each is an ASCII or superscript
# variant of a mark clean.py already stripped in another encoding, which is the
# failure mode a denylist cannot see.
SWEEP_CASES = [
    ("❨(δραχμ…)❩", "δραχμ"),        # U+2026, abbrev stratum
    ("ἀνθομολογηστ(…)", "ἀνθομολογηστ"),
    ("Εὐτυχ(…)", "Εὐτυχ"),
    ("α̣ν̣(…)", "αν"),
    ("οὐ-", "οὐ"),                   # U+002D line-split hyphen
    ("-δὲ", "δὲ"),
    ("~ἆλλα~", "ἆλλα"),              # U+007E, ASCII form of ∼∽
    ("⁽Μεσορὴ⁾", "Μεσορὴ"),          # U+207D/207E superscript parens
    ("⁽κ|num:20|⁾", "κ"),
    ("⁽⊢λ⊣|num:30|⁾", "λ"),
]

SWEEP_CHARS = "…⁽⁾-~"


def test_each_swept_char_is_registered():
    from clean import MARKUP_CHARS
    for ch in SWEEP_CHARS:
        assert ch in MARKUP_CHARS, f"{ch!r} (U+{ord(ch):04X}) not in MARKUP_CHARS"


def test_sweep_cases_reduce_to_the_plain_word():
    for marked, plain in SWEEP_CASES:
        assert clean(marked) == unicodedata.normalize("NFC", plain)


def test_latin_homoglyph_is_repaired_not_deleted():
    # κυρίoυ carries U+006F LATIN SMALL LETTER O -- a source typo, not markup.
    # Deleting the letter would leave κυρίυ, which is not a word; the letter
    # must become Greek omicron so the form can match κύριος.
    repaired = clean("κυ∤ρίoυ")
    assert repaired == "κυρίου"
    assert "o" not in repaired          # no LATIN SMALL LETTER O left
    assert "ο" in repaired              # GREEK SMALL LETTER OMICRON instead
    assert len(repaired) == len("κυρίου")    # repaired, not deleted


# -- residue(): the complement check that finds the next one ----------------


def test_residue_is_empty_for_ordinary_greek():
    for word in ("λόγος", "ἔχοντες", "ἀσππασώμεθα", "Σαραπίωνι", "κυρίου"):
        assert residue(word) == set()


def test_residue_allows_all_three_elision_apostrophes():
    # PapyGreek writes elision with U+02BC, U+2019 and U+0027 interchangeably.
    # All three are content, not markup, and must not trip the guard.
    for apostrophe in ("ʼ", "’", "'"):
        assert residue(f"οὐδ{apostrophe}") == set()


def test_residue_allows_combining_diacritics():
    assert residue(unicodedata.normalize("NFD", "ἔχοντες")) == set()


def test_residue_flags_markup_that_clean_does_not_yet_strip():
    # The guard's whole purpose: an unlisted mark is reported rather than
    # searched on. U+2E0E (editorial coronis) is not in MARKUP_CHARS today.
    assert residue("λόγος⸎") == {"⸎"}


def test_residue_flags_latin_letters_that_confusables_does_not_cover():
    # "q" has no Greek homoglyph, so it is neither repaired nor allowed.
    assert residue("λόγqς") == {"q"}


def test_cleaned_dataset_vocabulary_has_no_residue():
    # Guards the pairing of clean() and residue(): everything clean() emits for
    # the observed corpus vocabulary must be inside the allowed repertoire, or
    # run_eval.load() will refuse to run.
    for marked, _ in LEAK_CASES + SWEEP_CASES:
        assert residue(clean(marked)) == set()
