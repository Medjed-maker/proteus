"""Build the evaluation dataset from the PapyGreek treebanks.

Splits every annotated token into two strata:

  CLEAN   form_orig == form_reg  -> the papyrus spelling is already standard
  VARIANT form_orig != form_reg  -> the editor had to regularise the spelling

VARIANT is the Proteus UVP zone: a non-standard attested spelling that has to
be resolved back to a dictionary headword. Gold lemma is always ``lemma_reg``
(the lemma of the regularised reading), because that is the headword a
researcher would need to look up.
"""

import json
import re
import unicodedata
from pathlib import Path
from xml.etree import ElementTree as ET

from clean import classify

ROOT = Path(__file__).parent
TB = ROOT / "papygreek" / "ezhenrik-papygreek-treebanks-b87127c"
OUT = ROOT / "dataset.jsonl"

# Tokens the AGDT guidelines annotate as punctuation / non-words.
PUNCT_POSTAG = "u"
GREEK_RE = re.compile(r"[Ͱ-Ͽἀ-῿]")
INLINE_ANNOT = re.compile(r"\|[^|]*\|")


def is_greek_word(s: str) -> bool:
    return bool(s) and bool(GREEK_RE.search(s))


def main() -> None:
    rows = []
    files = sorted(TB.rglob("*.xml"))
    for path in files:
        tree = ET.parse(path)
        root = tree.getroot()
        meta = root.find("document_meta")
        doc = meta.get("name") if meta is not None else path.name
        date_before = meta.get("date_not_before") if meta is not None else ""
        date_after = meta.get("date_not_after") if meta is not None else ""
        for sent in root.iter("sentence"):
            for w in sent.iter("word"):
                form_orig = (w.get("form_orig") or "").strip()
                form_reg = (w.get("form_reg") or "").strip()
                # Lemmas carry inline annotations like ια|num:11|.
                lemma_reg = INLINE_ANNOT.sub("", w.get("lemma_reg") or "").strip()
                lemma_orig = INLINE_ANNOT.sub("", w.get("lemma_orig") or "").strip()
                postag_reg = (w.get("postag_reg") or "").strip()
                lang = w.get("lang") or ""

                if lang and lang != "grc":
                    continue
                if not form_orig or not form_reg or not lemma_reg:
                    continue
                if postag_reg[:1] == PUNCT_POSTAG:
                    continue
                if not is_greek_word(form_orig) or not is_greek_word(lemma_reg):
                    continue
                # Editorial placeholders for lost text.
                if "[" in form_orig or "]" in form_orig or "_" in lemma_reg:
                    continue

                stratum = classify(form_orig, form_reg)
                if stratum == "variant":
                    stratum = (
                        "variant_ortho"
                        if unicodedata.normalize("NFC", lemma_orig)
                        == unicodedata.normalize("NFC", lemma_reg)
                        else "variant_lex"
                    )
                rows.append({
                    "doc": doc,
                    "date_not_before": date_before,
                    "date_not_after": date_after,
                    "sent": sent.get("id"),
                    "wid": w.get("id"),
                    "form_orig": unicodedata.normalize("NFC", form_orig),
                    "form_reg": unicodedata.normalize("NFC", form_reg),
                    "lemma_gold": unicodedata.normalize("NFC", lemma_reg),
                    "lemma_orig": unicodedata.normalize("NFC", lemma_orig),
                    "postag": postag_reg,
                    "stratum": stratum,
                })

    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_clean = sum(1 for r in rows if r["stratum"] == "clean")
    n_var = len(rows) - n_clean
    print(f"files      : {len(files)}")
    print(f"tokens     : {len(rows)}")
    print(f"  clean    : {n_clean}")
    print(f"  variant  : {n_var}  ({n_var / max(len(rows), 1):.1%})")
    print(f"written to : {OUT}")


if __name__ == "__main__":
    main()
