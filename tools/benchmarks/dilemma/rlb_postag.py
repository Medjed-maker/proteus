"""Run Dilemma's own POS head over the UVP tokens and record what it predicts.

The re-ranking experiment in `rlb_rerank.py` showed POS is the one signal worth
having, but it measured that with PapyGreek's gold tag corrupted at an assumed
error rate. This replaces the assumption with the real tagger.

Two settings are recorded per token:

    pred_orig   the tagger sees the papyrus spelling -- the deployable setting
    pred_reg    the tagger sees the editor's regularised spelling -- an upper
                bound that isolates how much the misspelling costs the tagger

Output is a small JSONL keyed by (doc, sent, wid), so the expensive part runs
once and `rlb_rerank.py` just reads it. Needs a venv with `dilemma-nlp[onnx]`;
everything else in this directory does not.
"""

import argparse
import json
from pathlib import Path

from rlb_splits import tag
from run_eval import load, require_batch_result_count

ROOT = Path(__file__).parent

# Dilemma's POS head emits Wiktionary-style labels. lemma_attestation.json's
# `dominant_pos` uses a different vocabulary, and PapyGreek uses AGDT/Morpheus
# position-1 codes. Everything is mapped into the dominant_pos vocabulary.
DILEMMA_POS = {
    "verb": "verb", "noun": "noun", "adj": "adjective", "adv": "adverb",
    "pron": "pronoun", "num": "numeral", "prep": "preposition",
    "article": "article",
    # The head has no proper-noun class in dominant_pos terms; both of these
    # are nouns there.
    "name": "noun", "character": "noun",
}

AGDT_POS = {
    "n": "noun", "v": "verb", "a": "adjective", "d": "adverb",
    "l": "article", "g": "particle", "c": "conjunction", "r": "preposition",
    "p": "pronoun", "m": "numeral", "i": "interjection", "e": "interjection",
}

# Classes PapyGreek marks that Dilemma's head cannot emit at all.
UNREACHABLE = {"particle", "conjunction", "interjection"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", default="variant_ortho")
    ap.add_argument("--out", default="postags.jsonl")
    args = ap.parse_args()

    from dilemma import Dilemma

    rows = tag([r for r in load(ROOT / "dataset.jsonl")
                if r["stratum"] == args.stratum])
    print(f"{len(rows)} tokens", flush=True)

    d = Dilemma(lang="grc", resolve_articles=True, normalize=True,
                period="hellenistic")
    print("tagging papyrus spellings...", flush=True)
    pred_orig = d.predict_pos_batch([r["input"] for r in rows])
    require_batch_result_count(
        len(rows), len(pred_orig), "papyrus-spelling POS prediction")
    print("tagging regularised spellings...", flush=True)
    pred_reg = d.predict_pos_batch([r["input_reg"] for r in rows])
    require_batch_result_count(
        len(rows), len(pred_reg), "regularised-spelling POS prediction")

    with (ROOT / args.out).open("w", encoding="utf-8") as fh:
        for r, po, pr in zip(rows, pred_orig, pred_reg):
            fh.write(json.dumps({
                "doc": r["doc"], "sent": r["sent"], "wid": r["wid"],
                "split": r["split"],
                "gold_pos": AGDT_POS.get((r.get("postag") or "")[:1]),
                "raw_orig": po, "raw_reg": pr,
                "pred_orig": DILEMMA_POS.get(po),
                "pred_reg": DILEMMA_POS.get(pr),
            }, ensure_ascii=False) + "\n")
    print(f"written to {ROOT / args.out}")


if __name__ == "__main__":
    main()
