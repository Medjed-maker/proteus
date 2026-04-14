"""K-mer index builders over consonant skeletons for search seed discovery."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any
import warnings

from ..ipa_converter import apply_koine_consonant_shifts, tokenize_ipa
from ._constants import _DEFAULT_KMER_SIZE
from ._lookup import _entry_id, _entry_ipa
from ._query import _extract_consonant_skeleton
from ._types import KmerIndex


def _iter_kmers(tokens: list[str], k: int) -> Iterator[str]:
    """Yield space-joined k-mers for a token sequence.

    Args:
        tokens: List of IPA tokens.
        k: Size of each k-mer. Assumes k > 0 (validated by build_kmer_index).

    Yields:
        Space-joined k-mer strings.
    """
    if len(tokens) < k:
        return
    for index in range(len(tokens) - k + 1):
        yield " ".join(tokens[index : index + k])


def _build_entry_kmers(ipa_text: str, k: int) -> list[str]:
    """Return stable, de-duplicated seed k-mers for one lexicon IPA form.

    Each entry contributes both its stored Attic-oriented skeleton and the
    Koine-compatible skeleton produced by the shared consonant-shift logic.

    Args:
        ipa_text: IPA transcription string for a lexicon entry.
        k: Size of each consonant-skeleton k-mer.

    Returns:
        De-duplicated list of space-joined k-mer strings.
    """
    original_tokens = tokenize_ipa(ipa_text)
    original_skeleton = _extract_consonant_skeleton(original_tokens)
    koine_skeleton = _extract_consonant_skeleton(apply_koine_consonant_shifts(original_tokens))
    return list(
        dict.fromkeys(
            [
                *_iter_kmers(original_skeleton, k),
                *_iter_kmers(koine_skeleton, k),
            ]
        )
    )


def build_kmer_index(
    lexicon: Sequence[dict[str, Any]],
    k: int = _DEFAULT_KMER_SIZE,
) -> KmerIndex:
    """Build a consonant-skeleton k-mer index for lexicon lookup.

    Args:
        lexicon: Sequence of lexicon entry dicts to index.
        k: Size of each consonant-skeleton k-mer. Defaults to
            ``_DEFAULT_KMER_SIZE``.

    Returns:
        KmerIndex mapping each space-joined k-mer to a list of matching
        lexicon entry ids.

    Invalid lexicon entries are skipped with a warning that includes the entry
    id when available and the validation exception text.

    Raises:
        ValueError: If ``k <= 0``.
    """
    if k <= 0:
        raise ValueError(f"build_kmer_index requires k > 0 for k-mer size, got {k}")

    index: KmerIndex = {}
    for entry in lexicon:
        try:
            entry_id = _entry_id(entry)
            ipa = _entry_ipa(entry)
            for kmer in _build_entry_kmers(ipa, k):
                index.setdefault(kmer, []).append(entry_id)
        except (ValueError, KeyError) as exc:
            warnings.warn(
                f"Skipping invalid entry {entry.get('id', '<missing id>')}: {exc}",
                category=UserWarning,
                stacklevel=2,
            )
            continue
    return index
