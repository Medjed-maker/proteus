"""K-mer index builders over consonant skeletons for search seed discovery."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any
import warnings

from ..ipa_converter import apply_koine_consonant_shifts
from ._constants import _DEFAULT_KMER_SIZE
from ._lookup import _entry_id, _entry_ipa
from ._query import _extract_consonant_skeleton
from ._tokenization import tokenize_for_inventory
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


def _build_entry_kmers(
    ipa_text: str,
    k: int,
    *,
    phone_inventory: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None = None,
) -> list[str]:
    """Return stable, de-duplicated seed k-mers for one lexicon IPA form.

    Each entry contributes its original consonant skeleton plus any additional
    dialect-shifted skeletons produced by ``dialect_skeleton_builders``.

    When ``dialect_skeleton_builders`` is ``None`` (the default), the legacy
    Attic+Koine behavior is preserved for backward compatibility. Pass an
    explicit iterable (including an empty tuple) to opt into the new explicit
    dialect-shift model — the Ancient Greek profile passes
    ``(apply_koine_consonant_shifts,)``; custom profiles pass ``()``.

    Args:
        ipa_text: IPA transcription string for a lexicon entry.
        k: Size of each consonant-skeleton k-mer.
        phone_inventory: Optional phone inventory used to tokenize ``ipa_text``
            before skeleton and k-mer generation. If ``None``, default
            inventory behavior is used.
        dialect_skeleton_builders: Callables that each transform an IPA token
            list into a dialect variant for additional skeleton coverage.
            ``None`` triggers legacy Koine-always behavior.

    Returns:
        De-duplicated list of space-joined k-mer strings.
    """
    original_tokens = tokenize_for_inventory(ipa_text, phone_inventory)
    original_skeleton = _extract_consonant_skeleton(original_tokens)

    if dialect_skeleton_builders is None:
        # Legacy backward-compatible path: always add the Koine skeleton.
        extra_skeletons = [
            _extract_consonant_skeleton(apply_koine_consonant_shifts(original_tokens))
        ]
    else:
        extra_skeletons = [
            _extract_consonant_skeleton(builder(original_tokens))
            for builder in dialect_skeleton_builders
        ]

    return list(
        dict.fromkeys(
            [
                *_iter_kmers(original_skeleton, k),
                *(
                    kmer
                    for skeleton in extra_skeletons
                    for kmer in _iter_kmers(skeleton, k)
                ),
            ]
        )
    )


def build_kmer_index(
    lexicon: Sequence[dict[str, Any]],
    k: int = _DEFAULT_KMER_SIZE,
    *,
    phone_inventory: Iterable[str] | None = None,
    dialect_skeleton_builders: Iterable[Callable[[list[str]], list[str]]] | None = None,
) -> KmerIndex:
    """Build a consonant-skeleton k-mer index for lexicon lookup.

    Args:
        lexicon: Sequence of lexicon entry dicts to index.
        k: Size of each consonant-skeleton k-mer. Defaults to
            ``_DEFAULT_KMER_SIZE``.
        phone_inventory: Optional Iterable[str] of phones used to
            constrain/validate phone extraction for the consonant-skeleton
            k-mers. Passing None (the default) causes the function to infer
            the inventory from the lexicon. Invalid phones are ignored/skipped
            during tokenization.
        dialect_skeleton_builders: Optional callables that each transform a
            token list into a dialect variant for additional skeleton coverage.
            ``None`` (default) preserves legacy Koine-always behavior.
            Pass an explicit tuple (including ``()``) to use the new explicit
            dialect-shift model; the Ancient Greek profile passes
            ``(apply_koine_consonant_shifts,)``.

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
            for kmer in _build_entry_kmers(
                ipa,
                k,
                phone_inventory=phone_inventory,
                dialect_skeleton_builders=dialect_skeleton_builders,
            ):
                index.setdefault(kmer, []).append(entry_id)
        except (ValueError, KeyError) as exc:
            warnings.warn(
                f"Skipping invalid entry {entry.get('id', '<missing id>')}: {exc}",
                category=UserWarning,
                stacklevel=2,
            )
            continue
    return index
