"""Corpus source metadata adapters."""

from ._adapters import (
    CompositeCorpusAdapter,
    CorpusAdapter,
    CorpusSourceDataError,
    EMPTY_CORPUS_ADAPTER,
    EmptyCorpusAdapter,
    StaticCorpusAdapter,
    load_static_corpus_adapter,
    safe_lookup,
)
from ._models import AccessPolicy, SourceReference

__all__ = [
    "AccessPolicy",
    "CompositeCorpusAdapter",
    "CorpusAdapter",
    "CorpusSourceDataError",
    "EMPTY_CORPUS_ADAPTER",
    "EmptyCorpusAdapter",
    "SourceReference",
    "StaticCorpusAdapter",
    "load_static_corpus_adapter",
    "safe_lookup",
]
