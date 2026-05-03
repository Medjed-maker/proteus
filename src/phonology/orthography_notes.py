"""Language-independent payload types for orthographic notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

OrthographicNoteKind = Literal[
    "orthographic_correspondence",
    "beginner_aid",
    "pre_403_2_attic",
]
OrthographicNoteConfidence = Literal["low", "medium", "high"]
ResponseLanguage = Literal["en", "ja"]


class OrthographicNoteDataError(RuntimeError):
    """Raised when packaged orthographic-note data fails to load or validate."""


@dataclass(frozen=True, slots=True)
class OrthographicNotePayload:
    """Internal orthographic-note payload independent of API models."""

    kind: OrthographicNoteKind
    label: str
    messages: tuple[str, ...]
    confidence: OrthographicNoteConfidence
    normalized_form: str | None = None
    romanization: str | None = None
    period_label: str | None = None
    references: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.messages, str):
            raise TypeError("messages must be a sequence of strings, not str")
        if not isinstance(self.messages, tuple):
            object.__setattr__(self, "messages", tuple(self.messages))
        if not all(isinstance(item, str) for item in self.messages):
            raise TypeError("messages must contain only strings")
        if isinstance(self.references, str):
            raise TypeError("references must be a sequence of strings, not str")
        if not isinstance(self.references, tuple):
            object.__setattr__(self, "references", tuple(self.references))
        if not all(isinstance(item, str) for item in self.references):
            raise TypeError("references must contain only strings")


class OrthographicNoteBuilder(Protocol):
    """Protocol for language-specific orthographic-note builders."""

    def __call__(
        self,
        *,
        query_form: str,
        candidate_headword: str,
        candidate_ipa: str,
        query_ipa: str,
        response_language: ResponseLanguage,
        orthography_hint: str | None = None,
    ) -> list[OrthographicNotePayload]:
        """Build orthographic notes for a search candidate."""
        ...


__all__ = [
    "OrthographicNoteBuilder",
    "OrthographicNoteConfidence",
    "OrthographicNoteDataError",
    "OrthographicNoteKind",
    "OrthographicNotePayload",
    "ResponseLanguage",
]
