"""Tests for the smaller search helpers and per-stage components.

Covers ``TestFilterStage``, ``TestQueryModeHelpers``, ``TestBuildLexiconMap``,
``TestBuildKmerIndex``, and ``TestSeedStage`` — the small focused classes that
previously lived alongside ``TestSearch`` in ``tests/test_search.py``.
"""

from __future__ import annotations

import inspect
from types import ModuleType, SimpleNamespace
from typing import Any, Callable, get_args, get_type_hints
import unicodedata

import pytest

from phonology import search as search_module
import phonology.search.compat as search_compat
from phonology.languages.ancient_greek.ipa import tokenize_ipa
from phonology.search import (
    SearchResult,
    build_kmer_index,
    build_lexicon_map,
    classify_query_mode,
    extend_stage,
    filter_stage,
    normalize_query_for_search,
    prepare_query_ipa,
    seed_stage,
)
from phonology.search._types import PhoneInventory
from tests._helpers.fakes import install_test_language_profile


class TestFilterStage:
    """Verify final filtering sorts by confidence and rejects invalid limits."""

    def test_sorts_by_confidence_desc_then_lemma(
        self, sample_search_results: list[SearchResult]
    ) -> None:
        filtered = filter_stage(sample_search_results, max_results=2)

        assert [(result.lemma, result.confidence) for result in filtered] == [
            ("α", 0.9),
            ("β", 0.9),
        ]

    @pytest.mark.parametrize("max_results", [0, -1])
    def test_rejects_non_positive_max_results(self, max_results: int) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            filter_stage([], max_results=max_results)


class TestPublicCompatibilityBoundary:
    """Verify public wrappers resolve defaults before calling internal cores."""

    def test_compat_module_exports_only_public_wrappers(self) -> None:
        """Ensure compat exposes only the intended public wrapper functions."""
        assert search_compat.__all__ == [
            "build_kmer_index",
            "build_lexicon_map",
            "extend_stage",
            "prepare_query_ipa",
            "search",
            "search_execution",
            "seed_stage",
        ]

    @pytest.mark.parametrize(
        "target",
        [
            search_module._build_lexicon_map_core,
            search_module._build_lexicon_map_for_inventory,
            search_module._build_kmer_index_for_inventory,
            search_module._seed_stage_core,
            search_module._seed_stage_for_inventory,
            search_module._prepare_query_ipa_core,
            search_module._execute_search,
            search_module._LazySearchDependencies.__init__,
        ],
    )
    def test_core_phone_inventory_is_required(
        self, target: Callable[..., Any]
    ) -> None:
        params = inspect.signature(target).parameters
        assert "phone_inventory" in params, "phone_inventory parameter must exist"
        parameter = params["phone_inventory"]

        assert parameter.default is inspect.Parameter.empty
        assert get_type_hints(target)["phone_inventory"] == PhoneInventory

    def test_public_default_resolvers_return_phone_inventory_type(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Compat is the only layer that accepts optional public inventory."""
        install_test_language_profile(monkeypatch)
        normalize_hints = get_type_hints(search_compat._normalize_phone_inventory)
        resolve_hints = get_type_hints(search_compat._resolve_public_defaults)

        assert normalize_hints["return"] == PhoneInventory
        return_args = get_args(resolve_hints["return"])
        assert return_args, "Return type should have type arguments"
        assert return_args[0] == PhoneInventory

        resolved_inventory, resolved_vowels, resolved_builders = (
            search_compat._resolve_public_defaults(
                language="test",
                phone_inventory=None,
            )
        )
        assert resolved_inventory == ()
        assert resolved_vowels == ()
        assert type(resolved_inventory) is tuple
        assert type(resolved_vowels) is tuple
        assert resolved_builders == ()

    @pytest.mark.parametrize(
        "name",
        [
            "build_kmer_index",
            "build_lexicon_map",
            "prepare_query_ipa",
            "seed_stage",
            "extend_stage",
            "search_execution",
            "search",
        ],
    )
    def test_package_reexports_public_compat_functions(self, name: str) -> None:
        """Ensure public compatibility functions are re-exported by the package."""
        assert getattr(search_module, name) is getattr(search_compat, name)

    def test_core_module_caches_imports_and_can_force_reload(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify compat core resolution caches imports but supports explicit reload."""
        imported_names: list[str] = []
        first_module = ModuleType("first")
        second_module = ModuleType("second")
        modules = iter([first_module, second_module])

        def fake_import_module(name: str) -> ModuleType:
            imported_names.append(name)
            return next(modules)

        monkeypatch.setattr(search_compat, "_CACHED_CORE_MODULE", None)
        monkeypatch.setattr(search_compat, "import_module", fake_import_module)

        assert search_compat._core_module() is first_module
        assert search_compat._core_module() is first_module
        assert search_compat._core_module(force_reload=True) is second_module
        assert imported_names == ["phonology.search", "phonology.search"]

    def test_non_default_language_resolves_missing_inventory_to_empty_tuple(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-default languages pass an empty inventory tuple into the core."""
        captured: dict[str, object] = {}

        def fake_build_lexicon_map_core(
            lexicon: object,
            *,
            phone_inventory: PhoneInventory,
        ) -> dict[str, object]:
            captured["lexicon"] = lexicon
            captured["phone_inventory"] = phone_inventory
            return {}

        monkeypatch.setattr(
            search_module,
            "_build_lexicon_map_core",
            fake_build_lexicon_map_core,
        )
        install_test_language_profile(monkeypatch)

        assert build_lexicon_map([], language="test") == {}
        assert captured == {"lexicon": [], "phone_inventory": ()}

    def test_prepare_query_ipa_rejects_unknown_non_default_language(self) -> None:
        """Unknown profile ids must not fall back to the Ancient Greek converter."""
        with pytest.raises(ValueError, match="Unsupported language profile"):
            prepare_query_ipa("pa", language="missing_profile")

    def test_prepare_query_ipa_reuses_profile_for_defaults_and_converter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Registered non-default profiles are resolved once at the public boundary."""
        calls: list[str] = []
        captured: dict[str, object] = {}

        def converter(text: str, *, dialect: str) -> str:
            return f"{dialect}:{text}"

        profile = SimpleNamespace(
            phone_inventory=("ts", "p", "a"),
            vowel_phones=(),
            dialect_skeleton_builders=(),
            converter=converter,
        )

        def fake_get_language_profile(language_id: str) -> object:
            calls.append(language_id)
            return profile

        def fake_prepare_query_ipa_core(
            query: str,
            *,
            dialect: str = "attic",
            converter: object = None,
            phone_inventory: PhoneInventory,
            query_ipa: str | None = None,
        ) -> str:
            captured["query"] = query
            captured["dialect"] = dialect
            captured["converter"] = converter
            captured["phone_inventory"] = phone_inventory
            captured["query_ipa"] = query_ipa
            return "prepared"

        monkeypatch.setattr(
            search_module,
            "get_language_profile",
            fake_get_language_profile,
        )
        monkeypatch.setattr(
            search_module,
            "_prepare_query_ipa_core",
            fake_prepare_query_ipa_core,
        )

        assert prepare_query_ipa("pa", dialect="toy", language="toy_language") == (
            "prepared"
        )
        assert calls == ["toy_language"]
        assert captured == {
            "query": "pa",
            "dialect": "toy",
            "converter": converter,
            "phone_inventory": search_compat._normalize_phone_inventory(
                ("ts", "p", "a")
            ),
            "query_ipa": None,
        }

    def test_prepare_query_ipa_uses_default_profile_dialect_when_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitted dialect resolves from the selected default profile."""
        captured: dict[str, object] = {}

        def converter(text: str, *, dialect: str) -> str:
            return f"{dialect}:{text}"

        profile = SimpleNamespace(
            language_id="toy_default",
            default_dialect="toy",
            phone_inventory=("p", "a"),
            vowel_phones=(),
            dialect_skeleton_builders=(),
            converter=converter,
        )

        def fake_prepare_query_ipa_core(
            query: str,
            *,
            dialect: str | None = None,
            converter: object = None,
            phone_inventory: PhoneInventory,
            query_ipa: str | None = None,
        ) -> str:
            captured["query"] = query
            captured["dialect"] = dialect
            captured["converter"] = converter
            captured["phone_inventory"] = phone_inventory
            captured["query_ipa"] = query_ipa
            return "prepared"

        monkeypatch.setattr(search_module, "get_default_language_profile", lambda: profile)
        monkeypatch.setattr(
            search_module,
            "_prepare_query_ipa_core",
            fake_prepare_query_ipa_core,
        )

        assert prepare_query_ipa("pa") == "prepared"
        assert captured == {
            "query": "pa",
            "dialect": "toy",
            "converter": converter,
            "phone_inventory": search_compat._normalize_phone_inventory(("p", "a")),
            "query_ipa": None,
        }

    def test_default_language_resolves_profile_inventory_and_builders_when_omitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitted language resolves default profile search-token settings."""
        captured: dict[str, object] = {}

        def builder(tokens: list[str]) -> list[str]:
            return tokens

        profile = SimpleNamespace(
            language_id="toy_default",
            default_dialect="toy",
            phone_inventory=("ts", "p", "a"),
            vowel_phones=(),
            dialect_skeleton_builders=(builder,),
        )

        def fake_build_kmer_index_for_inventory(
            lexicon: object,
            *,
            k: int,
            phone_inventory: PhoneInventory,
            vowel_phones: object = None,
            dialect_skeleton_builders: object = None,
        ) -> dict[str, object]:
            captured["lexicon"] = lexicon
            captured["k"] = k
            captured["phone_inventory"] = phone_inventory
            captured["dialect_skeleton_builders"] = dialect_skeleton_builders
            return {}

        monkeypatch.setattr(search_module, "get_default_language_profile", lambda: profile)
        monkeypatch.setattr(
            search_module,
            "_build_kmer_index_for_inventory",
            fake_build_kmer_index_for_inventory,
        )

        assert build_kmer_index([], language=None) == {}
        assert captured == {
            "lexicon": [],
            "k": 2,
            "phone_inventory": search_compat._normalize_phone_inventory(("ts", "p", "a")),
            "dialect_skeleton_builders": (builder,),
        }

    def test_prepare_query_ipa_uses_registered_profile_default_dialect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit registered profiles supply converter, inventory, and dialect."""
        captured: dict[str, object] = {}

        def converter(text: str, *, dialect: str) -> str:
            return f"{dialect}:{text}"

        profile = SimpleNamespace(
            language_id="toy_language",
            default_dialect="toy",
            phone_inventory=("p", "a"),
            vowel_phones=(),
            dialect_skeleton_builders=(),
            converter=converter,
        )

        def fake_prepare_query_ipa_core(
            query: str,
            *,
            dialect: str | None = None,
            converter: object = None,
            phone_inventory: PhoneInventory,
            query_ipa: str | None = None,
        ) -> str:
            captured["query"] = query
            captured["dialect"] = dialect
            captured["converter"] = converter
            captured["phone_inventory"] = phone_inventory
            captured["query_ipa"] = query_ipa
            return "prepared"

        monkeypatch.setattr(search_module, "get_language_profile", lambda _id: profile)
        monkeypatch.setattr(
            search_module,
            "get_default_language_profile",
            lambda: SimpleNamespace(language_id="other_default"),
        )
        monkeypatch.setattr(
            search_module,
            "_prepare_query_ipa_core",
            fake_prepare_query_ipa_core,
        )

        assert prepare_query_ipa("pa", language="toy_language") == "prepared"
        assert captured == {
            "query": "pa",
            "dialect": "toy",
            "converter": converter,
            "phone_inventory": search_compat._normalize_phone_inventory(("p", "a")),
            "query_ipa": None,
        }

    def test_default_language_uses_profile_converter(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Omitted language resolves the default profile converter."""
        captured: dict[str, str] = {}

        def converter(query: str, *, dialect: str) -> str:
            captured["query"] = query
            captured["dialect"] = dialect
            return "p a"

        profile = SimpleNamespace(
            language_id="toy_default",
            default_dialect="toy",
            phone_inventory=("p", "a"),
            vowel_phones=(),
            dialect_skeleton_builders=(),
            converter=converter,
        )
        monkeypatch.setattr(search_module, "get_default_language_profile", lambda: profile)

        prepared = prepare_query_ipa("pa", phone_inventory=("p", "a"))

        assert prepared.query_ipa == "p a"
        assert captured == {"query": "pa", "dialect": "toy"}

    def test_default_language_resolves_missing_inventory_before_core(
        self, monkeypatch: pytest.MonkeyPatch, known_phones: tuple[str, ...]
    ) -> None:
        """Default seed_stage backfills known phones, unlike non-default languages."""
        captured: dict[str, object] = {}

        def fake_seed_stage_core(
            query_ipa: str,
            index: dict[str, list[str]],
            *,
            k: int,
            phone_inventory: PhoneInventory,
            vowel_phones: object = None,
        ) -> list[str]:
            captured["query_ipa"] = query_ipa
            captured["index"] = index
            captured["k"] = k
            captured["phone_inventory"] = phone_inventory
            return ["L1"]

        monkeypatch.setattr(search_module, "_seed_stage_core", fake_seed_stage_core)

        assert seed_stage("apʰlas", {"pʰ l": ["L1"]}) == ["L1"]
        assert captured["query_ipa"] == "apʰlas"
        assert captured["index"] == {"pʰ l": ["L1"]}
        assert captured["k"] == 2
        assert captured["phone_inventory"] == search_compat._normalize_phone_inventory(
            known_phones
        )

    def test_extend_stage_forwards_profile_annotation_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Public extend_stage backfills profile-specific annotation settings."""
        captured: dict[str, object] = {}

        def phone_matcher(actual: str, expected: str) -> bool:
            return actual == expected

        profile = SimpleNamespace(
            language_id="toy_language",
            default_dialect="toy",
            phone_inventory=("ts", "a", "p"),
            vowel_phones=("a",),
            phone_matcher=phone_matcher,
            dialect_skeleton_builders=(),
            always_match_contexts=("toy context",),
        )

        def fake_extend_stage_core(
            query_ipa: str,
            candidates: object,
            lexicon_map: object,
            matrix: object,
            language: str | None = None,
            *,
            phone_inventory: PhoneInventory,
            vowel_phones: object = None,
            phone_matcher: object = None,
            always_match_contexts: object = None,
        ) -> list[SearchResult]:
            captured["query_ipa"] = query_ipa
            captured["candidates"] = candidates
            captured["lexicon_map"] = lexicon_map
            captured["matrix"] = matrix
            captured["language"] = language
            captured["phone_inventory"] = phone_inventory
            captured["vowel_phones"] = vowel_phones
            captured["phone_matcher"] = phone_matcher
            captured["always_match_contexts"] = always_match_contexts
            return []

        monkeypatch.setattr(
            search_module,
            "get_language_profile",
            lambda _id: profile,
        )
        monkeypatch.setattr(
            search_module,
            "_extend_stage_core",
            fake_extend_stage_core,
        )

        assert extend_stage(
            "tsa",
            ["toy-entry"],
            {"toy-entry": {"headword": "tsa", "ipa": "tsa"}},
            {"ts": {"ts": 0.0}, "a": {"a": 0.0}},
            language="toy_language",
        ) == []
        assert captured == {
            "query_ipa": "tsa",
            "candidates": ["toy-entry"],
            "lexicon_map": {"toy-entry": {"headword": "tsa", "ipa": "tsa"}},
            "matrix": {"ts": {"ts": 0.0}, "a": {"a": 0.0}},
            "language": "toy_language",
            "phone_inventory": search_compat._normalize_phone_inventory(
                ("ts", "a", "p")
            ),
            "vowel_phones": ("a",),
            "phone_matcher": phone_matcher,
            "always_match_contexts": ("toy context",),
        }

    def test_extend_stage_core_forwards_annotation_settings(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Internal extend_stage core passes annotation settings through."""
        captured: dict[str, object] = {}
        scored_results = [SearchResult(lemma="tsa", confidence=1.0)]
        annotated_results = [SearchResult(lemma="annotated", confidence=1.0)]

        def phone_matcher(actual: str, expected: str) -> bool:
            return actual == expected

        def fake_score_stage_for_inventory(**kwargs: object) -> list[SearchResult]:
            captured["score_kwargs"] = kwargs
            return scored_results

        def fake_annotate_search_results_for_inventory(
            **kwargs: object,
        ) -> list[SearchResult]:
            captured["annotate_kwargs"] = kwargs
            return annotated_results

        monkeypatch.setattr(
            search_module,
            "_score_stage_for_inventory",
            fake_score_stage_for_inventory,
        )
        monkeypatch.setattr(
            search_module,
            "_annotate_search_results_for_inventory",
            fake_annotate_search_results_for_inventory,
        )

        result = search_module._extend_stage_core(
            "tsa",
            ["toy-entry"],
            {"toy-entry": {"headword": "tsa", "ipa": "tsa"}},
            {"ts": {"ts": 0.0}, "a": {"a": 0.0}},
            language="toy_language",
            phone_inventory=PhoneInventory(("ts", "a")),
            vowel_phones=("a",),
            phone_matcher=phone_matcher,
            always_match_contexts=("toy context",),
        )

        assert result == annotated_results
        assert captured["score_kwargs"] == {
            "query_ipa": "tsa",
            "candidates": ["toy-entry"],
            "lexicon_map": {"toy-entry": {"headword": "tsa", "ipa": "tsa"}},
            "matrix": {"ts": {"ts": 0.0}, "a": {"a": 0.0}},
            "phone_inventory": PhoneInventory(("ts", "a")),
        }
        assert captured["annotate_kwargs"] == {
            "query_ipa": "tsa",
            "results": scored_results,
            "lexicon_map": {"toy-entry": {"headword": "tsa", "ipa": "tsa"}},
            "matrix": {"ts": {"ts": 0.0}, "a": {"a": 0.0}},
            "language": "toy_language",
            "phone_inventory": PhoneInventory(("ts", "a")),
            "vowel_phones": ("a",),
            "phone_matcher": phone_matcher,
            "always_match_contexts": ("toy context",),
        }


class TestQueryModeHelpers:
    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("λόγος", "Full-form"),
            ("νυν", "Short-query"),
            ("ζηταω-", "Partial-form"),
            ("ζητω~", "Partial-form"),
            ("*λόγ", "Partial-form"),
            ("λο*γος", "Partial-form"),
            (unicodedata.normalize("NFD", "νῦν"), "Short-query"),
            (unicodedata.normalize("NFD", "λόγος"), "Full-form"),
            # Unicode dash markers (paste-from-word-processor, locale
            # keyboards, etc.) must classify as Partial-form just like ASCII
            # hyphen/asterisk. Mirrors _PARTIAL_QUERY_MARKERS in search.py.
            ("ζηταω\u2010", "Partial-form"),  # U+2010 HYPHEN
            ("ζηταω\u2011", "Partial-form"),  # U+2011 NON-BREAKING HYPHEN
            ("ζηταω\u2012", "Partial-form"),  # U+2012 FIGURE DASH
            ("ζηταω\u2013", "Partial-form"),  # U+2013 EN DASH
            ("ζηταω\u2014", "Partial-form"),  # U+2014 EM DASH
            ("ζηταω\u2015", "Partial-form"),  # U+2015 HORIZONTAL BAR
            ("ζηταω\u2212", "Partial-form"),  # U+2212 MINUS SIGN
            ("ζηταω\uff0d", "Partial-form"),  # U+FF0D FULLWIDTH HYPHEN-MINUS
        ],
    )
    def test_classify_query_mode(self, query: str, expected: str) -> None:
        assert classify_query_mode(query) == expected

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            (" ζηταω- ", "ζηταω"),
            (" ζητω~ ", "ζητω"),
            ("", ""),
            ("   ", ""),
            (" *λόγ ", "λόγ"),
            (" λο*γος ", "λογος"),
            ("λόγος", "λόγος"),
            # Unicode dash markers should be stripped the same way as ASCII
            # hyphen, leaving only the fragment to hand to IPA conversion.
            ("ζηταω\u2013", "ζηταω"),  # U+2013 EN DASH
            ("ζηταω\u2014", "ζηταω"),  # U+2014 EM DASH
            ("ζηταω\uff0d", "ζηταω"),  # U+FF0D FULLWIDTH HYPHEN-MINUS
        ],
    )
    def test_normalize_query_for_search(self, query: str, expected: str) -> None:
        assert normalize_query_for_search(query) == expected

    @pytest.mark.parametrize(
        "query",
        [
            "",
            "   ",
            "*",
            "~",
            "-",
            "-*",
            "~*",
            "*-",
            "ζη~τω-",
            "a-*",
            "a*-",
            "a*b*c",
        ],
    )
    def test_classify_query_mode_rejects_invalid_partial_syntax(
        self, query: str
    ) -> None:
        with pytest.raises(ValueError, match=r"wildcard marker|non-empty string"):
            classify_query_mode(query)

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            ("*", "*"),
            ("~", "~"),
            ("-", "-"),
            ("-*", "-*"),
            ("~*", "~*"),
            ("*-", "*-"),
            ("ζη~τω-", "ζη~τω-"),
            ("a-*", "a-*"),
            ("a*-", "a*-"),
            ("a*b*c", "a*b*c"),
        ],
    )
    def test_normalize_query_tolerates_invalid_partial_syntax(
        self, query: str, expected: str
    ) -> None:
        assert normalize_query_for_search(query) == expected


class TestBuildLexiconMap:
    """Verify lexicon map construction and duplicate detection."""

    def test_returns_empty_dict_for_empty_lexicon(self) -> None:
        assert build_lexicon_map([]) == {}

    def test_builds_map_with_token_counts(self) -> None:
        lexicon = [
            {"id": "L1", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
        ]
        result = build_lexicon_map(lexicon)
        assert "L1" in result
        assert result["L1"].entry == lexicon[0]
        assert result["L1"].token_count == 5

    def test_public_default_caches_ancient_greek_multicharacter_phones(
        self,
    ) -> None:
        """Ancient Greek defaults should tokenize multi-character phones.

        build_lexicon_map should cache apʰlas as ("a", "pʰ", "l", "a", "s")
        with token_count 5 for the sample lexicon.
        """
        lexicon = [
            {"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"},
        ]

        result = build_lexicon_map(lexicon)

        assert result["L1"].ipa_tokens == ("a", "pʰ", "l", "a", "s")
        assert result["L1"].token_count == 5

    def test_non_default_language_keeps_literal_fallback_tokens(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-default language tokenization should keep literal fallback tokens.

        For the apʰlas input lexicon with language="test", build_lexicon_map
        should cache ("a", "p", "ʰ", "l", "a", "s") with token_count 6.
        """
        install_test_language_profile(monkeypatch)
        lexicon = [
            {"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "test"},
        ]

        result = build_lexicon_map(lexicon, language="test")

        assert result["L1"].ipa_tokens == ("a", "p", "ʰ", "l", "a", "s")
        assert result["L1"].token_count == 6

    def test_trims_entry_ids_for_lookup_keys(self) -> None:
        lexicon = [
            {"id": " L1 ", "headword": "λόγος", "ipa": "lóɡos", "dialect": "attic"},
        ]
        result = build_lexicon_map(lexicon)

        assert list(result) == ["L1"]

    def test_raises_on_duplicate_entry_id(self) -> None:
        lexicon = [
            {"id": "L1", "headword": "αλφα", "ipa": "alfa", "dialect": "attic"},
            {"id": "L1", "headword": "βητα", "ipa": "beta", "dialect": "attic"},
        ]
        with pytest.raises(ValueError, match="Duplicate entry ID"):
            build_lexicon_map(lexicon)

    def test_raises_on_duplicate_entry_id_after_trimming(self) -> None:
        lexicon = [
            {"id": " L1 ", "headword": "αλφα", "ipa": "alfa", "dialect": "attic"},
            {"id": "L1", "headword": "βητα", "ipa": "beta", "dialect": "attic"},
        ]
        with pytest.raises(ValueError, match="Duplicate entry ID"):
            build_lexicon_map(lexicon)

    @pytest.mark.parametrize("ipa_text", ["", "   "])
    def test_raises_on_empty_or_whitespace_only_ipa(self, ipa_text: str) -> None:
        lexicon = [
            {"id": "L1", "headword": "test", "ipa": ipa_text, "dialect": "attic"},
        ]

        with pytest.raises(ValueError, match="non-empty 'ipa'"):
            build_lexicon_map(lexicon)

    @pytest.mark.parametrize("ipa_text", ["!?", "ã"])
    def test_builds_map_for_non_empty_special_character_ipa(
        self, ipa_text: str
    ) -> None:
        lexicon = [
            {"id": "L1", "headword": "test", "ipa": ipa_text, "dialect": "attic"},
        ]

        result = build_lexicon_map(lexicon)

        assert "L1" in result
        assert result["L1"].entry == lexicon[0]
        assert result["L1"].token_count == len(tokenize_ipa(ipa_text))


class TestBuildKmerIndex:
    """Verify consonant-skeleton k-mer index construction and parameter validation."""

    def test_builds_consonant_skeleton_kmer_index(
        self, sample_lexicon: list[dict[str, str]]
    ) -> None:
        index = build_kmer_index(sample_lexicon, k=2)

        assert index["p t"] == ["L1", "L2"]
        assert index["t n"] == ["L1", "L3"]
        assert "p n" not in index

    def test_public_default_kmer_index_adds_ancient_greek_compatibility_kmers(
        self,
    ) -> None:
        """Verify build_kmer_index with k=2 on IPA "apʰlas" generates compatibility kmers.

        The "pʰ l", "f l", and "l s" kmers are generated via the default Ancient Greek
        phone inventory and dialect skeleton builders, all mapping to id "L1".
        """
        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
        )

        assert index["pʰ l"] == ["L1"]
        assert index["f l"] == ["L1"]
        assert index["l s"] == ["L1"]

    def test_explicit_empty_builders_do_not_add_dialect_skeletons(self) -> None:
        from phonology.languages.ancient_greek.ipa import get_known_phones

        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
            phone_inventory=get_known_phones(),
            dialect_skeleton_builders=(),
        )

        assert index["pʰ l"] == ["L1"]
        assert "f l" not in index
        assert index["l s"] == ["L1"]

    def test_default_language_fills_in_ancient_greek_defaults(self) -> None:
        """Verify build_kmer_index fills in default profile search settings."""
        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
        )

        # "f l" kmer only exists when dialect_skeleton_builders adds compatibility kmers
        assert index["f l"] == ["L1"]
        # "pʰ l" kmer exists via tokenization with phone_inventory
        assert index["pʰ l"] == ["L1"]

    def test_default_language_adds_attic_tt_compatibility_kmers(self) -> None:
        """Verify Attic dialect σσ→ττ transformation generates compatibility k-mers.

        For θάλασσα (tʰálassa), the index should contain both the original
        k-mers ("l s") and Attic-variant k-mers ("l t", "t t") derived from
        the σσ→ττ shift that produces tʰálatta.
        """
        index = build_kmer_index(
            [
                {
                    "id": "LSJ-047735",
                    "headword": "θάλασσα",
                    "ipa": "tʰálassa",
                    "dialect": "attic",
                }
            ],
            k=2,
        )

        assert index["l s"] == ["LSJ-047735"]
        assert index["l t"] == ["LSJ-047735"]
        assert index["t t"] == ["LSJ-047735"]

    def test_normalized_default_language_fills_in_ancient_greek_defaults(self) -> None:
        """Default-language compatibility accepts the API/profile-normalized ID form."""
        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
            language=" Ancient_Greek ",
        )

        assert index["f l"] == ["L1"]
        assert index["pʰ l"] == ["L1"]

    def test_non_default_language_passes_through_supplied_parameters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify build_kmer_index passes through supplied phone_inventory and dialect_skeleton_builders for non-default languages unchanged."""
        custom_inventory = ["a", "p", "l", "s"]
        install_test_language_profile(monkeypatch, language_id="other_lang")

        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "aplas", "dialect": "generic"}],
            k=2,
            phone_inventory=custom_inventory,
            dialect_skeleton_builders=(),
            language="other_lang",
        )

        # With empty dialect_skeleton_builders, no compatibility kmers are added
        assert index["p l"] == ["L1"]
        # "f l" would only exist if Ancient Greek defaults were applied (they shouldn't be)
        assert "f l" not in index

    def test_explicit_koine_builder_adds_compatible_kmers(self) -> None:
        from phonology.languages.ancient_greek.ipa import (
            apply_koine_consonant_shifts,
            get_known_phones,
        )

        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "apʰlas", "dialect": "attic"}],
            k=2,
            phone_inventory=get_known_phones(),
            dialect_skeleton_builders=(apply_koine_consonant_shifts,),
        )

        assert index["pʰ l"] == ["L1"]
        assert index["f l"] == ["L1"]
        assert index["l s"] == ["L1"]

    def test_dialect_skeleton_builders_receive_independent_token_lists(self) -> None:
        """Verify each dialect skeleton builder receives an independent token list."""
        observed_tokens: list[list[str]] = []

        def mutating_builder(tokens: list[str]) -> list[str]:
            tokens.append("t")
            return tokens

        def observing_builder(tokens: list[str]) -> list[str]:
            observed_tokens.append(list(tokens))
            return tokens

        index = build_kmer_index(
            [{"id": "L1", "headword": "target", "ipa": "pa", "dialect": "attic"}],
            k=2,
            dialect_skeleton_builders=(mutating_builder, observing_builder),
        )

        assert index["p t"] == ["L1"]
        assert observed_tokens == [tokenize_ipa("pa")]

    @pytest.mark.parametrize("k", [0, -1])
    def test_rejects_non_positive_k(self, k: int) -> None:
        with pytest.raises(ValueError, match="build_kmer_index.*k"):
            build_kmer_index([], k=k)

    @pytest.mark.parametrize(
        ("entry", "message"),
        [
            ({"ipa": "pten"}, "non-empty 'id' or 'headword'"),
            (
                {"id": "   ", "headword": "   ", "ipa": "pten"},
                "non-empty 'id' or 'headword'",
            ),
            ({"id": "L1", "headword": "πτην"}, "non-empty 'ipa'"),
            ({"id": "L1", "headword": "πτην", "ipa": "   "}, "non-empty 'ipa'"),
        ],
    )
    def test_skips_invalid_lexicon_entries_with_warning(
        self, entry: dict[str, str], message: str
    ) -> None:
        # Invalid entries are skipped with a warning, not a hard error
        with pytest.warns(UserWarning, match=message):
            result = build_kmer_index([entry], k=2)
        assert result == {}


class TestSeedStage:
    """Verify stage-1 seed ranking by shared consonant-skeleton k-mers."""

    def test_ranks_candidates_by_shared_seed_count(
        self, sample_lexicon: list[dict[str, str]]
    ) -> None:
        index = build_kmer_index(sample_lexicon, k=2)

        candidates = seed_stage("pten", index, k=2)

        assert candidates == ["L1", "L2", "L3"]

    def test_returns_empty_list_when_query_has_no_seedable_consonant_skeleton(
        self,
    ) -> None:
        assert seed_stage("aː", {"p t": ["L1"]}, k=2) == []

    def test_k1_finds_candidates_for_single_consonant_query(self) -> None:
        """k=1 unigram index should find entries sharing a single consonant."""
        lexicon = [
            {"id": "L1", "headword": "πτω", "ipa": "pto", "dialect": "attic"},
            {"id": "L2", "headword": "κτω", "ipa": "kto", "dialect": "attic"},
            {"id": "L3", "headword": "ποι", "ipa": "poi", "dialect": "attic"},
        ]
        unigram_idx = build_kmer_index(lexicon, k=1)

        # Query IPA "poieɔː" has consonant skeleton ['p'], so k=1 produces ["p"].
        candidates = seed_stage("poieɔː", unigram_idx, k=1)

        # L1 (pto → skeleton p,t) and L3 (poi → skeleton p) contain "p"
        assert "L1" in candidates
        assert "L2" not in candidates
        assert "L3" in candidates

    def test_k1_returns_empty_for_pure_vowel_query(self) -> None:
        """k=1 should still return empty for a query with zero consonants."""
        unigram_idx: dict[str, list[str]] = {"p": ["L1"]}
        assert seed_stage("aː", unigram_idx, k=1) == []

    def test_default_language_seeds_ancient_greek_multicharacter_phones(
        self,
    ) -> None:
        """Verify seed_stage default fills in Ancient Greek phone inventory.

        With ``language="ancient_greek"`` (default) and no explicit
        ``phone_inventory``, ``seed_stage`` must tokenize ``pʰt`` as a single
        ``pʰ`` phone followed by ``t``, matching the consonant skeleton
        produced by ``build_kmer_index``. Before the public-defaults backfill
        was applied, the query was split into ``p``/``ʰ``/``t`` and seeding
        returned an empty list even though the index contained the entry.
        """
        lexicon = [
            {"id": "L1", "headword": "target", "ipa": "pʰt", "dialect": "attic"},
        ]
        index = build_kmer_index(lexicon, k=2)

        candidates = seed_stage("pʰt", index, k=2)

        assert candidates == ["L1"]

    def test_non_default_language_does_not_inject_ancient_greek_inventory(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-default language must keep the literal-character fallback."""
        install_test_language_profile(monkeypatch)
        lexicon = [
            {"id": "L1", "headword": "target", "ipa": "pʰt", "dialect": "test"},
        ]
        # Use a non-default language for both index and query so neither side
        # gets the Ancient Greek backfill.
        index = build_kmer_index(lexicon, k=2, language="test")

        candidates = seed_stage("pʰt", index, k=2, language="test")

        # Without the inventory, "pʰt" tokenizes to ["p", "ʰ", "t"] whose
        # consonant skeleton is ["p", "t"], producing the kmer "p t". The
        # index also produced "p t" for the same reason, so seeding succeeds
        # but via single-character matching, not multi-character phones.
        assert candidates == ["L1"]
