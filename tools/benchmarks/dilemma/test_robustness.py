"""Regression tests for benchmark input and aggregation contracts."""

import hashlib
import importlib.util
import io
import json
import sqlite3
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent


def _load(monkeypatch, name: str, dependencies: dict[str, object] | None = None):
    """Load a sibling script with only the dependencies relevant to the test."""
    for dependency, value in (dependencies or {}).items():
        monkeypatch.setitem(sys.modules, dependency, value)
    monkeypatch.delitem(sys.modules, name, raising=False)
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _write_safetensors(path, header, data=b""):
    """Write a minimal safetensors-shaped fixture."""
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + data)
    return path


def test_verify_model_accepts_valid_minimal_file(monkeypatch, tmp_path, capsys):
    module = _load(monkeypatch, "verify_model")
    path = _write_safetensors(
        tmp_path / "model.safetensors",
        {"lm_head.weight": {
            "dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        b"\0\0\0\0",
    )

    assert module.main(str(path)) == 0
    assert "OK:" in capsys.readouterr().out


def test_verify_model_reports_missing_file(monkeypatch, tmp_path, capsys):
    module = _load(monkeypatch, "verify_model")

    assert module.main(str(tmp_path / "missing.safetensors")) == 1
    assert "FAIL:" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("header", "data"),
    [
        ([], b""),
        ({"__metadata__": {}}, b""),
        ({"lm_head.weight": []}, b""),
        ({"lm_head.weight": {}}, b""),
        ({"lm_head.weight": {"data_offsets": [0]}}, b""),
        ({"lm_head.weight": {"data_offsets": ["0", 4]}}, b"\0" * 4),
        ({"lm_head.weight": {"data_offsets": [-1, 4]}}, b"\0" * 4),
        ({"lm_head.weight": {"data_offsets": [4, 3]}}, b"\0" * 4),
        ({"lm_head.weight": {"data_offsets": [0, 5]}}, b"\0" * 4),
    ],
)
def test_verify_model_rejects_invalid_header_structures(
        monkeypatch, tmp_path, capsys, header, data):
    module = _load(monkeypatch, "verify_model")
    path = _write_safetensors(tmp_path / "model.safetensors", header, data)

    assert module.main(str(path)) == 1
    output = capsys.readouterr().out
    assert "FAIL:" in output


def test_verify_model_rejects_short_file(monkeypatch, tmp_path, capsys):
    module = _load(monkeypatch, "verify_model")
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"short")

    assert module.main(str(path)) == 1
    assert "FAIL:" in capsys.readouterr().out


def test_verify_model_rejects_header_length_beyond_body(
        monkeypatch, tmp_path, capsys):
    module = _load(monkeypatch, "verify_model")
    path = tmp_path / "model.safetensors"
    path.write_bytes((9).to_bytes(8, "little") + b"{}")

    assert module.main(str(path)) == 1
    assert "FAIL:" in capsys.readouterr().out


def test_verify_model_cli_requires_path():
    completed = subprocess.run(
        [sys.executable, HERE / "verify_model.py"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "usage:" in completed.stderr
    assert "/tmp/greberta" not in completed.stderr


def test_papygreek_scan_requires_source_directory(monkeypatch, tmp_path):
    splits = types.SimpleNamespace(dev_docs=lambda: set())
    module = _load(monkeypatch, "rlb_leakcheck", {"rlb_splits": splits})
    monkeypatch.setattr(module, "PAPYGREEK", tmp_path / "missing")

    with pytest.raises(FileNotFoundError, match="PapyGreek"):
        module.papygreek_docs()


def test_build_dataset_classifies_and_writes_a_token(monkeypatch, tmp_path):
    module = _load(monkeypatch, "build_dataset")
    treebank = tmp_path / "papygreek"
    treebank.mkdir()
    (treebank / "sample.xml").write_text(
        """<treebank>
        <document_meta name="doc-1" date_not_before="1" date_not_after="2" />
        <sentence id="1">
          <word id="1" form_orig="λογος" form_reg="λόγος"
                lemma_orig="λόγος" lemma_reg="λόγος"
                postag_reg="n" lang="grc" />
        </sentence>
        </treebank>""",
        encoding="utf-8",
    )
    output = tmp_path / "dataset.jsonl"
    monkeypatch.setattr(module, "TB", treebank)
    monkeypatch.setattr(module, "OUT", output)

    module.main()

    rows = [json.loads(line) for line in output.read_text(
        encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["stratum"] == "variant_ortho"


def test_tm_ids_are_split_and_deduplicated(monkeypatch):
    splits = types.SimpleNamespace(dev_docs=lambda: set())
    module = _load(monkeypatch, "rlb_leakcheck", {"rlb_splits": splits})

    assert module._normalise_tm_ids("24174 24175 24174") == [
        "24174", "24175"]
    assert module._normalise_tm_ids("") == []


def test_generated_papygreek_ids_use_normalized_record_fields():
    data = json.loads((HERE / "papygreek_ids.json").read_text(encoding="utf-8"))
    assert all(isinstance(doc["tm_id"], list) for doc in data["docs"])
    assert all(ids == list(dict.fromkeys(ids)) for ids in
               (doc["tm_id"] for doc in data["docs"]))
    assert data["all_tm_ids"] == sorted({
        tm_id for doc in data["docs"] for tm_id in doc["tm_id"]})
    assert data["test_tm_ids"] == sorted({
        tm_id for doc in data["docs"] if doc["role"] == "test"
        for tm_id in doc["tm_id"]})


def test_key_cache_has_one_nonempty_key_per_newline(monkeypatch, tmp_path):
    monkeypatch.setenv("DILEMMA_DATA_DIR", str(tmp_path))
    (tmp_path / "lookup.db").touch()
    module = _load(monkeypatch, "rlb_lexicon")
    grc = tmp_path / "grc_keys.txt"
    b3 = tmp_path / "b3_keys.txt"

    module._write_key_cache(grc, ["alpha", "beta"])
    module._write_key_cache(b3, ["a", "b"])
    b3.write_text(b3.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    assert grc.read_bytes().endswith(b"\n")
    assert module._read_key_cache(grc) == ["alpha", "beta"]
    assert module._read_key_cache(b3) == ["a", "b"]
    assert grc.read_bytes().count(b"\n") == 2
    assert len(module._read_key_cache(grc)) == len(module._read_key_cache(b3))


def test_lexicon_escapes_sqlite_uri_paths(monkeypatch, tmp_path):
    data_dir = tmp_path / "data #1?"
    data_dir.mkdir()
    for name in ("lookup.db", "spell_index.db"):
        with sqlite3.connect(data_dir / name) as connection:
            connection.execute("CREATE TABLE marker (value TEXT)")
    monkeypatch.setenv("DILEMMA_DATA_DIR", str(data_dir))
    module = _load(monkeypatch, "rlb_lexicon")

    lexicon = module.Lexicon(data_dir)
    assert lexicon._lk.execute(
        "SELECT name FROM sqlite_master WHERE name = 'marker'").fetchone()
    assert lexicon._sp.execute(
        "SELECT name FROM sqlite_master WHERE name = 'marker'").fetchone()
    lexicon._lk.close()
    lexicon._sp.close()


def test_failed_fetch_removes_partial_file(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "LEMMA_DIR", tmp_path)

    class FailedProcess:
        """Stands in for subprocess.Popen, context manager protocol included.

        The real Popen has been a context manager since Python 3.2 and closes
        its pipes on exit; stream_lemmas relies on that to avoid leaking a pipe
        per text across a 1,027-text fetch, so the double has to model it or the
        test passes on a process shape that does not exist.
        """

        args = ["curl"]

        def __init__(self):
            self.stdout = iter([b'<word lemma="logos"/>\n'])
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            self.closed = True
            return False

        @staticmethod
        def wait():
            return 22

    calls = []
    processes = []

    def popen(args, stdout):
        calls.append(args)
        assert stdout is subprocess.PIPE
        processes.append(FailedProcess())
        return processes[-1]

    monkeypatch.setattr(module.subprocess, "Popen", popen)

    with pytest.raises(subprocess.CalledProcessError):
        module.stream_lemmas("tlg0001", 10)
    assert "--fail" in calls[0]
    assert not (tmp_path / "tlg0001.part").exists()
    assert not (tmp_path / "tlg0001.txt").exists()
    assert processes[0].closed, "the child's pipe was left open on the failure path"


class _CurlProcess:
    """Minimal Popen double for stream_lemmas outcome tests."""

    args = ["curl"]

    def __init__(self, lines, returncode):
        self.stdout = iter(lines)
        self.returncode = returncode

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def wait(self):
        return self.returncode


def test_timed_out_fetch_keeps_only_completed_sentences(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "LEMMA_DIR", tmp_path)
    process = _CurlProcess([
        b'<sentence id="1">\n',
        '<word lemma="λόγος"/>\n'.encode(),
        b'</sentence>\n',
        b'<sentence id="2">\n',
        '<word lemma="ἄνθρωπος"/>\n'.encode(),
    ], module.CURL_TIMEOUT)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_, **__: process)

    assert module.stream_lemmas("tlg0001", 10) == (1, 1)
    assert (tmp_path / "tlg0001.txt").read_text(encoding="utf-8") == "λόγος\n"
    assert not (tmp_path / "tlg0001.part").exists()


def test_timed_out_fetch_without_complete_sentence_fails(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "LEMMA_DIR", tmp_path)
    process = _CurlProcess([
        b'<sentence id="1">\n',
        '<word lemma="λόγος"/>\n'.encode(),
    ], module.CURL_TIMEOUT)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_, **__: process)

    with pytest.raises(subprocess.TimeoutExpired):
        module.stream_lemmas("tlg0001", 10)
    assert not (tmp_path / "tlg0001.part").exists()
    assert not (tmp_path / "tlg0001.txt").exists()


def test_successful_fetch_keeps_final_sentence(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "LEMMA_DIR", tmp_path)
    process = _CurlProcess([
        b'<sentence id="1">\n',
        '<word lemma="λόγος"/>\n'.encode(),
    ], 0)
    monkeypatch.setattr(module.subprocess, "Popen", lambda *_, **__: process)

    assert module.stream_lemmas("tlg0001", 10) == (1, 1)
    assert (tmp_path / "tlg0001.txt").read_text(encoding="utf-8") == "λόγος\n"


def _merge_module(monkeypatch):
    ladder = types.SimpleNamespace(report=lambda result: None)
    return _load(monkeypatch, "rlb_merge", {"rlb_ladder": ladder})


def _write_slice(path: Path, detail: list[dict], provenance=None) -> None:
    result = {
        "stage": "B3u", "seconds": 1.0, "detail": detail,
    }
    if provenance is not None:
        result["provenance"] = provenance
    path.write_text(json.dumps([result]), encoding="utf-8")


def _detail(doc: str = "doc-1", wid: str = "1") -> dict:
    return {
        "doc": doc, "sent": "1", "wid": wid, "form": "x", "gold": "x",
        "split": "test", "n_cand": 1, "rank": 0,
    }


def test_merge_rejects_duplicate_occurrence_ids(monkeypatch, tmp_path):
    module = _merge_module(monkeypatch)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    _write_slice(first, [_detail()])
    _write_slice(second, [_detail()])

    with pytest.raises(ValueError, match="duplicate"):
        module.merge([first, second], tmp_path / "out.json")


def test_merge_rejects_empty_detail(monkeypatch, tmp_path):
    module = _merge_module(monkeypatch)
    source = tmp_path / "empty.json"
    _write_slice(source, [])

    with pytest.raises(ValueError, match="empty"):
        module.merge([source], tmp_path / "out.json")


def test_merge_preserves_disjoint_occurrences(monkeypatch, tmp_path):
    module = _merge_module(monkeypatch)
    source = tmp_path / "slice.json"
    _write_slice(source, [_detail(), _detail("doc-2", "2")])

    result = module.merge([source], tmp_path / "out.json")

    assert result["per_split"]["all"]["n"] == 2


def test_merge_rejects_mixed_provenance(monkeypatch, tmp_path):
    module = _merge_module(monkeypatch)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    _write_slice(first, [_detail("doc-1")], {"dataset": "one"})
    _write_slice(second, [_detail("doc-2")], {"dataset": "two"})

    with pytest.raises(ValueError, match="different benchmark provenance"):
        module.merge([first, second], tmp_path / "out.json")


def _stats_module(monkeypatch):
    rerank = types.SimpleNamespace(load_attestation=lambda: {}, make_scorer=lambda *_: None)
    splits = types.SimpleNamespace(tag=lambda rows: rows)
    evaluation = types.SimpleNamespace(load=lambda path: [], norm_lenient=lambda value: value)
    return _load(monkeypatch, "rlb_stats", {
        "rlb_rerank": rerank, "rlb_splits": splits, "run_eval": evaluation,
    })


def test_rank_profile_returns_complete_empty_result(monkeypatch):
    module = _stats_module(monkeypatch)

    assert module.rank_profile({}, []) == {
        "n": 0,
        "unresolved": 0.0,
        "rank_median": None,
        "rank_p90": None,
        "recall@1": 0.0,
        "recall@5": 0.0,
        "recall@10": 0.0,
        "recall@20": 0.0,
        "recall@50": 0.0,
        "doc_coverage@5": 0.0,
        "doc_coverage@20": 0.0,
    }


def test_positional_join_is_validated_before_pairing(monkeypatch):
    module = _stats_module(monkeypatch)
    rows = [{"input": "x", "lemma_gold": "g", "split": "test"}]

    with pytest.raises(SystemExit, match="rows vs"):
        module._validated_detail_rows("B3u", [], rows)
    with pytest.raises(SystemExit, match="positional join failed"):
        module._validated_detail_rows(
            "B3u", [{"form": "y", "gold": "g", "split": "test"}], rows)

    detail = [{"form": "x", "gold": "g", "split": "test"}]
    assert module._validated_detail_rows("B3u", detail, rows) == [
        (detail[0], rows[0])]


def test_propnoun_report_names_capitalisation_proxy(monkeypatch, capsys):
    evaluation = types.SimpleNamespace(norm_lenient=lambda value: value)
    module = _load(monkeypatch, "rlb_propnoun", {"run_eval": evaluation})
    result = {
        "n_tokens": 1, "n_docs": 1, "k": 5, "dump_cap_truncated": 0,
        "base_rate_capitalised": 1.0,
        "buckets": {
            name: {"n": int(name == "a"), "capitalised": int(name == "a"),
                   "share": float(name == "a")}
            for name in ("hit", "a", "absent")
        },
    }

    module.report(result)
    output = capsys.readouterr().out.lower()
    assert "capitalised gold share" in output
    assert "onomasticon" not in output


def test_altprob_estimation_excludes_the_evaluation_documents(monkeypatch):
    rows = [
        {"doc": "dev", "stratum": "variant_ortho", "form_orig": "dv", "form_reg": "d"},
        {"doc": "dev", "stratum": "clean", "form_orig": "dev-leak", "form_reg": "x"},
        {"doc": "test", "stratum": "variant_ortho", "form_orig": "tv", "form_reg": "t"},
        {"doc": "test", "stratum": "clean", "form_orig": "test-leak", "form_reg": "x"},
        {"doc": "other", "stratum": "clean", "form_orig": "keep", "form_reg": "x"},
    ]
    module = _load(monkeypatch, "rlb_altprob", {
        "categorize": SimpleNamespace(labels=lambda a, b: [f"{a}>{b}"]),
        "clean": SimpleNamespace(clean=lambda value: value),
        "rlb_keys": SimpleNamespace(b1_key=lambda value: value),
        "rlb_stats": SimpleNamespace(bootstrap=lambda *_: {}),
        "run_eval": SimpleNamespace(
            load=lambda _: rows, norm_lenient=lambda value: value),
    })

    _, dev_counts = module.estimate({"dev"}, verbose=False)
    _, test_counts = module.estimate({"test"}, verbose=False)
    _, all_counts = module.estimate({"dev", "test"}, verbose=False)

    assert set(dev_counts) == {"tv>t", "test-leak>x", "keep>x"}
    assert set(test_counts) == {"dv>d", "dev-leak>x", "keep>x"}
    assert set(all_counts) == {"keep>x"}
    with pytest.raises(ValueError, match="must not be empty"):
        module.estimate(set(), verbose=False)


def test_altprob_estimation_rejects_empty_estimation_rows(monkeypatch):
    rows = [
        {"doc": "dev", "form_orig": "dv", "form_reg": "d"},
    ]
    module = _load(monkeypatch, "rlb_altprob", {
        "categorize": SimpleNamespace(labels=lambda a, b: [f"{a}>{b}"]),
        "clean": SimpleNamespace(clean=lambda value: value),
        "rlb_keys": SimpleNamespace(b1_key=lambda value: value),
        "rlb_stats": SimpleNamespace(bootstrap=lambda *_: {}),
        "run_eval": SimpleNamespace(
            load=lambda _: rows, norm_lenient=lambda value: value),
    })

    with pytest.raises(ValueError, match="no rows remain"):
        module.estimate({"dev"}, verbose=False)


def test_altprob_estimation_rejects_zero_alternation_total(monkeypatch):
    rows = [
        {"doc": "other", "form_orig": "same", "form_reg": "same"},
    ]
    module = _load(monkeypatch, "rlb_altprob", {
        "categorize": SimpleNamespace(labels=lambda *_: []),
        "clean": SimpleNamespace(clean=lambda value: value),
        "rlb_keys": SimpleNamespace(b1_key=lambda value: value),
        "rlb_stats": SimpleNamespace(bootstrap=lambda *_: {}),
        "run_eval": SimpleNamespace(
            load=lambda _: rows, norm_lenient=lambda value: value),
    })

    with pytest.raises(ValueError, match="no alternation sites"):
        module.estimate({"dev"}, verbose=False)


def test_altprob_cli_reports_estimation_validation_error(
        monkeypatch, tmp_path, capsys):
    dump = tmp_path / "dump.jsonl"
    dump.write_text(json.dumps({
        "split": "dev", "doc": "dev", "sent": "1", "wid": "1",
        "form": "x", "gold": "x", "cands": [],
    }) + "\n", encoding="utf-8")
    module = _load(monkeypatch, "rlb_altprob", {
        "categorize": SimpleNamespace(labels=lambda *_: []),
        "clean": SimpleNamespace(clean=lambda value: value),
        "rlb_keys": SimpleNamespace(b1_key=lambda value: value),
        "rlb_stats": SimpleNamespace(bootstrap=lambda *_: {}),
        "run_eval": SimpleNamespace(
            load=lambda _: [], norm_lenient=lambda value: value),
    })
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["rlb_altprob.py", "--dump", dump.name])

    with pytest.raises(SystemExit) as exc:
        module.main()

    assert exc.value.code == 2
    assert "no rows remain" in capsys.readouterr().err


def test_facet_headline_uses_dev_recall_winner_for_both_splits(monkeypatch):
    module = _load(monkeypatch, "rlb_facet", {
        "rlb_keys": SimpleNamespace(b1_key=lambda value: value),
        "rlb_rerank": SimpleNamespace(
            AGDT_POS={}, KOINE=(), centuries_for=lambda _: (),
            load_attestation=lambda: {}),
        "rlb_stats": SimpleNamespace(
            bootstrap=lambda *_: {}, rank_profile=lambda *_: {}),
        "run_eval": SimpleNamespace(norm_lenient=lambda value: value),
    })
    dev = {
        "none": {"recall@5": 0.75},
        "aggressive|pos": {
            "recall@5": 0.82,
            "recall5_ci": {"point": 7.0, "lo": 4.0, "hi": 9.0},
        },
        "aggressive|+pos+century+dialect": {
            "recall@5": 0.78,
            "recall5_ci": {"point": 3.0, "lo": -1.0, "hi": 6.0},
        },
    }
    test = {
        "none": {"recall@5": 0.73},
        "aggressive|pos": {
            "recall@5": 0.81,
            "recall5_ci": {"point": 8.0, "lo": 6.0, "hi": 11.0},
        },
        "aggressive|+pos+century+dialect": {
            "recall@5": 0.90,
            "recall5_ci": {"point": 17.0, "lo": 14.0, "hi": 20.0},
        },
    }

    dev_best = module.headline_selection(dev, "dev")
    test_best = module.headline_selection(test, "test")

    assert dev_best["name"] == "aggressive|pos"
    assert test_best["name"] == dev_best["name"]
    assert test_best["selection_metric"] == "dev recall@5"
    assert test_best["recall5_vs_unfiltered"] == test["aggressive|pos"]["recall5_ci"]


def test_bert_public_metadata_records_pinned_revisions(monkeypatch):
    module = _load(monkeypatch, "rlb_bert")

    metadata = module.portable_model_metadata(
        module.MODEL, module.MODEL_REVISION, module.TOKENIZER_REVISION)

    assert metadata == {
        "model": "bowphs/GreBerta",
        "model_revision": "3dce05464f1f429d68acd9b09e117632490c92d4",
        "tokenizer_revision": "3dce05464f1f429d68acd9b09e117632490c92d4",
    }


def test_bert_local_metadata_uses_content_hashes(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_bert")
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"model")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")

    metadata = module.portable_model_metadata(
        str(tmp_path), module.MODEL_REVISION, module.TOKENIZER_REVISION)

    assert metadata["model"] == "local-checkpoint"
    assert metadata["model_revision"].startswith("sha256:")
    assert metadata["tokenizer_revision"].startswith("sha256:")
    assert str(tmp_path) not in json.dumps(metadata)


def test_benchmark_provenance_is_portable_and_content_addressed(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "benchmark_provenance")
    artifact = tmp_path / "dump.jsonl"
    artifact.write_bytes(b"candidate dump\n")
    corpus = tmp_path / "private-corpus-path"
    corpus.mkdir()
    (corpus / "b.txt").write_text("beta\n", encoding="utf-8")
    (corpus / "a.txt").write_text("alpha\n", encoding="utf-8")

    file_result = module.file_identity(artifact)
    directory_result = module.directory_identity(corpus)

    assert file_result == {
        "name": "dump.jsonl",
        "bytes": 15,
        "sha256": hashlib.sha256(b"candidate dump\n").hexdigest(),
    }
    assert directory_result["name"] == corpus.name
    assert directory_result["files"] == 2
    assert len(directory_result["sha256"]) == 64
    assert str(tmp_path) not in json.dumps(directory_result)


def test_benchmark_code_identity_hashes_runtime_scripts_only(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "benchmark_provenance")
    (tmp_path / "runner.py").write_text("RUN = 1\n", encoding="utf-8")
    (tmp_path / "helper.py").write_text("HELPER = 1\n", encoding="utf-8")
    (tmp_path / "test_runner.py").write_text("TEST = 1\n", encoding="utf-8")
    monkeypatch.setattr(module, "BENCHMARK_DIR", tmp_path)

    first = module.benchmark_code_identity()
    (tmp_path / "test_runner.py").write_text("TEST = 2\n", encoding="utf-8")
    after_test_change = module.benchmark_code_identity()
    (tmp_path / "runner.py").write_text("RUN = 2\n", encoding="utf-8")
    after_runtime_change = module.benchmark_code_identity()

    assert first == after_test_change
    assert first["files"] == 2
    assert first["sha256"] != after_runtime_change["sha256"]
    assert str(tmp_path) not in json.dumps(first)


def _distribution(version="1.2.0", commit=None):
    direct_url = None
    if commit is not None:
        direct_url = json.dumps({"vcs_info": {"commit_id": commit}})
    return SimpleNamespace(
        version=version,
        read_text=lambda name: direct_url if name == "direct_url.json" else None,
    )


def test_benchmark_rejects_unpinned_dilemma_version(monkeypatch):
    module = _load(monkeypatch, "benchmark_provenance")
    monkeypatch.setattr(
        module.metadata, "distribution",
        lambda _: _distribution("9.9.9", module.DILEMMA_COMMIT))

    with pytest.raises(RuntimeError, match="version 1.2.0 required.*9.9.9"):
        module.dilemma_package_identity(require_expected=True)


def test_benchmark_rejects_unverifiable_dilemma_commit(monkeypatch):
    module = _load(monkeypatch, "benchmark_provenance")
    monkeypatch.setattr(
        module.metadata, "distribution", lambda _: _distribution())

    with pytest.raises(RuntimeError, match="no PEP 610 VCS commit"):
        module.dilemma_package_identity(require_expected=True)


def test_benchmark_rejects_wrong_dilemma_commit(monkeypatch):
    module = _load(monkeypatch, "benchmark_provenance")
    monkeypatch.setattr(
        module.metadata, "distribution",
        lambda _: _distribution(commit="wrong-commit"))

    with pytest.raises(RuntimeError, match="wrong-commit"):
        module.dilemma_package_identity(require_expected=True)


def test_benchmark_records_verified_dilemma_commit(monkeypatch):
    module = _load(monkeypatch, "benchmark_provenance")
    monkeypatch.setattr(
        module.metadata, "distribution",
        lambda _: _distribution(commit=module.DILEMMA_COMMIT))

    identity = module.dilemma_package_identity(require_expected=True)

    assert identity["source_commit"] == module.DILEMMA_COMMIT
    assert identity["expected_source_commit"] == module.DILEMMA_COMMIT


def test_dilemma_data_identity_records_pinned_source(monkeypatch, tmp_path):
    module = _load(monkeypatch, "benchmark_provenance")
    for name in module.DILEMMA_DATA_FILES:
        (tmp_path / name).write_text(name, encoding="utf-8")

    identity = module.dilemma_data_identity(tmp_path)

    assert identity["source_version"] == module.DILEMMA_VERSION
    assert identity["expected_source_commit"] == module.DILEMMA_COMMIT
    assert "source_commit" not in identity
    assert {item["name"] for item in identity["files"]} \
        == set(module.DILEMMA_DATA_FILES)


def _provenance_dependencies():
    provenance = types.SimpleNamespace(
        benchmark_code_identity=lambda: {"sha256": "code"},
        file_identity=lambda path: {"name": Path(path).name},
        dilemma_data_identity=lambda path: {"name": Path(path).name},
    )
    return provenance


def test_b0_result_provenance_identifies_every_material_input(
        monkeypatch, tmp_path):
    data = tmp_path / "dilemma-data"
    module = _load(monkeypatch, "rlb_b0", {
        "benchmark_provenance": _provenance_dependencies(),
        "rlb_keys": types.SimpleNamespace(b1_key=lambda value: value),
        "rlb_lexicon": types.SimpleNamespace(DATA=data, Lexicon=object),
        "rlb_splits": types.SimpleNamespace(tag=lambda rows: rows),
        "run_eval": types.SimpleNamespace(
            load=lambda path: [], norm_lenient=str, norm_strict=str),
    })
    monkeypatch.setattr(module, "ROOT", tmp_path)

    provenance = module.result_provenance()

    assert provenance["benchmark_code"] == {"sha256": "code"}
    assert provenance["inputs"] == {
        "dataset": {"name": "dataset.jsonl"},
        "splits": {"name": "splits.json"},
        "dilemma_data": {"name": "dilemma-data"},
    }


def test_ddb_result_provenance_identifies_every_material_input(
        monkeypatch, tmp_path):
    data = tmp_path / "dilemma-data"
    module = _load(monkeypatch, "rlb_ddb_run", {
        "benchmark_provenance": _provenance_dependencies(),
        "rlb_ddb_splits": types.SimpleNamespace(dev_docs=lambda path: set()),
        "rlb_ladder": types.SimpleNamespace(
            Ladder=object, serialize_candidates=lambda *_: []),
        "rlb_lexicon": types.SimpleNamespace(DATA=data),
        "run_eval": types.SimpleNamespace(candidate_ranks=lambda *_: (None, None)),
    })

    provenance = module.result_provenance(
        tmp_path / "pairs.jsonl", tmp_path / "splits.json")

    assert provenance["benchmark_code"] == {"sha256": "code"}
    assert provenance["inputs"] == {
        "pairs": {"name": "pairs.jsonl"},
        "splits": {"name": "splits.json"},
        "dilemma_data": {"name": "dilemma-data"},
    }


def test_bert_incomplete_gate_keeps_measurement_exploratory(monkeypatch):
    module = _load(monkeypatch, "rlb_bert")
    gate = {"n": 100, "A": {"verdict": "signal"}}
    diff = {"point": -0.014, "lo": -0.029, "hi": 0.004}

    status, verdict = module.measurement_conclusion(gate, diff)

    assert status == "exploratory"
    assert verdict == "EXPLORATORY -- formal 200-token gate incomplete"


def test_bert_result_artifacts_have_portable_exploratory_metadata():
    revision = "3dce05464f1f429d68acd9b09e117632490c92d4"
    for name in ("results_bert_gate.json", "results_bert_dev.json",
                 "results_bert_C_dev.json"):
        result = json.loads((HERE / name).read_text(encoding="utf-8"))
        assert result["model"] == "bowphs/GreBerta"
        assert result["model_revision"] == revision
        assert result["tokenizer_revision"] == revision
        assert result["preregistration_status"] == "exploratory"
        assert "/tmp/" not in json.dumps(result)

    for name in ("results_bert_dev.json", "results_bert_C_dev.json"):
        result = json.loads((HERE / name).read_text(encoding="utf-8"))
        assert result["verdict"].startswith("EXPLORATORY")


def test_facet_result_artifacts_select_dev_recall_winner():
    for name in ("results_facet_dev.json", "results_facet_test.json"):
        result = json.loads((HERE / name).read_text(encoding="utf-8"))
        best = result["_best"]
        assert best["name"] == "aggressive|pos"
        assert best["selection_metric"] == "dev recall@5"
        assert best["selected_on"] == "dev"
        assert (best["recall5_vs_unfiltered"]
                == result[best["name"]]["recall5_ci"])
        assert result["_preregistered_best"]["name"] \
            == "aggressive|+pos+century+dialect"


def test_lm_paired_rejects_headline_weight_outside_grid():
    completed = subprocess.run(
        [sys.executable, HERE / "rlb_lm_paired.py",
         "--weights", "1.0", "--headline-w", "2.0",
         "--dump", "/definitely/missing.jsonl"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "--headline-w must be included in --weights" in completed.stderr
    assert "FileNotFoundError" not in completed.stderr


def test_lm_paired_result_records_exact_inputs(monkeypatch, tmp_path):
    class FakeLM:
        def __init__(self, directory):
            self.sources = [{
                "dir": Path(directory).name,
                "group": 1,
                "files": 1,
                "tokens": 1,
            }]
            self.total = 1
            self.uni = {"lemma": 1}

    class FakeTrigram:
        @staticmethod
        def train(directories, _filters, _groups):
            return FakeLM(directories[0])

    lm_module = SimpleNamespace(
        Trigram=FakeTrigram,
        contexts=lambda *_, **__: {},
        nfc=lambda value: value,
    )
    stats_module = SimpleNamespace(bootstrap=lambda *_args, **_kwargs: {
        "point": 0.0, "lo": 0.0, "hi": 0.0,
        "n_docs": 1, "n_tokens": 1,
    })
    evaluation = SimpleNamespace(norm_lenient=lambda value: value)
    module = _load(monkeypatch, "rlb_lm_paired", {
        "rlb_lm": lm_module,
        "rlb_stats": stats_module,
        "run_eval": evaluation,
    })
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module, "hits_for", lambda rows, *_: {
            (row["doc"], row["sent"], row["wid"]): 1 for row in rows})

    for name in ("corpus-a", "corpus-b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "sample.txt").write_text("lemma\n", encoding="utf-8")
    dump = tmp_path / "dump.jsonl"
    dump.write_text(json.dumps({
        "doc": "d", "sent": "1", "wid": "1", "split": "dev",
        "gold": "lemma", "cands": [["lemma", 0, 1, 1, "source"]],
    }) + "\n", encoding="utf-8")
    (tmp_path / "context_lemmas.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "rlb_lm_paired.py",
        "--dump", "dump.jsonl",
        "--a-dir", "corpus-a",
        "--b-dir", "corpus-b",
        "--weights", "2.0",
        "--out", "paired.json",
    ])

    module.main()

    result = json.loads((tmp_path / "paired.json").read_text(encoding="utf-8"))
    assert result["a"] == "C_C (DDbDP, in-domain)"
    assert result["corpora"]["a"]["directories"][0]["name"] == "corpus-a"
    assert result["corpora"]["b"]["directories"][0]["name"] == "corpus-b"
    assert result["corpora"]["a"]["lm_tokens"] == 1
    assert result["parameters"] == {
        "context_mode": "dilemma",
        "weights": [2.0],
        "bootstrap_resamples": 1000,
    }
    assert set(result["inputs"]) == {"candidate_dump", "context_lemmas"}
    assert str(tmp_path) not in json.dumps(result)


def test_safe_ratios_return_zero_for_empty_denominators(monkeypatch):
    analyze = _load(monkeypatch, "rlb_analyze", {
        "categorize": SimpleNamespace(label=lambda *_: ""),
        "clean": SimpleNamespace(clean=lambda value: value),
        "rlb_keys": SimpleNamespace(b1_key=lambda value: value),
        "rlb_lexicon": SimpleNamespace(Lexicon=object),
        "run_eval": SimpleNamespace(load=lambda _: [], norm_lenient=lambda value: value),
    })
    b0 = _load(monkeypatch, "rlb_b0", {
        "rlb_keys": SimpleNamespace(b1_key=lambda value: value),
        "rlb_lexicon": SimpleNamespace(DATA=Path("data"), Lexicon=object),
        "rlb_splits": SimpleNamespace(tag=lambda rows: rows),
        "run_eval": SimpleNamespace(
            load=lambda _: [], norm_lenient=lambda value: value,
            norm_strict=lambda value: value),
    })

    assert analyze._ratio(0, 0) == 0.0
    assert analyze._ratio(1, 2) == 0.5
    assert b0._ratio(0, 0) == 0.0
    assert b0._ratio(1, 4) == 0.25


def test_line_count_and_resume_truncation_handle_unterminated_lines(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_build", {
        "rlb_keys": SimpleNamespace(collapse=lambda value: value),
        "rlb_lexicon": SimpleNamespace(Lexicon=object),
    })
    path = tmp_path / "keys.txt"

    assert module._line_count(path) == 0
    path.write_bytes(b"alpha\nbeta")
    assert module._line_count(path) == 2

    module._truncate_incomplete_line(path)
    assert path.read_bytes() == b"alpha\n"
    path.write_bytes(b"alpha\nbeta\n")
    module._truncate_incomplete_line(path)
    assert path.read_bytes() == b"alpha\nbeta\n"


def test_greek_range_excludes_overlapping_coptic_characters(monkeypatch):
    module = _load(monkeypatch, "rlb_ddb")

    assert module.is_greek("\u03e1")
    assert not module.is_greek("\u03e2")
    assert not module.is_greek("\u03ef")
    assert module.is_greek("\u03f0")
    assert module.is_greek("\u1f00")


def test_series_exclusions_require_the_scan_index(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_ddb")
    papygreek = tmp_path / "papygreek.json"
    papygreek.write_text(json.dumps({"all_tm_ids": ["1"]}), encoding="utf-8")
    missing = tmp_path / "missing.jsonl"

    assert module.load_exclusions("tm", missing, papygreek) == {"1"}
    with pytest.raises(FileNotFoundError, match="index"):
        module.load_exclusions("series", missing, papygreek)


def test_empty_ddb_exclusion_index_has_zero_document_shares(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_ddb")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "index.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "papygreek.json").write_text(
        json.dumps({"all_tm_ids": []}), encoding="utf-8")
    args = SimpleNamespace(
        index="index.jsonl", papygreek="papygreek.json", out="out.json")

    module.cmd_exclusions(args)

    result = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert result["policy_tm"]["share_docs"] == 0.0
    assert result["policy_series_volume"]["share_docs"] == 0.0


def test_ddb_ranking_continues_after_first_lenient_match(monkeypatch):
    module = _load(monkeypatch, "rlb_ddb_run", {
        "rlb_ddb_splits": SimpleNamespace(dev_docs=lambda _: set()),
        "rlb_ladder": SimpleNamespace(
            Ladder=object, serialize_candidates=lambda candidates, _: candidates),
        "run_eval": SimpleNamespace(
            candidate_ranks=lambda candidates, gold: (0, 1),
            norm_lenient=lambda value: value.lower(),
            norm_strict=lambda value: value),
    })

    assert module._gold_ranks(["GOLD", "Gold"], "Gold") == (0, 1)


def test_shared_ranking_continues_after_first_lenient_match(monkeypatch):
    module = _load(monkeypatch, "run_eval")

    assert module.candidate_ranks(["GOLD", "Gold"], "Gold") == (0, 1)


def test_batch_result_count_rejects_short_and_long_results(monkeypatch):
    module = _load(monkeypatch, "run_eval")

    for actual in (1, 3):
        with pytest.raises(RuntimeError, match=f"expected 2, got {actual}"):
            module.require_batch_result_count(2, actual, "test lemmatization")

    module.require_batch_result_count(2, 2, "test lemmatization")


def test_evaluate_rejects_short_batch_before_aggregation(monkeypatch):
    module = _load(monkeypatch, "run_eval")
    rows = [
        {"input": "a", "input_reg": "a", "stratum": "clean",
         "lemma_gold": "a"},
        {"input": "b", "input_reg": "b", "stratum": "clean",
         "lemma_gold": "b"},
    ]
    lemmatizer = SimpleNamespace(
        lemmatize_batch=lambda words, guess: ["a"])

    with pytest.raises(RuntimeError, match="expected 2, got 1"):
        module.evaluate(
            rows, lemmatizer, use_reg=False, guess=False, label="test")


def test_candidate_dump_serialization_keeps_every_candidate(monkeypatch):
    module = _load(monkeypatch, "rlb_ladder", {
        "rlb_index": SimpleNamespace(KeySpace=object, PairedIndex=object),
        "rlb_keys": SimpleNamespace(
            b1_key=lambda value: value, b3_key=lambda value: value,
            b3_variants=lambda value: [value]),
        "rlb_lexicon": SimpleNamespace(
            DATA=HERE, Lexicon=object, _read_key_cache=lambda _: [],
            form_freq=lambda: {}, lemma_freq=lambda: {},
            strip_for_freq=lambda value: value),
        "rlb_splits": SimpleNamespace(tag=lambda rows: rows),
        "run_eval": SimpleNamespace(candidate_ranks=lambda *_: (None, None),
                                    load=lambda _: []),
    })
    candidates = [f"lemma-{i}" for i in range(501)]
    scored = {candidate: (1.0, -2, -3, f"key-{i}")
              for i, candidate in enumerate(candidates)}

    dumped = module.serialize_candidates(candidates, scored)

    assert len(dumped) == 501
    assert dumped[-1] == ["lemma-500", 1.0, 2, 3, "key-500"]

    class FakeLadder:
        @staticmethod
        def scored(word, stage):
            return scored

    row = {
        "doc": "d", "sent": "1", "wid": "1", "postag": "n",
        "date_not_before": "", "date_not_after": "", "input": "query",
        "input_reg": "query", "lemma_gold": "gold", "split": "test",
    }
    dump = io.StringIO()
    module.score([row], FakeLadder(), "B3u", dump)
    record = json.loads(dump.getvalue())
    assert record["n_cand"] == len(record["cands"]) == 501


def _dataset(tmp_path, form_orig, form_reg=None):
    """One-row dataset.jsonl, enough for load() to have something to check."""
    path = tmp_path / "dataset.jsonl"
    path.write_text(json.dumps({
        "doc": "p.test.1.xml", "sent": "1", "wid": "1",
        "form_orig": form_orig, "form_reg": form_reg or form_orig,
        "lemma_gold": "λόγος", "lemma_orig": "λόγος",
        "postag": "n-s---mn-", "stratum": "clean",
    }) + "\n", encoding="utf-8")
    return path


def test_load_accepts_forms_that_clean_fully_strips(monkeypatch, tmp_path):
    module = _load(monkeypatch, "run_eval")

    rows = module.load(_dataset(tmp_path, "⎴λόγος⎵"))

    assert [r["input"] for r in rows] == ["λόγος"]


def test_load_refuses_markup_that_clean_let_through(monkeypatch, tmp_path):
    """The 2026-08-12 leak searched on the marked-up string and returned
    nothing, which reads as a method failure rather than a data bug. The guard
    must stop the run instead, naming the codepoint and an example token."""
    module = _load(monkeypatch, "run_eval")
    # U+2E0E EDITORIAL CORONIS: real Leiden markup, not in MARKUP_CHARS.
    path = _dataset(tmp_path, "λόγος⸎")

    with pytest.raises(SystemExit) as excinfo:
        module.load(path)

    message = str(excinfo.value)
    assert "U+2E0E" in message
    assert "p.test.1.xml" in message
    assert "MARKUP_CHARS" in message


def test_load_guard_also_inspects_the_regularised_side(monkeypatch, tmp_path):
    # input_reg feeds rlb_b0's reverse lookup and rlb_postag, so markup there
    # is just as corrupting as markup in the searched form.
    module = _load(monkeypatch, "run_eval")
    path = _dataset(tmp_path, "λόγος", form_reg="λόγος⸎")

    with pytest.raises(SystemExit, match="U\\+2E0E"):
        module.load(path)


def test_trigram_group_counts_nonempty_lines(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "sample.txt").write_text(
        "a\nb\nc\nd\n\ne\nf\n", encoding="utf-8")

    lm = module.Trigram.train(corpus, groups=[3], verbose=False)

    assert lm.tri["a", "b", "c"] == 1
    assert lm.tri["b", "c", "d"] == 0
    assert lm.tri[module.BOS, module.BOS, "d"] == 1
    assert lm.tri[module.BOS, module.BOS, "e"] == 1


def test_predicted_context_preserves_unresolved_token_positions(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    sequence = [
        {"doc": "d", "sent": "1", "wid": str(i), "lemma_gold": lemma}
        for i, lemma in enumerate(("one", "two", "three"), start=1)
    ]
    monkeypatch.setattr(
        module, "sentence_index", lambda: {("d", "1"): sequence})
    (tmp_path / "context_lemmas.json").write_text(
        json.dumps({"d|1|1": "one"}), encoding="utf-8")

    rows = [sequence[1], sequence[2]]
    predicted = module.contexts(
        rows, "dilemma", allow_legacy_context_cache=True)

    assert predicted[("d", "1", "2")] == (module.BOS, "one")
    assert predicted[("d", "1", "3")] == ("one", module.UNK)
    assert module.contexts([sequence[2]], "gold")[("d", "1", "3")] == (
        "one", "two")


def test_predicted_context_rejects_legacy_cache_by_default(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "sentence_index", lambda: {})
    (tmp_path / "context_lemmas.json").write_text(
        json.dumps({"d|1|1": "one"}), encoding="utf-8")

    with pytest.raises(SystemExit, match="legacy context cache"):
        module.contexts([], "dilemma")


def test_predicted_context_rejects_non_mapping_cache(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "sentence_index", lambda: {})
    (tmp_path / "context_lemmas.json").write_text("[]", encoding="utf-8")

    with pytest.raises(SystemExit, match="must be a mapping"):
        module.contexts(
            [], "dilemma", allow_legacy_context_cache=True)


def test_predicted_context_rejects_mismatched_dataset_provenance(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "sentence_index", lambda: {})
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("current\n", encoding="utf-8")
    provenance = module.file_identity(dataset)
    provenance["sha256"] = "0" * 64
    (tmp_path / "context_lemmas.json").write_text(json.dumps({
        "provenance": {"dataset": provenance},
        "lemmas": {},
    }), encoding="utf-8")

    with pytest.raises(SystemExit, match="dataset provenance mismatch"):
        module.contexts([], "dilemma")


def test_predicted_context_accepts_matching_dataset_provenance(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "sentence_index", lambda: {})
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("current\n", encoding="utf-8")
    (tmp_path / "context_lemmas.json").write_text(json.dumps({
        "provenance": {"dataset": module.file_identity(dataset)},
        "lemmas": {},
    }), encoding="utf-8")

    assert module.contexts([], "dilemma") == {}


def test_context_command_rejects_short_batch_before_writing(
        monkeypatch, tmp_path):
    class Dilemma:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def lemmatize_batch(words, guess):
            return []

    monkeypatch.setitem(
        sys.modules, "dilemma", SimpleNamespace(Dilemma=Dilemma))
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module, "dilemma_package_identity", lambda **_: {"version": "1.2.0"})
    monkeypatch.setattr(module, "sentence_index", lambda: {
        ("d", "1"): [{"doc": "d", "sent": "1", "wid": "1",
                        "form_orig": "λόγος"}],
    })

    with pytest.raises(RuntimeError, match="expected 1, got 0"):
        module.cmd_context(SimpleNamespace())
    assert not (tmp_path / "context_lemmas.json").exists()


def test_context_command_writes_versioned_provenance(monkeypatch, tmp_path):
    class Dilemma:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def lemmatize_batch(words, guess):
            return ["λόγος"]

    monkeypatch.setitem(
        sys.modules, "dilemma", SimpleNamespace(Dilemma=Dilemma))
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    package = {"version": "1.2.0", "source_commit": "pinned"}
    monkeypatch.setattr(module, "dilemma_package_identity", lambda **_: package)
    (tmp_path / "dataset.jsonl").write_text("fixture\n", encoding="utf-8")
    monkeypatch.setattr(module, "sentence_index", lambda: {
        ("d", "1"): [{
            "doc": "d", "sent": "1", "wid": "1", "form_orig": "λόγος",
        }],
    })

    module.cmd_context(SimpleNamespace())

    payload = json.loads(
        (tmp_path / "context_lemmas.json").read_text(encoding="utf-8"))
    assert payload["lemmas"] == {"d|1|1": "λόγος"}
    assert payload["provenance"]["dilemma"] == package
    assert payload["provenance"]["dataset"]["name"] == "dataset.jsonl"


def test_ddb_lemmatization_preserves_unresolved_token_positions(
        monkeypatch, tmp_path):
    class Dilemma:
        def __init__(self, **kwargs):
            pass

        @staticmethod
        def lemmatize_batch(words, guess):
            assert words == ["one", "two", "three"]
            return ["lemma-one", None, "lemma-three"]

    monkeypatch.setitem(
        sys.modules, "dilemma", SimpleNamespace(Dilemma=Dilemma))
    module = _load(monkeypatch, "rlb_lm")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(
        module, "dilemma_package_identity", lambda **_: {"version": "1.2.0"})
    source = tmp_path / "ddb_text"
    source.mkdir()
    (source / "1.txt").write_text("one two three\n", encoding="utf-8")

    module.cmd_ddb(SimpleNamespace(
        src="ddb_text", out="ddb_lemmas", batch=1, budget=0))

    output = (tmp_path / "ddb_lemmas" / "1.txt").read_text(encoding="utf-8")
    assert output.splitlines()[0] == "lemma-one <unk> lemma-three"


def test_trigram_rejects_nonpositive_group(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "sample.txt").write_text("a\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="positive"):
        module.Trigram.train(corpus, groups=[0], verbose=False)


def test_window_filter_requires_match_in_selected_directory(monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_lm")
    glaux = tmp_path / "glaux"
    ddb = tmp_path / "ddb"
    glaux.mkdir()
    ddb.mkdir()
    (glaux / "tlg0001.txt").write_text("a\n", encoding="utf-8")
    (ddb / "123.txt").write_text("b\n", encoding="utf-8")

    filters = module.window_filters([glaux, ddb], {"tlg0001"})
    assert filters == {glaux.resolve(): {"tlg0001"}}

    with pytest.raises(SystemExit, match="matches no files"):
        module.window_filters([glaux, ddb], {"missing"})


def _run_downsample(monkeypatch, module, *args):
    monkeypatch.setattr(sys, "argv", ["rlb_downsample.py", *map(str, args)])
    module.main()


def test_downsample_rejects_overlapping_paths_before_deleting(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_downsample")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    original = source / "keep.txt"
    original.write_text("one two\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="disjoint"):
        _run_downsample(
            monkeypatch, module, "--src", "source", "--out", "source",
            "--tokens", "1")

    assert original.read_text(encoding="utf-8") == "one two\n"


def test_downsample_rejects_invalid_budget_before_touching_output(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_downsample")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "source.txt").write_text("one\n", encoding="utf-8")
    existing = output / "existing.txt"
    existing.write_text("keep\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="positive token budget"):
        _run_downsample(
            monkeypatch, module, "--src", "source", "--out", "output")

    assert existing.exists()


def test_downsample_refuses_to_delete_regular_output_files(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_downsample")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "source.txt").write_text("one\n", encoding="utf-8")
    existing = output / "existing.txt"
    existing.write_text("keep\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="regular .txt"):
        _run_downsample(
            monkeypatch, module, "--src", "source", "--out", "output",
            "--tokens", "1")

    assert existing.exists()


def test_downsample_rejects_empty_source_before_touching_output(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_downsample")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    existing = output / "existing.txt"
    existing.write_text("keep\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="contains no .txt"):
        _run_downsample(
            monkeypatch, module, "--src", "source", "--out", "output",
            "--tokens", "1")

    assert existing.exists()


def test_downsample_rejects_match_overlapping_output_before_deleting(
        monkeypatch, tmp_path):
    module = _load(monkeypatch, "rlb_downsample")
    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "source.txt").write_text("one\n", encoding="utf-8")
    existing = output / "existing.txt"
    existing.write_text("keep\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="--match.*disjoint"):
        _run_downsample(
            monkeypatch, module, "--src", "source", "--out", "output",
            "--match", "output")

    assert existing.exists()


def _strata_module(monkeypatch, predictions):
    class Dilemma:
        def __init__(self, **kwargs):
            pass

        def lemmatize_batch(self, words, guess):
            return predictions

    monkeypatch.setitem(sys.modules, "dilemma", SimpleNamespace(Dilemma=Dilemma))
    return _load(monkeypatch, "rlb_ddb_strata")


def test_ddb_strata_rejects_empty_input(monkeypatch, tmp_path):
    module = _strata_module(monkeypatch, [])
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "pairs.jsonl").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["rlb_ddb_strata.py", "--pairs", "pairs.jsonl"])

    with pytest.raises(SystemExit, match="no pairs"):
        module.main()


def test_ddb_strata_validates_lemmatizer_result_count(monkeypatch, tmp_path):
    module = _strata_module(monkeypatch, ["only-one-result"])
    monkeypatch.setattr(module, "ROOT", tmp_path)
    (tmp_path / "pairs.jsonl").write_text(json.dumps({
        "form": "a", "gold": "b"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "rlb_ddb_strata.py", "--pairs", "pairs.jsonl", "--batch", "10"])

    with pytest.raises(RuntimeError, match="2.*1"):
        module.main()
