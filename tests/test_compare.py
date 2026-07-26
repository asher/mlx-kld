"""``mlx-kld compare``: results-root scanning, table shape, version gating,
run-spec grouping, and robustness against stray files."""

import json
import subprocess
import sys

from mlx_kld.compare import _publisher
from mlx_kld.report import FLOOR_CAUTION_MULT, build_locked_json

# ---------- publisher column (opt-in) ----------

def test_publisher_from_gguf_parent_dir():
    # The file name alone cannot tell two publishers apart; the ORG__REPO
    # directory holding it can.
    assert _publisher(
        "/llm/gguf/unsloth__Qwen3.6-27B-MTP-GGUF/Qwen3.6-27B-Q5_K_M.gguf"
    ) == "unsloth"
    assert _publisher(
        "/llm/gguf/bartowski__Qwen_Qwen3.6-27B-GGUF/Qwen_Qwen3.6-27B-Q5_K_M.gguf"
    ) == "bartowski"


def test_publisher_from_mlx_checkpoint_dir():
    assert _publisher("/llm/mlx/mlx-community__Qwen3.6-27B-bf16") == "mlx-community"


def test_publisher_absent_when_layout_has_no_org():
    assert _publisher("/llm/gguf/loose-collection/model-Q6_K.gguf") is None
    assert _publisher("/models/plain-dir") is None


def _record(student_name: str, mean: float, seed: int = 123,
            fmt: str = "mlx-affine", score_window=None,
            tok_mode: str = "mlx-student", floor_mean=None,
            tokens_hash: str = "aa00aa00aa00aa00") -> dict:
    return build_locked_json({
        "teacher": {"path": "org/teacher", "revision": None,
                    "precision": "bfloat16"},
        "student": {
            "path": f"/models/{student_name}",
            "format": fmt,
            "size_bytes": int(4e9),
            "effective_bpw": 4.5,
            "n_params": int(8e9),
            "quantization": {"kind": "affine", "bits": 4, "group_size": 64,
                             "mode": "affine"},
        },
        "tokenizer": {"mode": tok_mode, "identical": True,
                      "encoding_parity": True, "forced": False, "diffs": []},
        "calibration": {"corpus": "wikitext", "num_samples": 8,
                        "max_seq_len": 512, "seed": seed, "top_k": 128,
                        "score_window": score_window or [256, 512],
                        "long_context": False,
                        "corpus_tokens_hash": tokens_hash},
        "kld": {"mean": mean, "se": 0.0004, "floor_mean": floor_mean,
                "p50": 0.01, "p95": 0.1, "p99": 0.2, "p999": 0.5, "max": 1.0},
        "delta_p": {"mean": -0.001, "se": 0.0001, "rms": 0.01},
        "agreement": {"top1": 0.95, "top5": 0.99},
        "tokens_scored": 1000,
        "elapsed_seconds": 1.0,
        "scorer_version": "mlx-kld 0.1.0",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "by_position": [],
        "kld_histogram": {"bin_edges": [], "counts": []},
        "cache": {"dir": "/tmp/c", "status": "HIT", "top_k": 128},
    })


def _write(root, teacher_slug, name, record):
    d = root / teacher_slug
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(record))


def _run(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "mlx_kld", *args],
        capture_output=True, text=True, env=env,
    )


def test_compare_prints_table_sorted_by_mean(tmp_path):
    _write(tmp_path, "teacher", "worse-q3.aaaaaaaa.json",
           _record("worse-q3", mean=0.05))
    _write(tmp_path, "teacher", "better-q6.bbbbbbbb.json",
           _record("better-q6", mean=0.005))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "KLD comparison" in r.stdout
    assert "mean KLD +/- se" in r.stdout
    assert "0.0050 +/- 0.0004" in r.stdout
    # sorted ascending: better-q6 row before worse-q3
    assert r.stdout.index("better-q6") < r.stdout.index("worse-q3")


def test_compare_skips_foreign_schema_versions(tmp_path):
    _write(tmp_path, "teacher", "good.aaaaaaaa.json", _record("good", 0.01))
    stray = dict(_record("stray", 0.001))
    stray["schema_version"] = 2  # lab-lineage file: must be skipped
    _write(tmp_path, "teacher", "stray-lab.json", stray)
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0
    assert "good" in r.stdout
    assert "stray" not in r.stdout
    assert "schema_version" in r.stderr  # skip warning names the reason


def test_compare_mixed_spec_legend(tmp_path):
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", _record("a", 0.01, seed=123))
    _write(tmp_path, "teacher", "b.bbbbbbbb.json", _record("b", 0.02, seed=777))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0
    assert "Mixed run specs" in r.stdout
    assert "Spec legend" in r.stdout


def test_compare_single_spec_has_no_spec_column(tmp_path):
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", _record("a", 0.01))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0
    assert "| spec |" not in r.stdout
    assert "All runs share one spec" in r.stdout


def test_publisher_column_is_opt_in(tmp_path):
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", _record("a", 0.01))
    off = _run("compare", "--out-dir", str(tmp_path))
    assert off.returncode == 0
    assert "| publisher |" not in off.stdout
    on = _run("compare", "--out-dir", str(tmp_path), "--publisher")
    assert on.returncode == 0
    assert "| publisher |" in on.stdout
    # /models/a carries no ORG__REPO, so the cell degrades to n/a, not a crash.
    assert "n/a" in on.stdout


def test_compare_keeps_runs_differing_in_window(tmp_path):
    # The digest keeps these apart on disk; compare must not collapse them.
    _write(tmp_path, "teacher", "s.aaaaaaaa.json",
           _record("same-student", 0.03, score_window=[256, 512]))
    _write(tmp_path, "teacher", "s.bbbbbbbb.json",
           _record("same-student", 0.09, score_window=[0, 2048]))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "loaded 2 runs" in r.stderr
    assert r.stdout.count("same-student") >= 2  # 2 table rows (+ sources)
    assert "Spec legend" in r.stdout


def test_compare_merges_tok_modes_on_matching_tokens_hash(tmp_path):
    # Identical scored stream (same content hash): an hf-source GGUF run and
    # an mlx-student run are one ranking, not a mixed-spec table.
    _write(tmp_path, "teacher", "a.aaaaaaaa.json",
           _record("mlx-5bit", 0.02, tok_mode="mlx-student"))
    _write(tmp_path, "teacher", "b.bbbbbbbb.json",
           _record("gguf-q5", 0.01, tok_mode="hf-source"))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "All runs share one spec" in r.stdout
    assert "Mixed run specs" not in r.stdout
    assert "tokenizer=hf-source, mlx-student" in r.stdout


def test_compare_splits_tok_modes_without_tokens_hash(tmp_path):
    # Pre-hash records: no content witness, so tokenizer mode still splits.
    _write(tmp_path, "teacher", "a.aaaaaaaa.json",
           _record("a", 0.02, tok_mode="mlx-student", tokens_hash=None))
    _write(tmp_path, "teacher", "b.bbbbbbbb.json",
           _record("b", 0.01, tok_mode="hf-source", tokens_hash=None))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "Mixed run specs" in r.stdout


def test_compare_dedupes_same_student_across_tok_modes(tmp_path):
    # Same student, same scored stream: the tok-mode variant is the same
    # measurement and collapses to one row.
    _write(tmp_path, "teacher", "s.aaaaaaaa.json",
           _record("same-student", 0.03, tok_mode="mlx-student"))
    _write(tmp_path, "teacher", "s.cccccccc.json",
           _record("same-student", 0.03, tok_mode="hf-source"))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "loaded 1 runs" in r.stderr


def test_compare_skips_invalid_records_without_crashing(tmp_path):
    _write(tmp_path, "teacher", "good.aaaaaaaa.json", _record("good", 0.01))
    # Right schema_version but nothing else: must be skipped, not crash.
    (tmp_path / "teacher" / "stub.json").write_text('{"schema_version": 1}')
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "good" in r.stdout
    assert "skipping" in r.stderr


def test_compare_survives_null_mean(tmp_path):
    rec = _record("null-mean", 0.01)
    rec["kld"]["mean"] = None
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", rec)
    _write(tmp_path, "teacher", "b.bbbbbbbb.json", _record("fine", 0.02))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    # null mean sorts last and renders as n/a
    assert r.stdout.index("fine") < r.stdout.index("null-mean")
    assert "n/a" in r.stdout


def test_compare_tiny_se_renders_scientific(tmp_path):
    rec = _record("near-lossless", 1e-4)
    rec["kld"]["se"] = 1e-5
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", rec)
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0
    assert "+/- 0.0000" not in r.stdout
    assert "1.0e-05" in r.stdout


def test_compare_empty_root_fails_cleanly(tmp_path):
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 1
    assert "no records found" in r.stderr
    assert "[" not in r.stderr.split("no records found")[1].split("\n")[0]


def test_compare_marks_floor_limited_runs(tmp_path):
    _write(tmp_path, "teacher", "a.aaaaaaaa.json",
           _record("at-floor", 0.0002 * FLOOR_CAUTION_MULT * 0.5, floor_mean=0.0002))
    _write(tmp_path, "teacher", "b.bbbbbbbb.json",
           _record("well-above", 0.05, floor_mean=0.0002))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "(f)" in r.stdout
    assert "floor-limited" in r.stdout
    # only the at-floor row is marked
    above_row = [ln for ln in r.stdout.splitlines() if "well-above" in ln
                 and ln.startswith("|")][0]
    assert "(f)" not in above_row


def test_compare_no_floor_marker_without_floor(tmp_path):
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", _record("tiny", 1e-5))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0
    assert "floor-limited" not in r.stdout


def test_compare_splits_specs_on_corpus_hash_drift(tmp_path):
    # Same recorded spec, different scored tokens: never one ranking.
    _write(tmp_path, "teacher", "a.aaaaaaaa.json",
           _record("a", 0.01, tokens_hash="aa00aa00aa00aa00"))
    _write(tmp_path, "teacher", "b.aaaaaaaa2.json",
           _record("b", 0.02, tokens_hash="bb11bb11bb11bb11"))
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "Mixed run specs" in r.stdout
    assert "aa00aa00aa00aa00" in r.stdout
    assert "bb11bb11bb11bb11" in r.stdout


def test_rollup_subcommand_is_gone():
    r = _run("rollup")
    assert r.returncode != 0


def test_compare_survives_json_that_is_not_an_object(tmp_path):
    """Valid JSON need not be a JSON *object*. A stray list or bare string under
    the results root must not take out the whole invocation."""
    _write(tmp_path, "teacher", "good.aaaaaaaa.json", _record("good", 0.01))
    (tmp_path / "teacher" / "list.json").write_text("[1, 2, 3]")
    (tmp_path / "teacher" / "str.json").write_text('"just a string"')
    (tmp_path / "teacher" / "num.json").write_text("42")
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "good" in r.stdout
    assert "not a JSON object" in r.stderr


def test_compare_survives_a_scalar_score_window(tmp_path):
    """score_window is not type-checked by the locked schema, so a hand-edited
    record can carry a scalar there. Grouping must degrade, not raise."""
    _write(tmp_path, "teacher", "good.aaaaaaaa.json", _record("good", 0.01))
    bad = _record("bad-window", 0.02)
    bad["calibration"]["score_window"] = 5
    _write(tmp_path, "teacher", "bad.bbbbbbbb.json", bad)
    r = _run("compare", "--out-dir", str(tmp_path))
    assert r.returncode == 0, r.stderr
    assert "good" in r.stdout
    assert "bad-window" in r.stdout
