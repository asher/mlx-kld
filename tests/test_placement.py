"""JSON record placement: results root resolution + digest8 path layout.

Records must land under the results root, never inside model directories,
and distinct run specs must never overwrite each other.
"""

import subprocess
import sys
from pathlib import Path

from mlx_kld._constants import DEFAULT_RESULTS_DIR, RESULTS_ENV
from mlx_kld.report import resolve_results_dir, results_json_path, run_digest

_CAL = {
    "corpus": "Salesforce/wikitext:wikitext-103-raw-v1",
    "num_samples": 512,
    "max_seq_len": 512,
    "seed": 123,
    "top_k": 32768,
    "score_window": [256, 512],
    "long_context": False,
}


# ---------- results root resolution ----------

def test_resolve_explicit_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(RESULTS_ENV, str(tmp_path / "env"))
    assert resolve_results_dir(tmp_path / "cli") == tmp_path / "cli"


def test_resolve_env_beats_default(monkeypatch, tmp_path):
    monkeypatch.setenv(RESULTS_ENV, str(tmp_path / "env"))
    assert resolve_results_dir(None) == tmp_path / "env"


def test_resolve_default(monkeypatch):
    monkeypatch.delenv(RESULTS_ENV, raising=False)
    assert resolve_results_dir(None) == DEFAULT_RESULTS_DIR


# ---------- path layout ----------

def test_layout_teacher_slug_and_student_name():
    p = results_json_path(Path("out"), "Qwen/Qwen3-0.6B",
                          Path("/models/Qwen3-0.6B-q4"), _CAL, "mlx-student")
    # The org stays in the slug; without it two orgs' same-named models collide.
    assert p.parent == Path("out") / "qwen__qwen3-0.6b"
    assert p.name.startswith("Qwen3-0.6B-q4.")
    assert p.suffix == ".json"
    digest = p.name.split(".")[-2]
    assert len(digest) == 8


def test_teacher_slug_keeps_org_so_same_model_name_does_not_collide():
    a = results_json_path(Path("out"), "mlx-community/Qwen3-0.6B-bf16",
                          Path("/m/q4"), _CAL, "mlx-student")
    b = results_json_path(Path("out"), "Qwen/Qwen3-0.6B-bf16",
                          Path("/m/q4"), _CAL, "mlx-student")
    assert a.parent != b.parent
    assert a != b


def test_teacher_slug_recovers_org_name_from_hf_snapshot_path():
    """A snapshot basename is a commit sha, unreadable and unstable across
    re-pulls, so the models-- segment supplies the directory name."""
    snap = ("/home/u/.cache/huggingface/hub/models--mlx-community--Qwen3.6-27B-bf16"
            "/snapshots/0e0a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c")
    p = results_json_path(Path("out"), snap, Path("/m/q4"), _CAL, "mlx-student")
    assert p.parent == Path("out") / "mlx-community__qwen3.6-27b-bf16"


def test_same_basename_students_do_not_overwrite_each_other():
    """`./out/q4/` is a common quantizer output convention, so two unrelated
    students very plausibly share a basename."""
    a = results_json_path(Path("out"), "org/t", Path("/runA/q4"), _CAL, "mlx-student")
    b = results_json_path(Path("out"), "org/t", Path("/runB/q4"), _CAL, "mlx-student")
    assert a.parent == b.parent          # same teacher, same directory
    assert a.name.startswith("q4.")      # readable slug preserved
    assert a != b                        # but distinct records


def test_gguf_and_dir_students_sharing_a_stem_do_not_collide():
    a = results_json_path(Path("out"), "org/t", Path("/x/q4.gguf"), _CAL, "synthesized")
    b = results_json_path(Path("out"), "org/t", Path("/y/q4.gguf"), _CAL, "synthesized")
    assert a != b


def test_identical_rerun_is_still_idempotent(tmp_path):
    student = tmp_path / "q4"
    student.mkdir()
    args = (Path("out"), "org/t", student, _CAL, "mlx-student")
    assert results_json_path(*args) == results_json_path(*args)


def test_path_spelling_variants_resolve_to_one_record(tmp_path):
    student = tmp_path / "q4"
    student.mkdir()
    direct = results_json_path(Path("out"), "org/t", student, _CAL, "mlx-student")
    indirect = results_json_path(Path("out"), "org/t",
                                 tmp_path / "." / "q4", _CAL, "mlx-student")
    assert direct == indirect


def test_layout_gguf_uses_stem():
    p = results_json_path(Path("out"), "Qwen/Qwen3-0.6B",
                          Path("/gguf/Qwen3-0.6B-Q4_K_M.gguf"), _CAL, "synthesized")
    assert p.name.startswith("Qwen3-0.6B-Q4_K_M.")
    assert ".gguf" not in p.name


def test_record_path_is_never_inside_the_student_dir(tmp_path):
    student = tmp_path / "student-q4"
    p = results_json_path(tmp_path / "kld-results", "org/teacher",
                          student, _CAL, "mlx-student")
    assert student not in p.parents


# ---------- digest semantics ----------

def test_digest_stable_across_identical_reruns():
    assert run_digest(_CAL, "synthesized") == run_digest(dict(_CAL), "synthesized")


def test_digest_differs_across_tokenizer_modes():
    # The synthesized-vs-hf-source silent-overwrite bug: distinct modes must
    # write distinct records.
    assert run_digest(_CAL, "synthesized") != run_digest(_CAL, "hf-source")


def test_digest_differs_across_calibration_and_window():
    for key, value in [("corpus", "other"), ("num_samples", 64), ("seed", 7),
                       ("score_window", [0, 512]), ("top_k", 128)]:
        cal = dict(_CAL, **{key: value})
        assert run_digest(cal, "mlx-student") != run_digest(_CAL, "mlx-student"), key


# ---------- CLI surface ----------

def test_score_help_mentions_out_dir():
    r = subprocess.run(
        [sys.executable, "-m", "mlx_kld", "score", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "--out-dir" in r.stdout
    assert "kld-vs-" not in r.stdout  # model-dir sidecar default is gone
