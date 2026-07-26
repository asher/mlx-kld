"""CLI dispatcher integration tests (subprocess; no model load)."""

import subprocess
import sys


def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "mlx_kld", *args],
        capture_output=True, text=True,
    )


def test_self_test_subcommand_passes():
    r = _run("self-test")
    assert r.returncode == 0, r.stderr
    assert "All self-tests passed" in r.stderr


def test_subcommands_listed_in_help():
    r = _run("--help")
    assert r.returncode == 0
    for cmd in ("score", "compare", "plot", "cache", "floor-sweep", "self-test"):
        assert cmd in r.stdout
    assert "rollup" not in r.stdout


def test_score_help():
    r = _run("score", "--help")
    assert r.returncode == 0
    assert "--cache-max-gb" in r.stdout


def test_cache_list_on_missing_dir(tmp_path):
    r = _run("cache", "--cache-dir", str(tmp_path / "nope"), "list")
    assert r.returncode == 0
    assert "empty" in r.stdout


def test_cache_dir_accepted_after_the_op(tmp_path):
    r = _run("cache", "list", "--cache-dir", str(tmp_path / "nope"))
    assert r.returncode == 0
    assert "empty" in r.stdout


def test_cache_gc_rejects_zero_budget(tmp_path):
    r = _run("cache", "gc", "--cache-dir", str(tmp_path), "--max-gb", "0")
    assert r.returncode == 1
    assert "cache clear" in r.stderr


def test_cache_gc_no_args_errors_to_stderr(tmp_path):
    r = _run("cache", "gc", "--cache-dir", str(tmp_path))
    assert r.returncode == 1
    assert "--max-gb" in r.stderr


def test_cache_clear_all_requires_yes(tmp_path):
    r = _run("cache", "clear", "--cache-dir", str(tmp_path))
    assert r.returncode == 1
    assert "--yes" in r.stderr


def test_cache_clear_all_with_yes_proceeds(tmp_path):
    r = _run("cache", "clear", "--cache-dir", str(tmp_path), "--yes")
    assert r.returncode == 0
    assert "nothing removed" in r.stdout


def test_cache_clear_with_teacher_filter_needs_no_yes(tmp_path):
    r = _run("cache", "clear", "--cache-dir", str(tmp_path), "--teacher", "qwen")
    assert r.returncode == 0
    assert "nothing removed" in r.stdout


# ---------- score argument validation (all before any model load) ----------

def test_score_rejects_a_missing_gguf_student(tmp_path):
    r = _run("score", "org/teacher", str(tmp_path / "absent.gguf"))
    assert r.returncode != 0
    assert "does not exist" in r.stderr


def test_score_rejects_a_gguf_path_that_is_a_directory(tmp_path):
    d = tmp_path / "model.gguf"
    d.mkdir()
    r = _run("score", "org/teacher", str(d))
    assert r.returncode != 0
    assert "not a regular file" in r.stderr


def test_score_rejects_a_student_that_is_not_a_directory(tmp_path):
    f = tmp_path / "weights.bin"
    f.write_bytes(b"\0")
    r = _run("score", "org/teacher", str(f))
    assert r.returncode != 0
    assert "not a directory" in r.stderr


def test_hf_source_is_rejected_for_a_safetensors_student(tmp_path):
    """--hf-source swaps the student tokenizer for the teacher's, which only
    has meaning for a GGUF whose tokenizer was synthesized from metadata."""
    (tmp_path / "student").mkdir()
    r = _run("score", "org/teacher", str(tmp_path / "student"),
             "--hf-source", "org/teacher")
    assert r.returncode != 0
    assert "only applies to GGUF" in r.stderr


def test_score_window_must_parse_as_start_colon_end(tmp_path):
    (tmp_path / "student").mkdir()
    r = _run("score", "org/teacher", str(tmp_path / "student"),
             "--score-window", "256-512")
    assert r.returncode != 0
    assert "start:end" in r.stderr


def test_score_window_must_lie_inside_the_sequence(tmp_path):
    (tmp_path / "student").mkdir()
    r = _run("score", "org/teacher", str(tmp_path / "student"),
             "--max-seq-len", "512", "--score-window", "256:1024")
    assert r.returncode != 0
    assert "0 <= start < end <= max_seq_len" in r.stderr


def test_score_window_must_be_non_empty(tmp_path):
    (tmp_path / "student").mkdir()
    r = _run("score", "org/teacher", str(tmp_path / "student"),
             "--max-seq-len", "512", "--score-window", "256:256")
    assert r.returncode != 0
    assert "0 <= start < end <= max_seq_len" in r.stderr


def test_score_help_documents_the_streaming_flag():
    r = _run("score", "--help")
    assert r.returncode == 0
    assert "--stream-dataset" in r.stdout


def test_bare_invocation_is_a_usage_failure():
    """A script that runs `mlx-kld` with no subcommand asked for nothing and
    got nothing; help is the diagnostic, so it belongs on stderr behind a
    nonzero status rather than reading as successful output."""
    r = _run()
    assert r.returncode == 2
    assert "<command>" in r.stderr
    assert r.stdout == ""


def test_numeric_protocol_arguments_must_be_positive(tmp_path):
    """--top-k 0 would slice the whole vocab via `[..., -0:]` and --batch-size 0
    divides by zero, both far downstream of where the user could tell why."""
    student = tmp_path / "student"
    student.mkdir()
    for flag in ("--num-samples", "--top-k", "--batch-size", "--max-seq-len"):
        r = _run("score", "org/teacher", str(student), flag, "0")
        assert r.returncode == 1, f"{flag}: {r.returncode}"
        assert f"error: {flag} must be > 0" in r.stderr, f"{flag}: {r.stderr}"


def test_argument_errors_use_the_same_prefix_as_library_errors(tmp_path):
    """`error: ` on stderr, exit 1, the dispatcher's format for a deliberate
    failure. An argparse-level failure (exit 2) means the CLI never ran."""
    r = _run("score", "org/teacher", str(tmp_path / "nope.gguf"))
    assert r.returncode == 1
    assert r.stderr.startswith("error: ")
