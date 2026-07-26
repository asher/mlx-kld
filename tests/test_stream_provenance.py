"""One cache entry is shared by every student of a teacher, so a HIT replays
whatever ids the first student tokenized. These cover the detection of that case
and the way the report owns up to it."""

from __future__ import annotations

from mlx_kld.report import render_markdown
from mlx_kld.tokenizer import cached_stream_status


def _renderable_report(**tokenizer_overrides) -> dict:
    tokenizer = {
        "mode": "synthesized",
        "identical": True,
        "encoding_parity": True,
        "forced": False,
        "diffs": [],
    }
    tokenizer.update(tokenizer_overrides)
    return {
        "teacher": {"path": "org/teacher", "revision": None, "precision": "bfloat16"},
        "student": {"path": "/tmp/student.gguf", "format": "gguf", "size_bytes": 1,
                    "effective_bpw": 4.0, "n_params": 1000,
                    "quantization": {"kind": "gguf", "arch": "qwen3",
                                     "codecs": {"q4_k": 10}}},
        "tokenizer": tokenizer,
        "calibration": {"corpus": "wikitext", "num_samples": 8, "max_seq_len": 512,
                        "seed": 123, "top_k": 128, "score_window": [256, 512]},
        "kld": {"mean": 0.1, "se": 0.001, "floor_mean": 0.0002, "p50": 0.05,
                "p95": 0.3, "p99": 0.6, "p999": 1.0, "max": 2.0},
        "delta_p": {"mean": -0.002, "se": 0.0001, "rms": 0.02},
        "agreement": {"top1": 0.9, "top5": 0.99},
        "tokens_scored": 100,
        "elapsed_seconds": 1.5,
        "elapsed_phase": "warm: student only, cache HIT",
        "scorer_version": "mlx-kld 0.1.0",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "cache": {"dir": "/tmp/cache", "status": "HIT", "top_k": 128},
    }


# ---------- detection ----------

def test_matching_stream_is_the_students():
    assert cached_stream_status("aaaa1111", "aaaa1111", "mlx-student") == (True, None)


def test_pre_hash_cache_is_unknown_not_false():
    # An older manifest carries no witness; that is not evidence of a mismatch.
    assert cached_stream_status(None, "aaaa1111", "mlx-student") == (None, None)


def test_diverging_stream_is_flagged_with_both_hashes():
    ok, msg = cached_stream_status("aaaa1111", "bbbb2222", "mlx-student")
    assert ok is False
    assert "aaaa1111" in msg and "bbbb2222" in msg
    assert "--rebuild-cache" in msg


def test_synthesized_mode_gets_the_stronger_warning():
    """The default GGUF mode promises to test the GGUF as shipped, tokenizer
    included, so a borrowed stream defeats the mode rather than merely differing."""
    _, generic = cached_stream_status("aaaa1111", "bbbb2222", "mlx-student")
    _, synth = cached_stream_status("aaaa1111", "bbbb2222", "synthesized")
    assert "defeats" in synth and "--hf-source" in synth
    assert "defeats" not in generic


# ---------- the report owns up to it ----------

def test_report_owns_up_to_a_borrowed_stream():
    md = render_markdown(_renderable_report(stream_is_students=False))
    assert "NOT this student's tokenization" in md
    assert "--rebuild-cache" in md


def test_report_stays_quiet_when_the_stream_is_the_students():
    for value in (True, None):
        md = render_markdown(_renderable_report(stream_is_students=value))
        assert "NOT this student's tokenization" not in md
    # and when the field is absent entirely (records written before this check)
    assert "NOT this student's tokenization" not in render_markdown(_renderable_report())
