"""Tests for the locked schema_version=1 JSON record + validator."""

import pytest

from mlx_kld._constants import SCHEMA_VERSION
from mlx_kld.errors import RecordSchemaError
from mlx_kld.report import build_locked_json, validate_locked_schema


def _well_formed_report() -> dict:
    return {
        "teacher": {"path": "org/teacher", "revision": None, "precision": "bfloat16"},
        "student": {
            "path": "/tmp/student",
            "format": "mlx-affine",
            "size_bytes": 1,
            "effective_bpw": 4.0,
            "n_params": 1000,
            "quantization": {"kind": "affine", "bits": 4, "group_size": 64,
                             "mode": "affine"},
        },
        "tokenizer": {
            "mode": "mlx-student",
            "identical": True,
            "encoding_parity": True,
            "forced": False,
            "diffs": [],
        },
        "calibration": {
            "corpus": "wikitext",
            "num_samples": 8,
            "max_seq_len": 512,
            "seed": 123,
            "top_k": 128,
            "score_window": [256, 512],
            "long_context": False,
        },
        "kld": {"mean": 0.1, "se": 0.001, "floor_mean": 0.0002, "p50": 0.05,
                "p95": 0.3, "p99": 0.6, "p999": 1.0, "max": 2.0},
        "delta_p": {"mean": -0.002, "se": 0.0001, "rms": 0.02},
        "agreement": {"top1": 0.9, "top5": 0.99},
        "tokens_scored": 100,
        "elapsed_seconds": 1.5,
        "scorer_version": "mlx-kld 0.1.0",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "by_position": [],
        "kld_histogram": {"bin_edges": [], "counts": []},
        "cache": {"dir": "/tmp/cache", "status": "HIT", "top_k": 128},
    }


def test_schema_version_is_one():
    assert SCHEMA_VERSION == 1


def test_build_locked_json_passes_validator():
    payload = build_locked_json(_well_formed_report())
    assert payload["schema_version"] == SCHEMA_VERSION
    validate_locked_schema(payload)  # must not raise


def test_locked_json_has_no_recipe_key():
    payload = build_locked_json(_well_formed_report())
    assert "recipe" not in payload


def test_plausibility_field_present_and_non_locked():
    # Clean report: implausible False, validator still passes.
    payload = build_locked_json(_well_formed_report())
    assert payload["plausibility"] == {"implausible": False, "reason": None}
    validate_locked_schema(payload)
    # Broken-load report: flagged, and still schema-valid (non-locked field).
    report = _well_formed_report()
    report["agreement"]["top1"] = 0.002
    report["kld"]["mean"] = 16.4
    payload = build_locked_json(report)
    assert payload["plausibility"]["implausible"] is True
    assert payload["plausibility"]["reason"]
    validate_locked_schema(payload)


def test_validator_rejects_empty():
    with pytest.raises(RecordSchemaError):
        validate_locked_schema({})


@pytest.mark.parametrize("mutate,match", [
    (lambda p: p["student"].pop("quantization"), "student.quantization"),
    (lambda p: p["student"].__setitem__("format", "gptq"), "student.format"),
    (lambda p: p["student"].__setitem__("quantization", {}), "kind"),
    (lambda p: p.pop("tokenizer"), "tokenizer"),
    (lambda p: p["tokenizer"].pop("mode"), "tokenizer.mode"),
    (lambda p: p["kld"].pop("se"), "kld.se"),
    (lambda p: p["kld"].pop("floor_mean"), "kld.floor_mean"),
    (lambda p: p.pop("delta_p"), "delta_p"),
    (lambda p: p["delta_p"].pop("rms"), "delta_p.rms"),
])
def test_validator_rejects_mutations(mutate, match):
    payload = build_locked_json(_well_formed_report())
    mutate(payload)
    with pytest.raises(RecordSchemaError, match=match):
        validate_locked_schema(payload)


def test_validator_accepts_every_format():
    for fmt, q in [
        ("mlx-affine", {"kind": "affine", "bits": 4, "group_size": 64,
                        "mode": "affine"}),
        ("mlx-kquant", {"kind": "kquant", "codecs": {"q6_k": 100}}),
        ("gguf", {"kind": "gguf", "arch": "qwen3", "codecs": {"q6_k": 100}}),
        ("mlx", {"kind": "none", "dtype": "bfloat16"}),
    ]:
        report = _well_formed_report()
        report["student"]["format"] = fmt
        report["student"]["quantization"] = q
        validate_locked_schema(build_locked_json(report))


def test_provenance_is_optional_and_non_locked():
    report = _well_formed_report()
    payload = build_locked_json(report)
    validate_locked_schema(payload)  # absent: fine
    report["student"]["provenance"] = {"tool": "some-quantizer"}
    payload = build_locked_json(report)
    validate_locked_schema(payload)  # present: also fine, carried through
    assert payload["student"]["provenance"] == {"tool": "some-quantizer"}


def test_locked_key_set_is_pinned():
    """The locked top-level keys are a contract with downstream consumers, and
    the strict validator gates the *reader* too: adding a key to the locked set
    would make `compare` warn-skip every previously written record. New data
    goes under the extras; this pin is the tripwire."""
    LOCKED = {
        "schema_version", "teacher", "student", "tokenizer", "calibration",
        "kld", "delta_p", "agreement", "tokens_scored", "elapsed_seconds",
        "scorer_version", "timestamp",
    }
    EXTRAS = {
        "tokens_dropped_nonfinite", "by_position", "kld_histogram", "cache",
        "plausibility",
    }
    payload = build_locked_json(_well_formed_report())
    assert set(payload) == LOCKED | EXTRAS
    # A record stripped to the locked keys alone must still validate: that is
    # what makes every EXTRAS key genuinely optional for old/foreign readers.
    validate_locked_schema({k: v for k, v in payload.items() if k in LOCKED})
