"""Markdown report rendering: number formatting and per-format quantization
sections (edge cases only; the full report shape is covered by test_schema)."""

from mlx_kld.report import (
    FLOOR_CAUTION_MULT,
    _fmt_pm,
    _fmt_pp,
    _fmt_pp_pm,
    _fmt_stat,
    _quantization_lines,
    floor_limited,
    implausibility_reason,
)

# ---------- implausible-result guard ----------

def test_implausible_flags_broken_load_by_top1():
    # A checkpoint that loaded with double-shifted norms: near-chance top-1
    # agreement paired with a mean KLD orders of magnitude above any real quant.
    reason = implausibility_reason({"top1": 0.0022, "top5": 0.01},
                                   {"mean": 16.43})
    assert reason is not None
    assert "top-1" in reason and "mean KLD" in reason


def test_implausible_flags_high_mean_alone():
    reason = implausibility_reason({"top1": 0.7}, {"mean": 2.5})
    assert reason is not None and "mean KLD" in reason and "top-1" not in reason


def test_real_quant_is_plausible():
    # Worst real quant we measured (mlx 4bit): top-1 89.96%, mean 0.0573.
    assert implausibility_reason({"top1": 0.8996}, {"mean": 0.0573}) is None
    # Boundary: exactly at the threshold is not flagged (strict <).
    assert implausibility_reason({"top1": 0.5}, {"mean": 1.0}) is None


def test_implausible_tolerates_missing_metrics():
    assert implausibility_reason(None, None) is None
    assert implausibility_reason({}, {"mean": None}) is None


# ---------- floor-limited caution ----------

def test_floor_limited_thresholds():
    floor = 0.0002
    # Expressed relative to the constant so moving the threshold does not
    # silently invert what these assert.
    assert floor_limited({"mean": floor * (FLOOR_CAUTION_MULT * 0.5),
                          "floor_mean": floor})
    assert not floor_limited({"mean": floor * (FLOOR_CAUTION_MULT * 2),
                              "floor_mean": floor})
    # Strict <: exactly at the threshold is not flagged.
    assert not floor_limited({"mean": floor * FLOOR_CAUTION_MULT,
                              "floor_mean": floor})
    assert not floor_limited({"mean": 0.0005, "floor_mean": None})
    assert not floor_limited({"mean": None, "floor_mean": 0.0002})
    assert not floor_limited({"mean": 0.0005, "floor_mean": 0.0})


def test_floor_caution_does_not_flag_agreeing_precise_quants():
    """The 27B run's best quants sit at 2.9x floor and match published
    llama.cpp numbers; a threshold that flags them contradicts that evidence."""
    floor = 0.0018069
    for mean in (0.00521, 0.00592, 0.00824, 0.00867, 0.00906, 0.00943):
        assert not floor_limited({"mean": mean, "floor_mean": floor}), mean


# ---------- number formatting ----------

def test_fmt_stat_fixed_point_normal_range():
    assert _fmt_stat(0.0248) == "0.0248"
    assert _fmt_stat(0.0) == "0.0000"


def test_fmt_stat_scientific_below_resolution():
    # A nonzero value must never render as all zeros at the given precision.
    assert _fmt_stat(1e-5) == "1.0e-05"
    assert _fmt_stat(-3e-6, 5) == "-3.0e-06"
    assert _fmt_stat(3e-5, 5) == "0.00003"  # representable at 5dp: stays fixed


def test_fmt_pm_tiny_se_not_flattened():
    assert _fmt_pm(0.0001, 1e-5) == "0.0001 +/- 1.0e-05"
    assert _fmt_pm(0.0248, None) == "0.0248"
    assert _fmt_pm(None, None) == "n/a"


# ---------- quantization section ----------

def _lines(student):
    return _quantization_lines(student)


def test_unquantized_folds_bpw_into_one_line():
    lines = _lines({
        "format": "mlx",
        "quantization": {"kind": "none", "dtype": "bfloat16"},
        "effective_bpw": 16.0,
        "n_params": 68_000_000,
    })
    assert "- unquantized (bfloat16, 16.0 bpw)" in lines
    assert not any(line.startswith("- effective_bpw") for line in lines)
    # Sub-1B models report in M, not 0.07B.
    assert "- parameters: 68M" in lines


def test_kquant_empty_codecs_renders_na_not_empty_span():
    lines = _lines({
        "format": "mlx-kquant",
        "quantization": {"kind": "kquant", "codecs": {}},
        "effective_bpw": None,
        "n_params": None,
    })
    assert "- codecs: n/a" in lines
    assert "``" not in "\n".join(lines)


def test_gguf_section_has_arch_codecs_bpw_params():
    lines = _lines({
        "format": "gguf",
        "quantization": {"kind": "gguf", "arch": "qwen3",
                         "codecs": {"q4_k": 168, "q6_k": 29, "f32": 113}},
        "effective_bpw": 4.82,
        "n_params": 8_000_000_000,
    })
    joined = "\n".join(lines)
    assert "- arch: `qwen3`" in lines
    assert "q4_k:168" in joined
    assert "- effective_bpw: 4.820" in lines
    assert "- parameters: 8.00B" in lines


def test_zero_n_params_still_reported():
    lines = _lines({
        "format": "mlx-affine",
        "quantization": {"kind": "affine", "bits": 4, "group_size": 64,
                         "mode": "affine"},
        "effective_bpw": None,
        "n_params": 0,
    })
    assert "- parameters: 0M" in lines


# ---------- Delta-p rendering (llama.cpp prints percentage points) ----------

def test_delta_p_renders_as_percentage_points():
    # llama.cpp reports Mean/RMS dp scaled by 100; raw probability here would
    # sit 100x from any published table a reader compares against.
    assert _fmt_pp(-0.002) == "-0.200%"
    assert _fmt_pp(0.02) == "2.000%"
    assert _fmt_pp(None) == "n/a"


def test_delta_p_pm_renders_both_terms_scaled():
    assert _fmt_pp_pm(-0.002, 0.0001) == "-0.200% +/- 0.010%"
    assert _fmt_pp_pm(-0.002, None) == "-0.200%"
    assert _fmt_pp_pm(None, 0.1) == "n/a"


def test_tiny_delta_p_does_not_flatten_to_zero():
    # Near-lossless students live below the fixed-point resolution.
    assert _fmt_pp(1e-8) == "1.0e-06%"


def test_json_record_keeps_raw_probability():
    """Only the rendering scales; schema_version=1 consumers are unaffected."""
    from mlx_kld.report import build_locked_json
    from tests.test_schema import _well_formed_report

    payload = build_locked_json(_well_formed_report())
    assert payload["delta_p"]["mean"] == -0.002
    assert payload["delta_p"]["rms"] == 0.02


# ---------- model-card block ----------

def _card_report(**overrides) -> dict:
    from tests.test_schema import _well_formed_report

    report = _well_formed_report()
    report.setdefault("cache", {"dir": "/tmp/c", "status": "HIT", "top_k": 32768})
    for k, v in overrides.items():
        report[k] = v
    return report


def test_card_carries_the_headline_metrics():
    from mlx_kld.report import render_card

    card = render_card(_card_report())
    assert "Mean KL divergence" in card
    assert "Top-1 agreement" in card
    # The spec is what the numbers are comparable within, so it has to be there.
    assert "seed" in card and "top-k" in card


def test_card_never_leaks_a_local_teacher_path():
    """The card is written to be published. A home directory in it would travel
    into whatever model card it is pasted into."""
    from mlx_kld.report import render_card

    report = _card_report()
    report["teacher"] = dict(report["teacher"],
                             path="/Users/someone/models/Qwen3.6-27B-bf16")
    card = render_card(report)
    assert "/Users/someone" not in card
    assert "qwen3.6-27b-bf16" in card


def test_card_keeps_a_hub_id_intact():
    """An HF id is public and identifies the teacher exactly, so it survives."""
    from mlx_kld.report import render_card

    report = _card_report()
    report["teacher"] = dict(report["teacher"], path="Qwen/Qwen3-0.6B")
    assert "Qwen/Qwen3-0.6B" in render_card(report)


def test_card_never_leaks_the_student_path():
    from mlx_kld.report import render_card

    report = _card_report()
    report["student"] = dict(report["student"], path="/Users/someone/q4-build")
    card = render_card(report)
    assert "/Users/someone" not in card
    assert "<path-to-this-checkpoint>" in card


def test_card_flags_a_floor_limited_result():
    from mlx_kld.report import render_card

    report = _card_report()
    report["kld"] = dict(report["kld"], mean=0.002, floor_mean=0.0018)
    assert "reconstruction floor" in render_card(report)


def test_card_flags_an_implausible_result_over_the_floor_note():
    """A broken load is the more urgent thing to say, so it wins the slot."""
    from mlx_kld.report import render_card

    report = _card_report()
    report["agreement"] = dict(report["agreement"], top1=0.01)
    report["kld"] = dict(report["kld"], mean=0.002, floor_mean=0.0018)
    card = render_card(report)
    assert "implausible result" in card
    assert "reconstruction floor" not in card
