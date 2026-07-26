"""The vendored qwen3_5 norm-shift fix: the +1.0 zero-centered-norm shift must
gate on conv1d state only, not on leftover mtp weights (mlx-lm PR #990). This
is what keeps already-converted MLX checkpoints (mtp tensors present, norms
already standard) from being double-shifted into garbage."""

import types

import mlx.core as mx
import pytest

from mlx_kld.models import (
    _qwen35_conv1d_gated_sanitize,
    _sanitize_still_gates_on_mtp,
)


def _fake_text_model(tie: bool = False):
    return types.SimpleNamespace(
        args=types.SimpleNamespace(tie_word_embeddings=tie)
    )


def test_no_shift_when_conv1d_already_sanitized():
    # Already-converted checkpoint: conv1d transposed (last dim == 1) and mtp
    # tensors left in. Norms are in standard form and must be passed through
    # untouched, which is the observed mtp-carrying failure mode.
    w = {
        "model.layers.0.input_layernorm.weight": mx.full((4,), 0.9),
        "model.layers.0.linear_attn.conv1d.weight": mx.zeros((8, 4, 1)),
        "mtp.norm.weight": mx.ones((4,)),
    }
    out = _qwen35_conv1d_gated_sanitize(_fake_text_model(), w)
    assert "mtp.norm.weight" not in out  # mtp dropped
    assert bool(mx.allclose(
        out["model.layers.0.input_layernorm.weight"], mx.full((4,), 0.9)
    ))


def test_shift_when_conv1d_unsanitized():
    # Raw checkpoint: conv1d not yet transposed (last dim != 1). Norms are
    # zero-centered and must get the +1.0 shift; conv1d gets transposed.
    w = {
        "model.layers.0.input_layernorm.weight": mx.zeros((4,)),
        "model.norm.weight": mx.zeros((4,)),
        "model.layers.0.linear_attn.conv1d.weight": mx.zeros((8, 1, 4)),
    }
    out = _qwen35_conv1d_gated_sanitize(_fake_text_model(), w)
    assert out["model.layers.0.linear_attn.conv1d.weight"].shape[-1] == 1
    assert bool(mx.allclose(
        out["model.layers.0.input_layernorm.weight"], mx.ones((4,))
    ))
    assert bool(mx.allclose(out["model.norm.weight"], mx.ones((4,))))


def test_tied_lm_head_dropped():
    w = {
        "lm_head.weight": mx.zeros((2, 2)),
        "model.norm.weight": mx.ones((4,)),
    }
    out = _qwen35_conv1d_gated_sanitize(_fake_text_model(tie=True), w)
    assert "lm_head.weight" not in out


# ---------- the vendored patch gates itself off once upstream lands #990 ----------

def test_gate_detects_stock_mtp_proxy():
    def stock_sanitize(self, weights):
        has_mtp_weights = any("mtp." in k for k in weights)
        should_shift = has_mtp_weights or True
        return weights, should_shift
    assert _sanitize_still_gates_on_mtp(stock_sanitize) is True


def test_gate_detects_a_fixed_upstream():
    def fixed_sanitize(self, weights):
        should_shift = any("conv1d.weight" in k for k in weights)
        return weights, should_shift
    assert _sanitize_still_gates_on_mtp(fixed_sanitize) is False


def test_gate_fails_safe_when_source_is_unreadable():
    """Applying a strict narrowing when it was not needed is harmless; skipping
    it when it was needed silently corrupts logits. Default to applying."""
    assert _sanitize_still_gates_on_mtp(len) is True          # builtin, no source
    assert _sanitize_still_gates_on_mtp(object()) is True     # not a function


def test_real_mlx_lm_is_still_stock():
    """Guards the TODO: when this starts failing, upstream has shipped #990 and
    the vendored patch should be deleted."""
    qwen3_5 = pytest.importorskip("mlx_lm.models.qwen3_5")
    assert _sanitize_still_gates_on_mtp(qwen3_5.TextModel.sanitize) is True, (
        "mlx-lm now gates the qwen3_5 norm shift on conv1d state; drop the "
        "vendored patch in models.py"
    )
