"""Student-side vocab alignment (padded lm_head handling)."""

import mlx.core as mx
import pytest

from mlx_kld.errors import TokenizerMismatchError
from mlx_kld.scoring import _align_student_vocab


def test_matching_vocab_passthrough():
    x = mx.zeros((1, 2, 8))
    out, pad_mass = _align_student_vocab(x, 8)
    assert out is x
    assert pad_mass is None


def test_padded_lm_head_sliced_and_measured():
    out, pad_mass = _align_student_vocab(mx.zeros((1, 2, 12)), 8)
    assert out.shape == (1, 2, 8)
    # Uniform logits over 12 ids: 4/12 of the mass sits on the padded ids.
    assert pad_mass == pytest.approx(4 / 12, rel=1e-5)


def test_padded_lm_head_no_pad_mass():
    # All mass on an unpadded id: pad_mass ~ 0.
    logits = mx.zeros((1, 1, 12))
    logits = logits.at[..., 0].add(100.0)
    out, pad_mass = _align_student_vocab(logits, 8)
    assert out.shape == (1, 1, 8)
    assert pad_mass == pytest.approx(0.0, abs=1e-6)


def test_narrower_vocab_raises():
    with pytest.raises(TokenizerMismatchError, match="narrower"):
        _align_student_vocab(mx.zeros((1, 2, 4)), 8)
