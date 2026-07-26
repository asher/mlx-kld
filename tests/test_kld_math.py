"""Unit tests for the pure KLD math (no model load)."""

import math

import mlx.core as mx
import numpy as np
import pytest

from mlx_kld.kld_math import (
    Aggregator,
    clustered_se,
    delta_p_from_topk,
    kld_from_topk,
)


def _topk(log_p, K):
    idx = mx.argpartition(log_p, kth=-K, axis=-1)[..., -K:]
    v = mx.take_along_axis(log_p, idx, axis=-1)
    order = mx.argsort(-v, axis=-1)
    return (
        mx.take_along_axis(idx, order, axis=-1).astype(mx.int32),
        mx.take_along_axis(v, order, axis=-1),
    )


def test_uniform_vs_uniform_is_zero():
    V, K = 1024, 16
    teacher_log = mx.full((1, 1, V), -math.log(V), dtype=mx.float32)
    idx = mx.arange(K, dtype=mx.int32).reshape(1, 1, K)
    topk = mx.take_along_axis(teacher_log, idx, axis=-1)
    student = mx.zeros((1, 1, V), dtype=mx.float32)  # uniform
    kld, _, _ = kld_from_topk(topk, idx, student, V)
    assert abs(float(np.asarray(kld).item())) < 1e-4


def test_peaked_vs_uniform_is_large():
    V, K = 1024, 16
    raw = np.full((V,), -10.0, dtype=np.float32)
    raw[0] = 5.0
    teacher = mx.array(raw.reshape(1, 1, V))
    teacher_log = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    idx, topk = _topk(teacher_log, K)
    student = mx.zeros((1, 1, V), dtype=mx.float32)
    kld, _, _ = kld_from_topk(topk, idx, student, V)
    assert float(np.asarray(kld).item()) > 1.0


def test_kld_is_asymmetric():
    V, K = 1024, 16
    np_p = np.full((V,), -10.0, dtype=np.float32)
    np_p[0] = 5.0
    np_q = np.full((V,), -8.0, dtype=np.float32)
    np_q[1] = 4.0
    p_logits = mx.array(np_p.reshape(1, 1, V))
    q_logits = mx.array(np_q.reshape(1, 1, V))
    p_log = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    q_log = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)
    p_idx, p_topk = _topk(p_log, K)
    q_idx, q_topk = _topk(q_log, K)
    a = float(np.asarray(kld_from_topk(p_topk, p_idx, q_logits, V)[0]).item())
    b = float(np.asarray(kld_from_topk(q_topk, q_idx, p_logits, V)[0]).item())
    assert abs(a - b) > 1e-3


# ---------- Delta-p ----------

def _log_softmax_np(raw):
    return raw - np.log(np.exp(raw).sum())


def test_delta_p_true_token_in_topk_is_exact():
    V, K = 1024, 16
    rng = np.random.default_rng(1)
    t_raw = rng.normal(size=V).astype(np.float32) * 2.0
    s_raw = rng.normal(size=V).astype(np.float32) * 2.0
    t_log = _log_softmax_np(t_raw)
    s_log = _log_softmax_np(s_raw)
    true_id = int(np.argmax(t_log))  # guaranteed in the top-K

    teacher = mx.array(np.stack([t_log, t_log]).reshape(1, 2, V))
    student = mx.array(np.stack([s_raw, s_raw]).reshape(1, 2, V))
    idx, topk = _topk(teacher, K)
    token_ids = mx.array(np.array([[0, true_id]], dtype=np.int32))  # (B=1, T=2)

    dp = np.asarray(delta_p_from_topk(topk, idx, student, token_ids, V))
    assert dp.shape == (1, 1)  # T-1 positions
    expected = math.exp(s_log[true_id]) - math.exp(t_log[true_id])
    assert abs(float(dp[0, 0]) - expected) < 1e-6


def test_delta_p_true_token_in_tail_uses_uniform_reconstruction():
    V, K = 1024, 16
    raw = np.full((V,), -10.0, dtype=np.float32)
    raw[:K] = np.linspace(5.0, 1.0, K).astype(np.float32)
    t_log = _log_softmax_np(raw)
    true_id = V - 1  # far outside the top-K

    teacher = mx.array(np.stack([t_log, t_log]).reshape(1, 2, V))
    student = mx.zeros((1, 2, V), dtype=mx.float32)  # uniform student
    idx, topk = _topk(teacher, K)
    token_ids = mx.array(np.array([[0, true_id]], dtype=np.int32))

    dp = np.asarray(delta_p_from_topk(topk, idx, student, token_ids, V))
    head_mass = np.exp(t_log[np.argsort(-t_log)[:K]]).sum()
    p_tail = (1.0 - head_mass) / (V - K)
    expected = 1.0 / V - p_tail
    assert abs(float(dp[0, 0]) - expected) < 1e-7


def test_delta_p_identical_models_is_zero():
    V, K = 1024, 32
    rng = np.random.default_rng(2)
    raw = rng.normal(size=V).astype(np.float32)
    log = _log_softmax_np(raw)
    true_id = int(np.argsort(-log)[3])  # a top-K token

    teacher = mx.array(np.stack([log, log]).reshape(1, 2, V))
    student = mx.array(np.stack([raw, raw]).reshape(1, 2, V))
    idx, topk = _topk(teacher, K)
    token_ids = mx.array(np.array([[0, true_id]], dtype=np.int32))
    dp = np.asarray(delta_p_from_topk(topk, idx, student, token_ids, V))
    assert abs(float(dp[0, 0])) < 1e-6


# ---------- Aggregator: SE + Delta-p stats ----------

def test_clustered_se_collapses_to_iid_for_singleton_clusters():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    iid = x.std(ddof=1) / np.sqrt(x.size)
    assert clustered_se(x, np.arange(500)) == pytest.approx(iid, rel=1e-9)


def test_clustered_se_recovers_the_truth_under_within_sequence_correlation():
    """The whole point: 512 documents are 512 independent draws, not 262k. An
    i.i.d. SE understates the error by the design effect."""
    rng = np.random.default_rng(1)
    G, n = 256, 256
    doc = rng.normal(0, 1.0, G)
    tok = rng.normal(0, 1.0, (G, n))
    vals = (doc[:, None] + tok).ravel()
    ids = np.repeat(np.arange(G), n)
    truth = (doc + tok.mean(axis=1)).std(ddof=1) / np.sqrt(G)
    clustered = clustered_se(vals, ids)
    iid = vals.std(ddof=1) / np.sqrt(vals.size)
    assert clustered == pytest.approx(truth, rel=1e-3)
    assert clustered > 5 * iid          # design effect is large and real


def test_clustered_se_needs_at_least_two_clusters():
    assert clustered_se(np.array([1.0, 2.0, 3.0]), np.zeros(3, dtype=int)) is None
    assert clustered_se(np.array([1.0]), np.array([0])) is None


def test_aggregator_reports_clustered_se_and_labels_the_method():
    agg = Aggregator()
    rng = np.random.default_rng(2)
    for s in range(8):
        n = 16
        agg.update(rng.normal(1.0, 0.1, n), np.ones(n, bool), np.ones(n, bool),
                   np.arange(n, dtype=np.int32), np.full(n, s, dtype=np.int32))
    out = agg.finalize([(0, None)])
    assert out["kld"]["se_method"] == "clustered-by-sequence"
    assert out["kld"]["se"] > 0


def test_aggregator_falls_back_to_iid_without_sequence_ids():
    """External callers predating the sequence argument still get an exact mean;
    only the error bar degrades, and the record says so."""
    agg = Aggregator()
    n = 32
    agg.update(np.full(n, 0.5), np.ones(n, bool), np.ones(n, bool),
               np.arange(n, dtype=np.int32))
    out = agg.finalize([(0, None)])
    assert out["kld"]["se_method"] == "iid"
    assert out["kld"]["mean"] == pytest.approx(0.5)


def test_aggregator_se_and_delta_p_closed_form():
    agg = Aggregator()
    klds = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    flags = np.array([True, False, True, True])
    pos = np.array([300, 301, 302, 303], dtype=np.int32)
    agg.update(klds, flags, flags, pos)
    dps = np.array([0.01, -0.03, 0.02], dtype=np.float32)
    agg.update_delta_p(dps)
    out = agg.finalize()

    assert abs(out["kld"]["mean"] - klds.mean()) < 1e-7
    assert abs(out["kld"]["se"] - klds.std(ddof=1) / np.sqrt(4)) < 1e-7
    assert abs(out["delta_p"]["mean"] - dps.mean()) < 1e-7
    assert abs(out["delta_p"]["se"] - dps.std(ddof=1) / np.sqrt(3)) < 1e-7
    assert abs(out["delta_p"]["rms"] - np.sqrt((dps.astype(np.float64) ** 2).mean())) < 1e-9


def test_aggregator_drops_non_finite_kld():
    # A single inf per-token KLD (student gave ~0 prob to a token) must not
    # poison the mean; the token is dropped and the rest survive.
    agg = Aggregator()
    klds = np.array([0.1, np.inf, 0.3], dtype=np.float32)
    flags = np.array([True, True, True])
    pos = np.array([300, 301, 302], dtype=np.int32)
    agg.update(klds, flags, flags, pos)
    out = agg.finalize()
    assert out["tokens_scored"] == 2
    assert abs(out["kld"]["mean"] - 0.2) < 1e-6
    assert np.isfinite(out["kld"]["max"])


def test_aggregator_all_non_finite_kld_scores_zero_tokens():
    agg = Aggregator()
    agg.update(
        np.array([np.inf, np.nan], dtype=np.float32),
        np.array([True, True]),
        np.array([True, True]),
        np.array([300, 301], dtype=np.int32),
    )
    out = agg.finalize()
    assert out["tokens_scored"] == 0
    # The full result shape, all-null. A caller that reaches for out["kld"]
    # ["mean"] here is about to report a catastrophically broken student, and
    # must get to the plausibility guard rather than a KeyError.
    assert set(out) == {"tokens_scored", "tokens_dropped_nonfinite", "kld",
                        "delta_p", "agreement", "by_position", "kld_histogram"}
    assert out["tokens_dropped_nonfinite"] == 2
    assert out["kld"]["mean"] is None
    assert out["agreement"] == {"top1": None, "top5": None}


def test_zero_scored_tokens_reaches_the_plausibility_guard():
    """The empty result has to survive the whole report path, not just exist."""
    from mlx_kld.report import implausibility_reason

    out = Aggregator().finalize()
    reason = implausibility_reason(
        out["agreement"], out["kld"], out["tokens_scored"],
    )
    assert reason and "no token was scorable" in reason


def test_aggregator_floor_mean():
    agg = Aggregator()
    agg.update(
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([True, True]),
        np.array([True, True]),
        np.array([300, 301], dtype=np.int32),
    )
    agg.update_floor(np.array([0.001, 0.003], dtype=np.float32))
    out = agg.finalize()
    assert abs(out["kld"]["floor_mean"] - 0.002) < 1e-7


def test_aggregator_floor_none_when_any_shard_missing():
    # One pre-floor shard makes the whole run's floor unavailable, not biased.
    agg = Aggregator()
    agg.update(
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([True, True]),
        np.array([True, True]),
        np.array([300, 301], dtype=np.int32),
    )
    agg.update_floor(np.array([0.001], dtype=np.float32))
    agg.update_floor(None)
    out = agg.finalize()
    assert out["kld"]["floor_mean"] is None


def test_aggregator_without_floor_reports_none():
    agg = Aggregator()
    agg.update(
        np.array([0.1], dtype=np.float32),
        np.array([True]),
        np.array([True]),
        np.array([300], dtype=np.int32),
    )
    assert agg.finalize()["kld"]["floor_mean"] is None


def test_aggregator_without_delta_p_reports_none():
    agg = Aggregator()
    agg.update(
        np.array([0.1, 0.2], dtype=np.float32),
        np.array([True, True]),
        np.array([True, True]),
        np.array([300, 301], dtype=np.int32),
    )
    out = agg.finalize()
    assert out["delta_p"] == {"mean": None, "se": None, "rms": None}


def test_flat_tail_reconstruction_is_exact():
    V = 1024
    raw = np.full((V,), -10.0, dtype=np.float32)
    raw[:8] = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)
    teacher = mx.array(raw.reshape(1, 1, V))
    teacher_log = teacher - mx.logsumexp(teacher, axis=-1, keepdims=True)
    student = mx.zeros((1, 1, V), dtype=mx.float32)
    student_log = student - mx.logsumexp(student, axis=-1, keepdims=True)
    exact = float(np.asarray(
        mx.sum(mx.exp(teacher_log) * (teacher_log - student_log), axis=-1)
    ).item())
    for K in (16, 64, 128):
        idx, topk = _topk(teacher_log, K)
        approx = float(np.asarray(kld_from_topk(topk, idx, student, V)[0]).item())
        assert abs(approx - exact) / max(abs(exact), 1e-9) < 5e-3


def test_histogram_counts_every_scored_token_including_broken_students():
    """A finite top edge silently dropped everything above 10 nats, exactly
    the population a broken load produces, and what the histogram is for."""
    agg = Aggregator()
    vals = np.array([0.0, 1e-8, 0.5, 9.9, 10.5, 42.0, 1e6], dtype=np.float32)
    n = vals.size
    agg.update(vals, np.ones(n, bool), np.ones(n, bool),
               np.arange(n, dtype=np.int32), np.zeros(n, dtype=np.int32))
    out = agg.finalize([(0, None)])
    counts = out["kld_histogram"]["counts"]
    assert sum(counts) == out["tokens_scored"] == n
    assert counts[-1] == 3          # 10.5, 42.0, 1e6 land in the overflow bin
    edges = out["kld_histogram"]["bin_edges"]
    assert len(edges) == len(counts) + 1
    assert math.isinf(edges[-1])
