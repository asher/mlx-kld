"""Numerical core of the KLD scorer.

All KLD math runs in fp32 even when the disk cache is bf16. Summing ~250k
log-prob terms in bf16 loses ~0.5 nats of precision per position, which is more
than the actual signal between two competent quants. ``log1mexp`` near zero
also collapses to garbage in low precision.
"""

from __future__ import annotations

import math


def log1mexp_fp32(x):
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0``, in fp32.

    Two regimes (Machler 2012): ``log(-expm1(x))`` when x is close to 0
    (``|x| < log 2``), else ``log1p(-exp(x))``. Implemented via ``mx.where`` so
    it stays a fused MLX op.
    """
    import mlx.core as mx

    cutoff = -math.log(2.0)
    safe_close = mx.minimum(x, mx.array(-1e-30, dtype=mx.float32))
    safe_far = mx.minimum(x, mx.array(cutoff, dtype=mx.float32))
    a = mx.log(-mx.expm1(safe_close))
    b = mx.log1p(-mx.exp(safe_far))
    return mx.where(x > cutoff, a, b)


def kld_from_topk(
    teacher_topk_log_p,       # (B, T, K) fp32, log-softmax of teacher's top-K
    teacher_topk_indices,     # (B, T, K) int32, teacher's top-K vocab ids
    student_logits,           # (B, T, V) any float dtype; cast to fp32 internally
    vocab_size: int,
):
    """Per-token ``KLD(P||Q)`` where P is the top-K-reconstructed teacher and Q
    is the full student distribution.

    Reconstruction::

        log_p[i in topK] = teacher_topk_log_p
        log_p[i NOT in topK] = log((1 - sum_topK exp(log_p)) / (V - K))
                             = log1mexp(logsumexp(log_p_topK)) - log(V - K)

    Closed-form expansion of ``KL(P||Q)`` under this approximation, using the
    student's full log-softmax without materializing the ``(B,T,V)`` array.

    Returns ``(kld[B,T] fp32, student_top1[B,T] uint32, student_top5[B,T,5])``.
    ``student_top5`` is the student's top-5 index *set* (unordered); the
    scorer's "top-5 agreement" checks whether the teacher's argmax appears in
    it, which is a recall metric rather than a set overlap.
    """
    import mlx.core as mx

    K = teacher_topk_log_p.shape[-1]
    V = vocab_size
    tail_size = V - K
    # A real raise, not an assert: input validation must survive `python -O`,
    # and proceeding would take log() of a non-positive tail size.
    if tail_size <= 0:
        raise ValueError(f"vocab_size must exceed top-K (K={K}, V={V})")

    # Cast student logits to fp32 once; reuse for both KLD math and top-1/top-5.
    sl = student_logits.astype(mx.float32)
    lse = mx.logsumexp(sl, axis=-1)                         # (B, T)
    sum_logits = mx.sum(sl, axis=-1)                        # (B, T)

    # log_softmax(x) = x - lse, so sum over vocab equals sum(x) - V * lse.
    log_q_sum_full = sum_logits - float(V) * lse            # (B, T)
    # Gather student log_q at teacher's top-K positions (avoid full softmax).
    log_q_topk = (
        mx.take_along_axis(sl, teacher_topk_indices, axis=-1) - lse[..., None]
    )                                                       # (B, T, K)
    log_q_sum_topk = mx.sum(log_q_topk, axis=-1)            # (B, T)
    log_q_sum_tail = log_q_sum_full - log_q_sum_topk        # (B, T)

    log_p_topk = teacher_topk_log_p.astype(mx.float32)      # (B, T, K)
    log_p_topk_sum = mx.logsumexp(log_p_topk, axis=-1)      # (B, T)

    # Tail per-element log-prob (uniform residual). Clamp head sum strictly < 0
    # so log1mexp doesn't hit log(0) when teacher mass is fully captured.
    #
    # Do NOT "improve" this by caching the teacher's exact negentropy per
    # position (4 bytes/position) to remove the self-entropy approximation. The
    # self-entropy error and the cross-term error carry opposite signs and
    # partially cancel; removing only the first roughly doubles the measured
    # reconstruction floor. On a sharp distribution: self-entropy error
    # -0.000221, cross-term error +0.000444, net +0.000223 as written, versus
    # +0.000444 with "exact" entropy. The cancellation is fortuitous but real.
    log_p_topk_sum_safe = mx.minimum(
        log_p_topk_sum, mx.array(-1e-7, dtype=mx.float32)
    )
    log_p_tail = log1mexp_fp32(log_p_topk_sum_safe) - math.log(float(tail_size))  # (B, T)

    # Head term: sum over teacher's top-K of p_i * (log_p_i - log_q_i).
    head_term = mx.sum(
        mx.exp(log_p_topk) * (log_p_topk - log_q_topk), axis=-1
    )                                                       # (B, T)

    # Tail term: sum over (V-K) tail positions of p_tail * (log_p_tail - log_q_i)
    #          = exp(log_p_tail) * ((V-K) * log_p_tail - sum_{tail} log_q_i)
    tail_term = mx.exp(log_p_tail) * (
        float(tail_size) * log_p_tail - log_q_sum_tail
    )                                                       # (B, T)

    kld = head_term + tail_term

    # Top-1 / top-5 of the student. We only need the index sets; argpartition
    # is O(V) and avoids a full sort.
    student_top1 = mx.argmax(sl, axis=-1)                   # (B, T)
    student_top5 = mx.argpartition(sl, kth=-5, axis=-1)[..., -5:]  # (B, T, 5)

    return kld, student_top1, student_top5


def delta_p_from_topk(
    teacher_topk_log_p,       # (B, T, K) log-softmax of teacher's top-K
    teacher_topk_indices,     # (B, T, K) int32, teacher's top-K vocab ids
    student_logits,           # (B, T, V) any float dtype; cast to fp32 internally
    token_ids,                # (B, T) int32, the scored token stream
    vocab_size: int,
):
    """Per-position ``p_student(true next token) - p_teacher(true next token)``
    (llama.cpp's Delta-p family).

    Position ``t`` predicts token ``t+1``, so the result covers ``t`` in
    ``[0, T-1)`` and has shape ``(B, T-1)``, because the last position has no
    observed next token. Teacher ``p_true`` is read from the cached top-K row when the
    true token is in it, else from the uniform-tail reconstruction (the same
    approximation the KLD math uses); at the default K the tail case is
    negligible.
    """
    import mlx.core as mx

    K = teacher_topk_log_p.shape[-1]
    tail_size = vocab_size - K
    if tail_size <= 0:
        raise ValueError(f"vocab_size must exceed top-K (K={K}, V={vocab_size})")

    true_ids = token_ids[:, 1:].astype(mx.int32)                # (B, T-1)
    tk_log_p = teacher_topk_log_p[:, :-1, :].astype(mx.float32)  # (B, T-1, K)
    tk_idx = teacher_topk_indices[:, :-1, :]                    # (B, T-1, K)

    # One-hot (or all-zero) match of the true token inside the top-K row.
    match = tk_idx == true_ids[..., None]                       # (B, T-1, K)
    found = mx.any(match, axis=-1)                              # (B, T-1)
    head_log_p = mx.sum(
        mx.where(match, tk_log_p, mx.zeros_like(tk_log_p)), axis=-1
    )                                                           # (B, T-1)

    # Tail per-element log-prob, identical to the KLD reconstruction.
    head_sum = mx.logsumexp(tk_log_p, axis=-1)
    head_sum_safe = mx.minimum(head_sum, mx.array(-1e-7, dtype=mx.float32))
    log_p_tail = log1mexp_fp32(head_sum_safe) - math.log(float(tail_size))

    p_teacher = mx.where(found, mx.exp(head_log_p), mx.exp(log_p_tail))

    sl = student_logits[:, :-1, :].astype(mx.float32)           # (B, T-1, V)
    lse = mx.logsumexp(sl, axis=-1)                             # (B, T-1)
    log_q_true = (
        mx.take_along_axis(sl, true_ids[..., None], axis=-1)[..., 0] - lse
    )                                                           # (B, T-1)
    return mx.exp(log_q_true) - p_teacher


def clustered_se(values, cluster_ids):
    """Standard error of the mean, robust to clustering.

    Per-token KLDs inside one sequence share a document and a context, so they
    are not independent draws. At the default protocol a run scores 256
    positions in each of 512 sequences; ``std/sqrt(n)`` over those 131k tokens
    treats 512 sequences as 131k independent observations and understates the
    error by the design effect. Clustering by sequence is far closer to the
    truth, though still a little optimistic: sequences are fixed-length slices
    of one concatenated head-of-corpus stream, so several can come from a
    single document and share topical effects the estimator cannot see.

    Uses the standard cluster-robust (sandwich) estimator with the CR1
    finite-sample correction::

        Var(mean) = G/(G-1) * sum_g (sum_i (x_gi - xbar))^2 / N^2

    which handles unequal cluster sizes and collapses exactly to ``s/sqrt(n)``
    when every cluster holds one element. Returns ``None`` with fewer than two
    clusters.
    """
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return None
    _uniq, inv = np.unique(np.asarray(cluster_ids), return_inverse=True)
    n_clusters = int(inv.max()) + 1 if inv.size else 0
    if n_clusters < 2:
        return None
    resid = values - values.mean()
    per_cluster = np.bincount(inv, weights=resid, minlength=n_clusters)
    var = (per_cluster ** 2).sum() / (values.size ** 2)
    var *= n_clusters / (n_clusters - 1)
    return float(np.sqrt(var)) if var > 0 else 0.0


class Aggregator:
    """Streams per-token KLD into running stats + a per-position-bucket view.

    At the protocol sizes this tool targets (even 512x2048 is 1M tokens x
    4 bytes = 4 MB) we keep all per-token KLDs in memory and compute exact
    quantiles at the end. Avoids a t-digest dependency and lets the histogram
    be exact too.
    """

    def __init__(self):
        self._klds: list = []      # list of np.ndarray fp32 (B, T)
        self._top1: list = []      # list of np.ndarray bool (B, T)
        self._top5: list = []      # list of np.ndarray bool (B, T)
        self._positions: list = []  # list of np.ndarray int32 (B, T)
        self._sequences: list = []  # list of np.ndarray int32 (B, T)
        self._delta_p: list = []   # list of np.ndarray fp32 (masked, 1-D)
        self._delta_p_seq: list = []  # matching sequence ids
        self._floor: list = []     # list of np.ndarray fp32 (masked, 1-D)
        self._floor_missing = False
        self._sequences_missing = False

    def update(self, kld_bt, top1_match_bt, top5_match_bt, position_bt,
               sequence_bt=None):
        import numpy as np

        self._klds.append(np.asarray(kld_bt, dtype=np.float32))
        self._top1.append(np.asarray(top1_match_bt, dtype=bool))
        self._top5.append(np.asarray(top5_match_bt, dtype=bool))
        self._positions.append(np.asarray(position_bt, dtype=np.int32))
        # Sequence ids drive the clustered SE. Absent them the mean is still
        # exact; only the error bar falls back to the i.i.d. form.
        if sequence_bt is None:
            self._sequences_missing = True
        else:
            self._sequences.append(np.asarray(sequence_bt, dtype=np.int32))

    def update_delta_p(self, delta_p_values, sequence_ids=None):
        """Accumulate already-masked Delta-p values. Tracked separately from
        ``update`` because the last window position has a KLD but no observed
        next token, so the two masks differ."""
        import numpy as np

        self._delta_p.append(np.asarray(delta_p_values, dtype=np.float32))
        if sequence_ids is not None:
            self._delta_p_seq.append(np.asarray(sequence_ids, dtype=np.int32))

    def update_floor(self, floor_values):
        """Accumulate already-masked per-position reconstruction-floor KLDs
        (teacher scored against its own top-K reconstruction). Pass ``None``
        when the cache shard predates floor tracking; one missing shard makes
        the whole run's floor unavailable rather than biased."""
        import numpy as np

        if floor_values is None:
            self._floor_missing = True
            return
        self._floor.append(np.asarray(floor_values, dtype=np.float32))

    @staticmethod
    def _empty_result() -> dict:
        """The result shape for a run that scored nothing.

        Every key the populated result carries is present with a null value.
        Zero scored tokens is the signature of a catastrophically broken student
        (every position non-finite), which is exactly the case the report's
        plausibility guard exists to flag. The caller has to reach that
        guard, not die on a KeyError two frames earlier.
        """
        return {
            "tokens_scored": 0,
            "tokens_dropped_nonfinite": 0,
            "kld": {
                "mean": None, "se": None, "se_method": None, "floor_mean": None,
                "p50": None, "p95": None, "p99": None, "p999": None, "max": None,
            },
            "delta_p": {"mean": None, "se": None, "rms": None},
            "agreement": {"top1": None, "top5": None},
            "by_position": [],
            "kld_histogram": None,
        }

    def finalize(
        self,
        position_buckets: list[tuple[int, int | None]] | None = None,
    ) -> dict:
        import numpy as np

        from ._constants import POSITION_BUCKETS

        if not self._klds:
            return self._empty_result()
        kld = np.concatenate([a.ravel() for a in self._klds])
        top1 = np.concatenate([a.ravel() for a in self._top1])
        top5 = np.concatenate([a.ravel() for a in self._top5])
        pos = np.concatenate([a.ravel() for a in self._positions])
        have_seq = bool(self._sequences) and not self._sequences_missing
        seq = np.concatenate([a.ravel() for a in self._sequences]) if have_seq else None

        # The math is fully log-space, so a merely tiny student probability
        # still yields a finite KLD. A non-finite one means the student's
        # logits were themselves non-finite, or so extreme that the fp32 tail
        # accumulation overflowed, both worse failure classes. Drop those
        # tokens so one cannot poison the whole mean, but count them: dropping
        # the worst positions *improves* the mean, so the count travels to the
        # record as tokens_dropped_nonfinite.
        finite = np.isfinite(kld)
        n_bad = int((~finite).sum())
        if n_bad:
            from ._log import warn
            warn(f"dropping {n_bad} scored token(s) with non-finite KLD "
                 "(non-finite or overflowing student logits there), "
                 "recorded as tokens_dropped_nonfinite")
            kld, top1, top5, pos = kld[finite], top1[finite], top5[finite], pos[finite]
            if seq is not None:
                seq = seq[finite]
        if kld.size == 0:
            result = self._empty_result()
            result["tokens_dropped_nonfinite"] = n_bad
            return result

        # Defensive: clip any negative KLD from fp32 noise on near-identical
        # distributions. KLD is mathematically >= 0.
        kld = np.maximum(kld, 0.0)

        # Standard error of the mean, clustered by sequence: the protocol draws
        # 512 independent sequences, not 131k independent tokens.
        if seq is not None:
            kld_se = clustered_se(kld, seq)
            se_method = "clustered-by-sequence"
        else:
            kld_se = float(kld.std(ddof=1) / np.sqrt(kld.size)) if kld.size > 1 else None
            se_method = "iid"

        floor_mean = None
        if self._floor and not self._floor_missing:
            fl = np.concatenate([a.ravel() for a in self._floor])
            fl = np.maximum(fl[np.isfinite(fl)], 0.0)
            if fl.size:
                floor_mean = float(fl.mean())

        global_stats = {
            "mean": float(kld.mean()),
            "se": kld_se,
            # Self-describing, so a reader can tell a clustered error bar from
            # the i.i.d. one older records carry.
            "se_method": se_method,
            "floor_mean": floor_mean,
            "p50": float(np.quantile(kld, 0.5)),
            "p95": float(np.quantile(kld, 0.95)),
            "p99": float(np.quantile(kld, 0.99)),
            "p999": float(np.quantile(kld, 0.999)),
            "max": float(kld.max()),
        }
        agreement = {
            "top1": float(top1.mean()),
            "top5": float(top5.mean()),
        }

        dp_seq = None
        if self._delta_p:
            dp = np.concatenate([a.ravel() for a in self._delta_p])
            if len(self._delta_p_seq) == len(self._delta_p):
                dp_seq = np.concatenate([a.ravel() for a in self._delta_p_seq])
            dp_finite = np.isfinite(dp)
            dp = dp[dp_finite]
            if dp_seq is not None:
                dp_seq = dp_seq[dp_finite]
        else:
            dp = np.empty(0, dtype=np.float32)
        if dp.size:
            if dp_seq is not None:
                dp_se = clustered_se(dp, dp_seq)
            else:
                dp_se = float(dp.std(ddof=1) / np.sqrt(dp.size)) if dp.size > 1 else None
            delta_p = {
                "mean": float(dp.mean()),
                "se": dp_se,
                "rms": float(np.sqrt(np.mean(np.square(dp, dtype=np.float64)))),
            }
        else:
            delta_p = {"mean": None, "se": None, "rms": None}

        buckets = position_buckets or POSITION_BUCKETS
        by_position = []
        for start, end in buckets:
            if end is None:
                mask = pos >= start
                label = f"{start}-end"
            else:
                mask = (pos >= start) & (pos < end)
                label = f"{start}-{end}"
            n = int(mask.sum())
            if n == 0:
                by_position.append({"range": label, "tokens": 0})
                continue
            sub = kld[mask]
            by_position.append({
                "range": label,
                "tokens": n,
                "mean_kld": float(sub.mean()),
                "p95_kld": float(np.quantile(sub, 0.95)),
                "top1_agreement": float(top1[mask].mean()),
            })

        # Histogram (fixed log-spaced bins; exact counts). The final edge is
        # +inf, not 1e1: np.histogram drops values outside its edge range, so a
        # finite top edge silently discarded every token above 10 nats, which
        # is exactly the population a broken student produces, and exactly what
        # the histogram exists to show. counts now always sums to tokens_scored.
        bins = np.concatenate([
            np.array([0.0]),
            np.logspace(-6, 1, 29),  # 1e-6 .. 1e1: 29 edges
            np.array([np.inf]),
        ])                           # 31 edges -> 30 bins
        counts, edges = np.histogram(kld, bins=bins)
        histogram = {
            "bin_edges": [float(x) for x in edges],
            "counts": [int(c) for c in counts],
        }

        return {
            "tokens_scored": int(kld.size),
            "tokens_dropped_nonfinite": n_bad,
            "kld": global_stats,
            "delta_p": delta_p,
            "agreement": agreement,
            "by_position": by_position,
            "kld_histogram": histogram,
        }
