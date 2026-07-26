"""Student forward pass + online KLD against the cached teacher top-K."""

from __future__ import annotations

import sys
from pathlib import Path

from ._log import info, warn
from .errors import TokenizerMismatchError
from .kld_math import Aggregator, delta_p_from_topk, kld_from_topk
from .models import _extract_logits, _load_model, _make_fresh_cache

# Padded-id probability mass above which the slice-and-renormalize is worth
# interrupting the user for. Well-formed padded lm_heads train those ids to
# near-zero, so anything at 1e-4 is a signal, not rounding.
_PAD_MASS_WARN = 1e-4


def _align_student_vocab(student_logits, expected: int, config_vocab: int | None = None):
    """Reconcile the student lm_head width with the teacher cache's vocab.

    Wider (padded lm_head): slice to the cache vocab, renormalizing Q over it,
    and return the max probability mass the student put on the padded ids so
    the caller can report the worst case over the whole run. A narrower lm_head
    raises instead. Either the vocabularies genuinely differ, or the *teacher's*
    lm_head is
    padded and the student ships the unpadded width. ``config_vocab`` (the
    manifest's ``config_vocab_size``) tells those two apart, so the error can
    name the one that happened instead of asking the user to go read JSON.

    Returns ``(logits_sliced, pad_mass_or_None)``.
    """
    import mlx.core as mx

    sv = int(student_logits.shape[-1])
    if sv == expected:
        return student_logits, None
    if sv < expected:
        if config_vocab is not None and sv == config_vocab:
            raise TokenizerMismatchError(
                f"student logits vocab ({sv}) is narrower than the teacher "
                f"cache's ({expected}), but matches the teacher's configured "
                f"vocab_size ({config_vocab}) exactly: the teacher's lm_head "
                "is padded and this student ships the unpadded width. Rescore "
                "with --rebuild-cache against a teacher build whose lm_head "
                "is not padded, or score a student built to the same width."
            )
        raise TokenizerMismatchError(
            f"student logits vocab ({sv}) is narrower than the teacher "
            f"cache's ({expected}). Teacher and student use different "
            "vocabularies, so the comparison is invalid."
        )
    # Cheap next to the forward pass (two logsumexps), so measure every batch:
    # a student can put pathological mass on padded ids only in later batches.
    lf = student_logits.astype(mx.float32)
    lse_full = mx.logsumexp(lf, axis=-1)
    lse_head = mx.logsumexp(lf[..., :expected], axis=-1)
    pad_mass = float(mx.max(1.0 - mx.exp(lse_head - lse_full)).item())
    return student_logits[..., :expected], pad_mass


def score_loaded_student(
    student_model,
    cache_dir: Path,
    manifest: dict,
) -> dict:
    """Forward an already-loaded student against the cached teacher top-K and
    return the KLD aggregate.

    Public entry point used by ``student_pass`` (which loads from disk first) and
    by external tools that hold the model in memory and swap individual tensors
    between iterations. The student must be in eval mode and produce logits with
    vocab matching ``manifest["vocab_size"]``.
    """
    import mlx.core as mx
    import numpy as np
    from tqdm import tqdm

    student_vocab = manifest["vocab_size"]
    num_batches = manifest["num_batches"]
    seq_len = manifest["max_seq_len"]
    # The manifest carries score_window = [start, end). Caches missing the
    # optional field are treated as full-sequence.
    window = manifest.get("score_window")
    if window is None:
        window = [0, seq_len]
    win_start, win_end = int(window[0]), int(window[1])

    agg = Aggregator()
    info(f"student pass: {num_batches} batches, "
         f"score window=[{win_start}, {win_end})")
    # Running count of sequences seen, so every scored token carries the id of
    # the calibration sequence it came from. Shards are fixed-size and read in
    # order, but deriving the id from the running count rather than
    # i * batch_size keeps it correct for a short final shard too.
    seq_base = 0
    max_pad_mass: float | None = None
    pad_warned = False
    for i in tqdm(range(num_batches), desc="student", file=sys.stderr):
        shard = mx.load(str(cache_dir / f"batch-{i:05d}.safetensors"))
        token_ids = shard["token_ids"]                      # (B, L) int32
        top_log_p = shard["top_k_log_softmax"]              # (B, L, K) bf16
        top_idx = shard["top_k_indices"]                    # (B, L, K) int32
        attn_mask = shard["attention_mask"]                 # (B, L) bool

        cache = _make_fresh_cache(student_model)
        student_logits = _extract_logits(student_model(token_ids, cache=cache))  # (B, L, V)
        raw_width = int(student_logits.shape[-1])
        student_logits, pad_mass = _align_student_vocab(
            student_logits, student_vocab, manifest.get("config_vocab_size"),
        )
        if pad_mass is not None:
            if max_pad_mass is None:
                info(f"student lm_head is wider than the cache vocab "
                     f"({raw_width} > {student_vocab}). Scoring the first "
                     f"{student_vocab} ids and tracking padded-id mass across "
                     "the run")
            # Not `max`: `max(0.0, nan)` is 0.0, and a NaN here means the
            # student emitted non-finite logits, the case most worth
            # surfacing. `not (pad_mass <= max_pad_mass)` admits both a larger
            # value and a NaN, and the self-equality guard makes a NaN stick.
            if max_pad_mass is None or (
                max_pad_mass == max_pad_mass and not (pad_mass <= max_pad_mass)
            ):
                max_pad_mass = pad_mass
            # Warn on the batch that crosses the threshold, not after the whole
            # pass: on a large student that is hours of difference, and the
            # user's next move (abort and rescore) is the same either way.
            if not pad_warned and not (pad_mass <= _PAD_MASS_WARN):
                pad_warned = True
                warn(f"batch {i}: the student puts {pad_mass:.2e} of its "
                     f"probability mass on ids past the teacher vocab "
                     f"({student_vocab}). The sliced distribution is "
                     "renormalized over the teacher vocab, so that mass is "
                     "redistributed rather than scored")

        kld_bt, student_top1, student_top5 = kld_from_topk(
            top_log_p, top_idx, student_logits, student_vocab,
        )
        teacher_top1 = top_idx[..., 0]                      # (B, L)
        top1_match = (student_top1 == teacher_top1)
        # Top-5 recall: teacher's argmax appears in student's top-5 set.
        top5_match = mx.any(student_top5 == teacher_top1[..., None], axis=-1)

        delta_p_bt = delta_p_from_topk(
            top_log_p, top_idx, student_logits, token_ids, student_vocab,
        )                                                   # (B, L-1)

        # Mask invalid positions; intersect with the score window.
        attn = attn_mask.astype(mx.bool_)
        m = attn
        L = m.shape[1]
        if win_start > 0 or win_end < L:
            pos_idx = mx.arange(L, dtype=mx.int32)
            window_mask = (pos_idx >= win_start) & (pos_idx < win_end)
            m = m & window_mask[None, :]
        # Delta-p at position t needs an observed token at t+1, so drop the last
        # *sequence* position and require the next token to be valid. Positions
        # inside the window keep their Delta-p; only position L-1 has no
        # successor to compare against.
        m_dp = m[:, :-1] & attn[:, 1:]
        mx.eval(kld_bt, top1_match, top5_match, delta_p_bt, m, m_dp)

        m_np = np.asarray(m)
        positions = np.broadcast_to(
            np.arange(m_np.shape[1], dtype=np.int32), m_np.shape
        ).copy()
        seq_ids = np.broadcast_to(
            (seq_base + np.arange(m_np.shape[0], dtype=np.int32))[:, None],
            m_np.shape,
        ).copy()
        agg.update(
            np.asarray(kld_bt)[m_np],
            np.asarray(top1_match)[m_np],
            np.asarray(top5_match)[m_np],
            positions[m_np],
            seq_ids[m_np],
        )
        m_dp_np = np.asarray(m_dp)
        agg.update_delta_p(
            np.asarray(delta_p_bt)[m_dp_np], seq_ids[:, :-1][m_dp_np],
        )
        seq_base += m_np.shape[0]
        floor_bt = shard.get("floor_kld")  # absent in pre-floor caches
        agg.update_floor(
            np.asarray(floor_bt)[m_np] if floor_bt is not None else None
        )

    if max_pad_mass is not None:
        # `not (x <= t)` rather than `x > t` so a NaN reports as a warning.
        note = warn if not (max_pad_mass <= _PAD_MASS_WARN) else info
        note(f"max probability mass on padded lm_head ids across the run: "
             f"{max_pad_mass:.2e}")

    # Quartiles of the actual scored window (same logic for default and
    # --long-context modes).
    span = max(win_end - win_start, 4)
    q = span // 4
    buckets: list[tuple[int, int | None]] = [
        (win_start, win_start + q),
        (win_start + q, win_start + 2 * q),
        (win_start + 2 * q, win_start + 3 * q),
        (win_start + 3 * q, None),
    ]
    return agg.finalize(position_buckets=buckets)


def student_pass(
    student_dir: Path,
    cache_dir: Path,
    manifest: dict,
    batch_size: int,
) -> dict:
    """Forward the student loaded from disk, compute online KLD against the
    cached teacher logits. Thin wrapper around ``score_loaded_student``."""
    info(f"loading student: {student_dir}")
    student_model, _student_config = _load_model(str(student_dir), lazy=True)
    student_model.eval()

    cache_batch_size = manifest["batch_size"]
    if cache_batch_size != batch_size:
        info(f"Note: cache was written at batch_size={cache_batch_size}. "
             "The student pass replays at that batch size for shard alignment.")
    return score_loaded_student(student_model, cache_dir, manifest)
