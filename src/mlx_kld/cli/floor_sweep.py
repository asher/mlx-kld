"""``mlx-kld floor-sweep``: measure the top-K reconstruction floor on a model.

The top-K-with-uniform-tail reconstruction in ``kld_from_topk`` is exact when
the teacher's tail is uniform; on real models it has a residual that depends on
tail concentration. Running KLD with student==teacher (same forward, same
logits) at multiple K values from a single forward pass per batch measures that
residual: a student identical to its teacher would score 0, so whatever it does
score at each K *is* the measurement floor.

The floor is U-shaped in K, not monotone. A larger K shrinks the uniform-tail
error but adds bf16 rounding on more stored slots, and the two swap dominance
around the default ``--top-k``. That is why the sweep spans both sides of it.
"""

from __future__ import annotations

import argparse
import sys

from .._constants import DEFAULT_DATASET, DEFAULT_MAX_SEQ_LEN, DEFAULT_NUM_SAMPLES, DEFAULT_SEED
from .._log import info


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "floor-sweep",
        help="measure the top-K reconstruction floor on a model",
        description="Run teacher==student KLD at multiple K values from a single "
        "forward pass per batch and print a floor(K) table. Use it to pick a "
        "defensible --top-k.",
    )
    p.add_argument("model", help="HF id or local path to the model to sweep")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help="HF dataset (default: %(default)s)")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help="Number of sequences (default: %(default)s)")
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                   help="Tokens per sequence (default: %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Seed for the calibration-chunk permutation (default: "
                        "%(default)s). Ingest reads the head of the "
                        "corpus (~525k tokens at the default "
                        "protocol), so this permutes within that "
                        "window rather than sampling the corpus as a "
                        "whole.")
    p.add_argument("--stream-dataset", action="store_true",
                   help="Pull dataset rows over the network instead of "
                        "downloading the whole split first (see score)")
    p.add_argument("--k-values", default="128,512,2048,8192,32768,65536,131072",
                   metavar="K1,K2,...",
                   help="Comma-separated K values (default: %(default)s). "
                        "Values >= vocab size are dropped. The range spans "
                        "both sides of the default --top-k, because the floor "
                        "is U-shaped in K rather than monotone")
    p.add_argument("--score-window", default=None, metavar="START:END",
                   help="Window to measure over, as `start:end` (default: "
                        "second half, matching score)")
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    import mlx.core as mx
    import numpy as np
    from mlx_lm.utils import load_tokenizer
    from tqdm import tqdm

    from ..kld_math import kld_from_topk
    from ..models import _extract_logits, _load_model, _make_fresh_cache
    from ..tokenizer import load_calibration_tokens

    try:
        k_values = sorted({int(x) for x in args.k_values.split(",") if x.strip()})
    except ValueError:
        sys.exit(f"error: --k-values must be comma-separated ints, got: {args.k_values!r}")
    if not k_values:
        sys.exit("error: --k-values yielded no values")

    if args.score_window is not None:
        try:
            ws, we = (int(x) for x in args.score_window.split(":"))
        except ValueError:
            sys.exit(f"error: --score-window must be `start:end`, got: {args.score_window!r}")
    else:
        ws, we = args.max_seq_len // 2, args.max_seq_len
    if not (0 <= ws < we <= args.max_seq_len):
        sys.exit(
            f"error: --score-window ({ws}, {we}) must satisfy "
            f"0 <= start < end <= max_seq_len ({args.max_seq_len})"
        )

    info(f"K-floor sweep on {args.model}")
    info(f"  dataset={args.dataset}, samples={args.num_samples}, "
         f"seq={args.max_seq_len}, seed={args.seed}, window=[{ws}, {we})")

    tokenizer = load_tokenizer(args.model)
    tokens = load_calibration_tokens(
        tokenizer, args.dataset, args.num_samples, args.max_seq_len, args.seed,
        streaming=args.stream_dataset,
    )
    info(f"  loaded {tokens.shape[0]} sequences x {tokens.shape[1]} tokens")

    info("  loading model")
    model, _config = _load_model(args.model, lazy=False)
    model.eval()

    floor_sum: dict[int, float] = {K: 0.0 for K in k_values}
    floor_max: dict[int, float] = {K: 0.0 for K in k_values}
    floor_count = 0
    V: int | None = None

    for b in tqdm(range(tokens.shape[0]), desc="floor sweep", file=sys.stderr):
        batch = tokens[b:b + 1]
        cache = _make_fresh_cache(model)
        logits = _extract_logits(model(batch, cache=cache))
        logits_fp32 = logits.astype(mx.float32)

        if V is None:
            V = int(logits_fp32.shape[-1])
            k_values = [K for K in k_values if 0 < K < V]
            info(f"  V={V:,}, sweep K values = {k_values}")

        log_softmax = logits_fp32 - mx.logsumexp(logits_fp32, axis=-1, keepdims=True)

        for K in k_values:
            idx_part = mx.argpartition(log_softmax, kth=-K, axis=-1)[..., -K:]
            vals = mx.take_along_axis(log_softmax, idx_part, axis=-1)
            order = mx.argsort(-vals, axis=-1)
            idx_sorted = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
            # bf16, matching what score's cache stores and replays.
            topk_sorted = mx.take_along_axis(vals, order, axis=-1).astype(mx.bfloat16)

            kld, _t1, _t5 = kld_from_topk(topk_sorted, idx_sorted, logits_fp32, V)
            kld_np = np.maximum(np.asarray(kld[:, ws:we]).ravel(), 0.0)
            floor_sum[K] += float(kld_np.sum())
            floor_max[K] = max(floor_max[K], float(kld_np.max()))

        floor_count += we - ws
        del logits, logits_fp32, log_softmax
        mx.eval(mx.zeros(1))

    print()
    print(f"=== K-floor sweep on {args.model} ===")
    print(f"V={V:,}, dataset={args.dataset}, samples={args.num_samples}, "
          f"seq={args.max_seq_len}, seed={args.seed}, window=[{ws}, {we})")
    print(f"{floor_count:,} tokens scored")
    print()
    print(f"  {'K':>10}  {'mean floor (nats)':>20}  {'p100 (nats)':>14}")
    print(f"  {'-' * 10}  {'-' * 20}  {'-' * 14}")
    means = {K: floor_sum[K] / max(floor_count, 1) for K in k_values}
    best_K = min(means, key=means.get) if means else None
    for K in k_values:
        K_disp = f"{K:,}"
        if V is not None and K == V // 2:
            K_disp = f"{K:,} (V/2)"
        marker = "  <- lowest" if K == best_K else ""
        print(f"  {K_disp:>10}  {means[K]:>20.6f}  {floor_max[K]:>14.6f}{marker}")
    print()
    if best_K is not None:
        print(f"Lowest floor at --top-k {best_K}.")
        if best_K != max(k_values):
            # Worth saying outright: the floor is U-shaped in K, so the reflex
            # of "cache more of the distribution" makes the measurement worse
            # past the minimum, on top of costing proportionally more disk.
            print("The floor rises again above that K. More slots means more "
                  "bf16 rounding, and the tail approximation this corrects for "
                  "has already shrunk. Raising --top-k past the minimum costs "
                  "disk and accuracy both.")
        print("Entry size scales linearly with K: about "
              f"{6 * best_K + 9} bytes per cached position.")
    print()
    return 0
