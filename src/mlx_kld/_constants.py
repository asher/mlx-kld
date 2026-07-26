"""Locked protocol constants + cache defaults, shared across modules.

The scoring defaults match llama.cpp's ``--kl-divergence`` so the headline KLD
reads in the same units as the field's published numbers. Keeping these in one
import-cheap module avoids circular imports between the cache, scoring, and
report layers.
"""

from __future__ import annotations

from pathlib import Path

# The record contract downstream consumers read. `compare` and `plot` accept
# only this version, so a JSON file written by some other tool that happens to
# sit under the results root is skipped rather than misread.
SCHEMA_VERSION = 1

# The cache layout is independent of the scoring protocol. Adding an optional
# manifest field (as score_window was) does not bump this, so older entries
# stay readable; only a content-layout change that would corrupt an old reader
# does.
CACHE_FORMAT_VERSION = 1

# 32768 is not an arbitrary round number: on a 151936-vocab model it sits at
# the measured minimum of the reconstruction floor. `floor-sweep` on
# Qwen3-0.6B (V=151,936, default protocol) gives, in nats:
#
#     K       512     2,048    8,192   32,768   65,536   131,072
#     floor   0.1258  0.0359   0.0069  0.0030   0.0034   0.0056
#
# The curve is U-shaped, not monotone. Below the minimum the uniform-tail
# approximation dominates and a larger K helps. Above it two other errors take
# over. Every stored log-prob is rounded to bf16, so more slots means more
# rounding, and the head/tail error cancellation `kld_from_topk` documents
# degrades as the tail shrinks. Raising K past the minimum therefore costs disk
# *and* accuracy, which is the opposite of the intuitive reading.
#
# The minimum's location depends on the vocabulary and on how peaked the model
# is, so it is not a universal constant: measure it per model with
# `mlx-kld floor-sweep`. The floor's *magnitude* varies far more than its
# location, at 0.0030 nats here against 0.0018 on a 27B at the same K.
DEFAULT_TOP_K = 32_768
DEFAULT_NUM_SAMPLES = 512
DEFAULT_MAX_SEQ_LEN = 512       # the conventional llama.cpp KLD-protocol n_ctx
DEFAULT_SEED = 123
DEFAULT_DATASET = "Salesforce/wikitext:wikitext-103-raw-v1"
DEFAULT_CACHE_DIR = Path.home() / ".mlx-kld-cache"

# JSON records land under a results root, never inside model directories.
# Resolution order: --out-dir > $MLX_KLD_RESULTS > ./kld-results.
DEFAULT_RESULTS_DIR = Path("kld-results")
RESULTS_ENV = "MLX_KLD_RESULTS"

# Self-managing teacher cache: the root is held to a budget by evicting
# least-recently-used entries after each run (the entry in use is never
# evicted). A default-config entry is ~51.5 GB, so the default budget holds
# two teachers; a budget below one entry degenerates into keeping only the
# current teacher. 0 disables auto-eviction (entries grow unbounded until
# `mlx-kld cache gc`).
DEFAULT_CACHE_MAX_GB = 120.0

# Default position buckets: quartiles of the default protocol's scored window,
# [256, 512). The scorer derives buckets from the manifest's score_window and
# passes them explicitly, so these apply when Aggregator.finalize is called
# without them, which means tests and any caller running the default
# protocol.
POSITION_BUCKETS: list[tuple[int, int | None]] = [
    (256, 320),
    (320, 384),
    (384, 448),
    (448, None),
]
