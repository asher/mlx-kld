"""mlx-kld: KL-divergence scoring for quantized students (MLX safetensors or
GGUF) against a full-precision teacher.

Scores a quantized (or otherwise modified) student against a full-precision
teacher using top-K-truncated, disk-cached teacher logits. Text-only in v0.1.
K-quant safetensors checkpoints load via the optional ``[kquant]`` extra, and
GGUF students (K-quant / IQ / MXFP4) via the optional ``[gguf]`` extra. The
``mlx-kld`` CLI exposes ``score`` / ``compare`` / ``plot`` / ``cache`` /
``floor-sweep`` / ``self-test``.

The public Python API (used by external quantizer tooling) is
:func:`ensure_teacher_topk_cache` and :func:`score_loaded_student`, plus
:func:`entry_lock` to hold a cache entry against eviction across a replay.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .cache import ensure_teacher_topk_cache, entry_lock
from .errors import (
    CacheMismatchError,
    CalibrationCorpusError,
    MlxKldError,
    RecordSchemaError,
    TokenizerMismatchError,
)
from .scoring import score_loaded_student

__all__ = [
    "__version__",
    "ensure_teacher_topk_cache",
    "entry_lock",
    "score_loaded_student",
    "MlxKldError",
    "CacheMismatchError",
    "CalibrationCorpusError",
    "RecordSchemaError",
    "TokenizerMismatchError",
]
