"""Optional-dependency guards for the ``[kquant]`` and ``[gguf]`` extras.

Plain affine / bf16 safetensors checkpoints load through ``mlx_lm`` (a base
dependency). Scoring a *K-quant* checkpoint additionally needs ``mlx-kquant``
for both the loader and the codec geometry used to compute bits-per-weight;
scoring a *GGUF* student needs ``gmlx`` for the zero-conversion loader and
tokenizer synthesis. Keeping those imports optional lets the base install stay
lean; a missing extra surfaces as one clear, actionable message rather than a
raw ``ImportError``.
"""

from __future__ import annotations

_KQUANT_HINT = (
    "scoring a K-quant checkpoint needs the optional mlx-kquant dependency. "
    "Install it with:\n\n    pip install 'mlx-kld[kquant]'"
)

_GGUF_HINT = (
    "scoring a GGUF student needs the optional gmlx dependency "
    "Install it with:\n\n    pip install 'mlx-kld[gguf]'"
)


def require_kquant() -> None:
    """Raise ``ImportError`` with an actionable hint if ``[kquant]`` is absent."""
    try:
        import mlx_kquant  # noqa: F401
    except ImportError as e:
        raise ImportError(f"mlx-kld: {_KQUANT_HINT}") from e


def require_gguf() -> None:
    """Raise ``ImportError`` with an actionable hint if ``[gguf]`` is absent."""
    try:
        import gmlx  # noqa: F401
    except ImportError as e:
        raise ImportError(f"mlx-kld: {_GGUF_HINT}") from e
