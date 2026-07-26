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
    "(Python 3.11+). Install it with:\n\n    pip install 'mlx-kld[gguf]'"
)


def require_kquant() -> None:
    """Raise ``ImportError`` with an actionable hint if ``[kquant]`` is absent."""
    try:
        import mlx_kquant  # noqa: F401
    except ImportError as e:
        raise ImportError(f"mlx-kld: {_KQUANT_HINT}") from e


def require_gguf() -> None:
    """Raise ``ImportError`` with an actionable hint if ``[gguf]`` is absent."""
    import sys

    try:
        import gmlx  # noqa: F401
    except ImportError as e:
        # Diagnose the failure rather than gate on the version: gmlx's metadata
        # enforces 3.11+, so on 3.10 the install fails with a raw pip-resolver
        # error and "pip install 'mlx-kld[gguf]'" is advice that cannot work.
        # Checking the version only here keeps an importable gmlx usable on any
        # interpreter that managed to import it.
        if sys.version_info < (3, 11):
            raise ImportError(
                "mlx-kld: scoring a GGUF student needs Python 3.11+. This is "
                f"{sys.version_info[0]}.{sys.version_info[1]}, where the "
                "base install and safetensors scoring still work."
            ) from e
        raise ImportError(f"mlx-kld: {_GGUF_HINT}") from e
