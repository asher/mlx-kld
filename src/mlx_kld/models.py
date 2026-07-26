"""Model loading + logits extraction.

Text-only checkpoints. Plain affine / bf16 safetensors load via ``mlx_lm``;
K-quant safetensors (``quantization.mode == "kquant"``) load via the optional
``mlx-kquant`` package; GGUF files (K-quant / IQ / MXFP4 codecs) load via the
optional ``gmlx`` package. No VLM in v0.1.
"""

from __future__ import annotations

import json
from pathlib import Path

from ._log import info


def is_gguf_path(path_or_id: str) -> bool:
    """True iff ``path_or_id`` is a local ``.gguf`` file (GGUF student)."""
    p = Path(path_or_id)
    return p.suffix.lower() == ".gguf" and p.is_file()


def _peek_config(path_or_id: str) -> dict:
    """Read ``config.json`` without instantiating the model (for early dispatch).

    Local dirs are read directly; HF ids fetch just ``config.json`` via
    ``huggingface_hub`` (no model download, no mlx-vlm dependency).
    """
    p = Path(path_or_id)
    if p.is_dir():
        return json.loads((p / "config.json").read_text())
    if p.is_file() and p.name == "config.json":
        return json.loads(p.read_text())
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(repo_id=path_or_id, filename="config.json")
    return json.loads(Path(cfg_path).read_text())


def _is_kquant(cfg: dict) -> bool:
    qc = cfg.get("quantization_config") or {}
    if qc.get("mode") == "kquant":
        return True
    # mlx-kquant checkpoints carry the block under `quantization`, not
    # `quantization_config`; accept either spelling.
    q = cfg.get("quantization") or {}
    return q.get("mode") == "kquant"


# mlx-lm's stock qwen3_5 sanitize gates the zero-centered-norm +1.0 shift on
# `has_mtp_weights or has_unsanitized_conv1d`. The mtp term is a stale proxy:
# already-converted MLX checkpoints keep their norms in standard (post-shift)
# form yet still carry mtp. tensors, so the stock gate double-shifts them and
# corrupts the logits (e.g. Jundot Qwen3.6 oQ4e-mtp -> top-1 ~0%). The fix,
# mlx-lm PR #990 / commit b3b9639, narrows the gate to conv1d state only, which
# still shifts genuinely-raw checkpoints (their conv1d is not yet transposed).
#
# This replaces a third-party method process-wide, so it is gated: the patch is
# skipped once upstream's own sanitize stops gating on mtp (see
# _sanitize_still_gates_on_mtp). Still stock as of mlx-lm 0.31.3.
# TODO: drop this monkeypatch once the minimum supported mlx-lm includes #990.
def _qwen35_conv1d_gated_sanitize(self, weights):
    has_unsanitized_conv1d = any(
        "conv1d.weight" in k and v.shape[-1] != 1 for k, v in weights.items()
    )
    should_shift = has_unsanitized_conv1d
    weights = {k: v for k, v in weights.items() if "mtp." not in k}
    if getattr(self.args, "tie_word_embeddings", False):
        weights.pop("lm_head.weight", None)
    norm_keys = (
        ".input_layernorm.weight",
        ".post_attention_layernorm.weight",
        "model.norm.weight",
        ".q_norm.weight",
        ".k_norm.weight",
    )
    for k, v in weights.items():
        if "conv1d.weight" in k and v.shape[-1] != 1:
            weights[k] = v.moveaxis(2, 1)
        if should_shift and any(k.endswith(sfx) for sfx in norm_keys):
            if v.ndim == 1:
                weights[k] = v + 1.0
    return weights


_QWEN35_NORMSHIFT_PATCHED = False


def _sanitize_still_gates_on_mtp(sanitize_fn) -> bool:
    """True when mlx-lm's ``sanitize`` still uses the stale mtp proxy.

    Detected from the source rather than a version number, because the release
    carrying PR #990 is not known at the time of writing and a wrong pin would
    fail in the worse direction. Unreadable source (a C extension, a stripped
    install) answers True: the vendored gate is a strict narrowing of the stock
    one, so applying it when it was not needed is safe, while skipping it when
    it *was* needed silently corrupts logits.
    """
    import inspect

    try:
        return "has_mtp_weights" in inspect.getsource(sanitize_fn)
    except (OSError, TypeError):
        return True


def _apply_qwen35_normshift_fix() -> None:
    """Patch mlx-lm's ``qwen3_5.TextModel.sanitize`` to the conv1d-only norm
    shift gate (see ``_qwen35_conv1d_gated_sanitize``). Idempotent; a no-op if
    mlx-lm lacks the module, or already carries the fix. The VL wrapper's
    ``Model.sanitize`` delegates to the language model's, so patching
    ``TextModel`` covers both paths."""
    global _QWEN35_NORMSHIFT_PATCHED
    if _QWEN35_NORMSHIFT_PATCHED:
        return
    try:
        from mlx_lm.models import qwen3_5
    except ImportError:
        return
    if not _sanitize_still_gates_on_mtp(qwen3_5.TextModel.sanitize):
        info("mlx-lm already gates the qwen3_5 norm shift on conv1d state. "
             "The vendored PR #990 patch is no longer needed and can be dropped")
        _QWEN35_NORMSHIFT_PATCHED = True
        return
    qwen3_5.TextModel.sanitize = _qwen35_conv1d_gated_sanitize
    _QWEN35_NORMSHIFT_PATCHED = True


def _load_model(path_or_id: str, lazy: bool = True):
    """Dispatch: ``.gguf`` file -> gmlx loader; kquant config -> mlx-kquant
    loader; else mlx_lm.

    Returns ``(model, config)``. The tokenizer is sourced separately (via
    ``mlx_lm.utils.load_tokenizer`` for safetensors paths, or synthesized from
    GGUF metadata; see ``tokenizer.load_gguf_tokenizer``).
    """
    if is_gguf_path(path_or_id):
        from ._deps import require_gguf

        require_gguf()
        from gmlx import load_model as gmlx_load

        # gmlx returns a stock mlx-lm Model with quantized leaves swapped for
        # KQuant* modules (wire-byte read -> name remap -> config synth ->
        # module swap); it drives the same forward/cache path as mlx_lm models.
        #
        # `lazy` has no effect here: gmlx's zero-conversion loader mmaps the
        # wire bytes and has no eager/lazy switch to forward. Named in the
        # signature for one call shape across formats.
        model, config, _tok = gmlx_load(path_or_id)
        return model, config
    cfg = _peek_config(path_or_id)
    if _is_kquant(cfg):
        from ._deps import require_kquant

        require_kquant()
        from mlx_kquant.loader import load as kq_load

        # mlx-kquant's loader returns (model, config) with KQuant* modules
        # installed and weights loaded.
        model, config = kq_load(path_or_id, lazy=lazy)
        return model, config
    from mlx_lm.utils import load

    _apply_qwen35_normshift_fix()
    model, _tok, config = load(path_or_id, lazy=lazy, return_config=True)
    return model, config


def _extract_logits(out):
    """mlx-lm models return a logits array directly; some wrappers box it in a
    dataclass with a ``.logits`` attribute. Normalize to the array."""
    return out.logits if hasattr(out, "logits") else out


def _make_fresh_cache(model):
    """Build a fresh per-batch KV cache aligned to the model's LM stack.

    Some models skip RoPE bookkeeping in ``__call__`` when ``cache`` is None, so
    we always pass a cache and the teacher/student passes share one forward path.
    Harmless on plain text models (argmax unchanged with vs. without cache).
    """
    from mlx_lm.models import cache as cache_mod

    target = model.language_model if hasattr(model, "language_model") else model
    return cache_mod.make_prompt_cache(target, max_kv_size=None)
