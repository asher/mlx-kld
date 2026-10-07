"""Model loading + logits extraction.

Text-only checkpoints. Plain affine / bf16 safetensors load via ``mlx_lm``;
K-quant safetensors (``quantization.mode == "kquant"``) load via the optional
``mlx-kquant`` package; GGUF files (K-quant / IQ / MXFP4 codecs) load via the
optional ``gmlx`` package. No VLM in v0.1.
"""

from __future__ import annotations

import json
from pathlib import Path


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
