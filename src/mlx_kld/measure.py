"""Student checkpoint measurement: on-disk bytes, logical parameter count,
effective bits-per-weight, and a per-format ``quantization`` descriptor.

Every student measures to the same shape regardless of format::

    {size_bytes, effective_bpw, n_params, format, quantization}

``format`` is one of ``mlx-affine`` / ``mlx-kquant`` / ``gguf`` /
``mlx`` (unquantized MLX); ``quantization`` is the format-appropriate descriptor (affine
bits/group_size, or a codec histogram). K-quant checkpoints reuse
``mlx_kquant.codec_geometry`` for block layout; GGUF measurement is header-only
via ``gmlx.preflight`` + per-tensor wire bytes (no tensor data read).

The two size fields mean different things, and the same thing in every format:

``size_bytes``
    On-disk bytes of the weight files, meaning ``*.safetensors`` for an MLX
    student and the ``.gguf`` shards for a GGUF one. What a download costs. Container
    overhead is included because it is genuinely on disk, and config/tokenizer
    sidecars are excluded because GGUF has no separable counterpart to them.

``effective_bpw``
    Tensor wire bytes x 8 / logical parameters, in every format. Container
    overhead is excluded here, so a GGUF's embedded metadata does not inflate
    its apparent bit width against an MLX sibling.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from safetensors import safe_open

from .models import _is_kquant

# ---------- role classification ----------

# Order matters: more specific patterns first.
_ROLE_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("embedding",          re.compile(r"\b(embed_tokens|wte|word_embeddings)\b")),
    ("lm_head",            re.compile(r"(^|\.)lm_head$")),
    ("attn.q_proj",        re.compile(r"\.q_proj$")),
    ("attn.k_proj",        re.compile(r"\.k_proj$")),
    ("attn.v_proj",        re.compile(r"\.v_proj$")),
    ("attn.o_proj",        re.compile(r"\.o_proj$")),
    ("attn.qkv_proj",      re.compile(r"\.(qkv_proj|qkv)$")),
    ("attn.out_proj",      re.compile(r"\.(out_proj|attn\.proj)$")),
    ("linear_attn.in_proj_qkv", re.compile(r"\.linear_attn\.in_proj_qkv$")),
    ("linear_attn.in_proj_z",   re.compile(r"\.linear_attn\.in_proj_z$")),
    ("linear_attn.out_proj",    re.compile(r"\.linear_attn\.out_proj$")),
    ("linear_attn.conv1d",      re.compile(r"\.linear_attn\.conv1d$")),
    ("linear_attn.dt_proj",     re.compile(r"\.linear_attn\.dt_proj$")),
    ("linear_attn.norm",        re.compile(r"\.linear_attn\.norm")),
    ("linear_attn.other",       re.compile(r"\.linear_attn\.")),
    ("mlp.gate_proj",      re.compile(r"\.mlp\.gate_proj$")),
    ("mlp.up_proj",        re.compile(r"\.mlp\.up_proj$")),
    ("mlp.down_proj",      re.compile(r"\.mlp\.down_proj$")),
    ("mlp.gate_up_proj",   re.compile(r"\.mlp\.gate_up_proj$")),
    ("experts.gate_up",
     re.compile(r"\.experts\.(switch_mlp\.)?(gate_up_proj|gate_proj|up_proj)$")),
    ("experts.down",       re.compile(r"\.experts\.(switch_mlp\.)?down_proj$")),
    ("router",             re.compile(r"\.(router|gate)\.(weight|w[12])$|\.router$|\.gate$")),
    ("shared_expert",      re.compile(r"\.shared_expert(\.|_gate)")),
    ("norm",               re.compile(r"(\.norm$|\.layernorm$|_norm$|_layernorm$|\.rmsnorm$)")),
    ("vlm.merger",
     re.compile(r"(^|\.)(merger|connector|multi_modal_projector|vl_connector)\.")),
    ("vlm.patch_embed",    re.compile(r"\.patch_embed")),
    ("vlm.visual",         re.compile(r"(^|\.)(visual|vision_tower|vision_model)\.")),
    ("vlm.audio",          re.compile(r"(^|\.)(audio_tower|audio_model)\.")),
    ("vlm.projector",      re.compile(r"(^|\.)(embed_vision|embed_audio)\.")),
]


def classify_role(base: str) -> str:
    for role, pat in _ROLE_PATTERNS:
        if pat.search(base):
            return role
    return "other"


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index(base: str) -> int | None:
    m = _LAYER_RE.search(base)
    return int(m.group(1)) if m else None


# ---------- scored-weight split ----------

# The loader drops the MTP draft stack and the vision/audio towers before a
# single logit is produced, so bytes spent there buy no next-token quality.
# Reporting bits per *scored* weight alongside the raw figure keeps size
# comparisons honest between checkpoints carrying different amounts of such
# baggage (a VL+MTP build can be ~3% dead weight against a text-only one).
# Matched against the whole name, not via classify_role: the role table is
# ordered and its attn/mlp patterns fire first, so a vision tensor ending in
# ``.attn.qkv`` would classify as attention and be counted as scored.
_UNSCORED_KEY_RE = re.compile(
    r"(^|\.)(mtp|visual|vision_tower|vision_model|audio_tower|audio_model"
    r"|merger|connector|multi_modal_projector|vl_connector"
    r"|embed_vision|embed_audio)\.|\.patch_embed"
)


def is_scored_weight(base: str) -> bool:
    """False for MLX weights the scoring pass never loads (the MTP stack, and
    vision/audio towers), True otherwise."""
    if _UNSCORED_KEY_RE.search(base):
        return False
    return not classify_role(base).startswith("vlm.")


_GGUF_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


def gguf_scored_names(names) -> set[str]:
    """The subset of GGUF tensor names the scoring pass uses.

    GGUF spells the MTP head as an extra decoder block whose tensors carry
    ordinary ``attn_*``/``ffn_*`` names, so it cannot be spotted by name alone.
    Only the ``nextn`` projections sitting beside them in the same block mark
    it.
    Locate those blocks, then drop them whole. Inline vision/projector tensors
    (``v.``, ``mm.``) are dropped the same way.
    """
    names = list(names)
    mtp_blocks = {
        m.group(1) for n in names
        if ".nextn." in n and (m := _GGUF_BLOCK_RE.match(n))
    }
    scored = set()
    for n in names:
        m = _GGUF_BLOCK_RE.match(n)
        if m and m.group(1) in mtp_blocks:
            continue
        if n.startswith(("v.", "mm.")):
            continue
        scored.add(n)
    return scored


def _scored_fields(scored_bytes: int, scored_params: int) -> dict:
    """Uniform scored-* block. Equals the raw figures when nothing was excluded,
    so consumers never have to special-case a checkpoint without baggage."""
    return {
        "scored_n_params": scored_params,
        "scored_bytes": scored_bytes,
        "scored_bpw": (scored_bytes * 8 / scored_params) if scored_params else None,
    }


# ---------- safetensors metadata walk ----------

_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
    "BOOL": 1,
}


def _compute_nbytes(shape: list[int], dtype: str) -> int:
    elem_bytes = _DTYPE_BYTES.get(dtype, 0)
    if not elem_bytes:
        return 0
    return math.prod(shape) * elem_bytes


def collect_tensor_metadata(model_dir: Path) -> dict[str, dict]:
    """Return ``{tensor_name: {dtype, shape, nbytes, shard}}`` for every tensor,
    read from the safetensors headers (no data load)."""
    index_path = model_dir / "model.safetensors.index.json"
    single = model_dir / "model.safetensors"

    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
    elif single.exists():
        shards = ["model.safetensors"]
    else:
        raise FileNotFoundError(f"No safetensors found in {model_dir}")

    meta: dict[str, dict] = {}
    for shard in shards:
        path = model_dir / shard
        with safe_open(path, framework="numpy") as f:
            for key in f.keys():
                slice_view = f.get_slice(key)
                shape = list(slice_view.get_shape())
                dtype = slice_view.get_dtype()
                meta[key] = {
                    "dtype": dtype,
                    "shape": shape,
                    "shard": shard,
                    "nbytes": _compute_nbytes(shape, dtype),
                }
    return meta


def group_into_modules(meta: dict[str, dict]) -> dict[str, dict]:
    """Group ``.weight`` + ``.scales`` + ``.biases`` triples under a shared base
    name. Tensors without one of these suffixes still produce a base entry with
    weight only."""
    modules: dict[str, dict] = {}
    for name, info in meta.items():
        for suffix in (".weight", ".scales", ".biases"):
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                kind = suffix[1:]  # strip leading dot
                modules.setdefault(base, {})[kind] = info
                break
        else:
            modules.setdefault(name, {"weight": info})
    return modules


def derive_module_recipe(
    base: str,
    parts: dict,
    quant_cfg: dict,
    default_group_size: int | None,
) -> dict:
    """Determine bits, group_size, dtype, param count, effective bpw for one
    affine-quantized (or unquantized) module."""
    weight = parts.get("weight")
    scales = parts.get("scales")

    is_quantized = scales is not None
    declared = quant_cfg.get(base) if isinstance(quant_cfg.get(base), dict) else None

    info: dict[str, Any] = {
        "base": base,
        "role": classify_role(base),
        "layer": layer_index(base),
        "quantized": is_quantized,
        "declared_bits": declared["bits"] if declared else None,
        "declared_group_size": declared["group_size"] if declared else None,
        "weight_dtype": weight["dtype"] if weight else None,
    }

    total_bytes = sum(p["nbytes"] for p in parts.values() if p)
    info["total_bytes"] = total_bytes

    param_count = None
    if is_quantized and weight and scales:
        gs = info["declared_group_size"] or default_group_size or 64
        in_features = scales["shape"][-1] * gs
        out_dims = weight["shape"][:-1]
        param_count = math.prod(out_dims) * in_features
    elif weight:
        param_count = math.prod(weight["shape"])

    info["param_count"] = param_count
    return info


# ---------- quantization descriptors (schema v1 `student.quantization`) ----------

# The four public student formats (locked schema enum).
STUDENT_FORMATS = ("mlx-affine", "mlx-kquant", "gguf", "mlx")

_DTYPE_NAMES = {
    "F64": "float64", "F32": "float32", "F16": "float16", "BF16": "bfloat16",
}


def affine_descriptor(quant_cfg: dict) -> dict:
    """Descriptor for an mlx-lm affine-quantized checkpoint, from its config
    ``quantization`` block."""
    return {
        "kind": "affine",
        "bits": quant_cfg.get("bits"),
        "group_size": quant_cfg.get("group_size"),
        "mode": quant_cfg.get("mode") or "affine",
    }


def kquant_descriptor(per_tensor: dict) -> dict:
    """Descriptor for an mlx-kquant checkpoint: codec histogram from the
    per-tensor codec map (there is no global bits/group_size)."""
    codecs: dict[str, int] = {}
    for codec in per_tensor.values():
        codecs[codec] = codecs.get(codec, 0) + 1
    return {"kind": "kquant", "codecs": codecs}


def unquantized_descriptor(dtype_bytes: dict[str, int]) -> dict:
    """Descriptor for an unquantized checkpoint; ``dtype`` is the dominant
    weight dtype by byte count."""
    dtype = max(dtype_bytes, key=dtype_bytes.get) if dtype_bytes else None
    if dtype is not None:
        dtype = _DTYPE_NAMES.get(dtype, dtype.lower())
    return {"kind": "none", "dtype": dtype}


def weight_file_bytes(student_dir: Path) -> int:
    """On-disk bytes of an MLX student's ``*.safetensors`` files.

    The GGUF path reports shard file sizes, so the MLX path has to report file
    sizes too or the ``size_bytes`` column silently compares two different
    quantities across formats. Summing tensor ``nbytes`` instead would drop the
    safetensors header, which is small but not zero.
    """
    return sum(f.stat().st_size for f in student_dir.glob("*.safetensors"))


# ---------- kquant per-tensor geometry ----------

def _kquant_per_tensor(config: dict) -> dict:
    """The path -> codec map, accepting either config spelling.

    Current mlx-kquant checkpoints carry it under ``quantization.per_tensor``;
    some earlier ones used ``quantization_config.per_tensor``.
    """
    for key in ("quantization", "quantization_config"):
        block = config.get(key) or {}
        pt = block.get("per_tensor")
        if pt:
            return pt
    return {}


def _measure_kquant_student(student_dir: Path, config: dict) -> dict:
    """Measurement for a kquant checkpoint.

    kquant has no global (bits, group_size); it uses a per-tensor codec map.
    Logical param count per tensor is ``wire_bytes // bytes_per_block *
    weights_per_block`` for kquant entries, and ``prod(shape)`` for bf16
    pass-through tensors. Block geometry comes from ``mlx_kquant.codec_geometry``.
    """
    from ._deps import require_kquant

    require_kquant()
    from mlx_kquant.codec_geometry import CODEC_GEOMETRY

    per_tensor = _kquant_per_tensor(config)
    meta = collect_tensor_metadata(student_dir)
    total_bytes = 0
    total_params = 0
    scored_bytes = 0
    scored_params = 0
    for name, info in meta.items():
        nbytes = info["nbytes"]
        total_bytes += nbytes
        base = name.removesuffix(".weight") if name.endswith(".weight") else name
        codec = per_tensor.get(base)
        if codec is not None and codec in CODEC_GEOMETRY:
            _gs, _bits, bpb, wpb = CODEC_GEOMETRY[codec]
            params = (nbytes // bpb) * wpb
        else:
            shape = info.get("shape") or []
            params = 1
            for d in shape:
                params *= int(d)
        total_params += params
        if is_scored_weight(base):
            scored_bytes += nbytes
            scored_params += params
    bpw = (total_bytes * 8 / total_params) if total_params else None
    return {
        "size_bytes": weight_file_bytes(student_dir),
        "effective_bpw": bpw,
        "n_params": total_params,
        "format": "mlx-kquant",
        "quantization": kquant_descriptor(per_tensor),
        **_scored_fields(scored_bytes, scored_params),
    }


def measure_student(student_dir: Path) -> dict:
    """Measure an MLX safetensors student (affine, kquant, or unquantized)."""
    config = json.loads((student_dir / "config.json").read_text())
    if _is_kquant(config):
        return _measure_kquant_student(student_dir, config)
    quant_cfg = config.get("quantization", {}) or {}
    if not isinstance(quant_cfg, dict):
        quant_cfg = {}
    default_bits = quant_cfg.get("bits")   # read below to detect affine configs
    default_gs = quant_cfg.get("group_size")

    meta = collect_tensor_metadata(student_dir)
    modules_raw = group_into_modules(meta)
    modules = []
    scored_params = scored_bytes = 0
    for b, parts in modules_raw.items():
        if "weight" not in parts:
            continue
        m = derive_module_recipe(b, parts, quant_cfg, default_gs)
        modules.append(m)
        if is_scored_weight(b):
            scored_params += m["param_count"] or 0
            scored_bytes += m["total_bytes"]
    total_params = sum((m["param_count"] or 0) for m in modules)
    total_bytes = sum(m["total_bytes"] for m in modules)
    bpw = (total_bytes * 8 / total_params) if total_params else None

    if any(m["quantized"] for m in modules) or default_bits is not None:
        quantization = affine_descriptor(quant_cfg)
        fmt = "mlx-affine"
    else:
        dtype_bytes: dict[str, int] = {}
        for m in modules:
            d = m.get("weight_dtype")
            if d:
                dtype_bytes[d] = dtype_bytes.get(d, 0) + m["total_bytes"]
        quantization = unquantized_descriptor(dtype_bytes)
        fmt = "mlx"
    return {
        "size_bytes": weight_file_bytes(student_dir),
        "effective_bpw": bpw,
        "n_params": total_params,
        "format": fmt,
        "quantization": quantization,
        **_scored_fields(scored_bytes, scored_params),
    }


def measure_gguf_student(gguf_path: Path) -> dict:
    """Counterpart to ``measure_student`` for GGUF students. Header-only:
    ``gmlx.preflight`` supplies arch / codec histogram / logical param count
    (and shard discovery for split GGUFs); wire bytes come from the GGUF tensor
    table via ``GGUFReader``. ``size_bytes`` is the shard file total and
    ``effective_bpw`` uses tensor wire bytes, the same split as the MLX path,
    so GGUF metadata overhead doesn't inflate the bit width."""
    from ._deps import require_gguf

    require_gguf()
    from gguf import GGUFReader
    from gmlx.preflight import preflight

    pf = preflight(str(gguf_path))
    wire_bytes = 0
    tensors: list[tuple[str, int, int]] = []
    for shard in pf.shards:
        reader = GGUFReader(str(shard), "r")
        for t in reader.tensors:
            nbytes = int(t.n_bytes)
            wire_bytes += nbytes
            n = 1
            for d in t.shape:
                n *= int(d)
            tensors.append((str(t.name), nbytes, n))
    size_bytes = sum(Path(s).stat().st_size for s in pf.shards)
    bpw = (wire_bytes * 8 / pf.n_params) if pf.n_params else None
    scored_names = gguf_scored_names(name for name, _, _ in tensors)
    scored_bytes = sum(b for name, b, _ in tensors if name in scored_names)
    # Subtract from the preflight count rather than re-summing the scored
    # tensors, so scored_n_params stays on the same footing as n_params.
    unscored_params = sum(n for name, _, n in tensors if name not in scored_names)
    scored_params = max(pf.n_params - unscored_params, 0) if pf.n_params else 0
    return {
        "size_bytes": size_bytes,
        "effective_bpw": bpw,
        "n_params": pf.n_params,
        **_scored_fields(scored_bytes, scored_params),
        "format": "gguf",
        "quantization": {
            "kind": "gguf",
            "arch": pf.arch,
            # Lowercased to match the kquant descriptor's codec naming, so
            # cross-format consumers can group codecs without case-folding.
            "codecs": {k.lower(): v for k, v in pf.codec_histogram.items()},
        },
    }


# ---------- provenance sidecar (non-locked) ----------

# A quantizer may drop a recipe sidecar beside its output describing how the
# checkpoint was built (bit widths, per-role choices, tool version). When one is
# present it is folded into the record as free-form provenance, so a reader can
# see how a checkpoint was produced next to how it scored.
#
# Nothing here is required, validated, or interpreted: absence changes no
# measurement, and no key carries meaning to this tool. The names are checked in
# order, first match wins.
RECIPE_FILENAMES = ("quant-recipe.json", "recipe.json")

# Bounds on what gets copied into the record. A sidecar is written by another
# tool, so it is untrusted input: scalars only (a nested structure is that
# tool's business, not something to inline into a scoring record), and caps on
# both the key count and string length so an unexpected sidecar cannot bloat
# every record written against it.
_MAX_RECIPE_KEYS = 64
_MAX_RECIPE_STR = 200


def load_provenance(student_dir: Path) -> dict | None:
    """Read an optional quantizer recipe sidecar (see ``RECIPE_FILENAMES``).

    Returns its scalar fields, or ``None`` when no sidecar exists, it is not
    valid JSON, or it is not a JSON object. The result is free-form provenance
    in the JSON record. It is never required, never validated, and never
    measured against.
    """
    for name in RECIPE_FILENAMES:
        recipe_file = student_dir / name
        if not recipe_file.is_file():
            continue
        try:
            loaded = json.loads(recipe_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(loaded, dict):
            return None
        out: dict[str, Any] = {}
        for k, v in loaded.items():
            if len(out) >= _MAX_RECIPE_KEYS:
                break
            if isinstance(v, str):
                out[str(k)] = v[:_MAX_RECIPE_STR]
            elif v is None or isinstance(v, (bool, int, float)):
                out[str(k)] = v
        return out or None
    return None
