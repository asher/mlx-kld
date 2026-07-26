"""Student measurement: quantization descriptors, per-format measure_* output,
and the provenance sidecar reader. Safetensors cases use tiny synthetic
checkpoints; the GGUF case fakes gmlx/gguf at the module seam (header-only
logic, no real GGUF needed)."""

import json
import sys
import types

import numpy as np
from safetensors.numpy import save_file

from mlx_kld.measure import (
    affine_descriptor,
    gguf_scored_names,
    is_scored_weight,
    kquant_descriptor,
    load_provenance,
    measure_gguf_student,
    measure_student,
    unquantized_descriptor,
)

# ---------- descriptors (pure) ----------


def test_affine_descriptor():
    d = affine_descriptor({"bits": 4, "group_size": 64, "mode": "affine"})
    assert d == {"kind": "affine", "bits": 4, "group_size": 64, "mode": "affine"}


def test_affine_descriptor_defaults_mode():
    assert affine_descriptor({"bits": 5, "group_size": 32})["mode"] == "affine"


def test_kquant_descriptor_histogram():
    d = kquant_descriptor({"a.w": "q6_k", "b.w": "q6_k", "c.w": "q8_0"})
    assert d == {"kind": "kquant", "codecs": {"q6_k": 2, "q8_0": 1}}


def test_unquantized_descriptor_dominant_dtype():
    d = unquantized_descriptor({"BF16": 1000, "F32": 10})
    assert d == {"kind": "none", "dtype": "bfloat16"}


# ---------- measure_student on synthetic checkpoints ----------


def _write_affine_checkpoint(tmp_path):
    """One 4-bit module: 8x64 logical weights packed as uint32, gs=64."""
    (tmp_path / "config.json").write_text(json.dumps(
        {"quantization": {"bits": 4, "group_size": 64, "mode": "affine"}}
    ))
    save_file(
        {
            "model.layers.0.mlp.up_proj.weight":
                np.zeros((8, 8), dtype=np.uint32),
            "model.layers.0.mlp.up_proj.scales":
                np.zeros((8, 1), dtype=np.float16),
            "model.layers.0.mlp.up_proj.biases":
                np.zeros((8, 1), dtype=np.float16),
        },
        str(tmp_path / "model.safetensors"),
    )


def test_measure_student_affine(tmp_path):
    _write_affine_checkpoint(tmp_path)
    meta = measure_student(tmp_path)
    assert meta["format"] == "mlx-affine"
    assert meta["quantization"] == {
        "kind": "affine", "bits": 4, "group_size": 64, "mode": "affine",
    }
    # 8 out x (1 scale-col x 64 gs) in = 512 logical params; tensor bytes =
    # 8*8*4 (weight) + 8*2 (scales) + 8*2 (biases) = 288, which is what
    # effective_bpw divides. size_bytes is the file on disk, so it also carries
    # the safetensors header, the same convention as the GGUF path.
    assert meta["n_params"] == 512
    assert abs(meta["effective_bpw"] - 288 * 8 / 512) < 1e-9
    assert meta["size_bytes"] == (tmp_path / "model.safetensors").stat().st_size
    assert meta["size_bytes"] > 288


def test_measure_student_unquantized(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({}))
    save_file(
        {"model.layers.0.mlp.up_proj.weight": np.zeros((4, 4), dtype=np.float32)},
        str(tmp_path / "model.safetensors"),
    )
    meta = measure_student(tmp_path)
    assert meta["format"] == "mlx"
    assert meta["quantization"] == {"kind": "none", "dtype": "float32"}
    assert meta["n_params"] == 16
    assert abs(meta["effective_bpw"] - 32.0) < 1e-9


# ---------- measure_gguf_student (fake gmlx/gguf seam) ----------


def _install_fake_gguf_stack(monkeypatch, shards, n_params, wire_bytes_per_shard):
    fake_gmlx = types.ModuleType("gmlx")
    fake_preflight_mod = types.ModuleType("gmlx.preflight")

    def fake_preflight(path, **kwargs):
        assert path == str(shards[0])
        return types.SimpleNamespace(
            arch="qwen3",
            shards=[str(s) for s in shards],
            codec_histogram={"Q6_K": 3, "F32": 1},
            n_params=n_params,
        )

    fake_preflight_mod.preflight = fake_preflight
    fake_gmlx.preflight = fake_preflight_mod

    class FakeReader:
        def __init__(self, path, mode):
            entries = wire_bytes_per_shard[str(path)]
            self.tensors = [
                types.SimpleNamespace(
                    n_bytes=e[0] if isinstance(e, tuple) else e,
                    name=e[1] if isinstance(e, tuple) else f"blk.0.attn_q.w{i}",
                    shape=e[2] if isinstance(e, tuple) else [1, 1],
                )
                for i, e in enumerate(entries)
            ]

    fake_gguf = types.ModuleType("gguf")
    fake_gguf.GGUFReader = FakeReader
    monkeypatch.setitem(sys.modules, "gmlx", fake_gmlx)
    monkeypatch.setitem(sys.modules, "gmlx.preflight", fake_preflight_mod)
    monkeypatch.setitem(sys.modules, "gguf", fake_gguf)


def test_measure_gguf_student(tmp_path, monkeypatch):
    p = tmp_path / "student-Q6_K.gguf"
    p.write_bytes(b"\x00" * 1000)
    _install_fake_gguf_stack(
        monkeypatch, [p], n_params=1024,
        wire_bytes_per_shard={str(p): [800, 32]},
    )
    meta = measure_gguf_student(p)
    assert meta["format"] == "gguf"
    # Codec names lowercased to match the kquant descriptor's convention.
    assert meta["quantization"] == {
        "kind": "gguf", "arch": "qwen3", "codecs": {"q6_k": 3, "f32": 1},
    }
    assert meta["n_params"] == 1024
    # bpw from tensor wire bytes (832), size from the file (1000): metadata
    # overhead must not inflate bpw.
    assert abs(meta["effective_bpw"] - 832 * 8 / 1024) < 1e-9
    assert meta["size_bytes"] == 1000


def test_gguf_scored_names_drops_whole_mtp_block():
    # The MTP block's attn/ffn tensors carry ordinary names; only the sibling
    # `nextn` projections mark the block, so the block must be found by index.
    names = [
        "blk.0.attn_q.weight", "blk.0.ffn_up.weight",
        "blk.64.attn_q.weight", "blk.64.ffn_up.weight",
        "blk.64.nextn.eh_proj.weight",
        "output.weight", "v.blk.0.attn_q.weight",
    ]
    scored = gguf_scored_names(names)
    assert scored == {"blk.0.attn_q.weight", "blk.0.ffn_up.weight",
                      "output.weight"}


def test_gguf_scored_names_keeps_everything_without_mtp():
    names = ["blk.0.attn_q.weight", "blk.63.ffn_up.weight", "output.weight"]
    assert gguf_scored_names(names) == set(names)


def test_is_scored_weight_mlx():
    assert is_scored_weight("model.layers.0.self_attn.q_proj")
    assert not is_scored_weight("mtp.layers.0.self_attn.q_proj")
    assert not is_scored_weight("model.visual.blocks.0.attn.qkv")
    assert not is_scored_weight("vision_tower.encoder.layers.0.mlp.fc1")


def test_measure_gguf_student_scored_excludes_mtp_block(tmp_path, monkeypatch):
    p = tmp_path / "student-Q6_K.gguf"
    p.write_bytes(b"\x00" * 1000)
    # (n_bytes, name, shape): one real tensor + an MTP block worth 24 params.
    _install_fake_gguf_stack(
        monkeypatch, [p], n_params=1024,
        wire_bytes_per_shard={str(p): [
            (800, "blk.0.attn_q.weight", [10, 100]),
            (32, "blk.64.attn_q.weight", [4, 6]),
            (8, "blk.64.nextn.eh_proj.weight", [1, 1]),
        ]},
    )
    meta = measure_gguf_student(p)
    assert meta["effective_bpw"] == 840 * 8 / 1024      # unchanged, all bytes
    # scored drops both blk.64 tensors: 800 bytes over 1024-25 params.
    assert meta["scored_bytes"] == 800
    assert meta["scored_n_params"] == 1024 - (24 + 1)
    assert abs(meta["scored_bpw"] - 800 * 8 / 999) < 1e-9


def test_measure_gguf_student_split_shards(tmp_path, monkeypatch):
    s1 = tmp_path / "m-00001-of-00002.gguf"
    s2 = tmp_path / "m-00002-of-00002.gguf"
    s1.write_bytes(b"\x00" * 100)
    s2.write_bytes(b"\x00" * 60)
    _install_fake_gguf_stack(
        monkeypatch, [s1, s2], n_params=256,
        wire_bytes_per_shard={str(s1): [64], str(s2): [64]},
    )
    meta = measure_gguf_student(s1)
    assert meta["size_bytes"] == 160
    assert abs(meta["effective_bpw"] - 128 * 8 / 256) < 1e-9


# ---------- provenance sidecar ----------


def test_load_provenance_missing_returns_none(tmp_path):
    assert load_provenance(tmp_path) is None


def test_load_provenance_keeps_scalar_fields(tmp_path):
    """The sidecar is free-form, so any scalar field a quantizer writes is
    carried through rather than filtered against a fixed key list."""
    (tmp_path / "quant-recipe.json").write_text(json.dumps({
        "tool": "some-quantizer",
        "tool_version": "1.2.3",
        "bits": 4,
        "group_size": 64,
        "some_future_flag": False,
        "unset": None,
    }))
    prov = load_provenance(tmp_path)
    assert prov["tool"] == "some-quantizer"
    assert prov["bits"] == 4
    assert prov["some_future_flag"] is False
    assert prov["unset"] is None


def test_load_provenance_drops_non_scalar_fields(tmp_path):
    """Nested structures belong to the tool that wrote them, not inlined into a
    scoring record."""
    (tmp_path / "quant-recipe.json").write_text(json.dumps({
        "bits": 4,
        "per_layer": {"0": 4, "1": 6},
        "tags": ["a", "b"],
    }))
    prov = load_provenance(tmp_path)
    assert prov == {"bits": 4}


def test_load_provenance_bounds_untrusted_input(tmp_path):
    """A sidecar is another tool's output, so it cannot bloat every record
    written against the checkpoint."""
    from mlx_kld.measure import _MAX_RECIPE_KEYS, _MAX_RECIPE_STR

    (tmp_path / "quant-recipe.json").write_text(json.dumps(
        {"long": "x" * 5000, **{f"k{i}": i for i in range(200)}}
    ))
    prov = load_provenance(tmp_path)
    assert len(prov) == _MAX_RECIPE_KEYS
    assert len(prov["long"]) == _MAX_RECIPE_STR


def test_load_provenance_rejects_non_object_sidecar(tmp_path):
    (tmp_path / "quant-recipe.json").write_text(json.dumps(["not", "an", "object"]))
    assert load_provenance(tmp_path) is None


def test_load_provenance_rejects_corrupt_sidecar(tmp_path):
    (tmp_path / "quant-recipe.json").write_text("{not json")
    assert load_provenance(tmp_path) is None
