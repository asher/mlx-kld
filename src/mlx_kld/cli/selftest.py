"""``mlx-kld self-test``: numerical + schema unit suite (no model load)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tempfile
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "self-test",
        help="run the numerical + schema unit suite (no model load)",
        description="Unit-test the KLD math, top-K reconstruction, tokenizer "
        "hashing, locked-schema validator, and provenance loader. Needs only "
        "mlx + numpy, and loads no model.",
    )
    p.set_defaults(func=cmd)


def cmd(_args: argparse.Namespace) -> int:
    return run_self_test()


def run_self_test() -> int:
    """Fast numerical + schema unit checks. Returns a process exit code (0 = all passed)."""
    import mlx.core as mx
    import numpy as np

    from ..errors import RecordSchemaError
    from ..kld_math import kld_from_topk
    from ..measure import affine_descriptor, kquant_descriptor, load_provenance
    from ..report import validate_locked_schema
    from ..tokenizer import tokenizer_hash

    failures: list[str] = []

    # ---- KLD math: uniform vs uniform == 0 ----
    V = 1024
    K = 16
    teacher_log = mx.full((1, 1, V), -math.log(V), dtype=mx.float32)
    idx = mx.arange(K, dtype=mx.int32).reshape(1, 1, K)
    topk = mx.take_along_axis(teacher_log, idx, axis=-1)
    student_logits = mx.zeros((1, 1, V), dtype=mx.float32)
    kld, _t1, _t5 = kld_from_topk(topk, idx, student_logits, V)
    val = float(np.asarray(kld).item())
    if abs(val) > 1e-4:
        failures.append(f"uniform-vs-uniform KLD = {val} (expected ~0)")

    # ---- KLD math: peaked vs uniform > 0 (and large) ----
    np_logits = np.full((V,), -10.0, dtype=np.float32)
    np_logits[0] = 5.0
    teacher_logits = mx.array(np_logits.reshape(1, 1, V))
    teacher_log = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    idx_part = mx.argpartition(teacher_log, kth=-K, axis=-1)[..., -K:]
    vals = mx.take_along_axis(teacher_log, idx_part, axis=-1)
    order = mx.argsort(-vals, axis=-1)
    idx_sorted = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
    topk_sorted = mx.take_along_axis(vals, order, axis=-1)
    student_logits = mx.zeros((1, 1, V), dtype=mx.float32)
    kld, _t1, _t5 = kld_from_topk(topk_sorted, idx_sorted, student_logits, V)
    val_peaked = float(np.asarray(kld).item())
    if val_peaked < 1.0:
        failures.append(f"peaked-vs-uniform KLD = {val_peaked} (expected > 1)")

    # ---- KLD asymmetry: KL(P||Q) != KL(Q||P) ----
    np_p = np.full((V,), -10.0, dtype=np.float32)
    np_p[0] = 5.0
    np_q = np.full((V,), -8.0, dtype=np.float32)
    np_q[1] = 4.0
    p_logits = mx.array(np_p.reshape(1, 1, V))
    q_logits = mx.array(np_q.reshape(1, 1, V))
    p_log = p_logits - mx.logsumexp(p_logits, axis=-1, keepdims=True)
    q_log = q_logits - mx.logsumexp(q_logits, axis=-1, keepdims=True)

    def topk_of(log_p):
        i = mx.argpartition(log_p, kth=-K, axis=-1)[..., -K:]
        v = mx.take_along_axis(log_p, i, axis=-1)
        o = mx.argsort(-v, axis=-1)
        return (
            mx.take_along_axis(i, o, axis=-1).astype(mx.int32),
            mx.take_along_axis(v, o, axis=-1),
        )

    p_idx, p_topk = topk_of(p_log)
    q_idx, q_topk = topk_of(q_log)
    kl_pq, _, _ = kld_from_topk(p_topk, p_idx, q_logits, V)
    kl_qp, _, _ = kld_from_topk(q_topk, q_idx, p_logits, V)
    a = float(np.asarray(kl_pq).item())
    b = float(np.asarray(kl_qp).item())
    if abs(a - b) < 1e-3:
        failures.append(f"KLD symmetric? KL(P||Q)={a}, KL(Q||P)={b}")

    # ---- top-K reconstruction round-trip ----
    raw = np.full((V,), -10.0, dtype=np.float32)
    raw[:8] = np.array([8, 7, 6, 5, 4, 3, 2, 1], dtype=np.float32)
    teacher_logits = mx.array(raw.reshape(1, 1, V))
    teacher_log_full = teacher_logits - mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    student_logits = mx.zeros((1, 1, V), dtype=mx.float32)
    student_log_full = student_logits - mx.logsumexp(student_logits, axis=-1, keepdims=True)
    exact = float(np.asarray(
        mx.sum(mx.exp(teacher_log_full) * (teacher_log_full - student_log_full), axis=-1)
    ).item())
    for K_test in (16, 64, 128):
        idx_part = mx.argpartition(teacher_log_full, kth=-K_test, axis=-1)[..., -K_test:]
        vals = mx.take_along_axis(teacher_log_full, idx_part, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx_sorted = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
        topk_sorted = mx.take_along_axis(vals, order, axis=-1)
        approx, _, _ = kld_from_topk(topk_sorted, idx_sorted, student_logits, V)
        approx_v = float(np.asarray(approx).item())
        rel = abs(approx_v - exact) / max(abs(exact), 1e-9)
        if rel > 5e-3:
            failures.append(
                f"top-K reconstruction K={K_test} on flat-tail teacher: "
                f"rel error {rel:.4%} (>0.5%)"
            )
    np.random.seed(0)
    raw2 = np.random.randn(V).astype(np.float32) * 1.5
    raw2[0] = 6.0
    teacher2 = mx.array(raw2.reshape(1, 1, V))
    teacher2_log = teacher2 - mx.logsumexp(teacher2, axis=-1, keepdims=True)
    exact2 = float(np.asarray(
        mx.sum(mx.exp(teacher2_log) * (teacher2_log - student_log_full), axis=-1)
    ).item())
    last_err = float("inf")
    for K_test in (16, 64, 256, 512):
        idx_part = mx.argpartition(teacher2_log, kth=-K_test, axis=-1)[..., -K_test:]
        vals = mx.take_along_axis(teacher2_log, idx_part, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        idx_sorted = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
        topk_sorted = mx.take_along_axis(vals, order, axis=-1)
        approx, _, _ = kld_from_topk(topk_sorted, idx_sorted, student_logits, V)
        rel = abs(float(np.asarray(approx).item()) - exact2) / max(abs(exact2), 1e-9)
        if rel > last_err + 1e-6:
            failures.append(
                f"top-K reconstruction non-monotonic at K={K_test}: "
                f"rel {rel:.4%} > prior {last_err:.4%}"
            )
        last_err = rel

    # ---- tokenizer-parity hashing ----
    if hashlib.sha256(b"a").hexdigest() == hashlib.sha256(b"b").hexdigest():
        failures.append("hashlib parity: identical hashes for distinct inputs")

    class _FakeTok:
        def __init__(self, vocab, bos=None, eos=None, pad=None, unk=None):
            self._vocab = vocab
            self.vocab_size = len(vocab)
            self.bos_token_id = bos
            self.eos_token_id = eos
            self.pad_token_id = pad
            self.unk_token_id = unk

        def __len__(self):
            return len(self._vocab)

        def convert_ids_to_tokens(self, i):
            return self._vocab[i]

    a = _FakeTok(["<a>", "<b>", "<c>"], eos=2)
    a_dup = _FakeTok(["<a>", "<b>", "<c>"], eos=2)
    b = _FakeTok(["<a>", "<b>", "<X>"], eos=2)
    c = _FakeTok(["<a>", "<b>", "<c>"], eos=1)
    if tokenizer_hash(a) != tokenizer_hash(a_dup):
        failures.append("tokenizer_hash unstable across functionally identical inputs")
    if tokenizer_hash(a) == tokenizer_hash(b):
        failures.append("tokenizer_hash collides on differing vocab")
    if tokenizer_hash(a) == tokenizer_hash(c):
        failures.append("tokenizer_hash collides on differing special-token ids")

    # ---- locked-schema validator catches missing keys ----
    try:
        validate_locked_schema({})
    except RecordSchemaError:
        pass
    else:
        failures.append("validate_locked_schema accepted empty payload")

    # ---- quantization descriptors ----
    d = affine_descriptor({"bits": 4, "group_size": 64})
    if d != {"kind": "affine", "bits": 4, "group_size": 64, "mode": "affine"}:
        failures.append(f"affine_descriptor unexpected: {d!r}")
    d = kquant_descriptor({"a": "q6_k", "b": "q6_k", "c": "q8_0"})
    if d != {"kind": "kquant", "codecs": {"q6_k": 2, "q8_0": 1}}:
        failures.append(f"kquant_descriptor unexpected: {d!r}")

    # ---- provenance sidecar: scalars carried through, structures dropped ----
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        recipe = {
            "tool": "some-quantizer",
            "bits": 4,
            "future_flag": True,
            "per_layer": {"0": 4},
        }
        (td_path / "quant-recipe.json").write_text(json.dumps(recipe))
        loaded = load_provenance(td_path)
        if loaded is None:
            failures.append("load_provenance: returned None for existing sidecar")
        else:
            if loaded.get("tool") != "some-quantizer" or loaded.get("bits") != 4:
                failures.append(f"load_provenance: dropped scalar keys: {loaded!r}")
            if loaded.get("future_flag") is not True:
                failures.append(
                    "load_provenance: an unrecognized scalar key must still be "
                    f"carried through, got {loaded!r}"
                )
            if "per_layer" in loaded:
                failures.append(
                    "load_provenance: nested structures must be dropped, got "
                    f"{loaded!r}"
                )
    with tempfile.TemporaryDirectory() as td:
        if load_provenance(Path(td)) is not None:
            failures.append("load_provenance: missing sidecar should return None")

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("All self-tests passed.", file=sys.stderr)
    return 0
