"""``teacher_pass`` and ``cache_key``, driven by a fake teacher.

The teacher pass writes the ~51.5 GB artifact every score replays against, so
its manifest is a contract: get ``top_k``, ``vocab_size``, or
``corpus_tokens_hash`` wrong and every downstream number is wrong in a way no
later check catches. A stand-in model exercises all of it in milliseconds.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from mlx_kld import cache as C


class _FakeTeacher:
    """Returns deterministic logits of a fixed width, ignoring the KV cache."""

    def __init__(self, vocab: int):
        self.vocab = vocab
        self.calls = 0

    def eval(self):
        pass

    def __call__(self, batch, cache=None):
        self.calls += 1
        b, seq = batch.shape
        # Rank the vocab differently per position so top-K ordering is non-trivial.
        base = mx.arange(self.vocab, dtype=mx.float32)
        pos = mx.arange(seq, dtype=mx.float32)[:, None]
        return mx.broadcast_to((base + pos) % self.vocab, (b, seq, self.vocab))


@pytest.fixture
def fake_teacher(monkeypatch):
    """Install a fake model loader and stub out the hub revision lookup."""
    holder = {}

    def fake_load_model(path, lazy=True):
        model = _FakeTeacher(holder["vocab"])
        holder["model"] = model
        return model, {"vocab_size": holder.get("config_vocab", holder["vocab"])}

    monkeypatch.setattr(C, "_load_model", fake_load_model)
    monkeypatch.setattr(C, "_make_fresh_cache", lambda model: None)
    # Never touch the network from a unit test.
    monkeypatch.setattr(C, "hf_revision_of", lambda p: None)
    return holder


def _run(fake_teacher, tmp_path, *, vocab=64, top_k=8, n=4, seq=8, batch_size=2,
         config_vocab=None, score_window=None):
    fake_teacher["vocab"] = vocab
    if config_vocab is not None:
        fake_teacher["config_vocab"] = config_vocab
    tokens = mx.array(
        [[(i * seq + j) % vocab for j in range(seq)] for i in range(n)],
        dtype=mx.int32,
    )
    manifest = C.teacher_pass(
        "org/teacher", tokens, batch_size, top_k, tmp_path,
        "corpus/x", n, seq, 123, "tokhash", score_window=score_window,
    )
    return manifest, tokens


# ---------- top-K clamping ----------

def test_top_k_is_reclamped_against_the_measured_logits_width(fake_teacher, tmp_path):
    """The CLI clamps against the tokenizer's vocab_size, which is only a proxy.
    A tokenizer that over-reports would otherwise reach argpartition with
    kth >= V and die inside MLX."""
    manifest, _ = _run(fake_teacher, tmp_path, vocab=32, top_k=100, config_vocab=99999)
    assert manifest["vocab_size"] == 32
    assert manifest["top_k"] == 31
    shard = mx.load(str(tmp_path / "batch-00000.safetensors"))
    assert shard["top_k_indices"].shape[-1] == 31


def test_a_top_k_within_the_width_is_left_alone(fake_teacher, tmp_path):
    manifest, _ = _run(fake_teacher, tmp_path, vocab=64, top_k=8)
    assert manifest["top_k"] == 8
    shard = mx.load(str(tmp_path / "batch-00000.safetensors"))
    assert shard["top_k_indices"].shape[-1] == 8


def test_the_clamped_top_k_is_what_lands_in_the_manifest(fake_teacher, tmp_path):
    """Not the requested one. The manifest drives the student pass's math."""
    manifest, _ = _run(fake_teacher, tmp_path, vocab=16, top_k=64)
    assert manifest["top_k"] == 15
    assert json.loads((tmp_path / "manifest.json").read_text())["top_k"] == 15


# ---------- shard contents ----------

def test_shards_carry_every_key_the_student_pass_reads(fake_teacher, tmp_path):
    _run(fake_teacher, tmp_path, n=4, batch_size=2)
    shard = mx.load(str(tmp_path / "batch-00000.safetensors"))
    assert set(shard) == {"top_k_log_softmax", "top_k_indices", "token_ids",
                          "attention_mask", "floor_kld"}
    assert shard["top_k_log_softmax"].dtype == mx.bfloat16
    assert shard["top_k_indices"].dtype == mx.int32


def test_top_k_log_probs_are_sorted_descending(fake_teacher, tmp_path):
    """The student pass and delta_p both read index 0 as the teacher's argmax."""
    _run(fake_teacher, tmp_path, vocab=64, top_k=8)
    lp = mx.load(str(tmp_path / "batch-00000.safetensors"))["top_k_log_softmax"]
    diffs = lp[..., 1:].astype(mx.float32) - lp[..., :-1].astype(mx.float32)
    assert float(mx.max(diffs).item()) <= 0.0


def test_reconstruction_floor_is_recorded_and_non_negative(fake_teacher, tmp_path):
    _run(fake_teacher, tmp_path, vocab=64, top_k=8)
    floor = mx.load(str(tmp_path / "batch-00000.safetensors"))["floor_kld"]
    assert floor.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(floor)).item())
    assert float(mx.min(floor).item()) >= -1e-6


def test_batching_covers_every_sequence(fake_teacher, tmp_path):
    manifest, tokens = _run(fake_teacher, tmp_path, n=5, batch_size=2)
    assert manifest["num_batches"] == 3
    seen = mx.concatenate([
        mx.load(str(tmp_path / f"batch-{i:05d}.safetensors"))["token_ids"]
        for i in range(3)
    ])
    assert seen.shape == tokens.shape
    assert bool(mx.all(seen == tokens).item())


# ---------- manifest as a contract ----------

def test_corpus_tokens_hash_witnesses_the_exact_stream(fake_teacher, tmp_path):
    """This hash is how a later student learns that a cache HIT replayed
    someone else's tokenization, so it must move iff the ids move."""
    fake_teacher["vocab"] = 64

    def run(tokens, sub):
        return C.teacher_pass(
            "org/teacher", mx.array(tokens, dtype=mx.int32), 2, 8,
            tmp_path / sub, "corpus/x", len(tokens), len(tokens[0]), 123,
            "tokhash",
        )["corpus_tokens_hash"]

    base = [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert run(base, "a") == run([row[:] for row in base], "b")
    # One id different anywhere.
    assert run([[1, 2, 3, 4], [5, 6, 7, 9]], "c") != run(base, "d")
    # Same ids, different order.
    assert run([[5, 6, 7, 8], [1, 2, 3, 4]], "e") != run(base, "f")


def test_manifest_records_both_measured_and_config_vocab(fake_teacher, tmp_path):
    """They disagree exactly when a padded lm_head or an over-reporting config
    is in play, and the report needs to be able to say which."""
    manifest, _ = _run(fake_teacher, tmp_path, vocab=32, top_k=8, config_vocab=48)
    assert manifest["vocab_size"] == 32
    assert manifest["config_vocab_size"] == 48


def test_score_window_defaults_to_the_full_sequence(fake_teacher, tmp_path):
    manifest, _ = _run(fake_teacher, tmp_path, seq=8, score_window=None)
    assert manifest["score_window"] == [0, 8]


def test_score_window_is_recorded_when_given(fake_teacher, tmp_path):
    manifest, _ = _run(fake_teacher, tmp_path, seq=8, score_window=(4, 8))
    assert manifest["score_window"] == [4, 8]


def test_size_bytes_is_written_after_the_shards_exist(fake_teacher, tmp_path):
    """The LRU budget evicts on this number, so it has to be the whole entry,
    not just what existed when the manifest was first drafted."""
    _run(fake_teacher, tmp_path)
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    shard_bytes = sum(f.stat().st_size for f in tmp_path.glob("batch-*.safetensors"))
    assert shard_bytes > 0
    assert on_disk["size_bytes"] >= shard_bytes


def test_a_finished_pass_validates_as_a_cache_hit(fake_teacher, tmp_path):
    manifest, _ = _run(fake_teacher, tmp_path, n=5, batch_size=2)
    ok, why = C.cache_is_valid(tmp_path, manifest["num_batches"])
    assert ok, why


def test_a_missing_shard_invalidates_the_entry(fake_teacher, tmp_path):
    manifest, _ = _run(fake_teacher, tmp_path, n=5, batch_size=2)
    (tmp_path / "batch-00001.safetensors").unlink()
    ok, why = C.cache_is_valid(tmp_path, manifest["num_batches"])
    assert not ok and "batch-00001" in why


# ---------- cache_key ----------

_KEY_ARGS = ("org/teacher", "corpus/x", 512, 512, 123, 32768)


def test_cache_key_is_stable_for_identical_specs():
    assert C.cache_key(*_KEY_ARGS) == C.cache_key(*_KEY_ARGS)


@pytest.mark.parametrize("pos,value", [
    (0, "org/other-teacher"), (1, "corpus/y"), (2, 256), (3, 1024),
    (4, 999), (5, 128),
])
def test_every_cache_key_input_changes_the_key(pos, value):
    args = list(_KEY_ARGS)
    args[pos] = value
    assert C.cache_key(*args) != C.cache_key(*_KEY_ARGS)


def test_score_window_is_deliberately_absent_from_the_cache_key(fake_teacher, tmp_path):
    """Documented in cache.py's module docstring: every position is cached, so
    --score-window re-windows an existing entry without a teacher rebuild. The
    run digest carries the window instead, so records never blend across it."""
    import inspect

    assert "score_window" not in inspect.signature(C.cache_key).parameters
    a, _ = _run(fake_teacher, tmp_path, seq=8, score_window=(0, 8))
    b, _ = _run(fake_teacher, tmp_path / "second", seq=8, score_window=(4, 8))
    assert a["score_window"] != b["score_window"]
    assert a["corpus_tokens_hash"] == b["corpus_tokens_hash"]


# ---------- entry size estimate ----------

def test_entry_size_estimate_tracks_the_real_thing(fake_teacher, tmp_path):
    """`score` refuses to start when the estimate blows the cache budget, so an
    estimate that drifts low would let a run fill the disk."""
    n, seq, top_k = 4, 8, 8
    _run(fake_teacher, tmp_path, n=n, seq=seq, top_k=top_k, batch_size=n)
    actual = sum(f.stat().st_size for f in tmp_path.glob("batch-*.safetensors"))
    estimate = C.estimate_entry_bytes(n, seq, top_k)
    assert 0.5 * actual <= estimate <= 2.0 * actual
