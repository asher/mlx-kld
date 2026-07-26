"""The shared cache lifecycle (``ensure_cache_entry``) and the public
``ensure_teacher_topk_cache`` wrapper, driven by a fake teacher.

Both the CLI and the Python API resolve HIT/MISS/rebuild, the tokenizer-hash
gate, budget make-room, and locking through this one path, so its behavior is
the contract for every entry the tool ever builds.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from mlx_kld import cache as C
from mlx_kld.errors import CacheMismatchError


@pytest.fixture
def fake_teacher(monkeypatch):
    holder = {"vocab": 64}

    def fake_load_model(path, lazy=True):
        class _M:
            def eval(self):
                pass

            def __call__(self, batch, cache=None):
                b, seq = batch.shape
                base = mx.arange(holder["vocab"], dtype=mx.float32)
                pos = mx.arange(seq, dtype=mx.float32)[:, None]
                return mx.broadcast_to(
                    (base + pos) % holder["vocab"], (b, seq, holder["vocab"])
                )

        return _M(), {"vocab_size": holder["vocab"]}

    monkeypatch.setattr(C, "_load_model", fake_load_model)
    monkeypatch.setattr(C, "_make_fresh_cache", lambda m: None)
    monkeypatch.setattr(C, "hf_revision_of", lambda p: None)
    return holder


def _tokens(n=4, seq=8, vocab=64):
    return mx.array(
        [[(i * seq + j) % vocab for j in range(seq)] for i in range(n)],
        dtype=mx.int32,
    )


def _ensure(root, *, rebuild=False, tok_hash="tokhash", cache_max_gb=0.0,
            top_k=8, n=4, seq=8):
    return C.ensure_cache_entry(
        teacher_path="org/teacher",
        tokens=_tokens(n=n, seq=seq),
        teacher_tok_hash=tok_hash,
        dataset_name="corpus/x",
        num_samples=n,
        max_seq_len=seq,
        seed=123,
        top_k=top_k,
        batch_size=2,
        cache_root=root,
        rebuild=rebuild,
        score_window=(seq // 2, seq),
        cache_max_gb=cache_max_gb,
    )


def test_miss_builds_then_hit_reuses(fake_teacher, tmp_path):
    d1, m1, s1, secs1 = _ensure(tmp_path)
    assert s1 == "MISS"
    assert secs1 > 0  # the build was timed
    assert C.cache_is_valid(d1, m1["num_batches"]) == (True, "ok")
    d2, m2, s2, secs2 = _ensure(tmp_path)
    assert s2 == "HIT"
    assert secs2 == 0.0  # a HIT does no teacher pass
    assert d2 == d1
    assert m2["corpus_tokens_hash"] == m1["corpus_tokens_hash"]


def test_rebuild_rebuilds_a_valid_entry(fake_teacher, tmp_path):
    _ensure(tmp_path)
    _, _, status, _secs = _ensure(tmp_path, rebuild=True)
    assert status == "REBUILD"


def test_hit_overlays_this_runs_score_window(fake_teacher, tmp_path):
    d, m, _s, _secs = _ensure(tmp_path, seq=8)
    assert m["score_window"] == [4, 8]
    # Same key, different window: the entry replays, the window follows the run.
    _, m2, s2, _secs2 = C.ensure_cache_entry(
        teacher_path="org/teacher", tokens=_tokens(), teacher_tok_hash="tokhash",
        dataset_name="corpus/x", num_samples=4, max_seq_len=8, seed=123,
        top_k=8, batch_size=2, cache_root=tmp_path, score_window=(0, 8),
    )
    assert s2 == "HIT"
    assert m2["score_window"] == [0, 8]
    # The on-disk manifest keeps the build-time window; only the copy changes.
    on_disk = json.loads((d / "manifest.json").read_text())
    assert on_disk["score_window"] == [4, 8]


def test_changed_teacher_tokenizer_raises_cache_mismatch(fake_teacher, tmp_path):
    _ensure(tmp_path, tok_hash="tokhash-a")
    with pytest.raises(CacheMismatchError, match="tokenizer_hash"):
        _ensure(tmp_path, tok_hash="tokhash-b")


def test_build_makes_room_before_the_teacher_pass(
    fake_teacher, tmp_path, make_cache_entry
):
    """Eviction must run down to (budget - estimate) *before* the build, so the
    root peaks near the budget instead of budget + one entry."""
    make_cache_entry(tmp_path, "filler", size_bytes=3000, age_days=10)
    est = C.estimate_entry_bytes(4, 8, 8)
    budget_gb = (est + 1500) / 1e9  # filler cannot coexist with the new entry
    d, _, status, _secs = _ensure(tmp_path, cache_max_gb=budget_gb)
    assert status == "MISS"
    assert not (tmp_path / "filler").exists()
    assert d.exists()


def test_single_entry_over_budget_warns_but_builds(fake_teacher, tmp_path, capsys):
    _, _, status, _secs = _ensure(tmp_path, cache_max_gb=1e-9)
    assert status == "MISS"
    assert "exceeds the cache budget" in capsys.readouterr().err


def test_public_wrapper_returns_the_teacher_tokenizer(fake_teacher, tmp_path, monkeypatch):
    class _FakeTok:
        vocab_size = 3
        bos_token_id = eos_token_id = pad_token_id = unk_token_id = None

        def __len__(self):
            return 3

        def convert_ids_to_tokens(self, i):
            return f"<{i}>"

        def encode(self, s, add_special_tokens=False):
            return [ord(c) % 3 for c in s]

    tok = _FakeTok()
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda p: tok)
    monkeypatch.setattr(C, "load_calibration_tokens",
                        lambda *a, **k: _tokens(n=4, seq=8))
    cache_dir, manifest, returned_tok = C.ensure_teacher_topk_cache(
        teacher_path="org/teacher", dataset_name="corpus/x",
        num_samples=4, max_seq_len=8, seed=123, batch_size=2, top_k=8,
        cache_root=tmp_path,
    )
    assert returned_tok is tok
    assert manifest["top_k"] == 8
    assert C.cache_is_valid(cache_dir, manifest["num_batches"]) == (True, "ok")


def test_a_failed_build_leaves_no_unreclaimable_directory(fake_teacher, tmp_path, monkeypatch):
    """A manifest-less entry is rejected by cache_is_valid AND invisible to
    entry_info, so leaving one behind would strand up to a full entry of disk
    that no eviction path, not even `cache clear`, can reclaim."""
    def boom(*a, **k):
        raise RuntimeError("teacher exploded mid-pass")

    monkeypatch.setattr(C, "teacher_pass", boom)
    with pytest.raises(RuntimeError, match="exploded"):
        _ensure(tmp_path)
    assert C.list_entries(tmp_path) == []
    # The directory itself is gone, not merely unlisted.
    assert [p for p in tmp_path.iterdir() if p.is_dir()] == []


def test_keyboard_interrupt_mid_build_also_cleans_up(fake_teacher, tmp_path, monkeypatch):
    def interrupt(*a, **k):
        raise KeyboardInterrupt

    monkeypatch.setattr(C, "teacher_pass", interrupt)
    with pytest.raises(KeyboardInterrupt):
        _ensure(tmp_path)
    assert [p for p in tmp_path.iterdir() if p.is_dir()] == []
