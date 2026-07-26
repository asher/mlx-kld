"""Tests for the self-managing teacher cache (no model load)."""

import json
import shutil
from datetime import datetime, timedelta

from mlx_kld import cache as C


def test_list_entries_newest_used_first(tmp_path, make_cache_entry):
    make_cache_entry(tmp_path, "aaa", size_bytes=100, age_days=10)
    make_cache_entry(tmp_path, "bbb", size_bytes=100, age_days=1)
    make_cache_entry(tmp_path, "ccc", size_bytes=100, age_days=5)
    keys = [e["key"] for e in C.list_entries(tmp_path)]
    assert keys == ["bbb", "ccc", "aaa"]  # newest last_used first


def test_enforce_budget_evicts_oldest_first(tmp_path, make_cache_entry):
    make_cache_entry(tmp_path, "old", size_bytes=3000, age_days=10)
    make_cache_entry(tmp_path, "mid", size_bytes=2000, age_days=5)
    make_cache_entry(tmp_path, "new", size_bytes=1000, age_days=1)
    # Budget ~4000 bytes: must drop the oldest (3000) to get under it.
    evicted = C.enforce_budget(tmp_path, max_gb=4000 / 1e9)
    assert [k for k, _ in evicted] == ["old"]
    assert not (tmp_path / "old").exists()
    assert (tmp_path / "mid").exists() and (tmp_path / "new").exists()


def test_enforce_budget_never_evicts_protected(tmp_path, make_cache_entry):
    make_cache_entry(tmp_path, "keepme", size_bytes=3000, age_days=99)  # oldest
    make_cache_entry(tmp_path, "b", size_bytes=1000, age_days=5)
    make_cache_entry(tmp_path, "c", size_bytes=1000, age_days=1)
    evicted = {k for k, _ in C.enforce_budget(tmp_path, max_gb=1e-9, protect_keys=("keepme",))}
    assert evicted == {"b", "c"}
    assert (tmp_path / "keepme").exists()


def test_enforce_budget_disabled_when_zero(tmp_path, make_cache_entry):
    make_cache_entry(tmp_path, "a", size_bytes=1000, age_days=1)
    assert C.enforce_budget(tmp_path, max_gb=0) == []
    assert (tmp_path / "a").exists()


def test_gc_older_than(tmp_path, make_cache_entry):
    make_cache_entry(tmp_path, "fresh", size_bytes=100, age_days=0)
    make_cache_entry(tmp_path, "stale", size_bytes=100, age_days=30)
    evicted = {k for k, _ in C.gc(tmp_path, older_than_days=7)}
    assert evicted == {"stale"}
    assert (tmp_path / "fresh").exists()


def test_clear_by_teacher_substring(tmp_path, make_cache_entry):
    make_cache_entry(tmp_path, "x", size_bytes=100, age_days=1, teacher="org/gemma")
    make_cache_entry(tmp_path, "y", size_bytes=100, age_days=1, teacher="org/qwen")
    removed = {k for k, _ in C.clear_entries(tmp_path, teacher_substr="gemma")}
    assert removed == {"x"}
    assert (tmp_path / "y").exists()


def test_touch_entry_updates_last_used(tmp_path, make_cache_entry):
    d = make_cache_entry(tmp_path, "z", size_bytes=100, age_days=30)
    before = C.entry_info(d)["last_used"]
    C.touch_entry(d)
    after = C.entry_info(d)["last_used"]
    assert after > before


# ---------- cross-process coordination ----------

def test_eviction_skips_an_entry_whose_lock_is_held(tmp_path, make_cache_entry):
    """A concurrent run's replay holds the entry lock shared; every eviction
    path must leave that entry alone rather than delete shards mid-read.
    flock treats separate file descriptors as separate holders, so holding the
    lock in-process stands in for a second process."""
    make_cache_entry(tmp_path, "busy", size_bytes=3000, age_days=10)
    make_cache_entry(tmp_path, "idle", size_bytes=3000, age_days=5)
    with C.entry_lock(tmp_path, "busy", exclusive=False):
        evicted = {k for k, _ in C.enforce_budget(tmp_path, max_gb=1e-9)}
        assert evicted == {"idle"}
        assert (tmp_path / "busy").exists()
    # Lock released: the entry is an ordinary eviction candidate again.
    evicted = {k for k, _ in C.enforce_budget(tmp_path, max_gb=1e-9)}
    assert evicted == {"busy"}


def test_gc_and_clear_skip_locked_entries(tmp_path, make_cache_entry):
    make_cache_entry(tmp_path, "busy", size_bytes=100, age_days=30)
    with C.entry_lock(tmp_path, "busy", exclusive=True):
        assert C.gc(tmp_path, older_than_days=7) == []
        assert C.clear_entries(tmp_path) == []
        assert (tmp_path / "busy").exists()
    removed = {k for k, _ in C.clear_entries(tmp_path)}
    assert removed == {"busy"}


def test_removal_leaves_the_lock_file_in_place(tmp_path, make_cache_entry):
    """Lock files must outlive the entries they guard. flock binds to an inode:
    unlinking one while a process is blocked in flock() hands that process a
    lock on an orphan while the next arrival creates a fresh file at the same
    path and locks that, leaving two exclusive holders of one key. The stray is
    0-byte and invisible to list_entries, which is the cheaper trade."""
    make_cache_entry(tmp_path, "gone", size_bytes=100, age_days=30)
    with C.entry_lock(tmp_path, "gone", exclusive=False):
        pass
    assert C.clear_entries(tmp_path)
    assert not (tmp_path / "gone").exists()
    assert (tmp_path / ".gone.lock").exists()
    # ...and a stray lock file is not mistaken for a cache entry.
    assert C.list_entries(tmp_path) == []
    assert C.cache_total_bytes(tmp_path) == 0


def test_an_already_removed_entry_is_not_counted_as_freed(tmp_path, make_cache_entry):
    """enforce_budget subtracts what _try_remove_entry reports, and the CLI
    prints it as bytes freed; claiming another process's eviction would make
    both wrong."""
    e = make_cache_entry(tmp_path, "racy", size_bytes=100, age_days=1)
    info = C.entry_info(e)
    shutil.rmtree(e)  # a concurrent evictor got there first
    assert C._try_remove_entry(tmp_path, info) is False


def test_naive_last_used_timestamps_do_not_crash_listing(tmp_path, make_cache_entry):
    """Hand-built manifests may carry naive ISO timestamps; mixing them with
    aware ones must not raise from the sort or from age arithmetic."""

    d = tmp_path / "naive"
    d.mkdir()
    naive = (datetime.now() - timedelta(days=3)).isoformat()  # no tzinfo
    (d / "manifest.json").write_text(json.dumps({
        "format_version": 1, "teacher_path": "org/t", "dataset": "x",
        "top_k": 8, "num_samples": 8, "max_seq_len": 512,
        "last_used": naive, "created_at": naive,
    }))
    make_cache_entry(tmp_path, "aware", size_bytes=100, age_days=1)
    entries = C.list_entries(tmp_path)
    assert [e["key"] for e in entries] == ["aware", "naive"]
    assert all(e["last_used"].tzinfo is not None for e in entries)
    # Age math (what `cache list` and gc do) must also work.
    assert C.gc(tmp_path, older_than_days=2) and not d.exists()


def test_cache_key_resolves_local_teacher_path_spellings(tmp_path, monkeypatch):
    """./teacher, teacher/ and the absolute spelling must key one entry, not
    mint a ~51.5 GB duplicate each. Hub ids pass through untouched."""
    teacher = tmp_path / "teacher"
    teacher.mkdir()
    monkeypatch.chdir(tmp_path)
    spellings = ["teacher", "teacher/", "./teacher", str(teacher)]
    keys = {C.cache_key(s, "corpus/x", 512, 512, 123, 32768) for s in spellings}
    assert len(keys) == 1
    assert C.cache_key("org/teacher", "corpus/x", 512, 512, 123, 32768) not in keys


# ---------- measured teacher precision ----------

def test_measured_precision_reports_the_dominant_weight_dtype():
    """`teacher.precision` in a published record has to describe what was
    actually loaded, not whatever the command-line label defaulted to."""
    import mlx.core as mx

    from mlx_kld.cache import measured_precision

    class FakeModel:
        def parameters(self):
            return {
                "big": mx.zeros((64, 64), dtype=mx.bfloat16),
                "small": mx.zeros((2, 2), dtype=mx.float32),
            }

    assert measured_precision(FakeModel()) == "bfloat16"


def test_measured_precision_returns_none_rather_than_raising():
    """Precision is provenance, never worth failing a teacher pass over."""
    from mlx_kld.cache import measured_precision

    class Broken:
        def parameters(self):
            raise RuntimeError("no parameter tree")

    assert measured_precision(Broken()) is None


def test_measured_precision_returns_none_on_an_empty_tree():
    from mlx_kld.cache import measured_precision

    class Empty:
        def parameters(self):
            return {}

    assert measured_precision(Empty()) is None
