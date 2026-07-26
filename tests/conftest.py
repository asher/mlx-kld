"""Fixtures shared across the cache-facing test modules.

``make_cache_entry`` was previously a private helper in ``test_cache.py`` that
``test_lifecycle.py`` reached across for. That made a change to one test file's
internals break an unrelated one, and hid the coupling inside a function body
rather than declaring it. A fixture is the supported way to share it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def make_cache_entry():
    """Build a cache entry directory that ``list_entries``/``entry_info`` accept.

    Writes one shard of ``size_bytes`` and a manifest whose ``last_used`` sits
    ``age_days`` in the past, which is what the LRU ordering and the
    ``--older-than`` filters read.
    """

    def _make(root, key, *, size_bytes, age_days,
              teacher="org/teacher", top_k=128):
        d = root / key
        d.mkdir(parents=True)
        (d / "batch-00000.safetensors").write_bytes(b"\0" * size_bytes)
        last_used = (
            datetime.now(timezone.utc) - timedelta(days=age_days)
        ).isoformat()
        (d / "manifest.json").write_text(json.dumps({
            "format_version": 1,
            "teacher_path": teacher,
            "dataset": "wikitext",
            "top_k": top_k,
            "num_samples": 8,
            "max_seq_len": 512,
            "last_used": last_used,
            "created_at": last_used,
        }))
        return d

    return _make
