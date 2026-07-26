"""Durable writes: atomicity, parent creation, and the trailing newline."""

from __future__ import annotations

import json

import pytest

from mlx_kld._io import write_json_atomic, write_text_atomic


def test_creates_missing_parent_directories(tmp_path):
    """`score --md reports/nested/x.md` must not fail on a directory the user
    hasn't made yet; the JSON record path already relied on this."""
    target = tmp_path / "a" / "b" / "report.md"
    write_text_atomic(target, "# hi\n")
    assert target.read_text() == "# hi\n"


def test_failed_write_leaves_the_previous_file_intact(tmp_path):
    """The point of the temp-file dance: a crash mid-write must not truncate a
    good file into a broken one."""
    target = tmp_path / "manifest.json"
    write_json_atomic(target, {"num_batches": 512})

    class Exploding:
        def __repr__(self):  # pragma: no cover - json never gets this far
            return "boom"

    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": Exploding()})
    assert json.loads(target.read_text()) == {"num_batches": 512}


def test_no_temp_files_are_left_behind(tmp_path):
    target = tmp_path / "out.json"
    write_json_atomic(target, {"ok": True})
    with pytest.raises(TypeError):
        write_json_atomic(target, {"bad": object()})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out.json"]


def test_json_records_end_with_a_newline(tmp_path):
    target = tmp_path / "record.json"
    write_json_atomic(target, {"schema_version": 1})
    raw = target.read_text()
    assert raw.endswith("\n")
    assert json.loads(raw) == {"schema_version": 1}


def test_overwrite_replaces_rather_than_appends(tmp_path):
    target = tmp_path / "x.txt"
    write_text_atomic(target, "a long first version\n")
    write_text_atomic(target, "short\n")
    assert target.read_text() == "short\n"
