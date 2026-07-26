"""Durable file writes.

``Path.write_text`` truncates the target and then streams into it, so an
interrupted write (Ctrl-C during a long run, a full disk, a crash) leaves a
truncated file where a valid one used to be. That matters twice here: a
half-written cache manifest invalidates a ~51.5 GB entry, and a half-written
result record is indistinguishable from a real one until something tries to
parse it. Every write goes through a temp file in the destination directory and
an ``os.replace``, which is atomic on the same filesystem, so a reader sees the
old file or the new one and never a partial one.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # fsync the parent directory so the rename itself survives a crash;
        # best-effort, since not every filesystem allows fsync on a directory.
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_json_atomic(path: Path, payload: Any, indent: int = 2) -> None:
    """Write ``payload`` as pretty JSON with a trailing newline.

    The newline is not cosmetic: it makes records well-formed text files, so
    ``cat``, ``tail``, diff tools, and shell pipelines behave.
    """
    write_text_atomic(path, json.dumps(payload, indent=indent) + "\n")
