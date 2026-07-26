"""Minimal stderr progress logging, shared across modules."""

from __future__ import annotations

import sys


def info(msg: str) -> None:
    print(f"[INFO] {msg}", file=sys.stderr, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)
