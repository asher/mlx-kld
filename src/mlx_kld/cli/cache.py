"""``mlx-kld cache``: inspect and prune the teacher-logit cache."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from .._constants import DEFAULT_CACHE_DIR


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "cache",
        help="inspect / prune the teacher-logit cache",
        description="Manage the teacher top-K cache. A default-config entry "
        "is ~51.5 GB (512x512 positions at top-k 32768). The score command "
        "auto-evicts least-recently-used entries to --cache-max-gb, and these "
        "operations give manual control. With no <op>, lists entries.",
    )
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help=f"Cache root (default: {DEFAULT_CACHE_DIR})")
    ops = p.add_subparsers(dest="op", metavar="<op>")

    # Accepted before or after the op. SUPPRESS on the op-level copy so an
    # op-parser default can't clobber a value parsed at the group level.
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--cache-dir", type=Path, default=argparse.SUPPRESS,
                        help=f"Cache root (default: {DEFAULT_CACHE_DIR})")

    lst = ops.add_parser("list", parents=[shared],
                         help="table the cache entries (newest-used first)")
    lst.set_defaults(func=_list)

    g = ops.add_parser("gc", parents=[shared],
                       help="prune by age and/or down to a size budget")
    g.add_argument("--max-gb", type=float, default=None,
                   help="Evict least-recently-used entries to fit this budget")
    g.add_argument("--older-than", type=float, default=None, metavar="DAYS",
                   help="Drop entries not used in this many days")
    g.set_defaults(func=_gc)

    c = ops.add_parser("clear", parents=[shared],
                       help="remove all entries (or by teacher substring)")
    c.add_argument("--teacher", default=None,
                   help="Only remove entries whose teacher path contains this")
    c.add_argument("--yes", action="store_true",
                   help="Confirm removing every entry (required without --teacher)")
    c.set_defaults(func=_clear)

    p.set_defaults(func=_default)


def _default(args: argparse.Namespace) -> int:
    # Bare `mlx-kld cache` behaves like `list`.
    return _list(args)


def _list(args: argparse.Namespace) -> int:
    from ..cache import human_bytes, list_entries

    entries = list_entries(args.cache_dir)
    if not entries:
        print(f"cache is empty: {args.cache_dir}")
        return 0
    now = datetime.now(timezone.utc)
    hdr = f"{'key':16} {'size':>10} {'last used':>14}  {'top_k':>7}  teacher"
    print(hdr)
    print("-" * (len(hdr) + 20))
    total = 0
    for e in entries:
        total += e["size_bytes"]
        age_days = (now - e["last_used"]).total_seconds() / 86400
        print(f"{e['key']:16} {human_bytes(e['size_bytes']):>10} "
              f"{age_days:>11.1f} d  {str(e['top_k']):>7}  {e['teacher_path']}")
    print("-" * (len(hdr) + 20))
    print(f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'}, "
          f"{human_bytes(total)} total")
    return 0


def _gc(args: argparse.Namespace) -> int:
    from ..cache import gc, human_bytes

    if args.max_gb is None and args.older_than is None:
        print("error: nothing to do without --max-gb and/or --older-than",
              file=sys.stderr)
        return 1
    if args.max_gb is not None and args.max_gb <= 0:
        print("error: --max-gb must be > 0. Use 'mlx-kld cache clear' to "
              "remove everything", file=sys.stderr)
        return 1
    evicted = gc(args.cache_dir, max_gb=args.max_gb, older_than_days=args.older_than)
    if not evicted:
        print("nothing evicted (already within limits)")
        return 0
    freed = sum(sz for _, sz in evicted)
    for key, sz in evicted:
        print(f"evicted {key} ({human_bytes(sz)})")
    print(f"freed {human_bytes(freed)} across {len(evicted)} entries")
    return 0


def _clear(args: argparse.Namespace) -> int:
    from ..cache import clear_entries, human_bytes, list_entries

    if args.teacher is None and not args.yes:
        entries = list_entries(args.cache_dir)
        total = sum(e["size_bytes"] for e in entries)
        print(
            f"error: would remove {len(entries)} entr"
            f"{'y' if len(entries) == 1 else 'ies'} ({human_bytes(total)}). "
            "Pass --yes to confirm, or --teacher to target one teacher",
            file=sys.stderr,
        )
        return 1
    removed = clear_entries(args.cache_dir, teacher_substr=args.teacher)
    if not removed:
        print("nothing removed")
        return 0
    freed = sum(sz for _, sz in removed)
    print(f"removed {len(removed)} entries, freed {human_bytes(freed)}")
    return 0
