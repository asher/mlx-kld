"""``mlx-kld compare``: per-teacher comparison tables from the results root."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "compare",
        help="compare scored students per teacher (reads the results root)",
        description="Read JSON records written by `score`, group by teacher, "
        "sort each group by mean KLD ascending, and print a comparison table "
        "(--md/--json write files instead).",
    )
    p.add_argument(
        "--out-dir", type=Path, default=None,
        help="Results root to scan recursively (default: $MLX_KLD_RESULTS "
             "or ./kld-results)",
    )
    p.add_argument(
        "--pattern", action="append", default=None,
        help="Glob for record JSONs (repeatable, overrides --out-dir)",
    )
    p.add_argument(
        "--teacher", default=None,
        help="Substring filter on teacher.path. Emit only matching group(s)",
    )
    p.add_argument("--md", type=Path, default=None,
                   help="Write the table to this path instead of stdout (the "
                        "scan must resolve to one teacher group, so filter "
                        "with --teacher)")
    p.add_argument("--publisher", action="store_true",
                   help="Add a publisher column, read from the ORG__REPO "
                        "directory holding each student. Use when one table "
                        "mixes publishers: two publishers' same-named quants "
                        "can differ in bpw recipe")
    p.add_argument("--scored-bpw", action="store_true",
                   help="Add a bits-per-scored-weight column. This is bits per "
                        "weight over only the weights the scoring pass loads, "
                        "excluding any multi-token-prediction (MTP) stack and "
                        "any vision or audio tower, none of which contribute a "
                        "logit")
    p.add_argument("--json", dest="json_path", type=Path, default=None,
                   help="Write full per-run records to this path (the scan "
                        "must resolve to one teacher group, so filter with "
                        "--teacher)")
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    from ..compare import run_compare
    from ..report import resolve_results_dir

    if args.pattern:
        patterns, root_label = args.pattern, None
    else:
        root = resolve_results_dir(args.out_dir)
        patterns, root_label = [str(root / "**" / "*.json")], str(root)
    return run_compare(
        patterns,
        teacher_filter=args.teacher,
        md_path=args.md,
        json_path=args.json_path,
        root_label=root_label,
        show_publisher=args.publisher,
        show_scored_bpw=args.scored_bpw,
    )
