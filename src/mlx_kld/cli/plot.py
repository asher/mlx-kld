"""``mlx-kld plot`` - SVG scatter of mean KLD vs bpw from the results root."""

from __future__ import annotations

import argparse
from pathlib import Path


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "plot",
        help="render an SVG scatter of mean KLD vs bpw (or file size)",
        description="Read JSON records written by `score` and render one "
        "teacher's students as an SVG scatter: X = effective bpw or file "
        "size, Y = mean KLD, marker shape = student format. Unquantized "
        "baselines are excluded. The reconstruction floor is drawn when "
        "available.",
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
        help="Substring filter on teacher.path (required when the root holds "
             "more than one teacher)",
    )
    p.add_argument(
        "--svg", type=Path, default=None,
        help="Output path (default: ./<teacher>-kld-vs-<x>.svg)",
    )
    p.add_argument("--x", choices=("bpw", "size"), default="bpw",
                   help="X axis: effective bits per weight or file size "
                        "(default: bpw)")
    p.add_argument("--log-y", action="store_true",
                   help="Log-scale the KLD axis")
    p.add_argument("--no-labels", action="store_true",
                   help="Suppress per-point student labels")
    p.set_defaults(func=cmd)


def cmd(args: argparse.Namespace) -> int:
    from ..plot import run_plot
    from ..report import resolve_results_dir

    if args.pattern:
        patterns, root_label = args.pattern, None
    else:
        root = resolve_results_dir(args.out_dir)
        patterns, root_label = [str(root / "**" / "*.json")], str(root)
    return run_plot(
        patterns,
        teacher_filter=args.teacher,
        svg_path=args.svg,
        x_axis=args.x,
        log_y=args.log_y,
        labels=not args.no_labels,
        root_label=root_label,
    )
