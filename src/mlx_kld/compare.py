"""Aggregate JSON records from the results root into per-teacher comparison
tables.

Groups runs by teacher and sorts each group by mean KLD ascending. The table
prints to stdout by default (``--md``/``--json`` write files instead). Only
records that validate against the ``schema_version=1`` locked schema are read;
anything else is skipped with a warning.
"""

from __future__ import annotations

import glob
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._constants import SCHEMA_VERSION
from ._io import write_json_atomic, write_text_atomic
from ._log import info, warn
from .errors import RecordSchemaError
from .report import FLOOR_CAUTION_MULT, _fmt_stat, floor_limited, validate_locked_schema


def _run_key(d: dict) -> tuple:
    """What makes two runs directly comparable: the calibration spec plus
    top-K, score window, and the scored-token content hash. The hash splits
    runs whose spec matches but whose corpus drifted (upstream dataset
    revision, tokenizer change); when present it also subsumes tokenizer mode,
    since an identical scored stream is comparable no matter how it was
    produced (an hf-source GGUF run and an mlx-student run land in one
    ranking). Only pre-hash records fall back to splitting on tokenizer
    mode."""
    c = d.get("calibration", {})
    window = c.get("score_window")
    tokh = c.get("corpus_tokens_hash")
    tok = None if tokh else (d.get("tokenizer") or {}).get("mode")
    # score_window is not part of the locked schema's type checks, so a
    # hand-edited record can carry a scalar here. Group those together as
    # "unspecified" rather than raising out of a whole compare invocation.
    window = tuple(window) if isinstance(window, (list, tuple)) else None
    return (
        c.get("corpus"), c.get("num_samples"), c.get("max_seq_len"),
        c.get("seed"), c.get("top_k"),
        window,
        tok,
        tokh,
    )


def _tok_modes(runs: list[dict]) -> str:
    modes = sorted({(r.get("tokenizer") or {}).get("mode") or "n/a" for r in runs})
    return ", ".join(modes)


def _mean_or_inf(r: dict) -> float:
    m = r["kld"].get("mean")
    return m if m is not None else float("inf")


def load_runs(patterns: str | list[str]) -> list[dict]:
    """Walk pattern(s), parse JSON, drop records that don't validate against
    the locked schema. De-duplicates by (student.path, teacher.path, run key)."""
    if isinstance(patterns, str):
        patterns = [patterns]
    seen_paths: set[str] = set()
    paths: list[str] = []
    for pat in patterns:
        for p in sorted(glob.glob(os.path.expanduser(pat), recursive=True)):
            if p in seen_paths:
                continue
            seen_paths.add(p)
            paths.append(p)
    runs: list[dict] = []
    seen_keys: set[tuple] = set()
    for p in paths:
        try:
            with open(p) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warn(f"skipping {p}: {e}")
            continue
        # Valid JSON need not be a JSON *object*: a stray list or bare string
        # under the results root would otherwise take out the invocation on the
        # first attribute access below.
        if not isinstance(d, dict):
            warn(f"skipping {p}: not a JSON object ({type(d).__name__})")
            continue
        if d.get("schema_version") != SCHEMA_VERSION:
            warn(
                f"skipping {p}: schema_version {d.get('schema_version')!r} "
                f"(want {SCHEMA_VERSION})"
            )
            continue
        # A stray or truncated file must not poison the whole invocation.
        try:
            validate_locked_schema(d)
        except RecordSchemaError as e:
            warn(f"skipping {p}: {e}")
            continue
        key = (
            d.get("student", {}).get("path"),
            d.get("teacher", {}).get("path"),
            _run_key(d),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        d["_source_path"] = p
        runs.append(d)
    return runs


def group_by_teacher(runs: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for r in runs:
        groups.setdefault(r["teacher"]["path"], []).append(r)
    return groups


def _fmt(v: Any, kind: str = "auto") -> str:
    """Render a value for the MD table. Missing/None renders as n/a."""
    if v is None:
        return "n/a"
    if kind == "gb":
        return f"{v / 1e9:.2f} GB"
    if kind == "pct":
        return f"{v * 100:.2f}%"
    if kind == "kld":
        # One formatter for KLD values everywhere: report._fmt_stat switches to
        # scientific when fixed precision would floor a nonzero value to 0.
        return _fmt_stat(v, 4)
    if kind == "bpw":
        return f"{v:.3f}"
    return str(v)


def _publisher(student_path: str) -> str | None:
    """Publisher (HF org) of a student, recovered from the directory that holds
    it: the parent directory for a ``.gguf`` file, the checkpoint directory
    itself for MLX. Local mirrors are named ``ORG__REPO``, so the org is the
    part before the separator. None when the layout carries no org, which keeps
    this a display aid rather than something callers must satisfy.

    Worth surfacing because the table labels rows by file name alone, and two
    publishers ship the same quant name at different bpw recipes (an unsloth
    ``Q5_K_M`` is not a bartowski ``Q5_K_M``).
    """
    p = Path(student_path)
    container = p.parent if p.suffix.lower() == ".gguf" else p
    name = container.name
    return name.split("__", 1)[0] if "__" in name else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _window_str(window) -> str:
    if not window:
        return "n/a"
    return f"{window[0]}:{window[1]}"


def render_md(
    teacher_path: str,
    runs: list[dict],
    show_publisher: bool = False,
    show_scored_bpw: bool = False,
) -> str:
    """One Markdown comparison table, sorted by mean KLD ascending.

    ``show_publisher`` adds a publisher column (see ``_publisher``) and
    ``show_scored_bpw`` a bits-per-scored-weight column; both off by default so
    the table stays narrow. Records written before those fields existed render
    as n/a rather than dropping out.
    """
    out: list[str] = []
    teacher = runs[0]["teacher"]
    out.append(f"# KLD comparison: `{Path(teacher_path).name}`")
    out.append("")
    out.append(f"- teacher: `{teacher_path}`")
    out.append(f"- precision: `{teacher.get('precision')}`")
    out.append(f"- runs: **{len(runs)}**")
    out.append(f"- generated: {_now_iso()}")
    out.append("")

    by_key: dict[tuple, list[dict]] = {}
    for r in runs:
        by_key.setdefault(_run_key(r), []).append(r)
    cals = sorted(by_key, key=lambda k: tuple(str(x) for x in k))
    multi_cal = len(cals) > 1
    if not multi_cal:
        corpus, n, sl, seed, top_k, window, _tok, tokh = cals[0]
        out.append(
            f"All runs share one spec: corpus=`{corpus}`, num_samples={n}, "
            f"max_seq_len={sl}, seed={seed}, top_k={top_k}, "
            f"window={_window_str(window)}, tokenizer={_tok_modes(runs)}, "
            f"tokens={tokh or 'n/a'}."
        )
    else:
        out.append(
            f"**Mixed run specs ({len(cals)} distinct): KLD numbers across "
            f"spec boundaries are not directly comparable.** "
            f"The `spec` column groups rows. See the legend below."
        )
    out.append("")

    cal_index = {k: i + 1 for i, k in enumerate(cals)}
    runs_sorted = sorted(runs, key=_mean_or_inf)

    headers = [("student", "---")]
    if show_publisher:
        headers.append(("publisher", "---"))
    headers += [("format", "---"), ("size", "---:"), ("bpw", "---:")]
    if show_scored_bpw:
        headers.append(("scored bpw", "---:"))
    headers += [
        ("mean KLD +/- se", "---:"), ("p99", "---:"), ("top-1", "---:"),
    ]
    if multi_cal:
        headers.append(("spec", "---"))
    out.append("| " + " | ".join(h for h, _ in headers) + " |")
    out.append("|" + "|".join(a for _, a in headers) + "|")

    any_floor_limited = False
    for r in runs_sorted:
        s = r["student"]
        kld = r["kld"]
        ag = r["agreement"]
        mean_cell = _fmt(kld.get("mean"), "kld")
        if kld.get("se") is not None:
            mean_cell += f" +/- {_fmt(kld['se'], 'kld')}"
        if floor_limited(kld):
            mean_cell += " (f)"
            any_floor_limited = True
        row = [f"`{Path(s['path']).name}`"]
        if show_publisher:
            row.append(_fmt(_publisher(s.get("path") or "")))
        row += [
            _fmt(s.get("format")),
            _fmt(s.get("size_bytes"), "gb"),
            _fmt(s.get("effective_bpw"), "bpw"),
        ]
        if show_scored_bpw:
            row.append(_fmt(s.get("scored_bpw"), "bpw"))
        row += [
            mean_cell,
            _fmt(kld.get("p99"), "kld"),
            _fmt(ag.get("top1"), "pct"),
        ]
        if multi_cal:
            row.append(f"#{cal_index[_run_key(r)]}")
        out.append("| " + " | ".join(row) + " |")
    out.append("")
    if any_floor_limited:
        out.append(
            f"(f) floor-limited: mean KLD within {FLOOR_CAUTION_MULT:g}x of the "
            "top-K reconstruction floor, so much of the absolute value is "
            "measurement floor. These rows still rank correctly against their "
            "siblings, which carry the same floor."
        )
        out.append("")

    if multi_cal:
        out.append("## Spec legend")
        out.append("")
        out.append("| ID | corpus | samples | seq_len | seed | top_k | window "
                   "| tokenizer | tokens |")
        out.append("|---|---|---:|---:|---:|---:|---|---|---|")
        for key, idx in cal_index.items():
            corpus, n, sl, seed, top_k, window, _tok, tokh = key
            out.append(
                f"| #{idx} | `{corpus}` | {n} | {sl} | {seed} | {top_k} | "
                f"{_window_str(window)} | {_tok_modes(by_key[key])} | "
                f"{tokh or 'n/a'} |"
            )
        out.append("")

    out.append("## Sources")
    out.append("")
    for r in runs_sorted:
        out.append(f"- `{r['_source_path']}`")
    out.append("")
    return "\n".join(out)


def render_json(teacher_path: str, runs: list[dict]) -> dict:
    """Machine-readable comparison: full per-run records sorted by mean KLD
    ascending."""
    runs_sorted = sorted(runs, key=_mean_or_inf)
    return {
        "schema_version": SCHEMA_VERSION,
        "teacher": runs[0]["teacher"],
        "generated": _now_iso(),
        "num_runs": len(runs),
        "runs": [{k: v for k, v in r.items() if k != "_source_path"} for r in runs_sorted],
        "sources": [r["_source_path"] for r in runs_sorted],
    }


def run_compare(
    patterns: list[str],
    teacher_filter: str | None,
    md_path: Path | None,
    json_path: Path | None,
    root_label: str | None = None,
    show_publisher: bool = False,
    show_scored_bpw: bool = False,
) -> int:
    runs = load_runs(patterns)
    if not runs:
        where = f"under {root_label}" if root_label else "matching the given --pattern"
        warn(f"no records found {where}. Run 'mlx-kld score' first, "
             "or point --out-dir/--pattern at your results root")
        return 1

    info(f"loaded {len(runs)} runs")
    groups = group_by_teacher(runs)

    if teacher_filter:
        matched = {k: v for k, v in groups.items() if teacher_filter in k}
        if not matched:
            warn(
                f"no teacher path contains {teacher_filter!r}. "
                f"available: {sorted(groups.keys())}"
            )
            return 1
        groups = matched

    if (md_path or json_path) and len(groups) != 1:
        warn("--md/--json need the scan to resolve to exactly one teacher "
             "group. Filter with --teacher")
        return 1

    for teacher_path, group_runs in groups.items():
        md = render_md(teacher_path, group_runs, show_publisher=show_publisher,
                       show_scored_bpw=show_scored_bpw)
        if md_path:
            write_text_atomic(md_path, md)
            info(f"{teacher_path}: wrote {md_path} ({len(group_runs)} runs)")
        else:
            print(md)
        if json_path:
            write_json_atomic(json_path, render_json(teacher_path, group_runs))
            info(f"{teacher_path}: wrote {json_path} ({len(group_runs)} runs)")
    return 0
