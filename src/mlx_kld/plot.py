"""Render an SVG scatter of student quality from the results root.

X is effective bits-per-weight (or file size), Y is mean KLD. Unquantized
students (the bf16 baseline) are excluded: their mean is the reconstruction
floor, not a quantization result. The floor itself is drawn as a dashed
reference line when it falls inside the y range, so a reader can see which
students sit near the measurement floor. Marker shape + color encode the
student format.
All runs must share one comparability spec (same rule as ``compare``); mixed
specs are refused rather than silently blended into one picture.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

from ._io import write_text_atomic
from ._log import info, warn
from .compare import _publisher, _run_key, group_by_teacher, load_runs

# Geometry (px). ViewBox scales; these only set proportions.
_W, _H = 760, 480
_ML, _MR, _MT, _MB = 70, 24, 62, 56

_MARKERS = {
    "mlx-affine": ("circle", "#4878cf", "MLX affine"),
    "mlx-kquant": ("triangle", "#3a923a", "MLX k-quant"),
    "gguf": ("square", "#d65f2f", "GGUF"),
}
_FALLBACK_MARKER = ("diamond", "#777777", None)

# Label placement: one gap from the marker, one vertical step. Uniform by
# construction, so no two labels sit at different distances from their points.
_GAP, _STEP = 10.0, 15.0

_X_AXES = {
    "bpw": ("effective bits per weight", lambda s: s.get("effective_bpw")),
    "size": ("file size (GB)", lambda s: (s.get("size_bytes") or 0) / 1e9 or None),
}


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_tick(v: float) -> str:
    return f"{v:g}"


def _linear_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    step = 10.0 ** math.floor(math.log10(span / target))
    for mult in (1, 2, 2.5, 5, 10):
        if span / (step * mult) <= target:
            step *= mult
            break
    t = math.ceil(lo / step) * step
    ticks = []
    while t <= hi + step * 1e-9:
        ticks.append(round(t, 12))
        t += step
    return ticks


def _log_ticks(lo: float, hi: float) -> list[float]:
    ticks = []
    for e in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10.0 ** e
            if lo <= v <= hi:
                ticks.append(v)
    return ticks


def _marker_svg(shape: str, x: float, y: float, color: str, r: float = 5.0,
                hollow: bool = False) -> str:
    """A marker. ``hollow`` draws it as an outline on white, which is how one
    format family's second publisher is told apart without spending a second
    hue on it (see ``_publisher_fills``)."""
    paint = (f'fill="white" stroke="{color}" stroke-width="2"' if hollow
             else f'fill="{color}"')
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" {paint}/>'
    if shape == "square":
        s = r * 1.75
        return (f'<rect x="{x - s / 2:.1f}" y="{y - s / 2:.1f}" '
                f'width="{s:.1f}" height="{s:.1f}" {paint}/>')
    if shape == "triangle":
        s = r * 1.35
        pts = f"{x:.1f},{y - s:.1f} {x - s:.1f},{y + s * 0.8:.1f} {x + s:.1f},{y + s * 0.8:.1f}"
        return f'<polygon points="{pts}" {paint}/>'
    s = r * 1.3
    pts = f"{x:.1f},{y - s:.1f} {x + s:.1f},{y:.1f} {x:.1f},{y + s:.1f} {x - s:.1f},{y:.1f}"
    return f'<polygon points="{pts}" {paint}/>'


def _teacher_base(teacher_path: str) -> str:
    """The teacher's model name, org prefix and precision suffix stripped:
    ``mlx-community__Qwen3.6-27B-bf16`` -> ``Qwen3.6-27B``."""
    base = re.sub(r"^.*?__", "", Path(teacher_path).name)
    return re.sub(r"[-_.](bf16|fp16|f16|f32)$", "", base, flags=re.IGNORECASE)


# Words that add nothing to a quant label: containers, modalities, and
# structure the scorer ignores (mtp weights are dropped at load).
_LABEL_NOISE = {"mlx", "vl", "it", "instruct", "gguf", "mtp"}


def _short_label(name: str, teacher_base: str) -> str:
    """Point label: whatever the student name adds beyond the teacher's, so
    ``Qwen_Qwen3.6-27B-Q6_K_L.gguf`` -> ``Q6_K_L``. Noise words are dropped
    from both ends (``MLX-VL-oQ5`` -> ``oQ5``, ``oQ4e-mtp`` -> ``oQ4e``).
    Falls back to the name with any ``org__`` prefix removed."""
    n = name.removesuffix(".gguf")
    if teacher_base:
        m = re.search(re.escape(teacher_base), n, re.IGNORECASE)
        if m:
            parts = n[m.end():].strip("-_. ").split("-")
            while len(parts) > 1 and parts[0].lower() in _LABEL_NOISE:
                parts.pop(0)
            while len(parts) > 1 and parts[-1].lower() in _LABEL_NOISE:
                parts.pop()
            tail = "-".join(parts)
            if tail:
                return _normalize_label(tail)
    return _normalize_label(re.sub(r"^.*?__", "", n) or n)


def _normalize_label(label: str) -> str:
    """``4bit`` -> ``q4``, so an MLX affine checkpoint reads on the same axis as
    the K-quant names beside it (``Q4_K_S``) instead of in its own idiom."""
    return re.sub(r"^(\d+)bit$", r"q\1", label, flags=re.IGNORECASE)


_FAMILY_ORDER = ("K-quant", "IQ", "MXFP4", "GGUF")


def _codec_family(codec: str) -> str | None:
    c = str(codec).lower()
    if c in ("f32", "f16", "bf16"):
        return None
    if re.match(r"q\d+_k", c):
        return "K-quant"
    if c.startswith("iq"):
        return "IQ"
    if c.startswith("mxfp"):
        return "MXFP4"
    return "GGUF"


def _gguf_legend_label(runs: list[dict]) -> str:
    """Legend text for GGUF students: the codec families present, each file
    counted by its own dominant family. Aux tensors in other codecs (a few q8_0
    embeddings in an otherwise K-quant file) don't flip that file's label.

    Families are listed rather than reduced to the single most common one. A set
    holding both K-quant and IQ files under one "K-quant" heading would be wrong
    about the IQ ones, and the reader has no way to tell which is which."""
    fams = []
    for r in runs:
        q = r["student"].get("quantization") or {}
        if q.get("kind") != "gguf":
            continue
        per: dict[str, int] = {}
        for codec, count in (q.get("codecs") or {}).items():
            fam = _codec_family(codec)
            if fam:
                per[fam] = per.get(fam, 0) + int(count or 0)
        if per:
            fam = max(per, key=per.get)
            if fam not in fams:
                fams.append(fam)
    if not fams:
        return "GGUF"
    fams.sort(key=lambda f: _FAMILY_ORDER.index(f) if f in _FAMILY_ORDER else 99)
    return " / ".join(fams)


def _colliding_formats(points: list[tuple]) -> set:
    """Formats where two publishers ship the same quant name, so the file-name
    label alone names two different files. Those families get the fill split;
    the legend then says which publisher is which, for every point at once."""
    seen: dict = {}
    for _x, _y, _fmt, label, _pub in points:
        seen[label] = seen.get(label, 0) + 1
    return {fmt for _x, _y, fmt, label, pub in points if seen[label] > 1 and pub}


def _publisher_fills(points: list[tuple], split_formats: set) -> dict:
    """Map ``(format, publisher) -> hollow?`` for the families in
    ``split_formats``.

    Publisher is a second categorical dimension on top of format, and hue is
    already spent on format. Recoloring one publisher would also split a family
    that belongs together (two K-quant GGUFs are both K-quant). So the split
    rides the fill channel instead: the larger publisher keeps the solid marker,
    the next takes the outline. That reads in grayscale and under colorblindness,
    and it costs no palette budget.

    Only families whose file names actually collide are split. Several
    publishers shipping distinctly-named checkpoints need no marker distinction,
    and giving them one is noise. Only two fills exist, so a third publisher in a
    colliding family cannot be separated and the chart says so.
    """
    counts: dict = {}
    for _x, _y, fmt, _label, pub in points:
        if fmt in split_formats:
            counts.setdefault(fmt, {})
            counts[fmt][pub] = counts[fmt].get(pub, 0) + 1
    fills = {}
    for fmt, pubs in counts.items():
        named = sorted((p for p in pubs if p), key=lambda p: (-pubs[p], p))
        if len(named) < 2:
            continue
        for i, pub in enumerate(named[:2]):
            fills[(fmt, pub)] = bool(i)
        if len(named) > 2:
            warn(f"{fmt}: {len(named)} publishers ship colliding names but only "
                 f"2 marker fills exist, so {', '.join(named[2:])} cannot be told "
                 f"apart on the chart")
    return fills


def _overlaps(a: tuple, b: tuple, pad: float = 2.0) -> bool:
    return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                or a[3] + pad < b[1] or b[3] + pad < a[1])


def _box_dist(rect: tuple, pt: tuple) -> float:
    """Distance from a point to the nearest edge of a box."""
    x = min(max(pt[0], rect[0]), rect[2])
    y = min(max(pt[1], rect[1]), rect[3])
    return math.hypot(x - pt[0], y - pt[1])


def _claims_own(rect: tuple, own: tuple, centers: list) -> bool:
    """Whether ``rect`` is closer to ``own`` than to any other marker. A label
    nearer some other point than the one it names is misinformation, however
    tidy it looks, so placement treats this as a hard constraint."""
    d = _box_dist(rect, own)
    return all(_box_dist(rect, c) >= d for c in centers if c is not own)


def _place_labels(
    pts: list[tuple[float, float, str]],
    boxes: list[tuple],
    x0: float, y0: float, x1: float, y1: float,
) -> list[tuple]:
    """Greedy label placement: for each point walk candidate slots (a near
    ring around the marker, then a pushed-out ring), keep the first that
    stays inside the plot and clears every marker and already-placed label.
    Returns (x, baseline-y, text, leader) with leader either None or the
    (x1, y1, x2, y2) of a marker-to-label line.

    Every candidate sits exactly ``_GAP`` from the marker on the axis it is
    offset along, and vertical displacement comes in whole ``_STEP`` rows. Slots
    at mixed distances make identical relationships look different, which reads
    as sloppiness rather than as information.

    Two passes. The first also requires the label to be closer to its own marker
    than to any other, which is what lets a label stand alone. Only when no such
    slot exists does the second pass drop that requirement, and a label placed
    there gets a leader line, since it is then the line rather than proximity
    that ties it to its point."""
    h = 11.0
    placed = []
    centers = [(x, y) for x, y, _ in pts]
    for idx, (x, y, text) in enumerate(pts):
        w = 6.1 * len(text) + 2
        # Right and left at the same gap first, then the same two columns
        # stepped off the marker's row, then directly above/below.
        cands = [
            (x + _GAP, y - h / 2), (x - _GAP - w, y - h / 2),
            (x - w / 2, y - _GAP - h), (x - w / 2, y + _GAP),
        ]
        for dy in (_STEP, -_STEP, 2 * _STEP, -2 * _STEP):
            cands.append((x + _GAP, y - h / 2 + dy))
            cands.append((x - _GAP - w, y - h / 2 + dy))
        cands += [(x - w / 2, y - _GAP - h - _STEP),
                  (x - w / 2, y + _GAP + _STEP)]

        def fits(r):
            return not (r[0] < x0 or r[2] > x1 or r[1] < y0 or r[3] > y1
                        or any(_overlaps(r, b) for b in boxes))

        own = centers[idx]
        rects = [(cx, cy, cx + w, cy + h) for cx, cy in cands]
        rect = next((r for r in rects
                     if fits(r) and _claims_own(r, own, centers)), None)
        leader = None
        if rect is None:
            rect = next((r for r in rects if fits(r)),
                        (x + _GAP, y - h / 2, x + _GAP + w, y + h / 2))
            ax = min(max(x, rect[0]), rect[2])
            ay = min(max(y, rect[1]), rect[3])
            if math.hypot(ax - x, ay - y) > 1:
                leader = (x, y, ax, ay)
        boxes.append(rect)
        placed.append((rect[0], rect[3] - 2.5, text, leader))
    return placed


def render_svg(
    teacher_path: str,
    runs: list[dict],
    x_axis: str = "bpw",
    log_y: bool = False,
    labels: bool = True,
) -> str:
    x_label, x_of = _X_AXES[x_axis]
    tbase = _teacher_base(teacher_path)
    points = []
    for r in runs:
        s = r["student"]
        x, y = x_of(s), r["kld"].get("mean")
        if x is None:
            warn(f"skipping {Path(s['path']).name}: missing {x_axis}")
            continue
        if y is None or y <= 0:
            # y <= 0 is a real value the log axis (and the shared scatter
            # geometry) cannot place, so say that rather than "missing".
            warn(f"skipping {Path(s['path']).name}: missing or non-positive "
                 "mean KLD")
            continue
        points.append((x, y, s.get("format"),
                       _short_label(Path(s["path"]).name, tbase),
                       _publisher(s["path"])))
    if not points:
        raise ValueError("no plottable runs (need effective_bpw/size and mean KLD)")
    fills = _publisher_fills(points, _colliding_formats(points))

    # Every run in a group shares a floor (same teacher, same top-K), so the
    # first usable one speaks for all of them. Require > 0, not just
    # not-None: a 0.0 floor cannot be placed on a log axis and would shadow a
    # later run that carries a real one.
    floor = next((fm for r in runs
                  if (fm := r["kld"].get("floor_mean")) is not None and fm > 0), None)

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_lo, x_hi = min(xs), max(xs)
    pad = (x_hi - x_lo) * 0.06 or x_lo * 0.05 or 1.0
    x_lo, x_hi = x_lo - pad, x_hi + pad
    if log_y:
        # Range hugs the data; a floor below it is cited in the band caption
        # instead of stretching the axis into empty decades.
        y_lo = min(ys) * 0.75
        y_hi = max(ys) * 1.15
    else:
        y_lo, y_hi = 0.0, max(ys) * 1.1
        # Quant quality spans orders of magnitude once very low bit widths are
        # in the set. On a linear axis one bad student then flattens everything
        # else onto the baseline, which hides the differences worth reading.
        spread = max(ys) / min(ys)
        if spread > 25:
            warn(f"mean KLD spans {spread:.0f}x across these students, so a linear "
                 f"y axis compresses most of them onto the baseline. Use "
                 f"--log-y to read this set.")

    pw, ph = _W - _ML - _MR, _H - _MT - _MB

    def px(v: float) -> float:
        return _ML + (v - x_lo) / (x_hi - x_lo) * pw

    def py(v: float) -> float:
        if log_y:
            f = (math.log10(v) - math.log10(y_lo)) / (math.log10(y_hi) - math.log10(y_lo))
        else:
            f = (v - y_lo) / (y_hi - y_lo)
        return _MT + ph - f * ph

    precision = runs[0].get("teacher", {}).get("precision") or "full-precision"
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_W} {_H}" '
        'font-family="\'Helvetica Neue\', Helvetica, Arial, sans-serif">',
        f'<rect width="{_W}" height="{_H}" fill="white"/>',
        f'<text x="{_ML}" y="26" font-size="16" font-weight="bold" fill="#222">'
        f'{_esc(tbase)}: mean KL divergence by quantization</text>',
        f'<text x="{_ML}" y="45" font-size="12" fill="#555">'
        f'mean KLD vs the {_esc(precision)} teacher, lower is better</text>',
    ]

    # Dashed floor line when it falls inside the y range; its value is always
    # in the subtitle. Caption boxes are collected so point labels never land
    # on top of them.
    caption_boxes = []
    if floor is not None and y_lo < floor < y_hi:
        fy = py(floor)
        out.append(f'<line x1="{_ML}" y1="{fy:.1f}" x2="{_ML + pw}" y2="{fy:.1f}" '
                   f'stroke="#888" stroke-dasharray="5,4"/>')
        text = "top-K reconstruction floor"
        out.append(f'<text x="{_ML + pw - 6}" y="{fy - 5:.1f}" font-size="10" '
                   f'fill="#888" text-anchor="end">{text}</text>')
        caption_boxes.append(
            (_ML + pw - 6 - 6.1 * len(text), fy - 15, _ML + pw - 6, fy - 3)
        )

    y_ticks = _log_ticks(max(y_lo, 1e-12), y_hi) if log_y else _linear_ticks(y_lo, y_hi)
    for t in y_ticks:
        y = py(t)
        out.append(f'<line x1="{_ML}" y1="{y:.1f}" x2="{_ML + pw}" y2="{y:.1f}" '
                   f'stroke="#e8e8e8"/>')
        out.append(f'<text x="{_ML - 8}" y="{y + 4:.1f}" font-size="11" fill="#555" '
                   f'text-anchor="end">{_fmt_tick(t)}</text>')
    for t in _linear_ticks(x_lo, x_hi):
        x = px(t)
        out.append(f'<line x1="{x:.1f}" y1="{_MT + ph}" x2="{x:.1f}" y2="{_MT + ph + 5}" '
                   f'stroke="#999"/>')
        out.append(f'<text x="{x:.1f}" y="{_MT + ph + 20}" font-size="11" fill="#555" '
                   f'text-anchor="middle">{_fmt_tick(t)}</text>')

    out.append(f'<line x1="{_ML}" y1="{_MT + ph}" x2="{_ML + pw}" y2="{_MT + ph}" '
               f'stroke="#999"/>')
    out.append(f'<line x1="{_ML}" y1="{_MT}" x2="{_ML}" y2="{_MT + ph}" stroke="#999"/>')
    out.append(f'<text x="{_ML + pw / 2:.1f}" y="{_H - 14}" font-size="12" fill="#333" '
               f'text-anchor="middle">{_esc(x_label)}</text>')
    out.append(f'<text x="18" y="{_MT + ph / 2:.1f}" font-size="12" fill="#333" '
               f'text-anchor="middle" transform="rotate(-90 18 {_MT + ph / 2:.1f})">'
               f'mean KLD (nats)</text>')

    formats_seen = []
    pixel_pts = []
    # Sort on the numeric fields only. A full tuple sort falls through to
    # `format`, which is None for a record that never carried one, and
    # None < str raises.
    for x, y, fmt, name, pub in sorted(points, key=lambda p: (p[0], p[1])):
        shape, color, _ = _MARKERS.get(fmt, _FALLBACK_MARKER)
        if fmt not in formats_seen:
            formats_seen.append(fmt)
        pixel_pts.append((px(x), py(y), name))
        out.append(_marker_svg(shape, px(x), py(y), color,
                               hollow=fills.get((fmt, pub), False)))

    # One horizontal row in the header band, right-aligned to the plot edge.
    # Gridlines span the full plot width, so an in-plot legend always risks a
    # rule drawn through a row; out here nothing crosses it and no marker is
    # hidden underneath it.
    # A split family reads as one caption plus a marker per publisher
    # ("K-quant: [solid] unsloth [outline] bartowski"), so the family stays
    # named once and the fills sit side by side under it.
    entries = []
    for fmt in formats_seen:
        shape, color, label = _MARKERS.get(fmt, _FALLBACK_MARKER)
        if fmt == "gguf":
            label = _gguf_legend_label(runs)
        label = _esc(label or str(fmt))
        pubs = [p for (f, p) in fills if f == fmt]
        if pubs:
            entries.append((None, None, f"{label}:", False))
            for pub in sorted(pubs, key=lambda p: fills[(fmt, p)]):
                entries.append((shape, color, _esc(pub), fills[(fmt, pub)]))
        else:
            entries.append((shape, color, label, False))
    # Helvetica at 12px averages a hair over 6px per char; only used to right-
    # align the row, so a loose estimate is fine.
    widths = [(0 if shape is None else 16) + 6.4 * len(text)
              for shape, _, text, _ in entries]
    if entries:
        lx, ly = _ML + pw - (sum(widths) + 12 * (len(entries) - 1)), 45
        for (shape, color, text, hollow), w in zip(entries, widths, strict=True):
            tx = lx
            if shape is not None:
                out.append(_marker_svg(shape, lx + 4.5, ly - 4, color, r=4.5,
                                       hollow=hollow))
                tx = lx + 16
            out.append(f'<text x="{tx:.1f}" y="{ly}" font-size="12" '
                       f'fill="#333">{text}</text>')
            lx += w + 12

    if labels:
        boxes = list(caption_boxes)
        boxes += [(mx - 7, my - 7, mx + 7, my + 7) for mx, my, _ in pixel_pts]
        for tx, ty, text, leader in _place_labels(
            pixel_pts, boxes, _ML + 2, _MT + 2, _ML + pw - 2, _MT + ph - 2,
        ):
            if leader:
                out.append(f'<line x1="{leader[0]:.1f}" y1="{leader[1]:.1f}" '
                           f'x2="{leader[2]:.1f}" y2="{leader[3]:.1f}" '
                           f'stroke="#bbb" stroke-width="0.8"/>')
            out.append(f'<text x="{tx:.1f}" y="{ty:.1f}" font-size="10" '
                       f'fill="#444">{_esc(text)}</text>')

    out.append("</svg>")
    return "\n".join(out) + "\n"


def run_plot(
    patterns: list[str],
    teacher_filter: str | None,
    svg_path: Path | None,
    x_axis: str = "bpw",
    log_y: bool = False,
    labels: bool = True,
    root_label: str | None = None,
) -> int:
    runs = load_runs(patterns)
    if not runs:
        where = f"under {root_label}" if root_label else "matching the given --pattern"
        warn(f"no records found {where}. Run 'mlx-kld score' first")
        return 1

    groups = group_by_teacher(runs)
    if teacher_filter:
        groups = {k: v for k, v in groups.items() if teacher_filter in k}
        if not groups:
            warn(f"no teacher path contains {teacher_filter!r}")
            return 1
    if len(groups) != 1:
        warn("plot needs exactly one teacher group. Filter with --teacher "
             f"(available: {sorted(groups.keys())})")
        return 1
    teacher_path, group = next(iter(groups.items()))

    quantized = [r for r in group
                 if (r["student"].get("quantization") or {}).get("kind") != "none"]
    dropped = len(group) - len(quantized)
    if dropped:
        info(f"excluded {dropped} unquantized run(s) from the plot")
    if not quantized:
        warn("no quantized runs to plot")
        return 1

    specs = {_run_key(r) for r in quantized}
    if len(specs) > 1:
        warn(f"{len(specs)} distinct run specs under this teacher, and one chart "
             "must come from one spec. Narrow with --pattern, or rescore.")
        return 1

    svg = render_svg(teacher_path, quantized, x_axis=x_axis, log_y=log_y,
                     labels=labels)
    if svg_path is None:
        slug = Path(teacher_path).name.lower()
        svg_path = Path(f"{slug}-kld-vs-{x_axis}.svg")
    write_text_atomic(svg_path, svg)
    info(f"wrote {svg_path} ({len(quantized)} students)")
    return 0
