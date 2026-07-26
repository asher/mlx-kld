"""Markdown report + locked schema_version=1 JSON record + record placement."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from ._constants import DEFAULT_RESULTS_DIR, RESULTS_ENV, SCHEMA_VERSION
from .errors import RecordSchemaError
from .measure import STUDENT_FORMATS


def resolve_results_dir(out_dir: Path | None) -> Path:
    """Results root: ``--out-dir`` > ``$MLX_KLD_RESULTS`` > ``./kld-results``."""
    if out_dir is not None:
        return out_dir
    env = os.environ.get(RESULTS_ENV)
    return Path(env) if env else DEFAULT_RESULTS_DIR


def run_digest(
    calibration: dict,
    tok_mode: str,
    identity: str | None = None,
) -> str:
    """8-hex digest of everything that must not share a record file: the
    calibration spec, the tokenizer mode, the score window, and, when given,
    the run's *identity* (which teacher, which student on disk).

    Identity has to be in here. Slugs are basenames, so ``/runA/q4`` and
    ``/runB/q4`` produce the same student slug, and without an identity term two
    unrelated students would silently overwrite each other's records. Distinct
    runs never collide; an identical rerun stays idempotent.
    """
    payload = "|".join(str(x) for x in (
        calibration.get("corpus"),
        calibration.get("num_samples"),
        calibration.get("max_seq_len"),
        calibration.get("seed"),
        calibration.get("top_k"),
        calibration.get("score_window"),
        tok_mode,
        identity,
    ))
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def teacher_slug(teacher: str) -> str:
    """Directory name for a teacher, stable and collision-resistant.

    Keeps the org so ``mlx-community/Qwen3-0.6B`` and ``Qwen/Qwen3-0.6B`` land in
    different directories. For a local HF snapshot path
    (``.../models--ORG--NAME/snapshots/<sha>/``) the basename is a commit sha,
    which is both unreadable and unstable across re-pulls, so recover ``ORG__NAME``
    from the ``models--`` segment instead.
    """
    p = Path(teacher.rstrip("/"))
    for part in p.parts:
        if part.startswith("models--"):
            return part[len("models--"):].replace("--", "__").lower()
    stem = teacher.rstrip("/")
    if "/" in stem and not Path(stem).is_absolute():
        # An HF id: keep org and name.
        return "__".join(stem.split("/")[-2:]).lower()
    return Path(stem).name.lower()


def results_json_path(
    results_dir: Path,
    teacher: str,
    student: Path,
    calibration: dict,
    tok_mode: str,
) -> Path:
    """Default JSON record path:
    ``<results_dir>/<teacher-slug>/<student-slug>.<digest8>.json``."""
    student = Path(student)
    student_slug = student.stem if student.suffix.lower() == ".gguf" \
        else student.name
    # Resolve so two spellings of one path share a record, while two genuinely
    # different students never do.
    try:
        student_identity = str(student.resolve())
    except OSError:
        student_identity = str(student)
    digest = run_digest(calibration, tok_mode, f"{teacher}|{student_identity}")
    return results_dir / teacher_slug(teacher) / f"{student_slug}.{digest}.json"


# Below this multiple of the reconstruction floor, the *absolute* mean KLD is
# substantially measurement floor rather than student drift.
#
# Not a ranking caution. The floor is very nearly a common offset across the
# students of one teacher, so it inflates absolutes and compresses the ratios
# between students without reordering them. An earlier 10.0 flagged every quant
# at Q5 and above on a 27B, including the ones whose numbers agree with
# published llama.cpp results, which is precisely the evidence that they are
# trustworthy. Agreement demonstrably holds down to ~2.9x floor.
FLOOR_CAUTION_MULT = 2.0


def floor_limited(kld: dict) -> bool:
    """True when the reconstruction floor is a large enough share of the mean
    KLD that the absolute value should be read with it in mind."""
    mean, floor = kld.get("mean"), kld.get("floor_mean")
    return (
        mean is not None and floor is not None and floor > 0
        and mean < FLOOR_CAUTION_MULT * floor
    )


# A student whose next-token distribution is essentially unrelated to the
# teacher's is almost always a broken load or config (wrong norm convention,
# tokenizer/vocab mismatch, corrupt weights), not a real quantization. Every
# genuine quant we have measured keeps top-1 agreement well above 0.85 and mean
# KLD well under 0.1 nats; a broken load lands near chance. Flag results past
# these wide bounds loudly rather than emitting a silently-wrong number.
IMPLAUSIBLE_TOP1 = 0.5
IMPLAUSIBLE_MEAN_KLD = 1.0


def implausibility_reason(
    agreement: dict | None, kld: dict | None, tokens_scored: int | None = None,
) -> str | None:
    """Human-readable reason when a result looks like a broken load rather than
    a real quant, else None. Pure function of the already-computed metrics."""
    top1 = (agreement or {}).get("top1")
    mean = (kld or {}).get("mean")
    reasons = []
    if tokens_scored == 0:
        reasons.append(
            "no token was scorable (every position produced a non-finite KLD)"
        )
    if top1 is not None and top1 < IMPLAUSIBLE_TOP1:
        reasons.append(
            f"top-1 agreement {top1 * 100:.2f}% is far below any real quant "
            f"(< {IMPLAUSIBLE_TOP1 * 100:.0f}%)"
        )
    if mean is not None and mean > IMPLAUSIBLE_MEAN_KLD:
        reasons.append(
            f"mean KLD {mean:.3f} nats is implausibly high "
            f"(> {IMPLAUSIBLE_MEAN_KLD:.1f})"
        )
    return "; ".join(reasons) if reasons else None


def _fmt_stat(v, digits: int = 4) -> str:
    """Fixed-point at ``digits`` decimals; scientific once a nonzero value
    would floor to 0 at that precision (near-lossless students live there)."""
    if v is None:
        return "n/a"
    if v != 0 and abs(v) < 0.5 * 10 ** -digits:
        return f"{v:.1e}"
    return f"{v:.{digits}f}"


def _fmt_pct(v, digits: int = 2) -> str:
    """Render a fraction as a percentage. ``None`` (a run that scored nothing)
    renders as n/a rather than raising."""
    if v is None:
        return "n/a"
    return f"{v * 100:.{digits}f}%"


def _fmt_pm(mean, se, digits: int = 4) -> str:
    """Render ``mean +/- se``; falls back to plain mean when se is absent."""
    if mean is None:
        return "n/a"
    if se is None:
        return _fmt_stat(mean, digits)
    return f"{_fmt_stat(mean, digits)} +/- {_fmt_stat(se, digits)}"


def _fmt_pp(v, digits: int = 3) -> str:
    """Render a probability difference as percentage points.

    llama.cpp prints its Delta-p family scaled by 100 with a ``%`` sign, so a
    raw-probability rendering here would sit 100x away from any published table
    a reader is comparing against. Only the rendering scales. The JSON record
    keeps raw probability, so ``schema_version=1`` consumers are unaffected.
    """
    if v is None:
        return "n/a"
    return f"{_fmt_stat(v * 100.0, digits)}%"


def _fmt_pp_pm(mean, se, digits: int = 3) -> str:
    if mean is None:
        return "n/a"
    if se is None:
        return _fmt_pp(mean, digits)
    return f"{_fmt_pp(mean, digits)} +/- {_fmt_pp(se, digits)}"


def _fmt_params(n) -> str:
    return f"{n / 1e9:.2f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def _quantization_lines(student: dict) -> list[str]:
    """Format-appropriate quantization block for the Markdown report."""
    q = student.get("quantization") or {}
    kind = q.get("kind")
    out = [f"- format: `{student.get('format')}`"]
    bpw = student.get("effective_bpw")
    if kind == "affine":
        out.append(
            f"- affine: bits={q.get('bits')}, group_size={q.get('group_size')}, "
            f"mode={q.get('mode')}"
        )
    elif kind in ("kquant", "gguf"):
        if kind == "gguf" and q.get("arch"):
            out.append(f"- arch: `{q['arch']}`")
        codecs = q.get("codecs") or {}
        if codecs:
            hist = " ".join(
                f"{c}:{codecs[c]}"
                for c in sorted(codecs, key=lambda c: (-codecs[c], c))
            )
            out.append(f"- codecs: `{hist}`")
        else:
            out.append("- codecs: n/a")
    elif kind == "none":
        # bpw folds into this line; a separate effective_bpw row under a
        # "Quantization" heading reads like a bug for an unquantized model.
        dtype = q.get("dtype") or "unknown dtype"
        suffix = f", {bpw:.1f} bpw" if bpw is not None else ""
        out.append(f"- unquantized ({dtype}{suffix})")
        bpw = None
    if bpw is not None:
        out.append(f"- effective_bpw: {bpw:.3f}")
    if student.get("n_params") is not None:
        out.append(f"- parameters: {_fmt_params(student['n_params'])}")
    return out


def render_markdown(report: dict) -> str:
    out: list[str] = []
    p = out.append

    teacher = report["teacher"]
    student = report["student"]
    cal = report["calibration"]
    kld = report["kld"]
    delta_p = report.get("delta_p") or {}
    agreement = report["agreement"]

    teacher_label = teacher["path"]
    student_label = Path(student["path"]).name
    p(f"# KLD score: {student_label} vs {teacher_label}\n")

    p("## Calibration\n")
    p(f"- dataset: `{cal['corpus']}`")
    p(f"- num_samples: {cal['num_samples']}")
    p(f"- max_seq_len: {cal['max_seq_len']}")
    p(f"- seed: {cal['seed']}")
    # The window is part of the never-blend spec, so the human-readable report
    # must show it, not just the JSON record.
    window = cal.get("score_window")
    if window:
        p(f"- score_window: [{window[0]}, {window[1]})"
          + (" (long-context)" if cal.get("long_context") else ""))
    p(f"- top-k (cache): {report['cache']['top_k']}")
    p(f"- tokens scored: {report['tokens_scored']:,}")
    dropped = report.get("tokens_dropped_nonfinite") or 0
    if dropped:
        p(f"- **tokens dropped (non-finite KLD): {dropped:,}**. The student "
          "produced inf/NaN logits there, and the mean excludes those "
          "positions")
    p("")

    tok = report.get("tokenizer")
    if tok:
        p("## Tokenizer\n")
        p(f"- mode: {tok.get('mode')}")
        if tok.get("stream_is_students") is False:
            p("- **scored stream is NOT this student's tokenization**: a cache "
              "HIT replayed the token ids an earlier student built the cache "
              "from. The weights under test are this student's, but the "
              "tokenization is not. Rescore with `--rebuild-cache` to test "
              "this student's own tokenization.")
        if tok["identical"]:
            p("- teacher and student tokenizers are identical\n")
        else:
            p("- teacher vs student differ in metadata:")
            for d in tok.get("diffs") or []:
                p(f"  - {d}")
            if tok["encoding_parity"]:
                p("- encoding parity verified (text -> ids identical), so the "
                  "KLD is well-defined and the differences above don't affect "
                  "tokenization\n")
            else:
                forced = " (forced via --allow-tokenizer-mismatch)" if tok["forced"] else ""
                p(f"- **encoding DIVERGES{forced}**: KLD may be meaningless\n")

    p("## Headline\n")
    p("| Metric | Value |")
    p("|---|---:|")
    p(f"| Mean KLD (nats) | {_fmt_pm(kld['mean'], kld.get('se'))} |")
    floor = kld.get("floor_mean")
    if floor is not None:
        p(f"| Top-K reconstruction floor (mean) | {_fmt_stat(floor)} |")
    p(f"| Median KLD | {_fmt_stat(kld['p50'])} |")
    p(f"| P95 KLD | {_fmt_stat(kld['p95'])} |")
    p(f"| P99 KLD | {_fmt_stat(kld['p99'])} |")
    p(f"| P99.9 KLD | {_fmt_stat(kld['p999'])} |")
    p(f"| Max KLD | {_fmt_stat(kld['max'])} |")
    if delta_p.get("mean") is not None:
        # Percentage points, the unit llama.cpp prints Delta-p in.
        p(f"| Mean Delta-p (% pts) | {_fmt_pp_pm(delta_p['mean'], delta_p.get('se'))} |")
        p(f"| RMS Delta-p (% pts) | {_fmt_pp(delta_p['rms'])} |")
    p(f"| Top-1 agreement (llama.cpp \"Same top p\") | {_fmt_pct(agreement['top1'])} |")
    p(f"| Top-5 agreement | {_fmt_pct(agreement['top5'])} |\n")
    reason = implausibility_reason(agreement, kld, report.get("tokens_scored"))
    if reason:
        p("**WARNING - implausible result:** " + reason + ". This usually "
          "means a broken load or config (wrong norm convention, tokenizer/"
          "vocab mismatch, corrupt weights), not a real quantization. Treat "
          "the score as INVALID until the cause is found.\n")
    if floor_limited(kld):
        p(f"**Floor-limited:** the mean KLD is within {FLOOR_CAUTION_MULT:g}x of "
          "the top-K reconstruction floor, so a large share of the absolute "
          "value is measurement floor rather than student drift. Ranking "
          "against sibling students is still sound (they carry the same "
          "floor), but read the magnitude, and ratios against students well "
          "above the floor, with that in mind. To find a --top-k with a lower "
          "floor, run `mlx-kld floor-sweep` on the teacher. Note that the "
          "floor is U-shaped in K, so raising --top-k does not always lower "
          "it.\n")

    by_pos = report.get("by_position") or []
    if by_pos:
        p("## By position bucket\n")
        p("| Position range | Tokens scored | Mean KLD | P95 KLD | Top-1 agreement |")
        p("|---|---:|---:|---:|---:|")
        for row in by_pos:
            if row["tokens"] == 0:
                p(f"| {row['range']} | 0 | n/a | n/a | n/a |")
            else:
                p(f"| {row['range']} | {row['tokens']:,} | "
                  f"{row['mean_kld']:.4f} | {row['p95_kld']:.4f} | "
                  f"{row['top1_agreement'] * 100:.2f}% |")
        p("")

    p("## Quantization\n")
    for line in _quantization_lines(student):
        p(line)
    p("")

    prov = student.get("provenance")
    if prov:
        p("### Quantizer provenance\n")
        p("| Field | Value |")
        p("|---|---|")
        for k, v in prov.items():
            if isinstance(v, bool):
                cell = "true" if v else "false"
            elif v is None:
                cell = "n/a"
            else:
                cell = str(v)
            p(f"| `{k}` | {cell} |")
        p("")

    p("## Reproducibility\n")
    p(f"- teacher: `{teacher['path']}` "
      f"(revision {teacher.get('revision') or 'n/a'}, {teacher['precision']})")
    p(f"- student: `{student['path']}`")
    p(f"  ({student['size_bytes'] / 1e9:.2f} GB on disk)")
    p(f"- cache: `{report['cache']['dir']}/`  ({report['cache']['status']})")
    if report.get("json_path"):
        p(f"- JSON record: `{report['json_path']}`")
    p(f"- scorer: {report['scorer_version']}")
    p(f"- elapsed: {report['elapsed_seconds']:.1f}s ({report['elapsed_phase']})")
    p(f"- timestamp: {report['timestamp']}")
    p("")
    return "\n".join(out)


def _public_teacher_name(teacher_path: str) -> str:
    """A teacher name safe to publish.

    An HF id (``org/name``) is already public and identifies the model exactly,
    so it passes through. A local path is reduced to its final component. The
    directory layout of the machine that ran the score is not reproducible for a
    reader and does not belong in a model card.
    """
    raw = (teacher_path or "").rstrip("/")
    if not raw:
        return "unknown"
    p = Path(raw)
    if p.is_absolute() or raw.startswith(("~", ".")):
        # A local HF snapshot buries the readable name in a `models--ORG--NAME`
        # segment, and teacher_slug already knows how to recover it.
        return teacher_slug(raw)
    return raw


def render_card(report: dict) -> str:
    """A short Markdown block sized for a model card.

    The full report is written for someone diagnosing a run. This is written for
    a reader deciding whether to download a checkpoint, so it carries the
    headline numbers, the spec they are only comparable within, and the command
    that reproduces them.

    Local paths are deliberately absent. The output is meant to be published,
    and the directory layout of the machine that produced it is neither
    reproducible for a reader nor anyone else's business.
    """
    out: list[str] = []
    p = out.append
    teacher = report["teacher"]
    student = report["student"]
    cal = report["calibration"]
    kld = report["kld"]

    p("## Quantization quality (mlx-kld)")
    p("")
    teacher_name = _public_teacher_name(teacher.get("path", ""))
    p(f"Scored against `{teacher_name}` ({teacher.get('precision') or 'n/a'}) "
      f"with {report['scorer_version']}.")
    p("")
    p("| Metric | Value |")
    p("|---|---:|")
    p(f"| Mean KL divergence (nats) | {_fmt_pm(kld['mean'], kld.get('se'))} |")
    p(f"| P99 KL divergence (nats) | {_fmt_stat(kld['p99'])} |")
    p(f"| Top-1 agreement | {_fmt_pct(report['agreement'].get('top1'))} |")
    if student.get("effective_bpw") is not None:
        p(f"| Effective bits per weight | {student['effective_bpw']:.3f} |")
    p(f"| Size on disk | {student['size_bytes'] / 1e9:.2f} GB |")
    p("")
    p("Lower KL divergence means a next-token distribution closer to the "
      "full-precision teacher's.")
    p("")

    reason = implausibility_reason(
        report.get("agreement"), kld, report.get("tokens_scored")
    )
    if reason:
        p(f"**Warning, implausible result:** {reason}. This usually indicates a "
          "broken load rather than a real quantization.")
        p("")
    elif floor_limited(kld):
        p(f"Note: within {FLOOR_CAUTION_MULT:g}x of the top-K reconstruction "
          f"floor ({_fmt_stat(kld.get('floor_mean'))} nats), so the absolute "
          "value is substantially measurement floor. Ranking against other "
          "quants of this teacher is unaffected.")
        p("")

    window = cal.get("score_window")
    p(f"<sub>Corpus `{cal['corpus']}`, {cal['num_samples']} x "
      f"{cal['max_seq_len']} tokens, seed {cal['seed']}, top-k "
      f"{report['cache']['top_k'] if report.get('cache') else cal.get('top_k')}"
      + (f", window {window[0]}:{window[1]}" if window else "")
      + (f", token stream `{cal['corpus_tokens_hash']}`"
         if cal.get("corpus_tokens_hash") else "")
      + ". Numbers are comparable only against runs sharing that spec.</sub>")
    p("")
    p("```")
    p(f"mlx-kld score {teacher_name} <path-to-this-checkpoint>")
    p("```")
    p("")
    return "\n".join(out)


def build_locked_json(report: dict) -> dict:
    """Distill ``report`` down to the locked schema keys, keeping extras
    (by_position, histogram, cache, plausibility) under non-locked keys for
    downstream tools."""
    reason = implausibility_reason(
        report.get("agreement"), report.get("kld"), report.get("tokens_scored"),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "teacher": report["teacher"],
        "student": report["student"],
        "tokenizer": report["tokenizer"],
        "calibration": report["calibration"],
        "kld": report["kld"],
        "delta_p": report["delta_p"],
        "agreement": report["agreement"],
        "tokens_scored": report["tokens_scored"],
        "elapsed_seconds": report["elapsed_seconds"],
        "scorer_version": report["scorer_version"],
        "timestamp": report["timestamp"],
        # extras (non-locked, may be expanded over time)
        "tokens_dropped_nonfinite": report.get("tokens_dropped_nonfinite", 0),
        "by_position": report.get("by_position"),
        "kld_histogram": report.get("kld_histogram"),
        "cache": report.get("cache"),
        "plausibility": {"implausible": bool(reason), "reason": reason},
    }


def validate_locked_schema(payload: dict) -> None:
    """Sanity-check that the locked keys exist with the right primitive types.
    Called immediately before write so a broken scorer fails loud; raises
    :class:`~mlx_kld.errors.RecordSchemaError`."""
    required = {
        "schema_version": int,
        "teacher": dict,
        "student": dict,
        "tokenizer": dict,
        "calibration": dict,
        "kld": dict,
        "delta_p": dict,
        "agreement": dict,
        "tokens_scored": int,
        "elapsed_seconds": (int, float),
        "scorer_version": str,
        "timestamp": str,
    }
    for k, t in required.items():
        if k not in payload:
            raise RecordSchemaError(f"locked schema: missing key {k!r}")
        if not isinstance(payload[k], t):
            raise RecordSchemaError(
                f"locked schema: {k!r} should be {t}, got {type(payload[k])}"
            )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise RecordSchemaError(f"schema_version must be {SCHEMA_VERSION}")
    student = payload["student"]
    for k in ("path", "format", "size_bytes", "effective_bpw", "n_params",
              "quantization"):
        if k not in student:
            raise RecordSchemaError(f"locked schema: student.{k} missing")
    if student["format"] not in STUDENT_FORMATS:
        raise RecordSchemaError(
            f"locked schema: student.format {student['format']!r} not in "
            f"{STUDENT_FORMATS}"
        )
    if not isinstance(student["quantization"], dict) \
            or "kind" not in student["quantization"]:
        raise RecordSchemaError("locked schema: student.quantization needs a 'kind'")
    for k in ("mode", "identical", "encoding_parity", "forced", "diffs"):
        if k not in payload["tokenizer"]:
            raise RecordSchemaError(f"locked schema: tokenizer.{k} missing")
    for k in ("mean", "se", "floor_mean", "p50", "p95", "p99", "p999", "max"):
        if k not in payload["kld"]:
            raise RecordSchemaError(f"locked schema: kld.{k} missing")
    for k in ("mean", "se", "rms"):
        if k not in payload["delta_p"]:
            raise RecordSchemaError(f"locked schema: delta_p.{k} missing")
    for k in ("top1", "top5"):
        if k not in payload["agreement"]:
            raise RecordSchemaError(f"locked schema: agreement.{k} missing")
    for k in ("corpus", "num_samples", "max_seq_len", "seed"):
        if k not in payload["calibration"]:
            raise RecordSchemaError(f"locked schema: calibration.{k} missing")
