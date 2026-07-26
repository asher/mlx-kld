"""``mlx-kld score`` - KLD-score a student checkpoint (MLX safetensors or
GGUF) against a teacher."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NoReturn

from .. import __version__
from .._constants import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CACHE_MAX_GB,
    DEFAULT_DATASET,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    SCHEMA_VERSION,
)
from .._io import write_json_atomic, write_text_atomic
from .._log import info, warn


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "score",
        help="KLD-score a student checkpoint against a teacher",
        description="Score a quantized (or otherwise modified) student against "
        "a full-precision teacher. The student is an MLX safetensors directory "
        "or a .gguf file (the latter needs the [gguf] extra). Defaults follow "
        "the conventional llama.cpp protocol (n_ctx=512, second-half scoring).",
    )
    p.add_argument("teacher", help="HF id or local path to the (full-precision) teacher")
    p.add_argument("student", type=Path,
                   help="Local path to the quantized student: an MLX "
                        "safetensors directory, or a .gguf file")
    p.add_argument("--hf-source", type=str, default=None, metavar="ID_OR_DIR",
                   help="GGUF students only: load the student tokenizer from "
                        "this HF id or local dir instead of synthesizing it "
                        "from GGUF metadata. Teacher and student then share "
                        "the HF tokenizer, holding tokenization constant so "
                        "the test isolates the weights.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help=f"HF dataset (default: {DEFAULT_DATASET}). Accepts "
                        "`path:name` for HF subset configs. Chat-format corpora "
                        "(with a `messages` column) auto-render through the "
                        "scoring tokenizer's chat template. That is the "
                        "student's, which for a normal quant is identical to "
                        "the teacher's.")
    p.add_argument("--stream-dataset", action="store_true",
                   help="Pull dataset rows over the network instead of "
                        "downloading the whole split first. Needed for "
                        "web-scale corpora (fineweb-edu's sample-10BT is ~40 GB "
                        "on disk). It reads the same rows in the same order, so "
                        "the token stream and its hash are unchanged.")
    p.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                   help="Number of sequences to score (default: %(default)s)")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help=f"Tokens per sequence (default: {DEFAULT_MAX_SEQ_LEN}, "
                        "or 2048 with --long-context)")
    p.add_argument("--long-context", action="store_true",
                   help="Score full sequences at max_seq_len=2048 instead of "
                        "the default protocol. Results are not comparable "
                        "with default-protocol runs, and the cache entry is "
                        "4x larger")
    p.add_argument("--score-window", default=None, metavar="START:END",
                   help="Override the score window as `start:end` (default: "
                        "second half, or the full sequence under "
                        "--long-context)")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                   help="Top-K to cache per teacher position (default: "
                        "%(default)s). Any value up to the teacher vocab minus "
                        "one is accepted. The entry costs about 6 bytes per "
                        "position per slot, so this scales the cache linearly. "
                        "The default sits near the measured minimum of the "
                        "reconstruction floor. Run `mlx-kld floor-sweep` to "
                        "find it for your teacher, since a larger K does not "
                        "reliably mean a lower floor")
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size for a fresh teacher pass (default: 1). An "
                        "existing cache replays at its own batch size")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Seed for the calibration-chunk permutation (default: "
                        "%(default)s). Ingest reads the head of the "
                        "corpus (~525k tokens at the default "
                        "protocol), so this permutes within that "
                        "window rather than sampling the corpus as a "
                        "whole.")
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                   help=f"Teacher-logits cache root (default: {DEFAULT_CACHE_DIR})")
    p.add_argument("--cache-max-gb", type=float, default=DEFAULT_CACHE_MAX_GB,
                   help="Evict least-recently-used cache entries to keep the "
                        "cache root under this size (default: %(default)s GB, "
                        "where 0 disables auto-eviction)")
    p.add_argument("--rebuild-cache", action="store_true",
                   help="Force a teacher pass even if a cached one exists")
    p.add_argument("--allow-tokenizer-mismatch", action="store_true",
                   help="Force past a detected ENCODING divergence between the "
                        "teacher and student tokenizers (normally fatal)")
    p.add_argument("--md", type=Path, default=None, metavar="PATH",
                   help="Markdown report path (default: stdout)")
    p.add_argument("--card", type=Path, default=None, metavar="PATH",
                   help="Also write a short Markdown block sized for a model "
                        "card. It carries the headline metrics, the spec they "
                        "are comparable within, and the command that "
                        "reproduces them, but no local paths, so it is safe "
                        "to publish")
    p.add_argument("--out-dir", type=Path, default=None, metavar="DIR",
                   help="Results root for JSON records (default: "
                        "$MLX_KLD_RESULTS or ./kld-results)")
    p.add_argument("--json", type=Path, default=None, dest="json_path",
                   metavar="PATH",
                   help="Exact JSON record path (default: <out-dir>/"
                        "<teacher-slug>/<student-slug>.<digest8>.json)")
    p.add_argument("--teacher-precision", default=None,
                   help="Override the teacher precision shown in the report. "
                        "This is normally unnecessary, because the teacher "
                        "pass measures the dominant weight dtype and records "
                        "that. The label is used only for a cache entry built "
                        "before precision was measured (bfloat16 by default)")
    p.set_defaults(func=cmd)


def _die(msg: str) -> NoReturn:
    """Exit with the CLI's one error format (the same ``error:`` prefix the
    dispatcher gives library errors), so argument failures and library failures
    read alike."""
    sys.exit(f"error: {msg}")


def cmd(args: argparse.Namespace) -> int:
    from mlx_lm.utils import load_tokenizer

    from ..cache import (
        cache_is_valid,
        cache_total_bytes,
        enforce_budget,
        ensure_cache_entry,
        entry_lock,
        hf_revision_of,
        human_bytes,
    )
    from ..errors import MlxKldError
    from ..measure import load_provenance, measure_gguf_student, measure_student
    from ..models import is_gguf_path
    from ..report import (
        build_locked_json,
        render_card,
        render_markdown,
        resolve_results_dir,
        results_json_path,
        validate_locked_schema,
    )
    from ..scoring import student_pass
    from ..tokenizer import (
        _hf_inner,
        cached_stream_status,
        encoding_parity,
        load_calibration_tokens,
        load_gguf_tokenizer,
        tokenizer_hash,
        tokenizer_metadata_diffs,
    )

    student_dir = args.student
    using_gguf = is_gguf_path(str(student_dir))
    if not using_gguf:
        if student_dir.suffix.lower() == ".gguf":
            if student_dir.exists():
                _die(f"GGUF student is not a regular file: {student_dir}")
            _die(f"GGUF student does not exist: {student_dir}")
        if not student_dir.is_dir():
            _die(f"student is not a directory: {student_dir}")
        if args.hf_source:
            _die("--hf-source only applies to GGUF students")

    # Validate the numeric protocol arguments before anything expensive runs.
    # (--top-k 0 would otherwise slice the entire vocab via `[..., -0:]`, and
    # --batch-size 0 divides by zero computing the batch count.)
    for flag, value in (("--num-samples", args.num_samples),
                        ("--top-k", args.top_k),
                        ("--batch-size", args.batch_size)):
        if value <= 0:
            _die(f"{flag} must be > 0, got {value}")

    # Resolve protocol-mode defaults.
    if args.max_seq_len is None:
        max_seq_len = 2048 if args.long_context else DEFAULT_MAX_SEQ_LEN
    else:
        max_seq_len = args.max_seq_len
    if max_seq_len <= 0:
        _die(f"--max-seq-len must be > 0, got {max_seq_len}")
    if args.score_window is not None:
        try:
            sa, sb = (int(x) for x in args.score_window.split(":"))
        except ValueError:
            _die(f"--score-window must be `start:end`, got: {args.score_window!r}")
        score_window = (sa, sb)
    elif args.long_context:
        score_window = (0, max_seq_len)
    else:
        score_window = (max_seq_len // 2, max_seq_len)
    if not (0 <= score_window[0] < score_window[1] <= max_seq_len):
        _die(
            f"--score-window {score_window} must satisfy "
            f"0 <= start < end <= max_seq_len ({max_seq_len})"
        )
    args.max_seq_len = max_seq_len
    info(f"protocol: max_seq_len={max_seq_len}, top_k={args.top_k}, "
         f"score_window={list(score_window)}, dataset={args.dataset}")

    info(f"loading teacher tokenizer: {args.teacher}")
    teacher_tokenizer = load_tokenizer(args.teacher)

    # top-K must leave a non-empty tail (V - K > 0). Clamp against the teacher
    # vocab so common 32k-vocab families don't crash on the default top-K.
    #
    # The tokenizer's vocab is a proxy for the logits width, and the two can
    # differ in either direction (a padded lm_head is wider; some tokenizers
    # report wider than the head). teacher_pass re-clamps against the measured
    # width, which is authoritative. This early pass exists so the clamp shows
    # up in the cache key and the user sees it before a long teacher pass starts.
    #
    # Take the *smaller* of the two available readings. Erring low costs a
    # handful of top-K slots out of tens of thousands, which is immaterial next
    # to the uniform tail. Erring high is not symmetric: teacher_pass would
    # re-clamp, and the entry would then be filed under a key naming a K that
    # was never cached.
    inner = _hf_inner(teacher_tokenizer)
    try:
        inner_len = len(inner)
    except TypeError:
        inner_len = None
    sizes = [s for s in (getattr(inner, "vocab_size", None), inner_len) if s]
    teacher_vocab = min(sizes) if sizes else None
    if teacher_vocab and args.top_k >= teacher_vocab:
        clamped = teacher_vocab - 1
        warn(f"--top-k {args.top_k} >= teacher vocab {teacher_vocab}, "
             f"clamping to {clamped}")
        args.top_k = clamped

    if using_gguf and args.hf_source:
        # GGUF mode 2: teacher and student share an HF tokenizer, holding the
        # tokenization constant so the test isolates the weights.
        tok_mode = "hf-source"
        info(f"GGUF tokenizer mode: hf-source ({args.hf_source}). The student "
             "shares the HF tokenizer with the teacher (weights-only test).")
        student_tokenizer = load_tokenizer(args.hf_source)
    elif using_gguf:
        # GGUF mode 1 (default): synthesize the student tokenizer from GGUF
        # metadata, the same one the loaded GGUF model uses. This tests the
        # GGUF exactly as it ships, tokenizer included; it can differ
        # cosmetically from the teacher's HF tokenizer (reported below).
        student_tokenizer, gguf_arch = load_gguf_tokenizer(student_dir)
        tok_mode = "synthesized"
        info(f"GGUF tokenizer mode: synthesized from GGUF metadata "
             f"(arch={gguf_arch}). Tests the tokenizer the GGUF ships with.")
    else:
        tok_mode = "mlx-student"
        info(f"loading student tokenizer: {student_dir}")
        student_tokenizer = load_tokenizer(str(student_dir))

    # KLD is well-defined iff text -> token-ids agrees between the two
    # tokenizers. Metadata may differ (a GGUF-synthesized tokenizer reports the
    # model's padded vocab and the GGUF's bos id) without changing the ids of
    # the scored corpus; gate only on a direct encoding probe.
    teacher_tok_hash = tokenizer_hash(teacher_tokenizer)
    student_tok_hash = tokenizer_hash(student_tokenizer)
    tok_identical = teacher_tok_hash == student_tok_hash
    tok_diffs = tokenizer_metadata_diffs(teacher_tokenizer, student_tokenizer)
    if not tok_identical:
        info("\n".join(
            [f"TOKENIZER DIFFERENCE (teacher vs student), mode={tok_mode}:"]
            + [f"  {d}" for d in tok_diffs]
            + [f"  teacher hash : {teacher_tok_hash}",
               f"  student hash : {student_tok_hash}"]))

    enc_ok, enc_diff = encoding_parity(teacher_tokenizer, student_tokenizer)
    if not enc_ok:
        probe, ti_ids, si_ids = enc_diff
        msg = (
            "TOKENIZER ENCODING DIVERGES: the student tokenizer produces "
            "different token ids than the teacher's, so the teacher would be "
            "scored on a tokenization it never saw and the KLD is meaningless.\n"
            f"  probe       : {probe!r}\n"
            f"  teacher ids : {ti_ids[:24]}\n"
            f"  student ids : {si_ids[:24]}"
        )
        if args.allow_tokenizer_mismatch:
            info(msg + "\n  (forced past via --allow-tokenizer-mismatch)")
        elif using_gguf and not args.hf_source:
            _die(msg + "\n  Pass --hf-source <id-or-dir> (e.g. the teacher) to "
                 "score with that HF tokenizer instead, or "
                 "--allow-tokenizer-mismatch to force.")
        else:
            _die(msg + "\n  Use --allow-tokenizer-mismatch to force.")
    elif not tok_identical:
        info("encoding parity verified: text -> token-ids is identical despite "
             "the metadata differences above, so the KLD is well-defined. "
             f"Tokenization under test: {tok_mode}.")

    tokens = load_calibration_tokens(
        student_tokenizer, args.dataset, args.num_samples, args.max_seq_len,
        args.seed, streaming=args.stream_dataset,
    )
    info(f"calibration: {tokens.shape[0]} sequences x {tokens.shape[1]} tokens")
    # Content witness for the record: pins the exact scored token stream, so a
    # silent upstream dataset revision (or tokenizer change) is detectable even
    # when the recorded spec is identical.
    import numpy as np
    student_corpus_tokens_hash = hashlib.sha256(
        np.ascontiguousarray(np.asarray(tokens, dtype=np.int32)).tobytes()
    ).hexdigest()[:16]

    # One shared lifecycle (validity, tokenizer gate, HIT/MISS/rebuild, budget,
    # locking) for the CLI and the Python API. See cache.ensure_cache_entry.
    # CacheMismatchError propagates to the dispatcher's single error renderer.
    cache_dir, manifest, cache_status, teacher_pass_seconds = ensure_cache_entry(
        teacher_path=args.teacher,
        tokens=tokens,
        teacher_tok_hash=teacher_tok_hash,
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        top_k=args.top_k,
        batch_size=args.batch_size,
        cache_root=args.cache_dir,
        rebuild=args.rebuild_cache,
        score_window=score_window,
        cache_max_gb=args.cache_max_gb,
    )
    elapsed_phase = ("warm: student only, cache HIT" if cache_status == "HIT"
                     else "cold: teacher + student")
    key = cache_dir.name

    # Student pass + online KLD, under the entry's shared lock so a concurrent
    # run's cache eviction can never delete shards mid-replay.
    t1 = time.time()
    with entry_lock(args.cache_dir, key, exclusive=False):
        # Re-read the manifest now that the entry is pinned. The lifecycle
        # released its lock before returning, and in that window another
        # process can evict *and rebuild* this key from a different student's
        # token stream, meaning same key, same shard count, different corpus.
        # The
        # manifest returned above would then describe shards that are no
        # longer on disk, and every provenance field in the record (corpus
        # hash, top_k, vocab) would be a fiction. What is on disk under the
        # lock is what gets scored, so that is what the record must report.
        try:
            manifest = json.loads((cache_dir / "manifest.json").read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise MlxKldError(
                f"cache entry {cache_dir} became unreadable before the student "
                f"pass ({e}). Another process likely evicted it. Rerun."
            ) from e
        valid, reason = cache_is_valid(cache_dir, manifest["num_batches"])
        if not valid:
            raise MlxKldError(
                f"cache entry {cache_dir} became invalid before the student "
                f"pass ({reason}). Another process likely evicted it. Rerun."
            )
        manifest["score_window"] = [int(score_window[0]), int(score_window[1])]
        # Whether the stream actually scored is the one this student's own
        # tokenizer produces. Computed against the on-disk hash for every
        # status, not just HIT: after a rebuild-under-us even a MISS can end up
        # replaying somebody else's stream. None only for a pre-hash cache.
        stream_is_students, stream_warning = cached_stream_status(
            manifest.get("corpus_tokens_hash"), student_corpus_tokens_hash, tok_mode,
        )
        if stream_warning:
            warn(stream_warning)
        metrics = student_pass(
            student_dir, cache_dir, manifest, batch_size=args.batch_size,
        )
    student_pass_seconds = time.time() - t1
    elapsed_seconds = teacher_pass_seconds + student_pass_seconds
    # teacher_pass may have clamped top-K against the measured logits width;
    # the manifest carries the K actually cached, and the record must report
    # that one, not the requested value.
    cache_top_k = manifest.get("top_k", args.top_k)

    # The scored stream is the one in the cache shards; the manifest hashes it at
    # build time. Fall back to the student re-tokenization only for a pre-hash
    # cache (older manifest lacking the field).
    corpus_tokens_hash = manifest.get("corpus_tokens_hash") or student_corpus_tokens_hash

    if using_gguf:
        student_meta = measure_gguf_student(student_dir)
        provenance = None
    else:
        student_meta = measure_student(student_dir)
        provenance = load_provenance(student_dir)
    student_block = {
        "path": str(student_dir),
        "format": student_meta["format"],
        "size_bytes": student_meta["size_bytes"],
        "effective_bpw": student_meta["effective_bpw"],
        "n_params": student_meta["n_params"],
        "quantization": student_meta["quantization"],
        # Bits per weight the scoring pass actually loads (MTP stack and
        # vision/audio towers excluded). Absent for measurers that don't
        # report it, so consumers treat missing as "not measured".
        **{k: student_meta[k] for k in
           ("scored_n_params", "scored_bytes", "scored_bpw")
           if k in student_meta},
        # Non-locked free-form provenance (quantizer sidecar folded in);
        # always present so JSON consumers get one "missing" convention (null).
        "provenance": provenance,
    }
    report = {
        "teacher": {
            "path": args.teacher,
            # The manifest's revision was captured at build time; trust it
            # even when None (a local path that encodes no sha, or a hub that
            # was unreachable then). Either way, do not repeat the call.
            # Only manifests written before the field existed look it up fresh.
            "revision": (manifest["teacher_revision"]
                         if "teacher_revision" in manifest
                         else hf_revision_of(args.teacher)),
            # Measured at teacher-pass time and carried in the manifest. The
            # flag is a fallback for entries built before that was recorded,
            # so a published record names the precision actually loaded rather
            # than whatever the default label happened to say.
            "precision": (manifest.get("teacher_precision")
                          or args.teacher_precision
                          or "bfloat16"),
            "tokenizer_hash": teacher_tok_hash,
        },
        "student": student_block,
        "tokenizer": {
            "mode": tok_mode,
            "identical": tok_identical,
            "encoding_parity": enc_ok,
            "diffs": tok_diffs,
            "forced": bool(args.allow_tokenizer_mismatch and not enc_ok),
            # False when a cache HIT replayed a stream this student's tokenizer
            # would not produce, so `mode` describes what was requested and this
            # describes what was actually scored. None for a pre-hash cache.
            "stream_is_students": stream_is_students,
        },
        "calibration": {
            "corpus": args.dataset,
            "num_samples": args.num_samples,
            "max_seq_len": args.max_seq_len,
            "seed": args.seed,
            "top_k": cache_top_k,
            "score_window": list(score_window),
            "long_context": bool(args.long_context),
            "corpus_tokens_hash": corpus_tokens_hash,
        },
        # setdefault/fallback: locked keys stay present even if a caller's
        # aggregator predates the se/delta_p/floor metrics.
        "kld": {**{"se": None, "floor_mean": None}, **metrics["kld"]},
        "delta_p": metrics.get("delta_p") or {"mean": None, "se": None, "rms": None},
        "agreement": metrics["agreement"],
        "tokens_scored": metrics["tokens_scored"],
        # Non-locked: positions dropped because the student produced a
        # non-finite KLD there. Nonzero is a red flag the mean alone hides
        # (dropping the worst positions *improves* the mean).
        "tokens_dropped_nonfinite": metrics.get("tokens_dropped_nonfinite", 0),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "elapsed_phase": elapsed_phase,
        "scorer_version": f"mlx-kld {__version__}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "by_position": metrics.get("by_position"),
        "kld_histogram": metrics.get("kld_histogram"),
        "cache": {
            "dir": str(cache_dir),
            "status": cache_status,
            "top_k": cache_top_k,
        },
    }

    # Records land under the results root, never inside the model directory.
    json_path = args.json_path
    if json_path is None:
        json_path = results_json_path(
            resolve_results_dir(args.out_dir), args.teacher, student_dir,
            report["calibration"], tok_mode,
        )
    report["json_path"] = str(json_path)

    md = render_markdown(report)
    if args.md:
        write_text_atomic(args.md, md)
        info(f"Markdown report written to {args.md}")
    else:
        print(md)
    if args.card:
        write_text_atomic(args.card, render_card(report))
        info(f"model-card block written to {args.card}")

    payload = build_locked_json(report)
    validate_locked_schema(payload)
    write_json_atomic(json_path, payload)

    # Self-managing cache: keep the root under budget, never evicting the entry
    # this run just used.
    evicted = enforce_budget(args.cache_dir, args.cache_max_gb, protect_keys=(key,))
    total = cache_total_bytes(args.cache_dir)
    if evicted:
        freed = sum(sz for _, sz in evicted)
        info(f"cache: evicted {len(evicted)} LRU entr"
             f"{'y' if len(evicted) == 1 else 'ies'} ({human_bytes(freed)} freed)")
    info(f"cache root {args.cache_dir}: {human_bytes(total)}")
    info(f"done ({elapsed_phase}, {elapsed_seconds:.1f}s)")
    # The artifact last, so the record path is the final line on screen.
    info(f"JSON record (schema_version={SCHEMA_VERSION}): {json_path}")
    return 0
