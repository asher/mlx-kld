"""Teacher top-K logit cache: build, validate, and self-manage.

Each ``(teacher x calibration spec x top_k)`` gets a cache entry under
``~/.mlx-kld-cache/<key>/`` holding batched safetensors shards + a manifest.
An entry costs ~6 bytes per position per top-K slot (bf16 log-prob + int32
index), so the default config (512 x 512 positions, K=32768) writes ~51.5 GB.
The cache is therefore self-managing: the manifest tracks ``size_bytes`` +
``last_used`` and a budget evicts whole least-recently-used entries both
before a build, to make room for the new entry, and again after each run (the
default 120 GB budget keeps two default-config teachers resident; the entry in
use is never evicted). The ``mlx-kld cache`` subcommand exposes manual control
on top of the same primitives.

Every position is cached, not just the scored window. Under the default
second-half protocol that is 2x what any single run scores, roughly 25.8 GB
per entry that is read back but never contributes to the score. It buys
re-windowing for free: ``score_window`` is deliberately absent from
``cache_key``, so ``--score-window`` and ``--long-context`` reshape the
measurement without a teacher rebuild, while ``run_digest`` and ``compare``'s
run key both include the window so records never blend across it. Worth
revisiting only if disk pressure outweighs that.

Cross-process coordination is an advisory per-entry ``flock``
(:func:`entry_lock`): a build holds it exclusively, a replay holds it shared,
and every eviction path (``enforce_budget``, ``gc``, ``clear_entries``) skips
an entry whose lock another live process holds. That makes queued multi-run
scoring safe: run B's replay cannot be evicted mid-read by run A's post-run
budget pass, and two simultaneous misses on one key build it once.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ._constants import CACHE_FORMAT_VERSION, DEFAULT_CACHE_DIR
from ._io import write_json_atomic
from ._log import info, warn
from .errors import CacheMismatchError
from .kld_math import kld_from_topk
from .models import _extract_logits, _load_model, _make_fresh_cache
from .tokenizer import load_calibration_tokens, tokenizer_hash


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _teacher_identity(teacher_path: str) -> str:
    """Canonical spelling of a teacher for cache keying.

    Local paths resolve, so ``./teacher``, ``teacher/``, ``~/teacher``, and the
    absolute spelling key one entry instead of minting a ~51.5 GB entry each
    (the record layer resolves student paths for the same reason). Hub ids pass
    through untouched.
    """
    p = Path(teacher_path).expanduser()
    if p.exists():
        try:
            return str(p.resolve())
        except OSError:
            return str(p)
    return teacher_path


def cache_key(
    teacher_path: str,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    top_k: int,
) -> str:
    payload = "|".join([
        _teacher_identity(teacher_path),
        dataset_name,
        str(num_samples),
        str(max_seq_len),
        str(seed),
        str(top_k),
        f"v{CACHE_FORMAT_VERSION}",
    ])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def cache_is_valid(cache_dir: Path, expected_num_batches: int) -> tuple[bool, str]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        return False, "no manifest"
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return False, "manifest corrupt"
    if manifest.get("format_version") != CACHE_FORMAT_VERSION:
        return False, (
            f"format_version mismatch "
            f"({manifest.get('format_version')} != {CACHE_FORMAT_VERSION})"
        )
    declared = manifest.get("num_batches")
    if declared != expected_num_batches:
        return False, f"num_batches mismatch ({declared} != {expected_num_batches})"
    for i in range(declared):
        if not (cache_dir / f"batch-{i:05d}.safetensors").exists():
            return False, f"missing batch-{i:05d}.safetensors"
    return True, "ok"


def manifest_batch_size(cache_dir: Path) -> int | None:
    """The batch size an existing cache entry was written at, else None.

    Shards replay at their own batch size, so a differing --batch-size must not
    read as a MISS and trigger a full teacher rebuild."""
    mp = cache_dir / "manifest.json"
    if not mp.is_file():
        return None
    try:
        return json.loads(mp.read_text()).get("batch_size")
    except (json.JSONDecodeError, OSError):
        return None


def _looks_like_commit_sha(name: str) -> bool:
    return len(name) == 40 and all(c in "0123456789abcdefABCDEF" for c in name)


def measured_precision(model) -> str | None:
    """The dominant weight dtype of a loaded model, by byte count.

    Recorded at teacher-pass time so ``teacher.precision`` in the record states
    what was actually loaded. The alternative, a command-line label, is right
    only as often as the user remembers to set it, and a published record that
    misnames its teacher's precision is worse than one that says nothing.

    Returns None if the parameter tree cannot be walked, which leaves the
    caller's label in place.
    """
    try:
        from mlx.utils import tree_flatten
    except ImportError:
        return None
    by_dtype: dict[str, int] = {}
    try:
        for _name, arr in tree_flatten(model.parameters()):
            dtype = getattr(arr, "dtype", None)
            nbytes = getattr(arr, "nbytes", None)
            if dtype is None or nbytes is None:
                continue
            # mx.Dtype has no .name; its str() is "mlx.core.<dtype>".
            name = str(dtype).rsplit(".", 1)[-1]
            by_dtype[name] = by_dtype.get(name, 0) + int(nbytes)
    except Exception:
        # Precision is a provenance nicety. Never fail a teacher pass over it.
        return None
    if not by_dtype:
        return None
    return max(by_dtype, key=by_dtype.get)


def hf_revision_of(path_or_id: str) -> str | None:
    """Best-effort commit SHA for ``path_or_id``.

    Local HF snapshots encode it in the path (snapshot_download stores commits
    at ``.../models--ORG--NAME/snapshots/<sha>/``); hub ids resolve via a light
    API call, None when offline or not a hub id.

    The bare-sha fallback requires 40 *hex* characters. Length alone would report
    any 40-character directory name as a git revision.
    """
    p = Path(path_or_id)
    if p.exists():
        if p.parent.name == "snapshots" or _looks_like_commit_sha(p.name):
            return p.name
        return None
    try:
        from huggingface_hub import HfApi

        # Bounded: this runs at the end of a potentially hours-long score, and
        # a hung hub API must not stall the run past a short, fixed wait.
        return HfApi().model_info(path_or_id, timeout=5.0).sha
    except Exception:
        return None


# ---------- self-managing cache layer ----------

def _dir_size_bytes(d: Path) -> int:
    # A concurrent eviction (or a manual rm -rf) can delete files between the
    # glob and the stat; a vanished file counts as zero rather than raising.
    total = 0
    for f in d.glob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def human_bytes(n: float) -> str:
    # Decimal units, matching the report/compare tables' GB (bytes / 1e9).
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1000
    return f"{n:.1f} TB"


def estimate_entry_bytes(num_samples: int, max_seq_len: int, top_k: int) -> int:
    """Upfront size estimate for one cache entry: per position, top_k bf16
    log-probs (2 B) + top_k int32 indices (4 B), plus the token/mask/floor
    stream."""
    per_position = 6 * top_k + 9
    return num_samples * max_seq_len * per_position


def _entry_last_used(manifest: dict, entry_dir: Path) -> datetime:
    raw = manifest.get("last_used") or manifest.get("created_at")
    if raw:
        try:
            dt = datetime.fromisoformat(raw)
            # A hand-built manifest may carry a naive timestamp; normalize so
            # sorting and age math never mix naive and aware datetimes.
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    # Hand-built caches missing last_used: fall back to the manifest's mtime.
    ts = (entry_dir / "manifest.json").stat().st_mtime
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def entry_info(entry_dir: Path) -> dict | None:
    """Summarize one cache entry, or None if it isn't a valid entry."""
    manifest_path = entry_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return {
        "key": entry_dir.name,
        "path": entry_dir,
        "size_bytes": _dir_size_bytes(entry_dir),
        "last_used": _entry_last_used(manifest, entry_dir),
        "teacher_path": manifest.get("teacher_path"),
        "dataset": manifest.get("dataset"),
        "top_k": manifest.get("top_k"),
        "num_samples": manifest.get("num_samples"),
        "max_seq_len": manifest.get("max_seq_len"),
    }


def list_entries(cache_root: Path) -> list[dict]:
    """All cache entries under ``cache_root``, newest-used first."""
    if not cache_root.is_dir():
        return []
    entries = [
        e for d in sorted(cache_root.iterdir()) if d.is_dir()
        if (e := entry_info(d)) is not None
    ]
    entries.sort(key=lambda e: e["last_used"], reverse=True)
    return entries


def cache_total_bytes(cache_root: Path) -> int:
    return sum(e["size_bytes"] for e in list_entries(cache_root))


def touch_entry(cache_dir: Path) -> None:
    """Update an entry's ``last_used`` to now (called on cache HIT)."""
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return
    manifest["last_used"] = _now_iso()
    try:
        write_json_atomic(manifest_path, manifest)
    except OSError:
        pass


# ---------- cross-process coordination ----------

def _lock_path(cache_root: Path, key: str) -> Path:
    # A sibling of the entry directory, not inside it: eviction rmtree's the
    # entry while holding this lock, so the lock file has to sit outside the
    # tree being removed.
    #
    # Lock files are deliberately never unlinked. flock binds to an inode, not
    # a path: unlinking one while a process is blocked in flock() hands that
    # process a lock on an orphaned inode, while the next arrival creates a
    # fresh file at the same path and locks that instead, leaving two "exclusive"
    # holders of one key, which is exactly the mutual exclusion this exists to
    # provide. They are 0-byte and `list_entries` only walks directories, so
    # strays cost nothing and are invisible to every accounting path.
    return cache_root / f".{key}.lock"


@contextmanager
def entry_lock(cache_root: Path, key: str, exclusive: bool):
    """Advisory ``flock`` over one cache entry.

    A build (teacher pass, rebuild) holds it exclusively; a replay (student
    pass) holds it shared. Eviction paths take a non-blocking exclusive probe
    and skip entries another live process holds, so a concurrent run can never
    delete shards out from under a reader. Blocking: a second process that
    misses on the same key waits here, then re-checks validity and finds the
    entry the first one built.

    API callers replaying a cache with :func:`~mlx_kld.score_loaded_student`
    should hold ``entry_lock(cache_root, cache_dir.name, exclusive=False)``
    across the replay for the same protection the CLI gets.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    # O_RDONLY is enough for flock and keeps a cache root shared between users
    # usable: another user's 0644 lock file is still readable.
    fd = os.open(_lock_path(cache_root, key), os.O_CREAT | os.O_RDONLY, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield
    finally:
        os.close(fd)  # closing the fd releases the flock


def _try_remove_entry(cache_root: Path, entry: dict) -> bool:
    """Remove one entry unless its lock is held. True only when *this* call
    freed the entry's bytes, so callers can trust the freed totals they report.
    """
    try:
        fd = os.open(_lock_path(cache_root, entry["key"]),
                     os.O_CREAT | os.O_RDONLY, 0o666)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            # flock binds to the open file description, so a lock this same
            # process holds on another fd also lands here, hence "locked"
            # rather than naming another process.
            info(f"cache: skipping {entry['key']} (locked, in use)")
            return False
        if not Path(entry["path"]).exists():
            # Already gone (a concurrent evictor won the race). Nothing was
            # freed here, so it must not be counted against the budget.
            return False
        shutil.rmtree(entry["path"], ignore_errors=True)
        if Path(entry["path"]).exists():
            # Partial failure (permissions, a busy file): the entry still holds
            # disk, so it must not be reported as freed, or accounted so.
            warn(f"cache: could not fully remove {entry['key']}")
            return False
        return True
    finally:
        os.close(fd)


def enforce_budget(
    cache_root: Path,
    max_gb: float,
    protect_keys: tuple[str, ...] = (),
) -> list[tuple[str, int]]:
    """Evict least-recently-used entries until the cache is under ``max_gb``.

    ``max_gb <= 0`` disables eviction. Entries in ``protect_keys`` (e.g. the one
    the current run just used) are never evicted, and neither is an entry whose
    lock another live process holds. Returns the list of evicted
    ``(key, size_bytes)``.
    """
    if max_gb <= 0:
        return []
    budget = int(max_gb * 1e9)
    entries = list_entries(cache_root)
    total = sum(e["size_bytes"] for e in entries)
    if total <= budget:
        return []
    # Oldest-used first; protected entries are not eviction candidates.
    candidates = sorted(
        (e for e in entries if e["key"] not in protect_keys),
        key=lambda e: e["last_used"],
    )
    evicted: list[tuple[str, int]] = []
    for e in candidates:
        if total <= budget:
            break
        if not _try_remove_entry(cache_root, e):
            continue
        total -= e["size_bytes"]
        evicted.append((e["key"], e["size_bytes"]))
    return evicted


def gc(
    cache_root: Path,
    max_gb: float | None = None,
    older_than_days: float | None = None,
) -> list[tuple[str, int]]:
    """Prune entries older than ``older_than_days`` and/or down to ``max_gb``.
    Returns evicted ``(key, size_bytes)``."""
    evicted: list[tuple[str, int]] = []
    if older_than_days is not None:
        cutoff = datetime.now(timezone.utc).timestamp() - older_than_days * 86400
        for e in list_entries(cache_root):
            if e["last_used"].timestamp() < cutoff and _try_remove_entry(cache_root, e):
                evicted.append((e["key"], e["size_bytes"]))
    if max_gb is not None:
        evicted.extend(enforce_budget(cache_root, max_gb))
    return evicted


def clear_entries(cache_root: Path, teacher_substr: str | None = None) -> list[tuple[str, int]]:
    """Remove all entries, or those whose teacher path contains ``teacher_substr``."""
    removed: list[tuple[str, int]] = []
    for e in list_entries(cache_root):
        if teacher_substr and teacher_substr not in (e["teacher_path"] or ""):
            continue
        if _try_remove_entry(cache_root, e):
            removed.append((e["key"], e["size_bytes"]))
    return removed


# ---------- teacher pass ----------

def teacher_pass(
    teacher_path: str,
    tokens,                # mx.array (N, L) int32
    batch_size: int,
    top_k: int,
    cache_dir: Path,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    teacher_tokenizer_hash: str,
    score_window: tuple[int, int] | None = None,
):
    """Forward the teacher and write top-K cache shards + manifest."""
    import mlx.core as mx
    from tqdm import tqdm

    info(f"loading teacher: {teacher_path}")
    teacher_model, teacher_config = _load_model(teacher_path, lazy=True)
    teacher_model.eval()

    config_vocab = teacher_config.get("vocab_size")
    teacher_precision = measured_precision(teacher_model)

    cache_dir.mkdir(parents=True, exist_ok=True)
    N, L = tokens.shape
    num_batches = (N + batch_size - 1) // batch_size

    info(f"teacher pass: {N} sequences x {L} tokens, "
         f"batch_size={batch_size}, K={top_k}")

    measured_vocab = None
    for i in tqdm(range(num_batches), desc="teacher", file=sys.stderr):
        s = i * batch_size
        e = min(s + batch_size, N)
        batch = tokens[s:e]                                 # (B, L) int32
        cache = _make_fresh_cache(teacher_model)
        logits = _extract_logits(teacher_model(batch, cache=cache))  # (B, L, V)
        if measured_vocab is None:
            measured_vocab = int(logits.shape[-1])
            # The CLI clamps top_k against the tokenizer's vocab_size, which is
            # only a proxy for the logits width. Re-clamp against the width we
            # actually measured, so a tokenizer that over-reports cannot reach
            # argpartition with kth >= V and fail with a raw MLX error.
            if top_k >= measured_vocab:
                clamped = measured_vocab - 1
                warn(f"top-k {top_k} >= teacher logits width {measured_vocab}, "
                     f"clamping to {clamped}")
                top_k = clamped
        lf = logits.astype(mx.float32)
        log_softmax = lf - mx.logsumexp(lf, axis=-1, keepdims=True)
        idx_part = mx.argpartition(log_softmax, kth=-top_k, axis=-1)[..., -top_k:]
        vals_part = mx.take_along_axis(log_softmax, idx_part, axis=-1)
        order = mx.argsort(-vals_part, axis=-1)
        top_idx = mx.take_along_axis(idx_part, order, axis=-1).astype(mx.int32)
        top_log_p = mx.take_along_axis(vals_part, order, axis=-1).astype(mx.bfloat16)
        # Per-position reconstruction floor: the teacher scored against its own
        # bf16-rounded top-K reconstruction, exactly what the student pass
        # replays. Lets score report how much of a small KLD is floor.
        floor_kld, _f1, _f5 = kld_from_topk(top_log_p, top_idx, lf, int(lf.shape[-1]))
        floor_kld = floor_kld.astype(mx.float32)
        mx.eval(top_idx, top_log_p, floor_kld)
        # Calibration sequences are fixed-length slices of one concatenated token
        # stream, so every position is real and this is all-ones by construction.
        # Kept in the shard because the scorer's mask plumbing is what would let
        # a future padded/ragged corpus work without a cache-format bump.
        attention_mask = mx.ones(batch.shape, dtype=mx.bool_)
        mx.save_safetensors(
            str(cache_dir / f"batch-{i:05d}.safetensors"),
            {
                "top_k_log_softmax": top_log_p,
                "top_k_indices": top_idx,
                "token_ids": batch.astype(mx.int32),
                "attention_mask": attention_mask,
                "floor_kld": floor_kld,
            },
        )

    if score_window is None:
        score_window = (0, max_seq_len)
    now = _now_iso()
    # Witness the exact token stream that got scored, so every later student
    # reports the cached stream it actually replayed against.
    import numpy as np
    corpus_tokens_hash = hashlib.sha256(
        np.ascontiguousarray(np.asarray(tokens, dtype=np.int32)).tobytes()
    ).hexdigest()[:16]
    # Write manifest last so a partial cache won't pass cache_is_valid().
    manifest = {
        "format_version": CACHE_FORMAT_VERSION,
        "teacher_path": teacher_path,
        "teacher_revision": hf_revision_of(teacher_path),
        "dataset": dataset_name,
        "num_samples": num_samples,
        "max_seq_len": max_seq_len,
        "seed": seed,
        "top_k": top_k,
        "score_window": [int(score_window[0]), int(score_window[1])],
        "vocab_size": measured_vocab,
        "config_vocab_size": config_vocab,
        # Measured, not asserted. `score` reports this in preference to the
        # --teacher-precision label. None on a model whose parameter tree could
        # not be walked, in which case the label stands.
        "teacher_precision": teacher_precision,
        "tokenizer_hash": teacher_tokenizer_hash,
        "corpus_tokens_hash": corpus_tokens_hash,
        "num_batches": num_batches,
        "batch_size": batch_size,
        "logit_dtype": "bfloat16",
        "created_at": now,
        "last_used": now,
    }
    write_json_atomic(cache_dir / "manifest.json", manifest)
    # Record the on-disk footprint now that all shards + manifest exist.
    manifest["size_bytes"] = _dir_size_bytes(cache_dir)
    write_json_atomic(cache_dir / "manifest.json", manifest)

    info("freeing teacher")
    del teacher_model
    if hasattr(mx, "clear_cache"):
        mx.clear_cache()
    return manifest


# ---------- teacher cache lifecycle ----------

def ensure_cache_entry(
    *,
    teacher_path: str,
    tokens,                # mx.array (N, L) int32, the calibration stream
    teacher_tok_hash: str,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    top_k: int,
    batch_size: int = 1,
    cache_root: Path = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
    score_window: tuple[int, int] | None = None,
    cache_max_gb: float = 0.0,
) -> tuple[Path, dict, str, float]:
    """The single cache lifecycle: validity, tokenizer gate, HIT/MISS/rebuild,
    budget, locking.

    Both the CLI (which tokenizes with the student's tokenizer) and
    :func:`ensure_teacher_topk_cache` (which tokenizes with the teacher's)
    drive this, so a manifest check added here holds for both. Returns
    ``(cache_dir, manifest, status, build_seconds)`` with status ``"HIT"``,
    ``"MISS"``, or ``"REBUILD"``; the manifest's ``score_window`` reflects this
    run's window, and ``build_seconds`` times the teacher pass alone (0.0 on a
    HIT), excluding any wait for another process's lock.

    ``cache_max_gb > 0`` makes room *before* a build: least-recently-used
    entries are evicted down to ``cache_max_gb`` minus the new entry's
    estimated size, so the root peaks near the budget instead of budget + one
    entry. Free space is checked before every build, budget or not, and a
    shortfall is warned about up front rather than discovered hours in.

    Raises :class:`~mlx_kld.CacheMismatchError` when a valid entry was built
    against a different teacher tokenizer.
    """
    import time
    key = cache_key(teacher_path, dataset_name, num_samples, max_seq_len, seed, top_k)
    cache_dir: Path = cache_root / key
    if score_window is None:
        score_window = (0, max_seq_len)

    def _expected_batches() -> int:
        # An existing entry replays at its own batch size, so a differing
        # batch_size must not read as a MISS and trigger a teacher rebuild.
        replay_bs = manifest_batch_size(cache_dir) or batch_size
        return (int(tokens.shape[0]) + replay_bs - 1) // replay_bs

    def _hit() -> tuple[Path, dict, str, float]:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        # The cache is keyed to the teacher; any student that cleared the
        # encoding-parity gate scores on the same cached token ids. Only a
        # changed teacher tokenizer invalidates it.
        if manifest.get("tokenizer_hash") not in (None, teacher_tok_hash):
            raise CacheMismatchError(
                "cache tokenizer_hash mismatch: the entry was built against a "
                f"different teacher tokenizer ({manifest['tokenizer_hash']} != "
                f"{teacher_tok_hash}). Rebuild the entry (score "
                "--rebuild-cache, or rebuild=True from the API), or point "
                "--cache-dir / cache_root elsewhere."
            )
        info(f"teacher cache HIT: {cache_dir}")
        touch_entry(cache_dir)
        manifest = dict(manifest)
        manifest["score_window"] = [int(score_window[0]), int(score_window[1])]
        return cache_dir, manifest, "HIT", 0.0

    with entry_lock(cache_root, key, exclusive=False):
        valid, reason = cache_is_valid(cache_dir, _expected_batches())
        if valid and not rebuild:
            return _hit()

    # A build is needed. Trade the shared lock for the exclusive one (never
    # upgrade in place: two holders upgrading is a deadlock), then re-check.
    # Another process may have finished the same build while we waited.
    with entry_lock(cache_root, key, exclusive=True):
        valid, reason = cache_is_valid(cache_dir, _expected_batches())
        if valid and not rebuild:
            return _hit()

        est_bytes = estimate_entry_bytes(num_samples, max_seq_len, top_k)
        est = human_bytes(est_bytes)
        status = "REBUILD" if (valid and rebuild) else "MISS"
        if status == "REBUILD":
            info(f"rebuild requested: rewriting {cache_dir} (~{est})")
        else:
            info(f"teacher cache MISS ({reason}). The teacher pass will write "
                 f"a ~{est} cache entry under {cache_root} "
                 "(a smaller top_k shrinks it proportionally)")

        # Drop the doomed entry before making room, so its bytes are not
        # double-counted against the budget and the peak stays one entry lower.
        if cache_dir.exists():
            for f in cache_dir.glob("batch-*.safetensors"):
                f.unlink()
            (cache_dir / "manifest.json").unlink(missing_ok=True)

        budget = cache_max_gb * 1e9
        if budget > 0:
            if est_bytes > budget:
                warn(f"this single entry (~{est}) exceeds the cache budget of "
                     f"{cache_max_gb:g} GB: every other entry will be evicted "
                     "and the root will still be over budget. Raise "
                     "--cache-max-gb or lower --top-k.")
            # Evict to (budget - estimate), or to empty when one entry cannot
            # fit the budget at all, since that case needs the room most. A
            # positive floor because enforce_budget reads <= 0 as "disabled".
            target_gb = max((budget - est_bytes) / 1e9, 1e-9)
            pre = enforce_budget(cache_root, target_gb, protect_keys=(key,))
            if pre:
                freed = sum(sz for _, sz in pre)
                info(f"cache: pre-evicted {len(pre)} LRU entr"
                     f"{'y' if len(pre) == 1 else 'ies'} "
                     f"({human_bytes(freed)}) to make room")
        try:
            free = shutil.disk_usage(cache_root).free
        except OSError:
            free = None
        if free is not None and free < est_bytes:
            warn(f"free space under {cache_root} ({human_bytes(free)}) is below "
                 f"the estimated entry size (~{est}). The teacher pass is "
                 "likely to fail on a full disk")

        cache_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        try:
            manifest = teacher_pass(
                teacher_path, tokens, batch_size, top_k, cache_dir,
                dataset_name, num_samples, max_seq_len, seed,
                teacher_tok_hash, score_window=score_window,
            )
        except BaseException:
            # A manifest-less entry is unusable (cache_is_valid rejects it) AND
            # invisible to entry_info, so leaving it would strand up to a full
            # entry of disk that no eviction path can reclaim. Ctrl-C included.
            shutil.rmtree(cache_dir, ignore_errors=True)
            raise
        return cache_dir, manifest, status, time.time() - t0


# ---------- public entry for external tools ----------

def ensure_teacher_topk_cache(
    *,
    teacher_path: str,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    batch_size: int = 1,
    # Helper-level default is intentionally pinned to 128, NOT DEFAULT_TOP_K:
    # external callers doing within-loop delta ranking benefit from cache reuse
    # and a small K. The user-facing CLI passes top_k explicitly.
    top_k: int = 128,
    cache_root: Path = DEFAULT_CACHE_DIR,
    rebuild: bool = False,
    score_window: tuple[int, int] | None = None,
):
    """Ensure a teacher top-K cache exists for the given (teacher x calibration
    spec). Returns ``(cache_dir, manifest, tokenizer)``.

    The tokenizer is returned alongside because external callers typically need
    it for parity checks and to confirm the in-memory model's vocab matches what
    the cache was built against.

    .. important::
       ``cache_key`` does **not** cover the tokenizer, so one entry per
       ``(teacher x calibration spec x top_k)`` is shared by every student.
       Whichever caller builds the entry first fixes the token stream all later
       students replay. This helper tokenizes with the *teacher's* tokenizer;
       ``mlx-kld score`` tokenizes with the *student's*, which for a GGUF in
       synthesized-tokenizer mode can be a different stream. The manifest's
       ``corpus_tokens_hash`` is the witness: compare it against a stream you
       tokenized yourself before trusting that the cache matches your intent,
       and pass ``rebuild=True`` when it does not.

    .. note::
       To protect a long replay from another process's cache eviction, hold
       ``entry_lock(cache_root, cache_dir.name, exclusive=False)`` (importable
       from :mod:`mlx_kld`) across your
       :func:`~mlx_kld.score_loaded_student` call, and re-read
       ``manifest.json`` once you hold it: another process may have rebuilt
       the entry from a different token stream in the meantime, which leaves
       the manifest returned here describing shards that are no longer on
       disk. The build itself is already serialized internally.
    """
    from mlx_lm.utils import load_tokenizer

    info(f"loading teacher tokenizer: {teacher_path}")
    teacher_tokenizer = load_tokenizer(teacher_path)
    teacher_tok_hash = tokenizer_hash(teacher_tokenizer)

    tokens = load_calibration_tokens(
        teacher_tokenizer, dataset_name, num_samples, max_seq_len, seed,
    )
    info(f"calibration: {tokens.shape[0]} sequences x {tokens.shape[1]} tokens")

    cache_dir, manifest, _status, _build_seconds = ensure_cache_entry(
        teacher_path=teacher_path,
        tokens=tokens,
        teacher_tok_hash=teacher_tok_hash,
        dataset_name=dataset_name,
        num_samples=num_samples,
        max_seq_len=max_seq_len,
        seed=seed,
        top_k=top_k,
        batch_size=batch_size,
        cache_root=cache_root,
        rebuild=rebuild,
        score_window=score_window,
    )
    return cache_dir, manifest, teacher_tokenizer
