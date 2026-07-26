"""Tokenizer parity checks + calibration-corpus tokenization.

KLD is well-defined iff the tokenization fed to *both* models is the one the
teacher expects, meaning text -> token-ids agrees between teacher and student.
vocab_size / special-id metadata can differ without changing the ids of the
scored corpus, so those are reported but never gate the run; the direct
encoding-parity probe is what gates it.
"""

from __future__ import annotations

import hashlib

from ._log import info
from .errors import CalibrationCorpusError, TokenizerMismatchError


def _hf_inner(tokenizer):
    """The HF tokenizer behind an mlx_lm TokenizerWrapper, else the tokenizer
    itself. A raw PreTrainedTokenizerFast is already HF-like, and its
    ``_tokenizer`` attr is the backend Rust ``tokenizers.Tokenizer`` (no
    convert_ids_to_tokens / vocab_size / __len__), so only unwrap ``_tokenizer``
    when the inner object is still HF-like."""
    cand = getattr(tokenizer, "_tokenizer", None)
    if cand is not None and hasattr(cand, "convert_ids_to_tokens"):
        return cand
    return tokenizer


def tokenizer_hash(tokenizer) -> str:
    """Hash a loaded tokenizer's *behavior*: vocab map + special tokens.

    Hashing on-disk bytes is too strict, because mlx-lm's save() reformats
    tokenizer.json, producing different bytes for a functionally identical
    tokenizer. We hash the (id -> token) pairs that drive encoding/decoding, so
    the check passes iff KLD is actually defined.
    """
    h = hashlib.sha256()
    inner = _hf_inner(tokenizer)
    vocab_size = getattr(inner, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(inner)
    h.update(f"vocab_size={vocab_size}\n".encode())
    # One convert_ids_to_tokens call per id. convert_ids_to_tokens also accepts
    # a list, but batching it measured 0.11s against 0.13s on a 248k vocab,
    # noise next to a teacher pass, and not worth the extra code.
    try:
        for i in range(len(inner)):
            tok = inner.convert_ids_to_tokens(i)
            h.update(f"{i}\t{tok}\n".encode("utf-8", errors="replace"))
    except Exception as e:
        h.update(f"<walk-failed:{e}>".encode())
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        v = getattr(inner, name, None)
        h.update(f"{name}={v}\n".encode())
    return h.hexdigest()


def tokenizer_metadata_diffs(teacher, student) -> list[str]:
    """Human-readable list of metadata fields (vocab size + special-token ids)
    that differ between two tokenizers; empty when they match. These differences
    do not by themselves make KLD undefined; see ``encoding_parity``."""
    ti, si = _hf_inner(teacher), _hf_inner(student)
    tv = getattr(ti, "vocab_size", None) or len(ti)
    sv = getattr(si, "vocab_size", None) or len(si)
    out = []
    if tv != sv:
        out.append(f"vocab_size : teacher={tv} student={sv}")
    for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        t, s = getattr(ti, name, None), getattr(si, name, None)
        if t != s:
            out.append(f"{name} : teacher={t} student={s}")
    return out


# Deliberately diverse: ascii prose, code, unicode/CJK, the chat/tool special
# tokens, digits, and tricky whitespace. Two tokenizers that encode all of these
# to identical ids agree on the tokenization that drives KLD.
_ENCODING_PROBES = [
    "The quick brown fox jumps over the lazy dog.",
    "def f(x):\n    return x * 2  # double it\n",
    "Café résumé — числа 12345, 東京は日本の首都です。",  # ascii-exempt
    "<think>step by step</think> <tool_call>{\"a\": 1}</tool_call>",
    "Multiple   spaces\tand\ttabs, and a trailing space ",
]


def encoding_parity(teacher, student):
    """Probe whether two tokenizers encode text to identical ids. Returns
    ``(True, None)`` on full agreement, else ``(False, (probe, teacher_ids,
    student_ids))`` for the first divergent probe. This probe, not vocab_size
    or special-id metadata, is what a well-defined KLD actually requires."""
    ti, si = _hf_inner(teacher), _hf_inner(student)
    for s in _ENCODING_PROBES:
        try:
            a = list(ti.encode(s, add_special_tokens=False))
            b = list(si.encode(s, add_special_tokens=False))
        except Exception as e:  # defensive: never let the probe crash the run
            return False, (f"<encode failed: {e}>", [], [])
        if a != b:
            return False, (s, a, b)
    return True, None


def cached_stream_status(
    cached_hash: str | None, student_hash: str, tok_mode: str
) -> tuple[bool | None, str | None]:
    """Classify a cache HIT's token stream against the student's own.

    One cache entry per ``(teacher x calibration spec x top_k)`` is shared by
    every student, so a HIT replays whatever ids the first student tokenized.
    The encoding-parity probe is five short strings and cannot certify a
    262k-token corpus, so compare the streams directly.

    Returns ``(stream_is_students, warning)``. ``stream_is_students`` is ``None``
    for a pre-hash cache (no witness recorded), ``True`` when the cached stream
    is the one this student's tokenizer produces, and ``False`` otherwise, in
    which case ``warning`` carries the mode-appropriate explanation.
    """
    if cached_hash is None:
        return None, None
    if cached_hash == student_hash:
        return True, None
    detail = (
        f"the cached token stream ({cached_hash}) differs from the one this "
        f"student's tokenizer produces ({student_hash}). The score replays the "
        "cached stream"
    )
    if tok_mode == "synthesized":
        return False, (
            f"{detail}. This defeats the point of the default GGUF tokenizer "
            "mode, which is to test the GGUF as shipped: the weights are this "
            "student's but the tokenization is the cache builder's. Pass "
            "--rebuild-cache to score on this GGUF's own tokenization, or "
            "--hf-source to borrow the teacher's deliberately."
        )
    return False, (
        f"{detail}. Use --rebuild-cache to score on this student's own "
        "tokenization instead."
    )


def load_gguf_tokenizer(gguf_path) -> tuple[object, str]:
    """Synthesize the student tokenizer from GGUF metadata via gmlx.

    Returns ``(tokenizer, arch)`` where the tokenizer is an HF
    ``PreTrainedTokenizerFast`` built from the GGUF's embedded vocab/merges. That
    is the tokenizer the loaded GGUF model uses, so scoring with it tests the
    GGUF exactly as it ships. Encoding parity against the teacher still gates
    the run (see ``encoding_parity``).
    """
    from ._deps import require_gguf

    require_gguf()
    from gguf import GGUFReader

    try:
        # Public as of gmlx > 0.1.0; fall back to the module paths on 0.1.0.
        from gmlx import detect_arch, load_tokenizer_from_gguf
    except ImportError:
        from gmlx.remap import detect_arch
        from gmlx.tokenizer import load_tokenizer_from_gguf

    reader = GGUFReader(str(gguf_path), "r")
    arch = detect_arch(reader)
    return load_tokenizer_from_gguf(reader, arch), arch


def _split_dataset_spec(spec: str) -> tuple[str, str | None]:
    """Split ``path:name`` into ``(path, subset_name)`` for HF subset configs.

    Lets us pass corpora that require a ``name=`` kwarg to
    ``datasets.load_dataset`` (e.g. ``HuggingFaceFW/fineweb-edu:sample-10BT``).
    Plain ``path`` returns ``(spec, None)``.
    """
    path, sep, name = spec.partition(":")
    return (path, name) if sep and name else (spec, None)


def _render_chat_messages(tokenizer, dataset_name: str, ds):
    """Apply the tokenizer's chat template to each row's ``messages`` list.

    For chat-format corpora the per-row text is a list of role/content dicts;
    rendering through the chat template produces the deployment-shaped string
    the model would actually see at inference, so KLD calibrated on it lands
    in-distribution for instruct deploys. Bails clearly if the tokenizer has no
    ``chat_template`` set.

    Yields lazily, so ingest can stop as soon as it has enough tokens rather than
    rendering the whole corpus first.
    """
    # _hf_inner, not a bare _tokenizer unwrap: on a raw PreTrainedTokenizerFast
    # (the GGUF-synthesized case) `_tokenizer` is the backend Rust tokenizer,
    # which has no chat_template / apply_chat_template.
    inner = _hf_inner(tokenizer)
    if not getattr(inner, "chat_template", None):
        raise TokenizerMismatchError(
            f"Dataset {dataset_name!r} is chat-format (column 'messages') but "
            "the tokenizer used for calibration has no chat_template. Which "
            "tokenizer that is depends on the entry point: `mlx-kld score` "
            "uses the student's, `mlx-kld floor-sweep` the swept model's, and "
            "ensure_teacher_topk_cache the teacher's. Use a model that ships "
            "a chat template, or a text-format corpus."
        )
    for r in ds:
        msgs = r.get("messages") or []
        if not msgs:
            continue
        try:
            rendered = inner.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
        except Exception as e:
            raise CalibrationCorpusError(
                f"apply_chat_template failed on a row from {dataset_name!r}: {e}"
            ) from e
        if rendered.strip():
            yield rendered


def _iter_corpus_texts(tokenizer, dataset_name: str, ds, columns):
    """Lazily yield the non-empty text of each row, in dataset order.

    Lazy on purpose. Materializing the column first costs one Python string per
    row, 1.165M of them for wikitext-103, and enough to exhaust memory on a
    web-scale corpus, when ingest only ever reads the first few thousand.
    """
    if columns and "text" in columns:
        for r in ds:
            if r["text"].strip():
                yield r["text"]
    elif columns and "messages" in columns:
        yield from _render_chat_messages(tokenizer, dataset_name, ds)
    else:
        raise CalibrationCorpusError(
            f"Dataset {dataset_name!r} has neither a 'text' nor 'messages' "
            f"column (found {list(columns or [])}). --dataset must point at a "
            "text- or chat-format corpus."
        )


def load_calibration_tokens(
    tokenizer,
    dataset_name: str,
    num_samples: int,
    max_seq_len: int,
    seed: int,
    streaming: bool = False,
):
    """Tokenize the calibration corpus, slice into fixed-length chunks, and
    deterministically pick ``num_samples`` of them.

    Accepts ``path:subset`` syntax for HF subset configs. Auto-detects
    chat-format corpora (with a ``messages`` column) and renders them through
    the tokenizer's chat template before tokenizing.

    ``streaming=True`` pulls rows over the network instead of downloading the
    split first, which is what makes web-scale corpora usable at all. It reads
    the same rows in the same order, so the token stream is unchanged.

    .. note::
       Ingest stops as soon as it holds twice the tokens it needs, so the
       calibration set is drawn from the **head** of the corpus (~525k tokens at
       the default protocol) and ``seed`` permutes within that window rather than
       sampling the corpus as a whole. See the README's Limitations section.

    Returns an ``mx.array`` of shape ``(num_samples, max_seq_len)`` int32.
    """
    import mlx.core as mx
    import numpy as np
    from datasets import load_dataset

    if dataset_name == "wikitext-2-raw-v1":
        path, subset = "wikitext", "wikitext-2-raw-v1"
    else:
        path, subset = _split_dataset_spec(dataset_name)
    ds = load_dataset(path, name=subset, split="train", streaming=streaming)
    columns = ds.column_names
    if not columns:
        # Some streaming datasets do not declare a schema; peek one row.
        first = next(iter(ds), None)
        columns = list(first.keys()) if first else []

    info(f"tokenizing {dataset_name}"
         f"{' (streaming)' if streaming else ''}...")
    # Concatenate into one stream of ids, then chunk at max_seq_len, which is
    # the standard wikitext perplexity recipe (no padding, no BOS).
    all_ids: list[int] = []
    chunk_target = num_samples * max_seq_len + max_seq_len  # +slack
    rows_read = 0
    for t in _iter_corpus_texts(tokenizer, dataset_name, ds, columns):
        rows_read += 1
        all_ids.extend(tokenizer.encode(t, add_special_tokens=False))
        if len(all_ids) >= chunk_target * 2:
            break  # plenty for shuffling

    n_chunks = len(all_ids) // max_seq_len
    info(f"read {rows_read:,} rows -> {len(all_ids):,} tokens "
         f"({n_chunks:,} chunks of {max_seq_len})")
    if n_chunks < num_samples:
        raise CalibrationCorpusError(
            f"Calibration corpus too small: got {n_chunks} chunks of "
            f"length {max_seq_len}, need {num_samples}. Lower --num-samples "
            "or --max-seq-len."
        )

    ids_arr = np.asarray(all_ids[: n_chunks * max_seq_len], dtype=np.int32)
    chunks = ids_arr.reshape(n_chunks, max_seq_len)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_chunks)[:num_samples]
    chunks = chunks[perm]

    return mx.array(chunks)
