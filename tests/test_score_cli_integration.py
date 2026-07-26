"""End-to-end ``mlx-kld score`` through the real CLI dispatcher, with every
model/dataset loader faked (in-process, no subprocess, no network).

The argument-validation tests in test_cli.py exit before any model load; this
file covers the other side: a full run that builds a cache entry, replays it,
writes a record that validates, and prints the Markdown report to stdout.
"""

from __future__ import annotations

import json

import mlx.core as mx
import pytest

from mlx_kld import cache as C
from mlx_kld import measure, scoring, tokenizer
from mlx_kld.cli import main
from mlx_kld.report import validate_locked_schema

_VOCAB = 64


class _FakeModel:
    def eval(self):
        pass

    def __call__(self, batch, cache=None):
        b, seq = batch.shape
        base = mx.arange(_VOCAB, dtype=mx.float32)
        pos = mx.arange(seq, dtype=mx.float32)[:, None]
        return mx.broadcast_to((base + pos) % _VOCAB, (b, seq, _VOCAB))


class _FakeTokenizer:
    vocab_size = _VOCAB
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = None

    def __len__(self):
        return _VOCAB

    def convert_ids_to_tokens(self, i):
        return f"<tok{i}>"

    def encode(self, s, add_special_tokens=False):
        return [ord(c) % _VOCAB for c in s]


@pytest.fixture
def faked_run(monkeypatch, tmp_path):
    """Fake every loader the score path touches; return the CLI argv tail."""
    tok = _FakeTokenizer()
    model = _FakeModel()
    tokens = mx.array(
        [[(i * 8 + j) % _VOCAB for j in range(8)] for i in range(4)],
        dtype=mx.int32,
    )
    monkeypatch.setattr("mlx_lm.utils.load_tokenizer", lambda p: tok)
    monkeypatch.setattr(tokenizer, "load_calibration_tokens",
                        lambda *a, **k: tokens)
    monkeypatch.setattr(C, "_load_model",
                        lambda p, lazy=True: (model, {"vocab_size": _VOCAB}))
    monkeypatch.setattr(C, "_make_fresh_cache", lambda m: None)
    monkeypatch.setattr(C, "hf_revision_of", lambda p: None)
    monkeypatch.setattr(scoring, "_load_model",
                        lambda p, lazy=True: (model, {"vocab_size": _VOCAB}))
    monkeypatch.setattr(scoring, "_make_fresh_cache", lambda m: None)
    monkeypatch.setattr(measure, "measure_student", lambda p: {
        "format": "mlx-affine", "size_bytes": 1000, "effective_bpw": 4.5,
        "n_params": 2000, "quantization": {"kind": "affine", "bits": 4,
                                           "group_size": 64, "mode": "affine"},
        "scored_n_params": 1500, "scored_bytes": 800,
        "scored_bpw": 800 * 8 / 1500,
    })
    monkeypatch.setattr(measure, "load_provenance", lambda p: None)

    student = tmp_path / "student"
    student.mkdir()
    return [
        "score", "org/teacher", str(student),
        "--num-samples", "4", "--max-seq-len", "8", "--top-k", "8",
        "--batch-size", "2",
        "--cache-dir", str(tmp_path / "cache"),
        "--out-dir", str(tmp_path / "results"),
    ]


def _record_paths(tmp_path):
    return list((tmp_path / "results").rglob("*.json"))


def test_score_end_to_end_writes_a_valid_record(faked_run, tmp_path, capsys):
    assert main(faked_run) == 0
    out = capsys.readouterr().out
    assert "Mean KLD" in out           # the Markdown report went to stdout
    assert "score_window" in out       # the window is part of the printed spec
    records = _record_paths(tmp_path)
    assert len(records) == 1
    payload = json.loads(records[0].read_text())
    validate_locked_schema(payload)
    assert payload["cache"]["status"] == "MISS"
    assert payload["cache"]["top_k"] == 8
    assert payload["calibration"]["top_k"] == 8
    assert payload["tokenizer"]["stream_is_students"] is True
    assert payload["tokens_dropped_nonfinite"] == 0
    # The scored-bpw fields ride along in the student block (non-locked).
    assert payload["student"]["scored_bpw"] == pytest.approx(800 * 8 / 1500)
    # student == teacher: the mean is exactly the reconstruction floor.
    assert payload["kld"]["mean"] == pytest.approx(
        payload["kld"]["floor_mean"], rel=1e-5, abs=1e-9)


def test_second_run_hits_the_cache_and_stays_idempotent(faked_run, tmp_path, capsys):
    assert main(faked_run) == 0
    assert main(faked_run) == 0
    records = _record_paths(tmp_path)
    assert len(records) == 1          # identical rerun overwrote, not forked
    payload = json.loads(records[0].read_text())
    assert payload["cache"]["status"] == "HIT"
    err = capsys.readouterr().err
    assert "teacher cache HIT" in err


def test_requested_top_k_above_vocab_is_recorded_as_clamped(faked_run, tmp_path):
    """The record must carry the K actually cached, not the one requested."""
    argv = faked_run
    argv[argv.index("--top-k") + 1] = "9999"
    assert main(argv) == 0
    [record] = _record_paths(tmp_path)
    payload = json.loads(record.read_text())
    # Clamped early against the tokenizer vocab (64).
    assert payload["calibration"]["top_k"] == _VOCAB - 1
    assert payload["cache"]["top_k"] == _VOCAB - 1


def test_the_cache_key_spells_the_top_k_that_was_actually_cached(faked_run, tmp_path):
    """The entry directory is named by a hash over the requested K, while the
    manifest records the K the teacher pass actually used. If the early clamp
    ever lands above the real logits width, teacher_pass re-clamps and the two
    diverge, leaving an entry filed under a K it does not contain.

    Reconstructing the key from the recorded K is what proves they agree.
    """
    argv = faked_run
    argv[argv.index("--top-k") + 1] = "9999"
    assert main(argv) == 0
    [record] = _record_paths(tmp_path)
    payload = json.loads(record.read_text())

    manifest = json.loads((tmp_path / "cache" / C.cache_key(
        "org/teacher",
        payload["calibration"]["corpus"],
        payload["calibration"]["num_samples"],
        payload["calibration"]["max_seq_len"],
        payload["calibration"]["seed"],
        payload["cache"]["top_k"],
    ) / "manifest.json").read_text())
    assert manifest["top_k"] == payload["cache"]["top_k"] == _VOCAB - 1


def test_nonfinite_student_positions_are_counted_in_the_record(faked_run, tmp_path):
    """Dropping a position *improves* the mean, so the count has to reach the
    record rather than only being warned about."""
    real = scoring.kld_from_topk

    def poisoned(top_log_p, top_idx, student_logits, vocab):
        kld, t1, t5 = real(top_log_p, top_idx, student_logits, vocab)
        # Two positions of the first sequence go non-finite. They must sit
        # *inside* the scored window [4, 8). Positions outside it are dropped
        # by the mask before the aggregator ever sees them, and rightly are not
        # counted as scoring casualties.
        pos = mx.arange(kld.shape[1])[None, :]
        row = mx.arange(kld.shape[0])[:, None]
        bad = (row == 0) & (pos >= 4) & (pos < 6)
        return mx.where(bad, mx.array(float("inf")), kld), t1, t5

    scoring.kld_from_topk = poisoned
    try:
        assert main(faked_run) == 0
    finally:
        scoring.kld_from_topk = real
    [record] = _record_paths(tmp_path)
    payload = json.loads(record.read_text())
    # 4 sequences at batch_size 2 is two shards, and row 0 of each is poisoned:
    # 2 shards x 2 positions.
    assert payload["tokens_dropped_nonfinite"] == 4
    # The surviving positions still scored, so the run is not simply empty.
    assert payload["tokens_scored"] == 4 * 4 - 4
    validate_locked_schema(payload)   # still a valid record with the extra key


def test_record_describes_the_shards_actually_on_disk(faked_run, tmp_path, monkeypatch):
    """The lifecycle releases its lock before returning, so another process can
    evict *and rebuild* the key from a different student's stream in the gap.
    Same key, same shard count, different corpus: the count-only validity check
    passes, and the run would publish the in-memory manifest's provenance for
    shards it never scored. Simulate that by tampering with the on-disk
    manifest after the lifecycle returns."""
    real = C.ensure_cache_entry

    def rebuilt_under_us(**kwargs):
        cache_dir, manifest, status, secs = real(**kwargs)
        on_disk = json.loads((cache_dir / "manifest.json").read_text())
        on_disk["corpus_tokens_hash"] = "deadbeefdeadbeef"   # another stream
        (cache_dir / "manifest.json").write_text(json.dumps(on_disk))
        return cache_dir, dict(manifest), status, secs   # stale copy, as before

    # cmd() imports this inside the function body, so patch it at the source.
    monkeypatch.setattr(C, "ensure_cache_entry", rebuilt_under_us)
    assert main(faked_run) == 0
    [record] = _record_paths(tmp_path)
    payload = json.loads(record.read_text())
    # The record must witness the stream that was scored, not the one we built.
    assert payload["calibration"]["corpus_tokens_hash"] == "deadbeefdeadbeef"
    # ...and must not claim the scored tokenization was this student's.
    assert payload["tokenizer"]["stream_is_students"] is False


def test_evicted_entry_between_build_and_replay_fails_loudly(
        faked_run, tmp_path, monkeypatch, capsys):
    """An entry evicted in the build->replay gap must abort the run, not score
    against whatever is left behind."""
    import shutil as _shutil

    real = C.ensure_cache_entry

    def evicted_under_us(**kwargs):
        cache_dir, manifest, status, secs = real(**kwargs)
        _shutil.rmtree(cache_dir)
        return cache_dir, manifest, status, secs

    monkeypatch.setattr(C, "ensure_cache_entry", evicted_under_us)
    assert main(faked_run) == 1          # MlxKldError -> dispatcher exit 1
    err = capsys.readouterr().err
    assert "became unreadable" in err or "became invalid" in err
    assert _record_paths(tmp_path) == []   # no record written for a dead run


def test_a_record_that_fails_its_own_schema_reports_as_an_internal_error(
    faked_run, monkeypatch
):
    """RecordSchemaError subclasses MlxKldError so an embedding caller can catch
    the family, but a record failing the schema this tool just wrote is a bug in
    this tool, not a thing the user typed. It must not print as `error: ...`
    with exit 1, indistinguishable from a bad flag.
    """
    from mlx_kld import report
    from mlx_kld.errors import RecordSchemaError

    def boom(payload):
        raise RecordSchemaError("locked schema: kld.mean missing")

    monkeypatch.setattr(report, "validate_locked_schema", boom)
    assert main(faked_run) == 70          # sysexits.h EX_SOFTWARE
