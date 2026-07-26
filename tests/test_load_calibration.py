"""``load_calibration_tokens``: the ingest that decides what actually gets scored.

Every number the tool reports is conditioned on this function's output, and it
is the one place where a silent change (different rows, different order,
different slack) would move every score without moving any threshold. These
tests pin the stream, not the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlx_kld.errors import CalibrationCorpusError
from mlx_kld.tokenizer import load_calibration_tokens


class _FakeTokenizer:
    """One id per character, so a token stream is readable in a failure diff."""

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False, "calibration must not inject BOS/EOS"
        return [ord(c) % 4096 for c in text]


class _FakeDataset:
    """Mimics an HF Dataset/IterableDataset: ``iter()`` restarts.

    ``column_names`` is None for a streaming dataset that declares no schema,
    which is the case the peek in ``load_calibration_tokens`` exists for.
    """

    def __init__(self, rows, declare_columns=True):
        self._rows = rows
        self.column_names = list(rows[0].keys()) if (rows and declare_columns) else None
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self._rows)


def _rows(n: int, width: int = 200) -> list[dict]:
    # Distinct content per row so a dropped or reordered row is visible.
    return [{"text": f"row{i:05d} " + "abcdefghij" * (width // 10)} for i in range(n)]


@pytest.fixture
def patched(monkeypatch):
    """Install a fake ``datasets.load_dataset`` and record its kwargs."""
    calls = {}

    def fake_load_dataset(path, name=None, split=None, streaming=False):
        calls.update(path=path, name=name, split=split, streaming=streaming)
        return calls["dataset"]

    import datasets

    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    return calls


def _load(patched, ds, **kw):
    patched["dataset"] = ds
    kw.setdefault("num_samples", 4)
    kw.setdefault("max_seq_len", 16)
    kw.setdefault("seed", 123)
    return np.asarray(
        load_calibration_tokens(_FakeTokenizer(), kw.pop("dataset", "corpus/x"), **kw)
    )


# ---------- shape and determinism ----------

def test_returns_num_samples_by_max_seq_len(patched):
    out = _load(patched, _FakeDataset(_rows(60)))
    assert out.shape == (4, 16)
    assert out.dtype == np.int32


def test_same_seed_reproduces_the_same_calibration_set(patched):
    a = _load(patched, _FakeDataset(_rows(60)))
    b = _load(patched, _FakeDataset(_rows(60)))
    assert np.array_equal(a, b)


def test_seed_changes_which_chunks_are_drawn(patched):
    a = _load(patched, _FakeDataset(_rows(60)), seed=1)
    b = _load(patched, _FakeDataset(_rows(60)), seed=2)
    assert not np.array_equal(a, b)


# ---------- streaming parity ----------

def test_streaming_reads_the_same_stream_as_a_local_split(patched):
    """The README promises `--stream-dataset` changes only transport. If this
    ever diverges, every cache key silently stops meaning what it says."""
    local = _load(patched, _FakeDataset(_rows(60)), streaming=False)
    assert patched["streaming"] is False
    streamed = _load(patched, _FakeDataset(_rows(60)), streaming=True)
    assert patched["streaming"] is True
    assert np.array_equal(local, streamed)


def test_streaming_flag_is_forwarded_verbatim(patched):
    _load(patched, _FakeDataset(_rows(60)), streaming=True)
    assert patched["streaming"] is True
    assert patched["split"] == "train"


# ---------- dataset spec ----------

def test_path_colon_subset_is_split_into_load_dataset_kwargs(patched):
    _load(patched, _FakeDataset(_rows(60)),
          dataset="HuggingFaceFW/fineweb-edu:sample-10BT")
    assert patched["path"] == "HuggingFaceFW/fineweb-edu"
    assert patched["name"] == "sample-10BT"


def test_plain_path_passes_no_subset(patched):
    _load(patched, _FakeDataset(_rows(60)), dataset="Salesforce/wikitext")
    assert patched["path"] == "Salesforce/wikitext"
    assert patched["name"] is None


def test_legacy_wikitext_2_shorthand_still_resolves(patched):
    _load(patched, _FakeDataset(_rows(60)), dataset="wikitext-2-raw-v1")
    assert patched["path"] == "wikitext"
    assert patched["name"] == "wikitext-2-raw-v1"


# ---------- schema peek ----------

def test_undeclared_schema_peeks_a_row_without_losing_it(patched):
    """A streaming dataset may not declare column_names. The peek must not eat
    the first row, which would silently shift the whole calibration set."""
    declared = _load(patched, _FakeDataset(_rows(60), declare_columns=True))
    undeclared = _load(patched, _FakeDataset(_rows(60), declare_columns=False))
    assert np.array_equal(declared, undeclared)


# ---------- head-window ingest ----------

def test_ingest_stops_before_reading_the_whole_corpus(patched):
    """Documented behaviour (see the function's note and README Limitations):
    the calibration set comes from the head of the corpus."""
    consumed = []

    class _Counting(_FakeDataset):
        def __iter__(self):
            for r in self._rows:
                consumed.append(r)
                yield r

    patched["dataset"] = _Counting(_rows(20_000))
    load_calibration_tokens(_FakeTokenizer(), "corpus/x", num_samples=4,
                            max_seq_len=16, seed=123)
    assert len(consumed) < 200, "ingest read far more of the corpus than it needs"


# ---------- failure modes ----------

def test_too_small_a_corpus_raises_with_an_actionable_message(patched):
    with pytest.raises(CalibrationCorpusError) as e:
        _load(patched, _FakeDataset(_rows(1, width=10)), num_samples=64,
              max_seq_len=512)
    msg = str(e.value)
    assert "too small" in msg
    assert "--num-samples" in msg and "--max-seq-len" in msg


def test_missing_text_column_names_the_columns_it_found(patched):
    ds = _FakeDataset([{"content": "hello"}])
    with pytest.raises(CalibrationCorpusError) as e:
        _load(patched, ds)
    assert "content" in str(e.value)
