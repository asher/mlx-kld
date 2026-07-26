"""Calibration ingest: lazy row iteration and column dispatch.

Ingest reads only as far into the corpus as it needs, so it must never
materialize the text column first, which is one Python string per row (1.165M
of them for wikitext-103, enough to exhaust memory on a web-scale corpus).
"""

from __future__ import annotations

import pytest

from mlx_kld.errors import CalibrationCorpusError, MlxKldError
from mlx_kld.tokenizer import _iter_corpus_texts, _split_dataset_spec


class _CountingRows:
    """Stands in for a Dataset; records how far iteration actually got."""

    def __init__(self, rows):
        self._rows = rows
        self.consumed = 0

    def __iter__(self):
        for r in self._rows:
            self.consumed += 1
            yield r


def test_ingest_stops_early_instead_of_reading_the_whole_corpus():
    ds = _CountingRows([{"text": f"row {i}"} for i in range(10_000)])
    taken = []
    for t in _iter_corpus_texts(None, "ds", ds, ["text"]):
        taken.append(t)
        if len(taken) == 5:
            break
    assert taken == [f"row {i}" for i in range(5)]
    assert ds.consumed == 5, "generator must not run ahead of its consumer"


def test_blank_rows_are_skipped_in_order():
    rows = [{"text": "a"}, {"text": "   "}, {"text": ""}, {"text": "b"}]
    assert list(_iter_corpus_texts(None, "ds", _CountingRows(rows), ["text"])) == ["a", "b"]


def test_missing_text_and_messages_columns_raise_with_the_column_list():
    ds = _CountingRows([{"content": "x"}])
    with pytest.raises(CalibrationCorpusError) as e:
        list(_iter_corpus_texts(None, "ds", ds, ["content"]))
    assert "content" in str(e.value)


def test_empty_column_list_raises_rather_than_yielding_nothing():
    with pytest.raises(CalibrationCorpusError):
        list(_iter_corpus_texts(None, "ds", _CountingRows([]), []))


def test_calibration_errors_are_catchable_as_the_base_class():
    """An embedding caller must be able to handle these without catching
    BaseException, which is what sys.exit forced."""
    with pytest.raises(MlxKldError):
        list(_iter_corpus_texts(None, "ds", _CountingRows([{"c": 1}]), ["c"]))


# ---------- path:subset spec ----------

@pytest.mark.parametrize("spec,expected", [
    ("Salesforce/wikitext:wikitext-103-raw-v1",
     ("Salesforce/wikitext", "wikitext-103-raw-v1")),
    ("HuggingFaceFW/fineweb-edu:sample-10BT",
     ("HuggingFaceFW/fineweb-edu", "sample-10BT")),
    ("plain/dataset", ("plain/dataset", None)),
    ("trailing:", ("trailing:", None)),
])
def test_split_dataset_spec(spec, expected):
    assert _split_dataset_spec(spec) == expected
