"""``score_loaded_student`` round-trip against a real (fake-model) cache.

The student pass is the numeric heart of the product: shard replay order, the
window-mask intersection, the Delta-p mask offset, and the per-sequence
bookkeeping all live here. Scoring the *same* fake model that built the cache
pins the strongest property available without a real checkpoint: the measured
KLD must equal the reconstruction floor, position for position.

That property is only load-bearing if the floor actually *varies* across
positions. A fake whose output ignores its input produces one distribution
everywhere, so every mask has the same mean and the equality holds no matter
which positions are compared, the assertion looks strong and tests nothing.
``_FakeModel`` below therefore keys its logits on the token ids as well as the
position, and ``test_the_fixture_is_not_degenerate`` guards that.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from mlx_kld import cache as C
from mlx_kld import kld_math, scoring
from mlx_kld.scoring import score_loaded_student


class _FakeModel:
    """Deterministic logits that depend on both the input tokens and position.

    A circular-distance kernel peaked at an id derived from the token, with a
    sharpness that cycles with position. That makes the distribution, and so
    its entropy, and so the top-K reconstruction floor, differ from row to
    row and position to position, which is what gives the masked comparisons
    below their teeth.
    """

    def __init__(self, vocab: int):
        self.vocab = vocab

    def eval(self):
        pass

    def __call__(self, batch, cache=None):
        b, seq = batch.shape
        ids = mx.arange(self.vocab, dtype=mx.float32)[None, None, :]
        tok = batch.astype(mx.float32)[..., None]
        pos = mx.arange(seq, dtype=mx.float32)[None, :, None]
        peak = (tok * 7.0 + pos * 3.0) % self.vocab
        # Circular distance from the peak, so the distribution wraps cleanly.
        d = mx.minimum((ids - peak) % self.vocab, (peak - ids) % self.vocab)
        # Monotonic in position, not cyclic: a sharpness that repeated with any
        # period dividing the sequence length would give two different windows
        # the same multiset of distributions, and so the same mean floor. The
        # token term matters too. With sharpness keyed on position alone,
        # every row at a given position has the same *sorted* top-K values and
        # differs only by rotation, so swapping rows is undetectable.
        sharpness = 0.30 + 0.10 * pos + 0.07 * (tok % 5.0)
        return mx.broadcast_to(-d * sharpness, (b, seq, self.vocab))


@pytest.fixture
def built_cache(monkeypatch, tmp_path):
    """A real cache entry built by teacher_pass over the fake model."""
    vocab, n, seq, top_k, batch_size = 64, 5, 8, 8, 2
    model = _FakeModel(vocab)
    monkeypatch.setattr(C, "_load_model", lambda p, lazy=True: (model, {"vocab_size": vocab}))
    monkeypatch.setattr(C, "_make_fresh_cache", lambda m: None)
    monkeypatch.setattr(C, "hf_revision_of", lambda p: None)
    monkeypatch.setattr(scoring, "_make_fresh_cache", lambda m: None)
    tokens = mx.array(
        [[(i * seq + j) % vocab for j in range(seq)] for i in range(n)],
        dtype=mx.int32,
    )
    manifest = C.teacher_pass(
        "org/teacher", tokens, batch_size, top_k, tmp_path,
        "corpus/x", n, seq, 123, "tokhash", score_window=(4, 8),
    )
    return model, tmp_path, manifest, dict(n=n, seq=seq)


def _rewindow(manifest, start, end):
    m = dict(manifest)
    m["score_window"] = [start, end]
    return m


def test_the_fixture_is_not_degenerate(built_cache):
    """Guards every other test in this file.

    If the fake model ever stops depending on its input, the per-position
    floors collapse to a single value and the floor/window/Delta-p assertions
    below become true for any mask at all. Fail loudly here instead.
    """
    _model, cache_dir, manifest, dims = built_cache
    floors = []
    for i in range((dims["n"] + 1) // 2):
        shard = mx.load(str(cache_dir / f"batch-{i:05d}.safetensors"))
        floors.append(np.asarray(shard["floor_kld"]).ravel())
    floors = np.concatenate(floors)
    assert np.unique(floors).size > 1, "floor is constant: the fake is degenerate"
    # Specifically, it must vary *along the sequence axis*, since that is the
    # axis every window mask cuts on.
    shard0 = mx.load(str(cache_dir / "batch-00000.safetensors"))
    per_pos = np.asarray(shard0["floor_kld"])[0]
    assert np.unique(per_pos).size > 1, "floor is constant along the sequence"


def test_perfect_student_scores_exactly_the_floor(built_cache):
    """student == teacher: the only KLD left is the top-K reconstruction floor,
    computed at build time with the identical arithmetic the replay uses."""
    model, cache_dir, manifest, _ = built_cache
    metrics = score_loaded_student(model, cache_dir, manifest)
    kld = metrics["kld"]
    assert kld["floor_mean"] is not None
    assert kld["mean"] == pytest.approx(kld["floor_mean"], rel=1e-5, abs=1e-9)
    assert metrics["agreement"]["top1"] == 1.0
    assert metrics["agreement"]["top5"] == 1.0
    assert metrics["tokens_dropped_nonfinite"] == 0


def test_the_floor_is_masked_to_the_same_window_as_the_kld(built_cache):
    """The floor is averaged over the scored window, not the whole shard.

    Two different windows must yield two different floors; if the floor were
    accumulated unmasked, both would report the whole-shard mean.
    """
    model, cache_dir, manifest, _ = built_cache
    late = score_loaded_student(model, cache_dir, _rewindow(manifest, 4, 8))
    early = score_loaded_student(model, cache_dir, _rewindow(manifest, 0, 4))
    assert late["kld"]["floor_mean"] != pytest.approx(
        early["kld"]["floor_mean"], rel=1e-6
    )
    # And each still equals its own window's measured KLD.
    for m in (late, early):
        assert m["kld"]["mean"] == pytest.approx(
            m["kld"]["floor_mean"], rel=1e-5, abs=1e-9
        )


def test_window_masks_scoring_to_the_manifest_window(built_cache):
    """score_window [4, 8) over 5 sequences of 8: exactly 5 * 4 positions."""
    model, cache_dir, manifest, dims = built_cache
    metrics = score_loaded_student(model, cache_dir, manifest)
    assert metrics["tokens_scored"] == dims["n"] * 4
    # Bucket ranges are quartiles of the window and cover every scored token.
    ranges = [b["range"] for b in metrics["by_position"]]
    assert ranges == ["4-5", "5-6", "6-7", "7-end"]
    assert sum(b["tokens"] for b in metrics["by_position"]) == metrics["tokens_scored"]


def test_window_end_is_exclusive(built_cache):
    """A window that stops short of the sequence end exercises the upper bound.

    With the fixture's own [4, 8) window the end equals the sequence length, so
    the `win_end < L` guard skips the comparison entirely and an off-by-one
    there is unobservable. [2, 6) makes it live.
    """
    model, cache_dir, manifest, dims = built_cache
    metrics = score_loaded_student(model, cache_dir, _rewindow(manifest, 2, 6))
    assert metrics["tokens_scored"] == dims["n"] * 4
    assert [b["range"] for b in metrics["by_position"]] == ["2-3", "3-4", "4-5", "5-end"]
    # Position 6 must be excluded: including it would add one token per
    # sequence and move the mean toward the [2, 7) value.
    wider = score_loaded_student(model, cache_dir, _rewindow(manifest, 2, 7))
    assert wider["tokens_scored"] == dims["n"] * 5
    assert wider["kld"]["mean"] != pytest.approx(metrics["kld"]["mean"], rel=1e-9)


def _count_delta_p(monkeypatch, model, cache_dir, manifest):
    """Total number of Delta-p values the aggregator was actually handed."""
    seen: list[int] = []
    real = kld_math.Aggregator.update_delta_p

    def spy(self, values, seq_ids=None):
        seen.append(int(np.asarray(values).size))
        return real(self, values, seq_ids)

    monkeypatch.setattr(kld_math.Aggregator, "update_delta_p", spy)
    metrics = score_loaded_student(model, cache_dir, manifest)
    return sum(seen), metrics


def test_delta_p_drops_only_the_position_with_no_successor(built_cache, monkeypatch):
    """Delta-p compares probabilities at the *observed next* token, so a scored
    position needs a valid token after it.

    Only the final sequence position lacks one. A window ending at the sequence
    end therefore yields one fewer Delta-p than KLD per sequence; a window
    ending earlier yields exactly as many, because its last position's
    successor is still inside the sequence.
    """
    model, cache_dir, manifest, dims = built_cache
    n = dims["n"]

    n_dp, metrics = _count_delta_p(monkeypatch, model, cache_dir, manifest)
    assert metrics["tokens_scored"] == n * 4
    assert n_dp == n * 3

    n_dp_mid, metrics_mid = _count_delta_p(
        monkeypatch, model, cache_dir, _rewindow(manifest, 2, 6)
    )
    assert metrics_mid["tokens_scored"] == n * 4
    assert n_dp_mid == n * 4


def test_delta_p_is_near_zero_for_a_perfect_student(built_cache):
    """student == teacher, so p_student - p_teacher is reconstruction error."""
    model, cache_dir, manifest, _ = built_cache
    metrics = score_loaded_student(model, cache_dir, manifest)
    assert metrics["delta_p"]["mean"] is not None
    assert abs(metrics["delta_p"]["mean"]) < 1e-3
    assert metrics["delta_p"]["rms"] < 1e-2


class _DriftedModel(_FakeModel):
    """The fake teacher with its peak nudged: a student that is close but wrong.

    Every perfect-student assertion in this file is satisfied by a pass that
    silently drops its measurements (a zeroed Delta-p, a KLD that reads back
    the floor), because for an identical model the true answers *are* ~0 and
    the floor. Scoring a deliberately-different model is what forces the
    numbers to be real.
    """

    def __call__(self, batch, cache=None):
        b, seq = batch.shape
        ids = mx.arange(self.vocab, dtype=mx.float32)[None, None, :]
        tok = batch.astype(mx.float32)[..., None]
        pos = mx.arange(seq, dtype=mx.float32)[None, :, None]
        peak = (tok * 7.0 + pos * 3.0 + 1.0) % self.vocab
        d = mx.minimum((ids - peak) % self.vocab, (peak - ids) % self.vocab)
        sharpness = 0.30 + 0.10 * pos + 0.07 * (tok % 5.0)
        return mx.broadcast_to(-d * sharpness, (b, seq, self.vocab))


def test_a_drifted_student_produces_real_nonzero_metrics(built_cache):
    """A student that differs from the teacher must move every metric off its
    perfect-student value, in the direction drift implies."""
    _model, cache_dir, manifest, _ = built_cache
    drifted = score_loaded_student(_DriftedModel(64), cache_dir, manifest)

    # KLD must sit clearly above the reconstruction floor. The floor is the
    # error a *perfect* student would score, so a drifted one cannot match it.
    # (The fixture's top-K 8 of vocab 64 is a coarse reconstruction, so the
    # floor is a large fraction of the total; the ratio here is ~3.)
    assert drifted["kld"]["floor_mean"] is not None
    assert drifted["kld"]["mean"] > 2 * drifted["kld"]["floor_mean"]
    # Delta-p must carry real magnitude. A zeroed or mis-indexed Delta-p would
    # leave rms at ~0 while the KLD still looked plausible.
    assert drifted["delta_p"]["rms"] > 1e-3
    assert drifted["delta_p"]["se"] is not None
    # The peak moved by one id, so the argmax disagrees at every position while
    # the teacher's argmax stays well inside the student's top five. The two
    # agreement rates measure different things, and this separates them.
    assert drifted["agreement"]["top1"] == 0.0
    assert drifted["agreement"]["top5"] == 1.0
    assert drifted["tokens_dropped_nonfinite"] == 0


def test_short_final_shard_keeps_one_cluster_per_sequence(built_cache, monkeypatch):
    """5 sequences at batch_size 2 leaves a short final shard (2/2/1).

    The clustered SE needs every scored token tagged with the calibration
    sequence it came from, and every sequence to be its own cluster. Assert the
    ids the aggregator actually receives, not just that an SE came out: a
    collapsed or duplicated tagging still produces a number, just the wrong one.
    """
    model, cache_dir, manifest, dims = built_cache
    seen: list[np.ndarray] = []
    real = kld_math.Aggregator.update

    def spy(self, kld, top1, top5, positions, seq_ids):
        seen.append(np.asarray(seq_ids).copy())
        return real(self, kld, top1, top5, positions, seq_ids)

    monkeypatch.setattr(kld_math.Aggregator, "update", spy)
    metrics = score_loaded_student(model, cache_dir, manifest)

    ids = np.concatenate([s.ravel() for s in seen])
    assert sorted(set(ids.tolist())) == list(range(dims["n"]))
    # Each sequence contributes the same number of scored tokens (the window
    # width), so no sequence may be over- or under-represented.
    counts = np.bincount(ids, minlength=dims["n"])
    assert counts.tolist() == [4] * dims["n"]
    assert metrics["kld"]["se_method"] == "clustered-by-sequence"
    assert metrics["kld"]["se"] is not None


def test_rewindowing_the_same_cache_changes_only_the_mask(built_cache):
    """The window lives in the manifest, not the shards: overlaying a different
    window rescores the same entry without a rebuild."""
    model, cache_dir, manifest, dims = built_cache
    metrics = score_loaded_student(
        model, cache_dir, _rewindow(manifest, 0, dims["seq"])
    )
    assert metrics["tokens_scored"] == dims["n"] * dims["seq"]


class _PaddedModel(_FakeModel):
    """A wider lm_head than the teacher vocab, dumping mass on the padded ids.

    The extra ids stay cold until the token id crosses ``hot_from``, so with the
    fixture's tokens (row *i* holds 8i..8i+7) the first shard is clean and the
    later ones are not. That is what makes "which batch warned" observable.
    """

    def __init__(self, vocab, extra=4, hot_from=16, value=10.0):
        super().__init__(vocab)
        self.extra, self.hot_from, self.value = extra, hot_from, value

    def __call__(self, batch, cache=None):
        head = super().__call__(batch, cache)
        b, seq = batch.shape
        hot = (batch >= self.hot_from).astype(mx.float32)[..., None]
        pad = mx.broadcast_to(hot * self.value, (b, seq, self.extra))
        # Cold rows get a very negative pad logit so their padded mass is ~0.
        return mx.concatenate([head, pad - (1.0 - hot) * 50.0], axis=-1)


def _capture_notes(monkeypatch):
    """Record (level, message) for scoring's info/warn, in emission order."""
    notes: list[tuple[str, str]] = []
    monkeypatch.setattr(scoring, "info", lambda m: notes.append(("info", m)))
    monkeypatch.setattr(scoring, "warn", lambda m: notes.append(("warn", m)))
    return notes


def test_padded_lm_head_warns_at_the_batch_that_crosses_the_threshold(
    built_cache, monkeypatch
):
    """The user's move on a high padded mass is to abort and rescore. Telling
    them after the whole pass has run costs a full student pass, hours on a
    large model, for information available at batch 1.
    """
    _model, cache_dir, manifest, _ = built_cache
    notes = _capture_notes(monkeypatch)
    scoring.score_loaded_student(_PaddedModel(64), cache_dir, manifest)

    warns = [i for i, (lvl, _) in enumerate(notes) if lvl == "warn"]
    assert warns, f"expected a padded-mass warning, got {notes}"
    first = notes[warns[0]][1]
    assert "batch 1" in first, first
    # ...and it must precede the end-of-run summary, not be it.
    summary = [i for i, (_, m) in enumerate(notes) if "max probability mass" in m]
    assert summary and warns[0] < summary[0]


def test_a_clean_padded_lm_head_does_not_warn(built_cache, monkeypatch):
    """A padded head whose extra ids are trained to nothing is normal and must
    not be escalated, otherwise the warning means nothing when it matters."""
    _model, cache_dir, manifest, _ = built_cache
    notes = _capture_notes(monkeypatch)
    # hot_from above every token id in the fixture: no id is ever hot.
    scoring.score_loaded_student(
        _PaddedModel(64, hot_from=10_000), cache_dir, manifest
    )
    assert [m for lvl, m in notes if lvl == "warn"] == []
    assert any("max probability mass" in m for _lvl, m in notes)


def test_a_nan_padded_mass_is_reported_not_swallowed(built_cache, monkeypatch):
    """`max(0.0, nan)` is 0.0, so a naive running max would report "no padded
    mass" for a student emitting non-finite logits, the case most worth
    surfacing."""
    _model, cache_dir, manifest, _ = built_cache
    notes = _capture_notes(monkeypatch)
    scoring.score_loaded_student(
        _PaddedModel(64, hot_from=16, value=float("inf")), cache_dir, manifest
    )
    summary = [m for lvl, m in notes if "max probability mass" in m and lvl == "warn"]
    assert summary, f"NaN padded mass was not warned about: {notes}"
    assert "nan" in summary[0].lower()
