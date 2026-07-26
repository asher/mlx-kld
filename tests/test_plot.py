"""``mlx-kld plot``: SVG scatter rendering from the results root."""

import json
import re
import subprocess
import sys

from mlx_kld.plot import _MARKERS, _MT
from mlx_kld.report import build_locked_json


def _record(student_name: str, mean: float, seed: int = 123,
            fmt: str = "mlx-affine", floor_mean=None) -> dict:
    return build_locked_json({
        "teacher": {"path": "org/teacher", "revision": None,
                    "precision": "bfloat16"},
        "student": {
            "path": f"/models/{student_name}",
            "format": fmt,
            "size_bytes": int(4e9),
            "effective_bpw": 4.5,
            "n_params": int(8e9),
            "quantization": {"kind": "affine", "bits": 4, "group_size": 64,
                             "mode": "affine"},
        },
        "tokenizer": {"mode": "mlx-student", "identical": True,
                      "encoding_parity": True, "forced": False, "diffs": []},
        "calibration": {"corpus": "wikitext", "num_samples": 8,
                        "max_seq_len": 512, "seed": seed, "top_k": 128,
                        "score_window": [256, 512], "long_context": False,
                        "corpus_tokens_hash": "aa00aa00aa00aa00"},
        "kld": {"mean": mean, "se": 0.0004, "floor_mean": floor_mean,
                "p50": 0.01, "p95": 0.1, "p99": 0.2, "p999": 0.5, "max": 1.0},
        "delta_p": {"mean": -0.001, "se": 0.0001, "rms": 0.01},
        "agreement": {"top1": 0.95, "top5": 0.99},
        "tokens_scored": 1000,
        "elapsed_seconds": 1.0,
        "scorer_version": "mlx-kld 0.1.0",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "by_position": [],
        "kld_histogram": {"bin_edges": [], "counts": []},
        "cache": {"dir": "/tmp/c", "status": "HIT", "top_k": 128},
    })


def _write(root, teacher_slug, name, record):
    d = root / teacher_slug
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(record))


def _run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "mlx_kld", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def _bf16_record(mean=0.0018):
    rec = _record("teacher-bf16", mean, fmt="mlx", floor_mean=mean)
    rec["student"]["quantization"] = {"kind": "none", "dtype": "bfloat16"}
    return rec


def test_plot_writes_svg_excluding_unquantized(tmp_path):
    _write(tmp_path, "teacher", "bf16.aaaaaaaa.json", _bf16_record())
    _write(tmp_path, "teacher", "q4.bbbbbbbb.json",
           _record("affine-4bit", 0.05, floor_mean=0.0018))
    _write(tmp_path, "teacher", "q5.cccccccc.json",
           _record("gguf-q5.gguf", 0.02, fmt="gguf", floor_mean=0.0018))
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    assert "excluded 1 unquantized run(s)" in r.stderr
    svg = svg_path.read_text()
    assert svg.startswith("<svg")
    assert "MLX affine" in svg and "GGUF" in svg  # legend
    assert "reconstruction floor" in svg
    assert "teacher-bf16" not in svg


def test_plot_default_output_path_and_size_axis(tmp_path):
    _write(tmp_path, "results/teacher", "q4.aaaaaaaa.json", _record("q4", 0.05))
    r = _run("plot", "--out-dir", str(tmp_path / "results"), "--x", "size",
             cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr
    out = tmp_path / "teacher-kld-vs-size.svg"
    assert out.exists()
    assert "file size (GB)" in out.read_text()


def test_plot_refuses_mixed_specs(tmp_path):
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", _record("a", 0.01, seed=123))
    _write(tmp_path, "teacher", "b.bbbbbbbb.json", _record("b", 0.02, seed=777))
    r = _run("plot", "--out-dir", str(tmp_path), "--svg",
             str(tmp_path / "x.svg"))
    assert r.returncode == 1
    assert "distinct run specs" in r.stderr


def test_plot_refuses_multiple_teachers_without_filter(tmp_path):
    _write(tmp_path, "t1", "a.aaaaaaaa.json", _record("a", 0.01))
    rec = _record("b", 0.02)
    rec["teacher"]["path"] = "org/other-teacher"
    _write(tmp_path, "t2", "b.bbbbbbbb.json", rec)
    r = _run("plot", "--out-dir", str(tmp_path))
    assert r.returncode == 1
    assert "--teacher" in r.stderr


def test_plot_labels_default_on_and_log_y(tmp_path):
    _write(tmp_path, "teacher", "q4.aaaaaaaa.json",
           _record("my-quant.gguf", 0.05, fmt="gguf"))
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path),
             "--log-y")
    assert r.returncode == 0, r.stderr
    svg = svg_path.read_text()
    assert ">my-quant</text>" in svg  # .gguf suffix stripped
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path),
             "--no-labels")
    assert r.returncode == 0, r.stderr
    assert ">my-quant</text>" not in svg_path.read_text()


def test_plot_legend_names_dominant_codec_family(tmp_path):
    # A K-quant GGUF with a few q8_0 aux tensors is still "K-quant".
    rec = _record("q6.gguf", 0.02, fmt="gguf")
    rec["student"]["quantization"] = {
        "kind": "gguf", "arch": "qwen3",
        "codecs": {"q6_k": 300, "q8_0": 2, "f32": 100},
    }
    _write(tmp_path, "teacher", "q6.aaaaaaaa.json", rec)
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    svg = svg_path.read_text()
    assert ">K-quant</text>" in svg
    assert ">GGUF</text>" not in svg


def test_plot_legend_lists_every_codec_family_present(tmp_path):
    # A set holding both K-quant and IQ files must say so. Reducing it to the
    # most common family would mislabel the others, and the reader has no way
    # to tell which square is which.
    kq = _record("kq.gguf", 0.02, fmt="gguf")
    kq["student"]["quantization"] = {"kind": "gguf", "arch": "qwen3",
                                     "codecs": {"q6_k": 300, "f32": 100}}
    iq = _record("iq.gguf", 0.05, fmt="gguf")
    iq["student"]["quantization"] = {"kind": "gguf", "arch": "qwen3",
                                     "codecs": {"iq3_s": 200, "q4_k": 20}}
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", kq)
    _write(tmp_path, "teacher", "b.bbbbbbbb.json", iq)
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    assert ">K-quant / IQ</text>" in svg_path.read_text()


def test_plot_labels_strip_teacher_prefix_and_noise(tmp_path):
    # teacher org/teacher -> base "teacher"; the label keeps only the tail,
    # minus noise words on either end (containers, modality, ignored mtp).
    _write(tmp_path, "teacher", "q6.aaaaaaaa.json",
           _record("Some_teacher-Q6_K_L.gguf", 0.02, fmt="gguf"))
    _write(tmp_path, "teacher", "q4.bbbbbbbb.json",
           _record("Org__teacher-oQ4e-mtp", 0.05))
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    svg = svg_path.read_text()
    assert ">Q6_K_L</text>" in svg
    assert ">oQ4e</text>" in svg


def test_plot_splits_colliding_quant_names_by_publisher(tmp_path):
    # Two publishers ship "Q5_K_M" at different recipes. The markers must differ
    # by fill rather than by hue (both are K-quant and belong to one family),
    # and the legend must be what names them: publisher belongs there, not
    # prefixed onto some point labels and not others.
    _write(tmp_path, "teacher", "a.aaaaaaaa.json",
           _record("unsloth__Q-GGUF/teacher-Q5_K_M.gguf", 0.02, fmt="gguf"))
    _write(tmp_path, "teacher", "b.bbbbbbbb.json",
           _record("bartowski__Q-GGUF/teacher-Q5_K_M.gguf", 0.03, fmt="gguf"))
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    svg = svg_path.read_text()
    color = _MARKERS["gguf"][1]
    # One point outlined and one filled, plus the matching legend swatches.
    assert svg.count(f'fill="white" stroke="{color}"') == 2
    assert ">unsloth</text>" in svg and ">bartowski</text>" in svg
    assert svg.count(">Q5_K_M</text>") == 2
    assert "unsloth Q5_K_M" not in svg


def test_place_labels_keeps_each_label_nearest_its_own_point():
    # A label closer to some other point than to the one it names is
    # misinformation. Placement must guarantee that, or draw a leader.
    from mlx_kld.plot import _box_dist, _place_labels

    pts = [(100.0, 100.0, "alpha"), (118.0, 108.0, "beta"),
           (135.0, 96.0, "gamma"), (150.0, 112.0, "delta")]
    placed = _place_labels(pts, [], 0.0, 0.0, 400.0, 300.0)
    centers = [(x, y) for x, y, _ in pts]
    assert len(placed) == len(pts)
    for i, (tx, ty, text, leader) in enumerate(placed):
        box = (tx, ty - 8.5, tx + 6.1 * len(text) + 2, ty + 2.5)
        own = _box_dist(box, centers[i])
        nearest = min(_box_dist(box, c) for c in centers)
        assert own <= nearest + 1e-9 or leader is not None, (
            f"{text} is nearer another point and has no leader")


def test_plot_labels_affine_bit_counts_as_q(tmp_path):
    # "4bit" is MLX's idiom; on a chart beside "Q4_K_S" it should read on the
    # same axis as its neighbours.
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", _record("teacher-4bit", 0.05))
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    svg = svg_path.read_text()
    assert ">q4</text>" in svg
    assert ">4bit</text>" not in svg


def test_plot_leaves_distinct_names_unsplit(tmp_path):
    # Several publishers whose file names already differ need no fill split;
    # adding one would be noise.
    _write(tmp_path, "teacher", "a.aaaaaaaa.json",
           _record("unsloth__Q-GGUF/teacher-Q5_K_M.gguf", 0.02, fmt="gguf"))
    _write(tmp_path, "teacher", "b.bbbbbbbb.json",
           _record("bartowski__Q-GGUF/teacher-Q6_K.gguf", 0.01, fmt="gguf"))
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    svg = svg_path.read_text()
    assert f'fill="white" stroke="{_MARKERS["gguf"][1]}"' not in svg
    assert ">Q5_K_M</text>" in svg and ">Q6_K</text>" in svg
    assert "unsloth" not in svg and "bartowski" not in svg


def test_plot_legend_sits_above_the_plot_frame(tmp_path):
    # Gridlines span the full plot width, so a legend inside the frame ends up
    # with a rule drawn through a row. It belongs in the header band.
    _write(tmp_path, "teacher", "q4.aaaaaaaa.json", _record("q4", 0.05))
    _write(tmp_path, "teacher", "q5.bbbbbbbb.json",
           _record("q5.gguf", 0.02, fmt="gguf"))
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    svg = svg_path.read_text()
    for label in ("MLX affine", "GGUF"):
        m = re.search(rf'<text x="[\d.]+" y="([\d.]+)"[^>]*>{label}</text>', svg)
        assert m, f"no legend entry for {label}"
        assert float(m.group(1)) < _MT


def test_plot_only_unquantized_fails_cleanly(tmp_path):
    _write(tmp_path, "teacher", "bf16.aaaaaaaa.json", _bf16_record())
    r = _run("plot", "--out-dir", str(tmp_path), "--svg",
             str(tmp_path / "x.svg"))
    assert r.returncode == 1
    assert "no quantized runs" in r.stderr


def test_plot_svg_is_wellformed_xml(tmp_path):
    import xml.etree.ElementTree as ET
    _write(tmp_path, "teacher", "a.aaaaaaaa.json",
           _record("a&b<c>", 0.05, floor_mean=0.001))
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    ET.fromstring(svg_path.read_text())  # must parse


def test_plot_json_record_flow_matches_compare(tmp_path):
    # A record compare accepts must plot; a stray invalid file is skipped.
    _write(tmp_path, "teacher", "good.aaaaaaaa.json", _record("good", 0.03))
    (tmp_path / "teacher" / "stub.json").write_text(json.dumps({"schema_version": 1}))
    r = _run("plot", "--out-dir", str(tmp_path), "--svg",
             str(tmp_path / "x.svg"))
    assert r.returncode == 0, r.stderr
    assert "skipping" in r.stderr


def test_plot_tolerates_records_missing_a_format(tmp_path):
    """`format` is a locked key but can be null, and two students can land on
    the same (bpw, mean KLD) point. A full tuple sort then compares None to a
    str and raises, so the sort must key on the numeric fields only."""
    a = _record("q4-a", 0.05, floor_mean=0.0018)
    b = _record("q4-b", 0.05, floor_mean=0.0018)
    a["student"]["format"] = None
    b["student"]["format"] = "mlx-affine"
    _write(tmp_path, "teacher", "a.aaaaaaaa.json", a)
    _write(tmp_path, "teacher", "b.bbbbbbbb.json", b)
    svg_path = tmp_path / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    assert svg_path.read_text().startswith("<svg")


def test_plot_creates_the_svg_parent_directory(tmp_path):
    _write(tmp_path, "teacher", "q4.bbbbbbbb.json",
           _record("affine-4bit", 0.05, floor_mean=0.0018))
    svg_path = tmp_path / "charts" / "nested" / "chart.svg"
    r = _run("plot", "--out-dir", str(tmp_path), "--svg", str(svg_path))
    assert r.returncode == 0, r.stderr
    assert svg_path.is_file()
