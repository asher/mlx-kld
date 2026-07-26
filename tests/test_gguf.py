"""GGUF student support: dispatch seam, dependency guard, CLI validation.

No real GGUF is loaded. gmlx is faked at the module seam. End-to-end scoring
against a real .gguf is exercised manually (it needs multi-GB model files).
"""

import subprocess
import sys
import types

import pytest

from mlx_kld._deps import require_gguf
from mlx_kld.models import _load_model, is_gguf_path


def _make_gguf(tmp_path, name="student-Q4_K_M.gguf", size=64):
    p = tmp_path / name
    p.write_bytes(b"\x00" * size)
    return p


# ---------- is_gguf_path ----------

def test_is_gguf_path_true_for_existing_file(tmp_path):
    p = _make_gguf(tmp_path)
    assert is_gguf_path(str(p))


def test_is_gguf_path_false_for_missing_file(tmp_path):
    assert not is_gguf_path(str(tmp_path / "nope.gguf"))


def test_is_gguf_path_false_for_dir_and_other_files(tmp_path):
    assert not is_gguf_path(str(tmp_path))
    other = tmp_path / "model.safetensors"
    other.write_bytes(b"\x00")
    assert not is_gguf_path(str(other))


# ---------- dependency guard ----------

def test_require_gguf_hint_when_gmlx_missing(monkeypatch):
    """On an interpreter gmlx can be installed on, point at the install line."""
    # A None entry in sys.modules makes `import gmlx` raise ImportError. The
    # version is pinned too, so both branches are exercised on every
    # interpreter the matrix runs rather than only the one underfoot.
    monkeypatch.setitem(sys.modules, "gmlx", None)
    monkeypatch.setattr(sys, "version_info", (3, 11, 0))
    with pytest.raises(ImportError, match=r"mlx-kld\[gguf\]"):
        require_gguf()


def test_require_gguf_names_the_floor_below_3_11(monkeypatch):
    """Below 3.11, `pip install 'mlx-kld[gguf]'` cannot succeed, because gmlx's
    own metadata rejects the interpreter. Printing it anyway would send the user
    at a raw pip-resolver error, so the guard names the floor instead."""
    monkeypatch.setitem(sys.modules, "gmlx", None)
    monkeypatch.setattr(sys, "version_info", (3, 10, 18))
    with pytest.raises(ImportError, match="needs Python 3.11") as exc:
        require_gguf()
    assert "3.10" in str(exc.value)
    assert "pip install" not in str(exc.value)


# ---------- loader dispatch ----------

def test_load_model_dispatches_gguf_to_gmlx(tmp_path, monkeypatch):
    p = _make_gguf(tmp_path)
    sentinel_model, sentinel_config = object(), {"vocab_size": 128}
    fake = types.ModuleType("gmlx")
    seen = {}

    def fake_load(path, **kwargs):
        seen["path"] = path
        return sentinel_model, sentinel_config, object()

    fake.load_model = fake_load
    monkeypatch.setitem(sys.modules, "gmlx", fake)

    model, config = _load_model(str(p))
    assert model is sentinel_model
    assert config is sentinel_config
    assert seen["path"] == str(p)


# ---------- CLI validation (subprocess; no model load) ----------

def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "mlx_kld", *args],
        capture_output=True, text=True,
    )


def test_score_help_mentions_hf_source():
    r = _run("score", "--help")
    assert r.returncode == 0
    assert "--hf-source" in r.stdout


def test_score_missing_gguf_student_fails_cleanly(tmp_path):
    r = _run("score", "org/teacher", str(tmp_path / "nope.gguf"))
    assert r.returncode != 0
    assert "GGUF student does not exist" in r.stderr


def test_score_hf_source_rejected_for_dir_student(tmp_path):
    r = _run("score", "org/teacher", str(tmp_path), "--hf-source", "org/x")
    assert r.returncode != 0
    assert "--hf-source only applies to GGUF students" in r.stderr


def test_require_gguf_accepts_an_importable_gmlx_on_any_interpreter(monkeypatch):
    """The Python-3.11 floor is gmlx's, and it is enforced at install time.

    Checking the interpreter version *before* attempting the import would make
    require_gguf raise on 3.10 even where gmlx is importable, which is exactly
    what the test suite arranges, so the 3.10 CI leg would go red on five tests
    that never touch a real gmlx. Diagnose the failure; do not gate on it.
    """
    monkeypatch.setitem(sys.modules, "gmlx", types.ModuleType("gmlx"))
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    require_gguf()   # must not raise


def test_require_gguf_names_the_python_floor_when_the_import_fails_on_old_python(
    monkeypatch,
):
    """On 3.10 with gmlx absent, `pip install 'mlx-kld[gguf]'` is advice that
    cannot work, because the resolver will refuse. Say why instead."""
    monkeypatch.setitem(sys.modules, "gmlx", None)
    monkeypatch.setattr(sys, "version_info", (3, 10, 0, "final", 0))
    with pytest.raises(ImportError, match="Python 3.11"):
        require_gguf()
