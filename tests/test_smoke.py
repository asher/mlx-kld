import subprocess
import sys


def test_help_exits_zero():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_kld", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "mlx-kld" in result.stdout


def test_version_flag():
    result = subprocess.run(
        [sys.executable, "-m", "mlx_kld", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
