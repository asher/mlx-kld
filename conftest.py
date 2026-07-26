"""Test configuration.

``[tool.pytest.ini_options] pythonpath = ["src"]`` puts the package on ``sys.path``
for in-process imports. Several tests shell out to ``python -m mlx_kld``, and a
subprocess inherits ``PYTHONPATH`` rather than the parent's ``sys.path``, so export
it there too. Together these let a bare ``pytest`` pass from a fresh clone with no
editable install.
"""

from __future__ import annotations

import os
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"


# ``config`` is unused, but pytest matches hook arguments by name, so it stays.
def pytest_configure(config):
    existing = os.environ.get("PYTHONPATH", "")
    parts = [str(_SRC)] + [p for p in existing.split(os.pathsep) if p]
    os.environ["PYTHONPATH"] = os.pathsep.join(parts)
