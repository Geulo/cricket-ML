"""Pytest configuration and shared paths."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config) -> None:  # noqa: ARG001
    """Prefer bundled smoke CSV for tests when present (does not override user's ``IPL_CSV_PATH``)."""
    smoke = ROOT / "tests" / "fixtures" / "synthetic_smoke.csv"
    if "IPL_CSV_PATH" not in os.environ and smoke.is_file():
        os.environ["IPL_CSV_PATH"] = str(smoke)


def ipl_csv_for_tests() -> Path:
    """CSV path used by tests (env ``IPL_CSV_PATH`` or ``data/IPL.csv``)."""
    env = os.environ.get("IPL_CSV_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "data" / "IPL.csv").resolve()
