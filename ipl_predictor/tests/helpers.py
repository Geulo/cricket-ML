"""Test helpers."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def ipl_csv_path() -> Path:
    """Training CSV for tests (``pytest`` sets ``IPL_CSV_PATH`` to smoke fixture when bundled)."""
    env = os.environ.get("IPL_CSV_PATH")
    if env:
        return Path(env).expanduser().resolve()
    return (ROOT / "data" / "IPL.csv").resolve()
