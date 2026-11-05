"""Shared pytest setup.

The scripts are intentionally not a package (they run as plain files), so make
``scripts/`` importable for the tests without requiring installation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
