# ABOUTME: pytest shared fixtures and path setup. Puts the project root on
# ABOUTME: sys.path so test modules can import `scripts.*` and `configs.*`.

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
