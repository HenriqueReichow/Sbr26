"""Config loading (stdlib tomllib — no PyYAML needed)."""
from __future__ import annotations

import tomllib
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "rb") as f:
        return tomllib.load(f)
