from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

_ROOT = Path(__file__).resolve().parents[3]
_SETTINGS_PATH = _ROOT / "config" / "editor_selection.json"
_CACHE: Dict[str, Any] | None = None


def load_settings() -> Dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))
    return _CACHE


def get_config() -> Dict[str, Any]:
    return load_settings()
