"""Catálogo local de presets de Mejorar fotos.

Los JSON de config/enhance/presets son la fuente de las recetas. Los iconos
vectoriales viven en assets/enhance_presets y no requieren red ni paquetes.
"""

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

from .logic import load_enhance_config


ROOT = Path(__file__).resolve().parents[3]
PRESET_DIR = ROOT / "config" / "enhance" / "presets"
ICON_DIR = ROOT / "assets" / "enhance_presets"
SLIDER_KEYS = (
    "denoise_strength", "sharpen_strength", "recover_faces_strength",
    "upscale_factor", "exposure", "shadow", "highlight", "temperature",
    "grain_strength", "grain_size", "deband_strength", "vibrance",
    "face_model",
)


def _valid_recipe(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    sliders = data.get("sliders")
    if not isinstance(data.get("name"), str) or not isinstance(sliders, dict):
        return False
    if not all(key in sliders for key in SLIDER_KEYS):
        return False
    limits = load_enhance_config()["control_limits"]
    try:
        for key in SLIDER_KEYS:
            rule = limits[key]
            value = sliders[key]
            if "choices" in rule:
                if value not in rule["choices"]:
                    return False
            else:
                scaled = float(value) * 100 if key == "recover_faces_strength" else float(value)
                if not rule["minimum"] <= scaled <= rule["maximum"]:
                    return False
        return True
    except (ValueError, TypeError, KeyError):
        return False


def load_presets() -> "OrderedDict[str, dict]":
    """Devuelve recetas válidas, con fábrica en el orden elegido en config."""
    found = {}
    if PRESET_DIR.exists():
        for path in PRESET_DIR.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                print(f"[Enhance] Preset ilegible {path.name}: {exc}")
                continue
            if not _valid_recipe(data):
                print(f"[Enhance] Preset inválido {path.name}")
                continue
            name = data["name"].strip()
            if not name or name.casefold() in (item["name"].casefold() for item in found.values()):
                print(f"[Enhance] Preset duplicado/vacío {path.name}")
                continue
            data["_path"] = path
            data["_factory"] = not path.stem.startswith("custom_")
            found[path.stem] = data

    ordered = OrderedDict()
    for slug in load_enhance_config().get("preset_order", []):
        data = found.pop(slug, None)
        if data is not None:
            ordered[data["name"]] = data
    for data in sorted(found.values(), key=lambda item: item["name"].casefold()):
        ordered[data["name"]] = data
    return ordered


def icon_for_preset(data: dict) -> Path:
    candidate = ICON_DIR / (data["_path"].stem + ".svg")
    return candidate if candidate.is_file() else ICON_DIR / "custom.svg"


def save_custom_preset(name: str, sliders: dict) -> str:
    """Guarda sin rutas aportadas por el usuario ni sobrescribir otro preset."""
    name = (name or "").strip()
    if not re.fullmatch(r"[\w .-]{1,50}", name, flags=re.UNICODE):
        raise ValueError("Usa 1 a 50 letras, números, espacios, puntos o guiones.")
    if any(name.casefold() == existing.casefold() for existing in load_presets()):
        raise ValueError("Ya existe un preset con ese nombre; elige otro.")
    data = {"name": name, "description": "Preset personalizado", "sliders": sliders}
    if not _valid_recipe(data):
        raise ValueError("Los ajustes del preset están fuera del rango permitido.")
    PRESET_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:16]
    path = PRESET_DIR / f"custom_{digest}.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path.name
