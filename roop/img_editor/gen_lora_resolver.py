#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selección automática de LoRAs SDXL por escena — config JSON + solapamiento semántico."""

import os
from typing import Dict, List, Tuple

from roop.img_editor.gen_prompt_config import get_gen_thresholds, get_lora_catalog
from roop.img_editor.gen_semantic import is_multi_person_scene, score_anchor_overlap

LoraPick = Tuple[str, float, str]

MAX_SCENE_LORAS = 2


def _loras_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "ui", "tob", "ComfyUI", "models", "loras")


def _lora_exists(name: str) -> bool:
    path = _lora_path(name)
    return bool(path) and os.path.isfile(path)


def _lora_path(name: str) -> str:
    """Solo archivos dentro de la raíz de LoRAs, incluidos subdirectorios."""
    if not name or os.path.isabs(name) or os.path.splitdrive(name)[0]:
        return ""
    root = os.path.realpath(_loras_dir())
    path = os.path.realpath(os.path.join(root, name))
    try:
        if os.path.normcase(os.path.commonpath([root, path])) != os.path.normcase(root):
            return ""
    except ValueError:
        return ""
    return path


_SDXL_LORA_CACHE: Dict[Tuple[str, int, int], bool] = {}
_CLIP_LORA_CACHE: Dict[Tuple[str, int, int], bool] = {}


def lora_has_clip_keys(filename: str) -> bool:
    path = _lora_path(filename)
    if not path or not os.path.isfile(path):
        return False
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size)
    if key in _CLIP_LORA_CACHE:
        return _CLIP_LORA_CACHE[key]
    has = False
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("lora_te"):
                    has = True
                    break
    except Exception:
        has = False
    _CLIP_LORA_CACHE[key] = has
    return has


def _is_sdxl_lora(filename: str) -> bool:
    path = _lora_path(filename)
    if not path or not os.path.isfile(path):
        return False
    stat = os.stat(path)
    key = (path, stat.st_mtime_ns, stat.st_size)
    if key in _SDXL_LORA_CACHE:
        return _SDXL_LORA_CACHE[key]
    ok = False
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as sf:
            meta = sf.metadata() or {}
            base = meta.get("ss_base_model_version", "").lower()
            arch = meta.get("modelspec.architecture", "").lower()
            keys = list(sf.keys())
            adapter = any("lora_down.weight" in k or "lora_A.weight" in k for k in keys)
            ok = adapter and (base.startswith("sdxl") or "stable-diffusion-xl" in arch)
            if adapter and not base and not arch:
                # SDXL cross-attention usa contexto de 2048; SD1/2 no.
                ok = any(
                    k.startswith("lora_unet_") and "attn2_to_k" in k
                    and k.endswith("lora_down.weight")
                    and sf.get_slice(k).get_shape()[-1] == 2048
                    for k in keys
                )
    except Exception:
        ok = False
    _SDXL_LORA_CACHE[key] = ok
    return ok


def _is_pony_model(model_alias: str) -> bool:
    a = (model_alias or "").lower()
    return a in ("pony_realism", "cyberrealistic_pony") or "pony" in a


def lora_selection_error(filename: str, model_alias: str, is_sdxl: bool) -> str:
    if not _lora_path(filename):
        return "Ruta de LoRA inválida: debe estar dentro de la carpeta de LoRAs."
    if not _lora_exists(filename):
        return f"LoRA no instalada: {filename}."
    if not _is_sdxl_lora(filename):
        return f"LoRA no reconocida como adaptador SDXL compatible: {filename}."
    if not is_sdxl:
        return f"La LoRA {filename} requiere un modelo SDXL; no sirve con {model_alias}."
    entry = next((e for e in get_lora_catalog()
                  if os.path.normpath(e.get("file", "")).casefold()
                  == os.path.normpath(filename).casefold()), {})
    if entry.get("pony_only") and not _is_pony_model(model_alias):
        return f"La LoRA {filename} requiere un modelo Pony."
    if entry.get("sdxl_only") and _is_pony_model(model_alias):
        return f"La LoRA {filename} está configurada para SDXL base, no Pony."
    return ""


def get_available_lora_catalog(model_alias: str, is_sdxl: bool) -> List[Dict]:
    return [entry for entry in get_lora_catalog()
            if not lora_selection_error(entry.get("file", ""), model_alias, is_sdxl)]


def _score_entry(prompt_en: str, entry: Dict) -> float:
    best = 0.0
    for anchor in entry.get("anchors") or []:
        best = max(best, score_anchor_overlap(prompt_en, anchor))
    return best


def resolve_scene_loras(
    prompt_en: str,
    model_alias: str,
    base_strength: float = 0.55,
    image_type: str = "auto",
    include_nsfw: bool = True,
) -> Tuple[List[LoraPick], str]:
    """LoRAs desde catálogo JSON + scores semánticos (sin ifs por palabra del usuario).

    include_nsfw=False (checkbox «Contenido NSFW» de la UI): se omiten las
    entradas marcadas con "nsfw": true en config/generation_prompt.json.
    """
    pony = _is_pony_model(model_alias)
    picks: List[LoraPick] = []
    used_files = set()
    min_score = get_gen_thresholds().get("lora_scene_min_score", 0.12)
    multi_person = is_multi_person_scene(prompt_en)

    for entry in get_lora_catalog():
        if entry.get("manual_only"):
            continue
        if not include_nsfw and entry.get("nsfw"):
            continue
        if not entry.get("always"):
            continue
        if entry.get("pony_only") and not pony:
            continue
        if entry.get("sdxl_only") and pony:
            continue
        fname = entry["file"]
        if not _lora_exists(fname) or not _is_sdxl_lora(fname):
            if _lora_exists(fname) and not _is_sdxl_lora(fname):
                print(f"[GenLoRA] Omitida (no SDXL): {fname}")
            continue
        strength = float(entry.get("strength", base_strength))
        picks.append((fname, strength, entry.get("label", fname)))
        used_files.add(fname.lower())
        break

    scored = []
    for entry in get_lora_catalog():
        if entry.get("manual_only"):
            continue
        if not include_nsfw and entry.get("nsfw"):
            continue
        if entry.get("always"):
            continue
        if entry.get("pony_only") and not pony:
            continue
        if entry.get("sdxl_only") and pony:
            continue
        if entry.get("skip_when_multi_person") and multi_person:
            continue
        triggers = entry.get("image_type_trigger")
        if triggers:
            if (image_type or "auto") not in triggers:
                continue
            score = 1.0
        else:
            score = _score_entry(prompt_en, entry)
            if score < min_score:
                continue
        fname = entry["file"]
        if not _lora_exists(fname) or fname.lower() in used_files:
            continue
        if not _is_sdxl_lora(fname):
            print(f"[GenLoRA] Omitida escena (no SDXL): {fname}")
            continue
        scored.append((score, entry))

    scored.sort(key=lambda x: (-x[0], -x[1].get("strength", 0)))
    for _score, entry in scored[:MAX_SCENE_LORAS]:
        fname = entry["file"]
        picks.append((fname, float(entry["strength"]), entry.get("label", fname)))
        used_files.add(fname.lower())

    return picks, ""


def format_lora_log(picks: List[LoraPick]) -> str:
    if not picks:
        return "(ninguna)"
    return " + ".join(f"{label}@{st:.2f}" for _f, st, label in picks)
