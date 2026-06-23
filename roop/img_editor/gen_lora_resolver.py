#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Selección automática de LoRAs SDXL por escena (sin UI manual)."""

import os
import re
from typing import Dict, List, Tuple

LoraPick = Tuple[str, float, str]  # archivo, strength, motivo

# tags = palabras que pueden aparecer en prompt traducido (EN)
# boost = frases que se añaden al prompt si esta LoRA se activa (ayuda al modelo)
SCENE_LORA_CATALOG: List[Dict] = [
    {
        "file": "nsfw-xl-2.1.safetensors",
        "tags": ("nsfw", "nude", "naked", "explicit", "sex", "penis", "oral"),
        "strength": 0.50,
        "always": True,
        "boost": "",
        "label": "NSFW base",
    },
    {
        "file": "Nsfw Deepthroat.safetensors",
        "tags": ("sucking", "suck", "blowjob", "deepthroat", "oral", "fellatio", "penis", "kneeling", "on knees"),
        "strength": 0.72,
        "boost": "blowjob, deepthroat, oral sex, kneeling",
        "label": "Oral / deepthroat",
    },
    {
        "file": "Cuckold_Threesome_SDXL.safetensors",
        "tags": ("threesome", "another man", "second man", "two men", "waiting", "third person", "mmf", "cuckold"),
        "strength": 0.62,
        "sdxl_only": True,
        "boost": "threesome, two men, mmf, second man waiting",
        "label": "Threesome 2M+1F",
    },
    {
        "file": "Cuckold_Sex_-_Pony.safetensors",
        "tags": ("threesome", "another man", "second man", "two men", "waiting", "cuckold", "mmf"),
        "strength": 0.62,
        "pony_only": True,
        "boost": "threesome, two men, cuckold",
        "label": "Threesome Pony",
    },
    {
        "file": "NsfwPovAllInOneLoraSdxl-000009.safetensors",
        "tags": ("pov", "point of view", "first person"),
        "strength": 0.55,
        "boost": "pov, point of view",
        "label": "POV NSFW",
    },
    {
        "file": "realism3.safetensors",
        "tags": ("photorealistic", "skin", "texture", "detailed", "realistic", "raw photo"),
        "strength": 0.35,
        "boost": "detailed skin texture, photorealistic",
        "label": "Realism skin",
    },
]

MAX_SCENE_LORAS = 2  # además de la base NSFW


def _loras_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(root, "ui", "tob", "ComfyUI", "models", "loras")


def _lora_exists(name: str) -> bool:
    return bool(name) and os.path.isfile(os.path.join(_loras_dir(), name))


_SDXL_LORA_CACHE: Dict[str, bool] = {}


_CLIP_LORA_CACHE: Dict[str, bool] = {}


def lora_has_clip_keys(filename: str) -> bool:
    """True si la LoRA modifica CLIP (lora_te*). Si False → usar LoraLoaderModelOnly."""
    key = filename.lower()
    if key in _CLIP_LORA_CACHE:
        return _CLIP_LORA_CACHE[key]
    path = os.path.join(_loras_dir(), filename)
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
    """True si la LoRA tiene keys UNet/CLIP SDXL (no Flux/DiT)."""
    key = filename.lower()
    if key in _SDXL_LORA_CACHE:
        return _SDXL_LORA_CACHE[key]
    path = os.path.join(_loras_dir(), filename)
    ok = False
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as sf:
            for k in sf.keys():
                if k.startswith("lora_unet_") or k.startswith("lora_te"):
                    ok = True
                    break
                if k.startswith("diffusion_model.") or "lora_transformer" in k:
                    ok = False
                    break
    except Exception:
        ok = False
    _SDXL_LORA_CACHE[key] = ok
    return ok


def _is_pony_model(model_alias: str) -> bool:
    a = (model_alias or "").lower()
    return a in ("pony_realism", "cyberrealistic_pony") or "pony" in a


def _score_entry(prompt_l: str, entry: Dict) -> int:
    return sum(1 for tag in entry.get("tags", ()) if tag in prompt_l)


def _is_multi_person_scene(prompt_l: str) -> bool:
    markers = (
        "threesome", "two men", "another man", "second man", "mmf",
        "cuckold", "third person", "men", "stranger",
    )
    return sum(1 for m in markers if m in prompt_l) >= 1


def resolve_scene_loras(
    prompt_en: str,
    model_alias: str,
    base_strength: float = 0.55,
    image_type: str = "auto",
) -> Tuple[List[LoraPick], str]:
    """
    Devuelve lista de LoRAs a apilar y texto boost opcional para el prompt.
    """
    prompt_l = f" {(prompt_en or '').lower()} "
    pony = _is_pony_model(model_alias)
    picks: List[LoraPick] = []
    boosts: List[str] = []
    used_files = set()

    # 1) LoRA base NSFW siempre
    for entry in SCENE_LORA_CATALOG:
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

    # 2) LoRAs de escena por coincidencia de tags
    scored = []
    for entry in SCENE_LORA_CATALOG:
        if entry.get("always"):
            continue
        if entry.get("pony_only") and not pony:
            continue
        if entry.get("sdxl_only") and pony:
            continue
        fname = entry["file"]
        if not _lora_exists(fname) or fname.lower() in used_files:
            continue
        if entry.get("label") == "POV NSFW" and _is_multi_person_scene(prompt_l):
            continue
        score = _score_entry(prompt_l, entry)
        if score <= 0:
            continue
        if not _is_sdxl_lora(fname):
            print(f"[GenLoRA] Omitida escena (no SDXL): {fname}")
            boost = (entry.get("boost") or "").strip()
            if boost:
                boosts.append(boost)
            continue
        scored.append((score, entry))

    photoreal_modes = frozenset({"photoreal", "amateur_street", "smartphone"})
    want_realism = (image_type in photoreal_modes) or any(
        t in prompt_l for t in ("photorealistic", "raw photo", "realistic", "skin texture", "film grain")
    )

    scored.sort(key=lambda x: (-x[0], -x[1].get("strength", 0)))
    if want_realism:
        for entry in SCENE_LORA_CATALOG:
            if entry.get("label") != "Realism skin":
                continue
            fname = entry["file"]
            if not _lora_exists(fname) or fname.lower() in used_files or not _is_sdxl_lora(fname):
                continue
            scored.insert(0, (99, entry))
            break

    for _score, entry in scored[:MAX_SCENE_LORAS]:
        fname = entry["file"]
        picks.append((fname, float(entry["strength"]), entry.get("label", fname)))
        used_files.add(fname.lower())
        boost = (entry.get("boost") or "").strip()
        if boost:
            boosts.append(boost)

    boost_text = ", ".join(dict.fromkeys(
        b.strip() for part in boosts for b in part.split(",") if b.strip()
    ))
    return picks, boost_text


def format_lora_log(picks: List[LoraPick]) -> str:
    if not picks:
        return "(ninguna)"
    return " + ".join(f"{label}@{st:.2f}" for _f, st, label in picks)