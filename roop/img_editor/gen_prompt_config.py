#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Carga configuración de prompts GENERAR (sin strings hardcodeados en código)."""

import json
import math
import os
from typing import Any, Dict, List

_CACHE: Dict[str, Any] = {}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_generation_prompt_config() -> Dict[str, Any]:
    if _CACHE:
        return _CACHE
    path = os.path.join(_project_root(), "config", "generation_prompt.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            _CACHE.update(json.load(f))
    return _CACHE


def get_prompt_extras() -> Dict[str, str]:
    return dict(get_generation_prompt_config().get("prompt_extras") or {})


def get_gen_thresholds() -> Dict[str, float]:
    raw = get_generation_prompt_config().get("thresholds") or {}
    return {k: float(v) for k, v in raw.items()}


def get_lora_catalog() -> List[Dict[str, Any]]:
    return list(get_generation_prompt_config().get("lora_catalog") or [])


def get_route_editor_config() -> Dict[str, Any]:
    raw = get_generation_prompt_config().get("route_editor") or {}
    return {
        "default_engine": str(raw.get("default_engine") or "kontext"),
        "engines": dict(raw.get("engines") or {}),
    }


def get_kontext_model_config() -> Dict[str, Any]:
    """Read the Editor's configured Kontext model and native sampling recipe."""
    path = os.path.join(_project_root(), "config", "model_configs.json")
    with open(path, encoding="utf-8") as stream:
        model = json.load(stream)["kontext"]
    model_file = model["model_file"]
    steps = int(model["steps"])
    cfg = float(model["cfg"])
    if (
        not isinstance(model_file, str) or not model_file.strip()
        or "kontext" not in model_file.casefold()
        or steps < 1 or not math.isfinite(cfg) or cfg <= 0
    ):
        raise ValueError("Invalid Kontext model configuration")
    return {**model, "steps": steps, "cfg": cfg}


def get_generate_ui_config() -> Dict[str, Any]:
    return dict(get_generation_prompt_config().get("generate_ui") or {})


def get_editor_ui_config() -> Dict[str, Any]:
    return dict(get_generation_prompt_config().get("editor_ui") or {})


def get_rewriter_config() -> Dict[str, Any]:
    return dict(get_generation_prompt_config().get("rewriter") or {})


def get_style_gate_config() -> Dict[str, Any]:
    """Gate de estilo: decide si el prompt pide un estilo NO fotorreal.

    Concepts (descripciones EN/ES) + threshold viven en
    config/generation_prompt.json — sin palabras de usuario hardcodeadas.
    """
    raw = get_generation_prompt_config().get("style_gate") or {}
    return {
        "concepts": list(raw.get("concepts") or []),
        "threshold": float(raw.get("threshold", 0.25)),
    }


def get_adult_gate_config() -> Dict[str, Any]:
    """Gate adulto: decide si el PROMPT pide contenido adulto.

    El checkbox «Contenido NSFW» es solo permiso; sin esta señal medida el
    prefijo adulto, el anti-censor y las LoRAs adultas automáticas NUNCA se
    inyectan — el prompt manda siempre (v5.1210, «CASTILLO CON UNICORNIOS»
    salía mujer desnuda por el prefijo de 08/10).

    Concepts (descripciones EN/ES) + threshold calibrado con16 escenas
    neutrales y9 adultas (FP=0, FN=0; neutro máx 0,095 / adulto mín 0,152).
    """
    raw = get_generation_prompt_config().get("adult_gate") or {}
    return {
        "concepts": list(raw.get("concepts") or []),
        "threshold": float(raw.get("threshold", 0.13)),
    }
