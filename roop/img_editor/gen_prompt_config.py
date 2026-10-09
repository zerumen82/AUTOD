#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Carga configuración de prompts GENERAR (sin strings hardcodeados en código)."""

import json
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
        "default_engine": str(raw.get("default_engine") or "longcat"),
        "engines": dict(raw.get("engines") or {}),
    }


def get_generate_ui_config() -> Dict[str, Any]:
    return dict(get_generation_prompt_config().get("generate_ui") or {})


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
