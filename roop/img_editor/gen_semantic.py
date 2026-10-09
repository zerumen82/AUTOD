#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Señales semánticas para GENERAR — anclas del LightLocal, sin listas por palabra del usuario."""

from typing import Dict

from roop.img_editor.gen_prompt_config import get_gen_thresholds, get_style_gate_config

_analyzer = None


def _nlp():
    global _analyzer
    if _analyzer is None:
        from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer
        _analyzer = LightLocalIntentAnalyzer()
    return _analyzer


def get_scene_axis_scores(prompt_en: str) -> Dict[str, float]:
    return _nlp().get_axis_scores(prompt_en or "")


def scene_needs_anatomy_integrity(prompt_en: str) -> bool:
    axes = get_scene_axis_scores(prompt_en)
    th = get_gen_thresholds()
    return (
        axes.get("structural", 0.0) >= th.get("anatomy_structural", 0.06)
        and axes.get("attribute", 0.0) >= th.get("anatomy_attribute", 0.10)
    )


def is_multi_person_scene(prompt_en: str) -> bool:
    axes = get_scene_axis_scores(prompt_en)
    th = get_gen_thresholds()
    combo = axes.get("structural", 0.0) * 0.55 + axes.get("attribute", 0.0) * 0.45
    return combo >= th.get("multi_person_structural", 0.08) + 0.04


def route_prefers_editor(prompt_en: str) -> bool:
    """True si la instrucción pide cambiar algo concreto (añadir/quitar,
    ropa, fondo, pose/POV, apariencia, calidad) y por tanto conviene el
    Image Editor; False si describe una escena nueva (la foto de referencia
    se usa como base img2img). Umbrales en config/generation_prompt.json."""
    axes = get_scene_axis_scores(prompt_en)
    th = get_gen_thresholds()
    for th_name, axis in (
        ("route_structural", "structural"),
        ("route_pose", "pose"),
        ("route_attribute", "attribute"),
    ):
        limit = th.get(th_name)
        if limit is None:
            continue
        if axes.get(axis, 0.0) >= float(limit):
            return True
    return False


def score_anchor_overlap(prompt_en: str, anchor_phrase: str) -> float:
    # El analizador ligero expone _score_texts (no existe _score: la llamada
    # vieja lanzaba AttributeError y rompía el scoring de LoRAs SDXL).
    return _nlp()._score_texts(prompt_en or "", [anchor_phrase])


def score_style_gate(prompt_en: str) -> float:
    """0..1 — solape del prompt con los conceptos de estilo no fotorreal."""
    cfg = get_style_gate_config()
    concepts = cfg.get("concepts") or []
    if not concepts or not (prompt_en or "").strip():
        return 0.0
    scorer = getattr(_nlp(), "_score_texts", None) or getattr(_nlp(), "_score", None)
    if scorer is None:
        return 0.0
    return float(scorer(prompt_en, concepts))


def prompt_requests_nonphotoreal_style(prompt_en: str) -> bool:
    """True si el prompt pide un estilo no fotorreal (dibujo/ilustración/…).

    Umbrales y conceptos en config/generation_prompt.json → style_gate.
    """
    return score_style_gate(prompt_en) >= get_style_gate_config()["threshold"]