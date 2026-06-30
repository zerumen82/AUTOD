#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Señales semánticas para GENERAR — anclas del LightLocal, sin listas por palabra del usuario."""

from typing import Dict

from roop.img_editor.gen_prompt_config import get_gen_thresholds

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


def score_anchor_overlap(prompt_en: str, anchor_phrase: str) -> float:
    return _nlp()._score((prompt_en or "").lower(), [anchor_phrase])