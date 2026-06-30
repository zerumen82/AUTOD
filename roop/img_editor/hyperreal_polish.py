#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acabado post-imagen guiado por análisis CV (sin textos fijos en prompts)."""

from typing import Dict, Optional, Tuple

from PIL import Image


def polish_result_image(
    image: Image.Image,
    tier: str = "hd",
    *,
    fast: bool = False,
) -> Tuple[Image.Image, str, Optional[Dict]]:
    """
    Refuerza detalle/nitidez/textura según scores de analyze_image().
    Fuerza adaptativa: posterización alta → más textura; ya nítida → toque mínimo.
    """
    from roop.img_editor.image_quality_pipeline import get_quality_finisher

    from roop.img_editor.image_quality_pipeline import TIER_TARGET_SIDE

    finisher = get_quality_finisher()
    tier = (tier or "hd").lower()
    analysis = finisher.analyze_image(image, tier=tier)
    profile = analysis.get("profile", {})
    scores = analysis.get("scores", {})

    restore = float(scores.get("restore", scores.get("soft_need", 0.0)))
    poster = float(scores.get("poster", 0.0))
    sharp = float(scores.get("sharpness", 0.5))
    max_side = max(image.size)
    tier_target = TIER_TARGET_SIDE.get(tier, 1920)
    already_at_tier = max_side >= int(tier_target * 0.92)

    need = max(restore, poster * 0.85)
    if fast and need < 0.22 and sharp > 0.55:
        return image, "acabado rápido omitido", analysis
    if need < 0.08 and sharp > 0.72:
        return image, "acabado omitido (imagen ya limpia)", analysis

    out = image
    boost = float(profile.get("detail_boost_blend", 0.12)) * _lerp(0.55, 0.85, need)
    boost = max(0.08, min(0.30, boost))
    if fast:
        boost = min(boost, 0.12)
    if already_at_tier:
        boost = min(boost, 0.14)
    use_clahe = poster < 0.35 and max(out.size) <= 3200 and not fast
    if boost > 0.09:
        out = finisher.detail_boost(out, blend=boost, use_clahe=use_clahe)

    sharpen = float(profile.get("sharpen", 1.12))
    amt = sharpen * _lerp(0.88, 1.0, need)
    amt = max(1.05, min(1.32, amt))
    if fast:
        amt = min(amt, 1.08)
    if amt > 1.04:
        out = finisher.sharpen(out, amount=amt)

    notes = []
    if fast:
        notes.append("rápido")
    if boost > 0.09:
        notes.append(f"detalle {boost:.0%}")
    if amt > 1.04:
        notes.append(f"nitidez {amt:.2f}")

    if (
        not fast
        and not already_at_tier
        and poster > 0.22
        and float(profile.get("texture_blend", 0)) > 0
    ):
        tex = min(0.20, float(profile.get("texture_blend", 0.12)) * _lerp(0.35, 0.55, poster))
        out, tex_note = finisher.texture_recovery(
            out, tier=tier, blend=tex, texture_recovery=True,
        )
        if tex_note:
            notes.append(tex_note)

    note = "acabado adaptativo (" + ", ".join(notes) + ")" if notes else "sin acabado"
    return out, note, analysis


def _lerp(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, float(t)))
    return a + (b - a) * t