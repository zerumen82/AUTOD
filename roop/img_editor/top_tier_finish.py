#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acabado TOP para modo Solo mejorar y reutilizable en otros edits.
Híbrido: prep clásico → LongCat (realismo) → Lanczos 8K (sin tiles ONNX / sin rejilla).
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image


ProgressCallback = Callable[[Dict[str, Any]], None]
CancelCheck = Callable[[], bool]


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clip01(t)


def _quality_gen_prompt(scores: Dict[str, Any], tier: str) -> str:
    poster = float(scores.get("poster", 0.0))
    strength = "light" if poster < 0.42 else "strong" if poster < 0.58 else "full"
    tier = (tier or "hd").upper()
    return (
        f"Instruction: Enhance this exact photograph to ultra-realistic {tier} photographic quality. "
        "Keep identical composition, camera angle, lighting, background, faces, bodies and scene. "
        "Remove posterization, AI plastic skin and compression artifacts. "
        "Add natural skin pores, realistic fabric texture, hair detail and true camera-grade microcontrast. "
        f"Apply {strength} photorealistic refinement only — same photo, higher fidelity."
    )


def _edit_gen_prompt(analysis: Dict[str, Any]) -> str:
    mag = float(analysis.get("magnitude", 0.5))
    strength = "subtle" if mag < 0.62 else "moderate" if mag < 0.78 else "strong"
    return (
        "Instruction: Refine this edited photograph to ultra-realistic photographic quality. "
        "Keep the exact edit result unchanged: same composition, poses, body, clothing state, "
        "lighting, background, faces and scene. "
        "Add natural skin pores, realistic skin and fabric texture, hair detail and camera-grade microcontrast. "
        "Remove AI plastic look, posterization and edit artifacts. "
        f"Apply {strength} photorealistic polish only — preserve the edit, improve fidelity."
    )


def should_edit_top_finish(analysis: Dict[str, Any]) -> bool:
    """Activa acabado TOP post-edit por magnitud/ejes semánticos (sin hardcode de palabras)."""
    if bool(analysis.get("quality_only", False)):
        return False
    if bool(analysis.get("quality_hybrid", False)):
        return False
    mag = float(analysis.get("magnitude", 0.0))
    if bool(analysis.get("body_transform", False)):
        return True
    if bool(analysis.get("has_quality_request", False)):
        return True
    if float(analysis.get("body_transform_intensity", 0.0)) >= 0.12 and mag >= 0.48:
        return True
    return mag >= 0.55


def _generative_polish_pass(
    image: Image.Image,
    analysis: Dict[str, Any],
    tier: str,
    *,
    mode: str = "quality",
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> Tuple[Optional[Image.Image], str]:
    scores = analysis.get("scores", {})
    poster = float(scores.get("poster", 0.0))
    mag = float(analysis.get("magnitude", 0.5))

    if mode == "edit":
        if poster < 0.18 and mag < 0.52:
            return None, ""
    elif poster < 0.30:
        return None, ""

    try:
        from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    except Exception as e:
        print(f"[TopTier] Comfy client no disponible: {e}")
        return None, ""

    client = get_flux_edit_comfy_client()
    if not client.is_available():
        print("[TopTier] ComfyUI no responde — omitiendo paso generativo")
        return None, ""

    version = "LongCat-Image-Edit-Q4_K_S.gguf"
    ok, load_msg = client.load(version)
    if not ok:
        print(f"[TopTier] LongCat Full no cargado: {load_msg}")
        return None, ""

    ow, oh = image.size
    if mode == "edit":
        need = max(poster, mag * 0.55)
        denoise = _clip01(0.14 + need * 0.22)
        steps = int(_lerp(14, 18, need))
        guidance = 3.0
        prompt = _edit_gen_prompt(analysis)
        blend = _lerp(0.30, 0.52, need)
        label = "pulido edit"
    else:
        denoise = _clip01(0.24 + poster * 0.32)
        steps = int(_lerp(16, 22, poster))
        guidance = 3.2
        prompt = _quality_gen_prompt(scores, tier)
        blend = _lerp(0.50, 0.78, poster)
        label = "realismo"

    negative = "low quality, blurry, cartoon, plastic skin, posterization, grid, seams, artifacts"

    print(
        f"[TopTier] LongCat Full {label} — denoise={denoise:.2f}, steps={steps}, "
        f"poster={poster:.2f}, mag={mag:.2f}",
        flush=True,
    )
    if progress_callback:
        progress_callback({
            "phase": "Realismo generativo" if mode == "quality" else "Pulido TOP post-edit",
            "progress": 0.88 if mode == "edit" else 0.22,
            "detail": f"LongCat d={denoise:.2f}",
        })
    if cancel_check and cancel_check():
        return None, "cancelado"

    gen_result, err = client.generate(
        image,
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        denoise=denoise,
        reference_latents_method="index_timestep_zero",
        progress_callback=progress_callback,
    )
    if gen_result is None or gen_result.image is None:
        print(f"[TopTier] Generativo omitido: {err}")
        return None, ""

    gen = gen_result.image.convert("RGB")
    if gen.size != (ow, oh):
        gen = gen.resize((ow, oh), Image.LANCZOS)

    out = Image.blend(image.convert("RGB"), gen, blend)
    note = f"LongCat {label} ({blend:.0%}, d={denoise:.2f})"
    print(f"[TopTier] {note}", flush=True)
    return out, note


def _generative_realism_pass(
    image: Image.Image,
    analysis: Dict[str, Any],
    tier: str,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> Tuple[Optional[Image.Image], str]:
    return _generative_polish_pass(
        image, analysis, tier, mode="quality",
        progress_callback=progress_callback, cancel_check=cancel_check,
    )


def _preserve_faces(original: Image.Image, edited: Image.Image) -> Image.Image:
    try:
        from roop.img_editor.face_preserver import FacePreserver
        fp = FacePreserver()
        ok, _ = fp.initialize()
        if not ok:
            return edited
        return fp.preserve_faces(original, edited, method="swap")
    except Exception as e:
        print(f"[TopTier] Face preserve omitido: {e}")
        return edited


def run_top_tier_quality(
    image: Image.Image,
    *,
    enhance_tier: str = "hd",
    use_generative: bool = True,
    preserve_faces: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    from roop.img_editor.image_quality_pipeline import get_quality_finisher

    finisher = get_quality_finisher()
    tier = (enhance_tier or "hd").lower()
    analysis = finisher.analyze_image(image, tier=tier)
    profile = analysis.get("profile", {})
    scores = analysis.get("scores", {})
    notes = []

    gen_label = "generativo + " if use_generative else ""
    print(
        f"[TopTier] Pipeline TOP tier={tier.upper()} — "
        f"{gen_label}Lanczos (sin tiles ONNX) | {analysis.get('summary', '')}",
        flush=True,
    )

    out = image.convert("RGB")

    if not profile.get("skip_depixelize", False):
        dep = float(profile.get("depixelize_blend", 0.5))
        if progress_callback:
            progress_callback({"phase": "Desposterizar", "progress": 0.08, "detail": f"{dep:.0%}"})
        out = finisher.depixelize(out, blend_strength=dep, tier=tier)
        notes.append(f"desposterizar {dep:.0%}")

    if cancel_check and cancel_check():
        return image, "cancelado", analysis

    if use_generative:
        gen_img, gen_note = _generative_realism_pass(
            out, analysis, tier, progress_callback=progress_callback, cancel_check=cancel_check,
        )
        if gen_img is not None:
            out = gen_img
            notes.append(gen_note)
    else:
        notes.append("LongCat realismo omitido")

    if cancel_check and cancel_check():
        return out, " + ".join(notes), analysis

    if preserve_faces:
        if progress_callback:
            progress_callback({"phase": "Preservar caras", "progress": 0.58, "detail": "identidad original"})
        out = _preserve_faces(image, out)
        notes.append("caras preservadas")

    if cancel_check and cancel_check():
        return out, " + ".join(notes), analysis

    if progress_callback:
        progress_callback({
            "phase": "Upscale Lanczos",
            "progress": 0.62,
            "detail": f"→ {tier.upper()} sin rejilla",
        })
    out, up_note = finisher.upscale_lanczos_to_tier(
        out, tier, progress_callback=progress_callback,
    )
    if up_note:
        notes.append(up_note)

    band = float(profile.get("poster_band_blend", 0.0))
    if band > 0.05:
        if progress_callback:
            progress_callback({"phase": "Naturalizar bandas", "progress": 0.78, "detail": f"{band:.0%}"})
        out = finisher.soften_poster_bands(out, blend=band)
        notes.append(f"naturalizar bandas {band:.0%}")

    poster = float(scores.get("poster", 0.0))
    if tier == "8k" or max(out.size) >= 3600:
        stripe_mix = _lerp(0.12, 0.26, poster)
        if stripe_mix > 0.10:
            out = finisher.reduce_stripe_artifacts(out, blend=stripe_mix)
            notes.append(f"anti-rayas {stripe_mix:.0%}")
    color_mix = _lerp(0.12, 0.24, max(poster, float(scores.get("restore", 0.0))))
    if color_mix > 0.10:
        out = finisher.enhance_color(out, blend=color_mix)
        notes.append(f"color {color_mix:.0%}")

    if poster < 0.52:
        amt = min(1.12, float(profile.get("sharpen", 1.08)))
        out = finisher.sharpen(out, amount=amt)
        notes.append(f"nitidez {amt:.2f}")
    else:
        notes.append("nitidez omitida (evitar bandas)")

    note_str = " + ".join(notes) if notes else "top tier"
    print(f"[TopTier] Listo — {note_str}", flush=True)
    return out, note_str, analysis


def run_top_tier_edit_finish(
    image: Image.Image,
    original: Image.Image,
    analysis: Dict[str, Any],
    *,
    enhance_tier: str = "hd",
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """
    Acabado TOP ligero tras edits con prompt (alta mag, body transform, calidad pedida).
    Sin upscale ONNX — solo pulido generativo + naturalizar + caras.
    """
    from roop.img_editor.image_quality_pipeline import get_quality_finisher

    if not should_edit_top_finish(analysis):
        return image, "acabado edit omitido (mag baja)", {}

    finisher = get_quality_finisher()
    tier = (enhance_tier or "hd").lower()
    cv_analysis = finisher.analyze_image(image, tier=tier)
    profile = cv_analysis.get("profile", {})
    scores = cv_analysis.get("scores", {})
    poster = float(scores.get("poster", 0.0))
    mag = float(analysis.get("magnitude", 0.0))
    notes = []

    print(
        f"[TopTier] Post-edit TOP — mag={mag:.2f} poster={poster:.2f} "
        f"body={analysis.get('body_transform', False)}",
        flush=True,
    )

    out = image.convert("RGB")
    merged = {**analysis, "scores": scores, "profile": profile}

    if poster > 0.20 and not profile.get("skip_depixelize", False):
        dep = min(0.42, float(profile.get("depixelize_blend", 0.32)))
        if progress_callback:
            progress_callback({"phase": "Suavizar poster", "progress": 0.84, "detail": f"{dep:.0%}"})
        out = finisher.depixelize(out, blend_strength=dep, tier=tier)
        notes.append(f"depixelize {dep:.0%}")

    if cancel_check and cancel_check():
        return image, "cancelado", cv_analysis

    gen_img, gen_note = _generative_polish_pass(
        out, merged, tier, mode="edit",
        progress_callback=progress_callback, cancel_check=cancel_check,
    )
    if gen_img is not None:
        out = gen_img
        notes.append(gen_note)

    band = float(profile.get("poster_band_blend", 0.0))
    if band > 0.05 or poster > 0.38:
        band_use = max(band, _lerp(0.10, 0.28, poster))
        if progress_callback:
            progress_callback({"phase": "Naturalizar bandas", "progress": 0.92, "detail": f"{band_use:.0%}"})
        out = finisher.soften_poster_bands(out, blend=band_use)
        notes.append(f"naturalizar {band_use:.0%}")

    if progress_callback:
        progress_callback({"phase": "Preservar caras", "progress": 0.95, "detail": "original"})
    out = _preserve_faces(original.convert("RGB"), out)
    notes.append("caras preservadas")

    if poster < 0.48 and mag < 0.72:
        amt = min(1.10, float(profile.get("sharpen", 1.06)))
        out = finisher.sharpen(out, amount=amt)
        notes.append(f"nitidez {amt:.2f}")

    note_str = "acabado TOP edit (" + ", ".join(notes) + ")" if notes else "acabado TOP edit"
    print(f"[TopTier] {note_str}", flush=True)
    return out, note_str, cv_analysis


def run_quality_enhance_after_edit(
    image: Image.Image,
    original: Image.Image,
    *,
    enhance_tier: str = "hd",
    use_generative: bool = True,
    preserve_faces: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """
    Mejora TOP tras un edit con prompt (híbrido): opcional LongCat realismo + Lanczos + caras.
    """
    from roop.img_editor.image_quality_pipeline import get_quality_finisher

    finisher = get_quality_finisher()
    tier = (enhance_tier or "hd").lower()
    analysis = finisher.analyze_image(image, tier=tier)
    profile = analysis.get("profile", {})
    scores = analysis.get("scores", {})
    poster = float(scores.get("poster", 0.0))
    notes = []

    gen_note = " + LongCat realismo" if use_generative else ""
    print(
        f"[TopTier] Mejora post-edit tier={tier.upper()} (Lanczos{gen_note})",
        flush=True,
    )

    out = image.convert("RGB")
    max_side = max(out.size)
    heavy_tier = tier == "8k" or max_side >= 3600

    if not profile.get("skip_depixelize", False) and poster > 0.38:
        dep = min(0.30, float(profile.get("depixelize_blend", 0.22)))
        if heavy_tier:
            dep = min(dep, 0.18)
        if progress_callback:
            progress_callback({"phase": "Desposterizar", "progress": 0.90, "detail": f"{dep:.0%}"})
        out = finisher.depixelize(out, blend_strength=dep, tier=tier)
        notes.append(f"desposterizar {dep:.0%}")

    if cancel_check and cancel_check():
        return image, "cancelado", analysis

    if use_generative:
        merged = {"magnitude": 0.55, "scores": scores, "profile": profile}
        gen_img, gen_note = _generative_polish_pass(
            out, merged, tier, mode="edit",
            progress_callback=progress_callback, cancel_check=cancel_check,
        )
        if gen_img is not None:
            out = gen_img
            notes.append(gen_note)

    if cancel_check and cancel_check():
        return out, "cancelado", analysis

    if preserve_faces:
        out = _preserve_faces(original.convert("RGB"), out)
        notes.append("caras preservadas")

    if cancel_check and cancel_check():
        return out, "cancelado", analysis

    out, up_note = finisher.upscale_lanczos_to_tier(
        out, tier, progress_callback=progress_callback, progress_base=0.91, progress_span=0.05,
    )
    if up_note:
        notes.append(up_note)

    band = float(profile.get("poster_band_blend", 0.0))
    if band > 0.05 or poster > 0.32:
        band_use = max(band, _lerp(0.10, 0.24, poster))
        if heavy_tier:
            band_use = min(band_use, 0.20)
        out = finisher.soften_poster_bands(out, blend=band_use)
        notes.append(f"naturalizar {band_use:.0%}")

    stripe_mix = _lerp(0.14, 0.30, poster) if heavy_tier else _lerp(0.08, 0.18, poster)
    if stripe_mix > 0.10:
        out = finisher.reduce_stripe_artifacts(out, blend=stripe_mix)
        notes.append(f"anti-rayas {stripe_mix:.0%}")

    color_mix = _lerp(0.14, 0.26, max(poster, float(scores.get("restore", 0.0))))
    if color_mix > 0.10:
        out = finisher.enhance_color(out, blend=color_mix)
        notes.append(f"color {color_mix:.0%}")

    if poster < 0.50 and not heavy_tier:
        amt = min(1.10, float(profile.get("sharpen", 1.06)))
        out = finisher.sharpen(out, amount=amt)
        notes.append(f"nitidez {amt:.2f}")

    note_str = "mejora TOP post-edit (" + ", ".join(notes) + ")"
    print(f"[TopTier] {note_str}", flush=True)
    return out, note_str, analysis