#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acabado TOP para modo Solo mejorar y reutilizable en otros edits.
Híbrido: prep clásico → LongCat (realismo) → Lanczos 8K (sin tiles ONNX / sin rejilla).
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from PIL import Image


def _edge_mask_from_image(img: Image.Image, edge_thresh: float = 28.0, dilate_px: int = 3) -> Optional[np.ndarray]:
    """v5.1134 — mapa de BORDES del sujeto (gradiente Sobel sobre luminancia).
    Devuelve máscara float 0..1 (1 = borde fuerte, con dilatación) o None si cv2 no está."""
    try:
        import cv2 as _cv2
        g = np.array(img.convert("L"), dtype=np.float32)
        gx = _cv2.Sobel(g, _cv2.CV_32F, 1, 0, ksize=3)
        gy = _cv2.Sobel(g, _cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        m = (mag > edge_thresh).astype(np.uint8) * 255
        if dilate_px > 0:
            k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
            m = _cv2.dilate(m, k)
        m = _cv2.GaussianBlur(m, (0, 0), 2.0)
        return (m.astype(np.float32) / 255.0)
    except Exception:
        return None


def _edge_aware_blend(base: Image.Image, gen: Image.Image, blend: float) -> Tuple[Image.Image, float]:
    """v5.1134 — mezcla generativa CONSCIENTE DE BORDES. En los contornos del
    sujeto (pelo/hombros vs fondo) la imagen regenerada discrepa ligeramente de
    la original → 'borde'/halo visible + pérdida de fidelidad. Esta mezcla
    reduce el peso generativo SOLO en las bandas de borde (protege contornos,
    mejora fidelidad) y mantiene el blend pleno en zonas planas (piel, fondos)."""
    em = _edge_mask_from_image(base)
    if em is None:
        return Image.blend(base.convert("RGB"), gen.convert("RGB"), float(np.clip(blend, 0, 1))), float(blend)
    b = np.array(base.convert("RGB"), dtype=np.float32)
    g = np.array(gen.convert("RGB").resize(base.size, Image.LANCZOS), dtype=np.float32)
    alpha = float(np.clip(blend, 0, 1)) * (1.0 - 0.85 * em)[..., None]
    out = (b * (1.0 - alpha) + g * alpha).clip(0, 255).astype(np.uint8)
    eff = float(alpha.mean())
    return Image.fromarray(out), eff


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
        if mag >= 0.65:
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

    # v5.1135 — liberar los caches ONNX del proceso principal (CodeFormer de
    # _preserve_faces + ESRGAN/LSDIR del finisher) ANTES de cargar LongCat en
    # ComfyUI. Estos caches viven entre pasadas del pipeline y compiten por la
    # VRAM durante la carga → 75s+ de 'Cargando modelo' intra-preparación
    # (offload dance). Se recrean bajo demanda en ~1-2s.
    try:
        _cf = getattr(_preserve_faces, "_cf", None)
        if _cf is not None:
            _preserve_faces._cf = None
            del _cf
            print("[TopTier] Sesión CodeFormer liberada (se recarga bajo demanda)", flush=True)
    except Exception:
        pass
    try:
        from roop.img_editor.image_quality_pipeline import get_quality_finisher
        get_quality_finisher().release_gpu_caches()
    except Exception:
        pass

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
        reference_latents_method="offset",
        progress_callback=progress_callback,
    )
    if gen_result is None or gen_result.image is None:
        print(f"[TopTier] Generativo omitido: {err}")
        return None, ""

    gen = gen_result.image.convert("RGB")
    if gen.size != (ow, oh):
        gen = gen.resize((ow, oh), Image.LANCZOS)

    # v5.1134 — mezcla consciente de bordes: en los contornos del sujeto el
    # peso generativo cae (protege el contorno original = fidelidad + sin halo).
    if _edge_mask_from_image(image) is not None:
        out, _eff = _edge_aware_blend(image.convert("RGB"), gen, blend)
        blend_note = f"{blend:.0%}, bordes protegidos"
    else:
        out = Image.blend(image.convert("RGB"), gen.convert("RGB"), blend)
        blend_note = f"{blend:.0%}"
    note = f"LongCat {label} ({blend_note}, d={denoise:.2f})"
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
    # v5.1128 — era un STUB (devolvía `edited` sin tocar): la nota "caras
    # preservadas" mentía. Restauración REAL con CodeFormer (ONNX, rápido,
    # ya usado por _enhance_faces en el manager): restaura identidad facial
    # sobre el resultado del upscale/realismo sin regenerar contenido.
    try:
        import cv2 as _cv2
        import roop.face_util as _fu
        from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
        from roop.utilities import resolve_relative_path

        _cache = getattr(_preserve_faces, "_cf", None)
        if _cache is None:
            model_path = resolve_relative_path("../models/CodeFormer/CodeFormerv0.1.onnx")
            if not model_path or not __import__("os").path.exists(model_path):
                print("[TopTier] CodeFormer no disponible — caras sin restaurar")
                return edited
            cf = Enhance_CodeFormer()
            cf.Initialize({"devicename": "cuda"})
            _preserve_faces._cf = cf
            _cache = cf

        img_np = _cv2.cvtColor(np.array(edited.convert("RGB")), _cv2.COLOR_RGB2BGR)
        faces = _fu.extract_face_images(img_np, target_face_detection=True)
        if not faces:
            return edited

        h, w = img_np.shape[:2]
        result = img_np.copy()
        for face_obj, _crop in faces:
            x1, y1, x2, y2 = [int(v) for v in face_obj.bbox]
            margin = int(max(x2 - x1, y2 - y1) * 0.30)
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
            crop = result[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            enhanced, _sf = _cache.Run(None, None, crop)
            if enhanced.shape[:2] != crop.shape[:2]:
                enhanced = _cv2.resize(enhanced, (crop.shape[1], crop.shape[0]), interpolation=_cv2.INTER_LANCZOS4)
            det = float(getattr(face_obj, "det_score", 0.9) or 0.9)
            blend = float(np.clip(det * 0.18, 0.06, 0.30))
            fh, fw = crop.shape[:2]
            mask = np.zeros((fh, fw), dtype=np.float32)
            _cv2.ellipse(mask, (fw // 2, fh // 2), (max(8, int(fw * 0.46)), max(8, int(fh * 0.52))), 0, 0, 360, 1.0, -1)
            mask = _cv2.GaussianBlur(mask, (0, 0), max(6.0, min(fh, fw) * 0.08)) * blend
            m3 = mask[..., None]
            roi = result[y1:y2, x1:x2].astype(np.float32)
            result[y1:y2, x1:x2] = (roi * (1.0 - m3) + enhanced.astype(np.float32) * m3).astype(np.uint8)
        print(f"[TopTier] Caras restauradas con CodeFormer: {len(faces)} (blend<=30%)")
        return Image.fromarray(_cv2.cvtColor(result, _cv2.COLOR_BGR2RGB))
    except Exception as e:
        print(f"[TopTier] Preservación de caras omitida: {e}")
        return edited


def _measured_poster(finisher, image: Image.Image, tier: str) -> float:
    """v5.1129 — mide la posterización REAL del resultado (score del finisher).
    Verificación medida, no asumida: decide segundas pasadas por el valor
    medido, no por el análisis de la imagen de ENTRADA."""
    try:
        a = finisher.analyze_image(image, tier=tier)
        return float(a.get("scores", {}).get("poster", 0.0))
    except Exception:
        return 0.0


def normalize_quality_steps(steps: Optional[Dict[str, Any]]) -> Dict[str, bool]:
    """Pasos del Modo Calidad. None significa todos activos (API antigua)."""
    enabled = {"deposter": True, "upscale": True, "color": True, "realism": True}
    if not steps:
        return enabled
    for key in enabled:
        if key in steps and steps[key] is not None:
            enabled[key] = bool(steps[key])
    return enabled


def _classical_finish_strengths(
    scores: Dict[str, Any],
    profile: Dict[str, Any],
    *,
    after_edit: bool = False,
    heavy_tier: bool = False,
) -> Dict[str, Any]:
    """Fuerza de cada paso marcado en Modo Calidad.

    Si la foto ya es nítida y llega al tamaño pedido, la fuerza queda corta:
    un bilateral fuerte sobre piel y paredes lisas dejó grano
    (3840×5760, 2026-09-28).
    """
    sharpness = float(scores.get("sharpness", 0.0) or 0.0)
    poster = float(scores.get("poster", 0.0) or 0.0)
    restore = float(scores.get("restore", 0.0) or 0.0)
    max_side = int(scores.get("max_side", 0) or 0)
    target_side = int(scores.get("target_side", 0) or 0)
    sharp = sharpness >= 0.50
    at_size = max_side >= target_side > 0
    gentle = sharp and at_size

    if gentle:
        dep, color, light, sharpen = 0.12, 0.14, 0.12, 1.06
    elif sharp:
        dep, color, light, sharpen = 0.14, 0.16, 0.16, 1.08
    else:
        asked = float(profile.get("depixelize_blend", 0.22) or 0.22)
        dep = min(0.30, max(0.16, asked))
        if after_edit:
            dep = min(dep, 0.22)
        if heavy_tier:
            dep = min(dep, 0.18)
        drive = max(poster, restore)
        color = float(np.clip(_lerp(0.16, 0.22, drive), 0.16, 0.22))
        asked_light = float(profile.get("detail_boost_blend", 0.16) or 0.16)
        light = float(np.clip(max(0.16, asked_light), 0.16, 0.22))
        asked_sharp = float(profile.get("sharpen", 1.08) or 1.08)
        sharpen = float(np.clip(min(1.12, asked_sharp), 1.05, 1.12))

    return {
        "depixelize": dep,
        "color": color,
        "light": light,
        "sharpen": sharpen,
        "gentle": gentle,
    }


def _apply_quality_realism(finisher, image, moves, poster, *, heavy_tier=False):
    """Realismo sobre la misma foto.

    Con poster alto, el unsharp global marca las bandas. Ahí se recupera
    microdetalle local, sin CLAHE y sin afilar los contornos.
    """
    poster = float(poster or 0.0)
    banded = poster >= 0.52 or (bool(heavy_tier) and poster >= 0.45)
    # Una foto ya nítida y al tamaño pedido solo recibe un enfoque corto.
    # El grano fuerte sobre piel lisa ya estropeó una foto de 3840×5760.
    if moves.get("gentle"):
        amt = float(moves["sharpen"])
        return finisher.sharpen(image, amount=amt), f"nitidez {amt:.2f}"
    out = image
    note = ""
    if max(image.size) >= 720:
        textured, tex_note = finisher.budgeted_esrgan_at_size(
            image, image.size, blend=0.34 if banded else 0.40, max_tiles=48,
        )
        if tex_note:
            out = textured
            note = tex_note
    if not note:
        if banded:
            blend = 0.12 if poster >= 0.70 else 0.16
            out = finisher.detail_boost(out, blend=blend, use_clahe=False)
            note = f"realismo textura local {blend:.0%}"
        else:
            amt = float(moves["sharpen"])
            out = finisher.sharpen(out, amount=amt)
            note = f"nitidez {amt:.2f}"
    elif not banded:
        amt = float(moves["sharpen"])
        out = finisher.sharpen(out, amount=min(amt, 1.06))
        note += f" + nitidez {min(amt, 1.06):.2f}"
    return out, note


def run_top_tier_quality(
    image: Image.Image,
    *,
    enhance_tier: str = "hd",
    use_generative: bool = True,
    preserve_faces: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
    quality_steps: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    from roop.img_editor.image_quality_pipeline import get_quality_finisher

    finisher = get_quality_finisher()
    tier = (enhance_tier or "hd").lower()
    steps = normalize_quality_steps(quality_steps)
    if not preserve_faces and not any(steps.values()):
        return image.convert("RGB"), "ninguna opción de calidad marcada", {}
    analysis = finisher.analyze_image(image, tier=tier)
    profile = analysis.get("profile", {})
    scores = analysis.get("scores", {})
    notes = []
    moves = _classical_finish_strengths(scores, profile, after_edit=False)
    print(
        f"[TopTier] Pipeline TOP tier={tier.upper()} — "
        f"acabado clásico | {analysis.get('summary', '')}",
        flush=True,
    )

    out = image.convert("RGB")

    if steps["deposter"]:
        dep = float(moves["depixelize"])
        if progress_callback:
            progress_callback({"phase": "Desposterizar", "progress": 0.08, "detail": f"{dep:.0%}"})
        out = finisher.depixelize(out, blend_strength=dep, tier=tier)
        notes.append(f"desposterizar {dep:.0%}")
        # Segunda pasada solo si la foto está blanda y el poster medido sigue alto.
        # En una foto ya nítida ese segundo bilateral dejó grano.
        if not moves["gentle"]:
            _p_now = _measured_poster(finisher, out, tier)
            if _p_now > 0.22 and float(scores.get("sharpness", 0.0)) < 0.30:
                dep2 = min(0.18, _clip01(_lerp(0.12, 0.22, _p_now)))
                if progress_callback:
                    progress_callback({"phase": "Desposterizar (2ª pasada)", "progress": 0.14, "detail": f"poster={_p_now:.2f} → {dep2:.0%}"})
                out = finisher.depixelize(out, blend_strength=dep2, tier=tier)
                notes.append(f"desposterizar x2 {dep2:.0%} (poster={_p_now:.2f})")
    else:
        notes.append("paso desposterizar omitido")

    if cancel_check and cancel_check():
        return image, "cancelado", analysis

    if use_generative:
        gen_img, gen_note = _generative_realism_pass(
            out, analysis, tier, progress_callback=progress_callback, cancel_check=cancel_check,
        )
        if gen_img is not None:
            out = gen_img
            notes.append(gen_note)
            # v5.1129 — el paso LongCat a veces re-introduce posterización
            # (denoise+blend generan planos de color). Medición del RESULTADO:
            # si el poster medido subió >0.25, segunda pasada generativa corta
            # (denoise bajo, solo realismo) en vez de dejarlo así.
            _p_gen = _measured_poster(finisher, out, tier)
            if _p_gen > 0.25 and not (cancel_check and cancel_check()):
                if progress_callback:
                    progress_callback({"phase": "Realismo 2ª pasada", "progress": 0.45, "detail": f"poster={_p_gen:.2f}"})
                gen_img2, gen_note2 = _generative_realism_pass(
                    out, {**analysis, "scores": {**scores, "poster": _p_gen}},
                    tier, progress_callback=progress_callback, cancel_check=cancel_check,
                )
                if gen_img2 is not None:
                    out = gen_img2
                    notes.append(f"realismo x2 (poster={_p_gen:.2f})")

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
            "phase": "Upscale",
            "progress": 0.62,
            "detail": f"→ {tier.upper()}",
        })
    # v5.1128 — era `upscale_lanczos_to_tier` (Lanczos puro) aunque el plan IA
    # (`profile["upscale_plan"]`, LSDIR/ESRGAN por tiles con feather) ya existe
    # verificado en el finisher. `upscale_to_tier` usa el plan IA con fallback
    # Lanczos interno — detalle REAL en vez de interpolación.
    if steps["upscale"]:
        out, up_note = finisher.upscale_to_tier(
            out, tier, progress_callback=progress_callback,
            progress_base=0.55, progress_span=0.20,
            upscale_plan=profile.get("upscale_plan"),
        )
        if up_note:
            notes.append(up_note)
    else:
        notes.append("paso ampliar omitido")

    poster = float(scores.get("poster", 0.0))
    if steps["deposter"] and not moves["gentle"]:
        band = float(profile.get("poster_band_blend", 0.0))
        _p_final = _measured_poster(finisher, out, tier)
        if _p_final > 0.20:
            band = max(band, _lerp(0.0, 0.38, _p_final))
        if band > 0.05:
            if progress_callback:
                progress_callback({"phase": "Naturalizar bandas", "progress": 0.78, "detail": f"{band:.0%}"})
            out = finisher.soften_poster_bands(out, blend=band)
            notes.append(f"naturalizar bandas {band:.0%}")

        if tier == "8k" or max(out.size) >= 3600:
            stripe_mix = _lerp(0.12, 0.26, poster)
            if stripe_mix > 0.10:
                out = finisher.reduce_stripe_artifacts(out, blend=stripe_mix)
                notes.append(f"anti-rayas {stripe_mix:.0%}")

    if steps["color"]:
        out = finisher.enhance_color(out, blend=float(moves["color"]))
        notes.append(f"color {moves['color']:.0%}")
        use_clahe = (not moves["gentle"]) and max(out.size) <= 2800
        out = finisher.detail_boost(out, blend=float(moves["light"]), use_clahe=use_clahe)
        notes.append(f"luz {moves['light']:.0%}")
    else:
        notes.append("paso color omitido")

    if not steps["realism"]:
        notes.append("paso realismo omitido")
    else:
        out, realism_note = _apply_quality_realism(
            finisher, out, moves, poster, heavy_tier=(tier == "8k" or max(out.size) >= 3600),
        )
        notes.append(realism_note)

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
    quality_steps: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    """
    Mejora tras un edit con Modo Calidad: desposterizar, ampliar si falta
    tamaño, color, luz y detalle. LongCat solo si el llamador lo pide.
    Cada paso se puede apagar con quality_steps.
    """
    from roop.img_editor.image_quality_pipeline import get_quality_finisher

    finisher = get_quality_finisher()
    tier = (enhance_tier or "hd").lower()
    steps = normalize_quality_steps(quality_steps)
    if not preserve_faces and not any(steps.values()):
        return image.convert("RGB"), "ninguna opción de calidad marcada", {}
    analysis = finisher.analyze_image(image, tier=tier)
    profile = analysis.get("profile", {})
    scores = analysis.get("scores", {})
    poster = float(scores.get("poster", 0.0))
    notes = []
    max_side = max(image.size)
    heavy_tier = tier == "8k" or max_side >= 3600
    moves = _classical_finish_strengths(
        scores, profile, after_edit=True, heavy_tier=heavy_tier,
    )
    print(
        f"[TopTier] Mejora post-edit tier={tier.upper()} — acabado clásico",
        flush=True,
    )

    out = image.convert("RGB")
    if steps["deposter"]:
        dep = float(moves["depixelize"])
        if progress_callback:
            progress_callback({"phase": "Desposterizar", "progress": 0.90, "detail": f"{dep:.0%}"})
        out = finisher.depixelize(out, blend_strength=dep, tier=tier)
        notes.append(f"desposterizar {dep:.0%}")
    else:
        notes.append("paso desposterizar omitido")

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

    if steps["upscale"]:
        out, up_note = finisher.upscale_to_tier(
            out, tier, progress_callback=progress_callback,
            progress_base=0.85, progress_span=0.10,
            upscale_plan=profile.get("upscale_plan"),  # v5.1145: era cv_analysis (NameError — analysis es el nombre en esta función)
        )
        if up_note:
            notes.append(up_note)
    elif out.size != original.size and max(out.size) < max(original.size):
        out = out.resize(original.size, Image.Resampling.LANCZOS)
        notes.append(f"tamaño original {original.size[0]}×{original.size[1]}")
    else:
        notes.append("paso ampliar omitido")

    if steps["deposter"] and not moves["gentle"]:
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

    if steps["color"]:
        out = finisher.enhance_color(out, blend=float(moves["color"]))
        notes.append(f"color {moves['color']:.0%}")
        use_clahe = (not moves["gentle"]) and max(out.size) <= 2800
        out = finisher.detail_boost(out, blend=float(moves["light"]), use_clahe=use_clahe)
        notes.append(f"luz {moves['light']:.0%}")
    else:
        notes.append("paso color omitido")

    if not steps["realism"]:
        notes.append("paso realismo omitido")
    else:
        out, realism_note = _apply_quality_realism(
            finisher, out, moves, poster, heavy_tier=heavy_tier,
        )
        notes.append(realism_note)

    note_str = "mejora TOP post-edit (" + ", ".join(notes) + ")"
    print(f"[TopTier] {note_str}", flush=True)
    return out, note_str, analysis
