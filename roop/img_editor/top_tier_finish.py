#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Acabado TOP para modo Solo mejorar y reutilizable en otros edits.
Híbrido: prep clásico → Kontext (realismo) → Lanczos 8K (sin tiles ONNX / sin rejilla).
"""

from typing import Any, Callable, Dict, Optional, Tuple
import threading

import numpy as np
from PIL import Image


def _edge_mask_from_image(img: Image.Image, edge_thresh: float = 28.0, dilate_px: int = 2) -> Optional[np.ndarray]:
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
    progress_base: Optional[float] = None,
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
    # _preserve_faces + ESRGAN/LSDIR del finisher) ANTES de cargar Kontext en
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

    ok, load_msg = client.load()
    if not ok:
        print(f"[TopTier] Kontext no cargado: {load_msg}")
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
        f"[TopTier] Kontext {label} — denoise={denoise:.2f}, steps={steps}, "
        f"poster={poster:.2f}, mag={mag:.2f}",
        flush=True,
    )
    if progress_callback:
        default_progress = 0.88 if mode == "edit" else 0.22
        progress_callback({
            "phase": "Realismo generativo" if mode == "quality" else "Pulido TOP post-edit",
            "progress": float(default_progress if progress_base is None else progress_base),
            "detail": f"Kontext d={denoise:.2f}",
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
    note = f"Kontext {label} ({blend_note}, d={denoise:.2f})"
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
            mask = _face_restore_mask(crop, blend)
            if mask is None:
                continue
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


def _gate_sharpness(image: Image.Image) -> float:
    """Varianza del Laplaciano a 1280 px de lado (misma escala que analyze_image)."""
    try:
        import cv2 as _cv2
        g = np.array(image.convert("L"), dtype=np.uint8)
        m = max(g.shape[:2])
        if m > 1280:
            s = 1280.0 / m
            g = _cv2.resize(
                g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))),
                interpolation=_cv2.INTER_AREA,
            )
        return float(_cv2.Laplacian(g, _cv2.CV_64F).var())
    except Exception:
        return 0.0


def _gate_clipping(image: Image.Image) -> float:
    """Fracción de píxeles pegados a negro o blanco: recorte de tonos."""
    try:
        import cv2 as _cv2
        g = np.array(image.convert("L"), dtype=np.uint8)
        m = max(g.shape[:2])
        if m > 1280:
            s = 1280.0 / m
            g = _cv2.resize(
                g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))),
                interpolation=_cv2.INTER_AREA,
            )
        return float((g <= 4).mean() + (g >= 251).mean())
    except Exception:
        return 0.0


def _gate_banding(image: Image.Image, max_side: int = 1024) -> float:
    """Escalones de color en zonas planas (banding visible).

    v5.1321 — la métrica de posterización compara bloques y planos, pero no
    distingue bandas: en el fixture con bandas duras quedó 0,837→0,836 y la
    puerta no podía juzgar «desposterizar». Aquí se mide el síntoma real:
    bloques planos con pocos niveles de gris. Alto = escalones anchos.
    """
    import cv2 as _cv2

    try:
        probe = image.convert("L")
        size = probe.size
        scale = min(1.0, float(max_side) / float(max(size) or 1))
        if scale < 1.0:
            probe = probe.resize(
                (max(1, int(size[0] * scale)), max(1, int(size[1] * scale))),
                Image.BOX,
            )
        gray = np.asarray(probe, dtype=np.float32)
        h, w = gray.shape[:2]
        block = 24
        rows, cols = h // block, w // block
        if rows < 1 or cols < 1:
            return 0.0
        local = _cv2.blur(gray, (7, 7))
        flat = _cv2.blur((gray - local) ** 2, (7, 7)) < 3.0
        grid = gray[: rows * block, : cols * block].reshape(rows, block, cols, block)
        flat_grid = flat[: rows * block, : cols * block].reshape(rows, block, cols, block)
        cells = grid.transpose(0, 2, 1, 3).reshape(rows, cols, block * block)
        flat_share = flat_grid.transpose(0, 2, 1, 3).reshape(
            rows, cols, block * block
        ).mean(axis=-1)
        # 192 muestras por bloque: distinguen 2-6 niveles de un plano
        # posterizado y decenas de niveles de ruido de sensor.
        sample = cells[:, :, ::3].astype(np.uint8)
        levels = np.array(
            [[np.unique(cell).size for cell in row] for row in sample],
            dtype=np.float32,
        )
        plateau = np.clip(1.0 - (levels - 2.0) / 10.0, 0.0, 1.0)
        total = float(flat_share.sum())
        if total <= 1e-6:
            return 0.0
        return float((plateau * flat_share).sum() / total)
    except Exception as exc:
        print(f"[TopTier] Bandas no medibles: {exc}", flush=True)
        return 0.0


# Malla facial de 468 puntos (MediaPipe). El óvalo son los índices del contorno
# de cara; la máscara sigue ese contorno en vez de una elipse fija.
_FACE_OVAL_INDICES = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234,
    127, 162, 21, 54, 103, 67, 109, 151, 9, 8,
)
_MESH_STATE: Dict[str, Any] = {"app": None, "failed": False, "lock": threading.Lock()}


def _get_face_mesh():
    """FaceMesh de MediaPipe, una sola vez por proceso (None si no está)."""
    with _MESH_STATE["lock"]:
        if _MESH_STATE["app"] is not None:
            return _MESH_STATE["app"]
        if _MESH_STATE["failed"]:
            return None
        try:
            import mediapipe as mp

            if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
                raise RuntimeError("mediapipe sin solutions.face_mesh")
            _MESH_STATE["app"] = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=False,
                min_detection_confidence=0.4,
            )
            print("[TopTier] Malla facial 468 puntos inicializada", flush=True)
        except Exception as exc:
            _MESH_STATE["failed"] = True
            print(f"[TopTier] Malla facial no disponible ({exc}) — elipse con bordes protegidos", flush=True)
            return None
        return _MESH_STATE["app"]


def _face_mesh_mask(crop_bgr: np.ndarray, blend: float) -> Optional[np.ndarray]:
    """Máscara de la cara según el óvalo de 468 puntos, no una elipse.

    v5.1322 — la elipse difuminaba también fuera de la cara (pelo, orejas,
    fondo) y dejaba un óvalo suave en perfiles. El contorno medido de la malla
    sigue mandíbula, sienes y frente. Se conserva el protector de bordes para
    no cruzar montura, pelo o gafas que caigan dentro del óvalo.
    Devuelve None si no hay malla fiable para esta cara (se usa el fallback).
    """
    import cv2 as _cv2

    app = _get_face_mesh()
    if app is None:
        return None
    fh, fw = crop_bgr.shape[:2]
    if min(fh, fw) < 64:
        return None
    try:
        rgb = np.ascontiguousarray(_cv2.cvtColor(crop_bgr, _cv2.COLOR_BGR2RGB))
        result = app.process(rgb)
        faces = getattr(result, "multi_face_landmarks", None) if result else None
        if not faces:
            return None
        mesh = faces[0].landmark
        points = np.array(
            [
                [mesh[idx].x * (fw - 1), mesh[idx].y * (fh - 1)]
                for idx in _FACE_OVAL_INDICES
                if idx < len(mesh)
            ],
            dtype=np.float32,
        )
        if points.shape[0] < 8 or not np.isfinite(points).all():
            return None
        hull = _cv2.convexHull(points.astype(np.int32))
        if hull.shape[0] < 3:
            return None
        mask = np.zeros((fh, fw), dtype=np.uint8)
        _cv2.fillConvexPoly(mask, hull, 255, lineType=_cv2.LINE_AA)
        coverage = float((mask > 0).mean())
        # Una cara real ocupa una fracción razonable del recorte con margen 30%.
        if coverage < 0.03 or coverage > 0.92:
            return None
        soft = _cv2.GaussianBlur(
            mask.astype(np.float32) / 255.0, (0, 0),
            max(3.0, min(fh, fw) * 0.02),
        )
        gray = _cv2.cvtColor(crop_bgr, _cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = _cv2.Sobel(gray, _cv2.CV_32F, 1, 0, ksize=3)
        gy = _cv2.Sobel(gray, _cv2.CV_32F, 0, 1, ksize=3)
        edge = np.clip((np.sqrt(gx * gx + gy * gy) - 40.0) / 140.0, 0.0, 1.0)
        edge = _cv2.GaussianBlur(edge, (0, 0), 1.6)
        return (soft * (1.0 - 0.70 * edge) * float(blend)).astype(np.float32)
    except Exception as exc:
        print(f"[TopTier] Malla facial no usable ({exc}) — se usa elipse", flush=True)
        return None


def _face_restore_mask(
    crop_bgr: np.ndarray, blend: float, *, use_mesh: bool = True,
) -> Optional[np.ndarray]:
    """Máscara suave de restauración facial.

    v5.1321 — la elipse se atenúa en bordes fuertes (pelo, montura, mandíbula).
    v5.1322 — primero se intenta el óvalo de 468 puntos; la elipse queda como
    fallback para caras pequeñas, perfiles no detectados o sin MediaPipe.
    """
    import cv2 as _cv2

    if use_mesh:
        mesh_mask = _face_mesh_mask(crop_bgr, blend)
        if mesh_mask is not None:
            return mesh_mask

    fh, fw = crop_bgr.shape[:2]
    if fh < 8 or fw < 8:
        return None
    mask = np.zeros((fh, fw), dtype=np.float32)
    _cv2.ellipse(
        mask, (fw // 2, fh // 2),
        (max(8, int(fw * 0.46)), max(8, int(fh * 0.52))), 0, 0, 360, 1.0, -1,
    )
    mask = _cv2.GaussianBlur(mask, (0, 0), max(6.0, min(fh, fw) * 0.08))
    gray = _cv2.cvtColor(crop_bgr, _cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = _cv2.Sobel(gray, _cv2.CV_32F, 1, 0, ksize=3)
    gy = _cv2.Sobel(gray, _cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip((np.sqrt(gx * gx + gy * gy) - 40.0) / 140.0, 0.0, 1.0)
    edge = _cv2.GaussianBlur(edge, (0, 0), 1.6)
    return (mask * (1.0 - 0.85 * edge) * float(blend)).astype(np.float32)


def _gate_chroma_shift(before: Image.Image, after: Image.Image, max_side: int) -> Tuple[float, float]:
    """Mean/p95 Lab a,b distance on paired pixels; L changes remain allowed.

    Float RGB in 0..1 keeps OpenCV Lab in CIE units, rather than its scaled
    uint8 representation. Resize both images equally only to bound CPU cost.
    This color-only step must not change geometry; it cannot be registered by
    stretching a candidate back into place.
    """
    import cv2 as _cv2

    if before.size != after.size or min(before.size) < 1:
        raise ValueError("geometría incompatible")
    width, height = before.size
    scale = min(1.0, max_side / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    chroma = []
    for image in (before, after):
        probe = image.convert("RGB")
        if probe.size != size:
            probe = probe.resize(size, Image.Resampling.BOX)
        rgb = np.asarray(probe, dtype=np.float32) / 255.0
        chroma.append(_cv2.cvtColor(rgb, _cv2.COLOR_RGB2LAB)[..., 1:])
    distance = np.linalg.norm(chroma[1] - chroma[0], axis=-1)
    return float(distance.mean()), float(np.percentile(distance, 95))


class _StepGate:
    """Puerta de aceptación medida (v5.1245).

    Cada paso opcional se compara antes/después con la misma medida; sólo se
    revierte cuando EMPEORA (nitidez, posterización, recorte de tonos o croma). No
    decide por estética ni por listas de prompts.
    """

    def __init__(self, finisher, tier: str, cancel_check: Optional[CancelCheck] = None):
        from roop.img_editor.image_quality_pipeline import quality_settings

        self.finisher = finisher
        self.tier = tier
        self.cancel_check = cancel_check
        self.settings = quality_settings()
        self.enabled = bool(self.settings.get("acceptance_gate", True))
        gate = self.settings.get("gate") or {}
        self.tol_sharp = float(gate.get("sharpness_drop_tolerance", 0.06))
        self.tol_poster = float(gate.get("poster_rise_tolerance", 0.04))
        self.tol_clip = float(gate.get("clip_rise_tolerance", 0.008))
        self.tol_band = float(gate.get("band_rise_tolerance", 0.06))
        self.band_max_side = int(gate.get("band_max_side", 1024) or 1024)
        self.full_max = int(gate.get("full_metrics_max_side", 3600))
        self.chroma_policy = gate

    def snapshot(
        self,
        image: Image.Image,
        *,
        poster: Optional[float] = None,
        include_poster: bool = False,
        include_band: bool = False,
    ) -> Dict[str, float]:
        metrics = {"sharp": _gate_sharpness(image), "clip": _gate_clipping(image)}
        if poster is not None:
            metrics["poster"] = float(poster)
        elif include_poster and max(image.size) <= self.full_max:
            # La comparación es antes/después con la MISMA medida; se acota a 2400 px
            # para que la puerta no cueste más que el propio acabado en 4K.
            probe = image
            if max(image.size) > 2400:
                s = 2400.0 / max(image.size)
                probe = image.resize(
                    (max(1, int(image.size[0] * s)), max(1, int(image.size[1] * s))),
                    Image.LANCZOS,
                )
            metrics["poster"] = _measured_poster(self.finisher, probe, self.tier)
        if include_band:
            metrics["band"] = _gate_banding(image, self.band_max_side)
        return metrics

    def check(
        self,
        name: str,
        before_img: Image.Image,
        after_img: Image.Image,
        before_m: Dict[str, float],
        after_m: Dict[str, float],
        *,
        checks: Tuple[str, ...],
    ) -> Tuple[Image.Image, str]:
        if not self.enabled:
            return after_img, ""
        if "band" in checks and "band" in before_m and "band" in after_m:
            rise = after_m["band"] - before_m["band"]
            if rise > self.tol_band:
                msg = (
                    f"{name} revertido (bandas {before_m['band']:.2f}→"
                    f"{after_m['band']:.2f})"
                )
                print(f"[TopTier] {msg}", flush=True)
                return before_img, msg
        if "poster" in checks and "poster" in before_m and "poster" in after_m:
            if after_m["poster"] > before_m["poster"] + self.tol_poster:
                msg = f"{name} revertido (poster {before_m['poster']:.2f}→{after_m['poster']:.2f})"
                print(f"[TopTier] {msg}", flush=True)
                return before_img, msg
        if "sharp" in checks and before_m.get("sharp", 0.0) > 25.0:
            drop = 1.0 - (after_m.get("sharp", 0.0) / before_m["sharp"])
            if drop > self.tol_sharp:
                msg = f"{name} revertido (nitidez -{drop:.0%})"
                print(f"[TopTier] {msg}", flush=True)
                return before_img, msg
        if "clip" in checks:
            rise = after_m.get("clip", 0.0) - before_m.get("clip", 0.0)
            if rise > self.tol_clip:
                msg = f"{name} revertido (recorte de tonos +{rise:.1%})"
                print(f"[TopTier] {msg}", flush=True)
                return before_img, msg
        if "chroma" in checks:
            try:
                mean_limit = self.chroma_policy["chroma_mean_tolerance"]
                p95_limit = self.chroma_policy["chroma_p95_tolerance"]
                max_side = self.chroma_policy["chroma_max_side"]
                for value in (mean_limit, p95_limit):
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not np.isfinite(value) or value < 0):
                        raise ValueError("tolerancia de croma inválida")
                if type(max_side) is not int or max_side < 1:
                    raise ValueError("presupuesto de croma inválido")
                mean, p95 = _gate_chroma_shift(before_img, after_img, max_side)
                if not np.isfinite(mean) or not np.isfinite(p95) or min(mean, p95) < 0:
                    raise ValueError("medida de croma inválida")
                if mean <= mean_limit and p95 <= p95_limit:
                    return after_img, ""
                reason = f"croma media {mean:.2f}/{mean_limit:.2f}, p95 {p95:.2f}/{p95_limit:.2f}"
            except Exception as error:
                reason = f"croma no verificable: {error}"
            msg = f"{name} revertido ({reason})"
            print(f"[TopTier] {msg}", flush=True)
            return before_img, msg
        return after_img, ""


def _face_embeddings(image: Image.Image) -> Optional[np.ndarray]:
    """Embeddings ArcFace de la foto (None si no hay cara o no hay analizador)."""
    try:
        import cv2 as _cv2
        import roop.face_util as _fu

        app = _fu.get_face_analyser()
        if app is None:
            return None
        bgr = _cv2.cvtColor(np.array(image.convert("RGB")), _cv2.COLOR_RGB2BGR)
        faces = app.get(bgr)
        if not faces or any(getattr(f, "normed_embedding", None) is None for f in faces):
            return None
        embs = [np.asarray(f.normed_embedding, dtype=np.float32) for f in faces]
        return np.stack(embs)
    except Exception as exc:
        print(f"[TopTier] Identidad no medida: {exc}", flush=True)
        return None


def _matched_face_cosines(ref: np.ndarray, cand: np.ndarray) -> Optional[np.ndarray]:
    """Return one cosine per reference, using every candidate only once.

    SciPy is an existing application dependency (also used by roop.utilities).
    Invalid/incomplete evidence must not approve an irreversible face change.
    """
    from scipy.optimize import linear_sum_assignment

    if ref is None or cand is None:
        return None
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    if (
        ref.ndim != 2 or cand.ndim != 2 or ref.shape != cand.shape
        or not ref.size or not np.isfinite(ref).all() or not np.isfinite(cand).all()
    ):
        return None
    ref_norm = np.linalg.norm(ref, axis=1, keepdims=True)
    cand_norm = np.linalg.norm(cand, axis=1, keepdims=True)
    if (
        not np.isfinite(ref_norm).all() or not np.isfinite(cand_norm).all()
        or np.any(ref_norm == 0) or np.any(cand_norm == 0)
    ):
        return None
    sims = (ref / ref_norm) @ (cand / cand_norm).T
    rows, columns = linear_sum_assignment(-sims)
    return sims[rows, columns]


def _faces_keep_identity(
    original: Image.Image,
    before: Image.Image,
    after: Image.Image,
    settings: Dict[str, Any],
) -> Tuple[Image.Image, str]:
    """La restauración no puede empeorar el parecido de ninguna cara medida."""
    cfg = settings.get("identity") or {}
    if not bool(cfg.get("verify_after_face_restore", True)):
        return after, ""
    limit = int(cfg.get("max_side", 2048) or 2048)

    def _bounded(img: Image.Image) -> Image.Image:
        """v5.1321 — no saltar la verificación en fotos grandes: se mide el
        parecido sobre la misma versión reducida (la identidad no cambia)."""
        if limit <= 0 or max(img.size) <= limit:
            return img
        scale = limit / float(max(img.size))
        return img.resize(
            (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale))),
            Image.LANCZOS,
        )

    try:
        ref = _face_embeddings(_bounded(original))
        pre = _face_embeddings(_bounded(before))
        post = _face_embeddings(_bounded(after))
        sim_pre = _matched_face_cosines(ref, pre)
        sim_post = _matched_face_cosines(ref, post)
        max_drop = float(cfg.get("max_drop", 0.02))
        floor = float(cfg.get("min_absolute", 0.0) or 0.0)
        if not np.isfinite(floor) or floor < 0:
            floor = 0.0
        valid = (
            sim_pre is not None and sim_post is not None
            and np.isfinite(max_drop) and max_drop >= 0
        )
    except Exception as exc:
        print(f"[TopTier] Verificación individual de identidad falló: {exc}", flush=True)
        valid = False
    if not valid:
        return before, "caras revertidas (identidad individual no verificable)"
    drops = sim_pre - sim_post
    # El criterio relativo (no empeorar) aprobaba caras que ya no eran las de la
    # foto: 0.45→0.46 pasaba con blend. El mínimo absoluto corta ese caso: si la
    # restauración no llega al suelo, se revierte; si la cara YA estaba por
    # debajo, se entrega (revertir no devuelve el parecido) pero se dice.
    floor_note = ""
    if floor > 0:
        worst = int(np.argmin(sim_post))
        if sim_post[worst] < floor <= sim_pre[worst]:
            msg = (
                f"caras revertidas (cara {worst + 1}, identidad "
                f"{sim_pre[worst]:.2f}→{sim_post[worst]:.2f} < mínimo {floor:.2f})"
            )
            print(f"[TopTier] {msg}", flush=True)
            return before, msg
        if sim_pre.min() < floor:
            floor_note = (
                f"identidad baja {sim_pre.min():.2f}→{sim_post.min():.2f} "
                f"(mínimo {floor:.2f})"
            )
            print(f"[TopTier] {floor_note}", flush=True)
    if np.any(drops > max_drop):
        face = int(np.argmax(drops))
        msg = (
            f"caras revertidas (cara {face + 1}, "
            f"identidad {sim_pre[face]:.2f}→{sim_post[face]:.2f})"
        )
        print(f"[TopTier] {msg}", flush=True)
        return before, msg
    print(
        f"[TopTier] Identidad individual tras restaurar {len(sim_post)} caras: "
        f"mínima {sim_pre.min():.2f}→{sim_post.min():.2f}", flush=True,
    )
    return after, floor_note


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
        if after_edit:
            dep, color, light, sharpen = 0.12, 0.14, 0.12, 1.06
        else:
            # At full resolution the previous strengths often rounded back
            # to the input pixels, so Quality appeared to do nothing. Give a
            # clean photo a visible but restrained tonal/detail finish without
            # increasing bilateral smoothing or inventing texture.
            dep, color, light, sharpen = 0.12, 0.28, 0.17, 1.09
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

    # v5.1321 — el unsharp amplifica el grano: en fotos ruidosas el acabado
    # subía el ruido visible. El tope sale de la medida, no del prompt.
    noise = float(scores.get("noise", 0.0) or 0.0)
    if noise >= 0.42:
        sharpen = min(sharpen, 1.04)
        light = min(light, 0.14)
        color = min(color, 0.18)
    elif noise >= 0.28:
        sharpen = min(sharpen, 1.07)

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


def _final_quality_report(
    before: Image.Image,
    after: Image.Image,
    *,
    band_max_side: int = 1024,
) -> Dict[str, float]:
    """Medida final contra la entrada, a la misma escala de análisis.

    v5.1321 — las puertas comparan cada paso con el anterior; una cadena de
    pasos correctos puede dejar el conjunto peor que la entrada. Esto no
    revierte por sí solo (el color pedido debe poder cambiar L y croma):
    informa y avisa cuando pierde nitidez o gana bandas/recorte.
    """
    before_view = before
    after_view = after
    if after_view.size != before_view.size:
        before_view = before_view.resize(after_view.size, Image.LANCZOS)
    return {
        "sharp_before": _gate_sharpness(before_view),
        "sharp_after": _gate_sharpness(after_view),
        "clip_before": _gate_clipping(before_view),
        "clip_after": _gate_clipping(after_view),
        "band_before": _gate_banding(before_view, band_max_side),
        "band_after": _gate_banding(after_view, band_max_side),
    }


def _append_final_report(
    notes: list,
    analysis: Dict[str, Any],
    before: Image.Image,
    out: Image.Image,
    gate: "_StepGate",
) -> Dict[str, Any]:
    """Añade la medida final al análisis y avisa sólo si el conjunto empeora."""
    final_check = _final_quality_report(
        before, out, band_max_side=gate.band_max_side,
    )
    print(
        f"[TopTier] Final medido — nitidez {final_check['sharp_before']:.1f}→"
        f"{final_check['sharp_after']:.1f}, recorte "
        f"{final_check['clip_before']:.2%}→{final_check['clip_after']:.2%}, "
        f"bandas {final_check['band_before']:.2f}→{final_check['band_after']:.2f}",
        flush=True,
    )
    warnings = []
    if final_check["sharp_before"] > 25.0:
        drop = 1.0 - (final_check["sharp_after"] / final_check["sharp_before"])
        if drop > gate.tol_sharp:
            warnings.append(f"nitidez final -{drop:.0%} vs entrada")
    if final_check["band_after"] - final_check["band_before"] > gate.tol_band:
        warnings.append("bandas finales +")
    if warnings:
        notes.append("aviso final: " + ", ".join(warnings))
    return {**analysis, "final_check": final_check}


def run_top_tier_quality(
    image: Image.Image,
    *,
    enhance_tier: str = "hd",
    use_generative: bool = False,
    preserve_faces: bool = True,
    progress_callback: Optional[ProgressCallback] = None,
    cancel_check: Optional[CancelCheck] = None,
    quality_steps: Optional[Dict[str, Any]] = None,
) -> Tuple[Image.Image, str, Dict[str, Any]]:
    from roop.img_editor.image_quality_pipeline import get_quality_finisher

    finisher = get_quality_finisher()
    tier = (enhance_tier or "hd").lower()
    from roop.img_editor.image_quality_pipeline import limit_quality_tier, quality_settings

    tier, tier_note = limit_quality_tier(image, tier)
    settings = quality_settings()
    steps = normalize_quality_steps(quality_steps)
    if not preserve_faces and not any(steps.values()):
        return image.convert("RGB"), "ninguna opción de calidad marcada", {}
    analysis = finisher.analyze_image(image, tier=tier)
    profile = analysis.get("profile", {})
    scores = analysis.get("scores", {})
    notes = [tier_note] if tier_note else []
    gate = _StepGate(finisher, tier, cancel_check)
    moves = _classical_finish_strengths(scores, profile, after_edit=False)
    print(
        f"[TopTier] Pipeline TOP tier={tier.upper()} — "
        f"acabado clásico | {analysis.get('summary', '')}",
        flush=True,
    )

    out = image.convert("RGB")

    if steps["deposter"]:
        dep = float(moves["depixelize"])
        dep_before = out
        dep_before_m = gate.snapshot(
            out, poster=float(scores.get("poster", 0.0)), include_band=True,
        )
        if progress_callback:
            progress_callback({"phase": "Desposterizar", "progress": 0.08, "detail": f"{dep:.0%}"})
        out = finisher.depixelize(out, blend_strength=dep, tier=tier)
        notes.append(f"desposterizar {dep:.0%}")
        # Segunda pasada si sigue blanda y el poster MEDIDO o las bandas
        # siguen altos. En una foto ya nítida ese segundo bilateral dejó grano.
        if not moves["gentle"]:
            _p_now = _measured_poster(finisher, out, tier)
            _b_now = _gate_banding(out, gate.band_max_side)
            if (
                (_p_now > 0.22 and float(scores.get("sharpness", 0.0)) < 0.30)
                or _b_now > 0.32
            ):
                dep2 = min(0.18, _clip01(_lerp(0.12, 0.22, max(_p_now, _b_now))))
                if progress_callback:
                    progress_callback({"phase": "Desposterizar (2ª pasada)", "progress": 0.14, "detail": f"poster={_p_now:.2f} bandas={_b_now:.2f} → {dep2:.0%}"})
                out = finisher.depixelize(out, blend_strength=dep2, tier=tier)
                notes.append(f"desposterizar x2 {dep2:.0%} (poster={_p_now:.2f}, bandas={_b_now:.2f})")
        out, dep_revert = gate.check(
            "desposterizar", dep_before, out, dep_before_m,
            gate.snapshot(out, include_poster=True, include_band=True),
            checks=("band", "poster"),
        )
        if dep_revert:
            notes.append(dep_revert)
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
            # v5.1129 — el paso generativo a veces re-introduce posterización
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
            cancel_check=cancel_check,
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
            band_before = out
            band_before_m = gate.snapshot(out, poster=_p_final)
            out = finisher.soften_poster_bands(out, blend=band)
            notes.append(f"naturalizar bandas {band:.0%}")
            out, band_revert = gate.check(
                "naturalizar bandas", band_before, out, band_before_m,
                gate.snapshot(out, include_poster=True), checks=("poster",),
            )
            if band_revert:
                notes.append(band_revert)

        if tier == "8k" or max(out.size) >= 3600:
            stripe_mix = _lerp(0.12, 0.26, poster)
            if stripe_mix > 0.10:
                out = finisher.reduce_stripe_artifacts(out, blend=stripe_mix)
                notes.append(f"anti-rayas {stripe_mix:.0%}")

    if steps["color"]:
        color_before = out
        color_before_m = gate.snapshot(out)
        out = finisher.enhance_color(out, blend=float(moves["color"]))
        notes.append(f"color {moves['color']:.0%}")
        use_clahe = (not moves["gentle"]) and max(out.size) <= 2800
        out = finisher.detail_boost(out, blend=float(moves["light"]), use_clahe=use_clahe)
        notes.append(f"luz {moves['light']:.0%}")
        out, color_revert = gate.check(
            "color/luz", color_before, out, color_before_m,
            gate.snapshot(out), checks=("clip", "sharp", "chroma"),
        )
        if color_revert:
            notes.append(color_revert)
    else:
        notes.append("paso color omitido")

    if not steps["realism"]:
        notes.append("paso realismo omitido")
    else:
        banded = poster >= 0.30
        realism_before = out
        realism_before_m = gate.snapshot(out, include_poster=banded)
        out, realism_note = _apply_quality_realism(
            finisher, out, moves, poster, heavy_tier=(tier == "8k" or max(out.size) >= 3600),
        )
        notes.append(realism_note)
        out, realism_revert = gate.check(
            "realismo", realism_before, out, realism_before_m,
            gate.snapshot(out, include_poster=banded),
            checks=("sharp", "poster") if banded else ("sharp",),
        )
        if realism_revert:
            notes.append(realism_revert)

    if cancel_check and cancel_check():
        return out, "cancelado", analysis

    if preserve_faces:
        if progress_callback:
            progress_callback({"phase": "Preservar caras", "progress": 0.95, "detail": "identidad tras el acabado"})
        faces_before = out
        restored = _preserve_faces(image, out)
        out, faces_revert = _faces_keep_identity(image, faces_before, restored, settings)
        notes.append("caras preservadas")
        if faces_revert:
            notes.append(faces_revert)

    analysis = _append_final_report(notes, analysis, image, out, gate)
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
    from roop.img_editor.image_quality_pipeline import quality_settings

    settings = quality_settings()
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
    faces_before = out
    restored = _preserve_faces(original.convert("RGB"), out)
    out, faces_revert = _faces_keep_identity(
        original.convert("RGB"), faces_before, restored, settings,
    )
    notes.append("caras preservadas")
    if faces_revert:
        notes.append(faces_revert)

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
    tamaño, color, luz y detalle. Kontext solo si el llamador lo pide.
    Cada paso se puede apagar con quality_steps.
    """
    from roop.img_editor.image_quality_pipeline import get_quality_finisher

    finisher = get_quality_finisher()
    tier = (enhance_tier or "hd").lower()
    from roop.img_editor.image_quality_pipeline import limit_quality_tier, quality_settings

    tier, tier_note = limit_quality_tier(image, tier)
    settings = quality_settings()
    steps = normalize_quality_steps(quality_steps)
    if not preserve_faces and not any(steps.values()):
        return image.convert("RGB"), "ninguna opción de calidad marcada", {}
    analysis = finisher.analyze_image(image, tier=tier)
    profile = analysis.get("profile", {})
    scores = analysis.get("scores", {})
    poster = float(scores.get("poster", 0.0))
    notes = [tier_note] if tier_note else []
    gate = _StepGate(finisher, tier, cancel_check)
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
        dep_before = out
        dep_before_m = gate.snapshot(out, poster=poster, include_band=True)
        if progress_callback:
            progress_callback({"phase": "Desposterizar", "progress": 0.80, "detail": f"{dep:.0%}"})
        out = finisher.depixelize(out, blend_strength=dep, tier=tier)
        notes.append(f"desposterizar {dep:.0%}")
        out, dep_revert = gate.check(
            "desposterizar", dep_before, out, dep_before_m,
            gate.snapshot(out, include_poster=True, include_band=True),
            checks=("band", "poster"),
        )
        if dep_revert:
            notes.append(dep_revert)
    else:
        notes.append("paso desposterizar omitido")

    if cancel_check and cancel_check():
        return image, "cancelado", analysis

    if use_generative:
        merged = {"magnitude": 0.55, "scores": scores, "profile": profile}
        gen_img, gen_note = _generative_polish_pass(
            out, merged, tier, mode="edit", progress_base=0.82,
            progress_callback=progress_callback, cancel_check=cancel_check,
        )
        if gen_img is not None:
            out = gen_img
            notes.append(gen_note)

    if cancel_check and cancel_check():
        return out, "cancelado", analysis

    if cancel_check and cancel_check():
        return out, "cancelado", analysis

    if steps["upscale"]:
        out, up_note = finisher.upscale_to_tier(
            out, tier, progress_callback=progress_callback,
            progress_base=0.85, progress_span=0.10,
            upscale_plan=profile.get("upscale_plan"),  # v5.1145: era cv_analysis (NameError — analysis es el nombre en esta función)
            cancel_check=cancel_check,
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
        color_before = out
        color_before_m = gate.snapshot(out)
        out = finisher.enhance_color(out, blend=float(moves["color"]))
        notes.append(f"color {moves['color']:.0%}")
        use_clahe = (not moves["gentle"]) and max(out.size) <= 2800
        out = finisher.detail_boost(out, blend=float(moves["light"]), use_clahe=use_clahe)
        notes.append(f"luz {moves['light']:.0%}")
        out, color_revert = gate.check(
            "color/luz", color_before, out, color_before_m,
            gate.snapshot(out), checks=("clip", "sharp", "chroma"),
        )
        if color_revert:
            notes.append(color_revert)
    else:
        notes.append("paso color omitido")

    if not steps["realism"]:
        notes.append("paso realismo omitido")
    else:
        banded = poster >= 0.30
        realism_before = out
        realism_before_m = gate.snapshot(out, include_poster=banded)
        out, realism_note = _apply_quality_realism(
            finisher, out, moves, poster, heavy_tier=heavy_tier,
        )
        notes.append(realism_note)
        out, realism_revert = gate.check(
            "realismo", realism_before, out, realism_before_m,
            gate.snapshot(out, include_poster=banded),
            checks=("sharp", "poster") if banded else ("sharp",),
        )
        if realism_revert:
            notes.append(realism_revert)

    if cancel_check and cancel_check():
        return out, "cancelado", analysis

    if preserve_faces:
        if progress_callback:
            progress_callback({"phase": "Preservar caras", "progress": 0.95, "detail": "identidad tras el acabado"})
        faces_before = out
        restored = _preserve_faces(image, out)
        out, faces_revert = _faces_keep_identity(image, faces_before, restored, settings)
        notes.append("caras preservadas")
        if faces_revert:
            notes.append(faces_revert)

    analysis = _append_final_report(notes, analysis, image, out, gate)
    note_str = "mejora TOP post-edit (" + ", ".join(notes) + ")"
    print(f"[TopTier] {note_str}", flush=True)
    return out, note_str, analysis
