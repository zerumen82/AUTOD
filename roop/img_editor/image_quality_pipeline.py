#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-procesado de calidad para Image Editor: restauración por tiles + upscale ESRGAN + nitidez."""

from typing import Any, Callable, Dict, Optional, Tuple
import os
import cv2
import numpy as np
from PIL import Image

# Tier UI = solo objetivo de resolución máxima; la fuerza de cada paso sale del análisis de la imagen.
TIER_TARGET_SIDE = {"hd": 1920, "4k": 3840, "8k": 7680}
TIER_SCALE = {"hd": 0.78, "4k": 0.90, "8k": 1.0}

# Motores ONNX en models/Frame/ — el pipeline elige según análisis (no hardcode por prompt).
UPSCALE_MODEL_FILES = {
    "esrganx2": "real_esrgan_x2.onnx",
    "esrganx4": "real_esrgan_x4.onnx",
    "lsdirx4": "lsdir_x4.onnx",
}

import roop.globals
from roop.utilities import resolve_relative_path

ProgressCallback = Callable[[Dict[str, Any]], None]


def _emit_quality_progress(
    progress_callback: Optional[ProgressCallback],
    phase: str,
    progress: float,
    detail: str = "",
) -> None:
    line = f"[QualityFinisher] {phase}"
    if detail:
        line += f" — {detail}"
    print(line, flush=True)
    if progress_callback:
        try:
            progress_callback({"phase": phase, "progress": progress, "detail": detail})
        except Exception:
            pass


def _clip01(v: float) -> float:
    return float(np.clip(v, 0.0, 1.0))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clip01(t)


class ImageQualityFinisher:
    def __init__(self):
        self._upscaler = None
        self._upscale_subtype = None

    @staticmethod
    def _estimate_noise_level(gray: np.ndarray) -> float:
        blur = cv2.GaussianBlur(gray, (0, 0), 1.4)
        residual = np.abs(gray.astype(np.float32) - blur)
        return _clip01(float(np.median(residual)) / 18.0)

    @staticmethod
    def _estimate_sharpness(gray: np.ndarray) -> float:
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        var = float(lap.var())
        # ~30=muy blanda, ~400+=ya nítida
        return _clip01((var - 35.0) / 380.0)

    def analyze_image(self, image: Image.Image, tier: str = "hd") -> Dict[str, Any]:
        """
        Analiza la imagen y devuelve scores + perfil adaptativo de fuerza.
        El tier UI solo fija el objetivo de resolución; el resto depende de la foto.
        """
        tier = (tier or "hd").lower()
        if tier not in TIER_TARGET_SIDE:
            tier = "hd"

        bgr = self._pil_to_bgr(image)
        h, w = bgr.shape[:2]
        max_side = max(w, h)
        target_side = TIER_TARGET_SIDE[tier]
        tier_mul = TIER_SCALE.get(tier, 1.0)

        artifact_map, art_metrics = self._compute_artifact_map(bgr, tier=tier)
        art_mean = float(artifact_map.mean())
        art_max = float(artifact_map.max())
        art_p90 = float(art_metrics["score_p90"])
        flat_ratio = float(art_metrics["critical_ratio"])
        block_p90 = float(art_metrics["block_p90"])
        band_p90 = float(art_metrics["band_p90"])
        poster_p90 = float(art_metrics["poster_p90"])
        poster_ratio = float(art_metrics["poster_ratio"])
        flat_tone_mean = float(art_metrics["flat_tone_mean"])

        analysis_side = min(1280, max_side)
        scale = analysis_side / float(max_side)
        small = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        sharpness = self._estimate_sharpness(gray)
        noise = self._estimate_noise_level(gray.astype(np.float32))

        # Posterización IA: bandas de color + zonas planas (independiente de nitidez Laplacian).
        poster_need = _clip01(
            poster_p90 * 0.42 + poster_ratio * 0.33 + flat_tone_mean * 0.25
        )

        raw_artifact = block_p90 * 0.52 + band_p90 * 0.33 + art_p90 * 0.22
        artifact_need = _clip01((raw_artifact - 0.10) / 0.52)
        artifact_need = _clip01(artifact_need * 0.55 + poster_need * 0.45)

        soft_need = _clip01(1.0 - sharpness * 0.94)
        if poster_need > 0.32:
            soft_need = _clip01(max(soft_need, poster_need * 0.72))
        if block_p90 > 0.34 and sharpness < 0.42:
            soft_need = _clip01(soft_need + 0.12)

        if max_side >= int(target_side * 0.94):
            upscale_need = 0.0
        else:
            upscale_need = _clip01(1.0 - (max_side / float(target_side)) ** 0.85)

        at_target = upscale_need < 0.06
        # Nitidez Laplacian alta NO implica foto limpia si hay posterización.
        if sharpness > 0.48 and poster_need < 0.28:
            sharp_damp = _lerp(1.0, 0.25, sharpness)
            artifact_need *= sharp_damp
            if poster_need < 0.18:
                soft_need *= _lerp(1.0, 0.55, sharpness)
        if at_target and sharpness > 0.50 and poster_need < 0.25:
            artifact_need *= _lerp(0.75, 0.45, sharpness)

        restore_need = _clip01(
            poster_need * 0.48 + soft_need * 0.32 + artifact_need * 0.28 + upscale_need * 0.18
        )
        clean_sharp = sharpness > 0.62 and poster_need < 0.22 and noise < 0.22
        # Posterización global → ESRGAN full-frame (sin tiles que dejan rayas verticales).
        global_restore = poster_need > 0.28 or (flat_tone_mean > 0.50 and band_p90 > 0.26)

        work_intensity = _clip01(
            restore_need * 0.55 + artifact_need * 0.25 + upscale_need * 0.30
        )
        if clean_sharp:
            work_intensity = _clip01(upscale_need * 0.55 + soft_need * 0.15)
        force_mul = _lerp(0.78, tier_mul, work_intensity)

        # --- Fuerza adaptativa por eje ---
        tile_threshold = _lerp(0.32, 0.10, artifact_need)
        tile_blend_cap = _lerp(0.22, 0.62, artifact_need) * _lerp(1.0, 0.58, sharpness) * force_mul
        tile_blend_cap = _clip01(tile_blend_cap)

        tile_size = int(_lerp(288, 416, artifact_need + upscale_need * 0.35))
        tile_size = int(tile_size * _lerp(0.94, 1.06, force_mul))

        depixelize_blend = _lerp(0.28, 0.72, max(artifact_need, poster_need * 0.92)) * force_mul
        depixelize_blend = _clip01(depixelize_blend)

        if noise < 0.14:
            denoise_strength = "off"
        elif noise < 0.28 or (max_side > 2600 and noise < 0.42):
            denoise_strength = "light"
        elif max_side > 3400 and noise < 0.50:
            denoise_strength = "light"
        else:
            denoise_strength = "normal"

        texture_need = _clip01(
            poster_need * 0.40 + soft_need * 0.35 + artifact_need * 0.18 + upscale_need * 0.28
        )
        texture_recovery = (texture_need > 0.18 or global_restore) and max_side >= 900
        if global_restore:
            texture_blend = _lerp(0.24, 0.44, poster_need) * force_mul
        else:
            texture_blend = _lerp(0.0, 0.34, texture_need) * _lerp(1.0, 0.55, sharpness) * force_mul
        texture_blend = _clip01(texture_blend)

        detail_boost_blend = _lerp(0.08, 0.38, restore_need) * _lerp(1.0, 0.62, sharpness) * force_mul
        detail_boost_blend = _clip01(detail_boost_blend)

        sharpen_amt = _lerp(1.22, 1.55, restore_need) * _lerp(1.0, 0.82, sharpness)
        sharpen_amt *= _lerp(1.0, 0.88, noise)
        if global_restore and poster_need > 0.45:
            sharpen_amt *= _lerp(0.92, 0.78, poster_need)
        sharpen_amt = float(np.clip(sharpen_amt, 1.05, 1.55))

        sharpen_fine = 0.0
        if restore_need > 0.45 and noise < 0.38:
            sharpen_fine = _lerp(1.04, 1.12, restore_need)

        post_upscale_refine = (
            not global_restore
            and not clean_sharp
            and tier in ("4k", "8k")
            and upscale_need > 0.18
            and sharpness < 0.48
            and block_p90 > 0.38
            and artifact_need > 0.35
        )
        post_refine_blend = _lerp(0.10, 0.20, artifact_need) * force_mul
        post_texture_refine = global_restore and upscale_need > 0.12 and tier in ("4k", "8k")
        post_texture_blend = _clip01(_lerp(0.14, 0.26, poster_need) * force_mul)

        face_enhance_cap = _lerp(0.12, 0.36, restore_need) * _lerp(1.0, 0.72, sharpness)
        if global_restore and poster_need > 0.45:
            face_enhance_cap = min(face_enhance_cap, _lerp(0.22, 0.14, poster_need))

        skip_depixelize = clean_sharp or (
            poster_need < 0.14
            and artifact_need < 0.16
            and sharpness > 0.62
        )
        # Tiles solo para manchas locales JPEG; posterización global usa textura full-frame.
        skip_tiles = global_restore or clean_sharp or (
            poster_need < 0.18
            and artifact_need < 0.22
            and block_p90 < 0.30
            and art_max < 0.42
        )

        if clean_sharp:
            depixelize_blend = min(depixelize_blend, 0.16)
            texture_blend = min(texture_blend, _lerp(0.08, 0.14, upscale_need))
            texture_recovery = upscale_need > 0.12
            detail_boost_blend = min(detail_boost_blend, 0.08)
            post_texture_refine = False

        skip_pre_texture = global_restore and upscale_need > 0.22

        if global_restore and not clean_sharp:
            depixelize_blend = max(depixelize_blend, _lerp(0.50, 0.76, poster_need))
            texture_recovery = True
            if skip_pre_texture:
                texture_blend = 0.0
                depixelize_blend = min(depixelize_blend, _lerp(0.50, 0.60, poster_need))
                post_texture_blend = max(post_texture_blend, _lerp(0.10, 0.14, poster_need))
                # Refino micro ya corre post-upscale; evitar segundo pase fuerte (rejilla/duplicado).
                detail_boost_blend = min(detail_boost_blend, _lerp(0.10, 0.16, restore_need))
            else:
                texture_blend = max(texture_blend, _lerp(0.28, 0.46, poster_need))
                detail_boost_blend = max(detail_boost_blend, _lerp(0.22, 0.40, restore_need))

        if work_intensity < 0.22 and at_target:
            plan_mode = "ligero"
        elif global_restore or restore_need > 0.42:
            plan_mode = "restaurar"
        elif work_intensity < 0.48:
            plan_mode = "medio"
        else:
            plan_mode = "fuerte"

        upscale_plan = self._pick_upscale_plan(
            w, h, tier, scores={
                "poster": poster_need,
                "restore": restore_need,
                "upscale_need": upscale_need,
                "sharpness": sharpness,
            },
            profile={
                "clean_sharp": clean_sharp,
                "global_restore": global_restore,
            },
        )

        profile = {
            "tile_threshold": tile_threshold,
            "tile_blend_cap": tile_blend_cap,
            "tile_size": tile_size,
            "depixelize_blend": depixelize_blend,
            "denoise_strength": denoise_strength,
            "texture_recovery": texture_recovery,
            "texture_blend": texture_blend if texture_recovery else 0.0,
            "detail_boost_blend": detail_boost_blend,
            "sharpen": sharpen_amt,
            "sharpen_fine": sharpen_fine,
            "post_upscale_refine": post_upscale_refine,
            "post_refine_blend": post_refine_blend,
            "face_enhance_cap": face_enhance_cap,
            "skip_depixelize": skip_depixelize,
            "skip_tiles": skip_tiles,
            "plan_mode": plan_mode,
            "force_mul": round(force_mul, 3),
            "clean_sharp": clean_sharp,
            "global_restore": global_restore,
            "post_texture_refine": post_texture_refine,
            "post_texture_blend": post_texture_blend,
            "skip_pre_texture": skip_pre_texture,
            "upscale_plan": upscale_plan,
            "poster_band_blend": _clip01(_lerp(0.12, 0.40, poster_need)) if global_restore else 0.0,
        }

        scores = {
            "artifact": round(artifact_need, 3),
            "poster": round(poster_need, 3),
            "restore": round(restore_need, 3),
            "sharpness": round(sharpness, 3),
            "clean_sharp": clean_sharp,
            "global_restore": global_restore,
            "noise": round(noise, 3),
            "soft_need": round(soft_need, 3),
            "upscale_need": round(upscale_need, 3),
            "work_intensity": round(work_intensity, 3),
            "block": round(block_p90, 3),
            "banding": round(band_p90, 3),
            "art_mean": round(art_mean, 3),
            "art_max": round(art_max, 3),
            "flat_ratio": round(flat_ratio, 3),
            "max_side": max_side,
            "target_side": target_side,
        }

        if clean_sharp:
            route = "upscale-ligero"
        elif global_restore:
            route = "naturalidad-global"
        else:
            route = plan_mode
        up_engine = upscale_plan.get("label", "ESRGAN x2")
        clahe_flag = "off" if global_restore or tier == "8k" else "auto"
        summary = (
            f"plan={plan_mode} route={route} poster={scores['poster']:.2f} "
            f"rest={scores['restore']:.2f} nit={scores['sharpness']:.2f} "
            f"up={scores['upscale_need']:.2f} engine={up_engine} clahe={clahe_flag} | "
            f"tiles={'omit' if skip_tiles else f'≤{tile_blend_cap:.0%}'} "
            f"tex={texture_blend:.0%} dep={depixelize_blend:.0%} "
            f"sharpen={sharpen_amt:.2f}"
        )
        print(f"[QualityFinisher] Análisis adaptativo tier={tier.upper()} {w}×{h}: {summary}", flush=True)

        return {"tier": tier, "scores": scores, "profile": profile, "summary": summary}

    @staticmethod
    def _upscale_model_path(subtype: str) -> str:
        fname = UPSCALE_MODEL_FILES.get(subtype, UPSCALE_MODEL_FILES["esrganx2"])
        return resolve_relative_path(f"../models/Frame/{fname}")

    @staticmethod
    def _upscale_model_available(subtype: str) -> bool:
        return os.path.isfile(ImageQualityFinisher._upscale_model_path(subtype))

    @staticmethod
    def _upscale_label(subtype: str) -> str:
        return {
            "esrganx2": "ESRGAN x2",
            "esrganx4": "ESRGAN x4",
            "lsdirx4": "LSDIR x4",
        }.get(subtype, subtype)

    @staticmethod
    def _estimate_tile_count(w: int, h: int, subtype: str = "lsdirx4") -> int:
        """Estima tiles ONNX (x4 usa 128px; x2 usa tiles más grandes)."""
        scale = 4 if subtype in ("lsdirx4", "esrganx4") else 2
        max_dim = max(w, h)
        if scale >= 4:
            tile_size, pad = 128, 6
        elif max_dim >= 1200:
            tile_size, pad = 384, 12
        elif max_dim >= 640:
            tile_size, pad = 256, 10
        else:
            tile_size, pad = 128, 4
        core = max(32, tile_size - 2 * pad)
        cols = max(1, int(np.ceil(w / float(core))))
        rows = max(1, int(np.ceil(h / float(core))))
        return cols * rows

    def _pick_upscale_plan(
        self,
        w: int,
        h: int,
        tier: str,
        scores: Dict[str, Any],
        profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Elige motor de upscale según foto + tier (sin hardcode de prompt).
        LSDIR x4 ya está en disco y suele verse más natural en retratos posterizados.
        """
        tier = (tier or "hd").lower()
        target_side = TIER_TARGET_SIDE.get(tier, 1920)
        max_side = max(w, h)
        pixels = w * h
        poster = float(scores.get("poster", 0.0))
        restore = float(scores.get("restore", 0.0))
        upscale_need = float(scores.get("upscale_need", 0.0))
        clean_sharp = bool(profile.get("clean_sharp", False))
        global_restore = bool(profile.get("global_restore", False))

        if upscale_need < 0.06:
            return {"subtype": "esrganx2", "scale": 2, "mode": "direct", "label": "sin upscale"}

        photo_like = not clean_sharp and (poster > 0.30 or global_restore or restore > 0.45)
        remaining = target_side / float(max_side) if max_side > 0 else 1.0

        if photo_like and self._upscale_model_available("lsdirx4"):
            # Imagen pequeña: LSDIR 4× directo si caben pocos tiles (sin rejilla).
            if pixels <= 2_600_000 and max_side * 4 <= int(target_side * 1.12):
                est = self._estimate_tile_count(w, h, "lsdirx4")
                if est <= 55:
                    return {
                        "subtype": "lsdirx4",
                        "scale": 4,
                        "mode": "direct",
                        "label": f"LSDIR x4 (~{est} tiles)",
                    }
            # Retratos medianos/grandes: ESRGAN x2 a resolución completa evita 200+ tiles de LSDIR.
            if tier in ("4k", "8k") and pixels <= 9_000_000 and remaining > 1.05:
                if pixels > 3_000_000 and self._upscale_model_available("esrganx2"):
                    est2 = self._estimate_tile_count(w, h, "esrganx2")
                    if est2 <= 90:
                        return {
                            "subtype": "esrganx2",
                            "scale": 2,
                            "mode": "direct",
                            "label": f"ESRGAN x2 (~{est2} tiles)",
                        }
                pre_scale = float(np.clip(remaining / 4.0, 0.32, 0.92))
                for _ in range(8):
                    ww, wh = max(1, int(w * pre_scale)), max(1, int(h * pre_scale))
                    if self._estimate_tile_count(ww, wh, "lsdirx4") <= 55:
                        break
                    pre_scale *= 0.88
                pre_scale = round(float(np.clip(pre_scale, 0.30, 0.92)), 3)
                ww, wh = max(1, int(w * pre_scale)), max(1, int(h * pre_scale))
                est = self._estimate_tile_count(ww, wh, "lsdirx4")
                if est <= 55:
                    return {
                        "subtype": "lsdirx4",
                        "scale": 4,
                        "mode": "scaled",
                        "pre_scale": pre_scale,
                        "label": f"LSDIR x4 ({pre_scale:.0%}→4×, ~{est} tiles)",
                    }
                if self._upscale_model_available("esrganx2"):
                    est2 = self._estimate_tile_count(w, h, "esrganx2")
                    return {
                        "subtype": "esrganx2",
                        "scale": 2,
                        "mode": "direct",
                        "label": f"ESRGAN x2 (~{est2} tiles)",
                    }

        if (
            upscale_need > 0.35
            and pixels <= 1_600_000
            and self._upscale_model_available("esrganx4")
            and max_side * 4 <= int(target_side * 1.15)
        ):
            return {"subtype": "esrganx4", "scale": 4, "mode": "direct", "label": "ESRGAN x4"}

        subtype = "esrganx2" if self._upscale_model_available("esrganx2") else "esrganx4"
        return {
            "subtype": subtype,
            "scale": 2 if subtype == "esrganx2" else 4,
            "mode": "direct",
            "label": self._upscale_label(subtype),
        }

    def _get_upscaler(self, subtype: str = "esrganx2"):
        from roop.processors.Frame_Upscale import Frame_Upscale

        if subtype not in UPSCALE_MODEL_FILES:
            subtype = "esrganx2"
        if not self._upscale_model_available(subtype):
            subtype = "esrganx2" if self._upscale_model_available("esrganx2") else "esrganx4"

        if self._upscaler is not None and self._upscale_subtype == subtype:
            return self._upscaler

        up = Frame_Upscale()
        dev = "cuda" if "CUDAExecutionProvider" in getattr(roop.globals, "execution_providers", []) else "cpu"
        up.Initialize({"devicename": dev, "subtype": subtype})
        self._upscaler = up
        self._upscale_subtype = subtype
        return up

    @staticmethod
    def _pil_to_bgr(img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    TIER_MAX_SIDE = TIER_TARGET_SIDE

    def _choose_scale(self, w: int, h: int, requested: int = 0, force: bool = False) -> int:
        if requested in (2, 4):
            return requested
        if force:
            return 2
        pixels = w * h
        if pixels >= 1_600_000:
            return 0
        return 2

    def _run_upscale_plan(
        self,
        image: Image.Image,
        plan: Optional[Dict[str, Any]] = None,
        *,
        scale: int = 0,
        force: bool = False,
    ) -> Tuple[Image.Image, str]:
        w, h = image.size
        plan = dict(plan or {})
        mode = plan.get("mode", "direct")

        if mode == "scaled" and plan.get("pre_scale"):
            pre = float(plan["pre_scale"])
            ww, wh = max(1, int(w * pre)), max(1, int(h * pre))
            work = image.resize((ww, wh), Image.LANCZOS)
            subtype = plan.get("subtype", "lsdirx4")
            model_path = self._upscale_model_path(subtype)
            if not os.path.isfile(model_path):
                nw, nh = max(1, int(w * pre * plan.get("scale", 4))), max(1, int(h * pre * plan.get("scale", 4)))
                out = image.resize((nw, nh), Image.LANCZOS)
                return out, f"upscale Lanczos ({plan.get('label', subtype)})"
            try:
                up = self._get_upscaler(subtype)
                bgr = self._pil_to_bgr(work)
                out_bgr = up.Run(bgr)
                nw, nh = out_bgr.shape[1], out_bgr.shape[0]
                label = plan.get("label", self._upscale_label(subtype))
                print(f"[QualityFinisher] Upscale {label}: {w}×{h} → {ww}×{wh} → {nw}×{nh}")
                return self._bgr_to_pil(out_bgr), f"upscale {label}"
            except Exception as e:
                print(f"[QualityFinisher] Upscale scaled error ({e}) → Lanczos")
                nw, nh = max(1, int(w * pre * 4)), max(1, int(h * pre * 4))
                return image.resize((nw, nh), Image.LANCZOS), "upscale Lanczos (fallback)"

        use_scale = scale if scale in (2, 4) else int(plan.get("scale", 0) or 0)
        if use_scale not in (2, 4):
            use_scale = self._choose_scale(w, h, scale, force=force)
        if use_scale == 0:
            return image, "upscale omitido (imagen ya grande)"

        subtype = plan.get("subtype") or ("esrganx2" if use_scale <= 2 else "esrganx4")
        model_path = self._upscale_model_path(subtype)
        label = plan.get("label") or self._upscale_label(subtype)

        if not os.path.isfile(model_path):
            nw, nh = w * use_scale, h * use_scale
            out = image.resize((nw, nh), Image.LANCZOS)
            print(f"[QualityFinisher] {label} no instalado → Lanczos {use_scale}x ({w}×{h}→{nw}×{nh})")
            return out, f"upscale {use_scale}x Lanczos"

        try:
            up = self._get_upscaler(subtype)
            bgr = self._pil_to_bgr(image)
            out_bgr = up.Run(bgr)
            nw, nh = out_bgr.shape[1], out_bgr.shape[0]
            print(f"[QualityFinisher] Upscale {label}: {w}×{h} → {nw}×{nh}")
            return self._bgr_to_pil(out_bgr), f"upscale {label}"
        except Exception as e:
            nw, nh = w * use_scale, h * use_scale
            out = image.resize((nw, nh), Image.LANCZOS)
            print(f"[QualityFinisher] {label} error ({e}) → Lanczos {use_scale}x")
            return out, f"upscale {use_scale}x Lanczos"

    def upscale(
        self,
        image: Image.Image,
        scale: int = 0,
        force: bool = False,
        plan: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, str]:
        return self._run_upscale_plan(image, plan, scale=scale, force=force)

    @staticmethod
    def soften_poster_bands(image: Image.Image, blend: float = 0.35) -> Image.Image:
        """Rompe bandas de color posterizadas sin rejilla (mediana suave en croma + grano micro)."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        mix = float(np.clip(blend, 0.0, 0.55))
        if mix <= 0.02:
            return image

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        a_soft = cv2.medianBlur(a_ch, 3)
        b_soft = cv2.medianBlur(b_ch, 3)
        chroma_mix = mix * 0.65
        a_out = cv2.addWeighted(a_ch, 1.0 - chroma_mix, a_soft, chroma_mix, 0)
        b_out = cv2.addWeighted(b_ch, 1.0 - chroma_mix, b_soft, chroma_mix, 0)

        gray = l_ch.astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), 2.2)
        flat = 1.0 - np.clip(np.abs(gray - blur) / 18.0, 0.0, 1.0)
        rng = np.random.default_rng(42)
        grain = rng.normal(0.0, 1.0, gray.shape).astype(np.float32) * flat * (mix * 2.2)
        l_out = np.clip(l_ch.astype(np.float32) + grain, 0, 255).astype(np.uint8)

        out = cv2.cvtColor(cv2.merge([l_out, a_out, b_out]), cv2.COLOR_LAB2BGR)
        out = cv2.addWeighted(bgr, 1.0 - mix * 0.45, out, mix * 0.45, 0)
        return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    @staticmethod
    def sharpen(image: Image.Image, amount: float = 1.35) -> Image.Image:
        """Unsharp mask suave — halos mínimos, sin textura inventada."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(bgr, (0, 0), 1.0)
        sharp = cv2.addWeighted(bgr, amount, blurred, -(amount - 1.0), 0)
        return Image.fromarray(cv2.cvtColor(np.clip(sharp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    @staticmethod
    def light_denoise(image: Image.Image, strength: str = "normal") -> Image.Image:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        if strength == "light":
            h, h_color = 3, 3
        elif strength == "ultra":
            h, h_color = 6, 6
        else:
            h, h_color = 4, 4
        den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h_color, 7, 21)
        return Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))

    @staticmethod
    def reduce_stripe_artifacts(image: Image.Image, blend: float = 0.28) -> Image.Image:
        """Suaviza rayas/bandas verticales (costuras inpaint, croma posterizado) sin rejilla CLAHE."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        mix = float(np.clip(blend, 0.0, 0.45))
        if mix <= 0.02:
            return image

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        # Kernel ancho en horizontal → atenúa franjas verticales en croma.
        a_soft = cv2.GaussianBlur(a_ch, (11, 3), 0)
        b_soft = cv2.GaussianBlur(b_ch, (11, 3), 0)
        chroma_mix = mix * 0.72
        a_out = cv2.addWeighted(a_ch, 1.0 - chroma_mix, a_soft, chroma_mix, 0)
        b_out = cv2.addWeighted(b_ch, 1.0 - chroma_mix, b_soft, chroma_mix, 0)
        l_soft = cv2.GaussianBlur(l_ch, (5, 3), 0)
        l_mix = mix * 0.18
        l_out = cv2.addWeighted(l_ch, 1.0 - l_mix, l_soft, l_mix, 0)

        out = cv2.cvtColor(cv2.merge([l_out, a_out, b_out]), cv2.COLOR_LAB2BGR)
        out = cv2.addWeighted(bgr, 1.0 - mix * 0.35, out, mix * 0.35, 0)
        return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    @staticmethod
    def enhance_color(image: Image.Image, blend: float = 0.22) -> Image.Image:
        """Curva en S + calidez suave — contraste fotográfico realista."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        mix = float(np.clip(blend, 0.0, 0.40))
        if mix <= 0.02:
            return image

        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_f32 = l_ch.astype(np.float32)

        l_norm = l_f32 / 255.0
        s = mix * 5.0
        l_s = l_norm + s * l_norm * (1.0 - l_norm) * (2.0 * l_norm - 1.0)
        l_s = np.clip(l_s * 255.0, 0, 255).astype(np.uint8)

        warm = mix * 0.18
        a_out = np.clip(a_ch.astype(np.float32) + warm * 3.0, 0, 255).astype(np.uint8)
        b_out = np.clip(b_ch.astype(np.float32) + warm * 2.0, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(cv2.cvtColor(cv2.merge([l_s, a_out, b_out]), cv2.COLOR_LAB2BGR), cv2.COLOR_BGR2HSV).astype(np.float32)
        s_ch = hsv[:, :, 1]
        boost = 1.0 + mix * 0.18 * (1.0 - s_ch / 255.0)
        hsv[:, :, 1] = np.clip(s_ch * boost, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        out = cv2.addWeighted(bgr, 1.0 - mix * 0.65, out, mix * 0.65, 0)
        return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    @staticmethod
    def detail_boost(
        image: Image.Image,
        blend: float = 0.35,
        *,
        use_clahe: bool = True,
    ) -> Image.Image:
        """Microdetalle + contraste local. CLAHE solo con pocas celdas (evita rejilla en 4K/8K)."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        mix = float(np.clip(blend, 0.0, 0.45))

        blur = cv2.GaussianBlur(bgr, (0, 0), 1.6)
        high = bgr.astype(np.float32) - blur.astype(np.float32)
        micro = np.clip(bgr.astype(np.float32) + high * 0.35, 0, 255)

        # tileGridSize = nº de divisiones (NO píxeles). Muchas divisiones → cuadrícula visible.
        max_side = max(h, w)
        clahe_ok = use_clahe and mix >= 0.12 and max_side <= 4200
        if clahe_ok:
            divisions = int(max(4, min(16, max_side // 640)))
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.15, tileGridSize=(divisions, divisions))
            l_enh = clahe.apply(l_ch)
            contrast = cv2.cvtColor(
                cv2.merge([l_enh, a_ch, b_ch]), cv2.COLOR_LAB2BGR,
            ).astype(np.float32)
            clahe_w = 0.18 if max_side > 2800 else 0.28
            enhanced = contrast * clahe_w + micro * (1.0 - clahe_w)
        else:
            enhanced = micro

        out = bgr.astype(np.float32) * (1.0 - mix) + enhanced * mix
        return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    def texture_recovery(
        self,
        image: Image.Image,
        tier: str = "hd",
        blend: float = 0.30,
        progress_callback: Optional[ProgressCallback] = None,
        texture_recovery: bool = True,
    ) -> Tuple[Image.Image, str]:
        """ESRGAN a media resolución + mezcla: recupera textura/formas sin miles de tiles."""
        if not texture_recovery or blend <= 0:
            return image, ""

        w, h = image.size
        max_side = max(w, h)
        if max_side < 1200:
            return image, ""

        tier = (tier or "hd").lower()
        tier_target = TIER_TARGET_SIDE.get(tier, 1920)
        if max_side >= int(tier_target * 0.92):
            return image, ""

        model_path = resolve_relative_path("../models/Frame/real_esrgan_x2.onnx")
        if not os.path.isfile(model_path):
            return image, ""

        work_scale = _lerp(0.58, 0.48, blend) if tier == "8k" else _lerp(0.68, 0.55, blend)
        ww, wh = max(1, int(w * work_scale)), max(1, int(h * work_scale))
        small = image.resize((ww, wh), Image.LANCZOS)
        _emit_quality_progress(
            progress_callback, "Recuperación textura", 0.38, f"{ww}×{wh}",
        )
        # Textura = un paso ESRGAN x2 (nunca LSDIR 4×: en 8K dispara 1000+ tiles y bandas).
        tex_subtype = "esrganx2"
        if self._estimate_tile_count(ww, wh, tex_subtype) > 80:
            return image, ""
        try:
            up = self._get_upscaler(tex_subtype)
            bgr = self._pil_to_bgr(small)
            enhanced = up.Run(bgr)
            enhanced_bgr = self._fit_bgr_to_size(enhanced, h, w)
            enhanced_pil = self._bgr_to_pil(enhanced_bgr)
            mix = float(np.clip(blend, 0.12, 0.40))
            out = Image.blend(image, enhanced_pil, mix)
            tex_label = self._upscale_label(tex_subtype)
            return out, f"textura {tex_label} ({int(mix * 100)}%)"
        except Exception as e:
            print(f"[QualityFinisher] Texture recovery omitido: {e}")
            return image, ""

    @staticmethod
    def depixelize(image: Image.Image, blend_strength: float = 0.55, tier: str = "hd") -> Image.Image:
        """Suaviza bloques, pixelación y bandas de color preservando bordes."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        tier = (tier or "hd").lower()
        strength = _clip01(blend_strength)
        if tier == "8k":
            d, sc, ss = 11, 90, 90
            chroma_d, chroma_sc = 9, 80
        elif tier == "4k":
            d, sc, ss = 9, 70, 70
            chroma_d, chroma_sc = 7, 60
        else:
            d, sc, ss = 7, 55, 55
            chroma_d, chroma_sc = 5, 45
        d = max(5, int(d * _lerp(0.75, 1.0, strength)))
        sc = int(sc * _lerp(0.8, 1.0, strength))
        ss = int(ss * _lerp(0.8, 1.0, strength))
        blend = _lerp(0.32, 0.78, strength)

        smooth = cv2.bilateralFilter(bgr, d, sc, ss)
        lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        a_ch = cv2.bilateralFilter(a_ch, chroma_d, chroma_sc, chroma_sc)
        b_ch = cv2.bilateralFilter(b_ch, chroma_d, chroma_sc, chroma_sc)
        anti_band = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        out = cv2.addWeighted(anti_band, blend, bgr, 1.0 - blend, 0)
        return Image.fromarray(cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    def upscale_to_tier(
        self,
        image: Image.Image,
        tier: str = "hd",
        progress_callback: Optional[ProgressCallback] = None,
        progress_base: float = 0.55,
        progress_span: float = 0.28,
        upscale_plan: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, str]:
        tier = (tier or "hd").lower()
        target_side = self.TIER_MAX_SIDE.get(tier, 1920)
        w, h = image.size
        max_side = max(w, h)

        if max_side >= int(target_side * 0.92):
            return image, f"ya ~{tier.upper()} ({w}×{h})"

        plan = dict(upscale_plan or {})
        engine = plan.get("label", "ESRGAN x2")
        out = image
        notes = []

        if plan.get("mode") == "scaled" and plan.get("subtype") == "lsdirx4":
            _emit_quality_progress(
                progress_callback,
                f"Upscale {engine}",
                progress_base + progress_span * 0.5,
                f"{w}×{h} → ~{tier.upper()}",
            )
            out, up_note = self._run_upscale_plan(out, plan)
            if up_note:
                notes.append(up_note)
        else:
            max_passes = {"hd": 1, "4k": 2, "8k": 3}.get(tier, 1)
            for i in range(max_passes):
                ms = max(out.size)
                if ms >= int(target_side * 0.92):
                    break
                remaining = target_side / float(ms)
                if remaining <= 1.02:
                    break
                if remaining < 1.85:
                    plan_label = (plan.get("label") or "").lower()
                    if "sin upscale" not in plan_label and "esrgan" in plan_label:
                        w_cur, h_cur = out.size
                        est = self._estimate_tile_count(w_cur, h_cur, "esrganx2")
                        if est <= 90:
                            _emit_quality_progress(
                                progress_callback,
                                f"Upscale {engine} (detail)",
                                progress_base + progress_span * 0.5,
                                f"{w_cur}×{h_cur} → ~{tier.upper()}",
                            )
                            try:
                                mid, mid_note = self.upscale(out, scale=2, force=True, plan=plan)
                                if mid_note and "lanczos" not in mid_note.lower() and "omitido" not in mid_note.lower():
                                    notes.append(mid_note)
                                    out = mid
                                    ms2 = max(out.size)
                                    if ms2 > target_side:
                                        final_ratio = target_side / float(ms2)
                                        nw = max(1, int(out.size[0] * final_ratio))
                                        nh = max(1, int(out.size[1] * final_ratio))
                                        out = out.resize((nw, nh), Image.LANCZOS)
                                        notes.append(f"cap {tier}")
                                    break
                            except Exception as e:
                                print(f"[QualityFinisher] ESRGAN pre-Lanczos fallback: {e}", flush=True)
                        else:
                            print(f"[QualityFinisher] ESRGAN {est}tiles omitido (too many) → Lanczos", flush=True)
                    nw, nh = max(1, int(out.size[0] * remaining)), max(1, int(out.size[1] * remaining))
                    out = out.resize((nw, nh), Image.LANCZOS)
                    notes.append(f"lanczos {remaining:.2f}x")
                    break
                frac = (i + 0.5) / max(max_passes, 1)
                _emit_quality_progress(
                    progress_callback,
                    f"Upscale {engine} ({i + 1}/{max_passes})",
                    progress_base + progress_span * frac,
                    f"{out.size[0]}×{out.size[1]} → ~{tier.upper()}",
                )
                pass_plan = plan if i == 0 and plan.get("subtype") else None
                out, up_note = self.upscale(out, scale=2, force=True, plan=pass_plan)
                notes.append(up_note or f"pass{i + 1}")

        ow, oh = out.size
        ms = max(ow, oh)
        if ms > target_side:
            ratio = target_side / float(ms)
            nw, nh = max(1, int(ow * ratio)), max(1, int(oh * ratio))
            out = out.resize((nw, nh), Image.LANCZOS)
            notes.append(f"cap {tier}")

        return out, f"upscale {tier.upper()} ({' → '.join(notes)})"

    def esrgan_x2_fullframe_blend(
        self,
        image: Image.Image,
        tier: str = "hd",
        blend: float = 0.35,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Tuple[Image.Image, str]:
        """ESRGAN x2 a resolución completa (un paso), misma resolución — textura real antes de Lanczos."""
        if blend <= 0:
            return image, ""

        w, h = image.size
        max_side = max(w, h)
        if max_side > 2560:
            return image, ""

        tier_l = (tier or "hd").lower()
        est_tiles = self._estimate_tile_count(w, h, "esrganx2")
        if tier_l == "8k" or est_tiles > 12 or (max_side <= 1280 and est_tiles > 8):
            print(
                f"[QualityFinisher] ESRGAN omitido ({est_tiles} tiles, tier={tier_l}) — "
                "evita cuadrícula; usa Lanczos",
                flush=True,
            )
            return image, ""

        model_path = resolve_relative_path("../models/Frame/real_esrgan_x2.onnx")
        if not os.path.isfile(model_path):
            return image, ""

        try:
            upscaler = self._get_upscaler("esrganx2")
            bgr = self._pil_to_bgr(image)
            _emit_quality_progress(
                progress_callback, "ESRGAN x2 full-frame", 0.58, f"{w}×{h}",
            )
            enhanced = self._esrgan_same_size(upscaler, bgr)
            enhanced_pil = self._bgr_to_pil(enhanced)
            mix = float(np.clip(blend, 0.18, 0.48))
            out = Image.blend(image, enhanced_pil, mix)
            print(
                f"[QualityFinisher] ESRGAN x2 full-frame {w}×{h} ({int(mix * 100)}%)",
                flush=True,
            )
            return out, f"ESRGAN x2 full-frame ({int(mix * 100)}%)"
        except Exception as e:
            print(f"[QualityFinisher] ESRGAN full-frame omitido: {e}")
            return image, ""

    def upscale_lanczos_to_tier(
        self,
        image: Image.Image,
        tier: str = "hd",
        progress_callback: Optional[ProgressCallback] = None,
        progress_base: float = 0.55,
        progress_span: float = 0.20,
    ) -> Tuple[Image.Image, str]:
        """Solo Lanczos hasta tier — cero tiles ONNX, cero rejilla de upscaler."""
        tier = (tier or "hd").lower()
        target_side = self.TIER_MAX_SIDE.get(tier, 1920)
        w, h = image.size
        max_side = max(w, h)

        if max_side >= int(target_side * 0.92):
            return image, f"ya ~{tier.upper()} ({w}×{h})"

        remaining = target_side / float(max_side)
        nw = max(1, int(w * remaining))
        nh = max(1, int(h * remaining))
        _emit_quality_progress(
            progress_callback,
            "Upscale Lanczos",
            progress_base + progress_span * 0.5,
            f"{w}×{h} → {nw}×{nh}",
        )
        out = image.resize((nw, nh), Image.LANCZOS)
        print(f"[QualityFinisher] Lanczos {remaining:.2f}x: {w}×{h} → {nw}×{nh}", flush=True)
        return out, f"upscale Lanczos {tier.upper()} ({remaining:.2f}x, sin rejilla)"

    @staticmethod
    def _tile_feather_mask(h: int, w: int, feather: int) -> np.ndarray:
        """Ventana 2D suave (coseno) para evitar costuras verticales/horizontales entre tiles."""
        f = max(8, min(int(feather), h // 2, w // 2))
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f, dtype=np.float32))
        mask_y = np.ones(h, dtype=np.float32)
        mask_x = np.ones(w, dtype=np.float32)
        mask_y[:f] *= ramp
        mask_y[-f:] *= ramp[::-1]
        mask_x[:f] *= ramp
        mask_x[-f:] *= ramp[::-1]
        return mask_y[:, None] * mask_x[None, :]

    def _compute_artifact_map(
        self, bgr: np.ndarray, tier: str = "hd"
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Mapa 0-1 de zonas con bloques/bandas reales + métricas escalares."""
        h, w = bgr.shape[:2]
        analysis_side = 1024 if tier == "8k" else 768 if tier == "4k" else 512
        scale = min(1.0, analysis_side / float(max(h, w)))
        if scale < 1.0:
            aw, ah = max(1, int(w * scale)), max(1, int(h * scale))
            small = cv2.resize(bgr, (aw, ah), interpolation=cv2.INTER_AREA)
        else:
            small = bgr

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        blur = cv2.GaussianBlur(gray, (0, 0), 2.0)
        local_var = cv2.GaussianBlur((gray - blur) ** 2, (0, 0), 5.0)
        flat_tone = 1.0 - np.clip(local_var / 120.0, 0.0, 1.0)

        lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        block_raw = cv2.boxFilter(lap, ddepth=-1, ksize=(8, 8))
        br_mean = float(block_raw.mean()) + 1e-6
        br_p10 = float(np.percentile(block_raw, 10))
        br_p90 = float(np.percentile(block_raw, 90))
        br_spread = (br_p90 - br_p10) / br_mean
        local_block = cv2.boxFilter(block_raw, ddepth=-1, ksize=(16, 16))
        blockiness_map = np.clip(
            (block_raw - local_block * 0.88) / (br_p90 - br_p10 + 1e-6),
            0.0,
            1.0,
        )

        lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
        chroma_grad = np.hypot(
            cv2.Sobel(lab[:, :, 1], cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(lab[:, :, 2], cv2.CV_32F, 1, 0, ksize=3),
        )
        luma_grad = np.abs(cv2.Sobel(lab[:, :, 0], cv2.CV_32F, 1, 0, ksize=3))
        banding = np.clip(chroma_grad / (luma_grad + 8.0) - 0.12, 0.0, 1.0)

        structure_gate = np.clip(blockiness_map * 1.15 + banding * 0.75, 0.0, 1.0)
        flat_artifact = flat_tone * structure_gate
        score_map = np.clip(
            blockiness_map * 0.50 + banding * 0.30 + flat_artifact * 0.20,
            0.0,
            1.0,
        ).astype(np.float32)

        poster_map = np.clip(flat_tone * 0.62 + banding * 0.38, 0.0, 1.0).astype(np.float32)

        metrics = {
            "block_p90": float(np.clip(br_spread / 2.4, 0.0, 1.0)),
            "band_p90": float(np.percentile(banding, 90)),
            "flat_tone_mean": float(flat_tone.mean()),
            "poster_p90": float(np.percentile(poster_map, 90)),
            "poster_ratio": float((poster_map > 0.38).mean()),
            "score_p90": float(np.percentile(score_map, 90)),
            "score_max": float(score_map.max()),
            "critical_ratio": float((score_map > 0.32).mean()),
        }

        if scale < 1.0:
            score_map = cv2.resize(score_map, (w, h), interpolation=cv2.INTER_LINEAR)
        return score_map, metrics

    def _build_artifact_map(self, bgr: np.ndarray, tier: str = "hd") -> np.ndarray:
        score_map, _ = self._compute_artifact_map(bgr, tier=tier)
        return score_map

    @staticmethod
    def _fit_bgr_to_size(bgr: np.ndarray, th: int, tw: int) -> np.ndarray:
        eh, ew = bgr.shape[:2]
        if eh == th and ew == tw:
            return bgr
        return cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)

    def _esrgan_same_size(self, upscaler, tile_bgr: np.ndarray) -> np.ndarray:
        th, tw = tile_bgr.shape[:2]
        if th < 32 or tw < 32:
            return tile_bgr
        try:
            enhanced = upscaler.Run(tile_bgr)
            return self._fit_bgr_to_size(enhanced, th, tw)
        except Exception:
            return tile_bgr

    def tile_restore(
        self,
        image: Image.Image,
        tier: str = "hd",
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        progress_base: float = 0.15,
        progress_span: float = 0.35,
        profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, str]:
        """ESRGAN por tiles solo en zonas con artefactos; misma resolución, sin cambiar composición."""
        bgr = self._pil_to_bgr(image)
        h, w = bgr.shape[:2]
        tier = (tier or "hd").lower()

        profile = profile or {}
        artifact_map, _ = self._compute_artifact_map(bgr, tier=tier)
        threshold = float(profile.get("tile_threshold", 0.24))
        if float(artifact_map.max()) < threshold:
            return image, "tiles omitidos (sin zonas críticas)"

        model_path = resolve_relative_path("../models/Frame/real_esrgan_x2.onnx")
        if not os.path.isfile(model_path):
            return image, "tiles omitidos (ESRGAN no instalado)"

        try:
            upscaler = self._get_upscaler("esrganx2")
        except Exception as e:
            print(f"[QualityFinisher] Tile restore init error: {e}")
            return image, "tiles omitidos (ESRGAN error)"

        tile_size = int(profile.get("tile_size", 352))
        overlap = max(64, tile_size // 3)
        stride = max(48, tile_size - overlap)
        blend_cap = float(profile.get("tile_blend_cap", 0.48))
        blend_cap = min(blend_cap, 0.42)

        acc = np.zeros((h, w, 3), dtype=np.float64)
        weight = np.zeros((h, w), dtype=np.float64)
        tiles_done = 0
        tile_coords = []
        for y0 in range(0, h, stride):
            for x0 in range(0, w, stride):
                y1 = min(y0 + tile_size, h)
                x1 = min(x0 + tile_size, w)
                if y1 - y0 < 48 or x1 - x0 < 48:
                    continue
                if float(artifact_map[y0:y1, x0:x1].mean()) >= threshold:
                    tile_coords.append((y0, y1, x0, x1))
        total_tiles = len(tile_coords)
        if total_tiles == 0:
            return image, "tiles omitidos (sin zonas críticas)"

        _emit_quality_progress(
            progress_callback,
            "Tiles ESRGAN",
            progress_base,
            f"{total_tiles} zona(s) en {w}×{h}",
        )

        for tile_idx, (y0, y1, x0, x1) in enumerate(tile_coords):
            if cancel_check and cancel_check():
                return image, "tiles cancelados"

            th, tw = y1 - y0, x1 - x0
            tile = bgr[y0:y1, x0:x1]
            if tile.shape[0] != th or tile.shape[1] != tw:
                continue
            region_score = float(artifact_map[y0:y1, x0:x1].mean())
            restored = self._esrgan_same_size(upscaler, tile)
            restored = self._fit_bgr_to_size(restored, th, tw)
            mix = min(blend_cap, region_score * 0.85)
            restored = cv2.addWeighted(restored, mix, tile, 1.0 - mix, 0)

            mask = self._tile_feather_mask(th, tw, overlap) * region_score
            try:
                acc[y0:y1, x0:x1] += restored.astype(np.float64) * mask[..., None]
                weight[y0:y1, x0:x1] += mask
            except ValueError:
                print(f"[QualityFinisher] Tile skip shape mismatch {restored.shape} vs {(th, tw)}")
                continue
            tiles_done += 1
            if tile_idx == 0 or (tile_idx + 1) % max(1, total_tiles // 8) == 0 or tile_idx + 1 == total_tiles:
                frac = (tile_idx + 1) / total_tiles
                _emit_quality_progress(
                    progress_callback,
                    "Tiles ESRGAN",
                    progress_base + progress_span * frac,
                    f"{tiles_done}/{total_tiles}",
                )

        if tiles_done == 0:
            return image, "tiles omitidos (sin zonas críticas)"

        weight = np.maximum(weight, 1e-6)
        merged = (acc / weight[..., None]).clip(0, 255).astype(np.uint8)
        global_cap = min(0.45, blend_cap + 0.04)
        global_blend = np.clip(artifact_map * 0.65, 0.0, global_cap)[..., None]
        touched = (weight > 1e-5)[..., None]
        out = bgr.astype(np.float32)
        out = np.where(
            touched,
            merged.astype(np.float32) * global_blend + bgr.astype(np.float32) * (1.0 - global_blend),
            out,
        )
        out = out.clip(0, 255).astype(np.uint8)

        print(f"[QualityFinisher] Tile restore: {tiles_done} zona(s), tier={tier}")
        return self._bgr_to_pil(out), f"restauración tiles ({tiles_done} zonas)"

    def finish(
        self,
        image: Image.Image,
        *,
        upscale: bool = True,
        upscale_scale: int = 0,
        enhance_tier: str = "hd",
        sharpen_image: bool = True,
        denoise: bool = True,
        ultra: bool = False,
        depixelize_image: bool = False,
        tile_restore: bool = False,
        cancel_check: Optional[Callable[[], bool]] = None,
        progress_callback: Optional[ProgressCallback] = None,
        adaptive_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Image.Image, str, Optional[Dict[str, Any]]]:
        notes = []
        out = image
        tier = (enhance_tier or "hd").lower()
        ultra = ultra or tier in ("4k", "8k")
        w, h = out.size

        analysis = adaptive_profile or self.analyze_image(out, tier=tier)
        profile = analysis.get("profile", {})
        scores = analysis.get("scores", {})

        _emit_quality_progress(
            progress_callback,
            "Mejorando imagen",
            0.06,
            analysis.get("summary", f"tier={tier.upper()} {w}×{h}"),
        )

        if cancel_check and cancel_check():
            return image, "cancelado", analysis

        if (depixelize_image or ultra) and not profile.get("skip_depixelize", False):
            dep_blend = float(profile.get("depixelize_blend", 0.5))
            _emit_quality_progress(progress_callback, "Desposterizar", 0.10, f"{dep_blend:.0%}")
            out = self.depixelize(out, blend_strength=dep_blend, tier=tier)
            notes.append(f"desposterizar {dep_blend:.0%}")
        elif depixelize_image or ultra:
            notes.append("desposterizar omitido (imagen limpia)")

        if (tile_restore or (ultra and depixelize_image)) and not profile.get("skip_tiles", False):
            try:
                out, tile_note = self.tile_restore(
                    out,
                    tier=tier,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                    profile=profile,
                )
                if tile_note:
                    notes.append(tile_note)
            except Exception as e:
                print(f"[QualityFinisher] Tile restore omitido: {e}")
                notes.append("tiles error (continúa pipeline)")
        elif tile_restore or ultra:
            if profile.get("global_restore"):
                notes.append("tiles omitidos (restauración global sin costuras)")
            else:
                notes.append("tiles omitidos (pocos artefactos)")

        if cancel_check and cancel_check():
            return out, " + ".join(notes) if notes else "cancelado", analysis

        pre_texture = bool(profile.get("texture_recovery", False)) and not profile.get(
            "skip_pre_texture", False
        )
        out_tex, tex_note = self.texture_recovery(
            out,
            tier=tier,
            blend=float(profile.get("texture_blend", 0.0)),
            progress_callback=progress_callback,
            texture_recovery=pre_texture,
        )
        if tex_note:
            out = out_tex
            notes.append(tex_note)
        elif profile.get("skip_pre_texture"):
            notes.append("textura pre-upscale omitida (1 paso ESRGAN menos)")

        denoise_mode = profile.get("denoise_strength", "off")
        if denoise and denoise_mode != "off":
            _emit_quality_progress(progress_callback, "Denoise", 0.48, denoise_mode)
            out = self.light_denoise(out, strength=denoise_mode)
            notes.append(f"denoise {denoise_mode}")
        elif denoise:
            notes.append("denoise omitido (bajo ruido o preservar detalle)")

        if upscale:
            if ultra and tier in self.TIER_MAX_SIDE:
                out, up_note = self.upscale_to_tier(
                    out,
                    tier,
                    progress_callback=progress_callback,
                    upscale_plan=profile.get("upscale_plan"),
                )
            else:
                _emit_quality_progress(progress_callback, "Upscale", 0.60)
                out, up_note = self.upscale(out, upscale_scale, force=bool(upscale_scale))
            if up_note:
                notes.append(up_note)

        band_blend = float(profile.get("poster_band_blend", 0.0))
        if band_blend > 0.05:
            _emit_quality_progress(
                progress_callback, "Naturalizar bandas", 0.72, f"{band_blend:.0%}",
            )
            out = self.soften_poster_bands(out, blend=band_blend)
            notes.append(f"naturalizar bandas {band_blend:.0%}")

        if profile.get("post_upscale_refine") and (tile_restore or depixelize_image) and not profile.get("skip_tiles", False):
            try:
                ow, oh = out.size
                ms = max(ow, oh)
                work = out
                work_scale = 1.0
                if ms > 3200:
                    work_scale = 2800.0 / float(ms)
                    work = out.resize(
                        (max(1, int(ow * work_scale)), max(1, int(oh * work_scale))),
                        Image.LANCZOS,
                    )
                _emit_quality_progress(
                    progress_callback,
                    "Refino post-upscale",
                    0.78,
                    f"{work.size[0]}×{work.size[1]}",
                )
                out_ref, ref_note = self.tile_restore(
                    work,
                    tier=tier,
                    cancel_check=cancel_check,
                    progress_callback=progress_callback,
                    progress_base=0.78,
                    progress_span=0.10,
                    profile=profile,
                )
                if ref_note and "omitidos" not in ref_note:
                    if work_scale < 1.0:
                        out_ref = out_ref.resize((ow, oh), Image.LANCZOS)
                        blend_ref = float(profile.get("post_refine_blend", 0.22))
                        out = Image.blend(out, out_ref, blend_ref)
                    else:
                        out = out_ref
                    notes.append("refino " + ref_note)
            except Exception as e:
                print(f"[QualityFinisher] Refino post-upscale omitido: {e}")

        if profile.get("post_texture_refine"):
            try:
                if profile.get("skip_pre_texture"):
                    blend_c = min(0.12, float(profile.get("post_texture_blend", 0.18)) * 0.45)
                    _emit_quality_progress(
                        progress_callback,
                        "Refino post-upscale",
                        0.82,
                        f"clásico {blend_c:.0%}",
                    )
                    out = self.detail_boost(
                        out,
                        blend=blend_c,
                        use_clahe=not profile.get("global_restore", False),
                    )
                    notes.append(f"refino micro post-upscale ({blend_c:.0%}, sin CLAHE)")
                else:
                    _emit_quality_progress(
                        progress_callback,
                        "Textura post-upscale",
                        0.82,
                        f"{int(profile.get('post_texture_blend', 0.18) * 100)}%",
                    )
                    out_pt, pt_note = self.texture_recovery(
                        out,
                        tier=tier,
                        blend=float(profile.get("post_texture_blend", 0.18)),
                        progress_callback=progress_callback,
                        texture_recovery=True,
                    )
                    if pt_note:
                        out = out_pt
                        notes.append("textura post-upscale")
            except Exception as e:
                print(f"[QualityFinisher] Refino post-upscale omitido: {e}")

        boost_blend = float(profile.get("detail_boost_blend", 0.0))
        if boost_blend > 0:
            _emit_quality_progress(progress_callback, "Detalle/contraste", 0.86, tier.upper())
            out = self.detail_boost(
                out,
                blend=boost_blend,
                use_clahe=not profile.get("global_restore", False),
            )
            notes.append(
                "detalle+contraste (sin CLAHE)"
                if profile.get("global_restore")
                else "detalle+contraste"
            )

        if sharpen_image:
            _emit_quality_progress(progress_callback, "Nitidez", 0.90)
            amt = float(profile.get("sharpen", 1.35))
            out = self.sharpen(out, amount=amt)
            fine = float(profile.get("sharpen_fine", 0.0))
            if fine > 1.0:
                out = self.sharpen(out, amount=fine)
            notes.append("nitidez")

        _emit_quality_progress(progress_callback, "Post-proceso listo", 0.92, " + ".join(notes) if notes else "")
        note_str = " + ".join(notes) if notes else "sin post-proceso"
        return out, note_str, analysis


_finisher: Optional[ImageQualityFinisher] = None


def get_quality_finisher() -> ImageQualityFinisher:
    global _finisher
    if _finisher is None:
        _finisher = ImageQualityFinisher()
    return _finisher


def analyze_quality_plan(image: Image.Image, tier: str = "hd") -> Dict[str, Any]:
    """API pública para UI/preview: plan adaptativo sin ejecutar el pipeline."""
    return get_quality_finisher().analyze_image(image, tier=tier)