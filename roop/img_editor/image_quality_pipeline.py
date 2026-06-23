#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-procesado de calidad para Image Editor: upscale ESRGAN + nitidez + denoise ligero."""

from typing import Optional, Tuple
import os
import cv2
import numpy as np
from PIL import Image

import roop.globals
from roop.utilities import resolve_relative_path


class ImageQualityFinisher:
    def __init__(self):
        self._upscaler = None
        self._upscale_subtype = None

    def _get_upscaler(self, scale: int = 2):
        from roop.processors.Frame_Upscale import Frame_Upscale

        subtype = "esrganx2" if scale <= 2 else "esrganx4"
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

    TIER_MAX_SIDE = {"hd": 1920, "4k": 3840, "8k": 7680}

    def _choose_scale(self, w: int, h: int, requested: int = 0, force: bool = False) -> int:
        if requested in (2, 4):
            return requested
        if force:
            return 2
        pixels = w * h
        if pixels >= 1_600_000:
            return 0
        return 2

    def upscale(self, image: Image.Image, scale: int = 0, force: bool = False) -> Tuple[Image.Image, str]:
        w, h = image.size
        use_scale = self._choose_scale(w, h, scale, force=force)
        if use_scale == 0:
            return image, "upscale omitido (imagen ya grande)"

        model_name = "real_esrgan_x2.onnx" if use_scale <= 2 else "real_esrgan_x4.onnx"
        model_path = resolve_relative_path(f"../models/Frame/{model_name}")

        if not os.path.isfile(model_path):
            nw, nh = w * use_scale, h * use_scale
            out = image.resize((nw, nh), Image.LANCZOS)
            print(f"[QualityFinisher] ESRGAN no instalado → Lanczos {use_scale}x ({w}x{h}→{nw}x{nh})")
            return out, f"upscale {use_scale}x Lanczos"

        try:
            up = self._get_upscaler(use_scale)
            bgr = self._pil_to_bgr(image)
            out = up.Run(bgr)
            nw, nh = out.shape[1], out.shape[0]
            print(f"[QualityFinisher] Upscale ESRGAN {use_scale}x: {w}x{h} → {nw}x{nh}")
            return self._bgr_to_pil(out), f"upscale ESRGAN {use_scale}x"
        except Exception as e:
            nw, nh = w * use_scale, h * use_scale
            out = image.resize((nw, nh), Image.LANCZOS)
            print(f"[QualityFinisher] ESRGAN error ({e}) → Lanczos {use_scale}x")
            return out, f"upscale {use_scale}x Lanczos"

    @staticmethod
    def sharpen(image: Image.Image, amount: float = 1.35) -> Image.Image:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        blurred = cv2.GaussianBlur(bgr, (0, 0), 1.0)
        sharp = cv2.addWeighted(bgr, amount, blurred, -(amount - 1.0), 0)
        return Image.fromarray(cv2.cvtColor(np.clip(sharp, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

    @staticmethod
    def light_denoise(image: Image.Image, strength: str = "normal") -> Image.Image:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        h = 6 if strength == "ultra" else 4
        h_color = 6 if strength == "ultra" else 4
        den = cv2.fastNlMeansDenoisingColored(bgr, None, h, h_color, 7, 21)
        return Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))

    @staticmethod
    def depixelize(image: Image.Image, tier: str = "hd") -> Image.Image:
        """Suaviza bloques/pixelación preservando bordes (sin leer el prompt)."""
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        if tier == "8k":
            d, sc, ss = 11, 90, 90
        elif tier == "4k":
            d, sc, ss = 9, 70, 70
        else:
            d, sc, ss = 7, 55, 55
        smooth = cv2.bilateralFilter(bgr, d, sc, ss)
        return Image.fromarray(cv2.cvtColor(smooth, cv2.COLOR_BGR2RGB))

    def upscale_to_tier(self, image: Image.Image, tier: str = "hd") -> Tuple[Image.Image, str]:
        tier = (tier or "hd").lower()
        target_side = self.TIER_MAX_SIDE.get(tier, 1920)
        w, h = image.size
        max_side = max(w, h)

        if max_side >= int(target_side * 0.92):
            return image, f"ya ~{tier.upper()} ({w}×{h})"

        out = image
        notes = []
        max_passes = {"hd": 1, "4k": 2, "8k": 3}.get(tier, 1)

        for i in range(max_passes):
            if max(out.size) >= target_side:
                break
            out, up_note = self.upscale(out, scale=2, force=True)
            notes.append(up_note or f"pass{i + 1}")

        ow, oh = out.size
        ms = max(ow, oh)
        if ms > target_side:
            ratio = target_side / float(ms)
            nw, nh = max(1, int(ow * ratio)), max(1, int(oh * ratio))
            out = out.resize((nw, nh), Image.LANCZOS)
            notes.append(f"cap {tier}")

        return out, f"upscale {tier.upper()} ({' → '.join(notes)})"

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
    ) -> Tuple[Image.Image, str]:
        notes = []
        out = image
        tier = (enhance_tier or "hd").lower()
        ultra = ultra or tier in ("4k", "8k")

        if denoise:
            out = self.light_denoise(out, strength="ultra" if ultra else "normal")
            notes.append("denoise")

        if depixelize_image or ultra:
            out = self.depixelize(out, tier=tier)
            notes.append("desposterizar")

        if upscale:
            if ultra and tier in self.TIER_MAX_SIDE:
                out, up_note = self.upscale_to_tier(out, tier)
            else:
                out, up_note = self.upscale(out, upscale_scale, force=bool(upscale_scale))
            if up_note:
                notes.append(up_note)

        if sharpen_image:
            amt = {"hd": 1.45, "4k": 1.58, "8k": 1.68}.get(tier, 1.55 if ultra else 1.35)
            out = self.sharpen(out, amount=amt)
            notes.append("nitidez")

        return out, " + ".join(notes) if notes else "sin post-proceso"


_finisher: Optional[ImageQualityFinisher] = None


def get_quality_finisher() -> ImageQualityFinisher:
    global _finisher
    if _finisher is None:
        _finisher = ImageQualityFinisher()
    return _finisher