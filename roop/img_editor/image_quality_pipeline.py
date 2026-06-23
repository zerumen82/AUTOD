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

    def _choose_scale(self, w: int, h: int, requested: int = 0) -> int:
        if requested in (2, 4):
            return requested
        pixels = w * h
        if pixels >= 1_600_000:
            return 0
        if max(w, h) >= 1400:
            return 2
        return 2

    def upscale(self, image: Image.Image, scale: int = 0) -> Tuple[Image.Image, str]:
        w, h = image.size
        use_scale = self._choose_scale(w, h, scale)
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
    def light_denoise(image: Image.Image) -> Image.Image:
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        den = cv2.fastNlMeansDenoisingColored(bgr, None, 4, 4, 7, 21)
        return Image.fromarray(cv2.cvtColor(den, cv2.COLOR_BGR2RGB))

    def finish(
        self,
        image: Image.Image,
        *,
        upscale: bool = True,
        upscale_scale: int = 0,
        sharpen_image: bool = True,
        denoise: bool = True,
        ultra: bool = False,
    ) -> Tuple[Image.Image, str]:
        notes = []
        out = image

        if denoise:
            out = self.light_denoise(out)
            notes.append("denoise")

        if upscale:
            out, up_note = self.upscale(out, upscale_scale)
            if up_note:
                notes.append(up_note)

        if sharpen_image:
            amt = 1.55 if ultra else 1.35
            out = self.sharpen(out, amount=amt)
            notes.append("sharpen")

        return out, " + ".join(notes) if notes else "sin post-proceso"


_finisher: Optional[ImageQualityFinisher] = None


def get_quality_finisher() -> ImageQualityFinisher:
    global _finisher
    if _finisher is None:
        _finisher = ImageQualityFinisher()
    return _finisher