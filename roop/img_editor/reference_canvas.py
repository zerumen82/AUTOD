"""Reversible, isotropic preparation of small measured image-edit crops."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ReferenceCanvas:
    source_size: tuple[int, int]
    content_size: tuple[int, int]
    canvas_size: tuple[int, int]
    content_box: tuple[int, int, int, int]

    def restore(self, image: Image.Image) -> Image.Image:
        if image.size != self.canvas_size:
            raise ValueError(f"Unexpected reference canvas: {image.size}, expected {self.canvas_size}")
        return image.crop(self.content_box).resize(self.source_size, Image.Resampling.LANCZOS)


def prepare_reference_canvas(image: Image.Image, *, max_side: int, multiple: int):
    """Scale content uniformly, then pad instead of stretching each dimension.

    Both small and large crops use the same configured sampling budget. The
    padding repeats border pixels; it does not invent objects or alter masks.
    The mapping must be undone before composing into the source photo.
    """
    if (type(max_side) is not int or type(multiple) is not int
            or multiple < 1 or max_side < multiple or max_side % multiple):
        raise ValueError("Reference canvas needs a positive aligned pixel budget")
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 1 or height < 1:
        raise ValueError("Empty reference image")
    scale = max_side / max(width, height)
    scaled = (max(1, round(width * scale)), max(1, round(height * scale)))
    canvas = tuple(math.ceil(side / multiple) * multiple for side in scaled)
    left, top = (canvas[0] - scaled[0]) // 2, (canvas[1] - scaled[1]) // 2
    right, bottom = canvas[0] - scaled[0] - left, canvas[1] - scaled[1] - top
    content = rgb.resize(scaled, Image.Resampling.LANCZOS)
    padded = np.pad(np.asarray(content), ((top, bottom), (left, right), (0, 0)), mode="edge")
    mapping = ReferenceCanvas(rgb.size, scaled, canvas, (left, top, left + scaled[0], top + scaled[1]))
    return Image.fromarray(padded, "RGB"), mapping
