# -*- coding: utf-8 -*-
"""8K con muchos bloques usa ESRGAN reducido. Una nariz fuera del grupo no es cara."""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Face:
    def __init__(self, bbox, kps):
        self.bbox = bbox
        self.kps = np.asarray(kps, dtype=np.float32)


def main():
    fails = 0

    def check(name, cond, extra=""):
        nonlocal fails
        print(("OK  " if cond else "FAIL") + f" {name}" + (f" — {extra}" if extra else ""))
        if not cond:
            fails += 1

    from roop.face_util import _has_face_geometry
    from roop.img_editor.image_quality_pipeline import ImageQualityFinisher

    good = Face(
        [0, 0, 200, 240],
        [[50, 80], [140, 82], [95, 120], [70, 170], [130, 172]],
    )
    bad = Face(
        [0, 0, 200, 240],
        [[50, 80], [140, 82], [10, 120], [70, 170], [130, 172]],
    )
    check("cara con nariz en el grupo pasa", _has_face_geometry(good, max_tilt=45))
    check("nariz fuera del grupo se rechaza", not _has_face_geometry(bad, max_tilt=45))

    finisher = ImageQualityFinisher()
    ww, hh = finisher._fit_esrgan_work_size(2973, 4263, 48)
    tiles = finisher._estimate_tile_count(ww, hh, "esrganx2")
    check(
        "la foto del registro cabe en 48 bloques",
        tiles <= 48 and ww >= 64 and hh >= 64,
        f"{ww}×{hh} tiles={tiles}",
    )

    src = Image.fromarray(np.full((4152, 200, 3), 90, dtype=np.uint8))
    real_estimate = ImageQualityFinisher._estimate_tile_count
    real_fit = finisher._fit_esrgan_work_size
    real_upscale = finisher.upscale
    real_model = finisher._upscale_model_path

    def fake_estimate(w, h, subtype="esrganx2"):
        if max(w, h) >= 400:
            return 108
        return 12

    def fake_fit(w, h, max_tiles=48):
        return 180, 120

    def fake_upscale(image, scale=2, force=False, plan=None):
        w, h = image.size
        painted = Image.new("RGB", (w, h), (200, 40, 40))
        return painted.resize((w * 2, h * 2), Image.NEAREST), "upscale ESRGAN x2"

    ImageQualityFinisher._estimate_tile_count = staticmethod(fake_estimate)
    finisher._fit_esrgan_work_size = fake_fit
    finisher.upscale = fake_upscale
    finisher._upscale_model_path = lambda subtype: str(Path(__file__))
    try:
        out, note = finisher.upscale_to_tier(
            src, "8k", upscale_plan={"label": "ESRGAN x2", "subtype": "esrganx2"},
        )
    finally:
        ImageQualityFinisher._estimate_tile_count = real_estimate
        finisher._fit_esrgan_work_size = real_fit
        finisher.upscale = real_upscale
        finisher._upscale_model_path = real_model

    long_side = max(out.size)
    red = np.array(out)[:, :, 0].mean()
    check("8K no se queda solo en Lanczos", "ESRGAN textura" in note, note)
    check("la salida llega al lado 8K", long_side >= int(7680 * 0.92), str(out.size))
    check("la textura ESRGAN entra en la mezcla", red > 100, f"rojo={red:.1f}")
    print(f"TOTAL FAIL={fails}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
