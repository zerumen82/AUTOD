# -*- coding: utf-8 -*-
"""Modo Calidad v5.1321: bandas medidas, identidad sin saltos y coste.

Comprueba, sin GPU ni ComfyUI:
  1. La métrica de bandas distingue un degradado posterizado de uno limpio.
  2. La máscara de restauración facial pesa menos en un borde fuerte.
  3. La identidad se verifica también en fotos mayores que max_side.
  4. `run_top_tier_quality` no lanza generación por defecto.
  5. El ruido medido topa el afilado del acabado.
  6. `upscale_to_tier` respeta la cancelación antes de cargar ONNX.
  7. El A/B de upscale rechaza textura inventada (estructura, no sólo varianza).
  8. Los pasos del Modo Calidad conservan el resto de opciones al apagar uno.
  9. La puerta revierte un paso que aumenta bandas.
 10. La UI publica la estimación de resolución y está cableada.
"""
from pathlib import Path
import inspect
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roop.img_editor.top_tier_finish as ttf  # noqa: E402
from roop.img_editor.image_quality_pipeline import (  # noqa: E402
    ImageQualityFinisher,
    get_quality_finisher,
)


def _check(fails, name, condition, detail=""):
    mark = "OK  " if condition else "FAIL"
    print(f"{mark} {name}" + (f" — {detail}" if detail else ""), flush=True)
    return fails + (0 if condition else 1)


def _banded(size=512):
    yy, xx = np.mgrid[0:size, 0:size]
    level = ((xx // 8) * 8).astype(np.int32)
    rgb = np.stack([level, level, level], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def _dithered(size=512):
    yy, xx = np.mgrid[0:size, 0:size]
    base = (xx / float(size)) * 255.0
    rng = np.random.default_rng(7)
    noisy = np.clip(base + rng.uniform(-2.5, 2.5, base.shape), 0, 255)
    gray = noisy.astype(np.uint8)
    rgb = np.stack([gray, gray, gray], axis=-1)
    return Image.fromarray(rgb)


def _textured(size=(640, 480)):
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = 90 + 55 * np.sin(xx / 5.0) + 45 * np.sin(yy / 4.0)
    # Grano fino: sin él la referencia Lanczos queda por debajo del mínimo
    # medible (var_ref < 25) y el A/B se omite, como en una foto muy lisa.
    base = base + np.random.default_rng(5).normal(0.0, 12.0, base.shape)
    rgb = np.stack([
        np.clip(base, 0, 255),
        np.clip(base * 0.85 + 20, 0, 255),
        np.clip(base * 0.7 + 30, 0, 255),
    ], axis=-1).astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    fails = 0

    banded = ttf._gate_banding(_banded())
    dithered = ttf._gate_banding(_dithered())
    fails = _check(
        fails, "bandas: posterizado > limpio",
        banded > 0.25 and banded > dithered + 0.15,
        f"posterizado={banded:.3f} limpio={dithered:.3f}",
    )

    crop = np.zeros((200, 200, 3), dtype=np.uint8)
    crop[:, :100] = 200
    crop[:, 100:] = 40
    mask = ttf._face_restore_mask(crop, 1.0)
    center = float(mask[90:110, 30:70].mean())
    edge = float(mask[90:110, 96:104].mean())
    fails = _check(
        fails, "máscara facial protege el borde",
        mask is not None and center > edge + 0.05,
        f"centro={center:.3f} borde={edge:.3f}",
    )

    calls = []
    seen_sizes = []

    def fake_embeddings(image):
        seen_sizes.append(max(image.size))
        return calls.pop(0)

    original = Image.new("RGB", (2400, 1200), (200, 20, 20))
    before = Image.new("RGB", (2400, 1200), (20, 200, 20))
    after = Image.new("RGB", (2400, 1200), (20, 20, 200))
    calls.extend([
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
    ])
    saved = ttf._face_embeddings
    ttf._face_embeddings = fake_embeddings
    try:
        out, note = ttf._faces_keep_identity(
            original, before, after,
            {"identity": {
                "verify_after_face_restore": True,
                "max_side": 512,
                "max_drop": 0.02,
                "min_absolute": 0.0,
            }},
        )
    finally:
        ttf._face_embeddings = saved
    fails = _check(
        fails, "identidad verificada en foto grande",
        out is before and "revertidas" in note and all(s <= 512 for s in seen_sizes),
        f"medidas={seen_sizes} nota={note}",
    )

    default_gen = inspect.signature(ttf.run_top_tier_quality).parameters[
        "use_generative"
    ].default
    fails = _check(
        fails, "calidad pura no pide generación por defecto",
        default_gen is False, str(default_gen),
    )

    noisy_moves = ttf._classical_finish_strengths(
        {"sharpness": 0.20, "poster": 0.45, "restore": 0.6, "noise": 0.60,
         "max_side": 1600, "target_side": 3840},
        {},
    )
    clean_moves = ttf._classical_finish_strengths(
        {"sharpness": 0.20, "poster": 0.45, "restore": 0.6, "noise": 0.05,
         "max_side": 1600, "target_side": 3840},
        {},
    )
    fails = _check(
        fails, "ruido medido topa el afilado",
        noisy_moves["sharpen"] <= 1.04 and noisy_moves["sharpen"] < clean_moves["sharpen"],
        f"ruido={noisy_moves['sharpen']:.2f} limpio={clean_moves['sharpen']:.2f}",
    )

    finisher = ImageQualityFinisher()
    small = Image.new("RGB", (100, 80), (120, 120, 120))
    cancelled, cancel_note = finisher.upscale_to_tier(
        small, "4k", cancel_check=lambda: True,
    )
    fails = _check(
        fails, "upscale cancelado antes de cargar ONNX",
        cancelled is small and "cancelado" in cancel_note,
        cancel_note,
    )

    saved_available = ImageQualityFinisher._upscale_model_available
    saved_upscale = ImageQualityFinisher.upscale

    def fake_available(_subtype):
        return True

    def fake_upscale(self, image, scale=0, force=False, plan=None):
        w, h = image.size
        rng = np.random.default_rng(11)
        arr = rng.integers(0, 255, (h * 2, w * 2, 3), dtype=np.uint8)
        return Image.fromarray(arr), "fake AI"

    ImageQualityFinisher._upscale_model_available = staticmethod(fake_available)
    ImageQualityFinisher.upscale = fake_upscale
    try:
        worth, gain_note = finisher._ai_detail_gain(
            _textured(), {"subtype": "esrganx2", "scale": 2, "mode": "direct"},
        )
    finally:
        ImageQualityFinisher._upscale_model_available = saved_available
        ImageQualityFinisher.upscale = saved_upscale
    fails = _check(
        fails, "A/B rechaza textura inventada",
        worth is False and "estructura" in gain_note,
        gain_note,
    )

    steps = ttf.normalize_quality_steps({"upscale": False})
    fails = _check(
        fails, "apagar un paso conserva los demás",
        steps == {"deposter": True, "upscale": False, "color": True, "realism": True},
        str(steps),
    )

    gate = ttf._StepGate(get_quality_finisher(), "hd")
    before_img = Image.new("RGB", (200, 120), (100, 100, 100))
    after_img = Image.new("RGB", (200, 120), (110, 90, 130))
    kept, gate_note = gate.check(
        "poster", before_img, after_img,
        {"band": 0.10}, {"band": 0.50}, checks=("band",),
    )
    fails = _check(
        fails, "la puerta revierte si suben las bandas",
        kept is before_img and "bandas" in gate_note,
        gate_note,
    )

    ui_text = (ROOT / "ui/tabs/img_editor/ui.py").read_text(encoding="utf-8")
    events_text = (ROOT / "ui/tabs/img_editor/events.py").read_text(encoding="utf-8")
    fails = _check(
        fails, "UI publica y cablea la resolución prevista",
        '"quality_estimate": quality_estimate' in ui_text
        and "fn=on_quality_estimate" in events_text
        and "quality_estimate" in events_text,
        "",
    )

    print(f"TOTAL FAIL={fails}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
