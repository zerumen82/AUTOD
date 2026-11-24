"""Auditoría etapa por etapa del pipeline de «Mejorar fotos».

El probe de realismo (probe_enhance_realism_20261123.py) comparó recetas, pero
todas a 1x. Los lotes reales del usuario se exportaron a 2x (`up2x`), así que
aquí se mide:

1. Qué etapa se lleva el poro, aplicándolas UNA a UNA con la fuerza de los logs
   (denoise 65, deband 34, sharpen 40, caras GFPGAN 80, upscale 2x).
2. Si el upscale Real-ESRGAN x2 suaviza la piel que ya se había arreglado
   (la textura reinyectada antes de escalar puede perderse).
3. Dos órdenes completos con la misma receta (base + Ultra realista, 2x):
   A) el actual  denoise → deband → sharpen → caras → upscale
   B) el propuesto denoise → deband → upscale → caras → sharpen
   midiendo a 2x contra el original ampliado y a 1x tras reducir de vuelta.

Métricas:
- corr_poro: correlación de la banda de microdetalle (σ=1.5) con la referencia.
  1.0 = conserva el poro real; bajo = lo borró o inventó otro.
- banda: energía de esa banda respecto a la referencia (1.0 = igual).
- croma: desviación típica del croma en la mejilla respecto al original
  (<1 significa croma aplanado, que es la mitad «plástica» del muñeco).
- clip: % de píxeles quemados o negros.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.tabs.enhance import presets  # noqa: E402
from ui.tabs.enhance.logic import (  # noqa: E402
    apply_deband_cv2,
    apply_denoise_cv2,
    apply_grain_cv2,
    apply_lighting_color_cv2,
    apply_recover_faces_onnx,
    apply_sharpen_cv2,
    apply_upscale_onnx,
    apply_vibrance_cv2,
    load_enhance_config,
    process_single_image,
)

SOURCE = ROOT / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png"
OUT = ROOT / "diagnostics" / "out" / "enhance_audit_20261123"
SKIN_ROI = (215, 135, 300, 205)  # mejilla derecha del retrato 512x512
FACE_ROI = (150, 60, 370, 280)


def _scale_roi(roi, factor):
    return tuple(int(v * factor) for v in roi)


def _band(image: Image.Image, roi) -> np.ndarray:
    x1, y1, x2, y2 = roi
    g = np.array(image.convert("L"), dtype=np.float32)[y1:y2, x1:x2]
    return g - cv2.GaussianBlur(g, (0, 0), 1.5)


def detail_correlation(image: Image.Image, reference: Image.Image, roi=SKIN_ROI) -> float:
    a = _band(image, roi).ravel()
    b = _band(reference, roi).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def band_ratio(image: Image.Image, reference: Image.Image, roi=SKIN_ROI) -> float:
    a = _band(image, roi)
    b = _band(reference, roi)
    ref = float(np.sqrt((b * b).mean()))
    return float(np.sqrt((a * a).mean()) / ref) if ref > 0 else 0.0


def chroma_ratio(image: Image.Image, reference: Image.Image, roi=SKIN_ROI) -> float:
    def chroma(img):
        x1, y1, x2, y2 = roi
        lab = cv2.cvtColor(np.array(img.convert("RGB"))[y1:y2, x1:x2], cv2.COLOR_RGB2LAB)
        return lab[:, :, 1:].astype(np.float32).std()

    ref = chroma(reference)
    return float(chroma(image) / ref) if ref > 0 else 0.0


def clip_ratio(image: Image.Image) -> float:
    g = np.array(image.convert("L"), dtype=np.uint8)
    return float((g <= 4).mean() + (g >= 251).mean())


def same_scale(image: Image.Image, size) -> Image.Image:
    if image.size == size:
        return image
    return image.resize(size, Image.LANCZOS)


def report(label: str, image: Image.Image, reference: Image.Image, roi=SKIN_ROI) -> None:
    print(
        f"  {label:34s} corr={detail_correlation(image, reference, roi):5.2f} "
        f"banda={band_ratio(image, reference, roi):5.2f} "
        f"croma={chroma_ratio(image, reference, roi):5.2f} "
        f"clip={clip_ratio(image):5.2%}"
    )


def main() -> int:
    with Image.open(SOURCE) as opened:
        original = opened.convert("RGB")
    assert original.size == (512, 512), f"muestra inesperada: {original.size}"
    OUT.mkdir(parents=True, exist_ok=True)

    config = load_enhance_config()
    defaults = dict(config["default_slider_values"])
    ultra = dict(presets.load_presets()["Ultra realista"]["sliders"])

    print("== Etapas por separado (fuerza de los logs del 17/11) ==")
    stages = [
        ("denoise 65", lambda im: apply_denoise_cv2(im, 65)),
        ("deband 34", lambda im: apply_deband_cv2(im, 34, 7)),
        ("sharpen 40", lambda im: apply_sharpen_cv2(im, 40)),
        ("caras GFPGAN 80%", lambda im: apply_recover_faces_onnx(im, 0.8, "GFPGAN")),
        ("upscale 2x (Real-ESRGAN)", lambda im: apply_upscale_onnx(im, 2)),
    ]
    stage_images = []
    for label, fn in stages:
        out = same_scale(fn(original), original.size)
        stage_images.append((label, out))
        report(label, out, original)

    # La misma receta completa con el orden actual: sirve de referencia de lote.
    stack = dict(defaults)
    for key in ("denoise_strength", "sharpen_strength", "deband_strength", "grain_strength"):
        stack[key] = float(defaults[key]) + float(ultra[key])
    stack["recover_faces_strength"] = min(1.0, float(defaults["recover_faces_strength"]) + float(ultra["recover_faces_strength"]))
    stack["upscale_factor"] = 2
    stack["grain_size"] = 2
    print(
        f"== Receta completa a 2x: denoise {stack['denoise_strength']:g} · "
        f"deband {stack['deband_strength']:g} · sharpen {stack['sharpen_strength']:g} · "
        f"caras {stack['recover_faces_strength']:.0%} · upscale 2x =="
    )

    tmp = OUT / "source.png"
    original.save(tmp)
    new_pipeline, notes_new, _ = process_single_image(tmp, dict(stack))
    print(f"  [B · pipeline nuevo] {notes_new}")

    # A) Reproducción del comportamiento anterior, mismo orden y receta:
    #    sharpen completo antes de escalar, GFPGAN sin conservar tono y
    #    Real-ESRGAN sin devolverle el grano del original.
    legacy = apply_denoise_cv2(original, stack["denoise_strength"])
    legacy = apply_deband_cv2(legacy, stack["deband_strength"], 7)
    legacy = apply_sharpen_cv2(legacy, stack["sharpen_strength"])
    legacy = apply_recover_faces_onnx(
        legacy, stack["recover_faces_strength"], "GFPGAN", tone_match=False
    )
    legacy = apply_upscale_onnx(legacy, 2, resample_mix=0.0)
    legacy = apply_lighting_color_cv2(
        legacy, stack["exposure"], stack["shadow"], stack["highlight"], stack["temperature"]
    )
    legacy = apply_vibrance_cv2(legacy, stack["vibrance"])
    legacy = apply_grain_cv2(legacy, stack["grain_strength"], 2, seed=11)
    print("  [A · pipeline anterior] sharpen completo + GFPGAN sin tono + ESRGAN sin grano")

    # Comparación a 2x (lo que se exporta) contra el original ampliado.
    up2 = original.resize((1024, 1024), Image.BICUBIC)
    roi2 = _scale_roi(SKIN_ROI, 2)
    print("== A 2x nativo (contra el original x2 bicúbico) ==")
    for label, image in (("A · anterior", legacy), ("B · nuevo", new_pipeline)):
        report(label, image, up2, roi2)

    print("== A 1x (reducido de vuelta para comparar el poro con el original) ==")
    for label, image in (("A · anterior", legacy), ("B · nuevo", new_pipeline)):
        report(label, same_scale(image, original.size), original)

    # Cara aislada: sin/con conservación de tono. Una cara más clara y con
    # más píxeles al límite es el «blanqueo» de GFPGAN.
    def bright_stats(image: Image.Image, roi=FACE_ROI) -> tuple:
        x1, y1, x2, y2 = roi
        g = np.array(image.convert("L"), dtype=np.float32)[y1:y2, x1:x2]
        return float(g.mean()), float(np.percentile(g, 99.5)), float((g >= 245).mean())

    base_mean, base_p999, base_hot = bright_stats(original)
    print("== Cara sola con GFPGAN 80% (media / p99.5 / %≥245) ==")
    print(f"  Original                       {base_mean:6.1f} {base_p999:6.1f} {base_hot:5.2%}")
    for tone in (False, True):
        face_only = apply_recover_faces_onnx(original, 0.8, "GFPGAN", tone_match=tone)
        mean, p999, hot = bright_stats(face_only)
        tag = "con tono" if tone else "sin tono"
        print(f"  GFPGAN {tag:22s} {mean:6.1f} {p999:6.1f} {hot:5.2%}")

    # Dientes: GFPGAN los aclaraba hasta reventarlos.
    teeth = (228, 125, 262, 145)
    base = float(np.array(original.convert("L"), dtype=np.float32)[teeth[1]:teeth[3], teeth[0]:teeth[2]].mean())
    print("== Brillo medio de los dientes (original = 1.00) ==")
    for label, image in (("A · anterior", legacy), ("B · nuevo", new_pipeline)):
        flat = same_scale(image, original.size)
        val = float(np.array(flat.convert("L"), dtype=np.float32)[teeth[1]:teeth[3], teeth[0]:teeth[2]].mean())
        print(f"  {label:28s} {val / base:5.2f}")

    # Hoja visual: mejilla al 200% y cara completa.
    tile, pad = 300, 26
    rows = [("Original", original)] + stage_images + [
        ("A · pipeline anterior (2x)", same_scale(legacy, original.size)),
        ("B · pipeline nuevo (2x)", same_scale(new_pipeline, original.size)),
    ]
    cols = 3
    sheet_h = ((len(rows) + cols - 1) // cols) * (tile + pad)
    sheet = Image.new("RGB", (tile * cols, sheet_h), "#171717")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(rows):
        x, y = (index % cols) * tile, (index // cols) * (tile + pad) + pad
        sheet.paste(image.crop(SKIN_ROI).resize((tile, tile), Image.NEAREST), (x, y))
        draw.text((x + 8, y - 18), label, fill="white")
    skin_path = OUT / "skin_stages.png"
    sheet.save(skin_path)

    faces = Image.new("RGB", (tile * 2, (tile + pad) * 2), "#171717")
    draw = ImageDraw.Draw(faces)
    for index, (label, image) in enumerate((rows[0], rows[-2], rows[-1], rows[-3])):
        x, y = (index % 2) * tile, (index // 2) * (tile + pad) + pad
        faces.paste(image.crop(FACE_ROI).resize((tile, tile), Image.LANCZOS), (x, y))
        draw.text((x + 8, y - 18), label, fill="white")
    face_path = OUT / "face_orders.png"
    faces.save(face_path)

    # Las 15 recetas completas con su upscale real: recortes, gate y textura.
    print("== Las 15 recetas completas (clip = píxeles al límite) ==")
    catalog = presets.load_presets()
    clip0 = clip_ratio(original)
    preset_faces = []
    for name, entry in catalog.items():
        recipe = dict(entry["sliders"])
        out, notes, _ = process_single_image(tmp, recipe)
        flat = same_scale(out, original.size)
        reverts = [part for part in notes.split(" + ") if "revertido" in part]
        print(
            f"  {name:15s} clip {clip0:5.2%}→{clip_ratio(out):5.2%} "
            f"corr={detail_correlation(flat, original):4.2f} banda={band_ratio(flat, original):4.2f}"
            + (f"  REVERTIDO: {'; '.join(reverts)}" if reverts else "")
        )
        preset_faces.append((name, flat))

    cols_p = 5
    cell_p = 190
    rows_p = (len(preset_faces) + cols_p - 1) // cols_p
    presets_sheet = Image.new("RGB", (cell_p * cols_p, (cell_p + 20) * rows_p), "#171717")
    draw = ImageDraw.Draw(presets_sheet)
    for index, (name, image) in enumerate(preset_faces):
        x = (index % cols_p) * cell_p
        y = (index // cols_p) * (cell_p + 20) + 20
        presets_sheet.paste(image.crop(FACE_ROI).resize((cell_p, cell_p), Image.LANCZOS), (x, y))
        draw.text((x + 6, y - 15), name, fill="white")
    presets_path = OUT / "presets_sheet.png"
    presets_sheet.save(presets_path)

    print(f"HOJA {skin_path}")
    print(f"HOJA {face_path}")
    print(f"HOJA {presets_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
