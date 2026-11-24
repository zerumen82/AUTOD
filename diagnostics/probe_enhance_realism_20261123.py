"""Probe de realismo de «Mejorar fotos» (v5.1437).

El usuario reportó que las últimas mejoras salían «muy muñecos». Los logs del
22-23/11 muestran la receta con la que se abrió la pestaña (denoise 65,
sharpen 40, caras 80, grano 20/tamaño 10) y las salidas de `enhance/` llevan
las etiquetas `dn-sh-rf-up2x-db-co`, compatibles con esa base más un preset
acumulado (v5.1417 suma denoise/sharpen/grano).

Este probe reproduce:
  A) la receta de los logs (denoise 65 + sharpen 40 + grano 20/10),
  B) la pila probable (preset Ultra realista sumado a esa base: 95/85/100/30),
  C) el preset Ultra realista nuevo (v5.1437, con recuperación de caras),
  D) los valores por defecto nuevos.
Mide energía de alta frecuencia (textura) en una zona de piel y escribe una
hoja de contactos para revisión visual. A y B usan las fórmulas anteriores
(denoise bilateral único y unsharp sin umbral) para reproducir el defecto.
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
    apply_grain_cv2,
    load_enhance_config,
    process_single_image,
)

SOURCE = ROOT / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png"
OUT = ROOT / "diagnostics" / "out" / "enhance_realism_20261123"
# Zona de piel (mejilla derecha del retrato 512x512) sin pelo ni fondo.
SKIN_ROI = (215, 135, 300, 205)


def old_denoise(image: Image.Image, strength: float) -> Image.Image:
    """Fórmula anterior: bilateral único, sin devolver detalle."""
    d = 5 + int(strength / 100 * 15)
    sigma = 10 + int(strength / 100 * 40)
    arr = np.array(image.convert("RGB"))
    return Image.fromarray(cv2.bilateralFilter(arr, d, sigma, sigma))


def old_sharpen(image: Image.Image, strength: float) -> Image.Image:
    """Fórmula anterior: unsharp sin umbral ni recorte de halo."""
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    sigma = 1.0 + strength / 100 * 2.0
    blur = cv2.GaussianBlur(arr, (3, 3), sigma)
    amount = 0.3 + strength / 100 * 1.2
    return Image.fromarray((np.clip(arr + amount * (arr - blur), 0, 1) * 255).astype(np.uint8))


def old_recipe(image: Image.Image, denoise: float, sharpen: float, grain: float) -> Image.Image:
    out = old_denoise(image, denoise)
    out = old_sharpen(out, sharpen)
    return apply_grain_cv2(out, grain, 10, grain_type="gaussian", seed=7)


def _high_pass(image: Image.Image) -> np.ndarray:
    """Banda de microdetalle (poro, borde fino) de la mejilla."""
    x1, y1, x2, y2 = SKIN_ROI
    g = np.array(image.convert("L"), dtype=np.float32)[y1:y2, x1:x2]
    return g - cv2.GaussianBlur(g, (0, 0), 1.5)


def detail_correlation(image: Image.Image, reference: Image.Image) -> float:
    """Cuánto del microdetalle real del original sobrevive en la salida.

    Un denoise que borra el poro deja poca correlación; un sharpen que añade
    ruido propio también. Lo que se conserva de verdad correlaciona.
    """
    a = _high_pass(image).ravel()
    b = _high_pass(reference).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def check_presets() -> None:
    """Guardia: ningún preset de caras puede volver a la receta «muñeco»."""
    catalog = presets.load_presets()
    assert len(catalog) == 15, f"catálogo incompleto: {list(catalog)}"
    face_presets = ("Realista", "Ultra realista", "Retrato", "Selfie", "Archivo", "Color vivo", "Nocturna", "Noche clara")
    for name in face_presets:
        s = catalog[name]["sliders"]
        assert s["recover_faces_strength"] <= 0.5, f"{name}: recuperación de caras {s['recover_faces_strength']}"
        assert s["grain_size"] <= 4, f"{name}: grano {s['grain_size']} demasiado grueso"
        assert s["denoise_strength"] <= 60, f"{name}: denoise {s['denoise_strength']} vuelve a borrar el poro"
        assert s["sharpen_strength"] <= 35, f"{name}: sharpen {s['sharpen_strength']} deja halo"
    print("PASS — 15 presets válidos y sin recetas de piel de cera")


def main() -> int:
    check_presets()
    with Image.open(SOURCE) as opened:
        original = opened.convert("RGB")
    assert original.size == (512, 512), f"muestra inesperada: {original.size}"
    OUT.mkdir(parents=True, exist_ok=True)

    config = load_enhance_config()
    defaults = dict(config["default_slider_values"])
    ultra = dict(presets.load_presets()["Ultra realista"]["sliders"])

    variants = []
    variants.append(("A · logs 65/40/80/20", old_recipe(original, 65, 40, 20)))
    variants.append(("B · pila 95/85/100/30", old_recipe(original, 95, 85, 30)))

    tmp = OUT / "source.png"
    original.save(tmp)
    for label, recipe in (("C · Ultra realista nueva", ultra), ("D · valores por defecto", defaults)):
        settings = dict(recipe)
        settings["upscale_factor"] = 1  # misma escala para comparar textura
        out, notes, metrics = process_single_image(tmp, settings)
        variants.append((label, out))
        print(f"[{label}] {notes}")

    # E) El caso que produjo el lote real: preset sumado dos veces a la base
    # (v5.1417 acumula denoise/sharpen/grano), ahora con el tope de pila real.
    from ui.tabs.enhance.events import _merge_settings, _neutral_settings  # noqa: E402

    add = dict(ultra)
    add["accumulated_presets"] = ["Ultra realista"]
    stacked = _neutral_settings()
    for _ in range(2):
        stacked = _merge_settings(stacked, dict(add))
    stacked["upscale_factor"] = 1
    out, notes, _metrics = process_single_image(tmp, stacked)
    variants.append((f"E · pila con tope {stacked['denoise_strength']:g}/{stacked['sharpen_strength']:g}", out))
    print(f"[E · pila con tope] {notes} tope={stacked.get('stack_capped')}")

    rows = [("Original", original)] + variants
    print("Detalle real conservado (correlación de la banda de poro con el original):")
    for label, image in variants:
        print(f"  {label:28s} {detail_correlation(image, original):5.2f}")

    tile = 300
    pad = 26
    sheet = Image.new("RGB", (tile * len(rows), tile + pad), "#171717")
    draw = ImageDraw.Draw(sheet)
    for index, (label, image) in enumerate(rows):
        crop = image.crop(SKIN_ROI).resize((tile, tile), Image.LANCZOS)
        sheet.paste(crop, (index * tile, pad))
        draw.text((index * tile + 8, 7), label, fill="white")
    sheet_path = OUT / "skin_sheet.png"
    sheet.save(sheet_path)

    # Cara completa (2x2) para ver el «muñeco» de un vistazo.
    face = (150, 60, 370, 280)
    face_sheet = Image.new("RGB", (tile * 2, (tile + pad) * 2), "#171717")
    draw = ImageDraw.Draw(face_sheet)
    for index, (label, image) in enumerate((rows[0], rows[1], rows[3], rows[4])):
        crop = image.crop(face).resize((tile, tile), Image.LANCZOS)
        x, y = (index % 2) * tile, (index // 2) * (tile + pad) + pad
        face_sheet.paste(crop, (x, y))
        draw.text((x + 8, y - 19), label, fill="white")
    face_path = OUT / "face_sheet.png"
    face_sheet.save(face_path)

    # Zoom de poro (vecino más próximo) sobre una mejilla plana.
    zoom_roi = (225, 150, 275, 200)
    zoom_sheet = Image.new("RGB", (tile * len(rows), tile + pad), "#171717")
    draw = ImageDraw.Draw(zoom_sheet)
    for index, (label, image) in enumerate(rows):
        crop = image.crop(zoom_roi).resize((tile, tile), Image.NEAREST)
        zoom_sheet.paste(crop, (index * tile, pad))
        draw.text((index * tile + 8, 7), label, fill="white")
    zoom_path = OUT / "pore_zoom.png"
    zoom_sheet.save(zoom_path)
    print(f"HOJA {sheet_path}")
    print(f"HOJA {face_path}")
    print(f"HOJA {zoom_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
