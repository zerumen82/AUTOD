"""Render real de los cinco presets nuevos y prueba de guardado aislado."""

import tempfile
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui.tabs.enhance import presets  # noqa: E402
from ui.tabs.enhance.logic import process_single_image  # noqa: E402


OUT = ROOT / "diagnostics" / "out" / "enhance_presets_20261118"
SOURCES = (
    ROOT / "diagnostics" / "out" / "office" / "window.png",
    ROOT / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png",
)
NEW_PRESETS = ("Blanco y negro", "HDR", "Selfie", "Vintage", "Noche clara")


def main():
    catalog = presets.load_presets()
    factory = {name: data for name, data in catalog.items() if data["_factory"]}
    assert len(factory) == 15, f"catálogo de fábrica incompleto: {list(factory)}"
    for name, data in factory.items():
        icon = presets.icon_for_preset(data)
        assert icon.is_file(), f"sin icono: {name}"
        ET.parse(icon)

    with tempfile.TemporaryDirectory() as directory:
        original_dir = presets.PRESET_DIR
        presets.PRESET_DIR = Path(directory)
        try:
            (Path(directory) / "malformed.json").write_text("[]", encoding="utf-8")
            sliders = dict(catalog["Realista"]["sliders"])
            name = "Prueba temporal"
            saved = presets.save_custom_preset(name, sliders)
            assert (Path(directory) / saved).is_file()
            assert name in presets.load_presets()
            try:
                presets.save_custom_preset(name, sliders)
            except ValueError:
                pass
            else:
                raise AssertionError("El guardado sobrescribió un preset")
        finally:
            presets.PRESET_DIR = original_dir

    OUT.mkdir(parents=True, exist_ok=True)
    for source in SOURCES:
        with Image.open(source) as opened:
            before = ImageOps.exif_transpose(opened).convert("RGB")
        before.thumbnail((512, 512), Image.Resampling.LANCZOS)
        prepared = OUT / f"{source.parent.name}_before.png"
        before.save(prepared)
        tiles = [("Antes", before)]
        for name in NEW_PRESETS:
            after, notes, metrics = process_single_image(prepared, catalog[name]["sliders"])
            after.save(OUT / f"{source.parent.name}_{catalog[name]['_path'].stem}.png")
            if name == "Blanco y negro":
                rgb = np.asarray(after.convert("RGB"), dtype=np.int16)
                delta = max(
                    int(np.abs(rgb[:, :, 0] - rgb[:, :, 1]).max()),
                    int(np.abs(rgb[:, :, 1] - rgb[:, :, 2]).max()),
                )
                assert delta <= 1, f"Blanco y negro retiene color: delta={delta}"
            print(f"{source.parent.name} | {name} | {metrics['elapsed']:.2f}s | {after.size} | {notes}")
            tile = after.copy()
            tile.thumbnail(before.size, Image.Resampling.LANCZOS)
            tiles.append((name, tile))
        width, height = before.size
        sheet = Image.new("RGB", (width * 3, (height + 30) * 2), "#171717")
        draw = ImageDraw.Draw(sheet)
        for index, (name, tile) in enumerate(tiles):
            x, y = (index % 3) * width, (index // 3) * (height + 30)
            sheet.paste(tile, (x, y))
            draw.text((x + 8, y + height + 5), name, fill="white")
        sheet_path = OUT / f"{source.parent.name}_sheet.png"
        sheet.save(sheet_path)
        print(f"HOJA {sheet_path}")
    print("PASS — presets 15/15 válidos, iconos SVG, guardado y renders reales")


if __name__ == "__main__":
    main()
