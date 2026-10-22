"""Freeze sparse manual positive cores on the real, unedited coffee photo.

Coordinates below were selected by sd-tester by inspecting the SOURCE, never
inferred by a model. They are conservative interior samples, not a dense object
outline or human-certified ground truth. Unknown pixels cannot establish recall,
precision or IoU for the whole object. These annotations must not enter production
selection, conditioning, or an end-to-end render.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_RGB_HASH = "10db431a865f5d1a23b4c4b388291d86867477b96320e0a336c347af27014d65"

# Explicitly diagnostic, source-specific manual annotation; never routing rules.
PARTS = {
    "cup_body": {"object": "cup", "polygon": [
        [258, 222], [300, 229], [331, 222], [327, 241],
        [310, 260], [284, 268], [261, 263], [250, 246],
    ]},
    "cup_handle": {"object": "cup", "polygon": [
        [209, 268], [216, 270], [217, 282], [213, 289], [207, 286], [207, 278],
    ]},
    "cup_rim": {"object": "cup", "polygon": [
        [268, 197], [287, 199], [312, 198], [313, 201], [290, 202], [268, 200],
    ]},
    "spoon_bowl": {"object": "spoon", "polygon": [
        [347, 266], [363, 264], [378, 275], [385, 290], [379, 306],
        [363, 315], [349, 309], [339, 298], [339, 283],
    ]},
    "spoon_neck": {"object": "spoon", "polygon": [
        [337, 240], [344, 235], [354, 237], [352, 245], [344, 250], [337, 248],
    ]},
    "spoon_rear_handle": {"object": "spoon", "polygon": [
        [407, 76], [414, 74], [418, 77], [418, 86], [414, 96], [409, 106], [407, 98],
    ]},
    "saucer_left": {"object": "saucer", "polygon": [
        [103, 166], [128, 138], [151, 126], [163, 135], [157, 162], [165, 191],
        [177, 216], [173, 232], [157, 249], [131, 255], [110, 233], [101, 211],
    ]},
    "saucer_bottom": {"object": "saucer", "polygon": [
        [190, 329], [234, 346], [285, 350], [331, 343], [376, 330],
        [396, 337], [379, 350], [346, 364], [298, 374], [247, 370],
        [205, 354], [186, 339],
    ]},
    "saucer_right": {"object": "saucer", "polygon": [
        [430, 188], [452, 200], [458, 220], [453, 241], [443, 259],
        [429, 275], [421, 266], [430, 241], [432, 219],
    ]},
    "coffee_content": {"object": "coffee", "polygon": [
        [227, 127], [239, 108], [266, 100], [292, 101], [322, 110],
        [340, 128], [334, 145], [309, 158], [277, 160], [248, 151], [231, 139],
    ]},
    "table": {"object": "table", "polygon": [
        [498, 35], [566, 35], [566, 160], [502, 160],
    ]},
}
COLORS = {"cup": (255, 40, 180), "spoon": (30, 255, 80),
          "saucer": (30, 170, 255), "coffee": (255, 230, 30), "table": (220, 220, 220)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=ROOT / "venv/Lib/site-packages/skimage/data/coffee.png")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "diagnostics/out/editor_selection_gt_20261019/coffee_v1")
    args = parser.parse_args()
    source = Image.open(args.source).convert("RGB")
    digest = hashlib.sha256(f"RGB:{source.width}:{source.height}:".encode("ascii"))
    digest.update(source.tobytes())
    assert source.size == (600, 400) and digest.hexdigest() == EXPECTED_RGB_HASH, (
        "These manually traced polygons belong only to the inspected coffee source"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    source.save(args.output / "source.png")
    overlay = source.convert("RGBA")
    paint = Image.new("RGBA", source.size)
    draw = ImageDraw.Draw(paint)
    known = np.zeros((source.height, source.width), dtype=bool)
    families = {}
    records = []
    for index, (name, part) in enumerate(PARTS.items(), 1):
        mask = Image.new("L", source.size)
        ImageDraw.Draw(mask).polygon(part["polygon"], fill=255)
        area = np.asarray(mask) > 0
        assert area.any() and not np.any(known & area), f"Manual cores overlap: {name}"
        known |= area
        families.setdefault(part["object"], np.zeros_like(known))
        families[part["object"]] |= area
        mask.save(args.output / f"core_{name}.png")
        color = COLORS[part["object"]]
        draw.polygon(part["polygon"], fill=(*color, 100), outline=(*color, 255))
        ys, xs = np.where(area)
        draw.text((int(xs.mean()), int(ys.mean())), str(index), fill=(0, 0, 0, 255),
                  stroke_fill=(255, 255, 255, 255), stroke_width=1)
        records.append({"id": index, "name": name, **part, "positive_core_pixels": int(area.sum())})
    for family, area in families.items():
        Image.fromarray(area.astype(np.uint8) * 255).save(args.output / f"core_object_{family}.png")
    Image.fromarray((~known).astype(np.uint8) * 255).save(args.output / "unknown.png")
    canvas = Image.new("RGB", (source.width, source.height + len(records) * 17 + 28), "white")
    canvas.paste(Image.alpha_composite(overlay, paint).convert("RGB"), (0, 0))
    legend = ImageDraw.Draw(canvas)
    legend.text((8, 405), "SPARSE SOURCE CORES - unknown elsewhere; NOT dense ground truth", fill="black")
    for index, part in enumerate(records):
        legend.text((8, 426 + 17 * index), f"{part['id']}: {part['name']} ({part['positive_core_pixels']} px)", fill="black")
    canvas.save(args.output / "core_overlay.png")
    manifest = {
        "annotation_type": "sparse_source_manual_positive_cores",
        "dense_ground_truth": False, "human_certified": False,
        "annotator": "sd-tester (agent; manually inspected source)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()), "source_size": source.size,
        "source_rgb_fingerprint": digest.hexdigest(),
        "provenance": "Polygons chosen by inspecting the original source and source pixel values; no inference.",
        "uncertainty": "All pixels outside the annotated positive cores are UNKNOWN, including boundaries, holes, occluded surfaces and untraced visible parts.",
        "limitations": "Part-core coverage can reveal missing parts; these sparse cores cannot establish whole-object recall, precision or IoU, nor prove full coverage.",
        "use_restriction": "Diagnostic evaluation only. Never provide these masks to production selection, conditioning, or an end-to-end render.",
        "parts": records,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    (args.output / "annotations.json").write_text(payload, encoding="utf-8")
    (args.output / "annotations.sha256").write_text(hashlib.sha256(payload.encode("utf-8")).hexdigest() + "\n", encoding="ascii")
    print(f"Frozen {len(records)} manually inspected sparse cores: {args.output}")


if __name__ == "__main__":
    main()
