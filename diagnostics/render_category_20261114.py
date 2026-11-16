"""Render one or more Image Editor categories on a real photo and keep evidence.

Usage:
    python diagnostics/render_category_20261114.py clothing remove
    python diagnostics/render_category_20261114.py --all
    python diagnostics/render_category_20261114.py --source <path> --out <dir> clothing
"""

import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png"
DEFAULT_OUT = ROOT / "diagnostics" / "out" / "editor_categories_20261114"

CASES = {
    "pose": "Raise the astronaut's right arm",
    "clothing": "Change the astronaut suit to blue",
    "body": "Make the astronaut taller",
    "features": "Change the astronaut's hair to black",
    "background": "Change the background to a forest",
    "camera": "Show the astronaut from a high camera angle",
    "add": "Add a small flag beside the astronaut",
    "remove": "Remove the black helmet from the photo",
    "people": "Add another astronaut beside the first one",
}


def parse_args(argv):
    source = DEFAULT_SOURCE
    out = DEFAULT_OUT
    categories = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--source":
            index += 1
            source = Path(argv[index])
        elif token == "--out":
            index += 1
            out = Path(argv[index])
        elif token == "--all":
            categories = list(CASES)
        else:
            categories.append(token)
        index += 1
    return source, out, categories


def main(argv):
    source_path, out_dir, categories = parse_args(argv)
    if not categories:
        print("no categories requested", flush=True)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    source = Image.open(source_path).convert("RGB")
    manager = get_img_editor_manager()
    report = []
    for category in categories:
        prompt = CASES.get(category)
        if prompt is None:
            print(f"unknown category {category!r}", flush=True)
            continue
        started = time.monotonic()
        row = {"category": category, "prompt": prompt, "source": str(source_path)}
        try:
            image, message, _ = manager.generate_intelligent(
                image=source, prompt=prompt, engine="auto", scope_hint=category,
                use_semantic=True, use_rewriter=False, enhance_faces=False,
                auto_upscale=False, quality_mode=False,
            )
            row["message"] = message
            if image is not None:
                path = out_dir / f"{category}.png"
                image.save(path)
                row["output"] = str(path)
                row["changed_bbox"] = ImageChops.difference(
                    source, image.convert("RGB")).getbbox()
                row["changed_fraction"] = round(_changed_fraction(source, image), 4)
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["elapsed_s"] = round(time.monotonic() - started, 1)
        report.append(row)
        (out_dir / "report_render.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"CATEGORY {category}: {row}", flush=True)
    return 0


def _changed_fraction(source: Image.Image, edited: Image.Image) -> float:
    import numpy as np

    left = np.asarray(source.convert("RGB"), dtype=np.int16)
    right = np.asarray(edited.convert("RGB").resize(source.size), dtype=np.int16)
    delta = np.max(np.abs(right - left), axis=2)
    return float((delta > 30).mean())


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
