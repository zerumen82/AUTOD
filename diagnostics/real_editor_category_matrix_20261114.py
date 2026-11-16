"""Render one real photograph per Image Editor category and retain evidence."""

import json
import sys
import time
from pathlib import Path

from PIL import Image, ImageChops

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/out/editor_camera_neutral_20261110/before.png"
OUT = ROOT / "diagnostics/out/editor_categories_20261114"
CASES = (
    ("pose", "Raise the astronaut's right arm"),
    ("clothing", "Change the astronaut suit to blue"),
    ("body", "Make the astronaut taller"),
    ("features", "Change the astronaut's hair to black"),
    ("background", "Change the background to a forest"),
    ("camera", "Show the astronaut from a high camera angle"),
    ("add", "Add a small flag beside the astronaut"),
    ("remove", "Remove the black helmet from the photo"),
    ("people", "Add another astronaut beside the first one"),
)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    wanted = [arg.casefold() for arg in sys.argv[1:] if not arg.startswith("-")]
    source = Image.open(SOURCE).convert("RGB")
    report = []
    manager = get_img_editor_manager()
    for category, prompt in CASES:
        if wanted and category.casefold() not in wanted:
            continue
        started = time.monotonic()
        row = {"category": category, "prompt": prompt, "source": str(SOURCE)}
        try:
            image, message, _ = manager.generate_intelligent(
                image=source, prompt=prompt, engine="auto", scope_hint=category,
                use_semantic=True, use_rewriter=False, enhance_faces=False,
                auto_upscale=False, quality_mode=False,
            )
            row["message"] = message
            if image is not None:
                path = OUT / f"{category}.png"
                image.save(path)
                row["output"] = str(path)
                row["changed_bbox"] = ImageChops.difference(
                    source, image.convert("RGB")).getbbox()
            else:
                row["output"] = None
        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
        row["elapsed_s"] = round(time.monotonic() - started, 1)
        report.append(row)
        report_name = "report.json" if not wanted else f"report_{category}.json"
        (OUT / report_name).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"CATEGORY {category}: {row}", flush=True)


if __name__ == "__main__":
    main()
