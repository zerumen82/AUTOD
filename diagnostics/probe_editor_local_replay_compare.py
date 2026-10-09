"""Compose a real saved local sample through the production paste function."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("call_directory")
    parser.add_argument("sample")
    args = parser.parse_args()
    original = Image.open(args.source).convert("RGB")
    call = Path(args.call_directory)
    window = Image.open(call / "window.png").convert("RGB")
    mask = Image.open(call / "mask.png").convert("L")
    matches = cv2.matchTemplate(np.asarray(original), np.asarray(window), cv2.TM_SQDIFF)
    _, _, location, _ = cv2.minMaxLoc(matches)
    x, y = location
    box = (x, y, x + window.width, y + window.height)
    error = np.abs(np.asarray(original.crop(box), np.int16) - np.asarray(window, np.int16))
    assert error.max() == 0, "Saved window is not an exact crop of this source"
    full_mask = Image.new("L", original.size)
    full_mask.paste(mask, location)
    sample = Image.open(args.sample).convert("RGB")
    output = ImgEditorManager()._integrate_local_generation(
        original, sample, full_mask, box, keep_measured_mask=True,
    )
    target = Path("diagnostics/out") / f"local_replay_compare_{time.time_ns()}"
    target.mkdir()
    output.save(target / "composed.png")
    before = np.asarray(original, np.int16)
    delta = np.abs(before - np.asarray(output, np.int16)).max(axis=2)
    metadata = {"box": box, "changed_fraction": float((delta > 10).mean()),
                "source": args.source, "sample": args.sample, "source_crop_max_error": 0}
    (target / "meta.json").write_text(json.dumps(metadata, indent=2))
    strip = Image.new("RGB", (original.width * 2 + 12, original.height), (24, 24, 24))
    strip.paste(original)
    strip.paste(output, (original.width + 12, 0))
    strip.save(target / "compare.png")
    print(metadata)
    print(target.resolve())


if __name__ == "__main__":
    main()
