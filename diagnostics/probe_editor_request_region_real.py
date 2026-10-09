"""Measure an arbitrary requested part with the installed segmenter on CPU."""
import argparse
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("subject")
    parser.add_argument("request")
    args = parser.parse_args()
    source = Image.open(args.photo).convert("RGB")
    manager = ImgEditorManager()
    masker = manager._get_clipseg_masker()
    subject = masker.generate_mask(source, args.subject)
    assert subject is not None, "Subject was not measured"
    target = Path("diagnostics/out") / f"request_region_{time.time_ns()}"
    target.mkdir()
    subject.save(target / "subject.png")
    box = Image.fromarray((np.asarray(subject) >= 96).astype(np.uint8) * 255).getbbox()
    assert box is not None
    for name, image, mask in (
        ("full", source, subject),
        ("subject_crop", source.crop(box), subject.crop(box)),
    ):
        image.save(target / f"{name}_source.png")
        found = manager._request_region_on_subject(image, mask, [args.request])
        print(f"{name}: size={image.size}, region={found is not None}, box={box}")
        if found is not None:
            found.save(target / f"{name}_mask.png")
            print(f"{name}: coverage={(np.asarray(found) >= 96).mean():.4%}")
    print(target.resolve())


if __name__ == "__main__":
    main()
