#!/usr/bin/env python3
import cv2
import numpy as np
from PIL import Image
from roop.img_editor.image_quality_pipeline import get_quality_finisher

finisher = get_quality_finisher()
paths = [
    ("input_3980", "img_editor/edit_1782368164.png"),
    ("output_7680", "img_editor/edit_1782369536.png"),
    ("small_jpg", "debug_test/test_result_same.jpg"),
]

for label, p in paths:
    img = Image.open(p)
    bgr = finisher._pil_to_bgr(img)
    for tier in ("hd", "8k"):
        am, m = finisher._compute_artifact_map(bgr, tier=tier)
        print(
            f"{label} tier={tier} size={img.size}: "
            f"block_p90={m['block_p90']:.3f} band_p90={m['band_p90']:.3f} "
            f"score_mean={am.mean():.3f} score_p90={m['score_p90']:.3f} "
            f"score_max={m['score_max']:.3f} crit>{0.32}={m['critical_ratio']:.3f}"
        )