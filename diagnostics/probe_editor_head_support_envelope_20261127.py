"""Preview upper subject-envelope support using already recorded model outputs."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics/out/editor_head_support_envelope_20261127"
CASES = {
    "grace": (
        ROOT / "diagnostics/out/appearance_reference_20261023/1795764767693508900/source.png",
        ROOT / "diagnostics/out/appearance_reference_20261023/1795764767693508900/raw_0.png",
        ROOT / "diagnostics/out/editor_grace_replay_20261127",
    ),
    "saved_head": (
        ROOT / "ui/tob/ComfyUI/input/fast_1795733009.png",
        ROOT / "ui/tob/ComfyUI/output/FastEdit_01438_.png",
        ROOT / "diagnostics/out/editor_head_paste_latest_replay_20261127",
    ),
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, (source_path, raw_path, masks) in CASES.items():
        source = np.asarray(Image.open(source_path).convert("RGB"))
        raw = np.asarray(Image.open(raw_path).convert("RGB").resize(
            (source.shape[1], source.shape[0]), Image.Resampling.LANCZOS))
        head = np.asarray(Image.open(masks / "head_measured.png" if name == "saved_head" else masks / "head.png").convert("L")) >= 96
        refined = np.asarray(Image.open(masks / "head_refined.png").convert("L")) >= 96
        subject = np.asarray(Image.open(masks / "subject.png").convert("L")) >= 96
        ys, xs = np.where(refined)
        span = max(np.ptp(xs) + 1, np.ptp(ys) + 1)
        radius = max(1, round(span * .035))
        ceiling = int(ys.min() + (ys.max() - ys.min()) * .4)
        wider = cv2.dilate(subject.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0
        ygrid = np.indices(refined.shape)[0]
        xgrid = np.indices(refined.shape)[1]
        horizontal = (xgrid >= int(xs.min()) - radius) & (xgrid <= int(xs.max()) + radius)
        support = refined | (head & wider & horizontal & (ygrid <= ceiling))
        alpha = cv2.GaussianBlur(support.astype(np.float32), (0, 0),
                                 max(1, round(span * .016)))[:, :, None]
        output = np.clip(source * (1 - alpha) + raw * alpha, 0, 255).astype(np.uint8)
        Image.fromarray(support.astype(np.uint8) * 255).save(OUT / f"{name}_support.png")
        Image.fromarray(output).save(OUT / f"{name}_candidate.png")
        print(name, "span", span, "radius", radius, "ceiling", ceiling,
              "support", int(support.sum()), "added", int((support & ~refined).sum()))


if __name__ == "__main__":
    main()
