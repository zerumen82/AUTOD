"""Measure CLIPSeg visual prompts from an actual seed; no production change."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from roop.img_editor.clipseg_masker import CLIPSegMasker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("seed")
    args = parser.parse_args()
    photo = Image.open(args.photo).convert("RGB")
    seed = np.asarray(Image.open(args.seed).convert("L")) >= 96
    assert seed.shape == (photo.height, photo.width) and seed.any()
    masker = CLIPSegMasker()
    ok, message = masker.initialize()
    assert ok and "SIN pesos" not in message, message
    assert masker.device == "cpu"
    transform = transforms.Compose([
        transforms.Resize((352, 352)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    target = transform(photo).unsqueeze(0)
    seed_tensor = torch.from_numpy(seed.astype(np.float32)).unsqueeze(0)
    ys, xs = np.where(seed)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    cropped = transform(photo.crop(bbox)).unsqueeze(0)
    output = Path("diagnostics/out") / f"visual_region_{time.time_ns()}"
    output.mkdir()
    panels = [np.asarray(photo)]
    metadata = {"photo": args.photo, "seed": args.seed, "bbox": bbox, "device": masker.device}
    with torch.inference_mode():
        # This is the masked support embedding used by CLIPDensePredTMasked.
        masked_cond, _, _ = masker.model.visual_forward(
            target, mask=("all", "cls_token", seed_tensor),
        )
        for name, conditional in (("masked_full", masked_cond), ("seed_crop", cropped)):
            logits = masker.model(target, conditional)[0]
            heat = torch.sigmoid(logits[0, 0]).cpu().numpy()
            heat = cv2.resize(heat, photo.size)
            quantized = (heat * 255).astype(np.uint8)
            cutoff, _ = cv2.threshold(quantized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            _, labels = cv2.connectedComponents((quantized > cutoff).astype(np.uint8))
            touching = np.unique(labels[seed])
            touching = touching[touching != 0]
            selected = np.isin(labels, touching)
            Image.fromarray(quantized).save(output / f"{name}_probability.png")
            Image.fromarray((selected * 255).astype(np.uint8)).save(output / f"{name}_mask.png")
            overlay = np.asarray(photo).copy()
            overlay[selected] = (overlay[selected] * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
            panels.append(overlay)
            metadata[name] = {"cutoff": float(cutoff) / 255, "max": float(heat.max()),
                              "coverage": float(selected.mean())}
    Image.fromarray(np.concatenate(panels, axis=1)).save(output / "compare.png")
    (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output, metadata, flush=True)


if __name__ == "__main__":
    main()
