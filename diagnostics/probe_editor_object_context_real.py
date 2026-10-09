"""Diagnose whether a measured full-object context box completes the plant.

The previous crop diagnostic enlarged only the visible seed bbox, so the
prompt never reached the pot area. Here the context box is derived from the
measured seed itself: vertical closure down to the strongest floor evidence
below the seed, then horizontal opening, always from image measurements
(gradients/edges) plus the exclusion mask. No annotation coordinates feed the
inference; the manual masks are used only afterwards for evaluation.
"""
import argparse
import hashlib
import json
import os
import time
import uuid
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

from roop.img_editor.clipseg_masker import CLIPSegMasker  # noqa: E402


def find_floor(seed, exclusion, gray):
    """Row evidence for ground contact under the seed, from image gradients."""
    h, w = seed.shape
    ys, xs = np.where(seed)
    left, right = int(xs.min()), int(xs.max() + 1)
    top, bottom = int(ys.min()), int(ys.max() + 1)
    sobel = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    strip = sobel[:, left:right]
    free = ~exclusion[:, left:right]
    # The occluder sits in front of the object, so rows behind it still carry
    # object-edge evidence on their free pixels. Requiring a fully exclusion-
    # free row stopped the closure at the occluder instead of the floor.
    strength = np.where(free, strip, 0.0).sum(axis=1) / np.maximum(free.sum(axis=1), 1)
    row = float(strength[top:bottom].mean())
    best_row, best_score = None, 0.0
    for y in range(bottom, min(h, int(bottom + (bottom - top) * 0.6))):
        if free[y].sum() < (right - left) * 0.25:
            continue
        if strength[y] > best_score:
            best_score, best_row = strength[y], y
    return best_row if best_score > row * 1.05 else bottom


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('seed', type=Path)
    parser.add_argument('exclusion', type=Path)
    parser.add_argument('query')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--truth', type=Path, help='manual diagnostic mask, evaluation only')
    parser.add_argument('--neighbour', type=Path, help='manual neighbouring object, evaluation only')
    parser.add_argument('--extend-floor', type=int, default=0,
                        help='diagnostic: extend the derived box bottom by N px (default 0)')
    args = parser.parse_args()
    assert args.extend_floor >= 0
    started = time.monotonic()
    torch.set_num_threads(4)
    source = Image.open(args.source).convert('RGB')
    rgb = np.asarray(source)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    seed = np.asarray(Image.open(args.seed).convert('L')) >= 96
    excluded = np.asarray(Image.open(args.exclusion).convert('L')) >= 128
    assert seed.shape == excluded.shape == gray.shape
    seed &= ~excluded
    assert seed.any()
    ys, xs = np.where(seed)
    left, right = int(xs.min()), int(xs.max() + 1)
    top, bottom = int(ys.min()), int(ys.max() + 1)
    floor = find_floor(seed, excluded, gray)
    pad_x = int(0.25 * (right - left))
    box = (max(0, left - pad_x), max(0, top - int(0.10 * (bottom - top))),
           min(source.width, right + pad_x),
           min(source.height, floor + 4 + args.extend_floor))
    output = args.report / f'object_context_{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True)
    backbone = Path.home() / '.cache/clip/ViT-B-16.pt'
    assert backbone.is_file(), 'CLIP backbone must already exist; no implicit download allowed'
    masker = CLIPSegMasker()
    assert masker.device == 'cpu'
    metadata = dict(completed=False, device='cpu', query=args.query,
                    seed_box=[left, top, right, bottom], floor_row=floor, box=list(box),
                    source=str(args.source), seed=str(args.seed), exclusion=str(args.exclusion),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    variants=[])
    try:
        # baseline: full frame
        plans = [('full', source, (0, 0, source.width, source.height)),
                 ('context_box', source.crop(box), box)]
        for label, image, bounds in plans:
            mask = masker.generate_mask(image, args.query)
            probability = masker.probability(image, args.query)
            assert probability is not None and np.isfinite(probability).all()
            lx, ty, rx, by = bounds
            selected = np.zeros(gray.shape, bool)
            raw_excl = 0
            if mask is not None:
                selected[ty:by, lx:rx] = np.asarray(mask) >= 96
                raw_excl = int((selected & excluded).sum())
                selected &= ~excluded
            Image.fromarray(selected.astype(np.uint8) * 255).save(output / f'{label}_mask.png')
            overlay = rgb.copy()
            overlay[selected] = (overlay[selected] * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
            Image.fromarray(overlay).save(output / f'{label}_overlay.png')
            heat = np.zeros(gray.shape, np.float32)
            heat[ty:by, lx:rx] = probability
            Image.fromarray((heat * 255).astype(np.uint8)).save(output / f'{label}_probability.png')
            entry = dict(label=label, box=list(bounds), coverage=float(selected.mean()),
                         raw_exclusion_overlap=raw_excl,
                         crop_boundary_foreground=bool(np.any(probability[0] > .4)
                             or np.any(probability[-1] > .4) or np.any(probability[:, 0] > .4)
                             or np.any(probability[:, -1] > .4)))
            if args.truth:
                truth = np.asarray(Image.open(args.truth).convert('L')) > 127
                inter = int((selected & truth).sum())
                union = int((selected | truth).sum())
                entry.update(truth_iou=inter / union if union else None,
                             truth_recall=inter / int(truth.sum()),
                             truth_precision=inter / int(selected.sum()) if selected.any() else None)
                if args.neighbour:
                    neighbour = np.asarray(Image.open(args.neighbour).convert('L')) > 127
                    entry['neighbour_overlap'] = int((selected & neighbour).sum())
            metadata['variants'].append(entry)
            print(label, json.dumps(entry), flush=True)
        metadata.update(completed=True, seconds=time.monotonic() - started)
    finally:
        masker.unload()
        (output / 'meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    main()
