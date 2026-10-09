"""Compare real local SAM2 prompting and source-color GrabCut, without production edits.

All prompts derive from measured masks. Predicted IoU is model confidence,
not an annotated object-completeness score. Save every alternative for review.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Sam2Model, Sam2Processor

from diagnostics.probe_editor_sam2_region_real import interior_point


def spread_points(mask, count):
    """Pick interior points distributed across an actually measured mask."""
    if not mask.any():
        return []
    depth = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    yy, xx = np.indices(mask.shape)
    first = interior_point(mask)
    points = [first]
    nearest = np.full(mask.shape, np.inf)
    for _ in range(count - 1):
        px, py = points[-1]
        nearest = np.minimum(nearest, (xx-px)**2 + (yy-py)**2)
        score = np.sqrt(nearest) * np.sqrt(depth)
        score[~mask] = 0
        for px, py in points:
            score[py, px] = 0
        if score.max() <= 0:
            break
        py, px = np.unravel_index(score.argmax(), score.shape)
        points.append([int(px), int(py)])
    assert all(mask[y, x] for x, y in points)
    return points


def refine(image, candidate, seed, excluded, band, iterations):
    support = (candidate | seed) & ~excluded
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band*2+1,)*2)
    near = cv2.dilate(support.astype(np.uint8), kernel) > 0
    core = cv2.erode(candidate.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    classes = np.full(seed.shape, cv2.GC_BGD, np.uint8)
    classes[near] = cv2.GC_PR_BGD
    classes[support] = cv2.GC_PR_FGD
    classes[core] = cv2.GC_FGD
    classes[excluded] = cv2.GC_BGD
    if not np.any(classes == cv2.GC_FGD) or not np.any(classes == cv2.GC_BGD):
        raise ValueError('No foreground/background evidence for GrabCut')
    cv2.grabCut(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), classes, None,
                np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                iterations, cv2.GC_INIT_WITH_MASK)
    return np.isin(classes, (cv2.GC_FGD, cv2.GC_PR_FGD)) & ~excluded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo', type=Path)
    parser.add_argument('seed', type=Path)
    parser.add_argument('exclusion', type=Path)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--points', type=int, required=True)
    parser.add_argument('--band', type=int, required=True)
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--context-mask', type=Path,
                        help='Already measured SAM mask used only to derive a context box')
    parser.add_argument('--context-box',
                        help='Measured context box JSON [l,t,r,b] from object-context probe; evaluation-free prompting')
    parser.add_argument('--single-output', action='store_true')
    parser.add_argument('--positive-heat', type=Path,
                        help='Measured probability map (e.g. CLIPSeg) for extra positive points below the seed')
    parser.add_argument('--negative-mask', type=Path,
                        help='Measured binary mask (e.g. CLIPSeg neighbour) used as extra negative points')
    args = parser.parse_args()
    total_started = time.monotonic()
    assert args.points > 0 and args.band > 0 and args.iterations > 0
    torch.set_num_threads(4)
    photo = Image.open(args.photo).convert('RGB')
    rgb = np.asarray(photo)
    original_seed = np.asarray(Image.open(args.seed).convert('L')) >= 96
    excluded = np.asarray(Image.open(args.exclusion).convert('L')) >= 128
    assert original_seed.shape == excluded.shape == rgb.shape[:2]
    seed = original_seed & ~excluded
    assert seed.any()
    processor = Sam2Processor.from_pretrained(args.model, local_files_only=True)
    model, info = Sam2Model.from_pretrained(args.model, local_files_only=True, output_loading_info=True)
    assert not any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
    model.eval()
    base = processor(images=photo, return_tensors='pt')
    with torch.inference_mode():
        embeddings = model.get_image_embeddings(base['pixel_values'])
    count, components = cv2.connectedComponents((excluded & ~original_seed).astype(np.uint8))
    old_negative = [interior_point(components == index) for index in range(1, count)]
    positive = spread_points(seed, args.points)
    negative = spread_points(excluded & ~original_seed, args.points)
    yy, xx = np.where(seed)
    box = [int(xx.min()), int(yy.min()), int(xx.max()+1), int(yy.max()+1)]
    plans = [('single', [interior_point(seed)], old_negative, None),
             ('distributed', positive, negative, None),
             ('distributed_box', positive, negative, box)]
    context_bounds = None
    if args.context_box:
        # Derived from image measurements upstream (probe_editor_object_context_real);
        # never from manual annotation coordinates.
        context = args.context_box.strip()
        values = json.loads(context)
        while isinstance(values, list) and len(values) == 1 and isinstance(values[0], list):
            values = values[0]
        context = [int(v) for v in values]
        assert len(context) == 4 and context[0] < context[2] and context[1] < context[3]
        assert 0 <= context[0] and context[2] <= seed.shape[1] and 0 <= context[1] and context[3] <= seed.shape[0]
        context_bounds = context
        plans.append(('context_box', positive, negative, context))
    if args.context_mask:
        measured_context = np.asarray(Image.open(args.context_mask).convert('L')) >= 128
        assert measured_context.shape == seed.shape and measured_context.any()
        cy,cx = np.where(measured_context & ~excluded)
        context_box=[int(cx.min()),int(cy.min()),int(cx.max()+1),int(cy.max()+1)]
        plans.append(('measured_context_box',positive,negative,context_box))
    extra_positive, extra_negative = [], []
    if args.positive_heat:
        # Extra positives come from measured heat below the seed inside the
        # context box (pot area), never from annotation coordinates.
        heat = np.asarray(Image.open(args.positive_heat).convert('L')).astype(np.float32) / 255.0
        assert heat.shape == seed.shape
        assert context_bounds is not None, '--positive-heat requires --context-box'
        below = int(np.where(seed)[0].max() + 1) + 10
        top = max(below, context_bounds[1])
        assert top < context_bounds[3], 'no measured band below the seed inside the context box'
        band = np.zeros(seed.shape, bool)
        band[top:context_bounds[3], context_bounds[0]:context_bounds[2]] = True
        candidates = band & (heat >= 0.10) & ~excluded
        assert candidates.any(), 'measured positive heat has no pixels below the seed'
        extra_positive = spread_points(candidates, max(2, args.points // 2))
    if args.negative_mask:
        measured_negative = np.asarray(Image.open(args.negative_mask).convert('L')) >= 128
        assert measured_negative.shape == seed.shape and measured_negative.any()
        candidates_negative = measured_negative & ~seed
        assert candidates_negative.any(), 'measured negative mask empty after excluding the seed'
        extra_negative = spread_points(candidates_negative, max(2, args.points // 2))
    if context_bounds is not None and (extra_positive or extra_negative):
        plans.append(('context_box_excl', positive + extra_positive, negative + extra_negative, context_bounds))
    output = args.report / f'segmentation_{time.time_ns()}_{uuid.uuid4().hex[:8]}'
    output.mkdir(parents=True)
    metadata = dict(completed=False, photo=str(args.photo), seed=str(args.seed), exclusion=str(args.exclusion),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    model=str(args.model), device='cpu', points=args.points, band=args.band,
                    iterations=args.iterations, seed_coverage=float(seed.mean()),
                    single_output=args.single_output, context_mask=str(args.context_mask),
                    context_box=args.context_box, positive_heat=str(args.positive_heat),
                    negative_mask=str(args.negative_mask),
                    weights_sha256=hashlib.sha256((args.model/'model.safetensors').read_bytes()).hexdigest(),
                    production_modified=False, variants=[])
    Image.fromarray(seed.astype(np.uint8)*255).save(output/'seed.png')
    for name, pos, neg, bounds in plans:
        points, labels = pos+neg, [1]*len(pos)+[0]*len(neg)
        kwargs = dict(images=photo, input_points=[[points]], input_labels=[[labels]], return_tensors='pt')
        if bounds is not None:
            kwargs['input_boxes'] = [[bounds]]
        inputs = processor(**kwargs)
        started = time.monotonic()
        with torch.inference_mode():
            result = model(image_embeddings=embeddings, **{key: value for key, value in inputs.items()
                           if key not in ('pixel_values', 'original_sizes')},
                           multimask_output=not args.single_output)
            masks = processor.post_process_masks(result.pred_masks, inputs['original_sizes'])[0][0].numpy()
        entries, panels = [], [rgb]
        for index, mask in enumerate(masks):
            raw = mask.astype(bool)
            for mode, chosen in [('sam', raw & ~excluded),
                                 ('grabcut', refine(rgb, raw, seed, excluded, args.band, args.iterations))]:
                filename = f'{name}_{index}_{mode}'
                Image.fromarray(chosen.astype(np.uint8)*255).save(output/f'{filename}.png')
                overlay = rgb.copy()
                overlay[chosen] = (overlay[chosen]*.55 + np.array([255, 0, 0])*.45).astype(np.uint8)
                panels.append(overlay)
                entries.append(dict(index=index, mode=mode, predicted_iou=float(result.iou_scores[0, 0, index]),
                                    coverage=float(chosen.mean()), seed_recall=float(chosen[seed].mean()),
                                    raw_exclusion_overlap=int((raw & excluded).sum()),
                                    final_exclusion_overlap=int((chosen & excluded).sum())))
        Image.fromarray(np.concatenate(panels, axis=1)).save(output/f'{name}_compare.png')
        row=dict(name=name, points=points, labels=labels, box=bounds,
                 decode_refine_save_seconds=time.monotonic()-started, measurements=entries)
        metadata['variants'].append(row)
        print(name, json.dumps(row), flush=True)
        (output/'meta.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    metadata.update(completed=True,total_seconds=time.monotonic()-total_started)
    (output/'meta.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(output, flush=True)


if __name__ == '__main__':
    main()
