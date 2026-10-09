"""Evaluate saved segmentation masks against manual diagnostic annotations.

No segmentation or generation inference. Manual IoU is approximate: refer to
annotations/README.md, especially leaves and occluded surfaces.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / 'diagnostics' / 'out' / 'editor_object_complete_20261006' / 'annotations'


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object', choices=('plant', 'chair'), required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    metadata = json.loads((args.run / 'meta.json').read_text(encoding='utf-8'))
    annotation = json.loads((ANNOTATIONS / 'annotation.json').read_text(encoding='utf-8'))
    truth = mask(ANNOTATIONS / f'{args.object}_visible_object.png')
    core = mask(ANNOTATIONS / f'{args.object}_certain_core.png')
    outer = core | mask(ANNOTATIONS / f'{args.object}_boundary_uncertainty.png')
    neighbour_name = 'plant' if args.object == 'chair' else 'chair'
    neighbour = mask(ANNOTATIONS / f'{neighbour_name}_visible_object.png')
    neighbour_core = mask(ANNOTATIONS / f'{neighbour_name}_certain_core.png')
    arm = mask(ANNOTATIONS / 'manual_occluding_arm.png')
    exclusion = mask(ROOT / metadata['exclusion'])
    parts = {part: mask(ANNOTATIONS / info['mask'])
             for part, info in annotation['objects'][args.object]['parts'].items()}
    paths = sorted(args.run.glob('*_sam.png')) + sorted(args.run.glob('*_grabcut.png'))
    paths += [args.run / 'seed.png']
    rows = []
    for path in paths:
        selected = mask(path)
        assert selected.shape == truth.shape, (path, selected.shape, truth.shape)
        intersection = int((selected & truth).sum())
        selected_pixels = int(selected.sum())
        union = int((selected | truth).sum())
        rows.append({
            'mask': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'manual_approx_iou': intersection / union if union else None,
            'manual_approx_recall': intersection / int(truth.sum()),
            'manual_approx_precision': intersection / selected_pixels if selected_pixels else None,
            'manual_certain_core_recall': float(selected[core].mean()) if core.any() else None,
            'selected_pixels': selected_pixels,
            'outside_manual_outer_pixels': int((selected & ~outer).sum()),
            'manual_part_recall': {part: float(selected[part_mask].mean()) for part, part_mask in parts.items()},
            'neighbour_object': neighbour_name,
            'manual_neighbour_overlap_pixels': int((selected & neighbour).sum()),
            'manual_neighbour_core_overlap_pixels': int((selected & neighbour_core).sum()),
            'manual_arm_overlap_pixels': int((selected & arm).sum()),
            'measured_exclusion_overlap_pixels': int((selected & exclusion).sum()),
        })
    report = {'object': args.object, 'run': str(args.run),
              'annotation_source_sha256': annotation['source_sha256'],
              'annotation_sha256': hashlib.sha256((ANNOTATIONS / 'annotation.json').read_bytes()).hexdigest(),
              'scope': 'Saved real CPU masks, independent MANUAL approximate annotations used only for evaluation, no inference',
              'rows': rows}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    for row in rows:
        if row['mask'].startswith('distributed_') or row['mask'] == 'seed.png':
            print(json.dumps(row))


if __name__ == '__main__':
    main()
