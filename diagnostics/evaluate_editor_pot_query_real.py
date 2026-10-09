"""Evaluate the alternate-query sweep for the plant pot (H-EDIT-004).

Reads runs produced by probe_editor_object_context_real.py, measures each
variant mask per annotated part (crown/trunk/pot), the whole-object IoU and
the CLIPSeg heat inside the pot region, then measures the union of the
"plant" run with each alternate-query run. Manual annotations are used only
for this evaluation; no inference runs here.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / 'diagnostics' / 'out' / 'editor_object_complete_20261006' / 'annotations'
VARIANTS = ('full', 'context_box')


def mask(path):
    return np.asarray(Image.open(path).convert('L')) > 127


def stats(selected, truth, neighbour, parts):
    inter = int((selected & truth).sum())
    union = int((selected | truth).sum())
    return dict(
        pixels=int(selected.sum()),
        iou=inter / union if union else None,
        recall=inter / int(truth.sum()),
        precision=inter / int(selected.sum()) if selected.any() else None,
        part_recall={name: float(selected[m].mean()) if m.any() else None
                     for name, m in parts.items()},
        neighbour_overlap=int((selected & neighbour).sum()),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, action='append', required=True,
                        help='output dir of probe_editor_object_context_real; repeatable')
    parser.add_argument('--plant-run', type=Path, required=True,
                        help='run whose query is "plant", used for the union')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()

    truth = mask(ANNOTATIONS / 'plant_visible_object.png')
    neighbour = mask(ANNOTATIONS / 'chair_visible_object.png')
    annotation = json.loads((ANNOTATIONS / 'annotation.json').read_text(encoding='utf-8'))
    parts = {name: mask(ANNOTATIONS / info['mask'])
             for name, info in annotation['objects']['plant']['parts'].items()}
    pot = parts['pot']

    report = {'scope': 'saved masks from probe_editor_object_context_real, '
                       'manual annotations evaluation-only, no inference',
              'runs': [], 'union_with_plant': []}
    plant_masks = {}

    def load(run, label):
        meta = json.loads((run / 'meta.json').read_text(encoding='utf-8'))
        selected = mask(run / f'{label}_mask.png')
        assert selected.shape == truth.shape, (run, label)
        return meta, selected

    for run in args.run:
        meta = json.loads((run / 'meta.json').read_text(encoding='utf-8'))
        entry = {'run': run.name, 'query': meta.get('query'), 'box': meta.get('box')}
        for label in VARIANTS:
            path = run / f'{label}_mask.png'
            if not path.is_file():
                continue
            selected = mask(path)
            row = stats(selected, truth, neighbour, parts)
            prob_path = run / f'{label}_probability.png'
            if prob_path.is_file():
                heat = np.asarray(Image.open(prob_path).convert('L')).astype(np.float32) / 255.0
                if heat.shape == pot.shape and pot.any():
                    row['pot_heat_mean'] = float(heat[pot].mean())
                    row['pot_heat_max'] = float(heat[pot].max())
            row['mask_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            entry[label] = row
            if label == 'context_box':
                plant_masks[meta.get('query')] = selected
        report['runs'].append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    plant_key = None
    for key in plant_masks:
        if key and key.strip().lower() == 'plant':
            plant_key = key
    if plant_key is None and plant_masks:
        plant_key = next(iter(plant_masks))
    plant_meta, plant_mask = load(args.plant_run, 'context_box')
    assert plant_meta.get('query') == plant_key or plant_key is None, (plant_meta.get('query'), plant_key)
    for query, alt in sorted(plant_masks.items()):
        if query == plant_key:
            continue
        union = plant_mask | alt
        row = {'query_union': f'{plant_key} + {query}'}
        row.update(stats(union, truth, neighbour, parts))
        report['union_with_plant'].append(row)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    for row in report['union_with_plant']:
        print(json.dumps(row, ensure_ascii=False))
    print(args.report, flush=True)


if __name__ == '__main__':
    main()
