"""Diagnostic-only: do any natural query variants generate evidence for the parts CLIPSeg misses?

Runs the real installed CLIPSeg (CPU) with probe phrasings chosen by inspecting the
photo; results are measurements about the model, never production routing rules and
never fed into selection. Reports per-core mean heat for every query. No downloads,
no production mutation, GPU untouched.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.clipseg_masker import CLIPSegMasker  # noqa: E402

GT = ROOT / 'diagnostics/out/editor_selection_gt_20261019/coffee_v1/annotations.json'
SOURCE = ROOT / 'diagnostics/out/editor_selection_gt_20261019/coffee_v1/source.png'
OUT = ROOT / 'diagnostics/out/editor_selection_gt_20261019/query_sweep'

QUERIES = [
    # candidates for the real saucer (red plate under cup+spoon)
    'plate', 'the plate under the cup', 'red plate', 'dish', 'red saucer plate',
    # candidates for the cup body (dark red outer wall)
    'the outside of the cup', 'cup exterior', 'red ceramic cup wall',
    # candidates for the cup handle
    'the handle of the cup', 'mug handle',
    # candidates for the rear spoon handle behind the cup
    'spoon handle behind the cup', 'spoon sticking out',
    # controls (must stay strong: known-good measured queries)
    'silver spoon', 'coffee cup', 'cup', 'wooden table', 'saucer',
]


def main():
    gt = json.loads(GT.read_text(encoding='utf-8'))
    size = tuple(gt['source_size'])
    cores = {}
    for part in gt['parts']:
        image = Image.new('L', size)
        ImageDraw.Draw(image).polygon([tuple(q) for q in part['polygon']], fill=255)
        cores[part['name']] = (part['object'], np.asarray(image) > 0)

    settings = es.load_settings()
    seg = settings['segmenter']
    import torch
    torch.set_num_threads(4)
    masker = CLIPSegMasker(backbone_path=str(Path(seg['backbone_path']).expanduser()))
    masker.device = 'cpu'
    masker._WEIGHT_CANDIDATES = [(str(ROOT / seg['weights_path']), 'uni')]
    ok, message = masker.initialize()
    if not ok:
        raise RuntimeError(message)

    source = Image.open(SOURCE).convert('RGB')
    OUT.mkdir(parents=True, exist_ok=True)
    names = list(cores)
    print(f'{"query":30} ' + ' '.join(f'{n[:10]:>10}' for n in names))
    report = {}
    try:
        for query in QUERIES:
            heat = masker.probability(source, query)
            if heat is None:
                print(f'{query:30} SIN CALOR')
                continue
            np.save(OUT / (query.replace(' ', '_') + '_heat.npy'), heat.astype(np.float32))
            row = [f'{heat[core].mean():10.3f}' for _, core in cores.values()]
            report[query] = {name: round(float(heat[core].mean()), 4)
                             for name, (_, core) in cores.items()}
            print(f'{query:30} ' + ' '.join(row), flush=True)
    finally:
        masker.model = None
    (OUT / 'query_sweep_core_heat.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print('EVIDENCE', OUT)


if __name__ == '__main__':
    main()
