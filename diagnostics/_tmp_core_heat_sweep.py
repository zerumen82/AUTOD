"""Diagnostic-only sweep: per-core mean heat of every measured query on the coffee photo.

Reads saved heat arrays from the multiscale evidence run and the sparse GT cores.
No model runs, no production changes, no downloads. Prints a query x core table
to locate where each text query actually puts its evidence.
"""
import json
import os

import numpy as np
from PIL import Image, ImageDraw

RUN = 'diagnostics/out/editor_selection_multiscale_20261019/1792422366041859700'
GT = 'diagnostics/out/editor_selection_gt_20261019/coffee_v1/annotations.json'

gt = json.load(open(GT, encoding='utf-8'))
size = tuple(gt['source_size'])
cores = {}
for part in gt['parts']:
    image = Image.new('L', size)
    ImageDraw.Draw(image).polygon([tuple(q) for q in part['polygon']], fill=255)
    cores[part['name']] = (part['object'], np.asarray(image) > 0)

data = json.load(open(os.path.join(RUN, 'results.json'), encoding='utf-8'))
coffee = data['photos']['coffee']['labels']

records = []
for label, cands in coffee.items():
    for candidate in cands:
        records.append((candidate['query'], candidate['method'],
                        candidate.get('context_fraction'),
                        os.path.join(RUN, 'coffee', candidate['heat'])))

names = list(cores)
print(f'{"query":16} {"method":16} {"ctx":>5} ' + ' '.join(f'{n[:10]:>10}' for n in names))
seen = set()
for query, method, ctx, heat_path in records:
    key = (query, method, ctx)
    if key in seen:
        continue
    seen.add(key)
    heat = np.load(heat_path)
    row = [f'{heat[core].mean():10.3f}' for _, core in cores.values()]
    print(f'{query:16} {method:16} {str(ctx):>5} ' + ' '.join(row))
