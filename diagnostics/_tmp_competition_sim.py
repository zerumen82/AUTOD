"""Diagnostic-only simulation of cross-label competition and alias union on saved heats.

Uses the real CLIPSeg heat arrays already captured in the multiscale evidence run
plus the sparse GT cores. No model runs, no production mutation: it only replays
thresholding/assignment rules offline to quantify candidate fixes before any
production decision. CPU, local files only.
"""
import json
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw

RUN = 'diagnostics/out/editor_selection_multiscale_20261019/1792422366041859700'
GT = 'diagnostics/out/editor_selection_gt_20261019/coffee_v1/annotations.json'
THRESHOLD = 0.4
MIN_COMPONENT = 0.002

# Which measured query belongs to which selectable object (measured inventory
# labels + the LLM alias the real run recorded; no hand-written vocabulary).
OBJECT_QUERIES = {
    'cup': ['coffee cup', 'cup'],
    'spoon': ['silver spoon'],
    'table': ['wooden table', 'table', 'wood'],
    'saucer': ['saucer'],
}

gt = json.load(open(GT, encoding='utf-8'))
size = tuple(gt['source_size'])
cores = {}
for part in gt['parts']:
    image = Image.new('L', size)
    ImageDraw.Draw(image).polygon([tuple(q) for q in part['polygon']], fill=255)
    cores[part['name']] = (part['object'], np.asarray(image) > 0)

data = json.load(open(os.path.join(RUN, 'results.json'), encoding='utf-8'))
coffee = data['photos']['coffee']['labels']

heat_by_query = {}
for label, cands in coffee.items():
    for candidate in cands:
        if candidate['method'] == 'global' and candidate.get('context_fraction') is None:
            heat_by_query[candidate['query']] = np.load(
                os.path.join(RUN, 'coffee', candidate['heat']))


def clean(binary):
    count, labels_img = cv2.connectedComponents(binary.astype(np.uint8))
    filtered = np.zeros(binary.shape, dtype=bool)
    for component in range(1, count):
        if float((labels_img == component).mean()) >= MIN_COMPONENT:
            filtered |= labels_img == component
    return filtered


def score(mask, own_object):
    own, other = {}, {}
    for name, (obj, core) in cores.items():
        hits = float((mask & core).sum()) / float(core.sum())
        (own if obj == own_object else other)[name] = round(hits, 4)
    return own, other


def report(title, masks):
    print(f'== {title}')
    for obj, mask in masks.items():
        if mask is None or not mask.any():
            print(f'  {obj:8} (sin región)')
            continue
        own, other = score(mask, obj)
        foreign = {k: v for k, v in other.items() if v >= 0.05}
        print(f'  {obj:8} px={int(mask.sum()):6d} recall={own} foreign={foreign}')


def baseline():
    masks = {}
    for obj, queries in OBJECT_QUERIES.items():
        query = queries[0]  # production uses only the inventory label
        heat = heat_by_query.get(query)
        masks[obj] = clean(heat >= THRESHOLD) if heat is not None else None
    return masks


def alias_union():
    masks = {}
    for obj, queries in OBJECT_QUERIES.items():
        stack = [heat_by_query[q] for q in queries if q in heat_by_query]
        masks[obj] = clean(np.maximum.reduce(stack) >= THRESHOLD) if stack else None
    return masks


def competition(union_masks=None):
    """Per-pixel argmax across objects; ties keep the first object's pixel."""
    source = union_masks or {obj: clean(np.maximum.reduce(
        [heat_by_query[q] for q in queries if q in heat_by_query]) >= THRESHOLD)
        for obj, queries in OBJECT_QUERIES.items()}
    objects = [obj for obj, mask in source.items() if mask is not None]
    stack = np.stack([source[obj].astype(np.float32) for obj in objects])
    winner = np.argmax(stack, axis=0)
    has_any = stack.max(axis=0) > 0
    masks = {}
    for index, obj in enumerate(objects):
        masks[obj] = clean((winner == index) & has_any)
    return masks


def heat_competition():
    """Assign each above-threshold pixel to the object whose heat is highest."""
    objects = list(OBJECT_QUERIES)
    stack = np.stack([np.maximum.reduce(
        [heat_by_query[q] for q in OBJECT_QUERIES[obj] if q in heat_by_query])
        for obj in objects])
    binary = stack.max(axis=0) >= THRESHOLD
    winner = np.argmax(stack, axis=0)
    masks = {}
    for index, obj in enumerate(objects):
        masks[obj] = clean(binary & (winner == index))
    return masks


report('BASELINE (etiqueta de inventario sola, umbral 0.4)', baseline())
report('ALIAS UNION (max de las queries medidas por objeto)', alias_union())
report('HEAT COMPETITION (argmax de calor por píxel)', heat_competition())
