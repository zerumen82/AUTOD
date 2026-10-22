"""Diagnostic-only: test a measured duplicate-resolution rule on all three saved photos.

Rule under evaluation (no word lists, no new models): when region B's mask is
mostly contained inside region A's mask AND A's own heat over B's pixels beats
B's own heat there, B is measuring A's territory with weaker evidence and is
dropped as a mislabeled duplicate. Runs offline over the saved real heat arrays;
production untouched. CPU, local files only.
"""
import json
import os
from pathlib import Path

import cv2
import numpy as np

RUN = Path('diagnostics/out/editor_selection_multiscale_20261019/1792422366041859700')
THRESHOLD = 0.4
MIN_COMPONENT = 0.002
CONTAINMENT = 0.80

# Object -> measured queries already produced by the real run (inventory label
# plus the local LLM alias). No hand-written vocabulary.
QUERIES = {
    'coffee': {'cup': ['coffee cup', 'cup'], 'spoon': ['silver spoon'],
               'table': ['wooden table', 'table', 'wood'], 'saucer': ['saucer'],
               'background': ['gray background']},
    'two_people': {'suits': ['men in suits'], 'purple_tie': ['purple tie'],
                   'yellow_tie': ['yellow tie'], 'curtain': ['blue curtain'],
                   'chandelier': ['gold chandelier']},
    'astronaut': {'woman': ['woman'], 'suit': ['spacesuit'],
                  'patch': ['nasa patch'], 'shuttle_model': ['space shuttle model']},
}


def clean(binary):
    count, labels_img = cv2.connectedComponents(binary.astype(np.uint8))
    filtered = np.zeros(binary.shape, dtype=bool)
    for component in range(1, count):
        if float((labels_img == component).mean()) >= MIN_COMPONENT:
            filtered |= labels_img == component
    return filtered


def resolve(photo):
    results = json.loads((RUN / 'results.json').read_text(encoding='utf-8'))
    entry = results['photos'][photo]
    heat_by_query = {}
    for label, cands in entry['labels'].items():
        for candidate in cands:
            if candidate['method'] == 'global' and candidate.get('context_fraction') is None:
                heat_by_query[candidate['query']] = np.load(
                    RUN / photo / candidate['heat'])
    objects = {}
    for obj, queries in QUERIES[photo].items():
        stack = [heat_by_query[q] for q in queries if q in heat_by_query]
        if not stack:
            continue
        heat = np.maximum.reduce(stack)
        mask = clean(heat >= THRESHOLD)
        if mask.any():
            objects[obj] = {'mask': mask, 'heat': heat}
    print(f'== {photo}')
    names = list(objects)
    dropped = []
    for index_b, b in enumerate(names):
        for index_a, a in enumerate(names):
            if index_a == index_b:
                continue
            mask_b, mask_a = objects[b]['mask'], objects[a]['mask']
            inside = float((mask_b & mask_a).sum()) / float(mask_b.sum())
            if inside < CONTAINMENT:
                continue
            heat_a_on_b = float(objects[a]['heat'][mask_b].mean())
            heat_b_on_b = float(objects[b]['heat'][mask_b].mean())
            verdict = ('DROP' if heat_a_on_b > heat_b_on_b else 'KEEP')
            print(f'  {b:14} inside {a:14} {inside:.3f} '
                  f'heat_own={heat_b_on_b:.3f} heat_other={heat_a_on_b:.3f} -> {verdict}')
            if verdict == 'DROP':
                dropped.append((b, a))
    surviving = [name for name in names
                 if not any(name == b for b, _ in dropped)]
    for b, a in dropped:
        print(f'  DROPPED {b} as mislabeled duplicate of {a}')
    for obj in surviving:
        mask = objects[obj]['mask']
        print(f'  {obj:14} px={int(mask.sum()):7d}')


for photo in QUERIES:
    resolve(photo)
