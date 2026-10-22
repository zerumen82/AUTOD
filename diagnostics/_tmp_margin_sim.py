"""Diagnostic-only: margin numbers for a containment+dominance duplicate rule.

For every pair (B inside A) on the three saved photos, measures B's own heat vs
A's heat on B's pixels and the resulting dominance margin, to check whether a
single margin separates the mislabeled case (coffee 'saucer' = spoon bowl) from
legitimate subparts (ties on suits). Offline over saved heats; production
untouched.
"""
import json
from pathlib import Path

import numpy as np

RUN = Path('diagnostics/out/editor_selection_multiscale_20261019/1792422366041859700')
THRESHOLD = 0.4

PAIRS = [
    ('coffee', 'saucer', ['saucer'], 'spoon', ['silver spoon']),
    ('coffee', 'cup', ['coffee cup', 'cup'], 'spoon', ['silver spoon']),
    ('two_people', 'purple_tie', ['purple tie'], 'suits', ['men in suits']),
    ('two_people', 'yellow_tie', ['yellow tie'], 'suits', ['men in suits']),
    ('two_people', 'chandelier', ['gold chandelier'], 'suits', ['men in suits']),
    ('two_people', 'curtain', ['blue curtain'], 'suits', ['men in suits']),
    ('astronaut', 'patch', ['nasa patch'], 'suit', ['spacesuit']),
    ('astronaut', 'shuttle_model', ['space shuttle model'], 'suit', ['spacesuit']),
    ('astronaut', 'woman', ['woman'], 'suit', ['spacesuit']),
]

results = json.loads((RUN / 'results.json').read_text(encoding='utf-8'))
heats = {}
for photo, entry in results['photos'].items():
    heats[photo] = {}
    for label, cands in entry['labels'].items():
        for candidate in cands:
            if candidate['method'] == 'global' and candidate.get('context_fraction') is None:
                heats[photo][candidate['query']] = np.load(
                    RUN / photo / candidate['heat'])

print(f'{"photo":11} {"B":15} {"A":10} {"B px":>7} {"own":>6} {"other":>6} {"margin":>7} {"peak_own":>8}')
for photo, b, bq, a, aq in PAIRS:
    hb = np.maximum.reduce([heats[photo][q] for q in bq if q in heats[photo]])
    ha = np.maximum.reduce([heats[photo][q] for q in aq if q in heats[photo]])
    mask = hb >= THRESHOLD
    if not mask.any():
        print(f'{photo:11} {b:15} {a:10} no mask')
        continue
    own = float(hb[mask].mean())
    other = float(ha[mask].mean())
    print(f'{photo:11} {b:15} {a:10} {int(mask.sum()):7d} {own:6.3f} {other:6.3f} '
          f'{other - own:7.3f} {float(hb.max()):8.3f}')
