"""Find final batch photos that changed little from their uploaded input."""
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
START = datetime(2026, 9, 27, 1, 28).timestamp()
END = datetime(2026, 9, 27, 1, 29).timestamp()
inputs = {
    p.name: p for p in (ROOT / '.autodeep_temp').rglob('*')
    if p.is_file() and START <= p.stat().st_mtime < END
    and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
}
results = []
for output in (ROOT / 'output').iterdir():
    source = inputs.get(output.name)
    if source is None or not output.is_file():
        continue
    before, after = cv2.imread(str(source)), cv2.imread(str(output))
    if before is None or after is None or before.shape != after.shape:
        continue
    difference = cv2.absdiff(before, after)
    results.append((output.name, float(difference.mean()), float((np.max(difference, axis=2) > 20).mean())))

print(f'[AUDIT] matched={len(results)} uploaded={len(inputs)}')
for name, mean, fraction in sorted(results, key=lambda row: row[1])[:15]:
    print(f'[AUDIT] {name}: mean_change={mean:.2f} pixels_over_20={fraction:.2%}')
