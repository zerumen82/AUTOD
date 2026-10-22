"""Exercise the production instruction verifier with the vision guard active.

The verifier is the second production consumer of the Moondream projector, so
it must refuse to load the model when the guard says the vision path cannot
run, instead of entering the CPU projector route that aborts the process.

Real model, real image, no mocks and no downloads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from roop.img_editor.instruction_verifier import InstructionVerifier  # noqa: E402

IMAGE = ROOT / 'venv/Lib/site-packages/skimage/data/coffee.png'
OUT = ROOT / 'diagnostics/out/vision_guard_20261020'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(IMAGE).convert('RGB')
    verifier = InstructionVerifier()
    verdict = verifier.verify(photo, 'remove the cup and keep the spoon')
    report = {'verdict': verdict}
    (OUT / 'instruction_verifier.json').write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
