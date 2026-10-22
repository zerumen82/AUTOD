"""Measure the regional body change with fixed seeds, for a fair comparison.

A single render is not evidence: the model varies between runs. This runs
the production regional route with two fixed seeds and reports, for each, the
measured change of the torso and whether the photo outside the selection is
untouched.

Nothing is downloaded.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"
PROMPT = "la mujer esta desnuda, sin zapatos"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    analysis = es.analyze_elements(photo)
    objects = [region for region in analysis.regions if region.kind == "object"]
    region = max(objects, key=lambda item: item.coverage)
    support = np.asarray(es.union_mask(analysis, [region.id])) > 0
    source_arr = np.asarray(photo)
    for seed in (424242, 7):
        manager = ImgEditorManager()
        started = time.time()
        output, message, _mask = manager.generate_intelligent(
            image=photo, prompt=PROMPT, engine="kontext",
            selection_analysis=analysis, selected_region_ids=[region.id],
            use_semantic=True, use_rewriter=True, enhance_faces=False, auto_upscale=False,
            seed=seed,
        )
        print(f"[Seed {seed}] {time.time() - started:.1f}s: {message}", flush=True)
        if output is None:
            continue
        output.convert("RGB").save(OUT / f"regional_seed{seed}.png")
        changed = np.any(source_arr != np.asarray(output), axis=2)
        height, width = source_arr.shape[0], source_arr.shape[1]
        torso = np.zeros_like(support)
        torso[int(height * 0.40):int(height * 0.60), int(width * 0.32):int(width * 0.60)] = True
        torso &= support
        delta = float(np.abs(
            source_arr[torso].astype(np.float32) - np.asarray(output)[torso].astype(np.float32)
        ).mean()) if torso.any() else 0.0
        print(f"[Seed {seed}] torso mean absolute change={delta:.1f} levels, "
              f"changed inside={int((changed & support).sum())} "
              f"outside={int((changed & ~support).sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
