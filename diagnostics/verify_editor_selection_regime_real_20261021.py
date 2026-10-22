"""Real verification of the measured regime in the selected-region route.

Runs the production path on the real photo with the real instruction that
failed at 21:03 (regional selection, "la mujer esta desnuda, sin zapatos").

Steps:
  1. measure the selectable regions of the photo (production analyser);
  2. pick the object region that carries the person;
  3. run the regional route with the real Kontext client;
  4. measure the result: did the clothing change? is the exterior intact?

Nothing is downloaded. The photos are copied into diagnostics/out/.
"""
from __future__ import annotations

import shutil
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
PREVIOUS = ROOT / "img_editor" / "edit_1792523032.png"
PROMPT = "la mujer esta desnuda, sin zapatos"


def pick_person_region(analysis):
    """The object region that covers the most of the photo."""
    objects = [region for region in analysis.regions if region.kind == "object"]
    if not objects:
        return None
    return max(objects, key=lambda region: region.coverage)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    print(f"[Verify] source {photo.size} from {SOURCE.name}", flush=True)
    shutil.copy(SOURCE, OUT / "source.jpg")

    started = time.time()
    analysis = es.analyze_elements(photo)
    print(f"[Verify] analysis in {time.time() - started:.1f}s: "
          f"{[(r.id, r.kind, round(r.coverage, 4)) for r in analysis.regions]}", flush=True)
    region = pick_person_region(analysis)
    if region is None:
        print("[Verify] no measured object region; aborting", flush=True)
        return 1
    print(f"[Verify] selected {region.id} ({region.label}) coverage={region.coverage:.4f}", flush=True)

    manager = ImgEditorManager()
    started = time.time()
    output, message, mask = manager.generate_intelligent(
        image=photo, prompt=PROMPT, engine="kontext",
        selection_analysis=analysis, selected_region_ids=[region.id],
        use_semantic=True, use_rewriter=True, enhance_faces=False, auto_upscale=False,
    )
    elapsed = time.time() - started
    print(f"[Verify] route finished in {elapsed:.1f}s: {message}", flush=True)
    if output is None:
        print("[Verify] the route produced no image", flush=True)
        return 1
    output = output.convert("RGB")
    output.save(OUT / "regional_measured_regime.png")

    source_arr, output_arr = np.asarray(photo), np.asarray(output)
    support = np.asarray(es.union_mask(analysis, [region.id])) > 0
    changed = np.any(source_arr != output_arr, axis=2)
    inside = changed & support
    outside = changed & ~support
    print(f"[Verify] pixels changed: inside support={int(inside.sum())} "
          f"outside support={int(outside.sum())}", flush=True)
    print(f"[Verify] support pixels={int(support.sum())} "
          f"changed fraction inside={inside.sum() / max(1, support.sum()):.4f}", flush=True)
    if outside.any():
        print("[Verify] EXTERIOR CHANGED: the measured support was not respected", flush=True)
        return 1
    if not inside.any():
        print("[Verify] the selected region did not change at all", flush=True)
        return 1

    # The clothes are the darkest large region of the torso of the subject.
    torso = support.copy()
    ys, xs = np.where(support)
    y0, y1 = int(ys.min()), int(ys.max())
    height = max(1, y1 - y0)
    torso[:] = False
    torso[y0 + int(height * 0.25):y0 + int(height * 0.65)] = True
    torso &= support
    if torso.any():
        before = source_arr[torso].astype(np.float32)
        after = output_arr[torso].astype(np.float32)
        delta = np.abs(after - before).mean()
        print(f"[Verify] torso mean absolute change={delta:.2f} levels "
              f"(pixels={int(torso.sum())})", flush=True)

    if PREVIOUS.exists():
        previous = Image.open(PREVIOUS).convert("RGB")
        if previous.size == output.size:
            previous_arr = np.asarray(previous)
            previous_changed = np.any(previous_arr != source_arr, axis=2)
            print(f"[Verify] previous failed result changed inside support="
                  f"{int((previous_changed & support).sum())} px", flush=True)
            previous.save(OUT / "previous_failed_result.png")
    shutil.copy(SOURCE, OUT / "source_copy.jpg")
    print(f"[Verify] evidence in {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
