"""Does the reinterpretation recipe (vision engine, no hole) keep the OTHER people?

The selected body change now reinterprets the frame: the model redraws the
whole photo. This measures what that does to the people who were NOT
selected, using the two-person photo of the project corpus.

The instruction is benign on purpose (it changes the ROOM and asks for both
people to stay): what is measured is the engine's behaviour under the
reinterpretation recipe, not the content of the change.

Metrics: the mean absolute difference inside each measured head (the faces
of the two people) and over the whole frame.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
COMFY_OUT = ROOT / "ui" / "tob" / "ComfyUI" / "output"
SOURCE = ROOT / "ui" / "tob" / "ComfyUI" / "input" / "fast_1790845644.png"
PROMPT = ("Instruction: Edit this exact photo. Change the room into a bright "
          "living room with a big window. Keep both people exactly as they "
          "are: the same faces, the same hair, the same clothes and the same "
          "poses. Apply the requested change as completely, strongly and "
          "obviously as possible. The requested change must be clearly "
          "visible in the result; do not preserve the previous state of what "
          "was asked to change. Leave every other person in the photo exactly "
          "as they are.")
SEED = 424242


def newest_fastedit(after):
    candidates = [path for path in COMFY_OUT.glob("FastEdit_*.png") if path.stat().st_mtime > after]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    analysis = es.analyze_elements(photo)
    heads = [region for region in analysis.regions if region.kind == "head"]
    print(f"[Others] photo {photo.size}, measured heads: "
          f"{[(r.id, r.label, round(r.coverage, 4)) for r in heads]}", flush=True)

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    from roop.img_editor.selected_region_edit import VISION_BODY_MODEL
    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        print(f"[Others] model not available: {note}", flush=True)
        return 1
    started = time.time()
    result, note = client.generate(
        image=photo, prompt=PROMPT, negative_prompt="",
        guidance_scale=1.0, seed=SEED, num_inference_steps=12, denoise=1.0,
        prompt_authority=True, no_kontext=True, preserve_frame=False,
        mask_image=None, keep_source_pixels=False, progress_callback=None,
    )
    print(f"[Others] {time.time() - started:.1f}s: {note}", flush=True)
    if result is None or result.image is None:
        return 1
    raw = newest_fastedit(started - 1)
    if raw is not None:
        shutil.copy(raw, OUT / "others_vision_reinterpreted_raw.png")
    output = result.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    output.save(OUT / "others_vision_reinterpreted.png")
    out_arr = np.asarray(output).astype(np.int16)
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    print(f"[Others] whole frame mean delta={delta.mean():.1f} levels", flush=True)
    for region in heads:
        mask = np.asarray(region.mask.convert("L")) > 127
        if mask.shape != delta.shape or not mask.any():
            continue
        print(f"[Others] head {region.id} ({region.label}): mean delta="
              f"{float(delta[mask].mean()):.1f} levels, changed pixels="
              f"{int((delta[mask] > 8).sum())} of {int(mask.sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
