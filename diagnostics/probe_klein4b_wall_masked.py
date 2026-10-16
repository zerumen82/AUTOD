#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_klein4b_wall_masked.py - Klein 4B CON mascara en la misma foto Qwen.

Misma foto (fast_1791499927.png), misma instruccion (pared gris), pero con
mascara CLIPSeg de "wall" invertida desde "person": solo se edita el fondo,
las personas quedan intactas via keep_source_pixels + noise mask.

Receta: klein_base 4B Q4 (2.6GB, cabe en 8GB), 8 pasos, denoise 0.65,
keep_source_pixels=True, grow 4. Sin cambios de produccion.
"""
import os
import sys
import time
from pathlib import Path

from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1791499927.png"
OUTPUT = ROOT / "diagnostics/klein4b_wall_masked.png"
PROMPT = (
    "Change the wall color to neutral light grey. "
    "Keep the people, furniture, poses and framing exactly the same."
)


def main():
    from ui.tabs import comfy_launcher
    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    from roop.img_editor.clipseg_masker import get_clipseg_masker

    assert SOURCE.exists(), f"falta {SOURCE}"
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI en uso"

    photo = Image.open(SOURCE).convert("RGB")

    # Mascara de fondo = TODO menos personas (CLIPSeg "person").
    masker = get_clipseg_masker()
    ok, msg = masker.initialize()
    assert ok, msg
    person = masker.generate_mask(photo, "person")
    assert person is not None, "CLIPSeg no midio personas"
    fg = np.asarray(person.convert("L")) >= 96
    print(f"[PROBE] personas cubren {fg.mean():.1%} de la foto", flush=True)
    bg = Image.fromarray((~fg).astype(np.uint8) * 255).convert("L")
    masker.unload()

    ok, msg, _ = comfy_launcher.start(directly_run=True)
    process = comfy_launcher.comfy_process
    assert ok, msg
    try:
        client = get_flux_edit_comfy_client()
        ok, msg = client.load("flux-2-klein-base-4b-Q4_K_S.gguf")
        assert ok, msg
        t0 = time.time()
        result, msg = client.generate(
            image=photo,
            prompt=PROMPT,
            num_inference_steps=25,
            guidance_scale=4.0,
            denoise=0.65,
            mask_image=bg,
            grow_mask_by=4,
            keep_source_pixels=True,
        )
        dt = time.time() - t0
        print(f"[PROBE] status={msg} tiempo={dt:.0f}s", flush=True)
        assert result is not None, msg
        image = result.image if hasattr(result, "image") else result
        image.save(OUTPUT)
        print(f"[PROBE] output={OUTPUT} size={image.size}", flush=True)
    finally:
        process = process or comfy_launcher.comfy_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


if __name__ == "__main__":
    main()
