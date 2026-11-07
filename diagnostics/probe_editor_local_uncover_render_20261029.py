# -*- coding: utf-8 -*-
"""Render real de «sin zapatos» (zona local medida), 2026-10-29.

Reproduce la ruta de producción de `edit_described_subject` para el caso del
log 17:53:

1. `_described_subject_mask` (persona) + `_requested_removal_target` («shoes»)
   + `_measured_target_zone` (medida del calzado).
2. `measured_body_regime` + `_compose_generation_prompt` (sin sufijo de
   desnudez total).
3. `_kontext_zone_pass` con la receta de producción (denoise del régimen,
   Kontext Q4, semilla 424242).

La foto de entrada conserva los zapatos originales (`edit_1793292876.png`:
los escribió el render anterior solo dentro de su máscara, que terminaba en el
tobillo). No es la foto original completa: sirve para medir si el motor quita
el calzado en la zona nueva, no como resultado para el usuario.

Uso:
    python diagnostics/probe_editor_local_uncover_render_20261029.py
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from roop.img_editor import selected_region_edit as sre  # noqa: E402
from roop.img_editor import element_selection as selection  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client  # noqa: E402
from roop.img_editor.gen_prompt_config import get_kontext_model_config  # noqa: E402

SOURCE = ROOT / "img_editor" / "edit_1793292876.png"
RAW_PROMPT = "LA MUJER DEBE IR SIN ZAPATOS"
INSTRUCTION = "The woman must go without shoes"
SEED = 424242
OUT_DIR = ROOT / "diagnostics" / "out" / "local_uncover_20261029"
COMFY_LOG = OUT_DIR / "comfyui.log"


def log(msg):
    print(msg, flush=True)


def comfy_alive():
    import requests
    try:
        return requests.get("http://127.0.0.1:8188/system_stats", timeout=3).status_code == 200
    except Exception:
        return False


def start_comfyui():
    from ui.tabs.comfy_launcher import (
        COMFYUI_DIR, COMFYUI_PYTHON, COMFYUI_TEMP_ROOT,
    )
    os.makedirs(COMFYUI_TEMP_ROOT, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["COMFYUI_DISABLE_SAFETY_CHECKER"] = "true"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    env["CUDA_MODULE_LOADING"] = "LAZY"
    python_exe = COMFYUI_PYTHON if os.path.exists(COMFYUI_PYTHON) else sys.executable
    cmd = [python_exe, "-u", os.path.join(COMFYUI_DIR, "main.py"),
           "--port", "8188", "--normalvram",
           "--temp-directory", COMFYUI_TEMP_ROOT, "--disable-async-offload"]
    handle = open(COMFY_LOG, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        cmd, cwd=COMFYUI_DIR, env=env, stdout=handle, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP |
                       subprocess.DETACHED_PROCESS |
                       subprocess.CREATE_NO_WINDOW,
    )
    log(f"[Comfy] lanzado PID={proc.pid}")
    for _ in range(120):
        if comfy_alive():
            log("[Comfy] listo")
            return proc
        time.sleep(2)
    return None


def stop_comfyui(proc):
    if proc is None:
        return
    try:
        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
    except Exception:
        pass
    for _ in range(10):
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass
    log("[Comfy] detenido (lo lanzó el probe)")


def shoe_pixels(masker, image, box):
    """Píxeles de calzado medidos por CLIPSeg dentro de la caja del caso."""
    probability = masker.probability(image, "shoes")
    if probability is None:
        return None, None
    x1, y1, x2, y2 = box
    crop = np.asarray(probability)[y1:y2, x1:x2]
    before = int((crop > 0.5).sum())
    return before, probability


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=("kontext", "vision"),
                        default="kontext")
    parser.add_argument("--state", action="store_true",
                        help="añadir la frase de estado medido (pies descalzos)")
    parser.add_argument("--source", default=str(SOURCE),
                        help="foto de entrada (por defecto, el caso del log)")
    parser.add_argument("--zone-mask", default="",
                        help="máscara ya medida del caso (mismo tamaño que --source)")
    parser.add_argument("--prompt", default="",
                        help="prompt literal (sustituye al compuesto de producción)")
    parser.add_argument("--no-hole", action="store_true",
                        help="generar el recorte entero y escribir solo la zona")
    args = parser.parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        log(f"FAIL: falta {source_path}")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    photo = Image.open(source_path).convert("RGB")
    log(f"[Main] foto {photo.size}")

    manager = ImgEditorManager()
    if args.zone_mask and Path(args.zone_mask).exists():
        target = manager._requested_removal_target(RAW_PROMPT, INSTRUCTION)
        mask = Image.open(args.zone_mask).convert("L")
        if mask.size != photo.size:
            mask = mask.resize(photo.size, Image.Resampling.NEAREST)
        zone = np.asarray(mask) > 127
        person = zone.copy()
        query = "máscara del caso"
        log(f"[Main] máscara del caso: {int(zone.sum())} px")
    else:
        person, query = sre._described_subject_mask(manager, photo, ["The woman"])
        if person is None:
            log("FAIL: persona no medida")
            return 1
        log(f"[Main] persona «{query}»: {int(person.sum())} px")
        target = manager._requested_removal_target(RAW_PROMPT, INSTRUCTION)
        zone = sre._measured_target_zone(manager, photo, person, target)
        if zone is None:
            log(f"FAIL: objetivo «{target}» no medido")
            return 1
    zys, zxs = np.nonzero(zone)
    box = (int(zxs.min()), int(zys.min()), int(zxs.max()) + 1, int(zys.max()) + 1)
    log(f"[Main] objetivo «{target}»: {int(zone.sum())} px, caja {box}")

    regime = manager.measured_body_regime(RAW_PROMPT, INSTRUCTION, engine="kontext")
    prompt = manager._compose_generation_prompt(
        INSTRUCTION, engine="kontext",
        magnitude=max(0.75, float(regime["magnitude"])),
        body_transform=True, body_direction=regime["direction"],
        body_intensity=regime["intensity"],
    )
    if regime["enforce_clause"] and regime["enforce_clause"] not in prompt:
        prompt += regime["enforce_clause"]
    if args.state:
        prompt += " The woman is barefoot, bare feet, no shoes, no footwear."
    if args.prompt:
        prompt = args.prompt
    negative = regime["negative_prompt"]
    log(f"[Main] prompt: {prompt}")

    masker = manager._get_clipseg_masker()
    shoes_before, _ = shoe_pixels(masker, photo, box)
    log(f"[Main] calzado medido ANTES en la caja: {shoes_before} px")

    proc = None
    if not comfy_alive():
        proc = start_comfyui()
        if proc is None:
            log("FAIL: ComfyUI no arrancó")
            return 1
    else:
        log("[Comfy] ya estaba en marcha (no lo lanza el probe)")

    denoise = float(regime["engine"].get("img2img_denoise", 0.90))
    denoise = max(0.80, min(0.90, denoise))
    client = get_flux_edit_comfy_client()
    km = get_kontext_model_config()
    model = sre.VISION_BODY_MODEL if args.engine == "vision" else km["model_file"]
    ok, note = client.load(flux_version=model,
                           allow_vision_engine=(args.engine == "vision"))
    if not ok:
        log(f"FAIL: el motor no carga: {note}")
        stop_comfyui(proc)
        return 1

    settings = selection.load_settings()["edit"]
    manager._release_local_gpu_before_comfy()
    t0 = time.time()
    if args.engine == "vision":
        recipe = {"steps": 12, "guidance": 3.5, "denoise": 1.0,
                  "no_kontext": True}
        composed, note = sre._vision_zone_pass(
            manager, client, photo, zone, person, prompt, negative, SEED,
            settings, None, None,
            recipe=recipe, label="motor con visión (LongCat)",
            is_vision_zone=True,
        )
    else:
        composed, note = sre._kontext_zone_pass(
            manager, client, photo, zone, person | zone, prompt, negative, SEED,
            settings, None, None, denoise=denoise,
            label="Kontext img2img puro (crop)",
            write_mask=zone if args.no_hole else None, no_hole=args.no_hole,
        )
    elapsed = time.time() - t0
    if composed is None:
        log(f"FAIL: el motor no produjo imagen: {note}")
        stop_comfyui(proc)
        return 1
    tag = args.engine + ("_state" if args.state else "")
    if args.zone_mask:
        tag += "_mask"
    if args.prompt:
        tag += "_localprompt"
    if args.no_hole:
        tag += "_nohole"
    out_path = OUT_DIR / f"result_sin_zapatos_{tag}.png"
    composed.convert("RGB").save(out_path)
    log(f"[Main] render {elapsed:.0f}s → {out_path}")

    arr_in = np.asarray(photo, dtype=np.int16)
    arr_out = np.asarray(composed.convert("RGB"), dtype=np.int16)
    diff = np.abs(arr_out - arr_in).max(axis=2)
    inside = float((diff[zone] > 16).mean())
    outside = float((diff[~zone] > 16).mean())
    log(f"[Main] cambiado dentro={inside:.1%} fuera={outside:.2%}")

    shoes_after, _ = shoe_pixels(masker, composed.convert("RGB"), box)
    log(f"[Main] calzado medido DESPUÉS en la caja: {shoes_after} px")

    metrics = {
        "photo": str(SOURCE), "instruction": INSTRUCTION,
        "target": target, "zone_box": box, "zone_px": int(zone.sum()),
        "person_px": int(person.sum()), "denoise": denoise,
        "engine": args.engine, "state_clause": bool(args.state),
        "prompt": prompt, "seconds": round(elapsed, 1),
        "changed_inside_pct": round(inside * 100, 2),
        "changed_outside_pct": round(outside * 100, 2),
        "shoes_before_px": shoes_before, "shoes_after_px": shoes_after,
        "result": str(out_path), "note": str(note),
    }
    (OUT_DIR / f"metrics_{tag}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log(json.dumps(metrics, indent=2))
    stop_comfyui(proc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
