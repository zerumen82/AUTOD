#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full validation: denoise=1.0 vs 0.95 con auto-restart de ComfyUI.

Reproducción fiel de la ruta de producción:
1. `_described_subject_mask` + `_measured_clothing_zone` (zona de ropa medida).
2. `measured_body_regime` + `_compose_generation_prompt` → prompt compuesto.
3. `_vision_zone_pass` con la receta DE PRODUCCIÓN (denoise=1.0/0.95,
   guidance=3.5, steps=12, no_kontext=True).
4. Medición de ropa residual (CLIPSeg, umbral 0.45) dentro/fuera de la zona.

Salida: `diagnostics/out/denoise_1p0_val_<timestamp>/metrics.json`.
"""
from __future__ import annotations

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

SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1793020935.png"
RAW_PROMPT = "la mujer debe ir desnuda"
INSTRUCTION = "The woman must go naked"
SEED = 424242
COMFY_LOG = ROOT / "diagnostics" / "comfyui_run_log.txt"

def build_zone_and_prompt(manager, photo):
    """Reproduce la construcción de zona y prompt que usa producción."""
    person, _ = sre._described_subject_mask(manager, photo, ["The woman"])
    zone_mask = np.asarray(sre._measured_clothing_zone(manager, photo, person, None)) > 0
    regime = manager.measured_body_regime(RAW_PROMPT, INSTRUCTION, engine="kontext")
    body_prompt = manager._compose_generation_prompt(
        INSTRUCTION, engine="kontext",
        magnitude=max(0.75, float(regime["magnitude"])),
        body_transform=True,
        body_direction=regime["direction"],
        body_intensity=regime["intensity"],
    )
    if regime.get("enforce_clause") and regime["enforce_clause"] not in body_prompt:
        body_prompt += regime["enforce_clause"]
    body_prompt += regime["state_suffix"]
    return person, zone_mask, body_prompt, regime


def _ensure_comfyui_started():
    """Start ComfyUI freshly as a detached process."""
    from ui.tabs.comfy_launcher import (
        COMFYUI_DIR, COMFYUI_PYTHON, COMFYUI_TEMP_ROOT
    )
    import requests

    exe = os.path.join(COMFYUI_DIR, "main.py")
    python_exe = COMFYUI_PYTHON if os.path.exists(COMFYUI_PYTHON) else sys.executable
    os.makedirs(COMFYUI_TEMP_ROOT, exist_ok=True)

    env = os.environ.copy()
    env["COMFYUI_DISABLE_SAFETY_CHECKER"] = "true"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
    env["CUDA_MODULE_LOADING"] = "LAZY"

    cmd = [python_exe, "-u", exe, "--port", "8188", "--normalvram",
           "--temp-directory", COMFYUI_TEMP_ROOT, "--disable-async-offload"]
    lf = open(COMFY_LOG, "w")
    proc = subprocess.Popen(
        cmd, cwd=COMFYUI_DIR, env=env,
        stdout=lf, stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP |
                       subprocess.DETACHED_PROCESS |
                       subprocess.CREATE_NO_WINDOW,
    )
    log(f"[Launch] ComfyUI PID={proc.pid}")

    for i in range(150):
        try:
            r = requests.get("http://127.0.0.1:8188/system_stats", timeout=3)
            if r.status_code == 200:
                log(f"[Launch] ComfyUI ready after {i*2}s")
                return proc
        except Exception:
            pass
        time.sleep(2)
    return None


def _kill_comfyui(proc):
    if proc is None:
        return
    try:
        os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def is_comfy_alive():
    import requests
    try:
        r = requests.get("http://127.0.0.1:8188/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def log(msg):
    print(msg, flush=True)


def measure_clothing(manager, image, zone, threshold=0.45):
    """Mide ropa residual (CLIPSeg) dentro y fuera de la zona."""
    try:
        segmenter = manager._get_clothing_segmenter()
        if not segmenter or not segmenter.is_available():
            return {"error": "segmenter not available"}
        mask_img, _ = segmenter.segment_clothing(image, threshold=threshold)
        if mask_img is None:
            return {"error": "segment_clothing returned None"}
        clothes = np.asarray(mask_img.convert("L")) > 127
        person_img, _ = sre._described_subject_mask(manager, image, ["The woman"])
        person = np.asarray(person_img) > 0
        clothes_person = clothes & person
        zone_total = int(zone.sum())
        person_minus_zone = int((person & ~zone).sum())
        inside = int((clothes_person & zone).sum())
        outside = int((clothes_person & ~zone).sum())
        return {
            "threshold": threshold, "zone_px": zone_total,
            "person_px": int(person.sum()),
            "person_minus_zone_px": person_minus_zone,
            "clothing_inside_px": inside, "clothing_outside_px": outside,
            "pct_inside": round(inside / max(zone_total, 1) * 100, 2),
            "pct_outside": round(outside / max(person_minus_zone, 1) * 100, 2),
            "clothing_person_px": int(clothes_person.sum()),
        }
    except Exception as e:
        return {"error": str(e)}


def run_one(manager, photo, person, zone_mask, body_prompt, negative,
            denoise_val, out_dir):
    """Run _vision_zone_pass for a single denoise value."""
    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=sre.VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        return {"status": "fail", "error": f"load: {note}", "denoise": denoise_val}

    settings = selection.load_settings()["edit"]
    recipe = {"steps": 12, "guidance": 3.5, "denoise": denoise_val, "no_kontext": True}
    manager._release_local_gpu_before_comfy()
    t0 = time.time()
    try:
        composed, note = sre._vision_zone_pass(
            manager, client, photo, zone_mask, person, body_prompt,
            negative, SEED, settings, None, None,
            recipe=recipe, label=f"LongCat Turbo denoise={denoise_val}",
            is_vision_zone=True,
        )
    except Exception as e:
        return {"status": "fail", "error": f"exception: {e}",
                "denoise": denoise_val, "seconds": round(time.time() - t0, 1)}
    elapsed = time.time() - t0
    if composed is None:
        return {"status": "fail", "error": note,
                "denoise": denoise_val, "seconds": round(elapsed, 1)}

    result = composed.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    result_path = out_dir / f"result_denoise{denoise_val}.png"
    composed.save(out_dir / f"raw_denoise{denoise_val}.png")
    result.save(result_path)

    cloth = measure_clothing(manager, result, zone_mask)
    src = np.asarray(photo, dtype=float)
    out = np.asarray(result, dtype=float)
    diff = np.abs(out - src).mean(-1)
    inside_changed = float((diff[zone_mask] > 16).mean())
    outside_changed = float((diff[~zone_mask] > 16).mean())

    cm = cloth or {}
    log(f"[Main] denoise={denoise_val}: {elapsed:.0f}s | "
        f"changed_in={inside_changed:.1%} | changed_out={outside_changed:.2%} | "
        f"cloth_in={cm.get('clothing_inside_px', '?')}px "
        f"({cm.get('pct_inside', '?')}%)"
        f" cloth_out={cm.get('clothing_outside_px', '?')}px "
        f"({cm.get('pct_outside', '?')}%)")

    return {
        "status": "ok", "denoise": denoise_val, "seconds": round(elapsed, 1),
        "seed": SEED, "recipe": recipe, "zone_px": int(zone_mask.sum()),
        "changed_inside_pct": round(inside_changed * 100, 2),
        "changed_outside_pct": round(outside_changed * 100, 2),
        "clothing_metrics": cloth, "result": str(result_path), "note": str(note),
    }


def main():
    if not SOURCE.exists():
        log(f"FAIL: source not found: {SOURCE}")
        return 1
    photo = Image.open(SOURCE).convert("RGB")
    log(f"[Main] source: {SOURCE} size={photo.size}")
    manager = ImgEditorManager()
    person, zone_mask, body_prompt, regime = build_zone_and_prompt(manager, photo)
    if not zone_mask.any():
        log("FAIL: zona vacía")
        return 1
    negative = regime.get("negative_prompt") or ""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "diagnostics" / "out" / f"denoise_1p0_val_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"[Main] zona={int(zone_mask.sum())} px dir={regime.get('direction')}")
    log(f"[Main] prompt: {body_prompt[:200]}...")

    results = {}
    for dval in (1.0, 0.95):
        log(f"\n{'='*60}")
        log(f"[Main] === denoise={dval} ===")
        proc = None
        if not is_comfy_alive():
            log("[Main] ComfyUI not alive, starting fresh...")
            proc = _ensure_comfyui_started()
            if proc is None:
                results[f"denoise_{dval}"] = {
                    "status": "fail", "error": "ComfyUI no iniciado",
                    "denoise": dval
                }
                continue
        result = run_one(manager, photo, person, zone_mask, body_prompt,
                         negative, dval, out_dir)
        results[f"denoise_{dval}"] = result
        if proc is not None:
            log("[Main] Restarting ComfyUI for next run...")
            _kill_comfyui(proc)
            time.sleep(5)

    report = {
        "source": str(SOURCE), "raw_prompt": RAW_PROMPT,
        "instruction": INSTRUCTION, "seed": SEED, "timestamp": stamp,
        "zone_px": int(zone_mask.sum()),
        "body_prompt": body_prompt,
        "regime_direction": regime.get("direction"),
        "regime_magnitude": regime.get("magnitude"),
        "results": results,
    }
    rp = out_dir / "metrics.json"
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\n[Main] Report: {rp}")
    log(f"[Main] Output: {out_dir}")
    log("\n" + "=" * 60)
    log("RESUMEN")
    log("=" * 60)
    for dval in (1.0, 0.95):
        r = results.get(f"denoise_{dval}", {})
        if r.get("status") == "ok":
            cm = r.get("clothing_metrics", {})
            log(f"  denoise={dval}: PASS | {r['seconds']}s | "
                f"changed_in={r['changed_inside_pct']}% "
                f"changed_out={r['changed_outside_pct']}% | "
                f"cloth_in={cm.get('clothing_inside_px', '?')}px "
                f"({cm.get('pct_inside', '?')}%) "
                f"cloth_out={cm.get('clothing_outside_px', '?')}px "
                f"({cm.get('pct_outside', '?')}%)")
        else:
            log(f"  denoise={dval}: FAIL - {r.get('error', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
