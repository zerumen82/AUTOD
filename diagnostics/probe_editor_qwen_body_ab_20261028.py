#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A/B medido: Qwen-Image-Edit en la ruta corporal (zona de ropa medida).

Contexto (bitácora 2026-10-26/27): Kontext no obedece «la mujer debe ir
desnuda» en esta foto — 9 renders medidos terminan siempre en otra prenda;
LongCat-Turbo sí obedece pero degrada la zona fundida. El usuario aprueba
probar Qwen-Image-Edit (modelos ya en disco, sin descargas).

Este probe NO toca producción: reproduce la zona medida con la MISMA ruta de
producción (`_described_subject_mask` + `_measured_clothing_zone`), compone el
cuerpo del prompt con `measured_body_regime`/`_compose_generation_prompt` y
compone el resultado dentro de la máscara con el mismo feather de los A/B del
26/10, de modo que el exterior queda idéntico por construcción y la métrica
comparable con el baseline ya medido de Kontext/LongCat.

Métricas por corrida: píxeles cambiados dentro de la zona (umbral 16, mismo
que `compare_longcat.py`), fuera de la zona (objetivo 0,00 %), zona en px y
tiempo. La obediencia real (¿sale desnuda?) se juzga por inspección visual de
los PNG guardados, no por estas cifras.

Preflight (aborta ANTES de inferir): ComfyUI vivo, cola vacía, VRAM libre >=
umbral, nodos `UnetLoaderGGUF`/`TextEncodeQwenImageEdit`/`QwenVAELoader`
registrados y ficheros de modelo presentes. No mata procesos ajenos.

Uso:
    python -u -m diagnostics.probe_editor_qwen_body_ab_20261028
    python -u -m diagnostics.probe_editor_qwen_body_ab_20261028 --versions 2511
    python -u -m diagnostics.probe_editor_qwen_body_ab_20261028 --engines longcat
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from roop.img_editor import selected_region_edit as sre  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor.qwen_edit_comfy_client import get_qwen_edit_comfy_client  # noqa: E402

COMFY = "http://127.0.0.1:8188"
SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1793020935.png"
RAW_PROMPT = "la mujer debe ir desnuda"
INSTRUCTION = "The woman must go naked"
SEED = 424242
DENOISE = 0.90
STEPS = 8  # el cliente corta a 8 por diseño (euler_ancestral/simple)
CFG = 3.5
MIN_VRAM_MB = 5500
REQUIRED_NODES = ("UnetLoaderGGUF", "TextEncodeQwenImageEdit", "QwenVAELoader")
DEFAULT_VERSIONS = ("2511", "q2")


def _fail(msg: str) -> int:
    print(f"[QwenAB] PREFLIGHT FAIL: {msg}", flush=True)
    return 1


def vram_free_mb() -> int:
    """VRAM real (nvidia-smi). `torch.cuda.mem_get_info` no ve la de otros
    procesos — regla ya medida en la bitácora 2026-10-20."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20, check=True,
        )
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def ensure_comfyui(wait_s: int = 240) -> bool:
    """Si no hay ningún ComfyUI, arranca el nuestro con el launcher del
    proyecto (directly_run=True). Si ya hay uno vivo o el lanzamiento falla,
    devuelve False sin tocar nada: en ese caso el preflight dirá que no
    responde. No mata procesos ajenos."""
    try:
        requests.get(f"{COMFY}/system_stats", timeout=5).raise_for_status()
        return False  # ya había uno: no es nuestro, no lo cerramos
    except Exception:
        pass
    from ui.tabs.comfy_launcher import start as start_comfyui
    ok, msg, _port = start_comfyui(port=8188, directly_run=True)
    print(f"[QwenAB] arranque ComfyUI propio: ok={ok} {msg}", flush=True)
    if not ok:
        return False
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            requests.get(f"{COMFY}/system_stats", timeout=5).raise_for_status()
            return True
        except Exception:
            time.sleep(3)
    return False


def close_if_ours(started: bool) -> None:
    """Cierra SOLO el ComfyUI que arrancó este probe, y solo con cola vacía."""
    if not started:
        return
    try:
        queue = requests.get(f"{COMFY}/queue", timeout=10).json()
        if queue.get("queue_running") or queue.get("queue_pending"):
            print("[QwenAB] ComfyUI propio con cola no vacía: se deja abierto", flush=True)
            return
    except Exception as exc:
        print(f"[QwenAB] Cola ilegible, no se cierra ComfyUI: {exc}", flush=True)
        return
    from ui.tabs.comfy_launcher import stop as stop_comfyui
    ok, msg = stop_comfyui()
    print(f"[QwenAB] ComfyUI propio cerrado: ok={ok} {msg}", flush=True)


def preflight(versions: tuple[str, ...], min_vram_mb: int,
              required_nodes: tuple[str, ...] = REQUIRED_NODES) -> str | None:
    try:
        requests.get(f"{COMFY}/system_stats", timeout=5).raise_for_status()
    except Exception as exc:
        return f"ComfyUI no responde en {COMFY}: {exc}"

    try:
        queue = requests.get(f"{COMFY}/queue", timeout=10).json()
        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))
    except Exception as exc:
        return f"Cola ilegible: {exc}"
    if running or pending:
        return f"Cola no vacía (running={running}, pending={pending}) — esperar a que esté libre"

    free = vram_free_mb()
    if free >= 0 and free < min_vram_mb:
        return f"VRAM libre {free} MiB < umbral {min_vram_mb} MiB"

    for node in required_nodes:
        try:
            requests.get(f"{COMFY}/object_info/{node}", timeout=10).raise_for_status()
        except Exception:
            return f"Nodo ComfyUI no registrado: {node}"

    client = get_qwen_edit_comfy_client()
    for version in versions:
        ok, note = client.check_models(version)
        if not ok:
            return f"Modelos de '{version}': {note}"
    return None


def build_zone_and_prompt(manager: ImgEditorManager, photo: Image.Image):
    person, _ = sre._described_subject_mask(manager, photo, ["The woman"])
    zone_mask = np.asarray(sre._measured_clothing_zone(manager, photo, person, None)) > 0
    regime = manager.measured_body_regime(RAW_PROMPT, INSTRUCTION, engine="kontext")
    body_prompt = manager._compose_generation_prompt(
        INSTRUCTION, engine="kontext",
        magnitude=max(0.75, float(regime["magnitude"])), body_transform=True,
        body_direction=regime["direction"], body_intensity=regime["intensity"],
    )
    if regime["enforce_clause"] and regime["enforce_clause"] not in body_prompt:
        body_prompt += regime["enforce_clause"]
    body_prompt += regime["state_suffix"]
    return person, zone_mask, body_prompt, regime


def _compose(photo: Image.Image, zone_mask: np.ndarray, raw: Image.Image) -> Image.Image:
    """Misma composición de los A/B del 26/10: feather de la máscara de zona
    sobre la foto fuente → exterior idéntico por construcción."""
    radius = max(2, min(6, int(min(photo.size) * 0.006)))
    feathered = cv2.GaussianBlur(zone_mask.astype(np.float32) * 255.0,
                                 (radius * 2 + 1, radius * 2 + 1), 0)
    feathered[~zone_mask] = 0.0
    soft = Image.fromarray(feathered.astype(np.uint8), "L")
    return Image.composite(raw, photo, soft)


def _metrics(photo: Image.Image, zone_mask: np.ndarray, result: Image.Image) -> tuple[float, float]:
    src = np.asarray(photo, float)
    out = np.asarray(result, float)
    diff = np.abs(out - src).mean(-1)
    inside = float((diff[zone_mask] > 16).mean())
    outside = float((diff[~zone_mask] > 16).mean())
    return inside, outside


def run_variant(version: str, photo: Image.Image, zone_mask: np.ndarray,
                body_prompt: str, seed: int, denoise: float, steps: int,
                out_dir: Path) -> dict:
    client = get_qwen_edit_comfy_client()
    client.unload()
    ok, note = client.load(qwen_version=version)
    print(f"[QwenAB:{version}] load={ok} {note}", flush=True)
    if not ok:
        return {"version": version, "status": "load_fail", "note": note}

    started = time.time()
    generated, note = client.generate(
        image=photo,
        prompt=body_prompt,
        num_inference_steps=steps,
        guidance_scale=CFG,
        seed=seed,
        denoise=denoise,
    )
    elapsed = time.time() - started
    print(f"[QwenAB:{version}] generate {elapsed:.1f}s | {note}", flush=True)
    if generated is None or generated.image is None:
        return {"version": version, "status": "no_image", "note": str(note),
                "seconds": round(elapsed, 1)}

    raw = generated.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    raw_path = out_dir / f"raw_{version}.png"
    raw.save(raw_path)

    result = _compose(photo, zone_mask, raw)
    result_path = out_dir / f"result_{version}.png"
    result.save(result_path)

    inside, outside = _metrics(photo, zone_mask, result)

    metrics = {
        "version": version,
        "status": "ok",
        "seconds": round(elapsed, 1),
        "seed": seed,
        "denoise": denoise,
        "steps": steps,
        "cfg": CFG,
        "zone_px": int(zone_mask.sum()),
        "changed_inside": round(inside, 4),
        "changed_outside": round(outside, 6),
        "raw": str(raw_path),
        "result": str(result_path),
        "note": str(note),
    }
    print(f"[QwenAB:{version}] dentro={inside:.1%} fuera={outside:.2%} "
          f"zona={metrics['zone_px']}px", flush=True)
    return metrics


def run_longcat(manager: ImgEditorManager, photo: Image.Image, subject: np.ndarray,
                zone_mask: np.ndarray, body_prompt: str, negative: str,
                seed: int, out_dir: Path) -> dict:
    """Variante LongCat-Turbo por el MISMO camino que usa producción
    (`_vision_zone_pass`): recorte de la zona medida al presupuesto de
    referencia, máscara de la zona sin cabeza, keep_source_pixels y
    armonización de borde. Receta idéntica a producción: steps 12,
    guidance 3.5, denoise 0.95, no_kontext."""
    from roop.img_editor import element_selection as selection
    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client

    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=sre.VISION_BODY_MODEL, allow_vision_engine=True)
    print(f"[QwenAB:longcat] load={ok} {note}", flush=True)
    if not ok:
        return {"version": "longcat_turbo", "status": "load_fail", "note": note}

    settings = selection.load_settings()["edit"]
    recipe = {"steps": 12, "guidance": 3.5, "denoise": 0.95, "no_kontext": True}
    started = time.time()
    composed, note = sre._vision_zone_pass(
        manager, client, photo, zone_mask, subject, body_prompt,
        negative, seed, settings, None, None,
        recipe=recipe, label="motor con visión (LongCat Turbo) [A/B]",
        is_vision_zone=True)
    elapsed = time.time() - started
    print(f"[QwenAB:longcat] generate {elapsed:.1f}s | {note}", flush=True)
    if composed is None:
        return {"version": "longcat_turbo", "status": "no_image",
                "note": str(note), "seconds": round(elapsed, 1)}

    result = composed.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    result_path = out_dir / "result_longcat_turbo.png"
    result.save(result_path)

    inside, outside = _metrics(photo, zone_mask, result)
    metrics = {
        "version": "longcat_turbo",
        "status": "ok",
        "seconds": round(elapsed, 1),
        "seed": seed,
        "recipe": recipe,
        "zone_px": int(zone_mask.sum()),
        "changed_inside": round(inside, 4),
        "changed_outside": round(outside, 6),
        "result": str(result_path),
        "note": str(note),
    }
    print(f"[QwenAB:longcat] dentro={inside:.1%} fuera={outside:.2%} "
          f"zona={metrics['zone_px']}px", flush=True)
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--versions", nargs="+", default=list(DEFAULT_VERSIONS),
                    help="versiones qwen del cliente: 2511 q2 q3 nsfw")
    ap.add_argument("--engines", nargs="+", default=["qwen", "longcat"],
                    choices=["qwen", "longcat"],
                    help="qwen = variantes Qwen-Image-Edit; longcat = camino de "
                         "producción _vision_zone_pass (LongCat Turbo)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--denoise", type=float, default=DENOISE)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--min-vram-mb", type=int, default=MIN_VRAM_MB)
    ap.add_argument("--extra-prompt", default="",
                    help="frase que se añade al final del prompt compuesto "
                         "(solo texto de receta; no toca enrutado)")
    args = ap.parse_args()

    started_by_probe = ensure_comfyui()
    try:
        return _run(args, started_by_probe)
    finally:
        close_if_ours(started_by_probe)


def _run(args, started_by_probe: bool) -> int:
    # Solo se exigen ficheros y nodos Qwen si el motor qwen va a correr.
    check_versions = tuple(args.versions) if "qwen" in args.engines else ()
    check_nodes = REQUIRED_NODES if "qwen" in args.engines else ("UnetLoaderGGUF",)
    problem = preflight(check_versions, args.min_vram_mb, check_nodes)
    if problem:
        return _fail(problem)
    print(f"[QwenAB] preflight OK (VRAM libre {vram_free_mb()} MiB, cola vacía, "
          f"nodos y modelos presentes)", flush=True)

    if not SOURCE.exists():
        return _fail(f"fuente inexistente: {SOURCE}")
    photo = Image.open(SOURCE).convert("RGB")

    manager = ImgEditorManager()
    person, zone_mask, body_prompt, regime = build_zone_and_prompt(manager, photo)
    if args.extra_prompt.strip():
        body_prompt = f"{body_prompt} {args.extra_prompt.strip()}"
        print(f"[QwenAB] prompt extra: {args.extra_prompt.strip()}", flush=True)
    if not zone_mask.any():
        return _fail("la zona medida quedó vacía")
    print(f"[QwenAB] zona={int(zone_mask.sum())} px | prompt: {body_prompt[:200]}...",
          flush=True)

    manager._release_local_gpu_before_comfy()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "diagnostics" / "out" / "qwen_body_ab_20261028" / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    if "qwen" in args.engines:
        for version in args.versions:
            try:
                results.append(run_variant(version, photo, zone_mask, body_prompt,
                                           args.seed, args.denoise, args.steps, out_dir))
            except Exception as exc:  # un fallo de un modelo no aborta el otro
                print(f"[QwenAB:{version}] EXCEPCIÓN: {exc}", flush=True)
                results.append({"version": version, "status": "exception", "error": str(exc)})
    if "longcat" in args.engines:
        try:
            results.append(run_longcat(manager, photo, person, zone_mask, body_prompt,
                                       regime.get("negative_prompt") or "",
                                       args.seed, out_dir))
        except Exception as exc:
            print(f"[QwenAB:longcat] EXCEPCIÓN: {exc}", flush=True)
            results.append({"version": "longcat_turbo", "status": "exception", "error": str(exc)})

    payload = {
        "source": str(SOURCE),
        "raw_prompt": RAW_PROMPT,
        "instruction": INSTRUCTION,
        "prompt": body_prompt,
        "args": vars(args),
        "started_by_probe": started_by_probe,
        "vram_free_mb_at_end": vram_free_mb(),
        "results": results,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[QwenAB] métricas: {out_dir / 'metrics.json'}", flush=True)

    ok_count = sum(1 for r in results if r.get("status") == "ok")
    print(f"[QwenAB] FIN {ok_count}/{len(results)} corridas con imagen "
          f"(la obediencia se juzga por inspección visual de los PNG)",
          flush=True)
    return 0 if ok_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
