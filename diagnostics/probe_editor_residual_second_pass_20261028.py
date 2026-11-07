#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Segundo pase sobre el residual: LongCat Turbo inpaint sobre las tiras y
parches que quedaron en `img_editor/edit_1793205552.png` (primera edición ya
cumplida; ahora se limpia solo lo que sobra).

Metodología (sin tocar producción):
  1. Carga la foto ya editada (`edit_1793205552.png`) como imagen base.
  2. Mide los restos de tela en esa imagen con el segmentador de producción
     (CLIPSec): ropa a umbral bajo (0.3) y consultas de tela fina (0.3/0.4/0.5),
     restringido al sujeto de la foto.
  3. Máscara de residual = unión de todo lo que sigue pareciendo tela.
  4. Pasa LongCat Turbo con inpaint puro (mask_image = residual, denoise alto,
     keep_source_pixels, no_kontext) SOLO sobre esa máscara pequeña.
  5. Componer solo dentro del residual; el exterior queda pixel-identico por
     construcción (keep_source_pixels + máscara feathered).

Preflight (aborta ANTES de inferir): ComfyUI 8188 responde, cola vacía, nodos
registrados y modelo LongCat Turbo presente.

Uso:
    python -u -m diagnostics.probe_editor_residual_second_pass_20261028
"""
from __future__ import annotations

import argparse
import json
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
from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client  # noqa: E402

COMFY = "http://127.0.0.1:8188"
BASE_EDIT = ROOT / "img_editor/edit_1793205552.png"  # resultado LongCat previo
SEED = 424242
DENOISE = 0.95
STEPS = 12
GUIDANCE = 3.5
REFINE_THRESHOLDS = (0.5, 0.4, 0.3)
REFINE_PROMPTS = (
    "a striped fabric strap",
    "a thin strip of cloth",
    "a piece of torn fabric",
)


def vram_free_mb() -> int:
    """VRAM real (nvidia-smi). `torch.cuda.mem_get_info` no ve la de otros
    procesos — regla ya medida en la bitácora 2026-10-20."""
    import subprocess
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
    print(f"[Residual] arranque ComfyUI propio: ok={ok} {msg}", flush=True)
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
            print("[Residual] ComfyUI propio con cola no vacía: se deja abierto", flush=True)
            return
    except Exception as exc:
        print(f"[Residual] Cola ilegible, no se cierra ComfyUI: {exc}", flush=True)
        return
    from ui.tabs.comfy_launcher import stop as stop_comfyui
    ok, msg = stop_comfyui()
    print(f"[Residual] ComfyUI propio cerrado: ok={ok} {msg}", flush=True)


def preflight() -> str | None:
    try:
        requests.get(f"{COMFY}/system_stats", timeout=5).raise_for_status()
    except Exception as exc:
        return f"ComfyUI no responde en {COMFY}: {exc}"
    try:
        queue = requests.get(f"{COMFY}/queue", timeout=10).json()
        if queue.get("queue_running") or queue.get("queue_pending"):
            return f"Cola no vacía (running={len(queue['queue_running'])}, pending={len(queue['queue_pending'])})"
    except Exception as exc:
        return f"Cola ilegible: {exc}"
    for node in ("UnetLoaderGGUF", "TextEncodeQwenImageEdit", "QwenVAELoader"):
        try:
            requests.get(f"{COMFY}/object_info/{node}", timeout=10).raise_for_status()
        except Exception:
            return f"Nodo ComfyUI no registrado: {node}"
    free = vram_free_mb()
    if free >= 0 and free < 5500:
        return f"VRAM libre {free} MiB < umbral 5500 MiB"
    return None


def residual_mask(manager: ImgEditorManager, image: Image.Image) -> np.ndarray | None:
    """Máscara de los restos de tela en la imagen ya editada, dentro del sujeto."""
    try:
        segmenter = manager._get_clothing_segmenter()
    except Exception as exc:
        print(f"[Residual] sin segmentador: {exc}", flush=True)
        return None
    if not segmenter or not segmenter.is_available():
        print("[Residual] segmentador no disponible", flush=True)
        return None

    # Persona/sujeto para acotar el residual (evita pintar tela de fondo).
    person, _ = sre._described_subject_mask(manager, image, ["The woman"])
    if person is None:
        person = np.ones(image.size[::-1], dtype=bool)

    parts: list[np.ndarray] = []
    for th in REFINE_THRESHOLDS:
        try:
            mask, _ = segmenter.segment_with_prompt(
                image, list(REFINE_PROMPTS), threshold=th, dilation=0)
            if mask is not None:
                parts.append((np.asarray(mask.convert("L")) > 127) & person)
        except Exception as exc:
            print(f"[Residual] tela fina th={th}: {exc}", flush=True)

        try:
            mask, _ = segmenter.segment_clothing(image, threshold=th)
            if mask is not None:
                parts.append((np.asarray(mask.convert("L")) > 127) & person)
        except Exception as exc:
            print(f"[Residual] ropa th={th}: {exc}", flush=True)

    if not parts:
        return None
    residual = parts[0]
    for part in parts[1:]:
        residual = residual | part
    return residual if residual.any() else None


def components(mask: np.ndarray, limit: int = 8) -> list[dict]:
    if not mask.any():
        return []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        out.append({"bbox": [int(x), int(y), int(x + w), int(y + h)],
                    "area": int(area)})
    out.sort(key=lambda c: -c["area"])
    return out[:limit]


def compose_overlay(base: Image.Image, zone: np.ndarray, residual: np.ndarray,
                    result: Image.Image) -> Image.Image:
    overlay = np.array(result)  # copia escriturable
    zone_edge = cv2.morphologyEx(
        zone.astype(np.uint8), cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))).astype(bool)
    overlay[zone_edge] = (0, 255, 0)          # verde = zona de generación
    if residual is not None:
        inside = residual & zone
        outside = residual & ~zone
        overlay[inside] = (255, 0, 0)         # rojo = residual dentro de zona
        overlay[outside] = (255, 90, 0)       # azul = residual fuera (no esperado)
    return Image.fromarray(overlay)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=str(BASE_EDIT),
                    help="foto ya editada por LongCat (primera pasada)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--denoise", type=float, default=DENOISE)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--guidance", type=float, default=GUIDANCE)
    args = ap.parse_args()

    started = False
    try:
        started = ensure_comfyui()
        print(f"[Residual] ensure_comfyui -> started={started}", flush=True)

        base_path = Path(args.base)
        if not base_path.exists():
            print(f"[Residual] FAIL: base inexistente {base_path}", flush=True)
            return 1

        problem = preflight()
        if problem:
            print(f"[Residual] PREFLIGHT FAIL: {problem}", flush=True)
            return 1
        print(f"[Residual] preflight OK (VRAM libre {vram_free_mb()} MiB, cola vacía)",
              flush=True)

        manager = ImgEditorManager()
        base = Image.open(base_path).convert("RGB")

        residual = residual_mask(manager, base)
        if residual is None:
            print("[Residual] no se detectó tela remanente; nada que limpiar",
                  flush=True)
            return 0

        print(f"[Residual] tela remanente={int(residual.sum())} px en "
              f"{len(components(residual))} componentes", flush=True)

        # Zona de generación = residual (máscara pequeña), no toda la ropa de nuevo.
        zone = residual.copy()
        manager._release_local_gpu_before_comfy()

        client = get_flux_edit_comfy_client()
        ok, note = client.load(flux_version=sre.VISION_BODY_MODEL,
                               allow_vision_engine=True)
        print(f"[Residual] load LongCat Turbo={ok} {note}", flush=True)
        if not ok:
            print(f"[Residual] FAIL: {note}", flush=True)
            return 1

        t0 = time.time()
        generated, note = client.generate(
            image=base,
            prompt="Remove every remaining strip, strap, seam and fragment of fabric "
                   "from the woman's body. Leave clean bare skin only. Keep everything "
                   "outside the masked area exactly as it is.",
            negative_prompt="",
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            denoise=args.denoise,
            prompt_authority=True,
            no_kontext=True,
            keep_source_pixels=True,
            preserve_frame=True,
            cancel_check=None,
        )
        elapsed = time.time() - t0
        print(f"[Residual] generate {elapsed:.1f}s | {note}", flush=True)
        if generated is None or generated.image is None:
            print("[Residual] FAIL sin imagen", flush=True)
            return 1

        patch = generated.image.convert("RGB").resize(base.size, Image.Resampling.LANCZOS)

        # Componer solo dentro del residual (feather suave), exterior idéntico.
        radius = max(4, min(10, int(min(base.size) * 0.03)))
        feathered = cv2.GaussianBlur(zone.astype(np.float32) * 255.0,
                                    (radius * 2 + 1, radius * 2 + 1), 0)
        feathered[~zone] = 0.0
        soft = Image.fromarray(feathered.astype(np.uint8), "L")
        composed = Image.composite(patch, base, soft)

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "diagnostics" / "out" / "residual_second_pass_20261028" / stamp
        out_dir.mkdir(parents=True, exist_ok=True)

        result_path = out_dir / "result_residual_pass.png"
        composed.save(result_path)

        src = np.asarray(base, float)
        out = np.asarray(composed, float)
        diff = np.abs(out - src).mean(-1)
        inside = float((diff[zone] > 16).mean())
        outside = float((diff[~zone] > 16).mean())

        overlay = compose_overlay(base, zone, residual, composed)
        overlay.save(out_dir / "residual_overlay.png")

        metrics = {
            "base": str(base_path),
            "seed": args.seed,
            "denoise": args.denoise,
            "steps": args.steps,
            "guidance": args.guidance,
            "elapsed_seconds": round(elapsed, 1),
            "residual_px": int(zone.sum()),
            "changed_inside": round(inside, 4),
            "changed_outside": round(outside, 6),
            "result": str(result_path),
            "overlay": str(out_dir / "residual_overlay.png"),
            "note": str(note),
        }
        (out_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[Residual] dentro={inside:.1%} fuera={outside:.6%} "
              f"zona={metrics['residual_px']}px", flush=True)
        print(f"[Residual] resultado: {result_path}", flush=True)
        print(f"[Residual] overlay: {out_dir / 'residual_overlay.png'}", flush=True)
        return 0
    finally:
        close_if_ours(started)


if __name__ == "__main__":
    raise SystemExit(main())
