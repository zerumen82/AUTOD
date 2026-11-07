# -*- coding: utf-8 -*-
"""Variantes reales para «sin zapatos» (GPU), 2026-11-07.

Reproduce la ruta de producción (`edit_described_subject` →
`_vision_zone_pass`) sobre la última foto del usuario que aún conserva el
calzado (`img_editor/edit_1794006695.png`, v5.1317, 3 pasadas y el zapato
sigue: 64 % / 63 % / 73 % de residuo) y mide QUÉ variante baja el residuo:

  prod              prompt de producción, sin referencia (línea base)
  cover             prompt de producción + superficie cubierta para Qwen
  endstate          prompt solo de estado final (no nombra el objeto)
  cover_endstate    ambas

La superficie cubierta usa `ImgEditorManager._covered_surface_view`, el
mismo mecanismo que ya usa la ruta de apariencia («Qwen repite la superficie
que todavía ve»). No toca la UI del usuario; usa ComfyUI si ya está en
marcha y no lo detiene.

Uso:
    python diagnostics/probe_editor_shoe_removal_variants_20261107.py
"""

from __future__ import annotations

import argparse
import json
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

SOURCE = ROOT / "img_editor" / "edit_1794006695.png"
RAW_PROMPT = "LA MUJER DEBE IR SIN ZAPATOS"
INSTRUCTION = "The woman must go without shoes"
SEED = 424242
OUT_DIR = ROOT / "diagnostics" / "out" / "shoe_removal_variants_20261107"

END_STATE_PROMPT = (
    "The feet and legs are bare: bare skin with nothing on them, resting "
    "naturally, and no translucent, faded or darkened remnant of anything "
    "that covered them before. The floor and the bench under the feet keep "
    "their own texture and lighting. The requested change must be clearly "
    "visible in the result; do not preserve the previous state of what was "
    "asked to change. Leave every other person in the photo exactly as they "
    "are."
)


def log(msg):
    print(msg, flush=True)


def comfy_alive():
    import requests
    try:
        return requests.get("http://127.0.0.1:8188/system_stats", timeout=3).status_code == 200
    except Exception:
        return False


def crop_box(zone, settings):
    """Misma caja que `_vision_zone_pass` (bbox de la zona + contexto)."""
    import math
    ys, xs = np.nonzero(zone)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    context = float(settings.get("crop_context_fraction", 0.35))
    pad_x = math.ceil((x2 - x1) * context)
    pad_y = math.ceil((y2 - y1) * context)
    return (max(0, x1 - pad_x), max(0, y1 - pad_y),
            min(zone.shape[1], x2 + pad_x), min(zone.shape[0], y2 + pad_y))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variants", default="prod,cover,endstate,cover_endstate",
        help="lista de variantes a ejecutar",
    )
    args = parser.parse_args()
    wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
    if not SOURCE.exists():
        log(f"FAIL: falta {SOURCE}")
        return 1
    if not comfy_alive():
        log("FAIL: ComfyUI no está en marcha (el probe no arranca ni detiene servicios)")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    photo = Image.open(SOURCE).convert("RGB")
    log(f"[Main] foto {photo.size}")
    manager = ImgEditorManager()
    settings = selection.load_settings()["edit"]

    person, person_query = sre._described_subject_mask(manager, photo, ["The woman"])
    if person is None:
        log("FAIL: persona no medida")
        return 1
    target = manager._requested_removal_target(RAW_PROMPT, INSTRUCTION)
    zone = sre._measured_target_zone(manager, photo, person, target)
    if zone is None:
        log(f"FAIL: objetivo «{target}» no medido")
        return 1
    support = person | zone
    box = crop_box(zone, settings)
    before = sre._measured_residual(manager, photo, zone, target)
    log(f"[Main] persona «{person_query}» {int(person.sum())} px · "
        f"objetivo «{target}» {int(zone.sum())} px · caja {box}")
    log(f"[Main] residuo ANTES: {before} px "
        f"({100.0 * before / max(1, int(zone.sum())):.0f} % de la zona)")

    prod_prompt = manager._compose_local_target_prompt(
        INSTRUCTION, target, lower_body=True)
    prompts = {"prod": prod_prompt, "endstate": END_STATE_PROMPT}

    covered_full = manager._covered_surface_view(
        photo,
        Image.fromarray(zone.astype(np.uint8) * 255, "L"),
        Image.fromarray(support.astype(np.uint8) * 255, "L"),
    )
    covered_crop = None
    if covered_full is not None:
        covered_crop = covered_full.crop(box)
        covered_crop.save(OUT_DIR / "covered_crop.png")
        log("[Main] superficie cubierta calculada para Qwen")
    else:
        log("[Main] la superficie cubierta no pudo calcularse")

    client = get_flux_edit_comfy_client()
    ok, note = client.load(
        flux_version=sre.VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        log(f"FAIL: el motor no carga: {note}")
        return 1

    recipe = {"steps": 12, "guidance": 3.5, "denoise": 1.0, "no_kontext": True}
    variants = [
        ("prod", False, False, False),
        ("cover", True, False, False),
        ("endstate", False, True, False),
        ("cover_endstate", True, True, False),
        ("nohole", False, False, True),
        ("cover_nohole", True, False, True),
    ]
    results = {}
    for name, use_cover, use_endstate, use_nohole in variants:
        if name not in wanted:
            continue
        if use_cover and covered_crop is None:
            log(f"[{name}] omitida: sin superficie cubierta")
            continue
        prompt = prompts["endstate" if use_endstate else "prod"]
        log(f"--- variante {name} (cover={use_cover}, nohole={use_nohole}) ---")
        t0 = time.time()
        manager._release_local_gpu_before_comfy()
        composed, note = sre._vision_zone_pass(
            manager, client, photo, zone, support, prompt, "", SEED,
            settings, None, None,
            recipe=recipe, label=f"variante {name}",
            is_vision_zone=True, target_query=target, verify_zone=zone,
            reference_image=covered_crop if use_cover else None,
            no_hole=use_nohole,
            auto_cover=False,
        )
        elapsed = time.time() - t0
        if composed is None:
            log(f"[{name}] FAIL: {note}")
            results[name] = {"error": str(note)}
            continue
        out_path = OUT_DIR / f"result_{name}.png"
        composed.convert("RGB").save(out_path)
        after = sre._measured_residual(manager, composed, zone, target)
        ratio = None if after is None else after / max(1, int(zone.sum()))
        failed = sre._residual_ratio_failed(after, zone)
        log(f"[{name}] residuo DESPUÉS: {after} px "
            f"({'' if ratio is None else f'{100.0 * ratio:.0f} % de la zona'}), "
            f"falla={'sí' if failed else 'no'} · {elapsed:.0f}s → {out_path.name}")
        results[name] = {
            "prompt": prompt,
            "covered": bool(use_cover),
            "no_hole": bool(use_nohole),
            "seconds": round(elapsed, 1),
            "residual_px": after,
            "residual_fraction_of_zone": None if ratio is None else round(ratio, 4),
            "residual_fail": bool(failed),
            "note": str(note),
            "result": str(out_path),
        }

    metrics_path = OUT_DIR / "metrics.json"
    previous = {}
    if metrics_path.exists():
        try:
            previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            previous = {}
    merged_results = dict(previous.get("results") or {})
    merged_results.update(results)
    report = previous
    report.update({
        "source": str(SOURCE), "target": target, "zone_px": int(zone.sum()),
        "person_px": int(person.sum()), "box": box, "residual_before_px": before,
        "results": merged_results,
    })
    metrics_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
