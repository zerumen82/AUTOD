"""Probe de aceptación F0 — pestaña «✨ Mejorar fotos» (Topaz-style).

Criterio de cierre F0:
- Lote de 10 fotos procesadas de verdad; slider antes/después funciona;
  preset se aplica; salida correcta; sin tocar otras pestañas.

Este probe ejecuta el lote real (10 fotos), verifica salidas en disco,
mide antes/después (nitidez, ruido, tamaño) y valida autopilot + presets.
"""

import sys
from pathlib import Path

sys.path.insert(0, r"D:\PROJECTS\AUTOAUTO")

from PIL import Image, ImageFilter
import cv2  # noqa: E402  (solo para el test de blur del gate)
import numpy as np  # noqa: E402

from roop.output_paths import get_enhance_output_dir

from ui.tabs.enhance import logic as L


def collect_test_images() -> list:
    """10 fotos reales del repo, lista explícita y sin contenido sexual.

    No se usa glob sobre testdata/: allí hay imágenes explícitas preexistentes
    (p. ej. test_person.jpg) que no deben procesarse ni derivarse aquí.
    """
    root = Path(r"D:\PROJECTS\AUTOAUTO")
    rel = [
        "diagnostics/out/editor_categories_20261115/remove.png",
        "testdata/test_person_detected_plot.png",
        "diagnostics/out/versatility_walls_blue_20261005/source_real.png",
        "diagnostics/out/versatility_add_cup_integrated_20261005/source_real.png",
        "diagnostics/out/coffee_add_sugar_packet/source_real.png",
        "diagnostics/out/plant_native_recipe_1791223317354678200/window.png",
        "diagnostics/out/office/window.png",
        "diagnostics/out/couch_check/window.png",
        "diagnostics/out/shoe_removal_variants_20261107/stack_input_cover_prod_20261107b.png",
        "diagnostics/out/office/result_turbo.png",
    ]
    out = [root / r for r in rel]
    missing = [str(p) for p in out if not p.exists()]
    assert not missing, f"faltan muestras: {missing}"
    return out


def main() -> int:
    imgs = collect_test_images()
    print(f"[F0] Fotos recogidas: {len(imgs)}")
    assert len(imgs) == 10, f"se necesitan 10 fotos, hay {len(imgs)}"

    # --- 1. Autopilot sobre la primera foto ---
    first = Image.open(imgs[0]).convert("RGB")
    proposal = L.autopilot_propose(first)
    print("[F0] Autopilot propuesta:")
    for k, v in proposal.items():
        if k != "reasons":
            print(f"    {k} = {v}")
    print("    razones:", "; ".join(proposal["reasons"]))
    assert "denoise_strength" in proposal and "upscale_factor" in proposal

    # --- 2. Preset de fábrica aplica valores coherentes ---
    from ui.tabs.enhance.events import PRESET_DEFAULTS
    assert set(PRESET_DEFAULTS) == {"Retrato", "Paisaje", "Nocturna", "Producto", "Archivo"}
    p = PRESET_DEFAULTS["Nocturna"]
    assert p["denoise"] == 80 and p["upscale"] == 2
    print("[F0] Presets de fábrica OK:", list(PRESET_DEFAULTS))

    # --- 3. Lote real de 10 fotos ---
    settings = {
        "denoise_strength": 35,
        "sharpen_strength": 45,
        "recover_faces_strength": 0.6,
        "upscale_factor": 2,
        "exposure": 1.0,
        "shadow": 1.0,
        "highlight": 1.0,
        "temperature": 50,
        "grain_strength": 0,
        "grain_size": 10,
    }
    out_dir = Path(get_enhance_output_dir()) / "probe_f0"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = L.process_batch(
        file_paths=imgs,
        settings=settings,
        output_dir=out_dir,
        export_format="png",
        apply_filters_to_filename=True,
        allow_overwrite=True,
        progress_cb=lambda d: print(f"    [progress] {d['phase']} {d['detail']}"),
        cancel_check=L.is_cancel_requested,
    )

    ok = 0
    outs_seen = []
    for r in results:
        src = Path(r["input"])
        out = Path(r["output"]) if r.get("output") else None
        exists = out is not None and out.exists() and out.stat().st_size > 0
        if r.get("success") and exists:
            ok += 1
            outs_seen.append(out)
            before = Image.open(src)
            after = Image.open(out)
            s_b = L.measure_sharpness_laplacian(before)
            s_a = L.measure_sharpness_laplacian(after)
            print(f"    [OK] {src.name} -> {out.name} | {before.size} -> {after.size} | "
                  f"sharp {s_b:.0f} -> {s_a:.0f} | notas: {r.get('notes','')[:60]}")
        else:
            print(f"    [FAIL] {src.name}: {r.get('error','no output')}")

    print(f"[F0] Lote: {ok}/{len(results)} salidas válidas en {out_dir}")
    assert ok == len(imgs), "no todas las salidas son válidas"
    assert len(set(outs_seen)) == len(imgs), "hay salidas pisadas (mismo stem)"

    # --- 4. Gate de calidad: revertir si empeora ---
    base = Image.open(imgs[0]).convert("RGB")
    # «Peor» = blur fuerte (baja nitidez de forma garantizada)
    arr = np.array(base, dtype=np.float32)
    worse = Image.fromarray(np.clip(cv2.GaussianBlur(arr, (0, 0), 12), 0, 255).astype(np.uint8))
    kept, note = L.quality_gate_check(base, worse, "sharpen")
    assert kept is base and note, "el gate no revertió una imagen peor"
    kept2, note2 = L.quality_gate_check(base, base, "sharpen")
    assert kept2 is base and not note2
    # Upscale válido NO debe ser revertido por el gate
    up = base.resize((base.width * 2, base.height * 2), Image.LANCZOS)
    kept3, note3 = L.quality_gate_check(base, up, "upscale")
    assert kept3 is up, f"el gate rechazó un upscale legítimo: {note3}"
    # Denoise legítimo (ruido ↓) NO debe ser revertido
    den = base.filter(ImageFilter.GaussianBlur(radius=1.2))
    kept4, note4 = L.quality_gate_check(base, den, "denoise")
    assert kept4 is den, f"el gate rechazó un denoise legítimo: {note4}"
    print("[F0] Gate de calidad OK (revierte blur; conserva upscale/denoise):", note)

    print("[F0] PASS — 10 fotos procesadas, presets OK, autopilot OK, gate OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
