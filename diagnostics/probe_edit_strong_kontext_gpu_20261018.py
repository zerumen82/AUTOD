"""Compuerta GPU funcional: edicion FUERTE compuesta con default kontext.

Riesgo medido 2026-10-18: el splitter compuesto de la era LongCat divide la
instruccion en clausulas y conserva las regiones «sin mascara segura», asi
que con Kontext como default una edicion fuerte puede no llegar a su pase
global limpio (caso cuerpo: 1/3 regiones, chaqueta intacta). Ese riesgo es
GENERAL (no sexual): aplica a cualquier instruccion fuerte compuesta.

Esta compuerta usa una instruccion fuerte NO sexual, medible por color:
quitar la chaqueta vaquera y poner una blazer roja. El indicio cromatico
es la fraccion de pixeles ROJOS en la zona del torso antes/después (el
changed% de canvas completo es engañoso: mide regeneracion, no obediencia).
Captura el trace de enrutado (que motor cargo, cuantas pasadas, si el
compound splitter disparo) para diagnosticar.

Uso: con la cola de ComfyUI libre. No interrumpir renders.
"""
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, os.getcwd())

from ui.tabs.img_editor.events import on_generate  # noqa: E402
from roop.comfy_workflows import get_comfyui_url  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/out/gen_uncensored_20261018/lustify4.png"
OUTDIR = ROOT / "diagnostics/out/edit_strong_kontext_20261018"
INSTRUCCION = (
    "remove the denim jacket and put on a red leather blazer, "
    "same pose and same face"
)


def _red_fraction(img: Image.Image) -> float:
    """Fraccion de pixeles rojos dominantes (blazer roja) en la banda central."""
    w, h = img.size
    torso = img.crop((int(w * 0.33), int(h * 0.35), int(w * 0.66), int(h * 0.88)))
    a = np.asarray(torso, dtype=np.int16)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    red = (r > 100) & (r > g + 40) & (r > b + 40)
    return float(red.mean()) * 100


def main():
    assert SOURCE.is_file(), SOURCE
    session = requests.Session()
    session.trust_env = False
    queue = session.get(f"{get_comfyui_url()}/queue", timeout=5)
    queue.raise_for_status()
    state = queue.json()
    assert not state.get("queue_running") and not state.get("queue_pending"), (
        "ComfyUI ocupado; no se inicia el probe"
    )
    run_dir = OUTDIR / str(time.time_ns())
    run_dir.mkdir(parents=True, exist_ok=False)
    source = Image.open(SOURCE).convert("RGB")

    trace = io.StringIO()
    start = time.monotonic()
    with contextlib.redirect_stdout(trace):
        result, message, _ = on_generate(
            source, INSTRUCCION, "kontext", 0.0, False, 1, False, "hd",
        )
    elapsed = time.monotonic() - start
    (run_dir / "trace.log").write_text(trace.getvalue(), encoding="utf-8")

    print(f"[GATE] status={message} elapsed={elapsed:.1f}s", flush=True)
    assert result is not None, message
    out_path = run_dir / "kontext_strong_blazer.png"
    result.save(out_path)

    # --- Indicio cromatico: rojo en el torso (requiere inspeccion visual) ---
    before = _red_fraction(source)
    after = _red_fraction(result.convert("RGB"))
    print(
        f"[GATE] torso_red before={before:.1f}% after={after:.1f}% "
        f"delta={after - before:+.1f}pp output={out_path}",
        flush=True,
    )
    print(
        "[GATE] delta>5pp supera el criterio cromático de esta escena; "
        "la prenda, la pose y la cara requieren inspección visual",
        flush=True,
    )

    # --- Metrica secundaria: changed de canvas (NO mide obediencia) ---
    a = np.asarray(source, dtype=np.int16)
    b = np.asarray(result.convert("RGB").resize(source.size), dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    changed = float((diff > 12).mean()) * 100
    (run_dir / "metrics.json").write_text(json.dumps({
        "instruction": INSTRUCCION, "status": message,
        "elapsed_s": elapsed, "source": str(SOURCE),
        "torso_red_before_pct": before, "torso_red_after_pct": after,
        "canvas_changed_pct": changed,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[GATE] canvas_changed={changed:.1f}% (contexto, no obediencia)", flush=True)

    # --- Trace de enrutado ---
    keys = (
        "Enviando a ComfyUI", "multi-regi", "clausula", "cláusula",
        "sin mascara segura", "sin máscara segura", "mascara focal",
        "máscara focal", "compound", "LongCat", "kontext", "Kontext",
    )
    print("[TRACE] --- enrutado ---", flush=True)
    seen = set()
    for line in trace.getvalue().splitlines():
        if any(k in line for k in keys) and line not in seen:
            seen.add(line)
            print(f"[TRACE] {line[:160]}", flush=True)
    assert after - before > 5, (
        f"Blazer roja no medida: delta={after - before:+.1f}pp; ver {run_dir}"
    )


if __name__ == "__main__":
    main()
