"""Compuerta GPU: cambio de CUERPO con default kontext por la ruta real.

El plan deja Kontext como motor por defecto y el cuerpo se queda en Kontext
(decisión del usuario 2026-10-18). El riesgo medido histórico es v5.1101:
con el ANCLA del pipeline (ref=offset) Kontext no movía la instrucción
(«ropa intacta con guidance 4.8»). La sonda de censura de hoy corrió el
cliente DIRECTO (sin ancla) y Kontext obedeció 100 %.

Esta compuerta mide el caso que falta: la misma instrucción y la misma foto
de esa sonda, pero por la ruta REAL del Editor (on_generate con motor
kontext), para ver si el ancla del pipeline devuelve «ropa intacta» o si
obedece. Guarda la salida para inspección visual.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())

from ui.tabs.img_editor.events import on_generate  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/out/gen_uncensored_20261018/lustify4.png"
OUTDIR = ROOT / "diagnostics/out/edit_body_kontext_20261018"
INSTRUCCION = (
    "remove the denim jacket and the top, leave the woman bare-chested "
    "and completely naked, same pose and same face"
)


def main():
    assert SOURCE.is_file(), SOURCE
    OUTDIR.mkdir(parents=True, exist_ok=True)
    source = Image.open(SOURCE).convert("RGB")
    start = time.monotonic()
    result, message, _ = on_generate(
        source, INSTRUCCION, "kontext", 0.0, False, 1, False, "hd",
    )
    elapsed = time.monotonic() - start
    print(f"[GATE] status={message} elapsed={elapsed:.1f}s", flush=True)
    assert result is not None, message
    out_path = OUTDIR / "kontext_body_anchor.png"
    result.save(out_path)

    a = np.asarray(source, dtype=np.int16)
    b = np.asarray(result.convert("RGB").resize(source.size), dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    changed = float((diff > 12).mean()) * 100
    print(
        f"[GATE] size={source.size}->{result.size} changed={changed:.1f}% "
        f"mean_diff={float(diff.mean()):.2f} output={out_path}",
        flush=True,
    )
    print(
        "[GATE] changed mide variación de píxeles; no demuestra cumplimiento "
        "de la instrucción. La ropa y el cuerpo requieren inspección visual.",
        flush=True,
    )


if __name__ == "__main__":
    main()
