# -*- coding: utf-8 -*-
"""Contratos del gate de objeto oral (v5.1338).

Contexto (medido 2026-11-12 sobre las fotos reales del usuario):
la máscara de oclusión es «textura AND no-piel», así que la BARBA y la
CAVIDAD ORAL de una boca abierta producen las mismas señales (occ/rel/ring)
que un dedo delante de la boca. De 9 fotos que dispararon el pegado de la
boca del destino al 90 %, solo UNA tenía objeto real (un dedo); las otras
ocho NO (barba o cavidad) y el resultado era «boca mal».

La señal que sí discrimina es el diff de la zona de la boca entre la ya
swapeada y la del destino:
  - objeto delante -> la zona quedó protegida -> el swap no la tocó -> diff BAJO
  - boca normal    -> el swap cambió la boca    -> diff ALTO

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_v51338_mouth_object_gate_20261112.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roop.ProcessMgr import ProcessMgr  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'OK  ' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""), flush=True)


def main():
    gate = ProcessMgr._mouth_object_confirmed

    # Caso real medido: 215543979.jpg — un DEDO delante de la boca.
    check(
        "P1: objeto real (dedo, diff 0.043) se confirma",
        gate(0.0182, 0.745, 0.342, mouth_swap_diff=0.0433),
        "occ=0.0182 rel=0.745 ring=0.342 diff=0.0433",
    )
    # Falsos positivos medidos (barba / cavidad oral).
    for name, occ, rel, ring, diff in (
        ("15605642_026 (cavidad/sorrisa)", 0.0121, 0.534, 0.913, 0.0890),
        ("487378808 (boca abierta)", 0.0231, 0.738, 0.910, 0.0956),
    ):
        check(
            f"P2: falso positivo {name} se descarta (diff {diff:.3f})",
            not gate(occ, rel, ring, mouth_swap_diff=diff),
            f"occ={occ} rel={rel} ring={ring} diff={diff:.3f}",
        )
    # Sin el diff (no medible) se mantiene el comportamiento anterior.
    check(
        "P3: sin diff medible no se bloquea el pegado",
        gate(0.0182, 0.745, 0.342, mouth_swap_diff=-1.0),
        "diff=-1",
    )
    # El eje de perfil (caso especial ya existente) sigue mandando.
    check(
        "P4: el caso de eje de perfil se mantiene",
        gate(0.002, 0.20, 0.10, axis_object=True, mouth_swap_diff=0.20),
        "axis_object=True",
    )
    # Un objeto masivo (occ > 0.035) se acepta si la zona quedó INTACTA: un
    # objeto de verdad tapa la boca, así que el swap no la toca (diff bajo).
    check(
        "P5: objeto masivo (occ 0.040) con la zona intacta se acepta",
        gate(0.040, 0.50, 0.50, mouth_swap_diff=0.03),
        "occ=0.040 diff=0.03",
    )
    # Y se descarta si el swap SÍ cambió esa zona: entonces la boca del
    # destino no estaba tapada (no hay nada que preservar), por grande que
    # sea la señal de solape.
    check(
        "P8: objeto masivo con la zona cambiada por el swap se descarta",
        not gate(0.040, 0.50, 0.50, mouth_swap_diff=0.20),
        "occ=0.040 diff=0.20",
    )
    # Sin señales de solape no hay objeto.
    check(
        "P6: sin señales de solape no se pega",
        not gate(0.0002, 0.006, 0.218, mouth_swap_diff=0.04),
        "occ=0.0002 rel=0.006",
    )
    # La zona intacta (diff muy bajo) sí puede ser un objeto.
    check(
        "P7: zona intacta (diff 0.02) con señales de solape se confirma",
        gate(0.020, 0.60, 0.40, mouth_swap_diff=0.02),
        "diff=0.02",
    )

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} contratos OK", flush=True)
    if failed:
        print("FALLAN:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
