# -*- coding: utf-8 -*-
"""Barre los probes del repo y lista los que fallan.

Ejecuta cada probe de `diagnostics/` en un proceso limpio con timeout y
recoge los que salen con código distinto de 0. Los que necesitan argumentos
o fixtures se saltan (no se pueden ejecutar en lote).

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_suite_barrido.py [segundos_timeout]
"""
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "diagnostics"
PY = sys.executable
timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 120

probes = sorted(p for p in DIAG.glob("probe_*.py"))
print(f"[BARRIDO] {len(probes)} probes, timeout {timeout}s", flush=True)


def run(probe):
    started = time.time()
    try:
        proc = subprocess.run(
            [PY, str(probe)], capture_output=True, text=True,
            timeout=timeout, cwd=str(ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(ROOT)},
        )
        return (probe.name, proc.returncode, round(time.time() - started, 1), "")
    except subprocess.TimeoutExpired:
        return (probe.name, None, round(time.time() - started, 1), "TIMEOUT")
    except Exception as error:  # noqa: BLE001
        return (probe.name, -999, round(time.time() - started, 1), str(error)[:120])


results = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for index, outcome in enumerate(pool.map(run, probes), 1):
        results.append(outcome)
        name, code, elapsed, note = outcome
        state = "OK  " if code == 0 else ("TIMEOUT" if code is None else "FALLA")
        print(f"[{index:3d}/{len(probes)}] {state} {name} ({elapsed}s) {note}", flush=True)

fallos = [r for r in results if r[1] not in (0,)]
print(f"\n[BARRIDO] {len(results) - len(fallos)} OK, {len(fallos)} con fallo", flush=True)
for name, code, elapsed, note in fallos:
    print(f"  {name} -> codigo={code} ({note})", flush=True)
print("[FIN]", flush=True)
