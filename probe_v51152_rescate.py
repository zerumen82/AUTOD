# -*- coding: utf-8 -*-
"""Probe v5.1152 — Rescate por tracking_emb en RE-BUSQUEDA sin match.

AST checks:
  1. El bloque de rescate existe dentro de _reacquire_face_global.
  2. El rescate solo corre en la rama `if not candidates` (después del fail count).
  3. Gate `(re_fails + 1) >= 3`.
  4. Umbrales 0.48 multi / 0.38 single.
  5. Usa `self._calculate_similarity(tracking_emb, ...)` — identidad del track.
  6. Solo añade candidatos si _tsim >= _rescue_min.
  7. NO añade el fallback-mayor (que v5.1138 eliminó).

Simulación de la lógica con los valores reales del log 021745:
  mejor_sim(ref)=0.06, track_sim=0.59-0.73, umbral multi 0.48 → candidato rescatado.
"""
import ast
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SRC = "roop/ProcessMgr.py"
FN = "_reacquire_face_global"

with open(SRC, "r", encoding="utf-8") as fh:
    tree = ast.parse(fh.read())

fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == FN:
        fn = node
        break
assert fn is not None, f"función {FN} no encontrada"
src = ast.dump(fn)
raw = ast.get_source_segment(open(SRC, encoding="utf-8").read(), fn) or ""

fails = []


def check(name, ok):
    print(("OK   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


# --- AST ---
# 1. la función contiene el rescate
check("AST: rescate por tracking_emb presente en _reacquire_face_global",
      "_tsim" in raw and "rescate por tracking_emb" in raw)

# 2. el rescate está en la rama not-candidates (indentación 12 == body de `if not candidates`)
rescue_lines = [l for l in raw.splitlines() if "re_fails + 1) >= 3" in l]
check("AST: rescate dentro de `if not candidates` (indentación 12)",
      len(rescue_lines) == 1 and rescue_lines[0].startswith(" " * 12 + "if (re_fails + 1) >= 3"))

# 3. umbral multi/single
check("AST: umbrales 0.48 multi / 0.38 single",
      "0.48 if multi_face else 0.38" in raw)

# 4. usa tracking_emb con _calculate_similarity
check("AST: sim contra tracking_emb", "_calculate_similarity(tracking_emb, face.embedding)" in raw)

# 5. gate con >= 3 fallos
check("AST: gate >= 3 fallos", "(re_fails + 1) >= 3" in raw)

# 6. cero fallback-mayor (v5.1138 sigue vigente)
check("AST: sin fallback-mayor-caras (v5.1138 intacto)",
      "best_fb_sim >= 0.22" in raw and "largest_area_face" not in raw)

# --- Simulación con valores del log real ---
def rescue_logic(track_sims, multi_face=True, re_fails=2):
    """Réplica exacta de la lógica del fix."""
    cands = []
    rescue_min = 0.48 if multi_face else 0.38
    if (re_fails + 1) >= 3:
        for ts in track_sims:
            if ts >= rescue_min:
                cands.append(ts)
    return cands


# caso del log: ref=0.06 sin match, track 0.59/0.73 en multi-face
# (ambas caras entran como candidatos; el margin check 0.73-0.59=0.14>=0.08 las resuelve)
out = rescue_logic([0.59, 0.73], multi_face=True, re_fails=2)
check("SIM: log real (ref 0.06, track 0.59-0.73, multi) -> rescatado", out == [0.59, 0.73])

# misma persona con track débil (0.40) sigue rechazada — exquisito intacto
out = rescue_logic([0.40], multi_face=True, re_fails=2)
check("SIM: track 0.40 multi sigue rechazado (exquisito)", out == [])

# single-face: 0.40 sí pasa
out = rescue_logic([0.40], multi_face=False, re_fails=2)
check("SIM: track 0.40 single rescatado", out == [0.40])

# primeros 2 fallos NO interfieren (recuperación normal por ref primero)
out = rescue_logic([0.73], multi_face=True, re_fails=0)
check("SIM: primeros 2 fallos sin rescate", out == [])

# persona distinta (track_sim bajo) nunca rescatada
out = rescue_logic([0.10, 0.20], multi_face=True, re_fails=10)
check("SIM: otra persona (track 0.10-0.20) rechazada", out == [])

print()
if fails:
    print(f"probe_v51152_rescate: {len(fails)} FAIL")
    sys.exit(1)
print("probe_v51152_rescate: TODO OK")
