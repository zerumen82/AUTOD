# -*- coding: utf-8 -*-
"""probe_v51142_gate.py — Verificación del rescate anti-mutada v5.1142.

Contexto del log real (process_20260915_142427, Aggressivebbc):
- 76x "descartado (gigante con det bajo 0.21-0.69)" en bboxes 91-98% visibles
  → Lost 53 vs Success 88 (37% sin swap) + saltos de identidad 20%.
- El requisito det>=0.70 era para el zoom con alternancia (Sabrina 4K: gigante
  CORTADA + cara real pequeña en el MISMO frame).
"""
import ast
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

src = open("roop/ProcessMgr.py", encoding="utf-8").read()
tree = ast.parse(src)

OK, FAIL = "OK", "FALLA"
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"[{OK if cond else FAIL}] {name}")


# ---- AST: el rescate existe dentro del gate anti-mutada ----
gate_fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_find_target_face_for_selected_mode":
        gate_fn = node
        break
check("función _find_target_face_for_selected_mode existe", gate_fn is not None)

gate_src = ast.get_source_segment(src, gate_fn)
check(
    "rescate v5.1142 presente (visible>=0.85 + det>=0.30)",
    "_visible_frac >= 0.85" in gate_src and "_det >= 0.30" in gate_src,
)
check(
    "rescate exige IoU>=0.40 con la última swap (misma cara física)",
    ">= 0.40" in gate_src and "_last_swap_bbox" in gate_src,
)
check(
    "rescate también permite cara ÚNICA del frame sin historial",
    "len(valid_faces) == 1" in gate_src,
)
check(
    "el caso zoom-inflado sigue rechazado (visible<0.75 → cortada)",
    '"gigante cortada por borde del frame"' in gate_src,
)
check(
    "rechazo por det sigue existiendo cuando NO hay rescate",
    "gigante con det bajo" in gate_src,
)

# ---- Simulación funcional de la lógica del gate ----
def gate_logic(bx1, by1, bx2, by2, det, n_faces, last_bb, fw=1080, fh=1920):
    """Replica exacta de la lógica del gate (código v5.1142)."""
    bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
    ix = max(0.0, min(fw, bx2) - max(0.0, bx1))
    iy = max(0.0, min(fh, by2) - max(0.0, by1))
    vis = (ix * iy) / (bw * bh)
    if vis < 0.50:
        return False, "evita cara mutada"
    area = bw * bh
    if area <= 0.08 * fw * fh:
        return True, None  # no gigante
    if vis < 0.75:
        return False, "gigante cortada por borde del frame"
    if det < 0.70:
        rescued = False
        if det >= 0.30 and vis >= 0.85:
            if last_bb is not None:
                lx1, ly1, lx2, ly2 = last_bb
                iw = max(0.0, min(bx2, lx2) - max(bx1, lx1))
                ih = max(0.0, min(by2, ly2) - max(by1, ly1))
                inter = iw * ih
                union = (bw * bh) + max(1.0, (lx2 - lx1) * (ly2 - ly1)) - inter
                if inter / max(1.0, union) >= 0.40:
                    rescued = True
            elif n_faces == 1:
                rescued = True
        if not rescued:
            return False, f"gigante con det bajo ({det:.2f})"
    return True, None


frame_area = 1080 * 1920

# Caso 1 (del log real): closeup POV gigante 91-98% visible, det 0.36-0.63 — DEBE pasar
for det, bbox in [(0.36, (304, 140, 740, 712)), (0.62, (273, 113, 759, 725)),
                  (0.69, (224, 87, 755, 749)), (0.53, (236, 103, 753, 748))]:
    ok, why = gate_logic(bbox[0], bbox[1], bbox[2], bbox[3], det, 1, None)
    check(f"log real: gigante vis>=91% det={det} pasa (antes: rechazada)", ok)

# Caso 1b: mismo bbox pero CON historial solapado (misma cara) — pasa
ok, _ = gate_logic(304, 140, 740, 712, 0.36, 2, (300, 150, 735, 700))
check("gigante con IoU alto vs última swap pasa", ok)

# Caso 2 (Sabrina 4K): zoom-inflada 34% FUERA + cara real alternativa — sigue rechazada
# bbox=(2082,-477,3392,939) en 3840x2160, cara real 111px presente (n=2)
ok, why = gate_logic(2082, -477, 3392, 939, 0.42, 2, None, fw=3840, fh=2160)
check(f"zoom-inflada 66% visible + alternativa presente: rechazada ({why})", not ok and "cortada" in why)

# Caso 3: gigante casi-completa PERO IoU bajo con la última swap (cara distinta) y 2 caras → rechazada
ok, why = gate_logic(600, 300, 950, 800, 0.40, 2, (10, 10, 200, 300))
check(f"gigante lejos de la última swap + alternativa: rechazada ({why})", not ok)

# Caso 4: gigante casi-completa con det BASURA (0.21) → rechazada aunque sea única
ok, why = gate_logic(304, 140, 740, 712, 0.21, 1, None)
check(f"det=0.21 (<0.30) única cara: rechazada ({why})", not ok)

# Caso 5: cara pequeña normal (no gigante) — pasa igual que antes
ok, _ = gate_logic(100, 200, 200, 320, 0.40, 3, None)
check("cara normal <8% frame: pasa (intacto)", ok)

# Caso 6: cara mutada mayormente fuera — rechazada (intacto)
ok, why = gate_logic(-166, -195, 267, 325, 0.55, 2, None)
check("cara mutada 38% visible: rechazada (intacto)", not ok)

print()
passed = sum(1 for _, c in results if c)
print(f"RESULTADO: {passed}/{len(results)} OK")
sys.exit(0 if passed == len(results) else 1)
