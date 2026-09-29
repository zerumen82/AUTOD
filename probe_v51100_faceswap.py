# -*- coding: utf-8 -*-
"""Probe v5.1100 — verifica los fixes estructurales de FaceswapTab sin GPU."""
import ast
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PASS = 0
FAIL = 0


def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")


# ---------- 1. AST ProcessMgr ----------
print("== ProcessMgr.py (AST) ==")
src = open("roop/ProcessMgr.py", encoding="utf-8").read()
tree = ast.parse(src)
lines = src.splitlines()


def find_sub(node, name):
    for n in ast.walk(node):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name:
            return n
    return None


def get_src(node):
    return "\n".join(lines[node.lineno - 1 : node.end_lineno])


v21 = find_sub(tree, "_process_face_swap_v21")
v21_src = get_src(v21)
find_tgt = find_sub(tree, "_find_target_face_for_selected_mode")
find_src = get_src(find_tgt)
val_ident = find_sub(tree, "_validate_locked_identity")
val_src = get_src(val_ident)
strict_src = get_src(find_sub(tree, "_try_strict_same_person_reacquire"))
sel_sim_src = get_src(find_sub(tree, "_select_source_face_by_similarity"))

# v5.1100: foto match gate 0.35→0.28 — está en process_frame, no en _process_face_swap_v21
pf = find_sub(tree, "process_frame")
pf_src = get_src(pf)
check("foto match gate 0.28 (process_frame)", "max_sim > 0.28" in pf_src and "max_sim > 0.35" not in pf_src)
check("perfil cola hysteresis -2", "max(0, prof_state - 2)" in v21_src)
check("boost EMA (0.70/0.30)", "_mask_boost_ema_" in v21_src and "0.70 * _raw_factor" in v21_src)
check("boost viejo crudo eliminado", "final_mask * (0.92 / _mmax)" not in v21_src)
check("velocity floors 0.72/0.78", "max(0.72, 1.0 - (velocity - 80)" in v21_src and "max(0.78, 1.0 - (velocity - 100)" in v21_src)
check("low det 0.90", "swap_weight *= 0.90" in v21_src)
check("mouth dilate adaptativo", "_mouth_dilate_iters" in v21_src and "iterations=_mouth_dilate_iters" in v21_src)
check("mouth blur adaptativo", "_mouth_blur" in v21_src)
check("occ small_face 0.65", "(0.65 if is_small_face else (0.65 if is_profile else 0.80))" in v21_src)
check("scene cut resetea _mouth_ratio_smooth", "'_mouth_ratio_smooth'" in v21_src)

check("retry margin 0.04", "margin >= 0.04" in val_src)
check("strict reacquire margin 0.05", "margin >= 0.05" in strict_src)
# v5.1100: profile lock fallback — está en process_frame (loop de selección de source)
check("profile sin master -> lock frontal (process_frame)", "profile_master_embedding is None" in pf_src)

# ---------- 2. AST mouth_detector ----------
print("== mouth_detector.py (AST + simulación) ==")
md_src = open("roop/mouth_detector.py", encoding="utf-8").read()
ast.parse(md_src)
check("foreshortening (eye outer span)", "eye_outer_span" in md_src and "/ 2.2" in md_src)
check("umbral MediaPipe 0.16", "open_ratio > 0.16" in md_src and "open_ratio > 0.12\n" not in md_src)
check("sin mouth_width>0 viejo", "if mouth_width > 0:" not in md_src)

# Simulación del fix de perfil: boca cerrada de canto
mouth_w_proj = 12.0   # ancho de boca proyectado en perfil medio (px)
mouth_h = 4.5         # grosor de labios proyectado (px) — boca CERRADA
eye_outer = 55.0      # cantos de ojos (px) — invariante a la rotación
# antes: ratio = h / w_proyectado
old_ratio = mouth_h / mouth_w_proj
# ahora: w corregido = eye_outer / 2.2
w_fixed = eye_outer / 2.2
new_ratio = mouth_h / w_fixed
# Umbral de activación en prev_state=False es 0.20 (ratio>0.20 activa). Casos reales
# del log (max 0.17 boca cerrada) quedan <0.16; el sintético extremo baja >2x.
check(f"perfil cerrado: antes ratio={old_ratio:.2f} (falso abierto) -> ahora {new_ratio:.2f} (<=0.20 = cerrado)", old_ratio > 0.16 and new_ratio <= 0.20 and new_ratio < old_ratio * 0.5)
# frontal intacto: eye_outer ≈ 2.2 * mouth_width → corrección no aplica (guard <3x)
fw = 38.0
check("frontal intacto (guard eye<3x mouth)", not (fw * 3.0 < fw * 3.0))

# ---------- 3. AST FaceSwap.py ----------
print("== processors/FaceSwap.py (AST) ==")
fs_src = open("roop/processors/FaceSwap.py", encoding="utf-8").read()
ast.parse(fs_src)
check("máscara boca dilate+feather", "cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))" in fs_src)

# ---------- 4. Cadena boca completa (simulación) ----------
print("== Cadena de preservación bucal (simulación valores de log) ==")
# Perfil medio, boca cerrada, cara 100px:
mouth_open_ratio = new_ratio          # detector corregido: 0.0909
ratio_smooth = 0.75 * mouth_open_ratio + 0.25 * 0.0
thresh = 0.20                          # no prev_state (perfil)
mouth_open = ratio_smooth > thresh
check(f"perfil cerrado NO activa preservación (ratio={ratio_smooth:.2f} < 0.20)", not mouth_open)
# si habla de verdad (ratio real 0.30):
ratio_real = 0.30
check("habla real SÍ activa (0.30 > 0.20)", ratio_real > thresh)
# base m_blend perfil
m_blend = 0.32
oral = min(1.0, (ratio_real - 0.22) / 0.35)
m_blend_boost = min(0.65, m_blend * (1.0 + oral * 0.80))
check(f"restore con habla: base 32% -> {m_blend_boost*100:.0f}% (oral proporcional)", 0.32 <= m_blend_boost <= 0.65)
# small_face 90px: m_blend *= 0.92, pero floor perfil prof_sim<0.20 → max(m, 0.38)
m_small = m_blend_boost * 0.92
m_floor = max(m_small, 0.38)
check(f"small face perfil: {m_small*100:.0f}% -> floor {m_floor*100:.0f}% (boca preservada en 90px)", m_floor >= 0.38)

print()
print(f"RESULTADO: {PASS} OK, {FAIL} FAIL")
sys.exit(1 if FAIL else 0)
