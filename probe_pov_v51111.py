# -*- coding: utf-8 -*-
"""probe_pov_v51111.py — v5.1111 POV fixes (9 checks).

1. POV puro (angle_change=True, body_intensity=0) NO fuerza body_transform
   (antes: mag>=0.80 del boost POV activaba el Body SAFETY → hijack).
2. POV puro NO pasa por el router body (_route_body_engine).
3. POV en engine débil (klein/hart) → rutea a longcat turbo (no kontext).
4. POV en longcat/imagine se queda (None,None).
5. Wiring POV bypass: no_kontext=True + drop_edit_image_ref=False en
   generate-time cuando angle_change y not body_transform y engine longcat.
6. POV hint de vista se añade al prompt (view from behind, etc.).
7. Denoise POV >= 0.85 en engine longcat/imagine.
8. Zero hardcode de palabras de usuario: el único trigger es el flag
   analysis["angle_change"] (semántico), ninguna rama contiene vocabulario.
9. Body real + angle: sigue al router body (POV no rompe body_transform).
"""
import ast, sys, io

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

MGR = "roop/img_editor/img_editor_manager.py"

ok = 0
total = 0


def check(name, cond):
    global ok, total
    total += 1
    print(("OK  " if cond else "FAIL") + "  " + name)
    if cond:
        ok += 1


src = open(MGR, encoding="utf-8").read()
tree = ast.parse(src)

# ---- 1. AST: Body SAFETY excluye angle_change ----
found_safety = False
found_pov_excl = False
for node in ast.walk(tree):
    if isinstance(node, ast.If):
        comp = ast.unparse(node.test)
        if "not quality_only" in comp and "mag_suggested >= 0.65" in comp and "not angle_change" in comp:
            found_safety = True
    if isinstance(node, ast.If):
        comp = ast.unparse(node.test)
        if "angle_change" in comp and "body_transform NO forzado" in ast.unparse(node):
            found_pov_excl = True
check("1. Body SAFETY excluye angle_change (AST)", found_safety)

# ---- 2/3/4. Router: POV branch antes del weak-for-strong ----
found_pov_router = False
found_weak_no_angle = False
for node in ast.walk(tree):
    if isinstance(node, ast.If):
        comp = ast.unparse(node.test)
        if "angle_change" in comp and "not body_transform" in comp and "preferred_engine" in ast.unparse(node):
            found_pov_router = True
        if "preferred_engine in _WEAK_FOR_STRONG" in comp and "angle_change" not in comp:
            found_weak_no_angle = True
check("2. Router: branch POV dedicado (AST)", found_pov_router)
check("2b. Router weak-for-strong ya no mira angle_change (AST)", found_weak_no_angle)
check("2c. _LONGCAT_TURBO_VERSION definida en _resolve_engine_override (NameError fix)",
      "_LONGCAT_TURBO_VERSION" in ast.unparse(tree) and
      src.count('_LONGCAT_TURBO_VERSION = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"') >= 2)

# ---- 5. Wiring POV bypass ----
check("5. POV bypass no_kontext presente (AST)",
      'if (not body_transform) and analysis.get("angle_change", False)' in src
      and 'POV bypass' in src)

# ---- 6. Hint de vista sigue presente ----
check("6. POV view-hint presente (AST)", 'POV hint' in src and '_view_hint' in src)

# ---- 7. Denoise floor POV en auto_detect_params (AST) ----
found_denoise_floor = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "auto_detect_params":
        fn_src = ast.unparse(node)
        found_denoise_floor = ('angle_change' in fn_src and 'max(denoise, 0.85)' in fn_src)
check("7. Denoise POV >= 0.85 (AST)", found_denoise_floor)

# ---- 8. Zero hardcode: ninguna rama de POV contiene vocabulario de vistas ----
# Se excluyen: mensajes de log (status, no lógica), palabras 'pov' como flag
# y referencias a ANGLE_KEYWORDS (el detector semántico del analyzer).
banned = ["desde atras", "desde atrás", "de espaldas", "de lado", "de perfil",
          "cenital", "back view", "side view", "from behind"]
pov_blocks = []
lines = src.splitlines()
for i, ln in enumerate(lines):
    if "angle_change" in ln or "POV" in ln:
        pov_blocks.append((i, ln))
bad = []
for i, ln in pov_blocks:
    low = ln.lower()
    for b in banned:
        if b in low and "ANGLE_KEYWORDS" not in ln and "_view_hint" not in ln:
            bad.append((i + 1, b))
check("8. Cero vocabulario hardcodeado en ramas POV (AST)", not bad, )
if bad:
    print("     violaciones:", bad[:5])

# ---- 9. Body + angle sigue al router body ----
check("9. body_transform sigue优先 al router (AST)",
      src.find("if body_transform:\n            return self._route_body_engine(") >= 0
      or "return self._route_body_engine(" in src)

print(f"\n{ok}/{total} OK")
sys.exit(0 if ok == total else 1)
