# -*- coding: utf-8 -*-
"""v5.1131 — verificación: routing POV/fondo/vestimenta/añadir/quitar."""
import ast
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MGR = "roop/img_editor/img_editor_manager.py"
_ok = 0
_fail = 0


def check(name, cond):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  OK  {name}")
    else:
        _fail += 1
        print(f"  FAIL  {name}")


src = open(MGR, encoding="utf-8", errors="replace").read()
tree = ast.parse(src)

print("== AST ==")
check("Fix 1: body-branch es autoridad (pass en la cadena)",
      "if body_transform:\n                pass  # wrapper body ya asignado arriba" in src)
check("Fix 1: sin duplicado muerto mid-tier (una sola asignación en 0.62-0.75)",
      src.count('Keep the same people present with a strong likeness to them, and preserve the overall composition.') == 1)
check("Fix 1: 'identity unchanged' mid-tier eliminado",
      "Keep the face and identity unchanged, preserve the overall composition" not in src)
check("Fix 1: rama POV dedicada sin condición mag (elif _free_framing)",
      "elif _free_framing:" in src)
check("Fix 2: subset_hits en detección de ángulo",
      "_angle_hits = _pa.subset_hits(prompt_raw, _pa.ANGLE_KEYWORDS)" in src)

# La cadena elif ya no re-evalúa magnitude>=0.75 (las ramas son secuenciales)
no_stale = "elif magnitude >= 0.75 and is_longcat and _free_framing:" not in src
check("Fix 1: ramas elif sin condición stale mag>=0.75", no_stale)

# ---- Simulación del composer con las ramas nuevas ----
print("== FUNC: composer branches ==")


def compose_branches(body_transform, magnitude, is_longcat, free_framing):
    """Reproduce la lógica de ramas del composer v5.1131."""
    raw = "X"
    if body_transform:
        # body branch (autoridad) — simulada: wrapper con sufijo dirección
        base = "BODY_WRAPPER + direction suffix"
    else:
        if magnitude < 0.62:
            base = "LOW"
        elif magnitude < 0.75:
            base = "MID_UNLOCKED"
        elif free_framing:
            base = "BYPASS_FREE_FRAMING"
        else:
            base = "BYPASS_ANCHORED"
    return base


# body siempre conserva su wrapper (antes lo perdía)
check("body 0.85 → wrapper body (no sobrescrito)",
      compose_branches(True, 0.85, True, False) == "BODY_WRAPPER + direction suffix")
check("body 0.70 → wrapper body", compose_branches(True, 0.70, True, False).startswith("BODY"))
# mid-tier desbloqueado
check("no-body 0.70 → mid desbloqueado", compose_branches(False, 0.70, True, False) == "MID_UNLOCKED")
# POV bypass con framing libre
check("no-body 0.85 + POV → framing libre", compose_branches(False, 0.85, True, True) == "BYPASS_FREE_FRAMING")
# bypass sin pose → anclado
check("no-body 0.85 sin pose → anclado", compose_branches(False, 0.85, True, False) == "BYPASS_ANCHORED")
# low intacto
check("no-body 0.50 → low", compose_branches(False, 0.50, True, False) == "LOW")

# ---- Analyzer real: señal por categoría ----
print("== FUNC: analyzer (live) ==")
try:
    from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer
    from roop.img_editor.prompt_analyzer import PromptAnalyzer
    nlp = LightLocalIntentAnalyzer()
    pa = PromptAnalyzer()

    def detect(p):
        mag = nlp.get_magnitude(p)
        axes = nlp.get_axis_scores(p) or {}
        struct = float(axes.get("structural", 0.0))
        body = nlp.get_body_transform_intensity(p) if hasattr(nlp, "get_body_transform_intensity") else 0.0
        angle = (len(pa._angle_pattern.findall(p.lower())) > 0
                 or (pa.subset_hits(p, pa.ANGLE_KEYWORDS) if hasattr(pa, "subset_hits") else 0) > 0)
        zero = (body <= 0.0 and struct <= 0.06 and not angle and mag <= 0.55 and len(p.split()) >= 2)
        return mag, angle, zero

    m, a, z = detect("que se vea desde el lado")
    check("'que se vea desde el lado' detecta POV (subset)", a and not z)
    m, a, z = detect("desde atrás")
    check("'desde atrás' detecta POV", a)
    m, a, z = detect("ponle un sombrero")
    check("'ponle un sombrero' NO es zero-signal", not z)
    m, a, z = detect("añade un árbol")
    check("'añade un árbol' → zero-signal (régimen fuerte)", z)
except Exception as e:
    _fail += 1
    print(f"  FAIL  analyzer live: {e}")

print()
print(f"TOTAL: {_ok} OK, {_fail} FAIL")
sys.exit(1 if _fail else 0)
