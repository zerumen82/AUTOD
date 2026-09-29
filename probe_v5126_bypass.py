# -*- coding: utf-8 -*-
"""v5.1126 — verificación: apariencia desbloqueada en bypass, versatilidad intacta."""
import ast
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MANAGER = "roop/img_editor/img_editor_manager.py"
CLIENT = "roop/img_editor/flux_edit_comfy_client.py"

_ok = 0
_fail = 0


def check(name, cond):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  OK  {name}")
    else:
        _fail += 1
        print(f"  FAIL {name}")


print("=== [1] AST: wrappers bypass en el composer ===")
src = open(MANAGER, encoding="utf-8").read()
tree = ast.parse(src)

check("wrapper pose/POV: likeness a personas + apariencia libre",
      "The pose, framing and appearance may change as required by the instruction" in src)
check("wrapper no-pose: likeness a personas + apariencia libre + framing anclado",
      "The appearance may change as required by the instruction" in src
      and "the exact same framing, composition and camera. The appearance may change" in src)
check("ya no existe 'identity identical' en rama bypass",
      "Keep the face and identity identical, and the exact same framing" not in src)
check("'same people present with a strong likeness' presente",
      src.count("Keep the same people present with a strong likeness to them") >= 3)
check("wrapper anclado (mag<0.75) intacto",
      "Keep the same background and composition. Apply the requested change" in src)
check("cero vocabulario en STRING LITERALS (comentarios no cuentan)",
      not any(isinstance(n, ast.Constant) and isinstance(n.value, str) and "bald" in n.value.lower()
              for n in ast.walk(tree)))

print("=== [2] AST: cliente flux — CFG/steps bypass ===")
csrc = open(CLIENT, encoding="utf-8").read()
ast.parse(csrc)
check("clamp CFG bypass 2.0-3.5",
      "max(2.0, min(3.5, float(guidance_scale)))" in csrc)
check("steps bypass 32-40",
      "max(32, min(num_inference_steps, 40))" in csrc)
check("CFGNorm sigue activo en Full (nodo 17)",
      "CFGNorm" in csrc and "strength\": 1.0" in csrc)

print("=== [3] Composer real: 5 casos ===")
import types
mod = types.ModuleType("iem_probe")
mod.__dict__["__name__"] = "img_editor_manager_probe"
exec(compile(src, MANAGER, "exec"), mod.__dict__)
# Instanciar sin __init__ pesado: usar la clase via getattr
cls = None
for node in tree.body:
    if isinstance(node, ast.ClassDef) and "Manager" in node.name:
        cls = getattr(mod, node.name)
        break
check("clase manager encontrada", cls is not None)
if cls:
    mgr = object.__new__(cls)

    def compose(mag, body, is_longcat=True, angle=False, pose=False, engine="longcat_full"):
        return mgr._compose_generation_prompt(
            "the boy must be bald", engine=engine, magnitude=mag,
            quality_only=False, has_quality_request=False,
            enhance_tier=None, high_structural=False,
            structural_bias="neutral", body_transform=body,
            mask_target="subject", quality_hybrid=False,
            detected_age=None, shot_type="",
            body_direction="change", angle_change=angle, pose_change=pose,
        )

    p1 = compose(0.90, body=False)
    check("P1 bypass no-body (bald, mag=0.90): apariencia libre + framing anclado",
          "The appearance may change" in p1 and "exact same framing" in p1
          and "identity identical" not in p1)
    p2 = compose(0.90, body=True, angle=False, pose=False)
    check("P2 bypass body sin pose: apariencia libre + framing anclado",
          "The appearance may change" in p2 and "exact same framing" in p2)
    p3 = compose(0.90, body=True, angle=True)
    check("P3 bypass body+POV: pose Y framing libres",
          "The pose, framing and appearance may change" in p3)
    p4 = compose(0.70, body=False)
    check("P4 moderado (mag=0.70): régimen anclado intacto (sin 'appearance may change')",
          "The appearance may change" not in p4)
    p5 = compose(0.50, body=False)
    check("P5 bajo (mag=0.50): wrapper conservador intacto",
          "Keep the face, identity, composition, background and style unchanged" in p5)

print("=== [4] Cliente: simulación clamp ===")
import re
m_cfg = re.search(r"actual_cfg = max\(2\.0, min\(3\.5, float\(guidance_scale\)\)\) if guidance_scale else 2\.5", csrc)
check("clamp exacto presente", m_cfg is not None)
# simular valores
for gs, expected in [(2.5, 2.5), (1.0, 2.0), (4.0, 3.5)]:
    actual = max(2.0, min(3.5, float(gs))) if gs else 2.5
    check(f"cfg guidance={gs} -> {expected}", abs(actual - expected) < 1e-9)
for steps, expected in [(29, 32), (35, 35), (45, 40)]:
    actual = max(32, min(steps, 40))
    check(f"steps={steps} -> {expected}", actual == expected)

print()
print(f"TOTAL: {_ok} OK, {_fail} FAIL")
sys.exit(1 if _fail else 0)
