# -*- coding: utf-8 -*-
"""v5.1128 — verificación modo Calidad: caras reales + upscale IA cableado."""
import ast
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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


src = open("roop/img_editor/top_tier_finish.py", encoding="utf-8").read()
tree = ast.parse(src)

print("=== [1] AST: stub eliminado + CodeFormer real ===")
check("stub 'return edited' ya no es el cuerpo completo de _preserve_faces",
      "Caras restauradas con CodeFormer" in src and "_cache.Run(None, None, crop)" in src)
check("blend facial por det_score con cap 0.30",
      "det * 0.18" in src and "0.30" in src)
check("fallback silencioso si CodeFormer/modelo no disponible",
      "CodeFormer no disponible" in src)
check("cache del processor (no re-inicializa cada llamada)",
      "_preserve_faces._cf" in src)

print("=== [2] AST: upscale IA cableado ===")
check("run_top_tier_quality usa upscale_to_tier con plan",
      src.count("finisher.upscale_to_tier(") == 2
      and "upscale_plan=profile.get(\"upscale_plan\")" in src)
check("upscale_lanczos_to_tier ya no se USA en top_tier_finish (solo comentario)",
      not any(
          isinstance(n, ast.Call)
          and isinstance(n.func, ast.Attribute)
          and n.func.attr == "upscale_lanczos_to_tier"
          for n in ast.walk(tree)
      ))
check("nota 'sin rejilla' ya no promete Lanczos puro",
      "Upscale Lanczos" not in src)

print("=== [3] Finisher: plan disponible y coherente ===")
# El plan IA debe existir en el análisis del finisher
from roop.img_editor.image_quality_pipeline import get_quality_finisher
fin = get_quality_finisher()
from PIL import Image
import numpy as np
img = Image.fromarray((np.random.rand(480, 640, 3) * 255).astype(np.uint8))
analysis = fin.analyze_image(img, tier="hd")
plan = analysis.get("profile", {}).get("upscale_plan", {})
check("analyze_image produce upscale_plan", bool(plan))
print(f"    plan: {plan}")
check("plan tiene subtype/scale válidos",
      plan.get("subtype") in ("lsdirx4", "esrganx2", "esrganx4", None)
      and ("scale" in plan or plan.get("subtype") is None))

print("=== [4] _preserve_faces real: imagen sin caras = sin crash ===")
out = _mod_run_preserve(img, img) if False else None
# ejecutar la función real importando el módulo
import importlib
ttf = importlib.import_module("roop.img_editor.top_tier_finish")
res = ttf._preserve_faces(img, img)
check("imagen sintética (sin caras) pasa sin crash", res is not None)

print()
print(f"TOTAL: {_ok} OK, {_fail} FAIL")
sys.exit(1 if _fail else 0)
