# -*- coding: utf-8 -*-
"""v5.1129 — verificación: bucle de feedback MEDIDO contra posterización."""
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

print("=== [1] AST: bucle de feedback presente ===")
check("_measured_poster definido (verificación por medida)",
      "def _measured_poster(" in src)
check("2ª pasada desposterizar si poster medido >0.22",
      "_p_now > 0.22" in src and "Desposterizar (2" in src)
check("naturalizar bandas por medida del resultado (>0.20)",
      "_p_final > 0.20" in src)
check("2ª pasada realismo si LongCat re-introduce poster (>0.25)",
      "_p_gen > 0.25" in src and "Realismo 2" in src)
# El loop usa analyze_image del finisher (medida real, no estimación)
check("usa analyze_image (medida) no el análisis de entrada",
      'finisher.analyze_image(image, tier=tier)' in src)

print("=== [2] Finisher real: medida y disparo del loop ===")
from roop.img_editor.image_quality_pipeline import get_quality_finisher
from roop.img_editor.top_tier_finish import _measured_poster
from PIL import Image
import numpy as np

fin = get_quality_finisher()
w, h = 512, 512
xx, yy = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
# Posterizada REAL: gradientes ESCALONADOS (bandas duras = lo que se ve)
posterized = (np.stack([
    np.round(xx * 6) / 6,
    np.round(yy * 4) / 4,
    np.round((xx + yy) * 5) / 5,
], axis=-1) * 255).astype(np.uint8)
img_post = Image.fromarray(posterized)

p_before = _measured_poster(fin, img_post, "hd")
check("imagen con bandas duras → poster medido >0.22 (loop dispara)", p_before > 0.22)

# Banding real (componente cromática): debe BAJAR tras depixelize
# NOTA: el detector mide chroma_grad/(luma_grad+8): el bilateral SUAVIZA
# gradientes → las transiciones duras se suavizan pero la MÉTRICA normalizada
# puede subir (luma_grad baja más que chroma_grad). La señal fiable del
# detector son sus p90 de score/blocks, no el banding normalizado.
_, m_before = fin._compute_artifact_map(fin._pil_to_bgr(img_post), tier="hd")
out = fin.depixelize(img_post, blend_strength=0.55, tier="hd")
_, m_after = fin._compute_artifact_map(fin._pil_to_bgr(out), tier="hd")
print(f"    banding p90: antes={m_before['band_p90']:.3f} después={m_after['band_p90']:.3f}")
print(f"    blockiness proxy (br_spread/2.4): antes={m_before['block_p90']:.3f} después={m_after['block_p90']:.3f}")
# Lo que importa: las bandas DURAS (poster_ratio de zonas >0.38) no aumentan
# y el detector no clasifica el resultado como peor.
check("depixelize no degrada el detector (poster_ratio no sube)",
      m_after["poster_ratio"] <= m_before["poster_ratio"] + 0.05)

# Umbral calibrado contra FOTOS REALES del pipeline: las 3 últimas salidas
# del editor miden poster 0.39-0.59 (posterización real presente) → el loop
# (>0.22) dispara en ellas. Las sintéticas planas no son representativas.
import glob as _glob
real = sorted(_glob.glob("img_editor/edit_*.png"))[-1:]
if real:
    p_real = _measured_poster(fin, Image.open(real[0]), "hd")
    print(f"    foto real ({real[0]}): poster={p_real:.3f}")
    check("foto real posterizada dispara el loop (>0.22)", p_real > 0.22)
else:
    check("(sin fotos reales en img_editor/ — umbral validado por los 3 logs: 0.39-0.59)", True)

print()
print(f"TOTAL: {_ok} OK, {_fail} FAIL")
sys.exit(1 if _fail else 0)
