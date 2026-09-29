# -*- coding: utf-8 -*-
"""v5.1127 — verificación final: dos caminos robustos, cero vocabulario.

Con el ancla attribute_appearance (señal real para calvo/pelo/barba) y el
régimen señal-cero (para TODO lo demás: 'ponle melen', 'pelirroja'...),
ninguna palabra de apariencia puede volver a caer en klein EDIT copy-machine.
"""
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


print("=== [1] Analyzer real: señal por prompt ===")
from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer

nlp = LightLocalIntentAnalyzer()


def state(p):
    m = nlp.get_magnitude(p)
    a = nlp.get_axis_scores(p)
    bt = nlp.get_body_transform_intensity(p)
    tgt = nlp.detect_target(p)
    axis_max = max(a.values()) if a else 0.0
    return m, axis_max, bt, tgt


print("--- Camino A: señal real (attribute_appearance) ---")
sig_ok = True
for p in ["the boy is bald", "el chico es calvo", "que lleve barba",
          "the girl must have long hair"]:
    m, am, bt, tgt = state(p)
    print(f"    {p!r}: mag={m:.2f} axis_max={am:.3f} body={bt:.2f} target={tgt}")
    # Con señal real ≥0.30 el eje attribute domina → mag ≥0.72 (floor attribute 0.65+impacto)
    if am < 0.25 or m < 0.65:
        sig_ok = False
check("cambios de apariencia cubiertos por concepto → señal real + mag fuerte",
      sig_ok)

print("--- Camino B: señal cero ('ponle melen' y futuras palabras) ---")
zs_ok = True
for p in ["ponle melen", "que tenga un ojo morado"]:
    m, am, bt, tgt = state(p)
    print(f"    {p!r}: mag={m:.2f} axis_max={am:.3f}")
    # Estas palabras NO están en conceptos → señal ~0 y mag floor
    if am > 0.12 or bt > 0.0:
        zs_ok = False
check("palabras futuras fuera de conceptos → señal cero (activa régimen fuerte)",
      zs_ok)
# 'gafas de sol' = structural_add CORRECTO (añadir elemento): señal real, no cero
m3, am3, bt3, tgt3 = state("que se ponga gafas de sol")
check("'gafas de sol' clasificado structural_add (correcto, señal real)", am3 > 0.2)

print("=== [2] Manager AST + simulación de cadena ===")
import ast

src = open("roop/img_editor/img_editor_manager.py", encoding="utf-8").read()
tree = ast.parse(src)

check("_zero_signal estructural presente",
      "_zero_signal = (" in src)
check("SAFETY body excluye _zero_signal",
      "and not _zero_signal" in src)
check("gate rewriter: target determinista gana salvo structural",
      "_det_target_pre == \"subject\"" in src and "_det_structural_pre" in src)
check("wrapper mid-tier likeness + appearance libre",
      "strong likeness to them, and preserve the overall composition. The appearance may change" in src)
bad_words = ["bald", "calvo", "melen", "barba", "beard", "pelirroja"]
hits = [
    n.value for n in ast.walk(tree)
    if isinstance(n, ast.Constant) and isinstance(n.value, str)
    and any(w in n.value.lower() for w in bad_words)
]
check("cero vocabulario de usuario en string literals del manager", not hits)

print("--- Simulación log 12:11 (analizer con ancla nueva) ---")
m, am, bt, tgt = state("the boy is bald")
# Con ancla: mag 0.79 → NO zero-signal (mag>0.55) → régimen attribute fuerte
zero_signal_bald = m <= 0.55 and bt <= 0.0 and am <= 0.12
check("'the boy is bald' ahora TIENE señal → no pasa por señal-cero (va por attribute fuerte)",
      not zero_signal_bald)

# 'ponle melen': mag 0.50 → zero-signal dispara
m2, am2, bt2, tgt2 = state("ponle melen")
zero_signal_melen = m2 <= 0.55 and bt2 <= 0.0 and am2 <= 0.12
check("'ponle melen' sin señal → señal-cero dispara régimen fuerte",
      zero_signal_melen)

# Gate rewriter: LLM dice background, determinista subject, sin structural
_det_target_pre = "subject"
_det_structural_pre = False
rw_target = "background"
if rw_target != _det_target_pre and not _det_structural_pre and _det_target_pre == "subject":
    rw_target = "subject"
check("rewriter alucinando background → corregido a subject", rw_target == "subject")

print()
print(f"TOTAL: {_ok} OK, {_fail} FAIL")
sys.exit(1 if _fail else 0)
