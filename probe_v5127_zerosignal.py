# -*- coding: utf-8 -*-
"""v5.1127 — señal cero = régimen fuerte. Versatilidad total, cero vocabulario.

Valida con el ANALYZER REAL (LightLocalIntentAnalyzer) que:
1. Palabras fuera de conceptos ('calvo', 'ponle melen', inventadas) → señal 0
   en ejes → en el manager dispara _zero_signal → mag=0.90 global subject.
2. Pedidos CON señal (pose, body, structural) NO disparan _zero_signal.
3. El rewriter inestable (mag 0.1/target background del log 12:11) no puede
   abajar el régimen ni redirigir a background bajo _zero_signal.
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


print("=== [1] Analyzer real: señal de conceptos por prompt ===")
from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer

nlp = LightLocalIntentAnalyzer()


def axes_mag(p):
    m = nlp.get_magnitude(p)
    a = nlp.get_axis_scores(p)
    bt = nlp.get_body_transform_intensity(p)
    tgt = nlp.detect_target(p)
    return m, a, bt, tgt


# Prompts FUERA del vocabulario de conceptos (cualquier atributo no cubierto)
zero_prompts = [
    "the boy is bald",
    "el chico es calvo",
    "ponle melen",
    "que lleve barba",
    "el chico debe tener el pelo corto",
]
all_zero = True
for p in zero_prompts:
    m, a, bt, tgt = axes_mag(p)
    axis_max = max(a.values()) if a else 0.0
    print(f"    {p!r}: mag={m:.2f} axis_max={axis_max:.3f} body={bt:.2f} target={tgt}")
    if axis_max > 0.12 or bt > 0.0:
        all_zero = False
check("prompts fuera de conceptos → señal ~cero en ejes (mag cae al floor)",
      all_zero)

# Prompts CON señal: no deben disparar _zero_signal
signal_prompts = [
    ("de rodillas", "pose"),
    ("ella va desnuda", "body"),
    ("elimina al perro", "structural"),
    ("change the framing to a top view", "pose"),
]
none_zero = True
for p, expected_axis in signal_prompts:
    m, a, bt, tgt = axes_mag(p)
    axis_max = max(a.values()) if a else 0.0
    has = axis_max > 0.12 or bt > 0.0
    print(f"    {p!r}: mag={m:.2f} axis_max={axis_max:.3f} body={bt:.2f} [{expected_axis}] -> señal={'SI' if has else 'NO'}")
    if not has:
        none_zero = False
check("prompts con señal real (pose/body/structural) NO son señal-cero",
      none_zero)

print("=== [2] Manager AST: bloque _zero_signal + exclusiones ===")
import ast

src = open("roop/img_editor/img_editor_manager.py", encoding="utf-8").read()
tree = ast.parse(src)

check("_zero_signal definido (estructura, no vocabulario)",
      "_zero_signal = (" in src and "body_transform_intensity) <= 0.0" in src)
check("mag boost a 0.90 + global + subject bajo señal cero",
      "mag_suggested = 0.90" in src)
check("SAFETY body excluye _zero_signal",
      "and not _zero_signal" in src)
check("rewriter no puede redirigir a background bajo señal cero",
      "if _zero_signal:\n                    rw_target = \"subject\"" in src)
check("wrapper mid-tier (0.62-0.80) con likeness + appearance libre",
      "strong likeness to them, and preserve the overall composition. The appearance may change" in src)
# Cero vocabulario: sin palabras de usuario en string literals del manager
bad_words = ["bald", "calvo", "melen", "barba", "beard"]
hits = [
    n.value
    for n in ast.walk(tree)
    if isinstance(n, ast.Constant) and isinstance(n.value, str)
    and any(w in n.value.lower() for w in bad_words)
]
check("cero vocabulario en string literals del manager", not hits)

print("=== [3] Simulación cadena completa (log 12:11) ===")
# Reproducir la ruta: mag 0.50 señal-cero → boost 0.90 → rewriter mag=0.1 no baja
mag_suggested = 0.50
body_transform_intensity = 0.0
high_structural = False
angle_change = False
pose_change = False
quality_only = False
prompt_raw = "el chico es calvo"

_zero_signal = (
    not quality_only
    and float(body_transform_intensity) <= 0.0
    and not high_structural
    and not angle_change
    and not pose_change
    and mag_suggested <= 0.55
    and len((prompt_raw or "").split()) >= 2
)
check("_zero_signal dispara con el estado exacto del log 12:11", _zero_signal)
if _zero_signal:
    mag_suggested = 0.90
    mask_target = "subject"
# rewriter inestable devuelve mag=0.1 + background (como el log)
rw_new_mag = 0.10
mag_final = max(mag_suggested, rw_new_mag)
check("rewriter mag=0.1 NO baja el régimen (sigue 0.90)", mag_final == 0.90)
rw_target = "background"
if _zero_signal:
    rw_target = "subject"
check("rewriter target=background IGNORADO → subject", rw_target == "subject")

# Caso contraste: pedido con señal (pose) NO recibe el boost ciego
mag_suggested2 = 0.50
body2 = 0.0
high_structural2 = False
angle2 = False
pose2 = True  # 'de rodillas'
zs2 = (
    not quality_only and float(body2) <= 0.0 and not high_structural2
    and not angle2 and not pose2 and mag_suggested2 <= 0.55
)
check("pedido de pose (señal real) NO dispara señal-cero", not zs2)

print()
print(f"TOTAL: {_ok} OK, {_fail} FAIL")
sys.exit(1 if _fail else 0)
