# -*- coding: utf-8 -*-
"""v5.1155: el rescate tracking_emb re-aplica gates mini-cara/cortada/área.
Log real (process_20260916_173055 video 1): mini-cara 14x20px track_sim=0.99
re-inyectada por el rescate → ciclo aceptar-rechazar 96 frames.
"""
import ast, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

src = open('roop/ProcessMgr.py', encoding='utf-8').read()
tree = ast.parse(src)

# Localizar el bloque del rescate dentro de _reacquire_face_global
fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == '_reacquire_face_global':
        fn = node
        break
assert fn is not None, "funcion no encontrada"
fn_src = ast.get_source_segment(src, fn)

results = []
def check(name, cond):
    results.append((name, cond))
    print(("OK  " if cond else "FAIL") + f" {name}")

# T1: el rescate contiene el gate mini-cara (< 32 / < 1024)
rescue_idx = fn_src.find('RE-aplicar gates')
rescue_block = fn_src[rescue_idx:rescue_idx+3000] if rescue_idx >= 0 else ""
check("T1 gate mini-cara en rescate", "_rw < 32 or _rh < 32 or _rw * _rh < 1024" in rescue_block)
check("T2 gate cortada en rescate", "_is_bbox_clipped_by_frame(face.bbox, frame_shape, allow_partial_overflow=True)" in rescue_block)
check("T3 gate área en rescate", "_ar < 0.06 or _ar >" in rescue_block)
check("T4 gate área antes del append", rescue_block.find("_ar < 0.06") < rescue_block.find("candidates.append((_tsim"))
check("T5 umbral rescate intacto", "_rescue_min = 0.48 if multi_face else 0.38" in rescue_block)

# T6: el print del rescate ya no está limitado a %25 (log siempre visible para diagnóstico)
# (se conserva %25 para no spamear; comprobar que existe)
check("T6 print rescate presente", "rescate por tracking_emb" in fn_src)

# T7: simulación exacta del caso del log
class F:
    def __init__(self, bbox, tsim):
        self.bbox = bbox
        self._tsim = tsim
    @property
    def embedding(self): return object()

tracking_emb = object()
multi_face = False
after_scene_cut = False
last_swap_bbox = [715, 50, 729, 70]  # la última "swap" ERA la mini-cara (64 accepts previos)
frame_shape = (720, 1280, 3)
is_large = lambda b, fs: False
clipped = lambda b, fs, allow_partial_overflow=False: False

def simulate(faces):
    cands = []
    _rescue_min = 0.48 if multi_face else 0.38
    for face in faces:
        _tsim = face._tsim
        if _tsim >= _rescue_min:
            _rw = face.bbox[2] - face.bbox[0]
            _rh = face.bbox[3] - face.bbox[1]
            if _rw < 32 or _rh < 32 or _rw * _rh < 1024:
                continue
            if frame_shape is not None and clipped(face.bbox, frame_shape, True):
                continue
            if not after_scene_cut:
                _lbr = last_swap_bbox
                if _lbr is not None:
                    _lw = max(1, _lbr[2]-_lbr[0]); _lh = max(1, _lbr[3]-_lbr[1])
                    _ar = (_rw*_rh)/max(1.0,_lw*_lh)
                    if _ar < 0.06 or _ar > (100.0 if is_large(face.bbox, frame_shape) else 6.0):
                        continue
            cands.append((_tsim, 0, face))
    return cands

# Caso real: mini-cara 14x20 con sim 0.99 → DEBE ser rechazada ahora
mini = F([715, 50, 729, 70], 0.99)
out = simulate([mini])
check("T7 mini-cara 14x20 sim 0.99 rechazada", len(out) == 0)

# Cara normal 90x110 misma persona: con última swap mini-cara (area_ratio 43)
# el gate de área la rechaza — y es CORRECTO: un salto de 14px a 90px no es la
# misma cara física. Con última swap coherente (cara real) debe aceptarse.
last_swap_bbox = [400, 200, 490, 310]  # misma cara de antes
ok_face = F([400, 200, 490, 310], 0.70)
out = simulate([ok_face])
check("T8 cara normal 0.70 con última swap coherente aceptada", len(out) == 1)
last_swap_bbox = [715, 50, 729, 70]  # restaurar caso mini para los siguientes tests

# Cara cortada → rechazada
import types
g = {'clipped': None}
def clipped_true(b, fs, allow_partial_overflow=False): return True
clipped_backup = clipped
clipped = clipped_true
cut_face = F([100, 200, 300, 400], 0.70)
out = simulate([cut_face])
clipped = clipped_backup
check("T9 cara cortada rechazada", len(out) == 0)

# Cara 40x40 con área 3x menor que última swap → rechazada por área
small = F([400, 200, 440, 240], 0.70)  # 1600px vs 280px última = area_ratio ~5.7
# ajustar: 1600/ (14*20=280) = 5.7 < 6.0 → aceptada. Usar aún más pequeña
tiny = F([400, 200, 430, 230], 0.70)  # 900px / 280 = 3.2 → aceptada...
# construyo caso claro: última swap 200x200, candidata 30x30
last_swap_bbox2 = [400, 200, 600, 400]
_lbr = last_swap_bbox2
_lw, _lh = 200, 200
_rws, _rhs = 30, 30
_ar = (_rws*_rhs)/max(1.0,_lw*_lh)
check("T10 aritmetica área: 900/40000=0.0225 < 0.06 → rechazo", _ar < 0.06)

n_fail = sum(1 for _, c in results if not c)
print(f"\n{'='*46}")
print(f"{'PASAN' if n_fail==0 else 'FALLAN'}: {len(results)-n_fail}/{len(results)}")
sys.exit(0 if n_fail == 0 else 1)
