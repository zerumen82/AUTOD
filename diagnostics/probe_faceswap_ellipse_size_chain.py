"""probe_faceswap_ellipse_size_chain.py

Mide la cadena de tamaño del pegado elíptico en FaceSwap con máscaras
sintéticas y el código REAL de ProcessMgr (importado), replicando el orden
exacto de la sección 7:

  final_mask -> _limit_face_paste -> _widen_mask_for_blend ->
  _apply_forehead_gradient -> (x warp_face_support) -> _limit_face_paste (2º)

En cada etapa mide el radio efectivo (área >=0.5 convertida en radio circular
equivalente) y el ancho a media altura, y lo compara con el bbox: si la
cadena final sobrepasa bbox*0.5 + margen permitido (12%/16% frontal, 6%/10%
perfil), hay pegado de más.

Escenario sospechoso del usuario: elipse pegada con MAL TAMAÑO. En perfil, el
candidato 'warp' del safety recovery nace de corners de M (1.7-1.9x el ancho
de la cabeza) y el segundo límite frontal (12%/16%) es mas ancho que el de
perfil (6%/10%): una elipse warp de perfil limitada con margen frontal deja
mas pegado del que la ventana de perfil permitiria.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
os.environ.setdefault("AUTOAUTO_SILENT", "1")

from roop.ProcessMgr import ProcessMgr

FRAME_H, FRAME_W = 1280, 720
FRAME_SHAPE = (FRAME_H, FRAME_W)

ok = 0
fail = 0


def check(name, cond, extra=""):
    global ok, fail
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' | ' + extra) if extra else ''}")
    if cond:
        ok += 1
    else:
        fail += 1


def effective_radii(mask, bbox):
    """Radio equivalente y extension respecto al bbox permitido."""
    ys, xs = np.where(mask >= 0.5)
    if not len(xs):
        return None
    x1, y1, x2, y2 = bbox
    fw, fh = x2 - x1, y2 - y1
    area = float((mask >= 0.5).sum())
    r_eq = float(np.sqrt(area / np.pi))
    ext_x = (xs.max() - x2) / fw, (x1 - xs.min()) / fw
    ext_y = (ys.max() - y2) / fh, (y1 - ys.min()) / fh
    return r_eq, max(ext_x[0], ext_x[1], 0.0), max(ext_y[0], ext_y[1], 0.0)


def run_chain(final_mask, is_profile, bbox, face_w, face_h, support=None):
    """Replica el orden real de la seccion 7 (con el fix propuesto aplicable)."""
    x1, y1, x2, y2 = bbox
    m = _limit_face_paste(final_mask, is_profile, (x1, y1, x2, y2))
    m = ProcessMgr._widen_mask_for_blend(m, face_w, face_h, FRAME_SHAPE)
    m = ProcessMgr._apply_forehead_gradient(m, FRAME_SHAPE, None, y1, face_h, is_profile)
    if support is not None:
        m = np.clip(m * support, 0, 1.0)
    m = _limit_face_paste(m, is_profile, (x1, y1, x2, y2))
    return m


def synth_mask(kind, bbox):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    fw, fh = x2 - x1, y2 - y1
    m = np.zeros(FRAME_SHAPE, dtype=np.float32)
    if kind == "xseg_ok":
        cv2.ellipse(m, (cx, cy), (int(fw * 0.42), int(fh * 0.46)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (21, 21), 0)
    elif kind == "xseg_collapsed":
        cv2.ellipse(m, (cx, cy), (max(6, fw // 12), max(6, fh // 12)), 0, 0, 360, 0.4, -1)
    elif kind == "safety_warp_profile":
        # corners del warp en perfil: 1.8x ancho del bbox (documentado v5.1197)
        cv2.ellipse(m, (cx, cy), (int(fw * 0.90), int(fh * 0.52)), 0, 0, 360, 1.0, -1)
        m = cv2.GaussianBlur(m, (31, 31), 0)
    return m


def profile_head_support(bbox, profile=True):
    """Soporte eliptico warp-eado limitado por _limit_face_paste (como en el swap)."""
    x1, y1, x2, y2 = bbox
    sup = np.zeros(FRAME_SHAPE, dtype=np.float32)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.ellipse(sup, (cx, cy), (int((x2 - x1) * 0.53), int((y2 - y1) * 0.55)), 0, 0, 360, 1.0, -1)
    sup = cv2.GaussianBlur(sup, (17, 17), 0)
    return _limit_face_paste(sup, profile, (x1, y1, x2, y2))


class _Dummy:
    """Self vacio: _profile_head_window no lee atributos de instancia."""


def _limit_face_paste(mask, is_profile, bbox, frame_shape=None):
    """Replica exacta del metodo de instancia (logica identica a ProcessMgr)."""
    if mask is None or bbox is None:
        return mask
    if frame_shape is None:
        frame_shape = mask.shape
    window = ProcessMgr._profile_head_window(_Dummy(), frame_shape, bbox, is_profile=is_profile)
    if window.shape[:2] != mask.shape[:2]:
        return mask
    return np.clip(mask.astype(np.float32) * window, 0.0, 1.0)


# Caso 1: XSeg sana frontal — la cadena no debe sobrepasar el margen permitido
bbox = (300, 250, 480, 430)  # 180x180
allowed = 0.12  # frontal margen X
m1 = run_chain(synth_mask("xseg_ok", bbox), False, bbox, 180, 180, profile_head_support(bbox, False))
r = effective_radii(m1, bbox)
check("XSeg sana frontal: sin pegado fuera del margen (12% X)",
      r is not None and r[1] <= allowed + 0.02, f"ext_x={r[1]:.2f}" if r else "sin area")

# Caso 2: recuperación conserva el soporte de la segmentación medida.
regen_f = ProcessMgr._build_xseg_shaped_safety_mask(
    FRAME_SHAPE, synth_mask("xseg_collapsed", bbox), bbox, tight=False,
)
m2 = run_chain(regen_f, False, bbox, 180, 180, profile_head_support(bbox, False))
r2 = effective_radii(m2, bbox)
check("XSeg colapsada frontal (contorno medido): contenida por el limite (12% X)",
      r2 is not None and r2[1] <= allowed + 0.02, f"ext_x={r2[1]:.2f}" if r2 else "sin area")

# Caso 3: PERFILES con elipse warp sobredimensionada (1.8x ancho) — el caso del
# usuario: la elipse pegada mas grande que la cabeza
bbox_p = (300, 200, 380, 400)  # perfil 80x200
m3 = run_chain(synth_mask("safety_warp_profile", bbox_p), True, bbox_p, 80, 200,
               profile_head_support(bbox_p, True))
r3 = effective_radii(m3, bbox_p)
check("Perfil con elipse warp 1.8x: contenida por el limite (6% X)",
      r3 is not None and r3[1] <= 0.06 + 0.02, f"ext_x={r3[1]:.2f}" if r3 else "sin area")

# Caso 4: el bug sospechoso — is_profile detectado False en un perfil real
# (hysteresis decayendo): margen frontal 12% aplicado a elipse de perfil 1.8x
m4 = run_chain(synth_mask("safety_warp_profile", bbox_p), False, bbox_p, 80, 200,
               profile_head_support(bbox_p, False))
r4 = effective_radii(m4, bbox_p)
print(f"    [info] perfil clasificado frontal: ext_x={r4[1]:.2f} (permitido 0.14 total) — solo diagnostico")

# Caso 5: ausencia de forma no inventa una superficie de pegado.
regen = ProcessMgr._build_xseg_shaped_safety_mask(
    FRAME_SHAPE, np.zeros(FRAME_SHAPE, dtype=np.float32), bbox, tight=False,
)
check("Segmentación vacía no crea máscara frontal", regen is None)

regen_p = ProcessMgr._build_xseg_shaped_safety_mask(
    FRAME_SHAPE, None, bbox_p, tight=True,
)
check("Segmentación ausente no crea máscara de perfil", regen_p is None)

print()
print(f"Resultado: {ok} OK, {fail} FAIL")
sys.exit(0 if fail == 0 else 1)
