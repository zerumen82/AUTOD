"""probe_faceswap_video_reject_recovery_20261107.py — v5.1319

Valida con los datos REALES de los logs del lote 2026-11-07:

A) Recuperación de geometría en el gate kps↔bbox (v5.1202) en vídeo:
   - Vídeo 3 (VID_20261027_175214_860.mp4): 206 frames rechazados,
     rango eye_dist/bbox_w 0,003-0,050, bbox real [290, 553, 353, 604]
     (cara 63×51 px) → 28,5% del vídeo sin swap.
   - Vídeo 9 (VID_20261027_174936_599.mp4): 61 rechazos,
     rango 0,006-0,049, bbox real [413, 380, 482, 482].
   Antes: en vídeo la recuperación estaba deshabilitada y el frame salía
   sin swap (0 recuperaciones en todo el lote). Ahora:
   1) reusa el último kps sano de la persona (trasladado/escalado al bbox
      actual), 2) si no hay histórico, sintetiza los 5 kps del source,
   3) si nada pasa las comprobaciones → comportamiento original (sin swap).

B) Hold temporal en el gate «Contorno medido insuficiente» (línea ~6617):
   227 skips en el lote (219 en el vídeo 9) salían sin swap sin probar el
   hold que el gate 1 ya usa. Ahora se prueba el mismo hold; solo si no hay
   historia se registra el skip.

Son simulaciones aritméticas y llamadas a los helpers reales, no un render.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.faceswap_config import landmarks_fit_bbox  # noqa: E402

ok_count = 0
fail_count = 0


def check(name, cond):
    global ok_count, fail_count
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    if cond:
        ok_count += 1
    else:
        fail_count += 1


def make_manager(video="VID_20261027_175214_860.mp4"):
    m = ProcessMgr.__new__(ProcessMgr)
    m.options = SimpleNamespace(current_video_path=f"X:\\in\\{video}")
    m._current_video_key = f"video_{video}"
    return m


def gate_eye_rel(kps, bbox):
    """Réplica del gate v5.1202: True = rechazado."""
    kps = np.asarray(kps, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32)
    if kps.shape != (5, 2) or not np.isfinite(kps).all():
        return True
    if not landmarks_fit_bbox(kps, bbox):
        return True
    bw = max(1.0, float(bbox[2] - bbox[0]))
    eye_rel = float(np.linalg.norm(kps[1] - kps[0])) / bw
    return not (0.05 < eye_rel < 1.20)


# ----------------------------------------------------------------------------
# Datos REALES del log (video 3, frame 6051 y racha de 31 frames)
# ----------------------------------------------------------------------------
BBOX_V3 = np.array([290.0, 553.0, 353.0, 604.0], dtype=np.float32)  # 63×51 px
# kps degenerados reales: eye_dist/bbox_w = 0.013 → ojos a 0,82 px
EYE_REL_DEGEN = 0.013
KPS_DEGEN_V3 = np.array([
    [320.0, 570.0],
    [320.0 + EYE_REL_DEGEN * 63.0, 570.0],   # ojos prácticamente superpuestos
    [320.0, 576.0],
    [318.0, 582.0],
    [322.0, 582.0],
], dtype=np.float32)

# KPS sanos para ese mismo bbox (proporciones normales: eye_rel ≈ 0,41)
KPS_GOOD_V3 = np.array([
    [307.0, 567.0],
    [333.0, 567.0],
    [320.0, 580.0],
    [310.0, 590.0],
    [330.0, 590.0],
], dtype=np.float32)

# Source con bbox grande y kps sanos (simulación del source #79 elegido)
SRC_BBOX = np.array([100.0, 100.0, 300.0, 340.0], dtype=np.float32)
SRC_KPS = np.array([
    [140.0, 180.0],
    [220.0, 180.0],
    [180.0, 215.0],
    [150.0, 255.0],
    [210.0, 255.0],
], dtype=np.float32)


def face(bbox, kps, det=0.9):
    # SimpleNamespace: _face_with_kps usa copy + atributo (el acceso como dict
    # va en try/except en el helper real).
    return SimpleNamespace(bbox=np.asarray(bbox, dtype=np.float32),
                           kps=np.asarray(kps, dtype=np.float32),
                           det_score=det, embedding=np.ones(512, dtype=np.float32))


# ----------------------------------------------------------------------------
# A0. El caso real SÍ era rechazado por el gate (reproduce el fallo)
# ----------------------------------------------------------------------------
check("A0 gate rechaza los kps degenerados reales del video 3 "
      f"(eye_rel={EYE_REL_DEGEN})", gate_eye_rel(KPS_DEGEN_V3, BBOX_V3))

# ----------------------------------------------------------------------------
# A1. _face_with_kps no muta el objeto del tracker
# ----------------------------------------------------------------------------
mgr = make_manager()
tgt = face(BBOX_V3, KPS_DEGEN_V3)
fixed = mgr._face_with_kps(tgt, KPS_GOOD_V3)
check("A1 copia con kps sanos, tracker intacto "
      f"(original eye_rel={float(np.linalg.norm(tgt.kps[1]-tgt.kps[0]))/63:.3f})",
      np.array_equal(tgt.kps, KPS_DEGEN_V3)
      and np.array_equal(fixed.kps, KPS_GOOD_V3)
      and fixed is not tgt)

# ----------------------------------------------------------------------------
# A2. _remember_good_face_geom guarda solo geometría sana
# ----------------------------------------------------------------------------
mgr = make_manager()
mgr._remember_good_face_geom(mgr._current_video_key, 0, face(BBOX_V3, KPS_GOOD_V3))
mgr._remember_good_face_geom(mgr._current_video_key, 1, face(BBOX_V3, KPS_DEGEN_V3))
hist = getattr(mgr, "_last_good_face_geom", {})
check("A2 historia guarda la sana y rechaza la degenerada "
      f"(entradas={len(hist)})",
      f"{mgr._current_video_key}|0" in hist
      and f"{mgr._current_video_key}|1" not in hist)

# ----------------------------------------------------------------------------
# A3. Recuperación con histórico (misma persona, misma pose)
# ----------------------------------------------------------------------------
mgr = make_manager()
mgr._remember_good_face_geom(mgr._current_video_key, 0, face(BBOX_V3, KPS_GOOD_V3))
tgt = face(BBOX_V3, KPS_DEGEN_V3)
rec = mgr._recovered_coherent_kps(face(SRC_BBOX, SRC_KPS), tgt, 0)
check("A3 recupera con el último kps sano de la persona "
      "(vídeo 3, bbox del frame 6051)",
      rec is not None and not gate_eye_rel(rec, BBOX_V3))
if rec is not None:
    er = float(np.linalg.norm(rec[1] - rec[0])) / 63.0
    check(f"A3b geometría saneada con proporción sana (eye_rel={er:.3f} ≈ 0,41)",
          0.30 < er < 0.55)

# ----------------------------------------------------------------------------
# A4. Sin histórico: sintetiza desde el source elegido
# ----------------------------------------------------------------------------
mgr = make_manager()
tgt = face(BBOX_V3, KPS_DEGEN_V3)
rec = mgr._recovered_coherent_kps(face(SRC_BBOX, SRC_KPS), tgt, 0)
check("A4 sin histórico sintetiza kps del source y pasa el gate",
      rec is not None and not gate_eye_rel(rec, BBOX_V3))
if rec is not None:
    # La síntesis debe quedar DENTRO del bbox trackeado (nunca el warp 1,6×)
    pts = np.asarray(rec, dtype=np.float32)
    inside = (pts[:, 0] >= BBOX_V3[0] - 1e3).all() and (pts[:, 0] <= BBOX_V3[2] + 1e3).all()
    check("A4b kps sintetizados acotados al bbox del tracker", bool(inside))

# ----------------------------------------------------------------------------
# A5. Sin histórico y sin source utilizable → None (comportamiento original)
# ----------------------------------------------------------------------------
mgr = make_manager()
tgt = face(BBOX_V3, KPS_DEGEN_V3)
rec = mgr._recovered_coherent_kps(None, tgt, 0)
check("A5 sin histórico ni source → None (frame sin swap, nunca peor)",
      rec is None)

# ----------------------------------------------------------------------------
# A6. Histórico de OTRA persona no se reutiliza
# ----------------------------------------------------------------------------
mgr = make_manager()
mgr._remember_good_face_geom(mgr._current_video_key, 0, face(BBOX_V3, KPS_GOOD_V3))
tgt = face(BBOX_V3, KPS_DEGEN_V3)
rec = mgr._recovered_coherent_kps(None, tgt, 1)  # persona 1, sin source
check("A6 la historia es por persona (persona 1 no reusa la 0)", rec is None)

# ----------------------------------------------------------------------------
# A7. Histórico degenerado (basura) no se reutiliza → cae al source
# ----------------------------------------------------------------------------
mgr = make_manager()
# Entrada corrupta metida a mano (el guardado la rechaza, pero defensa en caliente)
mgr._last_good_face_geom = {
    f"{mgr._current_video_key}|0": (KPS_DEGEN_V3.copy(), BBOX_V3.copy()),
}
tgt = face(BBOX_V3, KPS_DEGEN_V3)
rec = mgr._recovered_coherent_kps(face(SRC_BBOX, SRC_KPS), tgt, 0)
check("A7 historia corrupta → se ignora y se sintetiza desde el source",
      rec is not None and not gate_eye_rel(rec, BBOX_V3))

# ----------------------------------------------------------------------------
# B. Hold temporal en «Contorno medido insuficiente» (helpers reales)
# ----------------------------------------------------------------------------
video = "VID_20261027_174936_599.mp4"  # vídeo 9: 219 de 227 skips
mgr = ProcessMgr.__new__(ProcessMgr)
mgr.options = SimpleNamespace(current_video_path=f"X:\\in\\{video}")
H, W = 1280, 720
prev_full = np.full((H, W, 3), 90, dtype=np.uint8)
original = np.zeros((H, W, 3), dtype=np.uint8)
last_bbox = (332, 61, 424, 183)  # bbox real del tracker (log frame 493)
prev_mask = np.zeros((H, W), dtype=np.float32)
prev_mask[61:183, 332:424] = 1.0

# B0. Con historia: el gate decide HOLD (replica de la nueva condición)
mgr._prev_frame_result = {f"video_{video}": prev_full}
mgr._last_swap_bbox = {video: last_bbox}
mgr._last_swap_mask = {video: prev_mask}
setattr(mgr, f"_frame_count_{video}", 500)
hold_ok = mgr._should_hold_previous_swap(video)
blend = mgr._faded_hold_blend(video, 0.88)
b0 = hold_ok and blend >= 0.10
check(f"B0 con historia decide HOLD (hold_ok={hold_ok}, blend={blend:.2f})", b0)

# B1. El hold real píxel a píxel devuelve la región swappeada, no el frame original
rng = np.random.default_rng(7)
original = (rng.integers(20, 60, (H, W, 3))).astype(np.uint8)  # escena real, no negro
held = mgr._hold_previous_face_region(original, prev_full, last_bbox,
                                       blend=blend, prev_mask=prev_mask)
inside_orig = float(original[61:183, 332:424].mean())
inside_held = float(held[61:183, 332:424].mean())
outside_same = np.array_equal(held[:50, :], original[:50, :])
check("B1 hold aplica los píxeles swappeados SOLO en la región de la cara "
      f"(antes={inside_orig:.1f} → después={inside_held:.1f}, fuera intacto)",
      inside_held > inside_orig + 15.0 and outside_same)

# B2. Sin historia (prev ausente) → no hay hold → skip (comportamiento original)
mgr2 = ProcessMgr.__new__(ProcessMgr)
mgr2.options = SimpleNamespace(current_video_path=f"X:\\in\\{video}")
setattr(mgr2, f"_frame_count_{video}", 500)
no_prev = mgr2._prev_frame_result if hasattr(mgr2, "_prev_frame_result") else {}
b2 = (no_prev.get(f"video_{video}") is not None
      and mgr2._should_hold_previous_swap(video))
check("B2 sin frame anterior → sin hold → se registra el skip", not b2)

# B3. Tras corte de escena duro no se hace hold (anti-swap-congelado intacto)
mgr3 = ProcessMgr.__new__(ProcessMgr)
mgr3.options = SimpleNamespace(current_video_path=f"X:\\in\\{video}")
mgr3._prev_frame_result = {f"video_{video}": prev_full}
mgr3._last_swap_bbox = {video: last_bbox}
setattr(mgr3, f"_frame_count_{video}", 510)
setattr(mgr3, f"_hard_scene_cut_frame_{video}", 505)
b3 = mgr3._should_hold_previous_swap(video)
check("B3 tras corte duro (<15 frames) no se hace hold", not b3)

# ----------------------------------------------------------------------------
print()
print(f"Resultado: {ok_count} OK, {fail_count} FAIL")
raise SystemExit(0 if fail_count == 0 else 1)
