"""probe_faceswap_align_and_mouth_gate.py — v5.1202

Valida con los datos REALES del log del lote de 93 fotos (2026-09-30):
1. Gate de coherencia kps↔bbox: Frame 1 (1994859933.jpg) tenía ojos a 17,6 px
   en una cara de 632 px → warp ~1,6× sobredimensionado. Los frames 2 y 3
   eran coherentes y deben pasar.
2. Gate de paste bucal endurecido en la banda occ 0,010-0,025: el Frame 21
   (64876795_014_3fe6.jpg) llegó a restore=90% con ratio 0,23 (boca mal).
   Los objetos reales documentados (v5.1157b) van con rel 0,53-1,0 y
   ring 0,66-1,0 y deben seguir pasando.

Son simulaciones aritméticas de los gates, no un render completo.
"""

import math

ok_count = 0
fail_count = 0


def check(name, cond):
    global ok_count, fail_count
    print(f"{'OK  ' if cond else 'FAIL'} {name}")
    if cond:
        ok_count += 1
    else:
        fail_count += 1


# ----------------------------------------------------------------------------
# 1. Gate kps↔bbox (réplica exacta de _process_face_swap_v21 v5.1202)
# ----------------------------------------------------------------------------
def align_gate(kps, bbox):
    """Devuelve (rechazar, eye_rel). Rango sano: 0.05 < eye_rel < 1.20."""
    import numpy as np
    kps = np.asarray(kps, dtype=np.float32)
    bbox = np.asarray(bbox, dtype=np.float32)
    if kps.shape != (5, 2) or not np.isfinite(kps).all():
        return False, None
    bw = max(1.0, float(bbox[2] - bbox[0]))
    eye_rel = float(np.linalg.norm(kps[1] - kps[0])) / bw
    return (eye_rel < 0.05 or eye_rel > 1.20), eye_rel


# Datos reales de [ALIGN_DEBUG] del log del usuario
frame1 = {
    "kps": [[1082.9767, 677.65625], [1098.4999, 669.1824],
            [1074.585, 840.67303], [870.927, 822.27386], [879.55383, 804.59344]],
    "bbox": [655.0, 366.0, 1287.0, 842.0],
}
frame2 = {
    "kps": [[294.4138, 262.14062], [326.83224, 312.18097],
            [263.15533, 338.91425], [228.46957, 350.9828], [253.74924, 389.5761]],
    "bbox": [224.0, 207.0, 373.0, 406.0],
}
frame3 = {
    "kps": [[542.53076, 417.326], [582.48566, 414.9236],
            [572.1413, 433.97232], [549.65594, 464.69034], [577.16046, 462.89935]],
    "bbox": [508.0, 362.0, 598.0, 501.0],
}

reject1, rel1 = align_gate(frame1["kps"], frame1["bbox"])
check(f"Frame 1 rechazado (eye_rel={rel1:.3f}, antes pasaba con warp 1.6x)", reject1)
reject2, rel2 = align_gate(frame2["kps"], frame2["bbox"])
check(f"Frame 2 pasa (eye_rel={rel2:.3f})", not reject2)
reject3, rel3 = align_gate(frame3["kps"], frame3["bbox"])
check(f"Frame 3 pasa (eye_rel={rel3:.3f})", not reject3)

# Caso límite de perfil real: eye_rel pequeño pero >=0.05 no se toca
reject_p, rel_p = align_gate(
    [[100.0, 100.0], [110.0, 101.0], [104.0, 130.0], [100.0, 150.0], [108.0, 151.0]],
    [80.0, 80.0, 200.0, 200.0],
)
check(f"Perfil de canto (eye_rel={rel_p:.3f} >= 0.05) pasa", not reject_p)

# ----------------------------------------------------------------------------
# 2. Gate de paste bucal (réplica de la sección 5c v5.1202)
# ----------------------------------------------------------------------------
def mouth_gate(occ, rel, ring, is_profile=False, mouth_open=True, axis_corrected=False):
    axis_object = (
        is_profile and mouth_open and axis_corrected
        and occ > 0.002 and rel > 0.15 and ring > 0.10
    )
    has_object = axis_object or (
        (occ > 0.010)
        and (rel > 0.20)
        and (ring > 0.04)
        and ((occ > 0.025) or (rel > 0.35 and ring > 0.30))
    ) or (occ > 0.035)
    return has_object


def restore_with_photo_base(ratio, occ, rel, ring, prev_ema=None,
                            is_profile=False, mouth_open=True, axis_corrected=False):
    """m_blend foto v5.1202 + boost oral + paste/EMA, réplica de la sección 5c."""
    m = 0.06 if ratio <= 0.22 else 0.32
    if ratio > 0.22:
        oral = min(1.0, (ratio - 0.22) / 0.35)
        m = min(0.65, m * (1.0 + oral * 0.80))
    if mouth_gate(occ, rel, ring, is_profile, mouth_open, axis_corrected):
        raw = min(0.96, 0.88 + occ * 1.2)
        prev = prev_ema if prev_ema is not None else raw
        paste = 0.60 * raw + 0.40 * prev
        m = max(m, paste)
    return min(0.92, m)


# Frame 21 real: ratio=0.23, restore llegaba a 0.90 con paste por RUIDO
# (señal débil típica: occ ~0.012, rel ~0.25, ring ~0.10 — banda de ruido v5.1157b)
r_noise = restore_with_photo_base(0.23, occ=0.012, rel=0.25, ring=0.10)
check(f"Ruido en banda occ 0.010-0.025 ya no pega boca (restore={r_noise:.2f} < 0.50)", r_noise < 0.50)

# Objetos REALES documentados v5.1157b en la misma banda: rel 0.53-1.0, ring 0.66-1.0
r_real = restore_with_photo_base(0.23, occ=0.012, rel=0.53, ring=0.66)
check(f"Objeto real fino en la banda sigue paste (restore={r_real:.2f} >= 0.80)", r_real >= 0.80)

# Objeto masivo: crop > 0.035 dispara solo
r_massive = restore_with_photo_base(0.10, occ=0.040, rel=0.10, ring=0.05)
check(f"Objeto masivo (occ=0.040) dispara (restore={r_massive:.2f} >= 0.80)", r_massive >= 0.80)

# Rescate de eje corregido en perfil (v5.1200) no se toca
r_axis = restore_with_photo_base(0.30, occ=0.004, rel=0.60, ring=0.50,
                                 is_profile=True, mouth_open=True, axis_corrected=True)
check(f"Rescate eje perfil intacto (restore={r_axis:.2f} >= 0.55)", r_axis >= 0.55)

# Foto normal boca cerrada sin objeto: swap puro con costura mínima
r_closed = restore_with_photo_base(0.10, occ=0.000, rel=0.0, ring=0.0)
check(f"Boca cerrada foto sin objeto (restore={r_closed:.2f} == 0.06)", abs(r_closed - 0.06) < 0.01)

# ----------------------------------------------------------------------------
print()
print(f"Resultado: {ok_count} OK, {fail_count} FAIL")
raise SystemExit(0 if fail_count == 0 else 1)
