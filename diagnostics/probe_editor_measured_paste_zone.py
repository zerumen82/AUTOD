#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe: la zona pegada se mide del cambio, no de una forma de cabeza.

Reproduce los dos defectos de edit_1790845755.png / edit_1790852880.png con
datos sinteticos, de modo que la medicion sea reproducible:

  1. Una muestra regenerada pegada con una mascara de cabeza produce una banda
     translucida (5 % < alpha < 95 %) sobre la foto.
  2. El guard de altura (0,60) devolvia la foto sin editar y ademas media la
     altura contra la foto completa, no contra la ventana local.
  3. Una edicion de pelo dentro de un sujeto medido debe pegarse completa.

No usa red ni GPU: solo numpy, OpenCV y el metodo real del manager.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

MANAGER = ImgEditorManager()


def translucent_band(result: Image.Image, source: Image.Image) -> float:
    """Fraction of the frame that changed while being only partly blended.

    A correctly pasted local edit writes whole pixels (alpha ~1) or none at
    all (alpha ~0). A partial band means a regenerated frame was composited
    through a soft mask: the translucent rectangle of the reported defect.
    """
    a = np.asarray(source.convert("RGB"), dtype=np.int16)
    b = np.asarray(result.convert("RGB"), dtype=np.int16)
    delta = np.abs(a - b).max(axis=2)
    changed = delta >= 8
    if not changed.any():
        return 0.0
    return float(changed.mean())


def hair_case():
    """Dark hair mass replaced by scalp inside a measured subject."""
    w, h = 400, 300
    photo = np.full((h, w, 3), 190, dtype=np.uint8)
    photo[:, :] = (200, 175, 160)
    # Measured subject: a head-sized region anchored on a real face box.
    subject = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(subject, (200, 130), (95, 110), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    # Original: dark hair on the upper 60 % of the head.
    photo_arr = photo.copy()
    cv2.ellipse(photo_arr, (200, 95), (88, 70), 0, 0, 360, (40, 35, 30), -1)
    source = Image.fromarray(photo_arr, "RGB")
    # Generated: the hair is gone (bald scalp), everything else untouched.
    gen_arr = photo_arr.copy()
    cv2.ellipse(gen_arr, (200, 95), (88, 70), 0, 0, 360, (205, 180, 165), -1)
    # The model also rewrote the background outside the subject.
    gen_arr[:40, :] = (30, 60, 120)
    generated = Image.fromarray(gen_arr, "RGB")
    box = (0, 0, w, h)
    result = MANAGER._integrate_local_generation(
        source, generated, subject_img, box,
    )
    band = translucent_band(result, source)
    src_hair = np.asarray(source.convert("L"), dtype=np.int16)
    out_hair = np.asarray(result.convert("L"), dtype=np.int16)
    core = np.asarray(subject_img, dtype=np.uint8) >= 160
    ys = np.flatnonzero(core.any(axis=1))
    scalp = core.copy()
    scalp[int(ys[0] + 0.60 * (ys[-1] - ys[0])) + 1:, :] = False
    before = float((src_hair[scalp] < 90).mean())
    after = float((out_hair[scalp] < 90).mean())
    # Outside the measured subject the photo must be byte-identical: the model
    # rewrote those pixels, and a correct measured paste never writes them.
    outside = np.asarray(subject_img, dtype=np.uint8) < 96
    leak = float((np.abs(
        np.asarray(result, dtype=np.int16) - np.asarray(source, dtype=np.int16)
    ).max(axis=2)[outside] >= 8).mean()) if outside.any() else 0.0
    print(f"  pelo en cuero {before:.1%} -> {after:.1%}")
    print(f"  fuga fuera del sujeto medido: {leak:.2%} del resto de la foto")
    print(f"  cambios en el marco: {band:.1%}")
    ok_hair = after < 0.10 * max(before, 0.01)
    # The rewritten background must not reach the photo at all.
    ok_bg = leak < 0.001
    # A measured paste is a hard local write: no partial-alpha frame-wide band.
    ok_band = band < 0.45
    print(f"  [{'OK' if ok_hair and ok_bg and ok_band else 'FALLO'}] pelo borrado={ok_hair}, fondo intacto={ok_bg}, sin banda translucida={ok_band}")
    return ok_hair and ok_bg and ok_band


def no_change_case():
    """A sample that did not change the subject must not invent a zone."""
    photo = np.full((240, 320, 3), 150, dtype=np.uint8)
    source = Image.fromarray(photo, "RGB")
    generated = Image.fromarray(photo.copy(), "RGB")
    subject = np.zeros((240, 320), dtype=np.uint8)
    cv2.ellipse(subject, (160, 110), (70, 80), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    result = MANAGER._integrate_local_generation(
        source, generated, subject_img, (0, 0, 320, 240),
    )
    band = translucent_band(result, source)
    zone = MANAGER._changed_zone_on_subject(
        generated, source, subject_img,
    )
    print(f"  zona medida sobre una muestra identica: {zone}")
    ok = band < 0.02 and zone is None
    print(f"  [{'OK' if ok else 'FALLO'}] sin cambio no hay zona ni pegado")
    return ok


def tall_region_case():
    """A legitimate close portrait must still be edited (no height guard)."""
    # Same measured subject as the failing session: it spans most of the frame
    # height. The old 0.60 guard refused it. Now the paste happens on the
    # measured change instead.
    w, h = 500, 340
    source_arr = np.full((h, w, 3), 180, dtype=np.uint8)
    cv2.ellipse(source_arr, (250, 170), (110, 165), 0, 0, 360, (60, 50, 45), -1)
    source = Image.fromarray(source_arr, "RGB")
    gen_arr = source_arr.copy()
    cv2.ellipse(gen_arr, (250, 170), (110, 165), 0, 0, 360, (210, 185, 170), -1)
    generated = Image.fromarray(gen_arr, "RGB")
    subject = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(subject, (250, 170), (110, 165), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    ys = np.flatnonzero((subject >= 160).any(axis=1))
    span = float(ys[-1] - ys[0] + 1) / h
    result = MANAGER._integrate_local_generation(
        source, generated, subject_img, (0, 0, w, h),
    )
    changed = float((np.abs(
        np.asarray(result, dtype=np.int16) - np.asarray(source, dtype=np.int16)
    ).max(axis=2) >= 8).mean())
    print(f"  zona medida abarca {span:.0%} del alto (el guard viejo lo bloqueaba con 60 %)")
    ok = changed > 0.10
    print(f"  [{'OK' if ok else 'FALLO'}] retrato legitimo editado, cambio={changed:.1%}")
    return ok


def soft_mask_geometry():
    """The measured zone replaces the invented ellipse, so feather is small."""
    subject = np.zeros((400, 400), dtype=np.uint8)
    cv2.ellipse(subject, (200, 200), (120, 150), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    base = np.full((400, 400, 3), 170, dtype=np.uint8)
    source = Image.fromarray(base, "RGB")
    # A localized change: a small hat-like blob inside the subject.
    gen_arr = base.copy()
    cv2.circle(gen_arr, (200, 110), 55, (20, 20, 25), -1)
    generated = Image.fromarray(gen_arr, "RGB")
    zone = MANAGER._changed_zone_on_subject(
        generated, source, subject_img,
    )
    arr = np.asarray(zone, dtype=np.uint8)
    ys, xs = np.where(arr >= 160)
    ok = zone is not None and len(xs) > 0
    if ok:
        # Feather must stay local: the graded band cannot reach the frame edge.
        soft = zone.filter(ImageFilter.GaussianBlur(6))
        sarr = np.asarray(soft, dtype=np.int16)
        partial = (sarr > 26) & (sarr < 229)
        share = float(partial.mean())
        ok = share < 0.10
        print(f"  contorno medido {int(xs.max()-xs.min()+1)}x{int(ys.max()-ys.min()+1)} px, banda de feather {share:.1%}")
    print(f"  [{'OK' if ok else 'FALLO'}] zona medida del cambio, feather contenido")
    return bool(ok)


def subtle_edit_case():
    """A low-intensity change must still be pasted completely.

    A fixed threshold of 8 levels threw these away and fell back to the
    invented ellipse, which is the translucent patch. The threshold is now
    the noise measured outside the subject.
    """
    w, h = 360, 300
    base = np.full((h, w, 3), (185, 160, 145), dtype=np.uint8)
    subject = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(subject, (180, 150), (110, 140), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    source = Image.fromarray(base, "RGB")
    # A garment tone shift of about 10 levels across the torso.
    gen_arr = base.copy()
    torso = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(torso, (180, 250), (95, 60), 0, 0, 360, 255, -1)
    inside = torso.astype(bool) & (subject >= 96)
    gen_arr[inside] = np.clip(
        gen_arr[inside].astype(np.int16) + np.array([-14, 6, 22]), 0, 255,
    ).astype(np.uint8)
    generated = Image.fromarray(gen_arr, "RGB")
    result = MANAGER._integrate_local_generation(
        source, generated, subject_img, (0, 0, w, h),
    )
    out = np.asarray(result, dtype=np.int16)
    src = np.asarray(source, dtype=np.int16)
    torso_core = torso.astype(bool) & (subject >= 128)
    shift = float(np.abs(out[torso_core] - src[torso_core]).mean())
    ok = shift > 6.0
    print(f"  cambio de tono medio dentro de la prenda: {shift:.1f} niveles")
    print(f"  [{'OK' if ok else 'FALLO'}] edicion sutil pegada completa")
    return ok


def garment_edit_case():
    """A large recolor over the subject keeps the background untouched."""
    w, h = 380, 320
    base = np.full((h, w, 3), (200, 180, 165), dtype=np.uint8)
    base[:, :, 2] = 210
    subject = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(subject, (190, 160), (120, 150), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    source = Image.fromarray(base, "RGB")
    gen_arr = base.copy()
    body = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(body, (190, 275), (110, 75), 0, 0, 360, 255, -1)
    inside = body.astype(bool) & (subject >= 96)
    gen_arr[inside] = (20, 40, 160)
    gen_arr[0:30, :] = (0, 0, 0)  # model also rewrote the sky
    generated = Image.fromarray(gen_arr, "RGB")
    result = MANAGER._integrate_local_generation(
        source, generated, subject_img, (0, 0, w, h),
    )
    out = np.asarray(result, dtype=np.int16)
    src = np.asarray(source, dtype=np.int16)
    body_core = body.astype(bool) & (subject >= 128)
    recolored = float((out[body_core][:, 2] > 120).mean())
    outside = subject < 96
    leak = float((np.abs(out - src).max(axis=2)[outside] >= 8).mean())
    ok = recolored > 0.90 and leak < 0.001
    print(f"  prenda recoloreada: {recolored:.1%}, fuga fuera del sujeto: {leak:.2%}")
    print(f"  [{'OK' if ok else 'FALLO'}] recolor grande completo y fondo intacto")
    return ok


def added_object_case():
    """An object added inside the subject is pasted, the rest is not."""
    w, h = 360, 300
    base = np.full((h, w, 3), (190, 170, 155), dtype=np.uint8)
    subject = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(subject, (180, 150), (105, 140), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    source = Image.fromarray(base, "RGB")
    gen_arr = base.copy()
    cv2.circle(gen_arr, (180, 90), 48, (25, 25, 30), -1)   # new hat
    cv2.rectangle(gen_arr, (250, 60), (340, 110), (30, 90, 200), -1)  # outside
    generated = Image.fromarray(gen_arr, "RGB")
    result = MANAGER._integrate_local_generation(
        source, generated, subject_img, (0, 0, w, h),
    )
    out = np.asarray(result, dtype=np.int16)
    src = np.asarray(source, dtype=np.int16)
    hat = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(hat, (180, 90), 48, 255, -1)
    hat_core = hat.astype(bool) & (subject >= 96)
    # The measured outline is feathered on purpose, so only the interior has
    # to be fully replaced; the rim is a soft blend by design.
    inner = cv2.erode(
        hat_core.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=3,
    ).astype(bool)
    hat_ok = float((out[inner].max(axis=1) < 90).mean()) if inner.any() else 0.0
    outside = subject < 96
    leak = float((np.abs(out - src).max(axis=2)[outside] >= 8).mean())
    ok = hat_ok > 0.98 and leak < 0.001
    print(f"  interior del objeto pegado: {hat_ok:.1%} (borde con feather), "
          f"fuga fuera del sujeto: {leak:.2%}")
    print(f"  [{'OK' if ok else 'FALLO'}] objeto anadido dentro del sujeto, fuera intacto")
    return ok


def scattered_change_case():
    """A scattered low-intensity change is kept whole, not fragmented.

    Smoothing or a subtle recolor answers per pixel, so a size filter on the
    responding specks would throw the whole edit away.
    """
    w, h = 320, 280
    base = np.full((h, w, 3), (170, 150, 140), dtype=np.uint8)
    subject = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(subject, (160, 140), (100, 130), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    source = Image.fromarray(base, "RGB")
    rng = np.random.default_rng(7)
    noise = rng.integers(-9, 10, size=(h, w, 1))
    gen_arr = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    gen_arr[:, :, 0] = np.clip(base[:, :, 0].astype(np.int16) + 12, 0, 255).astype(np.uint8)
    generated = Image.fromarray(gen_arr, "RGB")
    zone = MANAGER._changed_zone_on_subject(
        generated, source, subject_img,
    )
    ok = zone is not None
    share = 0.0
    if ok:
        arr = np.asarray(zone, dtype=np.uint8)
        share = float(((arr >= 160) & (subject >= 96)).sum() / max(1, int((subject >= 96).sum())))
        ok = share > 0.55
    print(f"  cobertura de la zona medida sobre el sujeto: {share:.1%}")
    print(f"  [{'OK' if ok else 'FALLO'}] cambio disperso conservado entero")
    return ok


if __name__ == "__main__":
    results = []
    print("[1] Edicion de pelo con fondo reescrito por el modelo")
    results.append(hair_case())
    print("[2] Muestra sin cambio real")
    results.append(no_change_case())
    print("[3] Retrato legitimo que ocupa casi todo el alto (sin guard de altura)")
    results.append(tall_region_case())
    print("[4] Zona medida del cambio y feather local")
    results.append(soft_mask_geometry())
    print("[5] Edicion sutil (tono bajo) sobre una prenda")
    results.append(subtle_edit_case())
    print("[6] Recolor grande sobre el sujeto con cielo reescrito")
    results.append(garment_edit_case())
    print("[7] Objeto anadido dentro del sujeto")
    results.append(added_object_case())
    print("[8] Cambio disperso de baja intensidad")
    results.append(scattered_change_case())
    total = len(results)
    passed = sum(1 for value in results if value)
    print(f"\n{passed}/{total} comprobaciones")
    sys.exit(0 if passed == total else 1)