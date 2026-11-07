#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fail→pass: pegado de región medida entera + residual que no se acepta.

Casos reales 2026-11-06 (log ui_start_20261106_001124):
  - ropa: zona pedida ~56 % → pegado Otsu 2,9 % (vestido intacto)
  - zapatos: residual 46 % anotado y aun así «Validación visual OK»

Contrato (CPU, sin GPU/Comfy):
  1. Con request_region medida, `_body_paste_zone` pega esa región entera
     (no solo picos Otsu), tras acreditar que hubo cambio en ella.
  2. Sin request_region, sigue person ∩ cambio (no ensancha colateral).
  3. Residual ≥40 % marca TARGET_RESIDUAL_FAIL; expansión crece dentro del
     soporte; el helper de nota detecta el fallo para el caller.
  4. Sin listas de prendas: la expansión usa solo geometría medida.

Uso:
    python -u diagnostics/probe_editor_measured_paste_and_residual_20261106.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor import selected_region_edit as sre  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    status = "PASS" if condition else "FAIL"
    print(f"{status} — {name}" + (f" | {detail}" if detail else ""))


def _photo():
    img = Image.new("RGB", (120, 160), (240, 240, 240))
    px = img.load()
    # Subject torso (blue cloth) and lower band (dark "object").
    for y in range(40, 110):
        for x in range(30, 90):
            px[x, y] = (40, 80, 180)
    for y in range(110, 145):
        for x in range(35, 85):
            px[x, y] = (20, 20, 20)
    return img


def _subject_mask(size):
    m = Image.new("L", size, 0)
    m.paste(255, (30, 40, 90, 145))
    return m


def _request_lower(size):
    m = Image.new("L", size, 0)
    m.paste(255, (35, 110, 85, 145))
    return m


class _StaticMasker:
    def __init__(self, residual_map):
        self.residual_map = residual_map

    def generate_mask(self, image, prompt, threshold=0.4):
        arr = np.zeros((image.height, image.width), np.uint8)
        key = (prompt or "").strip().lower()
        box = self.residual_map.get(key)
        if box is not None:
            x1, y1, x2, y2 = box
            arr[y1:y2, x1:x2] = 255
        return Image.fromarray(arr, "L")


def test_paste_full_request_region(manager):
    photo = _photo()
    subject = _subject_mask(photo.size)
    request = _request_lower(photo.size)
    # Generated: lower band becomes skin, plus a tiny speck change in torso
    # that Otsu would prefer if we only pasted peaks.
    generated = photo.copy()
    gpx = generated.load()
    for y in range(110, 145):
        for x in range(35, 85):
            gpx[x, y] = (210, 170, 140)
    for y in range(50, 55):
        for x in range(50, 55):
            gpx[x, y] = (255, 0, 0)

    # Script segmenter: request texts map to lower band; state on generated
    # still lower band (skin is not the query — return same box).
    manager._get_clipseg_masker = lambda: _StaticMasker({
        "change lower": (35, 110, 85, 145),
        "the person": (30, 40, 90, 145),
    })
    # `_request_region_on_subject` uses the masker; stub it to the measured part.
    manager._request_region_on_subject = lambda *_a, **_k: request

    zone = manager._body_paste_zone(
        photo, generated, subject, request, ["change lower"], (0, 0, 120, 160),
    )
    check("con región pedida hay zona", zone is not None)
    if zone is None:
        return
    z = np.asarray(zone.convert("L")) >= 160
    req = np.asarray(request.convert("L")) >= 160
    coverage = float((z & req).sum()) / float(max(1, req.sum()))
    check(
        "pega la región pedida entera (≥95 %), no solo picos",
        coverage >= 0.95,
        f"coverage={coverage:.3f} paste_px={int(z.sum())} req_px={int(req.sum())}",
    )
    # Torso speck must stay out: outside request region.
    check("colateral fuera de la región no entra", int(z[52, 52]) == 0)


def test_paste_without_request_keeps_change_only(manager):
    photo = _photo()
    subject = _subject_mask(photo.size)
    generated = photo.copy()
    gpx = generated.load()
    for y in range(110, 145):
        for x in range(35, 85):
            gpx[x, y] = (210, 170, 140)
    # Also repaint most of the subject — collateral.
    for y in range(40, 100):
        for x in range(30, 90):
            gpx[x, y] = (200, 30, 160)

    manager._request_region_on_subject = lambda *_a, **_k: None
    zone = manager._body_paste_zone(
        photo, generated, subject, None, ["go bare"], (0, 0, 120, 160),
    )
    check("sin región pedida hay zona de cambio", zone is not None)
    if zone is None:
        return
    z = np.asarray(zone.convert("L")) >= 160
    # Must not equal the entire subject (would paste dress collateral whole).
    subj = np.asarray(subject.convert("L")) >= 160
    ratio = float(z.sum()) / float(max(1, subj.sum()))
    check(
        "sin parte distinta no pega la persona entera",
        ratio < 0.95,
        f"ratio={ratio:.3f}",
    )


def test_residual_marker_and_expand():
    zone = np.zeros((80, 60), dtype=bool)
    zone[50:70, 20:40] = True
    support = np.zeros((80, 60), dtype=bool)
    support[30:75, 10:50] = True
    residual_high = int(zone.sum() * 0.5)
    residual_low = int(zone.sum() * 0.1)
    check(
        "residual 50 % falla",
        sre._residual_ratio_failed(residual_high, zone) is True,
    )
    check(
        "residual 10 % pasa",
        sre._residual_ratio_failed(residual_low, zone) is False,
    )
    note_fail = f"foo | el objetivo «x» sigue: {sre.TARGET_RESIDUAL_FAIL}"
    check(
        "nota residual detectada",
        sre._note_reports_residual_failure(note_fail) is True,
    )
    check(
        "nota limpia no falla",
        sre._note_reports_residual_failure("cambio real") is False,
    )
    expanded = sre._expand_zone_within_support(zone, support, grow_frac=0.35)
    check(
        "expansión crece dentro del soporte",
        int(expanded.sum()) > int(zone.sum()) and bool((expanded & ~support).sum() == 0),
        f"{int(zone.sum())} → {int(expanded.sum())}",
    )
    # No garment vocabulary in helper source — geometry only.
    import inspect
    src = inspect.getsource(sre._expand_zone_within_support)
    check(
        "expansión sin listas de prendas",
        "shoe" not in src.lower() and "zapato" not in src.lower()
        and "underwear" not in src.lower(),
    )


def main():
    manager = ImgEditorManager.__new__(ImgEditorManager)
    # Minimal stubs used by _changed_zone_on_subject / paste helpers.
    test_paste_full_request_region(manager)
    test_paste_without_request_keeps_change_only(manager)
    test_residual_marker_and_expand()
    failed = [name for name, ok, _ in RESULTS if not ok]
    print()
    if failed:
        print(f"FAIL: {len(failed)} checks: {failed}")
        return 1
    print(f"PASS: probe_editor_measured_paste_and_residual_20261106 ({len(RESULTS)} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
