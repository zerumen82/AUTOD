"""Ejecuta main() del probe compound con espía en _body_paste_zone.

Guarda generated/subject/box/zona de cada llamada para inspeccionar la del
caso skin (la que falla en la corrida completa pero no aislada).
"""
import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

spec = importlib.util.spec_from_file_location(
    "cfd", ROOT / "diagnostics" / "probe_editor_compound_fast_dispatch.py")
cfd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cfd)

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

OUT = ROOT / "diagnostics" / "out" / "halo_stages_20261013" / "skin_probe_spy"
OUT.mkdir(parents=True, exist_ok=True)

real_vrq = ImgEditorManager._visual_request_queries
real_rros = ImgEditorManager._request_region_on_subject


def vrq_spy(self, texts):
    queries = real_vrq(self, texts)
    print(f"[VRQ] texts={[str(t)[:40] for t in texts]} -> {queries}", flush=True)
    return queries


def rros_spy(self, image, subject_mask, texts):
    result = real_rros(self, image, subject_mask, texts)
    print(f"[RROS] texts={[str(t)[:40] for t in texts]} -> "
          f"{'None' if result is None else 'region'}", flush=True)
    return result


ImgEditorManager._visual_request_queries = vrq_spy
ImgEditorManager._request_region_on_subject = rros_spy

real_zone = ImgEditorManager._body_paste_zone
counter = {"n": 0}


def spy(self, source, generated, subject_mask, request_region, texts, box):
    zone = real_zone(self, source, generated, subject_mask, request_region, texts, box)
    n = counter["n"]
    counter["n"] += 1
    try:
        source.convert("RGB").save(OUT / f"{n}_source.png")
        generated.convert("RGB").save(OUT / f"{n}_generated.png")
        subject_mask.convert("L").save(OUT / f"{n}_subject.png")
        if request_region is not None:
            request_region.convert("L").save(OUT / f"{n}_request.png")
        if zone is not None:
            zone.convert("L").save(OUT / f"{n}_zone.png")
            px = int((np.asarray(zone.convert("L")) >= 160).sum())
        else:
            px = -1
        print(f"[SPY {n}] box={box} texts={texts} zona={px}px", flush=True)
    except Exception as exc:
        print(f"[SPY {n}] dump error: {exc}", flush=True)
    return zone


try:
    with patch.object(ImgEditorManager, "_body_paste_zone", new=spy):
        cfd.main()
except AssertionError:
    print(f"[SPY] AssertionError tras {counter['n']} llamadas", flush=True)
    raise
