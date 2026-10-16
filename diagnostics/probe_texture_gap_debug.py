"""Instrumenta _compensate_face_sharpness en una composición real (03429).

Guarda el parche de entrada, la referencia alineada que mide la función y
la M recibida, para localizar por qué la medición alineada (reporta 72%)
no coincide con la rejilla en marco (44%). CPU, sin GPU, sin UI.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location(
    "comp_probe", ROOT / "diagnostics" / "probe_faceswap_measured_composition_real.py")
comp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comp)

import roop.face_util as face_util
import roop.globals
from roop.ProcessMgr import ProcessMgr


def main():
    source_path = ROOT / "diagnostics" / "_texture_src.jpg"
    target_path = ROOT / "diagnostics" / "_texture_tgt.jpg"
    out = ROOT / "diagnostics" / "out" / "halo_stages_20261013" / "texture_gap_debug"
    out.mkdir(parents=True, exist_ok=True)
    meta_prev = json.load(open(
        ROOT / "diagnostics" / "out" / "halo_stages_20261013"
        / "composition_cpu_1791904675616500100" / "meta.json", encoding="utf-8"))
    # Copias de trabajo locales no son necesarias: usar las fotos directamente.
    source_path = Path(r"D:\PROJECTS\uee\FACES\MARETA\20230607_214550.jpg")
    target_path = Path(r"D:\PROJECTS\uee\FACES\MARETA\03429.jpg")

    captures = []
    orig_comp = ProcessMgr._compensate_face_sharpness

    def recording(patch, original_frame, M):
        ref = cv2.warpAffine(
            original_frame, M, (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )
        cap = {
            "patch_shape": list(patch.shape),
            "M": np.asarray(M).tolist(),
            "frame_shape": list(original_frame.shape),
        }
        cv2.imwrite(str(out / "ref.png"), ref)
        cv2.imwrite(str(out / "patch_in.png"), patch)
        result, note = orig_comp(patch, original_frame, M)
        cv2.imwrite(str(out / "patch_out.png"), result)
        cap["note"] = note
        cap["lap_patch_in"] = comp_module_lap(patch)
        cap["lap_ref"] = comp_module_lap(ref)
        cap["lap_patch_out"] = comp_module_lap(result)
        m128 = np.asarray(meta_prev["cases"][0]["expected_matrix"], float)
        cap["M_over_expected_x4_maxdiff"] = float(
            np.max(np.abs(np.asarray(M, float) - m128 * 4.0)))
        captures.append(cap)
        return result, note

    def comp_module_lap(img):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return float(cv2.Laplacian(g, cv2.CV_64F).var())

    ProcessMgr._compensate_face_sharpness = staticmethod(recording)
    try:
        with comp.persist_metadata({"captures": captures}, out):
            with comp.cpu_sessions_only():
                analyser = face_util.get_face_analyser_cpu()
                face_util.FACE_ANALYSER = analyser
                roop.globals.execution_providers = ["CPUExecutionProvider"]
                roop.globals.selected_enhancer = "GFPGAN"
                roop.globals.use_enhancer = True
                roop.globals.enhancer_blend_factor = 0.70
                source_faces = face_util.extract_face_images(str(source_path), is_source_face=True)
                source, _ = source_faces[0]
                faces = face_util.extract_face_images(str(target_path))
                target, _ = faces[0]
                frame = cv2.imdecode(np.fromfile(str(target_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                comp.run_variant(
                    ProcessMgr, source, target, frame, target_path, out,
                    "dbg", 1.0, 0.70,
                )
    finally:
        ProcessMgr._compensate_face_sharpness = orig_comp
    print(json.dumps(captures, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
