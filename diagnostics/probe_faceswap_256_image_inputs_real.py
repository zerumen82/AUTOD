"""Exercise the installed image-input ONNX with full source coordinates.

Production's embedding wrapper drops the source image. Its image-input branch
also supplies 0..255 although this ONNX starts by subtracting/dividing by 0.5.
This diagnostic bypasses those two input mistakes; it does not change production.
Run with CUDA_VISIBLE_DEVICES=-1 for the CPU pipeline.
"""
import argparse
import json
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align

import roop.face_util
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.types import FaceSet

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(r"D:\PROJECTS\uee\AUTO-DEEP\input\posti\1994859933.jpg")
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\IMG-20240111-WA0000.jpg")

# Model alignment data from FaceFusion's face_helper.py, not a production setting.
FFHQ_TEMPLATE = np.array([
    [0.37691676, 0.46864664], [0.62285697, 0.46912813],
    [0.50123859, 0.61331904], [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
], dtype=np.float32)


def aligned_crop(photo, landmarks, alignment):
    landmarks = np.asarray(landmarks, np.float32)
    if alignment == "arcface":
        return face_align.norm_crop2(photo, landmarks, 256)
    matrix, _ = cv2.estimateAffinePartial2D(
        landmarks, FFHQ_TEMPLATE * 256, method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )
    assert matrix is not None and np.isfinite(matrix).all()
    crop = cv2.warpAffine(photo, matrix, (256, 256),
                         borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_AREA)
    return crop, matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", choices=("arcface", "ffhq"), default="arcface")
    args = parser.parse_args()
    analyser = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models"),
        allowed_modules=["detection", "recognition", "landmark_2d_106"],
        providers=["CPUExecutionProvider"])
    analyser.prepare(ctx_id=-1, det_size=(1024, 1024), det_thresh=0.1)
    roop.face_util.FACE_ANALYSER = analyser
    target = extract_face_images(str(TARGET), (False, 0))[0][0]
    source = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)[0][0]
    frame, source_photo = cv2.imread(str(TARGET)), cv2.imread(str(SOURCE))
    assert frame is not None and source_photo is not None
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = True
    roop.globals.enhancer_blend_factor = 0.70
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [target], SimpleNamespace(blend_ratio=1.0,
        current_video_path=None), face_swap_mode="selected_faces")
    manager._is_single_photo = True
    engine = manager.processors["faceswap"]
    session = ort.InferenceSession(str(ROOT / "models/inswapper_256.onnx"),
        providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    assert [i.name for i in inputs] == ["target", "source"]
    assert all(i.shape == [1, 3, 256, 256] for i in inputs)
    source_crop, _ = aligned_crop(source_photo, source.kps, args.alignment)
    output = ROOT / "diagnostics/out" / f"faceswap_256_inputs_{time.time_ns()}"
    output.mkdir(parents=True)
    original_run = engine.Run
    count = 0
    measurements = []
    for colour in ("baseline128", "rgb", "bgr"):
        def image_run(_engine, _source, destination, photo, paste_back=False):
            nonlocal count
            aligned, matrix = aligned_crop(photo, destination.kps, args.alignment)
            def blob(image):
                if colour == "rgb":
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return (image.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
            result = session.run(None, {"target":blob(aligned), "source":blob(source_crop)})[0][0]
            assert np.isfinite(result).all()
            result = (result.transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            if colour == "rgb":
                result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            count += 1
            return result, matrix
        engine.Run = original_run if colour == "baseline128" else MethodType(image_run, engine)
        before = count
        ProcessMgr._swap_call_count = 0
        result = manager._process_face_swap_v21(source, target, frame.copy(), frame, photo_path=str(TARGET))
        assert result is not None
        if colour != "baseline128":
            assert count > before, "ONNX image-input inference did not execute"
        cv2.imwrite(str(output / f"{colour}.png"), result)
        measurements.append({"colour":colour, "alignment":args.alignment, "onnx_calls":count-before,
            "changed_over10":float((cv2.absdiff(result, frame).max(axis=2)>10).mean())})
    engine.Run = original_run
    (output / "meta.json").write_text(json.dumps(measurements, indent=2))
    print("[PROBE] output:", output, measurements, flush=True)


if __name__ == "__main__":
    main()
