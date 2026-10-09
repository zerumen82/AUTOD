"""Compare an explicitly installed Hyperswap model with the current 128 model.

No downloads and no production edits. Input/output normalization follows:
https://github.com/facefusion/facefusion/blob/master/facefusion/processors/modules/face_swapper/core.py
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

# ProcessMgr chooses its device from PyTorch availability, so provider globals
# alone do not force its baseline, masker or enhancer onto CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("--landmarks", choices=("reference", "68-rgb"), default="reference")
    parser.add_argument("--no-enhancer", action="store_true")
    args = parser.parse_args()
    model = Path(args.model)
    assert model.is_file(), "Install only after explicit user approval"
    root = Path(__file__).resolve().parents[1]
    analyser = FaceAnalysis(name="buffalo_l", root=str(root / "models"),
                           allowed_modules=["detection", "recognition", "landmark_2d_106", "landmark_3d_68"],
                           providers=["CPUExecutionProvider"])
    analyser.prepare(ctx_id=-1, det_size=(1024, 1024), det_thresh=0.1)
    roop.face_util.FACE_ANALYSER = analyser
    roop.globals.execution_providers = ["CPUExecutionProvider"]
    source = extract_face_images(args.source, (False, 0), is_source_face=True)[0][0]
    target = extract_face_images(args.target, (False, 0))[0][0]
    frame = cv2.imread(args.target)
    assert frame is not None
    if args.landmarks == "68-rgb":
        detected = analyser.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        assert detected, "No measured 68-point face"
        measured = min(detected, key=lambda face: np.linalg.norm(face.bbox - target.bbox))
        points = np.asarray(measured.landmark_3d_68, np.float32)[:, :2]
        target.kps = np.array([points[36:42].mean(axis=0), points[42:48].mean(axis=0),
                               points[30], points[48], points[54]], np.float32)
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = not args.no_enhancer
    roop.globals.enhancer_blend_factor = .70
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [target], SimpleNamespace(
        blend_ratio=1.0, current_video_path=None), face_swap_mode="selected_faces")
    manager._is_single_photo = True
    engine = manager.processors["faceswap"]
    assert engine.devicename == "cpu", "Comparison must release the shared GPU"
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inputs = {item.name: item for item in session.get_inputs()}
    assert set(inputs) == {"source", "target"}, list(inputs)
    embedding = np.asarray(source.embedding, np.float32).reshape(1, -1)
    embedding /= np.linalg.norm(embedding)
    count = 0

    def new_run(_engine, _source, destination, photo, paste_back=False):
        nonlocal count
        aligned, matrix = face_align.norm_crop2(photo, np.asarray(destination.kps, np.float32), 256)
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        feed = {"target": ((rgb - .5) / .5).transpose(2, 0, 1)[None], "source": embedding}
        for name, value in feed.items():
            assert inputs[name].type in ("tensor(float)", "tensor(float16)")
            feed[name] = value.astype(np.float16 if inputs[name].type == "tensor(float16)" else np.float32)
        swapped = session.run(None, feed)[0][0]
        assert np.isfinite(swapped).all()
        swapped = ((swapped.transpose(1, 2, 0) * .5 + .5) * 255).clip(0, 255).astype(np.uint8)
        count += 1
        swapped = cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)
        assert cv2.imwrite(str(output / f"aligned_input_{count}.png"), aligned)
        assert cv2.imwrite(str(output / f"raw_hyperswap_{count}.png"), swapped)
        return swapped, matrix

    output = root / "diagnostics/out" / f"faceswap_hyperswap_{time.time_ns()}"
    output.mkdir()
    assert cv2.imwrite(str(output / "original.png"), frame)
    original_run = engine.Run
    measurements = []
    try:
        for label in ("baseline128", "hyperswap256"):
            engine.Run = original_run if label == "baseline128" else MethodType(new_run, engine)
            ProcessMgr._swap_call_count = 0
            before = count
            started = time.perf_counter()
            result = manager._process_face_swap_v21(source, target, frame.copy(), frame,
                                                   photo_path=args.target)
            assert result is not None
            if label == "hyperswap256":
                assert count > before, "New ONNX inference did not run"
            assert cv2.imwrite(str(output / f"{label}.png"), result)
            measurements.append({"label": label, "seconds": time.perf_counter() - started,
                                 "onnx_calls": count - before,
                                 "changed_fraction": float((cv2.absdiff(result, frame).max(axis=2) > 10).mean())})
    finally:
        engine.Run = original_run
    digest = hashlib.sha256()
    with model.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    (output / "meta.json").write_text(json.dumps({"model": str(model), "source": args.source,
        "target": args.target, "sha256": digest.hexdigest(),
        "landmarks": args.landmarks, "kps": np.asarray(target.kps).tolist(),
        "enhancer": None if args.no_enhancer else "GFPGAN", "enhancer_blend": roop.globals.enhancer_blend_factor,
        "device": engine.devicename,
        "measurements": measurements}, indent=2), encoding="utf-8")
    print(output, measurements, flush=True)


if __name__ == "__main__":
    main()
