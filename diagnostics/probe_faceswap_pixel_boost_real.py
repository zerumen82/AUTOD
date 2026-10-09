"""Compare real subpixel inference with interpolation, using installed models.

The subpixel sampling method is documented by FaceFusion:
https://github.com/facefusion/facefusion/blob/master/facefusion/processors/pixel_boost.py
Only this diagnostic overrides the swapper call; production is unchanged.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import cv2
import numpy as np
from insightface.utils import face_align

import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]


def boosted_run(engine, source, target, frame, paste_back=False):
    native = int(engine.model.input_size[0])
    size = engine.diagnostic_size
    factor = size // native
    assert size % native == 0 and not engine.source_is_image
    aligned, matrix = face_align.norm_crop2(frame, np.asarray(target.kps, np.float32), size)
    embedding = np.asarray(source.embedding, np.float32).reshape(1, -1)
    embedding /= np.linalg.norm(embedding)
    embedding = embedding @ engine.model.emap
    embedding /= np.linalg.norm(embedding)
    swapped = np.empty_like(aligned)
    for row in range(factor):
        for col in range(factor):
            sample = aligned[row::factor, col::factor]
            blob = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = blob.transpose(2, 0, 1)[None]
            output = engine.model.session.run(None, {
                engine.model.input_names[0]: blob,
                engine.model.input_names[1]: embedding,
            })[0][0].transpose(1, 2, 0)
            if output.max() < 2.0:
                output *= 255.0
            output = cv2.cvtColor(output.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            swapped[row::factor, col::factor] = output
    return engine.paste_back_robust(frame, swapped, matrix) if paste_back else (swapped, matrix)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, default=Path(r"D:\PROJECTS\uee\AUTO-DEEP\input\posti\1994859933.jpg"))
    parser.add_argument("--source", type=Path, default=Path(r"D:\PROJECTS\uee\FACES\MARETA\IMG-20240111-WA0000.jpg"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512])
    args = parser.parse_args()
    output = ROOT / "diagnostics" / "out" / f"faceswap_subpixel_{time.time_ns()}"
    output.mkdir(parents=True)
    frame = cv2.imread(str(args.target))
    assert frame is not None
    target = extract_face_images(str(args.target), (False, 0))[0][0]
    source = extract_face_images(str(args.source), (False, 0), is_source_face=True)[0][0]
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = True
    roop.globals.enhancer_blend_factor = 0.70
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [target], SimpleNamespace(blend_ratio=1.0, current_video_path=None), face_swap_mode="selected_faces")
    manager._is_single_photo = True
    engine = manager.processors["faceswap"]
    assert engine.model is not None
    original_run = engine.Run
    print("[PROBE] providers:", engine.model.session.get_providers(), flush=True)
    cv2.imwrite(str(output / "original.png"), frame)
    measurements = []
    for index, size in enumerate([0] + args.sizes, 1):
        label = "interpolated" if size == 0 else f"subpixel_{size}"
        engine.Run = original_run if size == 0 else MethodType(boosted_run, engine)
        engine.diagnostic_size = size
        ProcessMgr._swap_call_count = 0
        started = time.monotonic()
        result = manager._process_face_swap_v21(source, copy.deepcopy(target), frame.copy(), frame, photo_path=str(args.target))
        assert result is not None
        elapsed = time.monotonic() - started
        assert cv2.imwrite(str(output / f"{label}.png"), result)
        debug_dir = output / label
        debug_dir.mkdir()
        for artifact in (ROOT / "debug_swap").glob("*_f1.png"):
            shutil.copyfile(artifact, debug_dir / artifact.name)
        measurements.append({"label": label, "seconds": elapsed})
        print(f"[PROBE] {label} {elapsed:.2f}s -> {output / (label + '.png')}", flush=True)
    engine.Run = original_run
    (output / "meta.json").write_text(json.dumps({"source": str(args.source), "target": str(args.target), "measurements": measurements}, indent=2))
    print("[PROBE] output:", output, flush=True)


if __name__ == "__main__":
    main()
