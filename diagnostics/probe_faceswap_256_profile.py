"""Compare the installed 256 image-source swapper on the difficult profile."""

from pathlib import Path
from types import SimpleNamespace
import sys

import cv2
import onnxruntime as ort

import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(r"D:\PROJECTS\uee\AUTO-DEEP\input\posti\1994859933.jpg")
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\IMG-20240111-WA0000.jpg")
MODEL = "inswapper_128.onnx" if "--standard-128" in sys.argv else "inswapper_256.onnx"
OUTPUT = ROOT / "diagnostics" / "out" / (
    "faceswap_codeformer_profile_pipeline.jpg"
    if "--codeformer" in sys.argv else (
        "faceswap_standard_128_profile_pipeline.jpg"
        if MODEL == "inswapper_128.onnx" else "faceswap_256_profile_pipeline.jpg"
    )
)


def main() -> None:
    frame = cv2.imread(str(TARGET))
    assert frame is not None
    target = extract_face_images(str(TARGET), (False, 0))[0][0]
    source = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)[0][0]
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = "CodeFormer" if "--codeformer" in sys.argv else "GFPGAN"
    roop.globals.use_enhancer = True
    manager = ProcessMgr()
    manager.initialize(
        [FaceSet([source])], [target],
        SimpleNamespace(blend_ratio=1.0, current_video_path=None),
        face_swap_mode="selected_faces",
    )
    if "--codeformer" in sys.argv:
        assert "enhance_codeformer" in manager.processors, "CodeFormer did not initialize"
    manager._is_single_photo = True
    engine = manager.processors["faceswap"]
    engine.model.session = ort.InferenceSession(
        str(ROOT / "models" / MODEL),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    engine.is_256 = MODEL == "inswapper_256.onnx"
    engine.source_is_image = engine.is_256
    print("providers:", engine.model.session.get_providers(), flush=True)
    result = manager._process_face_swap_v21(
        source, target, frame.copy(), frame, photo_path=str(TARGET),
    )
    assert cv2.imwrite(str(OUTPUT), result)
    print("output:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
