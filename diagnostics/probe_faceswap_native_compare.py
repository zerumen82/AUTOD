"""Compare the current profile composition with a direct model paste."""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from roop.face_util import extract_face_images
from roop.face_util_rotation import get_all_faces_smart
from roop.processors.FaceSwap import FaceSwap


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(r"D:\PROJECTS\uee\AUTO-DEEP\input\posti\1994859933.jpg")
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\IMG-20240111-WA0000.jpg")
OUT = ROOT / "diagnostics" / "out"


def main() -> None:
    frame = cv2.imread(str(TARGET))
    assert frame is not None
    target = extract_face_images(str(TARGET), (False, 0))[0][0]
    source = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)[0][0]
    target.kps = np.asarray(target.kps, dtype=np.float32)
    source.kps = np.asarray(source.kps, dtype=np.float32)
    source.embedding = np.asarray(source.embedding, dtype=np.float32)
    engine = FaceSwap()
    engine.Initialize({"devicename": "cuda"})
    assert engine.model is not None
    print("providers:", engine.model.session.get_providers(), flush=True)
    robust = engine.Run(source, target, frame.copy(), paste_back=True)
    assert robust is not None
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / "faceswap_robust_paste.jpg"), robust)
    model_256 = ROOT / "models" / "inswapper_256.onnx"
    engine.model.session = ort.InferenceSession(
        str(model_256), providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    engine.is_256 = True
    engine.source_is_image = True
    candidate_256 = engine.Run(source, target, frame.copy(), paste_back=True)
    assert candidate_256 is not None
    cv2.imwrite(str(OUT / "faceswap_256_paste.jpg"), candidate_256)
    box = np.asarray(target.bbox, dtype=int)
    x1, y1, x2, y2 = box
    for label, image in (("robust", robust), ("256", candidate_256)):
        change = cv2.absdiff(frame, image)[y1:y2, x1:x2].mean()
        print(f"{label} face_change={change:.2f}", flush=True)


if __name__ == "__main__":
    main()
