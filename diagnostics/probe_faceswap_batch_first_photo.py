"""Render the documented malformed alignment from the 93-photo batch."""

from pathlib import Path
from types import SimpleNamespace
import copy

import cv2
import numpy as np

import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images, get_face_analyser
from roop.face_util_rotation import get_all_faces_smart
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(r"D:\PROJECTS\uee\AUTO-DEEP\input\posti\1994859933.jpg")
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\IMG-20240111-WA0000.jpg")
OUTPUT = ROOT / "diagnostics" / "faceswap_batch_first_rescued.jpg"
ROTATED_OUTPUT = ROOT / "diagnostics" / "faceswap_batch_first_rotated_landmarks.jpg"
NO_ENHANCER_OUTPUT = ROOT / "diagnostics" / "faceswap_batch_first_no_enhancer.jpg"
NO_MOUTH_OUTPUT = ROOT / "diagnostics" / "faceswap_batch_first_no_mouth_restore.jpg"


def eye_ratio(face):
    kps = np.asarray(face.kps, dtype=np.float32)
    box = np.asarray(face.bbox, dtype=np.float32)
    return float(np.linalg.norm(kps[1] - kps[0])) / float(box[2] - box[0])


def main():
    frame = cv2.imread(str(TARGET))
    assert frame is not None
    detected = get_all_faces_smart(frame, for_target=True, include_180=True)
    bad = min(detected, key=lambda face: eye_ratio(face))
    assert eye_ratio(bad) < 0.05, eye_ratio(bad)
    reference = extract_face_images(str(TARGET), (False, 0))[0][0]
    assert eye_ratio(reference) > 0.05, eye_ratio(reference)
    source = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)[0][0]

    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = True
    previous_refs = getattr(roop.globals, "selected_face_references", None)
    previous_enhancer = getattr(roop.globals, "use_enhancer", None)
    previous_mouth = getattr(roop.globals, "preserve_mouth_expression", None)
    roop.globals.selected_face_references = {
        f"selected_face_ref_{TARGET.name}": [{
            "source_path": str(TARGET), "face_obj": reference,
            "embedding": reference.embedding, "bbox": reference.bbox,
        }],
    }
    try:
        manager = ProcessMgr()
        manager.initialize(
            [FaceSet([source])], [reference],
            SimpleNamespace(blend_ratio=1.0, current_video_path=None),
            face_swap_mode="selected_faces",
        )
        manager._is_single_photo = True
        result = manager._process_face_swap_v21(
            source, bad, frame.copy(), frame, photo_path=str(TARGET),
        )
        assert cv2.imwrite(str(OUTPUT), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        box = np.asarray(reference.bbox, dtype=np.int32)
        x1, y1, x2, y2 = box
        changed = float(cv2.absdiff(result, frame)[y1:y2, x1:x2].mean())
        print(
            f"[PROBE] bad_eye_rel={eye_ratio(bad):.3f} "
            f"reference_eye_rel={eye_ratio(reference):.3f} "
            f"face_change={changed:.2f} output={OUTPUT}", flush=True,
        )

        roop.globals.use_enhancer = False
        unenhanced = manager._process_face_swap_v21(
            source, copy.deepcopy(reference), frame.copy(), frame,
            photo_path=str(TARGET),
        )
        assert cv2.imwrite(str(NO_ENHANCER_OUTPUT), unenhanced)
        roop.globals.use_enhancer = True
        roop.globals.preserve_mouth_expression = False
        no_mouth = manager._process_face_swap_v21(
            source, copy.deepcopy(reference), frame.copy(), frame,
            photo_path=str(TARGET),
        )
        assert cv2.imwrite(str(NO_MOUTH_OUTPUT), no_mouth)
        roop.globals.preserve_mouth_expression = previous_mouth
        print(
            f"[PROBE] no_enhancer={NO_ENHANCER_OUTPUT} "
            f"no_mouth_restore={NO_MOUTH_OUTPUT}", flush=True,
        )

        # Diagnosis only: see whether a modest camera rotation finds better
        # target landmarks while retaining the exact photo's mask bbox.
        height, width = frame.shape[:2]
        affine = cv2.getRotationMatrix2D((width / 2, height / 2), -30, 1.0)
        inverse = cv2.invertAffineTransform(affine)
        rotated = cv2.warpAffine(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), affine, (width, height),
        )
        alternatives = []
        for detected_face in get_face_analyser().get(rotated):
            points = cv2.transform(
                np.asarray(detected_face.kps, dtype=np.float32)[None], inverse,
            )[0]
            center = points.mean(axis=0)
            if (reference.bbox[0] < center[0] < reference.bbox[2]
                    and reference.bbox[1] < center[1] < reference.bbox[3]):
                alternatives.append((float(detected_face.det_score), points))
        if alternatives:
            confidence, points = max(alternatives, key=lambda item: item[0])
            rotated_reference = copy.deepcopy(reference)
            rotated_reference.kps = points
            rotated_result = manager._process_face_swap_v21(
                source, rotated_reference, frame.copy(), frame, photo_path=str(TARGET),
            )
            assert cv2.imwrite(
                str(ROTATED_OUTPUT), rotated_result,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )
            print(
                f"[PROBE] rotated_confidence={confidence:.2f} "
                f"rotated_eye_rel={eye_ratio(rotated_reference):.3f} "
                f"output={ROTATED_OUTPUT}", flush=True,
            )
    finally:
        roop.globals.selected_face_references = previous_refs
        roop.globals.use_enhancer = previous_enhancer
        roop.globals.preserve_mouth_expression = previous_mouth


if __name__ == "__main__":
    main()
