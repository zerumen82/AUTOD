"""Check photo selection decisions without loading FaceSwap or occupying the GPU."""
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import roop.globals
import roop.face_util_rotation as rotation
import roop.face_util as face_util
from roop.ProcessMgr import ProcessMgr


ROOT = Path(__file__).resolve().parents[1]
FRAME = np.zeros((500, 600, 3), dtype=np.uint8)
PHOTO = str(ROOT / "diagnostics" / "selected_photo.jpg")
OTHER = str(ROOT / "diagnostics" / "other_photo.jpg")
SOURCE = SimpleNamespace(embedding=np.array([0.0, 0.0, 1.0], dtype=np.float32))


def face(box, embedding):
    x1, y1, x2, y2 = box
    w, h = x2-x1, y2-y1
    kps = [
        [x1+w*.3, y1+h*.35], [x1+w*.7, y1+h*.35],
        [x1+w*.5, y1+h*.57], [x1+w*.36, y1+h*.78],
        [x1+w*.64, y1+h*.78],
    ]
    return SimpleNamespace(bbox=box, kps=kps, embedding=np.asarray(embedding), det_score=.95)


def run_case(name, path, detected, references, expect_swap, expect_box=None, expect_boxes=None):
    manager = ProcessMgr.__new__(ProcessMgr)
    manager.is_initialized = True
    manager.processors = {"faceswap": object()}
    sources = [SOURCE]
    manager.input_facesets = [SimpleNamespace(faces=[src]) for src in sources]
    manager.options = SimpleNamespace(current_video_path=None, face_swap_mode="selected_faces", blend_ratio=1.0)
    manager._select_source_face = lambda *args, **kwargs: SOURCE
    selected = []
    selected_sources = []

    def fake_swap(source, target, result, original, *args, **kwargs):
        selected.append(target)
        selected_sources.append(source)
        output = result.copy()
        output[0, 0] = [255, 255, 255]
        return output

    manager._process_face_swap_v21 = fake_swap
    old_refs = getattr(roop.globals, 'selected_face_references', None)
    old_targets = getattr(roop.globals, 'TARGET_FACES', None)
    old_random = getattr(roop.globals, 'source_random', None)
    old_detector = rotation.get_all_faces_smart
    old_analyser = face_util.get_face_analyser
    old_rotation_analyser = rotation.get_face_analyser
    try:
        roop.globals.selected_face_references = references
        roop.globals.TARGET_FACES = [UI_FACE]
        roop.globals.source_random = False
        rotation.get_all_faces_smart = lambda *args, **kwargs: detected
        face_util.get_face_analyser = lambda: None
        rotation.get_face_analyser = lambda: None
        result = manager.process_frame(FRAME.copy(), file_path=path)
        assert bool(selected) is expect_swap, f"{name}: selected={selected}"
        assert bool(result[0, 0].any()) is expect_swap, f"{name}: output unchanged unexpectedly"
        if expect_box is not None:
            assert np.array_equal(np.asarray(selected[0].bbox), expect_box), name
        if expect_boxes is not None:
            assert len(selected) == len(expect_boxes), f"{name}: {len(selected)} swaps"
            assert all(np.array_equal(np.asarray(face.bbox), box) for face, box in zip(selected, expect_boxes)), name
        if expect_swap:
            assert selected_sources == [SOURCE] * len(selected), f"{name}: wrong source"
        print(f"[PROBE] {name}: {'swap' if expect_swap else 'no swap'}", flush=True)
    finally:
        roop.globals.selected_face_references = old_refs
        roop.globals.TARGET_FACES = old_targets
        roop.globals.source_random = old_random
        rotation.get_all_faces_smart = old_detector
        face_util.get_face_analyser = old_analyser
        rotation.get_face_analyser = old_rotation_analyser


UI_FACE = face([100, 100, 220, 250], [1.0, 0.0, 0.0])
EXACT_REF = {
    'face_obj': UI_FACE, 'bbox': UI_FACE.bbox, 'embedding': UI_FACE.embedding,
    'source_path': os.path.normcase(os.path.abspath(PHOTO)),
}
EXACT = {f"selected_face_ref_{os.path.basename(PHOTO)}": [EXACT_REF]}
WRONG = face([370, 100, 490, 250], [0.0, 1.0, 0.0])
SAME_PLACE = face([105, 105, 225, 255], [0.0, 1.0, 0.0])

run_case("same_file_low_similarity", PHOTO, [SAME_PLACE], EXACT, True, SAME_PLACE.bbox)
run_case("same_file_redetection_empty", PHOTO, [], EXACT, True, UI_FACE.bbox)
run_case("batch_one_face_low_similarity", OTHER, [WRONG], EXACT, True, WRONG.bbox)
run_case("batch_multiple_faces_ambiguous", OTHER, [SAME_PLACE, WRONG], EXACT, False)
run_case("same_file_multiple_faces_position", PHOTO, [WRONG, SAME_PLACE], EXACT, True, SAME_PLACE.bbox)
both_refs = {f"selected_face_ref_{os.path.basename(PHOTO)}": [
    EXACT_REF,
    {'face_obj': WRONG, 'bbox': WRONG.bbox, 'embedding': WRONG.embedding,
     'source_path': os.path.normcase(os.path.abspath(PHOTO))},
]}
run_case("two_selected_destinations_same_photo", PHOTO, [SAME_PLACE, WRONG], both_refs,
         True, expect_boxes=[SAME_PLACE.bbox, WRONG.bbox])
other_path_same_name = str(ROOT / "diagnostics" / "elsewhere" / "selected_photo.jpg")
other_ref = {
    'face_obj': WRONG, 'bbox': WRONG.bbox, 'embedding': WRONG.embedding,
    'source_path': os.path.normcase(os.path.abspath(other_path_same_name)),
}
same_name_bank = {f"selected_face_ref_{os.path.basename(PHOTO)}": [EXACT_REF, other_ref]}
run_case("same_basename_different_folder", PHOTO, [WRONG, SAME_PLACE], same_name_bank, True, SAME_PLACE.bbox)

import ui.tabs.faceswap.events as events
import ui.tabs.faceswap.logic as logic

assert events._face_detection_key(PHOTO, 1) != events._face_detection_key(OTHER, 1)
assert events._face_detection_key(PHOTO, 1) != events._face_detection_key(PHOTO, 2)

old_refs = roop.globals.selected_face_references
try:
    roop.globals.selected_face_references = {}
    array_face = face(np.array([100, 100, 220, 250]), [1.0, 0.0, 0.0])
    assert events._store_destination_reference("selected_photo.jpg", array_face, PHOTO) == (1, "ok")
    assert events._store_destination_reference("selected_photo.jpg", array_face, PHOTO) == (1, "dup")
    assert events._store_destination_reference("selected_photo.jpg", array_face, other_path_same_name) == (1, "ok")
    assert len(events._get_file_bank("selected_photo.jpg", PHOTO)) == 1
    assert len(events._get_file_bank("selected_photo.jpg", other_path_same_name)) == 1
    assert "Origen elegido" not in logic.update_pagination_info([None, None], "input")
    print("[PROBE] UI bank ndarray duplicate and no source selection indicator: pass", flush=True)
finally:
    roop.globals.selected_face_references = old_refs

print("[PROBE] photo selection 9/9", flush=True)
