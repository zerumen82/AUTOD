"""Check that an alignment rescue uses only this photo's selected face."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import roop.globals
from roop.ProcessMgr import ProcessMgr


def face():
    return SimpleNamespace(
        bbox=np.array([20, 20, 80, 90], dtype=np.float32),
        kps=np.array([[35, 40], [65, 40], [50, 55], [39, 74], [61, 74]], dtype=np.float32),
        embedding=np.ones(512, dtype=np.float32),
    )


def main():
    manager = ProcessMgr.__new__(ProcessMgr)
    manager.options = SimpleNamespace(current_video_path=None)
    target = face()
    original = np.zeros((120, 120, 3), dtype=np.uint8)
    current = np.full_like(original, 17)
    recovered = np.full_like(original, 51)
    wrong_path = r"D:\photos\another\same.jpg"
    right_path = r"D:\photos\selected\same.jpg"
    wrong_ref = {"source_path": wrong_path, "face_obj": face(), "embedding": target.embedding}
    right_ref = {"source_path": right_path, "face_obj": face(), "embedding": target.embedding}

    with patch.object(roop.globals, "selected_face_references", {"selected_face_ref_same.jpg": [wrong_ref]}, create=True):
        with patch.object(manager, "_process_face_swap_v21", return_value=recovered) as swap:
            result = manager._swap_from_saved_reference(
                None, target, current, original, 1, photo_path=right_path,
            )
            assert np.array_equal(result, current), "wrong photo reference was used or prior swap erased"
            swap.assert_not_called()

    with patch.object(roop.globals, "selected_face_references", {"selected_face_ref_same.jpg": [wrong_ref, right_ref]}, create=True):
        with patch.object(manager, "_process_face_swap_v21", return_value=recovered) as swap:
            result = manager._swap_from_saved_reference(
                None, target, current, original, 2, photo_path=right_path,
            )
            assert np.array_equal(result, recovered), "exact photo reference was not recovered"
            assert swap.call_count == 1
            assert swap.call_args.kwargs["photo_path"] == right_path

    bad_target = face()
    bad_target.kps[1] = bad_target.kps[0] + [1, 0]
    bad_ref = {"source_path": right_path, "face_obj": bad_target, "embedding": bad_target.embedding}
    with patch.object(roop.globals, "selected_face_references", {"selected_face_ref_same.jpg": [bad_ref, right_ref]}, create=True):
        with patch.object(manager, "_process_face_swap_v21", return_value=recovered) as swap:
            result = manager._swap_from_saved_reference(
                None, target, current, original, 3, photo_path=right_path,
            )
            assert np.array_equal(result, recovered), "degenerate first reference prevented valid later rescue"
            assert swap.call_count == 1
            assert np.array_equal(swap.call_args.args[1].kps, right_ref["face_obj"].kps)
    with patch.object(manager, "_swap_from_saved_reference", return_value=current) as rescue:
        result = ProcessMgr._process_face_swap_v21(
            manager, target, bad_target, current, original, photo_path=right_path,
        )
        assert np.array_equal(result, current)
        assert rescue.call_args.kwargs["photo_path"] == right_path
        assert rescue.call_args.args[2] is current

    print("OK: gate transmite el archivo exacto y conserva swaps previos del fotograma")


if __name__ == "__main__":
    main()
