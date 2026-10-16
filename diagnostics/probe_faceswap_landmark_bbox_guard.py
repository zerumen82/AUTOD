"""Logged 2026-10-13 geometry regression; CPU contracts, no model inference.

--baseline executes HEAD definitions at the affected entry points. Frames
are constant arrays. Sampling and profile classification are intercepted at
the neural boundary; reference selection, validation and alignment are real.
"""
import ast
import contextlib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
process_module = importlib.import_module("roop.ProcessMgr")
warp_module = importlib.import_module("roop.processors.FaceSwap")
rotation_module = importlib.import_module("roop.face_util_rotation")
config_module = importlib.import_module("roop.faceswap_config")
face_module = importlib.import_module("roop.face_util")
globals_module = importlib.import_module("roop.globals")
face_align = importlib.import_module("insightface.utils.face_align")
BASELINE = "--baseline" in sys.argv
hashes = {}
unavailable_baseline = []


def actual_definition(module, path, name):
    source = (subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT).decode("utf-8")
              if BASELINE else (ROOT / path).read_text(encoding="utf-8"))
    hashes[path] = hashlib.sha256(source.encode()).hexdigest()
    tree = ast.parse(source)
    node = next((n for n in ast.walk(tree) if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name), None)
    if node is None and BASELINE and name == "_filter_false_positive_faces":
        unavailable_baseline.append(f"{path}:{name} is absent from HEAD")
        return None
    if node is None:
        raise RuntimeError(f"definition absent: {path}:{name}")
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), vars(module))
    return vars(module)[name]


Process = actual_definition(process_module, "roop/ProcessMgr.py", "ProcessMgr")
Warp = actual_definition(warp_module, "roop/processors/FaceSwap.py", "FaceWarpEngine")
filter_faces = actual_definition(rotation_module, "roop/face_util_rotation.py", "_filter_false_positive_faces")
BAD_BOX = np.array([286, 142, 1040, 1067], dtype=np.float32)
BAD_POINTS = np.array([[587.97095, 457.11865], [735.9153, 457.13428],
                       [587.43286, 826.53345], [568.89087, 1236.5726],
                       [638.0598, 1207.3059]], dtype=np.float32)
ORIGINAL = np.zeros((2000, 3000, 3), dtype=np.uint8)
PRIOR = ORIGINAL.copy()
PRIOR[20:30, 20:30] = 17
EXACT_PATH = "D:/fixture/110276779.jpg"


def face(points=BAD_POINTS, bounds=BAD_BOX):
    return SimpleNamespace(kps=np.asarray(points, dtype=np.float32).copy(),
                           bbox=np.asarray(bounds, dtype=np.float32).copy(),
                           embedding=np.ones(512, dtype=np.float32), det_score=0.9)


def good_face():
    center = (BAD_BOX[:2] + BAD_BOX[2:]) / 2
    points = (face_align.arcface_dst - face_align.arcface_dst.mean(axis=0)) * 10 + center
    return face(points)


def manager():
    instance = Process.__new__(Process)
    instance.options = SimpleNamespace(current_video_path=None, blend_ratio=1.0)
    return instance


def shared_shape_and_config():
    valid = good_face()
    assert config_module.landmarks_fit_bbox(valid.kps, valid.bbox)
    for points, box in ((BAD_POINTS, BAD_BOX), (None, BAD_BOX), (np.zeros((4, 2)), BAD_BOX),
                        (np.full((5, 2), np.nan), BAD_BOX), (valid.kps, [2, 2, 1, 1]),
                        (valid.kps, [0, 0, np.inf, 100]), (valid.kps, [0, 0, 100])):
        assert not config_module.landmarks_fit_bbox(points, box), (points, box)
    settings = config_module.get_faceswap_config()
    assert settings["landmark_bbox_margin_fraction"] >= 0
    assert settings["landmark_bbox_margin_pixels"] >= 0
    for field in ("landmark_bbox_margin_fraction", "landmark_bbox_margin_pixels"):
        for value in (-1, np.inf, np.nan):
            with patch.object(config_module, "get_faceswap_config", return_value={**settings, field: value}):
                try:
                    config_module.landmarks_fit_bbox(valid.kps, valid.bbox)
                except ValueError:
                    pass
                else:
                    raise AssertionError(f"invalid config accepted: {field}={value}")


def swap_gate():
    for points in (BAD_POINTS, np.full((5, 2), np.nan), np.zeros((4, 2))):
        instance = manager()
        target = face(points)
        log = io.StringIO()
        with (patch.object(Process, "_swap_call_count", 99),
              patch.object(instance, "_swap_from_saved_reference", return_value=PRIOR) as rescue,
              patch.object(Process, "_is_profile_face", side_effect=RuntimeError("boundary reached before inference")) as profile,
              contextlib.redirect_stdout(log), contextlib.redirect_stderr(log)):
            output = instance._process_face_swap_v21(good_face(), target, PRIOR, ORIGINAL, photo_path=EXACT_PATH)
        assert rescue.call_count == 1, "invalid landmarks passed the production alignment gate"
        assert not profile.called, "invalid target reached the inference boundary"
        assert output is PRIOR, "rejected target erased prior composite"
        assert rescue.call_args.kwargs["photo_path"] == EXACT_PATH


def saved_reference_rejected():
    instance = manager()
    bad = face()
    assert instance._photo_reference_face({"face_obj": bad}, ORIGINAL.shape) is None, "bad UI reference accepted"
    refs = {"selected": [{"source_path": EXACT_PATH, "face_obj": bad, "embedding": bad.embedding}]}
    with (patch.object(globals_module, "selected_face_references", refs, create=True),
          patch.object(instance, "_process_face_swap_v21", return_value=ORIGINAL) as downstream):
        output = instance._swap_from_saved_reference(good_face(), bad, PRIOR, ORIGINAL, 100, photo_path=EXACT_PATH)
    assert not downstream.called, "invalid reference entered rescued swap"
    assert output is PRIOR, "unrecoverable geometry erased previous valid swap"
    assert instance._align_reject_active is False


def validation_error_stops_swap():
    settings = config_module.get_faceswap_config()
    instance = manager()
    log = io.StringIO()
    with (patch.object(Process, "_swap_call_count", 99),
          patch.object(config_module, "get_faceswap_config", return_value={**settings, "landmark_bbox_margin_fraction": -1}),
          patch.object(Process, "_is_profile_face", side_effect=RuntimeError("validation bypassed")) as profile,
          contextlib.redirect_stdout(log), contextlib.redirect_stderr(log)):
        output = instance._process_face_swap_v21(good_face(), face(), PRIOR, ORIGINAL, photo_path=EXACT_PATH)
    assert not profile.called, "configuration validation error bypassed landmark guard"
    assert output is PRIOR, "validation error erased earlier composite"


def later_valid_reference():
    instance = manager()
    bad, good = face(), good_face()
    refs = {"selected": [{"source_path": EXACT_PATH, "face_obj": f, "embedding": f.embedding}
                         for f in (bad, good)]}
    with (patch.object(globals_module, "selected_face_references", refs, create=True),
          patch.object(instance, "_process_face_swap_v21", return_value=PRIOR) as downstream):
        output = instance._swap_from_saved_reference(good, bad, PRIOR, ORIGINAL, 100, photo_path=EXACT_PATH)
    assert downstream.call_count == 1
    assert downstream.call_args.args[1] is good, "invalid first ref displaced later coherent ref"
    assert output is PRIOR
    assert instance._align_reject_active is False


def filter_high_confidence_invalid():
    bad, good = face(), good_face()
    kept = filter_faces([bad, good])
    assert len(kept) == 1 and kept[0] is good, "high detector score bypassed coordinate containment"
    assert not face_module._has_face_geometry(bad)


def alignment_invalid():
    try:
        Warp.estimate_constrained_norm((3000, 2000), BAD_POINTS, 256, bbox=BAD_BOX)
    except ValueError:
        pass
    else:
        raise AssertionError("logged out-of-face points produced a warp")


def rotated_controls():
    good = good_face()
    center = (BAD_BOX[:2] + BAD_BOX[2:]) / 2
    for degrees in (0, 45, 90, 180, 270):
        angle = np.deg2rad(degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = (good.kps-center) @ rotation.T + center
        rotated = face(points)
        assert config_module.landmarks_fit_bbox(rotated.kps, rotated.bbox), degrees
        assert Process._photo_reference_face({"face_obj": rotated}, ORIGINAL.shape) is rotated, degrees
        if filter_faces is not None:
            kept = filter_faces([rotated])
            assert len(kept) == 1 and kept[0] is rotated, degrees
        matrix, method = Warp.estimate_constrained_norm((3000, 2000), rotated.kps, 256, bbox=rotated.bbox)
        assert method == "kps_ok" and np.array_equal(matrix, face_align.estimate_norm(rotated.kps, 256)), degrees


results = []
for test in (shared_shape_and_config, swap_gate, validation_error_stops_swap, saved_reference_rejected, later_valid_reference,
             filter_high_confidence_invalid, alignment_invalid, rotated_controls):
    try:
        if BASELINE and test in (shared_shape_and_config, validation_error_stops_swap, filter_high_confidence_invalid):
            row = {"test": test.__name__, "pass": None,
                   "status": "SKIP", "reason": "new shared helper / named filter unavailable in HEAD"}
        else:
            test()
            row = {"test": test.__name__, "pass": True}
    except Exception as error:
        row = {"test": test.__name__, "pass": False, "error": repr(error)}
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
for source_path in ("roop/face_util.py", "roop/faceswap_config.py", "config/model_configs.json"):
    hashes[source_path] = hashlib.sha256((ROOT / source_path).read_bytes()).hexdigest()
print(json.dumps({"baseline": BASELINE, "source_hashes": hashes,
                  "unavailable_baseline": unavailable_baseline, "results": results}, indent=2))
sys.exit(0 if all(r["pass"] is not False for r in results) else 1)
