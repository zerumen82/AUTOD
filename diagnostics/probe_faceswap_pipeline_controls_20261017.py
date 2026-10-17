"""Contratos de cuerpos de producción en CPU; sin cargar modelos ni UI."""
import ast
from pathlib import Path
from types import SimpleNamespace
import traceback
from typing import List
import unittest

import cv2
import numpy as np

from roop.faceswap_config import get_faceswap_config
from roop.quality_enhancements import detect_foreground_occlusion


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / "roop/ProcessMgr.py").read_text(encoding="utf-8"))
CLASS = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "ProcessMgr")
SWAP = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == "_process_face_swap_v21")


def execute(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<production body>", "exec"), namespace)


def manager_class():
    methods = [n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name in {
        "_process_face_swap_v21", "_mouth_occlusion_evidence", "_localize_mouth_occlusion",
        "_profile_occlusion_guides", "_soothe_swap_ring"}]
    cls = ast.ClassDef(name="ProcessMgr", bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {"np": np, "cv2": cv2, "traceback": traceback,
                 "get_faceswap_config": get_faceswap_config}
    execute([ast.fix_missing_locations(cls)], namespace)
    return namespace["ProcessMgr"]


class PipelineControls(unittest.TestCase):
    def test_multiface_composite_preserves_previous_face(self):
        node = next(n for n in ast.walk(SWAP) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "result_frame" for t in n.targets)
                    and "warped_face" in ast.unparse(n.value))
        original = np.zeros((32, 64, 3), np.uint8)
        accumulated = original.copy()
        for x, value in ((4, 100), (48, 200)):
            mask = np.zeros_like(original, np.float32)
            mask[4:12, x:x+8] = 1
            ns = dict(np=np, warped_face=np.full_like(original, value), composite_mask=mask,
                      original_frame=original, result_frame=accumulated, composite_base=accumulated)
            execute([node], ns)
            accumulated = ns["result_frame"]
        self.assertEqual(accumulated[5, 5, 0], 100)
        self.assertEqual(accumulated[5, 49, 0], 200)

    def test_invalid_second_face_preserves_previous_swap(self):
        cls = manager_class()
        cls._swap_call_count = 10
        manager = cls()
        manager.options = SimpleNamespace(blend_ratio=1.0)
        original = np.zeros((8, 8, 3), np.uint8)
        previous = np.full_like(original, 100)
        result = manager._process_face_swap_v21(SimpleNamespace(embedding=None),
                                                SimpleNamespace(), previous, original)
        np.testing.assert_array_equal(result, previous)

    def test_post_blend_preserves_previous_face_outside_support(self):
        cls = manager_class()
        original = np.zeros((128, 128, 3), np.uint8)
        previous = original.copy()
        previous[50:70, 50:70] = 150
        mask = np.zeros((128, 128), np.float32)
        cv2.circle(mask, (85, 60), 13, 1, -1)
        mask = cv2.GaussianBlur(mask, (9, 9), 0)
        result = previous.copy()
        result[mask > 0] = 200
        soothe = next(n for n in ast.walk(SWAP) if isinstance(n, ast.Assign)
                      and "self._soothe_swap_ring(" in ast.unparse(n))
        keep = [n for n in ast.walk(SWAP) if isinstance(n, ast.Assign)
                and any(ast.unparse(target) in ("outside_current", "result_frame[outside_current]")
                        for target in n.targets)]
        self.assertEqual(len(keep), 2)
        ns = dict(np=np, cv2=cv2, self=cls(), result_frame=result, original_frame=original,
                  composite_base=previous, composite_mask=np.repeat(mask[:, :, None], 3, axis=2),
                  blend_mask=mask, face_w=100, face_h=100)
        execute([soothe] + sorted(keep, key=lambda n: n.lineno), ns)
        np.testing.assert_array_equal(ns["result_frame"][mask == 0], previous[mask == 0])

    def test_video_honors_smoothing_in_both_paths(self):
        method = next(n for n in CLASS.body if isinstance(n, ast.FunctionDef) and n.name == "run_batch_inmem")
        calls = [n for n in ast.walk(method) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "process_frame"]
        self.assertEqual(len(calls), 2)
        for flag in (False, True):
            for call in calls:
                received = []
                ns = dict(self=SimpleNamespace(process_frame=lambda *a, **k: received.append(k)),
                          frame=np.zeros((8, 8, 3), np.uint8), video_path="fixture.mp4",
                          temporal_smoothing=flag)
                execute([ast.copy_location(ast.Expr(value=call), call)], ns)
                self.assertIs(received[0]["enable_temporal_smoothing"], flag)

    def test_disabled_autorotate_skips_rotated_passes(self):
        path = ROOT / "roop/face_util_rotation.py"
        fn = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                  if isinstance(n, ast.FunctionDef) and n.name == "get_all_faces_with_rotation")
        for flag in (False, True):
            seen = []
            analyser = SimpleNamespace(det_size=(640, 640), det_model=SimpleNamespace(input_size=(640, 640)),
                                       get=lambda image: seen.append(image.copy()) or [])
            ns = dict(np=np, cv2=cv2, List=List, Face=object,
                      roop=SimpleNamespace(globals=SimpleNamespace(autorotate_faces=flag, distance_threshold=.2)),
                      get_face_analyser=lambda: analyser, MEDIAPIPE_DETECTOR_AVAILABLE=False,
                      _filter_false_positive_faces=lambda faces, *a, **k: faces)
            execute([fn], ns)
            image = np.arange(64*96*3, dtype=np.uint8).reshape(64, 96, 3)
            ns[fn.name](image, for_target=True, prefer_fast=True, include_180=True)
            rotated = [f for f in seen if f.shape[:2] == (96, 64)]
            self.assertEqual(bool(rotated), flag)
            self.assertTrue(any(f.shape == image.shape for f in seen))

    def test_video_transition_preserves_native_texture(self):
        blocks = [n for n in ast.walk(SWAP) if isinstance(n, ast.If)
                  and any(isinstance(b, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "content_blur_sz"
                          for t in b.targets) for b in n.body)]
        yy, xx = np.indices((128, 128))
        frame = np.repeat((((xx+yy) % 2)*150+50).astype(np.uint8)[:, :, None], 3, axis=2)
        mask = np.tile(np.linspace(0, 1, 128, dtype=np.float32), (128, 1))
        ns = dict(np=np, cv2=cv2, warped_face=frame.copy(), final_mask=mask,
                  is_very_small_face=False, self=SimpleNamespace(_is_single_photo=False), blur_sz=17)
        execute(blocks, ns)
        np.testing.assert_array_equal(ns["warped_face"], frame)

    def test_disabled_smoothing_has_no_mask_boost_history(self):
        block = next(n for n in ast.walk(SWAP) if isinstance(n, ast.If)
                     and "not occ_applied" in ast.unparse(n.test)
                     and "_mask_boost_ema_" in ast.unparse(n))
        manager = SimpleNamespace(_mask_boost_ema_fixture=2.5)
        ns = dict(np=np, final_mask=np.full((16, 16), .5, np.float32), occ_applied=False,
                  self=manager, video_basename="fixture", call_num=10,
                  enable_temporal_smoothing=False, is_single_photo=False)
        execute([block], ns)
        self.assertAlmostEqual(float(ns["final_mask"].max()), .92, places=5)
        self.assertFalse(hasattr(manager, "_mask_boost_ema_fixture"))

    def test_occlusion_rescue_uses_unique_measured_component(self):
        cls = manager_class()
        self.assertTrue(hasattr(cls, "_localize_mouth_occlusion"))
        points = np.array([[80, 80], [175, 80], [128, 125], [108, 180], [148, 180]], np.float32)
        background = np.zeros((256, 256), np.float32)
        background[:100] = 1
        background[:, :80] = 1
        positive = background.copy()
        positive[132:251, 112:145] = 1
        candidate = cls._localize_mouth_occlusion(positive, points)
        self.assertIsNotNone(candidate)
        self.assertTrue(np.all(candidate[:100] == 0))
        self.assertTrue(np.all(candidate[132:251, 112:145] == 1))
        self.assertIsNone(cls._localize_mouth_occlusion(background, points))
        connected = positive.copy()
        connected[90:140, 125:130] = 1
        self.assertIsNone(cls._localize_mouth_occlusion(connected, points))
        remote = background.copy()
        remote[132:251, 215:240] = 1
        self.assertIsNone(cls._localize_mouth_occlusion(remote, points))
        lips = background.copy()
        cv2.ellipse(lips, (128, 180), (28, 16), 0, 0, 360, 1, -1)
        self.assertIsNone(cls._localize_mouth_occlusion(lips, points))

    def test_detector_real_occlusion_rescue(self):
        cls = manager_class()
        self.assertTrue(hasattr(cls, "_localize_mouth_occlusion"))
        points = np.array([[80, 80], [175, 80], [128, 125], [108, 180], [148, 180]], np.float32)
        target = np.full((256, 256, 3), (90, 140, 190), np.uint8)
        target[:100] = 0
        target[:, :80] = 0
        target[132:251, 112:145] = (255, 255, 0)
        measured = detect_foreground_occlusion(target, target, points, np.ones((256, 256), np.float32))
        self.assertGreater(float((measured > .1).mean()), .4)
        candidate = cls._localize_mouth_occlusion(measured, points)
        self.assertIsNotNone(candidate)
        self.assertLessEqual(float((candidate > .1).mean()), .4)
        self.assertTrue(np.all(candidate <= measured))

    def test_two_valid_oral_components_are_ambiguous(self):
        cls = manager_class()
        points = np.array([[80, 80], [175, 80], [128, 125], [108, 180], [148, 180]], np.float32)
        left = np.zeros((256, 256), np.float32)
        right = left.copy()
        left[132:251, 92:112] = 1
        right[132:251, 144:164] = 1
        self.assertIsNotNone(cls._localize_mouth_occlusion(left, points))
        self.assertIsNotNone(cls._localize_mouth_occlusion(right, points))
        self.assertIsNone(cls._localize_mouth_occlusion(np.maximum(left, right), points))


if __name__ == "__main__":
    unittest.main()
