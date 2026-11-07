"""CPU regression: production temporal geometry must retain valid bbox/kps.

The smoothing branch is executed from the real ProcessMgr AST, without loading
models or rendering media. Inputs are synthetic measured geometry. This proves
the geometry contract, never visual quality or detector accuracy.
"""
from __future__ import annotations

import ast
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.faceswap_config import landmarks_fit_bbox  # noqa: E402


def _production_smoothing():
    tree = ast.parse((ROOT / "roop/ProcessMgr.py").read_text(encoding="utf-8-sig"))
    manager = next(node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "ProcessMgr")
    swap = next(node for node in manager.body
                if isinstance(node, ast.FunctionDef) and node.name == "_process_face_swap_v21")
    branch = next(node for node in ast.walk(swap)
                  if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Name)
                  and node.test.id == "enable_temporal_smoothing"
                  and any(isinstance(child, ast.Name) and child.id == "prev_bbox"
                          for child in ast.walk(node)))
    return compile(ast.fix_missing_locations(ast.Module(
        body=[copy.deepcopy(branch)], type_ignores=[],
    )), str(ROOT / "roop/ProcessMgr.py"), "exec")


class TemporalGeometryContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production = _production_smoothing()

    def run_motion(self, displacement, *, blend, profile, confidence, previous=True):
        previous_bbox = np.array([100., 100., 200., 200.])
        previous_kps = np.array([
            [101., 130.], [125., 132.], [110., 145.], [102., 165.], [130., 166.],
        ])
        raw_bbox = previous_bbox + np.array([displacement, 0., displacement, 0.])
        raw_kps = previous_kps + np.array([displacement, 0.])
        self.assertTrue(landmarks_fit_bbox(raw_kps, raw_bbox))
        target = SimpleNamespace(bbox=raw_bbox.copy(), kps=raw_kps.copy(), det_score=confidence)
        key = "video_neutral.mp4"
        state = SimpleNamespace(
            _prev_face_bbox={key: previous_bbox.copy()} if previous else {},
            _prev_face_kps={key: previous_kps.copy()} if previous else {},
            _prev_frame_result={},
        )
        # Real tracking has advanced before smoothing; zero would activate the
        # unrelated initial plan window through its existing default contract.
        setattr(state, "_frame_count_neutral.mp4", 50)
        namespace = {
            "np": np, "self": state, "video_key": key, "video_basename": "neutral.mp4",
            "enable_temporal_smoothing": True, "target_face": target, "call_num": 50,
            "user_blend": blend, "is_profile": profile, "video_path": "neutral.mp4",
            "landmarks_fit_bbox": landmarks_fit_bbox,
        }
        exec(self.production, namespace)
        self.assertTrue(landmarks_fit_bbox(target.kps, target.bbox),
                        "Temporal smoothing invalidated a valid measured face")
        np.testing.assert_array_equal(state._prev_face_bbox[key], target.bbox)
        np.testing.assert_array_equal(state._prev_face_kps[key], target.kps)
        return target, raw_bbox, raw_kps

    def test_large_motion_retains_valid_raw_geometry(self):
        target, bbox, kps = self.run_motion(140., blend=1., profile=False, confidence=.9)
        np.testing.assert_array_equal(target.bbox, bbox)
        np.testing.assert_array_equal(target.kps, kps)

    def test_profile_motion_retains_valid_raw_geometry(self):
        target, bbox, kps = self.run_motion(140., blend=1., profile=True, confidence=.9)
        np.testing.assert_array_equal(target.bbox, bbox)
        np.testing.assert_array_equal(target.kps, kps)

    def test_low_confidence_motion_keeps_one_coherent_pair(self):
        for profile in (False, True):
            with self.subTest(profile=profile):
                target, bbox, kps = self.run_motion(80., blend=.7, profile=profile, confidence=.3)
                np.testing.assert_array_equal(target.bbox, bbox)
                np.testing.assert_array_equal(target.kps, kps)

    def test_stable_valid_smoothing_is_preserved(self):
        for profile in (False, True):
            with self.subTest(profile=profile):
                target, bbox, kps = self.run_motion(2., blend=1., profile=profile, confidence=.9)
                self.assertFalse(np.array_equal(target.bbox, bbox))
                self.assertFalse(np.array_equal(target.kps, kps))

    def test_initial_frame_keeps_measurement(self):
        target, bbox, kps = self.run_motion(0., blend=.7, profile=True, confidence=.3, previous=False)
        np.testing.assert_array_equal(target.bbox, bbox)
        np.testing.assert_array_equal(target.kps, kps)


if __name__ == "__main__":
    unittest.main()
