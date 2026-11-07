"""Independent CPU invariants for temporal face geometry.

Executes the unchanged production smoothing AST on synthetic measured faces.
No images, detectors, GPU, services or video content are loaded. This checks
geometry consistency across translation, shape changes and plan windows only.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics.probe_faceswap_temporal_alignment_20261106 import _production_smoothing  # noqa: E402
from roop.faceswap_config import landmarks_fit_bbox  # noqa: E402


class IndependentTemporalGeometry(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.production = _production_smoothing()

    def test_measurement_remains_coherent_across_motion_and_shape_changes(self):
        normalized = np.array([[.01, .30], [.25, .32], [.10, .45], [.02, .65], [.30, .66]])
        previous_bbox = np.array([100., 100., 200., 200.])
        previous_kps = previous_bbox[:2] + normalized * 100.
        key = "independent.mp4"
        for delta in (0., 45., 140.):
            for size in ((40., 180.), (180., 40.)):
                for confidence in (.3, .9):
                    for blend in (.7, 1.):
                        for profile in (False, True):
                            for plan_window in (False, True):
                                with self.subTest(delta=delta, size=size, confidence=confidence,
                                                  blend=blend, profile=profile, plan_window=plan_window):
                                    center = np.array([150. + delta, 150.])
                                    extent = np.array(size)
                                    raw_bbox = np.concatenate((center - extent / 2, center + extent / 2))
                                    raw_kps = raw_bbox[:2] + normalized * extent
                                    self.assertTrue(landmarks_fit_bbox(raw_kps, raw_bbox))
                                    target = SimpleNamespace(bbox=raw_bbox.copy(), kps=raw_kps.copy(),
                                                             det_score=confidence)
                                    state = SimpleNamespace(_prev_face_bbox={key: previous_bbox.copy()},
                                                            _prev_face_kps={key: previous_kps.copy()},
                                                            _prev_frame_result={})
                                    setattr(state, "_frame_count_independent.mp4", 50)
                                    setattr(state, "_plan_change_smooth_until_independent.mp4",
                                            60 if plan_window else 0)
                                    namespace = {
                                        "np": np, "self": state, "video_key": key,
                                        "video_basename": key, "enable_temporal_smoothing": True,
                                        "target_face": target, "call_num": 50, "user_blend": blend,
                                        "is_profile": profile, "video_path": None,
                                        "landmarks_fit_bbox": landmarks_fit_bbox,
                                    }
                                    exec(self.production, namespace)
                                    self.assertTrue(landmarks_fit_bbox(target.kps, target.bbox))
                                    self.assertTrue(np.isfinite(np.asarray(target.kps)).all())
                                    np.testing.assert_array_equal(state._prev_face_bbox[key], target.bbox)
                                    np.testing.assert_array_equal(state._prev_face_kps[key], target.kps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
