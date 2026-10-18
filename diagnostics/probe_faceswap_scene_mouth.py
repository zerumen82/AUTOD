"""Scene-cut mouth memory stays per video, and a dead verifier does not drop a source."""
from types import SimpleNamespace

import numpy as np

from roop.ProcessMgr import ProcessMgr
import roop.source_face_validator as validator


def _seed(mgr):
    mgr._occ_frame_ema = {
        'video_a.mp4': np.ones((2, 2), np.float32),
        'video_b.mp4': np.ones((2, 2), np.float32),
    }
    mgr._mouth_blend_ema_by_person = {'video_a.mp4_p0': 0.65, 'video_b.mp4_p0': 0.40}
    mgr._mouth_force_paste_by_person = {'video_a.mp4_p0': True, 'video_b.mp4_p1': False}
    mgr._mouth_paste_ema_by_person = {'video_a.mp4_p0': 0.9, 'video_b.mp4_p0': 0.2}
    mgr._mouth_state_history = {'video_a.mp4_p0': [1], 'video_b.mp4_p0': [0]}
    mgr._mouth_ratio_smooth = {'video_a.mp4_p0': 0.5, 'video_b.mp4_p0': 0.1}
    mgr._mouth_hold_count = {'video_a.mp4_p0': 3, 'video_b.mp4_p0': 1}
    setattr(mgr, '_profile_hysteresis_video_a.mp4', 4)
    setattr(mgr, '_profile_hysteresis_video_b.mp4', 2)


def _assert_other_video(mgr):
    assert 'video_b.mp4' in mgr._occ_frame_ema
    assert mgr._mouth_blend_ema_by_person['video_b.mp4_p0'] == 0.40
    assert mgr._mouth_force_paste_by_person['video_b.mp4_p1'] is False
    assert mgr._mouth_paste_ema_by_person['video_b.mp4_p0'] == 0.2
    assert mgr._mouth_state_history['video_b.mp4_p0'] == [0]
    assert mgr._mouth_ratio_smooth['video_b.mp4_p0'] == 0.1
    assert mgr._mouth_hold_count['video_b.mp4_p0'] == 1
    assert getattr(mgr, '_profile_hysteresis_video_b.mp4') == 2


def main():
    mgr = ProcessMgr.__new__(ProcessMgr)
    _seed(mgr)
    mgr._reset_tracking_on_scene_cut(r'D:\clips\a.mp4', 12, reason='probe hard')
    for store in (
        mgr._mouth_blend_ema_by_person,
        mgr._mouth_force_paste_by_person,
        mgr._mouth_paste_ema_by_person,
        mgr._mouth_state_history,
        mgr._mouth_ratio_smooth,
        mgr._mouth_hold_count,
    ):
        assert 'video_a.mp4_p0' not in store
    assert 'video_a.mp4' not in mgr._occ_frame_ema
    assert not hasattr(mgr, '_profile_hysteresis_video_a.mp4')
    _assert_other_video(mgr)

    _seed(mgr)
    mgr._reset_tracking_on_scene_cut(r'D:\clips\a.mp4', 20, reason='probe soft', soft=True)
    assert mgr._mouth_blend_ema_by_person['video_a.mp4_p0'] == 0.65
    assert mgr._mouth_force_paste_by_person['video_a.mp4_p0'] is True
    assert 'video_a.mp4' in mgr._occ_frame_ema
    assert getattr(mgr, '_profile_hysteresis_video_a.mp4') == 4

    _seed(mgr)
    setattr(mgr, '_closeup_active_a.mp4', True)
    mgr._reset_tracking_on_scene_cut(r'D:\clips\a.mp4', 30, reason='probe closeup')
    assert 'video_a.mp4_p0' not in mgr._mouth_blend_ema_by_person
    assert 'video_a.mp4' not in mgr._occ_frame_ema
    _assert_other_video(mgr)

    _seed(mgr)
    mgr._reset_tracking_on_scene_cut(
        r'D:\clips\a.mp4', 40, reason='probe explicit soft closeup', soft=True,
    )
    assert mgr._mouth_blend_ema_by_person['video_a.mp4_p0'] == 0.65
    assert 'video_a.mp4' in mgr._occ_frame_ema

    _seed(mgr)
    mgr._clear_video_temporal_mouth_and_occlusion('video_a.mp4')
    assert 'video_a.mp4_p0' not in mgr._mouth_blend_ema_by_person
    assert 'video_a.mp4' not in mgr._occ_frame_ema
    _assert_other_video(mgr)

    ProcessMgr._swap_call_count = 6
    mgr.options = SimpleNamespace(blend_ratio=1.0)
    mgr._mouth_blend_mask = np.ones((4, 4), np.float32)
    mgr._mouth_blend_factor = 0.8
    frame = np.zeros((8, 8, 3), np.uint8)
    accumulated = frame.copy()
    accumulated[1:4, 1:4] = 73
    returned = mgr._process_face_swap_v21(
        SimpleNamespace(), SimpleNamespace(bbox=[0, 0, 4, 4]), accumulated, frame,
    )
    np.testing.assert_array_equal(returned, accumulated)
    assert not np.shares_memory(returned, accumulated)
    assert not frame.any(), 'failed swap changed the original reference'
    assert mgr._mouth_blend_mask is None
    assert mgr._mouth_blend_factor == 0.0

    image = np.zeros((80, 80, 3), np.uint8)
    face = SimpleNamespace(bbox=np.array([10, 10, 50, 50], dtype=float))
    rejected, _reason = validator.confirm_source_region(
        image, SimpleNamespace(bbox=np.array([1, 2, 3], dtype=float)),
    )
    assert rejected is False

    class Boom:
        def process(self, _rgb):
            raise RuntimeError('mp down')

    validator._detector = Boom()
    validator._mesh = Boom()
    skipped, message = validator.confirm_source_region(image, face)
    assert skipped is True and 'skipped' in message

    class _Rect:
        xmin = ymin = 0.0
        width = height = 0.05

    class _Det:
        class location_data:
            relative_bounding_box = _Rect()

    class _Detected:
        detections = [_Det()]

    class _Point:
        x = 0.0
        y = 0.0

    class _Landmarks:
        landmark = [_Point() for _ in range(4)]

    class _MeshOut:
        multi_face_landmarks = [_Landmarks()]

    class _FakeDet:
        def process(self, _rgb):
            return _Detected()

    class _FakeMesh:
        def process(self, _rgb):
            return _MeshOut()

    validator._detector = _FakeDet()
    validator._mesh = _FakeMesh()
    low_iou, iou_message = validator.confirm_source_region(image, face)
    assert low_iou is False and 'IoU' in iou_message

    class _NoneModel:
        def process(self, _rgb):
            return None

    validator._detector = _NoneModel()
    validator._mesh = _NoneModel()
    opened, open_message = validator.confirm_source_region(image, face)
    assert opened is True and 'no detection result' in open_message
    validator._detector = None
    validator._mesh = None
    print('scene-cut mouth reset keeps the other video; verifier failure keeps the source')


if __name__ == '__main__':
    main()
