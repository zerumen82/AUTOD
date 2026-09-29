"""Verify automatic source choice from a large bank without GPU sessions."""
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import roop.globals
import roop.face_util_rotation as rotation
from roop.ProcessMgr import ProcessMgr


def face(x, y, z=0.0, quality=0.5):
    return SimpleNamespace(embedding=np.array([x, y, z], dtype=np.float32), quality=quality)


manager = ProcessMgr.__new__(ProcessMgr)
manager._score_source_face_quality = lambda source: source.quality
bank = [face(0, 0, 1) for _ in range(700)]
bank[140] = face(0.4, 0.9)
bank[699] = face(0.9, 0.4)
views = [face(1, 0), face(0.8, 0.6), face(0.8, -0.6)]
old_refs = getattr(roop.globals, 'selected_face_references', None)
old_random = getattr(roop.globals, 'source_random', None)
try:
    roop.globals.selected_face_references = {}
    roop.globals.source_random = False
    manager.options = SimpleNamespace(current_video_path=None)
    assert manager._select_source_face(face(1, 0), bank, 'selected_faces', 'photo_1.jpg') is bank[0], (
        'an unrelated face with higher target cosine must not displace the source identity'
    )
    assert manager._select_source_face(face(0, 1), views, 'selected_faces', 'photo_2.jpg') is views[1]
    assert manager._select_source_face(face(0, -1), views, 'selected_faces', 'photo_3.jpg') is views[2]
    quality_bank = [face(0.8, 0.6, quality=0.2), face(0.9, 0.3, quality=4.5)]
    assert manager._select_source_face(face(0, 1), quality_bank, 'selected_faces', 'quality.jpg') is quality_bank[1]

    low_bank = [face(0, 0, 1), face(0.5, 0.866)]
    assert manager._select_source_face(face(1, 0), low_bank, 'selected_faces', 'photo_4.jpg') is low_bank[1]
    assert manager._select_source_face(face(1, 0), [face(0, 0, 0)], 'selected_faces', 'invalid.jpg') is None

    # UI stores a padded crop in face_img_ref while bbox remains in the full
    # photo. Off-origin boxes must still measure focus on the facial pixels.
    quality_manager = ProcessMgr.__new__(ProcessMgr)
    quality_manager.options = SimpleNamespace(current_video_path=None)
    yy, xx = np.indices((276, 276))
    pattern = np.uint8(((xx//4 + yy//4) % 2)*150 + 50)
    sharp_image = np.repeat(pattern[:, :, None], 3, axis=2)
    blurred_image = cv2.GaussianBlur(sharp_image, (41, 41), 12)
    crop_faces = []
    for image, confidence in ((blurred_image, .95), (sharp_image, .90)):
        candidate = face(1, 0)
        candidate.bbox = [500, 400, 756, 656]
        candidate.det_score = confidence
        candidate.face_img_ref = image
        candidate.face_img_origin = (490, 390)
        candidate.face_img_is_rgb = True
        candidate.source_image_size = (1200, 1000)
        crop_faces.append(candidate)
    assert quality_manager._score_source_face_quality(crop_faces[1]) > quality_manager._score_source_face_quality(crop_faces[0]) + 1
    saved_random = roop.globals.source_random
    roop.globals.source_random = False
    assert quality_manager._select_source_face(face(0, 1), crop_faces, 'selected_faces', 'photo.jpg') is crop_faces[1]
    roop.globals.source_random = saved_random

    majority_bank = [face(1, 0, quality=2), face(.98, .10, quality=2), face(0, 1, quality=5)]
    majority_choice = manager._select_source_face(face(0, 1), majority_bank, 'selected_faces', 'majority.jpg')
    assert majority_choice is majority_bank[0] or majority_choice is majority_bank[1]
    random_bank = [face(1, 0), face(1, 0), face(1, 0)]
    roop.globals.source_random = True
    with patch('random.choice', side_effect=lambda pool: pool[-1]), patch.object(
        manager, '_select_source_face_by_similarity', side_effect=AssertionError('automatic route entered in random mode')
    ):
        assert manager._select_source_face(face(1, 0), random_bank, 'selected_faces', 'random.jpg') is random_bank[-1]

    manager.options.current_video_path = 'video_1.mp4'
    roop.globals.selected_face_references['selected_face_ref_video_1.mp4'] = {
        'embedding': face(1, 0).embedding,
    }
    roop.globals.source_random = True  # A stale photo setting cannot randomize video.
    assert manager._select_source_face(face(0, 1), views, 'selected_faces', 'video_1.mp4') is views[0]
    weak_match_bank = [face(0.25, 0.96, quality=0.4), face(0.19, 0.98, quality=5.0)]
    assert manager._select_source_face(face(1, 0), weak_match_bank, 'selected_faces', 'video_1.mp4') is weak_match_bank[1], (
        'a blurry source must not win a whole video on a tiny cross-person cosine advantage'
    )
    mixed_identity_bank = [face(0.25, 0.96, quality=0.4), face(-0.05, 0.0, 1.0, quality=5.0)]
    assert manager._select_source_face(face(1, 0), mixed_identity_bank, 'selected_faces', 'video_1.mp4') is mixed_identity_bank[0], (
        'a sharp face from another source identity must not replace the best-matching identity'
    )
    video_quality_bank = [face(.2, .98, quality=5), face(.2, .98, quality=.2)]
    video_quality_bank[1].kps = [[40, 110], [70, 108], [18, 160], [48, 210], [95, 200]]
    assert manager._select_source_face(face(1, 0), video_quality_bank, 'selected_faces', 'video_1.mp4', is_profile=True) is video_quality_bank[0], (
        'a poor profile must not lock a whole video when a much cleaner source is available'
    )
    assert manager._video_source_lock_keys('folder_a/video.mp4') != manager._video_source_lock_keys('folder_b/video.mp4')
    reacquired = SimpleNamespace(
        bbox=np.array([200, 100, 280, 190], dtype=np.float32),
        kps=np.array([[210, 130], [250, 130], [230, 150], [215, 175], [245, 175]], dtype=np.float32),
        embedding=np.array([1, 0, 0], dtype=np.float32),
    )
    exact = manager._apply_reacquired_face(
        reacquired, '_probe_assigned', '_probe_lost', '_probe_positions', 'probe.mp4',
        last_bbox=[10, 10, 90, 100], last_kps=np.zeros((5, 2)),
    )
    assert exact is reacquired and np.array_equal(exact.kps, reacquired.kps), (
        'reacquisition must retain current-frame landmarks without blending old geometry'
    )
    clock_manager = ProcessMgr.__new__(ProcessMgr)
    clock_manager.is_initialized = True
    clock_manager.processors = {'faceswap': object()}
    clock_manager.input_facesets = [SimpleNamespace(faces=[bank[0]])]
    clock_manager.options = SimpleNamespace(current_video_path='clock_video.mp4')
    clock_manager._should_hold_previous_swap = lambda *args: False
    old_detector = rotation.get_all_faces_smart
    try:
        rotation.get_all_faces_smart = lambda *args, **kwargs: []
        for _ in range(3):
            clock_manager.process_frame(np.zeros((128, 128, 3), dtype=np.uint8), file_path='clock_video.mp4')
        assert getattr(clock_manager, '_frame_count_clock_video.mp4') == 3
    finally:
        rotation.get_all_faces_smart = old_detector
    print('[PROBE] source identity, quality, per-photo views, invalid embedding, video isolation and clock: OK')
finally:
    roop.globals.selected_face_references = old_refs
    roop.globals.source_random = old_random
