"""Profile paste width, closed-mouth floor, and occlusion guides.

Synthetic only: no CUDA, no video. A passing run shows the helpers clip a
profile mask to the detector box, keep a closed mouth at blend 0, and move
occlusion onto the 5 keypoints instead of the centered crop.
"""
import numpy as np

from roop.ProcessMgr import ProcessMgr
from roop.quality_enhancements import detect_foreground_occlusion


def _half_max_span(mask):
    cols = np.where(mask.max(axis=0) > 0.5)[0]
    if len(cols) == 0:
        return 0
    return int(cols[-1] - cols[0] + 1)


def _check_profile_window(manager):
    shape = (600, 800)
    bbox = (100, 80, 280, 380)  # 180 x 300
    face_w = 180
    ones = np.ones(shape, dtype=np.float32)
    limited = manager._limit_profile_paste(ones, True, bbox, shape)
    width = _half_max_span(limited)
    assert width < 1.35 * face_w, width
    assert limited[230, 190] > 0.95
    assert limited[230, 28] < 0.02
    same = manager._limit_profile_paste(ones, False, bbox, shape)
    assert same is ones
    widened = ProcessMgr._widen_mask_for_blend(limited, face_w, 300, shape)
    again = manager._limit_profile_paste(widened, True, bbox, shape)
    width_again = _half_max_span(again)
    assert width_again < 1.35 * face_w, width_again
    assert again[230, 190] > 0.95
    assert again[230, 28] < 0.02
    print(
        f'profile window half-max {width}px then {width_again}px '
        f'(limit {1.35 * face_w:.0f}); legacy helper leaves frontal unchanged'
    )


def _check_frontal_window(manager):
    shape = (600, 800)
    bbox = (100, 80, 280, 380)
    ones = np.ones(shape, dtype=np.float32)
    limited = manager._limit_face_paste(ones, False, bbox, shape)
    widened = ProcessMgr._widen_mask_for_blend(limited, 180, 300, shape)
    composited = manager._limit_face_paste(widened, False, bbox, shape)
    assert limited[230, 190] > 0.95
    assert composited[230, 190] > 0.95
    assert limited[230, 50] < 0.02
    assert composited[230, 50] < 0.02
    assert limited[20, 190] < 0.02
    assert composited[20, 190] < 0.02
    assert 180 < _half_max_span(composited) < 1.35 * 180
    # A frontal face keeps more forehead and cheek area than a profile.
    profile = manager._limit_face_paste(ones, True, bbox, shape)
    assert composited[230, 294] > profile[230, 294]
    print(f'frontal window half-max {_half_max_span(composited)}px; far background excluded')


def _check_mouth_floor(manager):
    closed = manager._apply_profile_mouth_floor(0.0, False, 0.9)
    open_match = manager._apply_profile_mouth_floor(0.24, True, 0.9)
    open_other = manager._apply_profile_mouth_floor(0.24, True, 0.2)
    open_from_zero = manager._apply_profile_mouth_floor(0.0, True, 0.9)
    assert np.isclose(closed, 0.0), closed
    assert open_match >= 0.30
    assert np.isclose(open_other, 0.24), open_other
    assert np.isclose(open_from_zero, 0.30), open_from_zero
    print('closed profile mouth stays 0; open matched mouth keeps the 0.30 floor')


def _check_mouth_occlusion_evidence():
    points = np.array([
        [40, 45], [120, 45], [80, 70], [60, 100], [100, 100],
    ], dtype=np.float32)
    empty = np.zeros((160, 160), dtype=np.float32)
    assert ProcessMgr._mouth_occlusion_evidence(empty, points) == (0.0, 0.0)
    object_mask = empty.copy()
    import cv2
    cv2.ellipse(object_mask, (80, 100), (58, 46), 0, 0, 360, 1.0, -1)
    lips, ring = ProcessMgr._mouth_occlusion_evidence(object_mask, points)
    assert lips > 0.18 and ring > 0.12, (lips, ring)
    print(f'oral object evidence lips={lips:.2f} ring={ring:.2f}; empty mask rejected')


def _check_mouth_depth_choice():
    kps = np.array([
        [40, 40], [80, 40], [60, 80], [48, 105], [72, 105],
    ], dtype=np.float32)
    assert ProcessMgr._mouth_depth_choice(kps, (55, 85), (60, 105)) is True
    assert ProcessMgr._mouth_depth_choice(kps, (60, 105), (55, 85)) is False
    assert ProcessMgr._mouth_depth_choice(kps, (60, 105), (62, 108)) is None
    print('mouth depth rejects a lip center almost level with the nose')


def _check_guides():
    matrix = np.array([[2, 0, 10], [0, 2, -4]], dtype=np.float32)
    kps = np.array([
        [100, 80], [140, 82], [130, 100], [110, 120], [150, 122],
    ], dtype=np.float32)
    mapped, interior = ProcessMgr._profile_occlusion_guides(
        matrix, kps, (40, 50, 90, 120), (256, 256), (512, 512),
    )
    assert mapped is not None and interior is not None
    assert np.allclose(mapped[0], [105, 78], atol=0.1)
    assert interior[78, 65] > 0.5
    assert interior[10, 10] == 0 and interior[250, 250] == 0
    assert ProcessMgr._profile_occlusion_guides(
        None, kps, (40, 50, 90, 120), (256, 256), (512, 512),
    ) == (None, None)
    print('occlusion guides follow the scaled matrix; empty matrix returns nothing')


def _synthetic_face():
    face = np.zeros((160, 160, 3), dtype=np.uint8)
    face[:, :] = (90, 140, 190)
    # Non-skin checker on the left half of the aligned square.
    for y in range(0, 160, 8):
        for x in range(0, 96, 8):
            color = (220, 30, 30) if ((x + y) // 8) % 2 == 0 else (40, 220, 40)
            face[y:y + 8, x:x + 8] = color
    face[61:79, 121:139] = (255, 255, 0)
    face[61:63, 121:123] = (0, 0, 0)
    return face


def _check_occlusion():
    face = _synthetic_face()
    plain = detect_foreground_occlusion(face, face)
    assert isinstance(plain, np.ndarray) and plain.shape == (160, 160)
    left = float(plain[:, :40].mean())
    frac = float(np.mean(plain > 0.1))
    assert left > 0.2, left
    assert frac > 0.40, frac

    kps = np.array([
        [110, 50], [150, 50], [140, 60], [122, 70], [138, 70],
    ], dtype=np.float32)
    interior = np.zeros((160, 160), dtype=np.float32)
    interior[30:140, 100:156] = 1.0
    guided = detect_foreground_occlusion(
        face, face, feature_points=kps, interior_mask=interior,
    )
    guided_left = float(guided[:, :40].mean())
    guided_frac = float(np.mean(guided > 0.1))
    object_mean = float(guided[64:76, 124:136].mean())
    assert guided_left < 0.02, guided_left
    assert guided_frac <= 0.40, guided_frac
    assert object_mean > 0.25, object_mean
    print(
        f'centered occlusion left={left:.2f} frac={frac:.2f}; '
        f'guided left={guided_left:.3f} frac={guided_frac:.2f} object={object_mean:.2f}'
    )


def main():
    manager = ProcessMgr.__new__(ProcessMgr)
    _check_profile_window(manager)
    _check_frontal_window(manager)
    _check_mouth_floor(manager)
    _check_mouth_occlusion_evidence()
    _check_mouth_depth_choice()
    _check_guides()
    _check_occlusion()
    print('profile paste, mouth floor, and occlusion guides OK')


if __name__ == '__main__':
    main()
