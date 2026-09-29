"""Verify lip masks follow their source crop and the swap alignment."""
from types import SimpleNamespace

import numpy as np

from roop.ProcessMgr import ProcessMgr
from roop.processors.FaceSwap import (
    MOUTH_OUTER_LANDMARK_INDICES, mouth_region_to_frame_points,
)


def main():
    landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(468)]
    angles = np.linspace(0, 2 * np.pi, len(MOUTH_OUTER_LANDMARK_INDICES), endpoint=False)
    for index, angle in zip(MOUTH_OUTER_LANDMARK_INDICES, angles):
        landmarks[index] = SimpleNamespace(
            x=0.5 + 0.16 * np.cos(angle), y=0.5 + 0.10 * np.sin(angle),
        )
    region = {'landmarks_468': SimpleNamespace(landmark=landmarks)}
    original_shape = (240, 320, 3)
    frame_points = mouth_region_to_frame_points(
        region, original_shape, crop_origin=(100, 80), crop_size=(100, 80),
    )
    assert np.allclose(frame_points.mean(axis=0), [150, 120], atol=1)

    aligned = np.zeros((256, 256, 3), dtype=np.uint8)
    transform = np.array([[2, 0, -200], [0, 2, -140]], dtype=np.float32)
    manager = ProcessMgr.__new__(ProcessMgr)
    for mouth_region in (region, frame_points):
        mask = manager._mouth_mask_in_aligned_face(
            aligned, mouth_region, original_shape, transform,
            (100, 70, 200, 170), (100, 80), (100, 80),
        )
        ys, xs = np.where(mask > 0.5)
        assert len(xs) > 0
        assert np.allclose([xs.mean(), ys.mean()], [100, 100], atol=2)
        assert mask[100, 100] > 0.5 and mask[20, 20] == 0

    enhanced = np.zeros((512, 512, 3), dtype=np.uint8)
    enhanced_transform = transform * 2
    enhanced_mask = manager._mouth_mask_in_aligned_face(
        enhanced, region, original_shape, enhanced_transform,
        (100, 70, 200, 170), (100, 80), (100, 80),
    )
    ys, xs = np.where(enhanced_mask > 0.5)
    assert np.allclose([xs.mean(), ys.mean()], [200, 200], atol=3)

    mask_no_transform = manager._mouth_mask_in_aligned_face(
        aligned, region, original_shape, None,
        (100, 70, 200, 170), (100, 80), (100, 80),
    )
    ys, xs = np.where(mask_no_transform > 0.5)
    assert np.allclose([xs.mean(), ys.mean()], [128, 128], atol=2)
    print('MediaPipe crop and 106-landmark masks align at 256/512px; bbox fallback aligns too')


if __name__ == '__main__':
    main()
