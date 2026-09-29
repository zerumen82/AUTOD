"""Check rotated face boxes and landmarks return to the photo coordinates."""
import numpy as np

from roop.face_util_rotation import transform_faces_from_rotation


class Detection(dict):
    def __getattr__(self, name):
        return self[name]


width, height = 467, 662
original_box = np.array([90, 380, 240, 510], dtype=np.int32)
original_point = np.array([130.0, 420.0], dtype=np.float32)

for angle, rotated_box, rotated_point in (
    (90, [height - 510, 90, height - 380, 240], [height - 1 - 420, 130]),
    (180, [width - 240, height - 510, width - 90, height - 380],
     [width - 1 - 130, height - 1 - 420]),
    (270, [380, width - 240, 510, width - 90], [420, width - 1 - 130]),
):
    detection = Detection(
        bbox=np.asarray(rotated_box, dtype=np.int32),
        kps=np.tile(np.asarray(rotated_point, dtype=np.float32), (5, 1)),
        landmark_2d_106=np.tile(np.asarray(rotated_point, dtype=np.float32), (106, 1)),
        embedding=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        det_score=0.8,
    )
    face = transform_faces_from_rotation([detection], angle, width, height)[0]
    np.testing.assert_array_equal(face.bbox, original_box)
    np.testing.assert_allclose(face.kps, np.tile(original_point, (5, 1)))
    np.testing.assert_allclose(face.landmark_106, np.tile(original_point, (106, 1)))

print('[PROBE] 90/180/270 face boxes, 5 points and 106 landmarks: OK')
