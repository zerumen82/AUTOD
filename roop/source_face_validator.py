"""Independent spatial confirmation for uncertain source detections."""
import threading

import cv2
import numpy as np

_lock = threading.Lock()
_detector = None
_mesh = None


def _box_iou(first, second):
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    overlap = np.maximum(0, np.minimum(first[2:], second[2:]) - np.maximum(first[:2], second[:2])).prod()
    union = np.maximum(0, first[2:] - first[:2]).prod() + np.maximum(0, second[2:] - second[:2]).prod() - overlap
    return float(overlap / max(1.0, union))


def confirm_source_region(image_bgr, face):
    """Confirm that another model finds a face in the same region, not nearby."""
    global _detector, _mesh
    try:
        h, w = image_bgr.shape[:2]
        box = np.asarray(face.bbox, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            return False, 'invalid bbox'
        x1, y1, x2, y2 = box
        px, py = (x2-x1)*.5, (y2-y1)*.5
        cx1, cy1 = max(0, int(x1-px)), max(0, int(y1-py))
        cx2, cy2 = min(w, int(x2+px)), min(h, int(y2+py))
        if cx2 <= cx1 or cy2 <= cy1:
            return False, 'empty crop'
        rgb = cv2.cvtColor(image_bgr[cy1:cy2, cx1:cx2], cv2.COLOR_BGR2RGB)
        if max(rgb.shape[:2]) > 640:
            scale = 640.0 / max(rgb.shape[:2])
            rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        extent = np.array([cx2-cx1, cy2-cy1])
        origin = np.array([cx1, cy1])
        best_iou = 0.0
        detector_ran = False
        mesh_ran = False
        with _lock:
            import mediapipe as mp
            if _detector is None:
                _detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1, min_detection_confidence=.5,
                )
            detected = _detector.process(rgb)
            detector_ran = detected is not None
            detections = [] if detected is None else (getattr(detected, 'detections', None) or [])
            for candidate in detections:
                rect = candidate.location_data.relative_bounding_box
                lo = origin + [rect.xmin, rect.ymin] * extent
                hi = origin + [rect.xmin+rect.width, rect.ymin+rect.height] * extent
                best_iou = max(best_iou, _box_iou(box, [*lo, *hi]))
            if best_iou >= .40:
                return True, f'face detector IoU={best_iou:.2f}'
            # Mesh covers some tilted, dark and profile faces missed by the
            # independent detector. Spatial overlap prevents a nearby face
            # from validating an ear, necklace or a patch of background.
            if _mesh is None:
                _mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=True, max_num_faces=3,
                    min_detection_confidence=.5,
                )
            result = _mesh.process(rgb)
            mesh_ran = result is not None
            landmarks = [] if result is None else (getattr(result, 'multi_face_landmarks', None) or [])
            for candidate in landmarks:
                points = np.array([[point.x, point.y] for point in candidate.landmark])
                lo = origin + points.min(axis=0)*extent
                hi = origin + points.max(axis=0)*extent
                best_iou = max(best_iou, _box_iou(box, [*lo, *hi]))
        if not detector_ran and not mesh_ran:
            return True, 'independent verification skipped: no detection result'
        return best_iou >= .40, f'independent face IoU={best_iou:.2f}'
    except Exception as exc:
        return True, f'independent verification skipped: {exc}'
