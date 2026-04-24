from typing import Optional
import cv2
import numpy as np

from roop.types import Frame


def get_image_frame(filename: str):
    try:
        # Método 1: np.fromfile (soporta Unicode)
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img
        # Método 2: fallback a imread regular
        return cv2.imread(filename, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
    return None


def get_video_frame(video_path: str, frame_number: int = 0) -> Optional[Frame]:
    capture = cv2.VideoCapture(video_path)
    frame_total = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.set(cv2.CAP_PROP_POS_FRAMES, min(frame_total, frame_number - 1))
    has_frame, frame = capture.read()
    capture.release()
    if has_frame:
        return frame
    return None


def get_video_frame_total(video_path: str) -> int:
    capture = cv2.VideoCapture(video_path)
    video_frame_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    return video_frame_total
