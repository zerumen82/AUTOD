from typing import Optional
import os
import subprocess as sp
import threading
import time
import cv2
import numpy as np

from roop.types import Frame


def _find_ffprobe() -> Optional[str]:
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.join(project_dir, 'ffmpeg', 'ffmpeg-master-latest-win64-gpl', 'bin', 'ffprobe.exe'),
        os.path.join(project_dir, 'ffmpeg', 'ffprobe.exe'),
        'ffprobe',
    ]
    for candidate in candidates:
        if candidate == 'ffprobe' or os.path.exists(candidate):
            return candidate
    return None


def _ffprobe_frame_total(video_path: str) -> Optional[int]:
    probe = _find_ffprobe()
    if probe is None:
        return None
    is_avi = video_path.lower().endswith(('.avi', '.wmv'))
    base = [probe, '-v', 'error', '-select_streams', 'v:0', '-of', 'default=noprint_wrappers=1:nokey=1']
    attempts = [
        ([], 'nb_frames'),
    ]
    if not is_avi:
        attempts.append((['-count_packets'], 'nb_read_packets'))
    for extra_args, key in attempts:
        try:
            result = sp.run(
                base + extra_args + ['-show_entries', f'stream={key}', video_path],
                capture_output=True, text=True, timeout=8,
            )
        except Exception:
            continue
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if lines and lines[0].isdigit() and int(lines[0]) > 0:
                return int(lines[0])
    return None


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
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_number) - 1))
    has_frame, frame = capture.read()
    capture.release()
    if has_frame:
        return frame
    return None


_TOTAL_CACHE = {}


def _cache_key(video_path: str):
    try:
        stat = os.stat(video_path)
        return (video_path, stat.st_mtime_ns, stat.st_size)
    except Exception:
        return None


def get_video_frame_total(video_path: str) -> int:
    key = _cache_key(video_path)
    if key and key in _TOTAL_CACHE:
        return _TOTAL_CACHE[key]
    results = {}

    def run_ffprobe():
        total = _ffprobe_frame_total(video_path)
        if total is not None:
            results['total'] = total

    def run_opencv():
        try:
            capture = cv2.VideoCapture(video_path)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
            if total > 0:
                results['total'] = total
        except Exception:
            pass

    ff_thread = threading.Thread(target=run_ffprobe, daemon=True)
    cv_thread = threading.Thread(target=run_opencv, daemon=True)
    ff_thread.start()
    cv_thread.start()
    deadline = time.time() + 10
    while time.time() < deadline:
        if 'total' in results:
            total = results['total']
            if key:
                _TOTAL_CACHE[key] = total
            return total
        time.sleep(0.05)
    total = results.get('total', 0)
    if key:
        _TOTAL_CACHE[key] = total
    return total
