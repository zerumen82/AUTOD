"""Read the local FaceSwap settings shared by alignment and composition."""

from functools import lru_cache
import json
from pathlib import Path
import numpy as np


@lru_cache(maxsize=1)
def _read_settings(path, modified_ns, size):
    with Path(path).open(encoding="utf-8") as config_file:
        settings = json.load(config_file)["faceswap"]
    if not isinstance(settings, dict):
        raise ValueError("faceswap configuration must be an object")
    return settings


def get_faceswap_config():
    path = Path(__file__).resolve().parents[1] / "config" / "model_configs.json"
    stat = path.stat()
    return dict(_read_settings(str(path), stat.st_mtime_ns, stat.st_size))


def landmarks_fit_bbox(kps, bbox):
    """Check coordinate consistency without assuming a head orientation.

    Detector confidence cannot repair points measured outside their face.
    The tolerance is the same containment margin used by source extraction.
    """
    settings = get_faceswap_config()
    fraction = float(settings["landmark_bbox_margin_fraction"])
    pixels = float(settings["landmark_bbox_margin_pixels"])
    if not np.isfinite([fraction, pixels]).all() or fraction < 0 or pixels < 0:
        raise ValueError("Landmark containment margins must be finite and nonnegative")
    try:
        points = np.asarray(kps, dtype=np.float64)
        bounds = np.asarray(bbox, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if (points.shape != (5, 2) or bounds.shape != (4,)
            or not np.isfinite(points).all() or not np.isfinite(bounds).all()):
        return False
    extent = bounds[2:] - bounds[:2]
    if np.any(extent <= 0):
        return False
    margin = max(pixels, fraction * float(extent.max()))
    return bool(np.all(points >= bounds[:2] - margin)
                and np.all(points <= bounds[2:] + margin))
