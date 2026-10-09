"""Read the local FaceSwap settings shared by alignment and composition."""

from functools import lru_cache
import json
from pathlib import Path


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
