"""Actual HTTP/color check with GPU housekeeping explicitly isolated.

Runs the existing served-UI diagnostic. Only hardware discovery/cleanup are
disabled; the CPU image pipeline, Gradio HTTP callbacks and PNG IO are real.
No image-model inference or GPU cleanup is certified by this diagnostic.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


def main():
    from diagnostics import probe_editor_engine_status_served_20261024 as served
    from roop.img_editor.img_editor_manager import ImgEditorManager

    out = ROOT / "diagnostics/out/editor_cpu_served_20261025" / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    isolation = {
        "scope": "real HTTP and classical CPU color; no image inference",
        "hardware_cleanup_disabled": True,
        "hardware_discovery_disabled": True,
        "output_directory": str(out),
    }
    (out / "isolation.json").write_text(json.dumps(isolation, indent=2), encoding="utf-8")
    print(json.dumps(isolation), flush=True)
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=0),
        patch.object(ImgEditorManager, "_release_local_gpu_before_comfy"),
        patch.object(served.events, "_cleanup_vram"),
        patch.object(served.events, "get_img_editor_output_dir", return_value=str(out)),
    ):
        served.main()


if __name__ == "__main__":
    main()
