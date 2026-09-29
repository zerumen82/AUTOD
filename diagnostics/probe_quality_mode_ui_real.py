"""Exercise prompt-free HD Quality Mode through the actual Gradio event."""
from pathlib import Path
import time

import numpy as np
from PIL import Image

from ui.tabs import comfy_launcher
from ui.tabs.img_editor.events import on_generate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "img_editor/edit_1790254710.png"
OUTPUT = ROOT / "diagnostics/quality_mode_hd_ui_probe.png"


def main():
    assert SOURCE.is_file(), SOURCE
    comfy_before = comfy_launcher.is_comfyui_running()
    original = Image.open(SOURCE).convert("RGB")
    start = time.monotonic()
    result, message, _ = on_generate(
        original, "", "longcat", 0.0, False, 1, True, "hd",
    )
    elapsed = time.monotonic() - start
    print(f"[PROBE] status={message}", flush=True)
    assert result is not None, message
    comfy_after = comfy_launcher.is_comfyui_running()
    assert comfy_after == comfy_before, "Quality-only changed ComfyUI state"
    result.save(OUTPUT)
    reduced = result.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    delta = np.abs(np.asarray(reduced, dtype=np.int16) - np.asarray(original, dtype=np.int16))
    print(
        f"[PROBE] size={original.size}->{result.size} "
        f"elapsed={elapsed:.1f}s mean_abs_delta={float(delta.mean()):.2f} "
        f"comfy_before={comfy_before} comfy_after={comfy_after} "
        f"output={OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
