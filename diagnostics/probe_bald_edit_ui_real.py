"""Repeat the user's failed follow-up through the actual ImgEditor event."""
from pathlib import Path

from PIL import Image

from ui.tabs.img_editor.events import on_generate
from ui.tabs import comfy_launcher


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1790204211.png"
OUTPUT = ROOT / "diagnostics/bald_edit_ui_real.png"


def main():
    assert SOURCE.is_file(), SOURCE
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI already in use"
    process = None
    try:
        result, message, _ = on_generate(
            Image.open(SOURCE).convert("RGB"), "el hombre debe ser calvo",
            "longcat", 0.0, False, 3, False, "hd",
        )
        process = comfy_launcher.comfy_process
        print(f"[PROBE] status={message}", flush=True)
        assert result is not None, message
        result.save(OUTPUT)
        print(f"[PROBE] output={OUTPUT} size={result.size}", flush=True)
    finally:
        process = process or comfy_launcher.comfy_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


if __name__ == "__main__":
    main()
