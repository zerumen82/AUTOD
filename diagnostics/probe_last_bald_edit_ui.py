"""Reproduce the final stalled bald edit through the real UI event."""
from pathlib import Path

from PIL import Image

from ui.tabs.img_editor.events import on_generate
from ui.tabs import comfy_launcher


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "img_editor/edit_1790254710.png"
OUTPUT = ROOT / "diagnostics/last_bald_edit_turbo_full_context.png"


def main():
    assert SOURCE.is_file(), SOURCE
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI already in use"
    try:
        result, message, _ = on_generate(
            Image.open(SOURCE).convert("RGB"), "el hombre debe ser calvo",
            "longcat", 0.0, False, 3, False, "hd",
        )
        print(f"[PROBE] status={message}", flush=True)
        assert result is not None, message
        result.save(OUTPUT)
        print(f"[PROBE] output={OUTPUT} size={result.size}", flush=True)
    finally:
        process = comfy_launcher.comfy_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


if __name__ == "__main__":
    main()
