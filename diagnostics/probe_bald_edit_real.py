"""Repeat one failed standalone edit with the user's actual image."""
from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager
from ui.tabs import comfy_launcher


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1790204739.png"
OUTPUT = ROOT / "diagnostics/bald_edit_square_tight.png"


def main():
    assert SOURCE.is_file(), SOURCE
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI already in use; leave the user's session alone"
    process = None
    try:
        manager = ImgEditorManager()
        result, message, _ = manager.generate_intelligent(
            image=Image.open(SOURCE).convert("RGB"),
            prompt="el hombre debe ser calvo",
            engine="longcat",
            use_rewriter=True,
            use_semantic=True,
            auto_upscale=False,
            enhance_faces=False,
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
