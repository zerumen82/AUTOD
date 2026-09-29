"""Exercise the integrated scene edit on the real first-pass image."""
from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager
from ui.tabs import comfy_launcher


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1790203567.png"
OUTPUT = ROOT / "diagnostics/img_editor_scene_office_final.png"


def main():
    assert SOURCE.is_file(), SOURCE
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI already in use"
    process = None
    try:
        result, message, _ = ImgEditorManager().generate_intelligent(
            image=Image.open(SOURCE).convert("RGB"),
            prompt="they're in an office",
            engine="longcat",
            use_rewriter=True,
            use_semantic=True,
            auto_upscale=False,
            enhance_faces=False,
            _compound_depth=1,
            _compound_region_hint="office",
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
