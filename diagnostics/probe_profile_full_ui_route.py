"""Exercise the edited Image Editor route with the recovered real profile photo."""
from pathlib import Path
from PIL import Image
from ui.tabs.img_editor.events import on_generate
from ui.tabs import comfy_launcher

ROOT = Path(__file__).resolve().parents[1]


def main():
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI already in use"
    try:
        source = Image.open(ROOT / "diagnostics/profile_reconstructed_source.png").convert("RGB")
        result, message, _ = on_generate(
            source, "EL HOMBRE DEBE SER CALVO", "longcat", 0.0, False, 3, False, "hd")
        print(f"[PROBE] status={message}", flush=True)
        assert result is not None, message
        output = ROOT / "diagnostics/profile_bald_full_ui_route.png"
        result.save(output)
        print(f"[PROBE] output={output} size={result.size}", flush=True)
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
