"""Real UI path for the user's profile-size and crop regression."""
from pathlib import Path
from PIL import Image

from ui.tabs.img_editor.events import on_generate
from ui.tabs import comfy_launcher

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("first", ROOT / "diagnostics/first_reconstructed_source.png"),
    ("profile", ROOT / "diagnostics/profile_reconstructed_source.png"),
]


def main():
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI already in use"
    try:
        for name, source in CASES:
            assert source.is_file(), source
            result, message, _ = on_generate(
                Image.open(source).convert("RGB"), "EL HOMBRE DEBE SER CALVO",
                "longcat", 0.0, False, 3, False, "hd",
            )
            print(f"[PROBE] {name} status={message}", flush=True)
            assert result is not None, message
            output = ROOT / "diagnostics" / f"{name}_bald_ui_final.png"
            result.save(output)
            print(f"[PROBE] {name} output={output} size={result.size}", flush=True)
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
