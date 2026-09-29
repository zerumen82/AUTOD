"""Compare Klein's local edit on the same failed image."""
from pathlib import Path

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
from ui.tabs import comfy_launcher


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1790204739.png"
OUTPUT = ROOT / "diagnostics/bald_klein_crop.png"
BOX = (459, 334, 998, 875)


def main():
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI already in use"
    process = None
    try:
        ok, msg, _ = comfy_launcher.start(directly_run=True)
        process = comfy_launcher.comfy_process
        assert ok, msg
        source = Image.open(SOURCE).convert("RGB").crop(BOX)
        client = get_flux_edit_comfy_client()
        ok, msg = client.load(flux_version="flux-2-klein-base-4b-Q4_K_S.gguf")
        assert ok, msg
        result, msg = client.generate(
            image=source,
            prompt="Make the man bald. Preserve his face, expression, pose and the other visible details of this exact photo.",
            num_inference_steps=4,
            guidance_scale=1.0,
            denoise=1.0,
        )
        print(f"[PROBE] status={msg} result={type(result)}", flush=True)
        assert result is not None, msg
        image = result.image if hasattr(result, "image") else result
        image.save(OUTPUT)
        print(f"[PROBE] output={OUTPUT} size={image.size}", flush=True)
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
