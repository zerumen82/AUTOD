"""Real Turbo inpaint probe on the saved profile crop."""
from pathlib import Path
from PIL import Image

from ui.tabs import comfy_launcher
from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client


ROOT = Path(__file__).resolve().parents[1]


def main():
    assert not comfy_launcher.is_comfyui_running(), "ComfyUI already in use"
    try:
        ok, status, _ = comfy_launcher.start(directly_run=True)
        assert ok, status
        client = get_flux_edit_comfy_client()
        ok, status = client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
        assert ok, status
        source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1790280681.png").convert("RGB")
        mask = Image.open(ROOT / "diagnostics/profile_hair_mask_00619.png").convert("L")
        output, status = client.generate(
            source,
            "Edit this exact photo: make the man bald. Keep his exact head size, "
            "profile angle, face, expression, ear and neck. Keep the woman and "
            "the hand exactly as in the photo.",
            negative_prompt="extra faces, changed head size, changed pose, cut edges",
            num_inference_steps=12,
            guidance_scale=1.0,
            denoise=1.0,
            mask_image=mask,
            grow_mask_by=0,
            no_kontext=True,
            reference_latents_method="index",
        )
        print(f"[PROBE] {status}", flush=True)
        assert output is not None, status
        output.image.save(ROOT / "diagnostics/profile_longcat_inpaint_raw.png")
        print(f"[PROBE] saved size={output.image.size}", flush=True)
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
