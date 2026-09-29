"""Check a two-pass benign camera + wall-color edit."""

from pathlib import Path

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client


ROOT = Path(__file__).resolve().parents[1]


def main():
    camera_result = Image.open(
        ROOT / "diagnostics/editor_viewpoint_qwen_multiangle.png"
    ).convert("RGB")
    client = get_flux_edit_comfy_client()
    loaded, note = client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
    assert loaded, note
    result, note = client.generate(
        image=camera_result,
        prompt=(
            "Change only the office walls to blue. Preserve the high-angle camera "
            "view, both people, desks, chairs, windows, plant and composition."
        ),
        num_inference_steps=12,
        guidance_scale=1.0,
        denoise=1.0,
        no_kontext=True,
    )
    assert result is not None, note
    path = ROOT / "diagnostics/editor_viewpoint_qwen_then_turbo.png"
    result.image.save(path)
    print(f"SAVED {path} {result.image.size} {note}", flush=True)


if __name__ == "__main__":
    main()
