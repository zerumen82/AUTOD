"""Test a weaker LongCat reference method for a camera change."""

from pathlib import Path

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client


ROOT = Path(__file__).resolve().parents[1]


def main():
    image = Image.open(ROOT / "diagnostics/img_editor_scene_office_final.png").convert("RGB")
    client = get_flux_edit_comfy_client()
    loaded, note = client.load("LongCat-Image-Edit-Q4_K_S.gguf")
    assert loaded, note
    result, note = client.generate(
        image=image,
        prompt=(
            "Change the viewpoint of this exact photo to a clear overhead camera angle. "
            "Show the same two adults in the same office with the same desks, "
            "chairs, windows and plant. Reconstruct their visible surfaces from above; "
            "the floor and desk tops should be visible. Keep their identities."
        ),
        num_inference_steps=36,
        guidance_scale=2.5,
        denoise=1.0,
        no_kontext=False,
        reference_latents_method="uxo/uno",
    )
    assert result is not None, note
    path = ROOT / "diagnostics/editor_viewpoint_middle_ref.png"
    result.image.save(path)
    print(f"SAVED {path} {result.image.size} {note}", flush=True)


if __name__ == "__main__":
    main()
