"""Compare installed FLUX.2 Klein 4B on the same benign POV request."""

from pathlib import Path

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client


ROOT = Path(__file__).resolve().parents[1]


def main():
    image = Image.open(ROOT / "diagnostics/img_editor_scene_office_final.png").convert("RGB")
    client = get_flux_edit_comfy_client()
    loaded, note = client.load("flux-2-klein-base-4b-Q4_K_S.gguf")
    assert loaded, note
    result, note = client.generate(
        image=image,
        prompt=(
            "Change the viewpoint of this exact office photo to an overhead "
            "bird's-eye camera angle. Show the same two adults, desks, chairs, "
            "windows and plant from above, with recognizable identities and layout. "
            "The tops of desks and the floor should be visible."
        ),
        num_inference_steps=4,
        guidance_scale=1.0,
        denoise=1.0,
    )
    assert result is not None, note
    path = ROOT / "diagnostics/editor_viewpoint_klein.png"
    result.image.save(path)
    print(f"SAVED {path} {result.image.size} {note}", flush=True)


if __name__ == "__main__":
    main()
