"""Test whether an unanchored Full pass can produce a new office viewpoint."""

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
            "Overhead bird's-eye photograph looking straight down on the same office. "
            "Reconstruct the camera view from directly above: floor and tops of "
            "desks and chairs are visible; two adults and the plant remain in the room. "
            "The original frontal camera angle must be replaced."
        ),
        num_inference_steps=36,
        guidance_scale=2.5,
        denoise=1.0,
        no_kontext=True,
        drop_edit_image_ref=True,
    )
    assert result is not None, note
    path = ROOT / "diagnostics/editor_viewpoint_weak_ref.png"
    result.image.save(path)
    print(f"SAVED {path} {result.image.size} {note}", flush=True)


if __name__ == "__main__":
    main()
