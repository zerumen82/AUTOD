"""Measure the removal target and the background zone of the P1 cases."""

from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png"
PROMPTS = (
    "Remove the black helmet from the photo",
    "Quita el casco negro de la foto",
    "Add a small flag beside the astronaut",
)


def main():
    image = Image.open(SOURCE).convert("RGB")
    manager = get_img_editor_manager()
    for prompt in PROMPTS:
        target = manager._extract_removal_target(prompt)
        print(f"prompt={prompt!r} removal_target={target!r}", flush=True)
        masker = manager._get_clipseg_masker()
        for query in (target, "black helmet", "helmet"):
            if not query:
                continue
            mask = masker.generate_mask(image, query)
            if mask is None:
                print(f"   query {query!r}: sin máscara", flush=True)
                continue
            print(
                f"   query {query!r}: cobertura={manager._mask_coverage(mask):.1%}",
                flush=True,
            )
    faces = manager._detect_edit_faces(image)
    print(f"caras detectadas: {len(faces or [])}", flush=True)
    background = manager._background_mask_from_people(image, masker, faces or [])
    if background is None:
        print("zona de fondo: NO medida", flush=True)
    else:
        import numpy as np

        arr = np.asarray(background.convert("L"))
        print(
            f"zona de fondo: {float((arr >= 96).mean()):.1%} del lienzo",
            flush=True,
        )
        background.save(ROOT / "diagnostics" / "out" / "editor_categories_20261114" /
                        "background_zone_debug.png")


if __name__ == "__main__":
    main()
