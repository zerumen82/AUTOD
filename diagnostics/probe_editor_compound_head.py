"""Check the first pass of a multi-region edit without rerunning ComfyUI."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.face_util import detect_faces_robust, get_face_analyser_cpu
from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1790324529.png").convert("RGB")
    generated = Image.open(ROOT / "ui/tob/ComfyUI/output/FastEdit_00631_.png").convert("RGB")
    source = source.resize((800, 533), Image.Resampling.LANCZOS)
    generated = generated.resize(source.size, Image.Resampling.LANCZOS)
    faces = detect_faces_robust(
        cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR), get_face_analyser_cpu(), 0.2,
    ) or []
    masker = CLIPSegMasker()
    masker.device = "cpu"
    manager = ImgEditorManager.__new__(ImgEditorManager)
    try:
        subject = masker.generate_mask(source, "man")
        head = masker.generate_mask(source, "head")
        hair = masker.generate_mask(source, "hair")
        seed = manager._head_mask_from_region_plan(source, faces, subject)
        assert seed is not None
        region = manager._union_grayscale_masks(head, hair)
        completed = manager._complete_localized_region_mask(seed, region)
        mask = np.asarray(completed)
        hair_pixels = np.asarray(hair.convert("L")) >= 96
        missed = int((hair_pixels & (mask < 128)).sum())
        print(f"hair outside completed region: {missed}/{int(hair_pixels.sum())}")
        completed.save(ROOT / "diagnostics/editor_compound_completed_mask.png")
        Image.composite(generated, source, completed).save(
            ROOT / "diagnostics/editor_compound_completed_composite.png"
        )
    finally:
        masker.unload()


if __name__ == "__main__":
    main()
