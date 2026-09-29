"""Measure whether the real subsequent edits removed the named man's hair."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.face_util import detect_faces_robust, get_face_analyser
from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]
PAIRS = (
    ("ui/tob/ComfyUI/input/fast_1790204211.png", "img_editor/edit_1790204403.png", True),
    ("ui/tob/ComfyUI/input/fast_1790204520.png", "img_editor/edit_1790204710.png", True),
    ("ui/tob/ComfyUI/input/fast_1790204211.png", "diagnostics/bald_edit_first_photo_v2.png", False),
    ("ui/tob/ComfyUI/input/fast_1790204211.png", "diagnostics/bald_edit_ui_real.png", False),
    ("ui/tob/ComfyUI/input/fast_1790204739.png", "img_editor/edit_1790204977.png", True),
    ("ui/tob/ComfyUI/input/fast_1790204739.png", "diagnostics/bald_edit_square_tight.png", False),
)


def main():
    analyser = get_face_analyser()
    masker = CLIPSegMasker()
    manager = ImgEditorManager.__new__(ImgEditorManager)
    try:
        for source_rel, output_rel, expected_fail in PAIRS:
            source = Image.open(ROOT / source_rel).convert("RGB")
            output = Image.open(ROOT / output_rel).convert("RGB").resize(source.size)
            faces = detect_faces_robust(
                cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR), analyser, 0.20,
            ) or []
            subject = masker.generate_mask(source, "man")
            head = manager._head_mask_from_region_plan(source, faces, subject)
            assert head is not None, source_rel
            before, after = manager._bald_hair_coverage(source, output, head, masker)
            failed = before >= 0.04 and after > max(0.04, 0.60 * before)
            assert failed is expected_fail, (output_rel, before, after)
            print(
                f"{Path(output_rel).name}: source_hair={before:.3f} "
                f"output_hair={after:.3f} fail={failed}",
                flush=True,
            )
    finally:
        masker.unload()


if __name__ == "__main__":
    main()
