"""Inspect the localized composite mask against a real generated edit."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.face_util import detect_faces_robust, get_face_analyser_cpu
from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1790323509.png"
GENERATED = ROOT / "ui/tob/ComfyUI/output/FastEdit_00630_.png"
DELIVERED = ROOT / "img_editor/edit_1790323674.png"


def main():
    source = Image.open(SOURCE).convert("RGB").resize((800, 533), Image.Resampling.LANCZOS)
    generated = Image.open(GENERATED).convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    delivered = Image.open(DELIVERED).convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
    faces = detect_faces_robust(
        cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR), get_face_analyser_cpu(), 0.20
    ) or []
    masker = CLIPSegMasker()
    masker.device = "cpu"
    try:
        man = masker.generate_mask(source, "man")
        hair = masker.generate_mask(source, "hair")
        head_seg = masker.generate_mask(source, "head")
        manager = ImgEditorManager.__new__(ImgEditorManager)
        head = manager._head_mask_from_region_plan(source, faces, man)
        assert head is not None
        head.save(ROOT / "diagnostics/editor_case_head_mask.png")
        for name, mask in (("man", man), ("hair", hair), ("head_seg", head_seg)):
            if mask is not None:
                mask.save(ROOT / f"diagnostics/editor_case_{name}_mask.png")
        src = np.asarray(source, dtype=np.int16)
        raw = np.asarray(generated, dtype=np.int16)
        out = np.asarray(delivered, dtype=np.int16)
        m = np.asarray(head.convert("L")) >= 128
        sy, sx = np.where(m)
        for radius in (6, 12, 24, 40, 60):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            ring = cv2.dilate(m.astype(np.uint8), k).astype(bool) & ~m
            diff_raw = np.abs(raw - src).max(axis=2)
            diff_final = np.abs(out - src).max(axis=2)
            hair_arr = np.asarray(hair.convert("L")) >= 96
            print(f"ring={radius}: pixels={ring.sum()} raw_diff>30={np.mean(diff_raw[ring]>30):.3f} final_diff>30={np.mean(diff_final[ring]>30):.3f} hair={int((ring & hair_arr).sum())}")
        print("head bbox", (sx.min(), sy.min(), sx.max(), sy.max()))
        print("hair outside mask", int(((np.asarray(hair.convert("L")) >= 96) & ~m).sum()))
        visual_region = manager._union_grayscale_masks(head_seg, hair)
        full = np.asarray(manager._complete_localized_region_mask(head, visual_region))
        Image.fromarray(full, "L").save(ROOT / "diagnostics/editor_case_completed_mask.png")
        rerender = Image.composite(generated, source, Image.fromarray(full, "L"))
        rerender.save(ROOT / "diagnostics/editor_case_completed_composite.png")
        hair_pixels = np.asarray(hair.convert("L")) >= 96
        missed = int((hair_pixels & (full < 128)).sum())
        print("hair outside completed mask", missed)
        assert missed < 0.05 * int(hair_pixels.sum())
        assert full[100, 600] < 8  # unrelated person remains protected

        seed = np.zeros((80, 100), np.uint8)
        seed[20:40, 15:35] = 255
        regions = np.zeros_like(seed)
        regions[15:45, 10:40] = 255
        regions[15:45, 60:90] = 255
        isolated = np.asarray(manager._complete_localized_region_mask(
            Image.fromarray(seed), Image.fromarray(regions),
        ))
        assert isolated[30, 25] > 200 and isolated[30, 75] < 8
        all_regions = np.asarray(manager._complete_localized_region_mask(
            Image.fromarray(regions), Image.fromarray(regions),
        ))
        assert all_regions[30, 25] > 200 and all_regions[30, 75] > 200
    finally:
        masker.unload()


if __name__ == "__main__":
    main()
