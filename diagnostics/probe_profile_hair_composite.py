"""Inspect saved LongCat crops without generating another image."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from insightface.app import FaceAnalysis

from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]
CASES = [(615, 1790275495), (617, 1790278400), (619, 1790280681)]


def main():
    detector = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models"),
                            providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, det_size=(640, 640))
    masker = CLIPSegMasker()
    editor = ImgEditorManager()
    for run, stamp in CASES:
        source = Image.open(ROOT / "ui/tob/ComfyUI/input" / f"fast_{stamp}.png").convert("RGB")
        generated = Image.open(ROOT / "ui/tob/ComfyUI/output" / f"FastEdit_{run:05d}_.png").convert("RGB")
        if generated.size != source.size:
            generated = generated.resize(source.size, Image.Resampling.LANCZOS)
        faces = detector.get(np.asarray(source)[:, :, ::-1].copy())
        generated_faces = detector.get(np.asarray(generated)[:, :, ::-1].copy())
        subject = masker.generate_mask(source, "man")
        hair = masker.generate_mask(source, "hair")
        head = editor._head_mask_from_region_plan(source, faces, subject)
        if head is None:
            raise AssertionError(f"case {run}: no grounded head")
        scalp = editor._grounded_hair_mask(source, head, subject, hair)
        if scalp is None:
            raise AssertionError(f"case {run}: no grounded scalp")
        soft = editor._feather_mask(scalp, amount=6)
        hair.save(ROOT / "diagnostics" / f"profile_hair_mask_{run:05d}.png")
        scalp.save(ROOT / "diagnostics" / f"profile_scalp_mask_{run:05d}.png")
        fixed = Image.composite(generated, source, soft)
        fixed.save(ROOT / "diagnostics" / f"profile_hair_crop_{run:05d}.png")
        hair_arr = np.asarray(hair) >= 96
        subj_arr = np.asarray(subject) >= 80
        raw_hair = (hair_arr & subj_arr).astype(np.uint8)
        raw_hair = cv2.dilate(raw_hair, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        tight = Image.fromarray(raw_hair * 255, "L")
        tight = editor._feather_mask(tight, amount=4)
        Image.composite(generated, source, tight).save(
            ROOT / "diagnostics" / f"profile_hair_tight_{run:05d}.png")
        if faces and generated_faces:
            source_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            edited_face = max(generated_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            affine, _ = cv2.estimateAffinePartial2D(
                np.asarray(edited_face.kps, np.float32),
                np.asarray(source_face.kps, np.float32),
                method=cv2.LMEDS,
            )
            if affine is not None:
                moved = cv2.warpAffine(np.asarray(generated), affine, source.size,
                                       flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
                aligned = Image.composite(Image.fromarray(moved), source, soft)
                aligned.save(ROOT / "diagnostics" / f"profile_hair_aligned_{run:05d}.png")
                mapped = cv2.transform(np.asarray(edited_face.kps, np.float32)[None], affine)[0]
                error = np.linalg.norm(mapped - np.asarray(source_face.kps), axis=1).mean()
                scale = np.linalg.norm(affine[0, :2])
                print(f"case={run} source_face={source_face.bbox.round().tolist()} "
                      f"generated_face={edited_face.bbox.round().tolist()} "
                      f"affine_scale={scale:.3f} landmark_error={error:.1f}px", flush=True)
        mask_array = np.asarray(soft)
        diff = np.max(np.abs(np.asarray(fixed).astype(np.int16) - np.asarray(source).astype(np.int16)), axis=2)
        assert np.count_nonzero(diff[mask_array == 0]) == 0
        print(f"case={run} size={source.size} faces={len(faces)} "
              f"mask_share={(mask_array >= 128).mean():.3f} "
              f"changed_share={(diff > 20).mean():.3f}", flush=True)
    masker.unload()


if __name__ == "__main__":
    main()
