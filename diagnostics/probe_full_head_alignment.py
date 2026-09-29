"""Measure landmark alignment of full-context bald edits on two real photos."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis
from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager

ROOT = Path(__file__).resolve().parents[1]
detector = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models"),
                        providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=-1, det_size=(1024, 1024))
masker = CLIPSegMasker()
editor = ImgEditorManager()
for name in ("first", "profile"):
    src = Image.open(ROOT / f"diagnostics/{name}_reconstructed_source.png").convert("RGB")
    raw = Image.open(ROOT / f"diagnostics/{name}_full_longcat_raw.png").convert("RGB")
    raw = raw.resize(src.size, Image.Resampling.LANCZOS)
    s = np.asarray(src)
    g = np.asarray(raw)
    sf = detector.get(s[:, :, ::-1].copy())
    gf = detector.get(g[:, :, ::-1].copy())
    subject = masker.generate_mask(src, "man")
    head = editor._head_mask_from_region_plan(src, sf, subject)
    assert head is not None
    h = np.asarray(head) >= 128
    source = max(sf, key=lambda f: h[
        max(0, int(f.bbox[1])):min(src.height, int(f.bbox[3])),
        max(0, int(f.bbox[0])):min(src.width, int(f.bbox[2]))
    ].mean())
    center = np.asarray(source.bbox).reshape(2, 2).mean(axis=0)
    target = min(gf, key=lambda f: np.linalg.norm(
        np.asarray(f.bbox).reshape(2, 2).mean(axis=0) - center))
    transform, inliers = cv2.estimateAffinePartial2D(
        np.asarray(target.kps, np.float32), np.asarray(source.kps, np.float32),
        method=cv2.LMEDS)
    assert transform is not None
    aligned = cv2.warpAffine(g, transform, src.size,
                             flags=cv2.INTER_LANCZOS4,
                             borderMode=cv2.BORDER_REFLECT_101)
    mask = editor._feather_mask(head, amount=6)
    Image.composite(Image.fromarray(aligned), src, mask).save(
        ROOT / f"diagnostics/{name}_full_head_aligned.png")
    print(name, "transform", transform.tolist(), "source box", source.bbox,
          "generated box", target.bbox, flush=True)
masker.unload()
