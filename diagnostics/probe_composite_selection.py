"""Measure seam energy for both localized composites on real outputs."""
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
CASES = [
    ("first", "first_reconstructed_source.png", "first_full_head_composite.png",
     "first_full_hair_composite.png"),
    ("profile", "profile_reconstructed_source.png", "profile_bald_ui_final.png",
     "profile_final_hair_replay.png"),
]


def metrics(source, output, mask):
    s = np.asarray(source.convert("RGB"), np.float32)
    o = np.asarray(output.convert("RGB"), np.float32)
    core = np.asarray(mask.convert("L")) >= 128
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    inner = core & ~(cv2.erode(core.astype(np.uint8), kernel) > 0)
    outer = ~core & (cv2.dilate(core.astype(np.uint8), kernel) > 0)
    ring = inner | outer
    gray_s = cv2.cvtColor(s.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_o = cv2.cvtColor(o.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap_s = cv2.Laplacian(gray_s, cv2.CV_32F)
    lap_o = cv2.Laplacian(gray_o, cv2.CV_32F)
    grad = np.abs(lap_o - lap_s)
    delta = np.max(np.abs(s - o), axis=2)
    return {
        "lap_ring": float(grad[ring].mean()),
        "lap_inner": float(grad[inner].mean()),
        "delta_ring": float(delta[ring].mean()),
        "delta_interior": float(delta[core & ~inner].mean()),
        "ring_area": float(ring.mean()),
    }


detector = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models"),
                        providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=-1, det_size=(1024, 1024))
masker = CLIPSegMasker()
editor = ImgEditorManager()
for name, src_name, head_name, hair_name in CASES:
    src = Image.open(ROOT / "diagnostics" / src_name).convert("RGB")
    head_out = Image.open(ROOT / "diagnostics" / head_name).convert("RGB")
    hair_out = Image.open(ROOT / "diagnostics" / hair_name).convert("RGB")
    faces = detector.get(np.asarray(src)[:, :, ::-1].copy())
    subject = masker.generate_mask(src, "man")
    hair = masker.generate_mask(src, "hair")
    head = editor._head_mask_from_region_plan(src, faces, subject)
    scalp = editor._grounded_hair_mask(src, head, subject, hair)
    print(name, "head", metrics(src, head_out, head), flush=True)
    print(name, "hair", metrics(src, hair_out, scalp), flush=True)
masker.unload()
