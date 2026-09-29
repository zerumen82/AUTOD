"""Measure the selected person's face drift in two real Turbo crops."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

ROOT = Path(__file__).resolve().parents[1]
CASES = [("first", 1790284132, 622), ("profile", 1790283837, 621)]
detector = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models"),
                        providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=-1, det_size=(640, 640))

for name, stamp, number in CASES:
    source = Image.open(ROOT / "ui/tob/ComfyUI/input" / f"fast_{stamp}.png").convert("RGB")
    generated = Image.open(ROOT / "ui/tob/ComfyUI/output" / f"FastEdit_{number:05d}_.png").convert("RGB")
    generated = generated.resize(source.size, Image.Resampling.LANCZOS)
    sf = detector.get(np.asarray(source)[:, :, ::-1].copy())
    gf = detector.get(np.asarray(generated)[:, :, ::-1].copy())
    print(name, "counts", len(sf), len(gf))
    for label, fs in (("source", sf), ("generated", gf)):
        for f in sorted(fs, key=lambda f: -((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])))[:3]:
            print(name, label, "bbox", np.round(f.bbox).tolist(), "kps", np.round(f.kps).tolist())
    s = max(sf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    g = max(gf, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    sd = np.asarray(s.bbox); gd = np.asarray(g.bbox)
    center_delta = np.linalg.norm((sd[:2]+sd[2:])/2 - (gd[:2]+gd[2:])/2)
    face_width = sd[2]-sd[0]
    kps_delta = np.linalg.norm(np.asarray(s.kps)-np.asarray(g.kps),axis=1)
    scale = ((gd[2]-gd[0])*(gd[3]-gd[1]) / ((sd[2]-sd[0])*(sd[3]-sd[1])))**.5
    print(name, "center_delta_fraction", center_delta/face_width,
          "kp_delta_fraction", np.mean(kps_delta)/face_width,
          "scale", scale)
