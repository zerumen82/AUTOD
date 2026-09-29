import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

VIDEO = r"D:\PROJECTS\AUTOAUTO\.autodeep_temp\84aca5160488e89760d6c7b591f617f3e86777f9b895768397382b67f1e655d2\Corn1982vzla_20260119__2013272405091766326_1_20132723701279703040.mp4"

import roop.face_util as fu
import roop.face_util_rotation as fur
from roop.face_util_rotation import get_all_faces_with_rotation, _filter_false_positive_faces
from roop.face_util import _has_face_geometry

def check_geom_reason(face):
    try:
        kps = getattr(face, 'kps', None)
        if kps is None or not hasattr(kps, 'shape') or kps.shape != (5, 2):
            return "kps_shape"
        kpsf = kps.astype(np.float32)
        if not np.all(np.isfinite(kpsf)):
            return "nan"
        le, re, nose, lm, rm = kpsf
        fw = face.bbox[2] - face.bbox[0]
        fh = face.bbox[3] - face.bbox[1]
        if fw <= 0 or fh <= 0:
            return "bbox_dim"
        tol_v = 0.20 * fh
        if not (le[1] < lm[1] + tol_v and re[1] < rm[1] + tol_v):
            return f"ojos_boca le_y={le[1]:.0f} lm_y={lm[1]:.0f} re_y={re[1]:.0f} rm_y={rm[1]:.0f} tol_v={tol_v:.0f}"
        if not (nose[1] > min(le[1], re[1]) - tol_v and nose[1] < max(lm[1], rm[1]) + tol_v):
            return f"nariz_entre nose_y={nose[1]:.0f} le_y={le[1]:.0f} re_y={re[1]:.0f} lm_y={lm[1]:.0f} rm_y={rm[1]:.0f}"
        eye_sep_x = abs(re[0] - le[0])
        if eye_sep_x > 0.02 * fw and le[0] >= re[0]:
            return f"orden_horiz sep_x={eye_sep_x:.1f} 2%fw={0.02*fw:.1f} le_x={le[0]:.0f} re_x={re[0]:.0f}"
        if not (0.005 < eye_sep_x / max(1.0, fw) < 1.5):
            return f"interocular sep_x={eye_sep_x:.1f} fw={fw:.0f} ratio={eye_sep_x/max(1.0,fw):.4f}"
        return "OK"
    except Exception as e:
        return f"exc {e}"

def main():
    cap = cv2.VideoCapture(VIDEO)
    if not cap.isOpened():
        print("NO ABRE VIDEO")
        return
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"frames totales: {total}")
    # frames a probar: varios repartidos
    for fidx in [0, 30, 60, 100, 150, 200, 250, 276]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        print(f"\n===== Frame {fidx} shape={frame.shape} =====")
        raw_faces = None
        import io, contextlib
        # captura los prints del filtro
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            faces = get_all_faces_with_rotation(frame, for_target=True, prefer_fast=True)
        filtro_msgs = [ln for ln in buf.getvalue().splitlines() if 'sin geometr' in ln]
        print(f"  REJECTS del filtro: {len(filtro_msgs)}")
        for msg in filtro_msgs[:6]:
            print(f"    {msg.strip()}")
        print(f"  detectadas tras pipeline: {len(faces)}")
        for fi, f in enumerate(faces):
            det = getattr(f, 'det_score', 0.0)
            kps = getattr(f, 'kps', None)
            reason = check_geom_reason(f)
            bb = f.bbox
            print(f"  face{fi}: det={det:.2f} bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) geom={reason}")
            if kps is not None:
                k = np.asarray(kps, dtype=np.float32)
                print(f"       kps.shape={k.shape} kps=[{k.tolist()}]")
        print(f"  tras _filter_false_positive_faces: {len(_filter_false_positive_faces(list(faces)))}/{len(faces)}")
    cap.release()

if __name__ == "__main__":
    main()