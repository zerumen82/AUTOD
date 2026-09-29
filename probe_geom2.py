import sys, os
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

VIDEO = r"D:\PROJECTS\AUTOAUTO\.autodeep_temp\84aca5160488e89760d6c7b591f617f3e86777f9b895768397382b67f1e655d2\Corn1982vzla_20260119__2013272405091766326_1_20132723701279703040.mp4"

import roop.face_util as fu
import roop.face_util_rotation as fur
from roop.face_util import _has_face_geometry
from roop.face_util_rotation import get_all_faces_with_rotation, convert_faces_to_face_objects, transform_faces_from_rotation, filter_duplicate_faces

def check_geom_reason(face):
    try:
        kps = getattr(face, 'kps', None)
        if kps is None:
            return "kps_none"
        k = np.asarray(kps, dtype=np.float32)
        if k.shape == (1, 5, 2):
            k = k[0]
        if k.shape != (5, 2):
            return f"kps_shape={k.shape}"
        kpsf = k.astype(np.float32)
        if not np.all(np.isfinite(kpsf)):
            return "nan"
        le, re, nose, lm, rm = kpsf
        fw = face.bbox[2] - face.bbox[0]
        fh = face.bbox[3] - face.bbox[1]
        if fw <= 0 or fh <= 0:
            return "bbox_dim"
        tol_v = 0.20 * fh
        if not (le[1] < lm[1] + tol_v and re[1] < rm[1] + tol_v):
            return f"R1 ojos_boca le_y={le[1]:.0f} lm_y={lm[1]:.0f} re_y={re[1]:.0f} rm_y={rm[1]:.0f} tol_v={tol_v:.0f}"
        if not (nose[1] > min(le[1], re[1]) - tol_v and nose[1] < max(lm[1], rm[1]) + tol_v):
            return f"R2 nariz_entre nose_y={nose[1]:.0f} le_y={le[1]:.0f} re_y={re[1]:.0f} lm_y={lm[1]:.0f} rm_y={rm[1]:.0f} tol_v={tol_v:.0f}"
        eye_sep_x = abs(re[0] - le[0])
        if eye_sep_x > 0.02 * fw and le[0] >= re[0]:
            return f"R3 orden_horiz sep_x={eye_sep_x:.1f} fw={fw:.0f} le_x={le[0]:.0f} re_x={re[0]:.0f}"
        if not (0.005 < eye_sep_x / max(1.0, fw) < 1.5):
            return f"R4 interocular sep_x={eye_sep_x:.1f} fw={fw:.0f} ratio={eye_sep_x/max(1.0,fw):.4f}"
        return "OK"
    except Exception as e:
        return f"exc {e}"

def main():
    analizador = fu.get_face_analyser()
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"frames totales: {total}")

    orig_det_size = analizador.det_size
    orig_input = analizador.det_model.input_size

    for fidx in [30, 60, 100, 250, 276]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"\n===== Frame {fidx} shape={frame.shape} =====")
        # replicar _run_passes con det 640 (prefer_fast)
        analizador.det_size = (640, 640)
        analizador.det_model.input_size = (640, 640)
        found = []
        try:
            faces = analizador.get(frame_rgb)
            if faces:
                valid = [f for f in faces if f.get('det_score', f.get('score', 0.0)) >= 0.10]
                converted = convert_faces_to_face_objects(valid)
                found.extend(converted)
            h, w = frame_rgb.shape[:2]
            for angle in [90, 270]:
                rotated = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE) if angle == 90 else cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                rot_faces = analizador.get(rotated)
                if rot_faces:
                    valid_rot = [f for f in rot_faces if f.get('det_score', f.get('score', 0.0)) >= 0.10]
                    if valid_rot:
                        converted_rot = convert_faces_to_face_objects(valid_rot)
                        transformed = transform_faces_from_rotation(converted_rot, angle, w, h)
                        found.extend(transformed)
        finally:
            analizador.det_size = orig_det_size
            analizador.det_model.input_size = orig_input

        print(f"  raw detectadas: {len(found)}")
        for fi, f in enumerate(found):
            det = getattr(f, 'det_score', 0.0)
            kps = getattr(f, 'kps', None)
            reason = check_geom_reason(f)
            bb = f.bbox
            print(f"  raw{fi}: det={det:.2f} bbox=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) geom={reason}")
            if kps is not None:
                k = np.asarray(kps, dtype=np.float32)
                print(f"       kps.shape={k.shape} kps=[{k.tolist()}]")
    cap.release()

if __name__ == "__main__":
    main()