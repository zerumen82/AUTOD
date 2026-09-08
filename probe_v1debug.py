import os, sys
import numpy as np
import cv2

DLL_DIR = os.path.abspath("dll")
dll_renamed = False
if os.path.isdir(DLL_DIR):
    backup = DLL_DIR + "_BACKUP"
    if not os.path.isdir(backup):
        os.rename(DLL_DIR, backup)
        dll_renamed = True

try:
    sys.path.insert(0, os.path.abspath("."))
    from insightface.utils import face_align
    from roop.face_util_rotation import get_all_faces_smart

    D = "debug_swap"
    for n in [1, 2, 3]:
        raw = cv2.imread(os.path.join(D, f"01_raw_model_f{n}.png"))
        orig = cv2.imread(os.path.join(D, f"07_original_frame_f{n}.png"))
        final = cv2.imread(os.path.join(D, f"04_final_result_f{n}.png"))
        mask = cv2.imread(os.path.join(D, f"05_mask_final_f{n}.png"))
        warped = cv2.imread(os.path.join(D, f"06_warped_face_f{n}.png"))
        if raw is None or orig is None:
            print(f"--- f{n}: archivos faltantes ---")
            continue
        print(f"--- f{n} ---")
        faces = get_all_faces_smart(orig, min_score=None, for_target=True) or []
        print(f"  caras en 07: {len(faces)}")
        if not faces:
            continue
        f0 = faces[0]
        b = [int(v) for v in f0.bbox]
        aimg, M = face_align.norm_crop2(orig, np.array(f0.kps, dtype=np.float32), 256)
        d = np.abs(raw.astype(np.int16) - aimg.astype(np.int16))
        print(f"  bbox={b}")
        print(f"  RAW(aligned256) vs TARGET(aligned256): mean={d.mean():.2f} max={d.max():.0f}")
        if mask is not None:
            gm = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255.0
            roi = gm[max(0, b[1]):b[3], max(0, b[0]):b[2]]
            print(f"  mask stats: max={gm.max():.3f} mean_bbox={roi.mean():.3f} mean_full={gm.mean():.3f}")
        x1, y1, x2, y2 = [max(0, v) for v in [b[0], b[1], b[2], b[3]]]
        df = np.abs(orig[y1:y2, x1:x2].astype(np.int16) - final[y1:y2, x1:x2].astype(np.int16))
        print(f"  FINAL vs ORIGINAL (bbox): mean={df.mean():.2f} max={df.max():.0f} px>15={int((df > 15).sum())}")
        dw = np.abs(orig[y1:y2, x1:x2].astype(np.int16) - warped[y1:y2, x1:x2].astype(np.int16))
        print(f"  WARPED vs ORIGINAL (bbox): mean={dw.mean():.2f} max={dw.max():.0f}")
    print("[DONE]")
finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
