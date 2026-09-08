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
    from roop.face_util_rotation import get_all_faces_smart

    D = "debug_swap"
    for n in [1, 2, 3]:
        orig = cv2.imread(os.path.join(D, f"07_original_frame_f{n}.png"))
        mask = cv2.imread(os.path.join(D, f"05_mask_final_f{n}.png"))
        warped = cv2.imread(os.path.join(D, f"06_warped_face_f{n}.png"))
        if orig is None or mask is None:
            continue
        print(f"--- f{n} ---")
        faces = get_all_faces_smart(orig, min_score=None, for_target=True) or []
        for i, f in enumerate(faces):
            b = [int(v) for v in f.bbox]
            print(f"  face{i}: bbox={b} det={f.det_score:.2f}")
        gm = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) / 255.0
        ys, xs = np.where(gm > 0.5)
        if len(xs):
            print(f"  MASK>0.5: centro=({xs.mean():.0f},{ys.mean():.0f}) bbox=({xs.min()},{ys.min()})-({xs.max()},{ys.max()}) n={len(xs)}")
        else:
            print(f"  MASK>0.5: ninguno (max={gm.max():.3f})")
        wm = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        yd, xd = np.where(wm > 0)
        if len(xd):
            print(f"  WARPED no-negro: bbox=({xd.min()},{yd.min()})-({xd.max()},{yd.max()}) n={len(xd)}")
    print("[DONE]")
finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
