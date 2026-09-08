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

    VIDEO_SRC = r".autodeep_temp\5f64a2298ca57e5f726ce95f21c3d102fc09de834b2e002a81dfda2fd3a6b226\Real_Kinky__20251102__1984997195783749785_1_19849971259611709450.mp4"
    VIDEO_OUT = r"output\Real_Kinky__20251102__1984997195783749785_1_19849971259611709450.mp4"

    CAPTION = {1: "vf2", 59: "vf60", 119: "vf120", 199: "vf200", 499: "vf500", 859: "vf860"}
    REGIONS = {
        "poster": (1129, 432, 1155, 465),
        "main":   (544, 126, 667, 380),
        "bighead": (733, 0, 1003, 262),
    }

    cap_s = cv2.VideoCapture(VIDEO_SRC)
    cap_o = cv2.VideoCapture(VIDEO_OUT)
    for idx in sorted(CAPTION.keys()):
        cap_s.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_o.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_s, f_s = cap_s.read()
        ok_o, f_o = cap_o.read()
        if not ok_s or not ok_o:
            print(f"[{CAPTION[idx]}] read fail")
            continue
        print(f"--- {CAPTION[idx]} ---")
        for name, (x1, y1, x2, y2) in REGIONS.items():
            fa = f_s[y1:y2, x1:x2]
            fb = f_o[y1:y2, x1:x2]
            d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
            print(f"  {name:8s} ({x1},{y1})-({x2},{y2}) mean={d.mean():5.2f} max={d.max():4.0f} px>15={int((d > 15).sum()):5d} px>30={int((d > 30).sum()):5d}")
    cap_s.release()
    cap_o.release()
    print("[DONE]")
finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
