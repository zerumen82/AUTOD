import os, sys, glob

DLL_DIR = os.path.abspath("dll")
dll_renamed = False
if os.path.isdir(DLL_DIR):
    backup = DLL_DIR + "_BACKUP"
    if not os.path.isdir(backup):
        os.rename(DLL_DIR, backup)
        dll_renamed = True

try:
    sys.path.insert(0, os.path.abspath("."))
    import numpy as np
    import cv2
    from roop.face_util import get_face_analyser, detect_faces_robust

    analyser = get_face_analyser()

    weak = [
        "14126097_014_91ad.jpg", "42551980.jpg", "64876795_115_b403.jpg",
        "41821452_028_a73f.jpg", "241247257.jpg", "290894114.jpg",
        "20260419_232343.jpg", "22102089_023_66c8.jpg", "63394449_015_d8e1.jpg",
    ]
    inputs = {}
    for inp in glob.glob(r".autodeep_temp\**\*", recursive=True):
        if os.path.isfile(inp):
            inputs[os.path.basename(inp)] = inp

    for name in weak:
        inp = inputs.get(name) or inputs.get(name[:-4] + ".jpeg") or inputs.get(name[:-4] + ".png")
        if not inp:
            continue
        a = cv2.imread(inp)
        b = cv2.imread(os.path.join("output", name))
        if a is None or b is None or b.shape != a.shape:
            print(f"=== {name} READ_FAIL ===")
            continue
        d = np.abs(a.astype(np.float32) - b.astype(np.float32))
        strong = d > 15  # cambios reales (no ruido JPEG)
        ys, xs = np.where(strong.any(axis=2))
        print(f"=== {name} ({a.shape[1]}x{a.shape[0]}) pix_fuertes={(strong.any(axis=2)).sum()} ===")
        if len(xs) == 0:
            print("  NINGUN pixel fuerte >15")
            continue
        # bounding box de los cambios fuertes
        print(f"  region_fuerte: x[{xs.min()},{xs.max()}] y[{ys.min()},{ys.max()}]")
        # caras detectadas en input con el detector del pipeline (get_all_faces_smart)
        a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        from roop.face_util_rotation import get_all_faces_smart
        faces = get_all_faces_smart(a, min_score=None, for_target=True) or []
        print(f"  caras(pipeline): {len(faces)}")
        for i, f in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            overlap = ((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)).sum()
            cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
            # diff medio en el bbox de la cara
            dx1, dy1 = max(0, x1), max(0, y1)
            dx2, dy2 = min(a.shape[1], x2), min(a.shape[0], y2)
            if dx2 - dx1 > 5 and dy2 - dy1 > 5:
                dm = float(d[dy1:dy2, dx1:dx2].mean())
            else:
                dm = -1
            print(f"    #{i}: bbox=({x1},{y1},{x2},{y2}) centro=({cx},{cy}) det={f.det_score:.2f} diff_media_bbox={dm:.2f} pix_fuertes_en_cara={overlap}")
        # caras del probe (detect_faces_robust) para comparar
        faces2 = detect_faces_robust(a_rgb, analyser) or []
        print(f"  caras(probe): {len(faces2)}")
        for i, f in enumerate(faces2):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            print(f"    #{i}: bbox=({x1},{y1},{x2},{y2}) det={f.det_score:.2f}")
finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
