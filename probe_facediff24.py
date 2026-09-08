import os, sys, glob, shutil

# dll rename convention (importing roop from standalone script)
DLL_DIR = os.path.abspath("dll")
dll_renamed = False
if os.path.isdir(DLL_DIR):
    backup = DLL_DIR + "_BACKUP"
    if not os.path.isdir(backup):
        os.rename(DLL_DIR, backup)
        dll_renamed = True

try:
    import cv2
    import numpy as np
    from roop.face_util import get_face_analyser, detect_faces_robust

    sys.path.insert(0, os.path.abspath("."))
    analyser = get_face_analyser()

    pairs = []
    for inp in glob.glob(r".autodeep_temp\**\*", recursive=True):
        if not os.path.isfile(inp):
            continue
        name = os.path.basename(inp)
        out = os.path.join("output", name[:-5] + ".jpg" if name.endswith(".jpeg") else name)
        if os.path.isfile(out):
            pairs.append((inp, out))

    print(f"pairs: {len(pairs)}")
    rows = []
    for inp, out in sorted(pairs, key=lambda p: os.path.basename(p[0])):
        a = cv2.imread(inp)          # BGR input
        b = cv2.imread(out)          # BGR output
        if a is None or b is None:
            rows.append((os.path.basename(inp), "READ_FAIL", "", ""))
            continue
        if b.shape != a.shape:
            b = cv2.resize(b, (a.shape[1], a.shape[0]))
        a_rgb = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        try:
            faces = detect_faces_robust(a_rgb, analyser) or []
        except Exception as e:
            rows.append((os.path.basename(inp), f"DET_ERR:{e}", "", ""))
            continue
        if not faces:
            rows.append((os.path.basename(inp), "NO_FACES", "", ""))
            continue
        stats = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(a.shape[1], x2), min(a.shape[0], y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            d = np.abs(a[y1:y2, x1:x2].astype(np.float32) - b[y1:y2, x1:x2].astype(np.float32))
            stats.append((round(float(d.mean()), 2), round(float(d.max()), 1), x2 - x1, y2 - y1, round(float(f.det_score), 3)))
        if not stats:
            rows.append((os.path.basename(inp), "NO_VALID", "", ""))
            continue
        best = max(stats, key=lambda s: s[0])
        rows.append((os.path.basename(inp), f"faces={len(stats)}", f"best_face_mean={best[0]} max={best[1]} {best[2]}x{best[3]} det={best[4]}", f"all={[(s[0], s[1]) for s in stats]}"))

    print(f"{'file':28s} {'faces':>6s} {'best face diff':>34s}")
    for r in sorted(rows, key=lambda r: r[1] if isinstance(r[1], str) else 0):
        print(f"{r[0]:28s} {r[1]:>6s}  {r[2]}")
        if r[3]:
            print(f"{'':28s} {'':>6s}  {r[3]}")
finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
