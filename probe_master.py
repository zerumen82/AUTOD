import os, sys, glob, time, pickle

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
    from roop.face_util import get_face_analyser

    analyser = get_face_analyser()

    CACHE = os.path.abspath("probe_cache_emb.pkl")
    if os.path.isfile(CACHE):
        with open(CACHE, "rb") as f:
            pool = pickle.load(f)
        print(f"[CACHE] pool cargado: {len(pool)} caras")
    else:
        import roop.globals
        from roop.types import FaceSet
        import ui.tabs.faceswap.logic as logic

        SRC_DIR = r"D:\PROJECTS\uee\FACES\MARETA"
        src_files = sorted(os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR) if os.path.isfile(os.path.join(SRC_DIR, f)))
        pool = []
        t0 = time.time()
        for i, f_path in enumerate(src_files):
            try:
                faces_data = logic.extract_face_images(f_path, is_source_face=True)
                for face_obj, face_img in faces_data:
                    if face_img.shape[0] < 64 or face_img.shape[1] < 64:
                        continue
                    pool.append({
                        'name': os.path.basename(f_path),
                        'embedding': np.asarray(face_obj.embedding, dtype=np.float32),
                        'quality': getattr(face_obj, 'quality_score', 0) or getattr(face_obj, 'quality', 0) or 0,
                        'bbox': list(face_obj.bbox),
                    })
            except Exception:
                pass
        with open(CACHE, "wb") as f:
            pickle.dump(pool, f)
        print(f"[POOL] {len(pool)} caras en {time.time()-t0:.0f}s (cacheado)")

    def cosine(a, b):
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    # Master = mejor calidad (como ProcessMgr.initialize l.227-253)
    pool_sorted = sorted(pool, key=lambda p: p['quality'], reverse=True)
    master = pool_sorted[0]
    print(f"[MASTER] {master['name']} quality={master['quality']:.2f}")

    # Sim entre MASTER y cada cara del pool (para contexto)
    sims = [cosine(master['embedding'], p['embedding']) for p in pool]
    print(f"[MASTER vs pool] max={max(sims):.3f} (self), top5={sorted(sims, reverse=True)[:5]}")

    # Targets: 24 archivos; para cada uno, cara #0 del pipeline y del probe
    inputs = {}
    for inp in glob.glob(r".autodeep_temp\**\*", recursive=True):
        if os.path.isfile(inp):
            inputs[os.path.basename(inp)] = inp

    from roop.face_util_rotation import get_all_faces_smart
    from roop.face_util import detect_faces_robust

    print(f"\n{'file':26s} {'master~tgt0':>11s} {'poolmax~tgt0':>13s} {'tgt0 det':>8s} {'user_diff':>9s}")
    weak = {
        "14126097_014_91ad.jpg": 1.66, "42551980.jpg": 2.38, "64876795_115_b403.jpg": 2.79,
        "41821452_028_a73f.jpg": 2.97, "241247257.jpg": 3.49, "290894114.jpg": 3.50,
        "20260419_232343.jpg": 3.80, "63394449_015_d8e1.jpg": 4.92, "22102089_023_66c8.jpg": 1.57,
    }
    strong = {
        "126349928.jpg": 7.44, "272348544.jpg": 7.66, "334609987_1.jpg": 7.68,
        "334718736.jpg": 6.94, "341150274.jpg": 7.94, "344864156.jpg": 6.02,
        "354190361_1.jpg": 5.30, "68931176_252_47db.jpg": 5.32, "76076949.jpg": 6.89,
        "95248439_009_8fef.jpg": 5.72, "97226212.jpg": 4.32, "124428913.jpg": 5.39,
        "2026228.png": 6.58, "31863376_1.jpg": 7.53, "47159067.jpg": 8.48,
    }
    rows = []
    for name in list(weak.keys()) + list(strong.keys()):
        inp = inputs.get(name) or inputs.get(name[:-4] + ".jpeg") or inputs.get(name[:-4] + ".png")
        if not inp:
            continue
        a = cv2.imread(inp)
        if a is None:
            continue
        faces = get_all_faces_smart(a, min_score=None, for_target=True) or []
        if not faces:
            print(f"{name:26s} SIN caras pipeline")
            continue
        # cara #0 = la del usuario (la que ellos clickeaban = primera de mayor det)
        t0f = faces[0]
        emb0 = np.asarray(t0f.embedding, dtype=np.float32)
        m_t0 = cosine(master['embedding'], emb0)
        pool_sims = [cosine(p['embedding'], emb0) for p in pool]
        pmax = max(pool_sims)
        ud = weak.get(name) or strong.get(name)
        tag = "WEAK" if name in weak else "strong"
        rows.append((name, m_t0, pmax, tag, ud))
        print(f"{name:26s} {m_t0:11.3f} {pmax:13.3f} {t0f.det_score:8.2f} {ud:9.2f} {tag}")

    print("\n[ANALISIS]")
    weak_rows = [r for r in rows if r[3] == "WEAK"]
    strong_rows = [r for r in rows if r[3] == "strong"]
    print(f"WEAK:   master~target0 media={np.mean([r[1] for r in weak_rows]):.3f} max={max(r[1] for r in weak_rows):.3f}")
    print(f"STRONG: master~target0 media={np.mean([r[1] for r in strong_rows]):.3f} max={max(r[1] for r in strong_rows):.3f}")
finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
