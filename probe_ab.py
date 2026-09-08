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
    import roop.globals
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions

    CACHE = os.path.abspath("probe_cache_emb.pkl")
    with open(CACHE, "rb") as f:
        pool = pickle.load(f)
    print(f"[POOL] {len(pool)} caras (cache sin face_img)")

    roop.globals.distance_threshold = 0.2
    roop.globals.blend_ratio = 1.0
    roop.globals.face_swap_mode = "selected_faces"
    roop.globals.use_enhancer = True
    roop.globals.blend_mode = "seamless"

    options = ProcessOptions({'face_swapper': {}}, 0.2, 1.0, "selected_faces", 0, "", None, 1, False, False, True, "seamless")
    mgr = ProcessMgr()
    mgr.initialize([], [], options)

    # fuentes de prueba
    class Wrap:
        def __init__(self, emb, img=None):
            self.embedding = np.asarray(emb, dtype=np.float32).tolist()
            if img is not None:
                self.face_img = img
                self.face_img_ref = img

    pool_sorted = sorted(pool, key=lambda p: p['quality'], reverse=True)
    master_emb = pool_sorted[0]['embedding']
    face193 = pool_sorted[193]['embedding'] if len(pool_sorted) > 193 else pool_sorted[0]['embedding']
    print(f"[SRC] master={pool_sorted[0]['name']} qual={pool_sorted[0]['quality']:.2f}; #193={pool_sorted[193]['name'] if len(pool_sorted)>193 else 'N/A'} qual={pool_sorted[193]['quality']:.2f}")

    # instrumentar detect_foreground_occlusion para medir cobertura de "occlusion"
    import roop.ProcessMgr as _pm
    _orig_occ = _pm.detect_foreground_occlusion
    def _wrap_occ(face_img, target_region):
        m = _orig_occ(face_img, target_region)
        frac = float(np.mean(m > 0.1))
        tr = target_region
        if tr is not None and tr.shape[:2] != face_img.shape[:2]:
            tr = cv2.resize(tr, (face_img.shape[1], face_img.shape[0]))
        if tr is not None:
            h, w = tr.shape[:2]
            tg = cv2.cvtColor(tr, cv2.COLOR_BGR2GRAY)
            mag = np.abs(cv2.Sobel(tg, cv2.CV_32F, 1, 1, ksize=3))
            thr = max(25, min(40, np.mean(mag) * 1.5))
            tex = (mag > thr).astype(np.uint8)
            hsv = cv2.cvtColor(tr, cv2.COLOR_BGR2HSV)
            skin = cv2.inRange(hsv, np.array([0, 12, 40]), np.array([20, 170, 220]))
            center = np.zeros((h, w), np.uint8)
            cv2.ellipse(center, (w // 2, h // 2), (int(w * 0.82), int(h * 0.85)), 0, 0, 360, 255, -1)
            notskin = (255 - skin) & center
            lap = np.abs(cv2.Laplacian(tg, cv2.CV_32F))
            lthr = max(35, np.mean(lap[lap > 0]) * 0.6 if np.any(lap > 0) else 40)
            lab = cv2.cvtColor(tr, cv2.COLOR_BGR2LAB).astype(np.float32)
            sp = lab[skin > 0]
            obj = np.zeros_like(skin)
            if len(sp) > 30:
                ld = np.sqrt(np.sum((lab - np.mean(sp, axis=0).reshape(1, 1, 3)) ** 2, axis=2))
                obj = ((ld > 65).astype(np.uint8) & center)
            print(f"    [OCC_SIGNALS] tex&notskin={np.mean(tex & notskin)*100:.1f}% detail={np.mean(lap > lthr)*100:.1f}% object(LAB65)={np.mean(obj>0)*100:.1f}% skin={np.mean(skin>0)*100:.1f}%")
        print(f"    [OCC_MASK] mean={np.mean(m):.3f} max={np.max(m):.1f} frac>0.1={frac*100:.1f}%")
        return m
    _pm.detect_foreground_occlusion = _wrap_occ

    # instrumentar el swapper para medir el diff RAW (aligned) vs original
    sw = mgr.processors["faceswap"]
    orig_run = sw.Run
    def wrap_run(source_face, target_face, frame, paste_back=False):
        out = orig_run(source_face, target_face, frame, paste_back)
        try:
            if isinstance(out, tuple):
                sf, M = out
            else:
                sf, M = out, None
            if M is not None:
                h, w = sf.shape[:2]
                ref = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
                d = np.abs(sf.astype(np.float32) - ref.astype(np.float32))
                print(f"    [RAW_SWAP] mean={d.mean():.3f} max={d.max():.1f} pct>8={(d.mean(axis=2)>8).mean()*100:.1f}%")
        except Exception as e:
            print(f"    [RAW_SWAP_ERR] {e}")
        return out
    sw.Run = wrap_run

    inputs = {}
    for inp in glob.glob(r".autodeep_temp\**\*", recursive=True):
        if os.path.isfile(inp):
            inputs[os.path.basename(inp)] = inp

    from roop.face_util_rotation import get_all_faces_smart

    def get_face0(name):
        inp = inputs.get(name) or inputs.get(name[:-4] + ".jpeg") or inputs.get(name[:-4] + ".png")
        a = cv2.imread(inp)
        faces = get_all_faces_smart(a, min_score=None, for_target=True) or []
        return a, faces[0] if faces else None

    def run_case(label, frame, target_face, source):
        mgr._current_video_key = "frame"
        t1 = time.time()
        out = mgr._process_face_swap_v21(source, target_face, frame.copy(), frame, False)
        dt = time.time() - t1
        x1, y1, x2, y2 = [int(v) for v in target_face.bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        d = np.abs(frame[y1:y2, x1:x2].astype(np.float32) - out[y1:y2, x1:x2].astype(np.float32))
        print(f"  [FINAL] {label}: mean={d.mean():.3f} max={d.max():.1f} n_pix>15={(d.max(axis=2)>15).sum()} ({dt:.1f}s)")
        try:
            kps = np.asarray(target_face.kps, dtype=np.float32)
            print(f"  [KPS] {label}: bbox=({x1},{y1},{x2},{y2}) kps0={kps[0].tolist()} kps3={kps[3].tolist()} kps4={kps[4].tolist()}")
            my = (kps[3][1] + kps[4][1]) / 2 - y1
            mx = (kps[3][0] + kps[4][0]) / 2 - x1
            mh, mw = (y2 - y1), (x2 - x1)
            mz = np.zeros_like(d[..., 0])
            cv2.ellipse(mz, (int(mx), int(my)), (max(4, int(mw * 0.30)), max(4, int(mh * 0.20))), 0, 0, 360, 1, -1)
            if mz.sum() > 30:
                dm = d[..., 0][mz > 0]
                db = d[..., 0][mz == 0]
                print(f"  [MOUTH_ZONE] {label}: boca mean={dm.mean():.3f} max={dm.max():.1f} | resto mean={db.mean():.3f} max={db.max():.1f}")
        except Exception as e:
            print(f"  [MOUTH_ZONE_ERR] {label}: {e}")
        return out

    print("\n=== CASO A: 14126097 face#0 + MASTER (repro usuario, esperado WEAK) ===")
    frame, tf = get_face0("14126097_014_91ad.jpg")
    run_case("MASTER", frame, tf, Wrap(master_emb))

    print("\n=== CASO B: 14126097 face#0 + FACE#193 (control, esperado STRONG?) ===")
    frame, tf = get_face0("14126097_014_91ad.jpg")
    run_case("FACE#193", frame, tf, Wrap(face193))

    print("\n=== CASO C: 126349928 face#0 + MASTER (control fuerte) ===")
    frame, tf = get_face0("126349928.jpg")
    run_case("MASTER", frame, tf, Wrap(master_emb))

    print("\n=== CASO D: 22102089 face#2 + MASTER (multitud, esperado WEAK) ===")
    inp = inputs.get("22102089_023_66c8.jpg")
    a = cv2.imread(inp)
    faces = get_all_faces_smart(a, min_score=None, for_target=True) or []
    tf2 = None
    for f in faces:
        if f.bbox[0] > 300 and f.bbox[2] < 600:  # la cara principal 393,145,542,353
            tf2 = f
            break
    if tf2 is None:
        tf2 = faces[2]
    run_case("MASTER face#2", a, tf2, Wrap(master_emb))

finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
