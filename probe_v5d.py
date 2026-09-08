import os, sys, pickle, time, traceback
import numpy as np
import cv2

# --- dll rename (roop standalone convention) ---
DLL_DIR = os.path.abspath("dll")
dll_renamed = False
if os.path.isdir(DLL_DIR):
    backup = DLL_DIR + "_BACKUP"
    if not os.path.isdir(backup):
        os.rename(DLL_DIR, backup); dll_renamed = True

try:
    sys.path.insert(0, os.path.abspath("."))
    import roop.globals
    from roop.types import FaceSet
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions
    import ui.tabs.faceswap.logic as logic

    # ---------- cache source embeddings ----------
    CACHE = os.path.abspath("probe_cache_emb.pkl")
    roop.globals.INPUT_FACESETS = []
    with open(CACHE, "rb") as f:
        pool = pickle.load(f)
    pool_sorted = sorted(pool, key=lambda p: p['quality'], reverse=True)

    class _F:
        def __init__(self, emb, name, bbox):
            self.embedding = np.asarray(emb, dtype=np.float32).tolist()
            self.name = name
            self.bbox = tuple(bbox) if bbox else None
            self.face_img = None
            self.face_img_ref = None
            self.det_score = 0.0
    # cargar face_img perezamente NO — ProcessMgr usa solo embedding+name en seleccion
    roop.globals.INPUT_FACESETS = [FaceSet(faces=[_F(p['embedding'], p['name'], p.get('bbox'))], name=p['name']) for p in pool_sorted]
    print(f"[PROBE] INPUT_FACESETS cache: {len(roop.globals.INPUT_FACESETS)} caras")

    # ---------- target face ref (MrT4887 frame 700 == clip start) ----------
    VIDEO = r".test_clip\mrt_clip.mp4"
    ORIG = r".autodeep_temp\ac3cb005e4f16b7b0d19a4a0e42bc546cb67fae2e123bf9e0dd0ed2c6bebff08\MrT4887_20260217__2023778421474816433_1_20237783603595223040.mp4"
    raw = detect_faces_robust(rgb, a)
    from roop.face_util import get_face_analyser, detect_faces_robust
    v0 = cv2.VideoCapture(ORIG); v0.set(cv2.CAP_PROP_POS_FRAMES, 700); _, f0 = v0.read(); v0.release()
    assert f0 is not None, "no se lee frame 700 orig"
    a = get_face_analyser()
    rgb = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)
    raw = detect_faces_robust(rgb, a)
    # coger la cara más grande con embedding válido
    cand = [f for f in raw if getattr(f, 'embedding', None) is not None and len(f.embedding) >= 128]
    cand = sorted(cand, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    assert cand, "sin cara objetivo con embedding válido en frame 700"
    target_face = cand[0]
    cand = sorted(cand, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]), reverse=True)
    assert cand, "sin cara objetivo con embedding válida en frame 700"
    target_face = cand[0]
    roop.globals.selected_face_references = {"selected_face_ref_mrt_clip": {'face_obj': target_face, 'embedding': target_face.embedding, 'bbox': target_face.bbox}}
    print(f"[PROBE] target face {int(target_face.bbox[2]-target_face.bbox[0])}x{int(target_face.bbox[3]-target_face.bbox[1])} emb={int(np.linalg.norm(target_face.embedding))}")

    # ---------- config ProcessMgr ----------
    roop.globals.distance_threshold = 0.2
    roop.globals.blend_ratio = 1.0
    roop.globals.face_swap_mode = "selected_faces"
    roop.globals.use_enhancer = True
    roop.globals.blend_mode = "seamless"
    roop.globals.output_path = os.path.abspath(".probe_v5d_out")
    os.makedirs(roop.globals.output_path, exist_ok=True)

    options = ProcessOptions({'face_swapper': {}}, roop.globals.distance_threshold, roop.globals.blend_ratio, roop.globals.face_swap_mode, 0, "", None, 1, False, False, True, roop.globals.blend_mode)
    pm = ProcessMgr()
    pm.initialize(roop.globals.INPUT_FACESETS, [], options)

    # ---------- procesar 60 frames ----------
    cap = cv2.VideoCapture(VIDEO)
    n = min(30, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    print(f"[PROBE] procesando {n} frames...")
    t0 = time.time()
    accepted = 0; rejected = 0; safety_hotspot = 0; safety_ok = 0
    for i in range(n):
        ret, frame = cap.read()
        if not ret: break
        try:
            out = pm.process_frame(frame, file_path=os.path.basename(VIDEO))
            if out is None: out = frame
        except Exception:
            traceback.print_exc(); out = frame
        # stats: proceso fuera del loop de tracking para no ralentizar
        cv2.imwrite(os.path.join(roop.globals.output_path, f"frame_{i:04d}.png"), out)
        if (i+1) % 20 == 0:
            print(f"  frame {i+1}/{n} ({time.time()-t0:.1f}s)", flush=True)
    cap.release()
    print(f"[PROBE] DONE {n} frames en {time.time()-t0:.1f}s")
finally:
    if dll_renamed and os.path.isdir(DLL_DIR+"_BACKUP") and not os.path.isdir(DLL_DIR):
        os.rename(DLL_DIR+"_BACKUP", DLL_DIR)
