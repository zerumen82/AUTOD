import os, sys, glob, shutil, time, traceback, pickle

# dll rename convention (importing roop from standalone script)
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

    # ---------- 0) Encontrar inputs reales del batch ----------
    inputs = {}
    for inp in glob.glob(r".autodeep_temp\**\*", recursive=True):
        if os.path.isfile(inp):
            inputs[os.path.basename(inp)] = inp
    out_names = sorted(os.listdir("output"))
    print(f"[PROBE] outputs: {len(out_names)}, inputs encontrados: {len(inputs)}")

    # ---------- 1) Cargar pool de fuentes como la UI (events.py:554-570) ----------
    # v5.1045: usa cache de embeddings (probe_cache_emb.pkl) si existe — evita re-extraer 732 fuentes
    import roop.globals
    from roop.types import FaceSet
    import ui.tabs.faceswap.logic as logic

    CACHE = os.path.abspath("probe_cache_emb.pkl")
    roop.globals.INPUT_FACESETS = []
    if os.path.isfile(CACHE):
        with open(CACHE, "rb") as f:
            pool = pickle.load(f)

        class _F:
            def __init__(self, emb, name):
                self.embedding = np.asarray(emb, dtype=np.float32).tolist()
                self.name = name

        pool_sorted = sorted(pool, key=lambda p: p['quality'], reverse=True)
        roop.globals.INPUT_FACESETS = [
            FaceSet(faces=[_F(p['embedding'], p['name'])], name=p['name']) for p in pool_sorted
        ]
        print(f"[PROBE] INPUT_FACESETS desde cache: {len(roop.globals.INPUT_FACESETS)} caras")
    else:
        SRC_DIR = r"D:\PROJECTS\uee\FACES\MARETA"
        src_files = sorted(os.path.join(SRC_DIR, f) for f in os.listdir(SRC_DIR) if os.path.isfile(os.path.join(SRC_DIR, f)))
        print(f"[PROBE] Pool MARETA: {len(src_files)} archivos")
        MIN_SRC_SIZE = 64
        t0 = time.time()
        for i, f_path in enumerate(src_files):
            try:
                faces_data = logic.extract_face_images(f_path, is_source_face=True)
                for face_obj, face_img in faces_data:
                    if face_img.shape[0] < MIN_SRC_SIZE or face_img.shape[1] < MIN_SRC_SIZE:
                        continue
                    face_img = np.ascontiguousarray(face_img)
                    face_obj.face_img = face_img
                    face_obj.face_img_ref = face_img
                    roop.globals.INPUT_FACESETS.append(FaceSet(faces=[face_obj], name=os.path.basename(f_path)))
            except Exception as e:
                print(f"[SRC_ERR] {os.path.basename(f_path)}: {e}")
            if (i + 1) % 200 == 0:
                print(f"[PROBE] {i+1}/{len(src_files)} fuentes ({time.time()-t0:.0f}s, {len(roop.globals.INPUT_FACESETS)} caras)")
        print(f"[PROBE] INPUT_FACESETS: {len(roop.globals.INPUT_FACESETS)} caras en {time.time()-t0:.0f}s")

    # ---------- 2) Refs por archivo como la UI (events.py:182-184, 146-150) ----------
    # Orden: sospechosos primero (frames 1-3 reciben debug detallado call_num<=3)
    weak_first = [
        "14126097_014_91ad.jpg", "22102089_023_66c8.jpg", "42551980.jpg",
        "64876795_115_b403.jpg", "41821452_028_a73f.jpg", "241247257.jpg",
        "290894114.jpg", "20260419_232343.jpg",
    ]
    if "--weak" in sys.argv:
        ordered = [n for n in weak_first if n in out_names]
        print("[PROBE] MODO --weak: solo los 8 archivos débiles")
    else:
        ordered = weak_first + [n for n in out_names if n not in weak_first]

    roop.globals.selected_face_references = {}
    for name in ordered:
        inp = inputs.get(name) or inputs.get(name[:-4] + ".jpeg") or inputs.get(name[:-4] + ".png")
        if not inp:
            print(f"[PROBE] INPUT NO ENCONTRADO para {name}")
            continue
        detected = logic.extract_face_images(inp, (True, 1), target_face_detection=True, ui_padding=0.5)
        if not detected:
            print(f"[PROBE] REF SIN CARAS: {name}")
            continue
        biggest = max(detected, key=lambda fd: (fd[0].bbox[2]-fd[0].bbox[0]) * (fd[0].bbox[3]-fd[0].bbox[1]))
        face_obj = biggest[0]
        roop.globals.selected_face_references[f"selected_face_ref_{name}"] = {
            'face_obj': face_obj, 'embedding': face_obj.embedding, 'bbox': face_obj.bbox,
        }
        print(f"[PROBE] ref {name}: cara {int(face_obj.bbox[2]-face_obj.bbox[0])}x{int(face_obj.bbox[3]-face_obj.bbox[1])}")

    # ---------- 3) ProcessMgr como core.batch_process_regular ----------
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions

    roop.globals.distance_threshold = 0.2
    roop.globals.blend_ratio = 1.0
    roop.globals.face_swap_mode = "selected_faces"
    roop.globals.use_enhancer = True
    roop.globals.blend_mode = "seamless"
    roop.globals.output_path = os.path.abspath("probe_repro_out")

    options = ProcessOptions(
        {'face_swapper': {}},
        roop.globals.distance_threshold,
        roop.globals.blend_ratio,
        roop.globals.face_swap_mode,
        0, "", None, 1, False, False, True, roop.globals.blend_mode,
    )
    process_mgr = ProcessMgr()
    process_mgr.initialize(roop.globals.INPUT_FACESETS, [], options)

    # ---------- 4) Instrumentacion ----------
    # v5.1045: medir diff RAW del swapper (aligned) para separar contenido vs máscara
    sw = process_mgr.processors["faceswap"]
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
                print(f"[RAW_SWAP] mean={d.mean():.3f} max={d.max():.1f} pct>8={(d.mean(axis=2)>8).mean()*100:.1f}%")
        except Exception as e:
            print(f"[RAW_SWAP_ERR] {e}")
        return out
    sw.Run = wrap_run

    orig_swap = ProcessMgr._process_face_swap_v21
    def wrapped_swap(self, source_face, target_face, result_frame, frame, enable_temporal_smoothing):
        out = orig_swap(self, source_face, target_face, result_frame, frame, enable_temporal_smoothing)
        tb = getattr(target_face, 'bbox', None)
        if tb is not None:
            x1, y1, x2, y2 = [int(v) for v in tb]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 - x1 > 10 and y2 - y1 > 10:
                d = np.abs(frame[y1:y2, x1:x2].astype(np.float32) - out[y1:y2, x1:x2].astype(np.float32))
                print(f"[WRAP_DIFF] bbox=({x1},{y1},{x2},{y2}) mean={d.mean():.3f} max={d.max():.1f} n_pix={(d>8).sum()}")
        return out
    ProcessMgr._process_face_swap_v21 = wrapped_swap

    orig_sel = ProcessMgr._select_source_face
    def wrapped_sel(self, target_face, candidate_faces, face_swap_mode, video_path=None, is_profile=False):
        src = orig_sel(self, target_face, candidate_faces, face_swap_mode, video_path, is_profile)
        kind = "MASTER" if (src is not None and not hasattr(src, 'bbox')) else ("FACE" if src is not None else "NONE")
        print(f"[WRAP_SRC] -> {kind}")
        return src
    ProcessMgr._select_source_face = wrapped_sel

    # ---------- 5) Procesar ----------
    from roop.capturer import get_image_frame
    os.makedirs("probe_repro_out", exist_ok=True)
    results = []
    for name in ordered:
        inp = inputs.get(name) or inputs.get(name[:-4] + ".jpeg") or inputs.get(name[:-4] + ".png")
        if not inp:
            continue
        frame = get_image_frame(inp)
        if frame is None:
            print(f"[PROBE] READ_FAIL {name}")
            continue
        print(f"\n===== {name} =====")
        t1 = time.time()
        out = process_mgr.process_frame(frame, file_path=inp)
        dt = time.time() - t1
        if out is None:
            out = frame
        cv2.imwrite(os.path.join("probe_repro_out", name), out)

        # diff cara-a-cara vs input (mismo detector que probe_facediff24)
        a_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detect_faces_robust(a_rgb, analyser) or []
        stats = []
        for f in faces:
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            if x2 - x1 < 10 or y2 - y1 < 10:
                continue
            d = np.abs(frame[y1:y2, x1:x2].astype(np.float32) - out[y1:y2, x1:x2].astype(np.float32))
            stats.append((round(float(d.mean()), 2), round(float(d.max()), 1), x2 - x1, y2 - y1))
        best = max(stats, key=lambda s: s[0]) if stats else (None, None, 0, 0)
        print(f"[REPRO] faces={len(stats)} best_face_mean={best[0]} max={best[1]} {best[2]}x{best[3]} ({dt:.1f}s)")

        # diff vs salida real del usuario
        user_out = cv2.imread(os.path.join("output", name))
        if user_out is not None and user_out.shape == out.shape:
            d = np.abs(user_out.astype(np.float32) - out.astype(np.float32))
            print(f"[VS_USER] mean={d.mean():.3f} max={d.max():.1f}")
        elif user_out is not None:
            print(f"[VS_USER] dims difieren: user={user_out.shape} repro={out.shape}")
        results.append((name, len(stats), best[0], best[1], best[2], best[3]))

    print("\n===== RESUMEN REPRO (vs baseline swap real ~10 mean / sin-swap ~1.5) =====")
    for r in results:
        print(f"{r[0]:28s} faces={r[1]:3d} best_face_mean={r[2]} max={r[3]} {r[4]}x{r[5]}")
finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
