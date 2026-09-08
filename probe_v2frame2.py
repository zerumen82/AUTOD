import os, sys, pickle
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
    import onnxruntime as ort
    from insightface.utils import face_align
    from roop.face_util_rotation import get_all_faces_smart

    VIDEO_SRC = r".autodeep_temp\5f64a2298ca57e5f726ce95f21c3d102fc09de834b2e002a81dfda2fd3a6b226\Real_Kinky__20251102__1984997195783749785_1_19849971259611709450.mp4"
    VIDEO_OUT = r"output\Real_Kinky__20251102__1984997195783749785_1_19849971259611709450.mp4"
    MODEL = r"models\inswapper_128_facefusion.onnx"

    cap = cv2.VideoCapture(VIDEO_SRC)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ok, frame = cap.read()
    cap.release()
    assert ok, "no se pudo leer frame 2 source"
    print(f"[FRAME] source vf2: {frame.shape}")

    faces = get_all_faces_smart(frame, min_score=None, for_target=True) or []
    print(f"[DETECT] {len(faces)} caras pipeline en vf2")
    for i, f in enumerate(faces):
        b = [int(v) for v in f.bbox]
        print(f"  face{i}: bbox={b} det={f.det_score:.2f} emb_norm={np.linalg.norm(np.asarray(f.embedding, dtype=np.float32)):.3f}")

    face = faces[0]
    tgt_kps = np.array(face.kps, dtype=np.float32)
    aimg_tgt, M = face_align.norm_crop2(frame, tgt_kps, 128)
    aimg_tgt_rgb = cv2.cvtColor(aimg_tgt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob_tgt = aimg_tgt_rgb.transpose(2, 0, 1)[np.newaxis, ...]

    sess = ort.InferenceSession(MODEL, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    emap = None
    for ini in sess.get_inputs():
        pass
    print(f"[MODEL] cargado. inputs={[i.name for i in sess.get_inputs()]}")

    try:
        raw = np.load(r"models\inswapper_128_facefusion_emap.npy")
        emap = raw
        print("[EMAP] emap externo cargado")
    except Exception:
        pass
    if emap is None:
        for f in os.listdir("models"):
            if "emap" in f.lower():
                try:
                    emap = np.load(os.path.join("models", f))
                    print(f"[EMAP] {f}")
                    break
                except Exception:
                    pass

    def run_swap(src_emb):
        src_emb = np.asarray(src_emb, dtype=np.float32).flatten()
        src_emb = src_emb / np.linalg.norm(src_emb)
        if emap is not None:
            blob_src = np.dot(src_emb.reshape((1, -1)), emap)
            blob_src /= np.linalg.norm(blob_src)
        else:
            blob_src = src_emb[np.newaxis, :]
        res = sess.run(None, {sess.get_inputs()[0].name: blob_tgt, sess.get_inputs()[1].name: blob_src.astype(np.float32)})[0]
        swapped = res[0].transpose(1, 2, 0)
        if swapped.max() < 2.0:
            swapped = swapped * 255.0
        swapped = swapped.clip(0, 255).astype(np.uint8)
        return cv2.cvtColor(swapped, cv2.COLOR_RGB2BGR)

    target_emb = np.asarray(face.embedding, dtype=np.float32)

    with open("probe_cache_emb.pkl", "rb") as fh:
        pool = pickle.load(fh)
    pool_sorted = sorted(pool, key=lambda p: p['quality'], reverse=True)
    master = pool_sorted[0]
    master_emb = np.asarray(master['embedding'], dtype=np.float32)
    print(f"[MASTER proxy] {master['name']} quality={master['quality']:.2f}")

    def cos(a, b):
        a = np.asarray(a, dtype=np.float32).flatten()
        b = np.asarray(b, dtype=np.float32).flatten()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    print(f"[SIM] proxy_master~target0 = {cos(master_emb, target_emb):.3f}")
    pool_sims = [cos(p['embedding'], target_emb) for p in pool]
    print(f"[SIM] pool_max~target0 = {max(pool_sims):.3f}")

    swapped_proxy = run_swap(master_emb)
    swapped_self = run_swap(target_emb)

    def diff_stats(a, b):
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()
        dm = np.abs(a.astype(np.int16) - b.astype(np.int16)).max()
        return d, dm

    dp = diff_stats(swapped_proxy, aimg_tgt)
    ds = diff_stats(swapped_self, aimg_tgt)
    print(f"[RAW] swap con MASTER proxy  vs target: mean={dp[0]:.2f} max={dp[1]:.0f}")
    print(f"[RAW] swap con SELF (control) vs target: mean={ds[0]:.2f} max={ds[1]:.0f}")

    cap2 = cv2.VideoCapture(VIDEO_OUT)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 1)
    ok2, out_frame = cap2.read()
    cap2.release()
    assert ok2, "no se pudo leer frame 2 output"
    print(f"[FRAME] output vf2: {out_frame.shape}")

    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    fa = frame[y1:y2, x1:x2]
    fb = out_frame[y1:y2, x1:x2]
    if fa.shape == fb.shape:
        d = np.abs(fa.astype(np.int16) - fb.astype(np.int16))
        print(f"[FINAL] bbox {face.bbox} diff mean={d.mean():.2f} max={d.max():.0f} px>{d.max()//1*0+15}: {(d > 15).sum()}")
    else:
        print(f"[FINAL] shapes no coinciden: {fa.shape} vs {fb.shape}")
    print("[DONE]")
finally:
    if dll_renamed:
        os.rename(DLL_DIR + "_BACKUP", DLL_DIR)
