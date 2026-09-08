import os, sys

_prj = r'D:\PROJECTS\AUTOAUTO'
sys.path.insert(0, _prj)
os.chdir(_prj)

_dll = os.path.join(_prj, 'dll')
_dll_bak = os.path.join(_prj, 'dll_BACKUP')
_renamed = False
if os.path.exists(_dll) and not os.path.exists(_dll_bak):
    os.rename(_dll, _dll_bak)
    _renamed = True
    print('dll dir temporarily renamed')

try:
    import numpy as np
    import roop.globals
    import ui.tabs.faceswap.logic as logic
    from roop.ProcessEntry import ProcessEntry
    from roop.types import FaceSet

    SRC_DIR = r'D:\PROJECTS\uee\FACES\MIAS'
    VIDEO = r'D:\PROJECTS\AUTOAUTO\.autodeep_temp\5f64a2298ca57e5f726ce95f21c3d102fc09de834b2e002a81dfda2fd3a6b226\Real_Kinky__20251102__1984997195783749785_1_19849971259611709450.mp4'
    OUT_DIR = os.path.join(_prj, 'output')
    OUT_MP4 = os.path.join(OUT_DIR, os.path.basename(VIDEO))
    # El mux final con ffmpeg (sin -y) se cuelga si el mp4 destino ya existe
    for stale in (OUT_MP4, OUT_MP4 + '.temp.mp4'):
        if os.path.exists(stale):
            os.remove(stale)
            print(f'output anterior eliminado: {os.path.basename(stale)}')

    # 1. Pool de origen (igual que events.py on_src_changed)
    roop.globals.INPUT_FACESETS = []
    src_files = sorted(
        os.path.join(SRC_DIR, f)
        for f in os.listdir(SRC_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    )
    loaded_files = 0
    for f_path in src_files:
        try:
            faces_data = logic.extract_face_images(f_path, is_source_face=True)
        except Exception as e:
            print(f"[POOL_ERR] {os.path.basename(f_path)}: {e}")
            continue
        if not faces_data:
            continue
        loaded_files += 1
        for face_obj, face_img in faces_data:
            if face_img.shape[0] < 64 or face_img.shape[1] < 64:
                continue
            face_obj.face_img = np.ascontiguousarray(face_img)
            face_obj.face_img_ref = face_obj.face_img
            roop.globals.INPUT_FACESETS.append(FaceSet(faces=[face_obj], name=os.path.basename(f_path)))
    print(f"[POOL] {len(roop.globals.INPUT_FACESETS)} caras en {loaded_files} archivos desde {SRC_DIR}")

    # 2. Cara destino: la principal del frame 1 (bbox=(542,121,666,379) segun run 09:47)
    from roop.capturer import get_video_frame
    faces_video = logic.extract_face_images(VIDEO, (True, 1), target_face_detection=True, ui_padding=0.5)
    target = None
    for fo, fi in faces_video:
        b = fo.bbox
        if abs(b[0] - 542) < 25 and abs(b[1] - 121) < 25:
            target = fo
            break
    if target is None and faces_video:
        target = faces_video[0][0]
        print(f"[TARGET] fallback: primera cara bbox={target.bbox}")
    if target is None:
        print("[TARGET] ERROR: sin cara destino")
        sys.exit(1)
    roop.globals.TARGET_FACES = [target]
    basename = os.path.basename(VIDEO)
    roop.globals.selected_face_references = {}
    roop.globals.selected_face_references[f'selected_face_ref_{basename}'] = {
        'face_obj': target, 'embedding': target.embedding, 'bbox': target.bbox,
    }
    print(f"[TARGET] referencia destino: bbox={target.bbox}")

    # 3. Globals (mismos que config cargado: selected_faces, dist 0.2, blend 1.0)
    roop.globals.face_swap_mode = 'selected_faces'
    roop.globals.use_enhancer = True
    roop.globals.distance_threshold = 0.2
    roop.globals.blend_ratio = 1.0
    roop.globals.blend_mode = 'seamless'
    roop.globals.output_path = OUT_DIR
    roop.globals.skip_audio = False
    roop.globals.num_swap_steps = 1
    roop.globals.temporal_smoothing = True
    roop.globals.processing = True

    # 4. Ejecutar el batch igual que logic.start_swap -> batch_process_regular
    from roop.core import batch_process_regular
    entry = ProcessEntry(VIDEO, 0, 0, 30.0)
    for prog, msg in batch_process_regular([entry], None, '', False, {'layers': []}, 1, None, 0, True):
        print(f"[UI] Progreso global: {prog:.1f}% - {msg}", flush=True)
    print("[DONE] Procesamiento completado")

finally:
    if _renamed and os.path.exists(_dll_bak) and not os.path.exists(_dll):
        os.rename(_dll_bak, _dll)
        print('dll dir restored')
