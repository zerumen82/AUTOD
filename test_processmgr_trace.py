import os, sys, cv2, numpy as np
sys.path.insert(0, '.')
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.ProcessOptions import ProcessOptions
from roop.capturer import get_video_frame
from roop.analyser import get_face_single
from roop.types import FaceSet

src_path = r'D:\PROJECTS\uee\FACES\MARETA\mar.JPG'
tgt_path = r'D:\PROJECTS\uee\AUTO-DEEP\input\VBIDE\MORINVIDS\1_4940920829606102276.mp4'
src_img = cv2.imread(src_path)
src_face = get_face_single(src_img)

ProcessMgr._swap_call_count = 0

roop.globals.INPUT_FACESETS = [FaceSet(faces=[src_face])]
roop.globals.TARGET_FACES = []
roop.globals.distance_threshold = 0.45
roop.globals.blend_ratio = 1.0
roop.globals.face_swap_mode = 'selected_faces'
roop.globals.use_enhancer = True
roop.globals.enhancer_blend_factor = 0.30

options = ProcessOptions(
    {'face_swapper': {}, 'mask_xseg': {}},
    0.45, 1.0, 'selected_faces', 0, '', None, 1, False, False, True, 'seamless'
)

mgr = ProcessMgr()
mgr.initialize(roop.globals.INPUT_FACESETS, [], options)
print(f"ProcessMgr initialized")

frame = get_video_frame(tgt_path, 1)

# Monkey-patch the _process_face_swap_v21 to capture the swapped_face_aligned BEFORE debug writes
original_process = mgr._process_face_swap_v21

def patched_process(target_face, source_face, result_frame, **kwargs):
    # Call original
    result = original_process(target_face, source_face, result_frame, **kwargs)
    
    # After original, check the processors' FaceSwap model output directly
    fs = mgr.processors.get("faceswap")
    if fs:
        print(f"\n[TRACE] FaceSwap model_path: {fs.model_path}")
        print(f"[TRACE] FaceSwap source_is_image: {fs.source_is_image}")
        
        # Run FaceSwap directly with the same inputs to compare
        from roop.analyser import get_face_many
        from roop.face_util_rotation import get_all_faces_smart
        
        # Get fresh faces for comparison
        faces = get_all_faces_smart(result_frame, min_score=None, for_target=True)
        if faces:
            tgt_face = faces[0]
            direct_result = fs.Run(source_face, tgt_face, result_frame, paste_back=False)
            if isinstance(direct_result, tuple):
                s, m = direct_result
                print(f"[TRACE] Direct FaceSwap.Run: mean(RGB)=({s[:,:,0].mean():.0f},{s[:,:,1].mean():.0f},{s[:,:,2].mean():.0f})")
                white = np.count_nonzero(np.all(s > 250, axis=2)) / (256*256) * 100
                print(f"[TRACE] Direct white: {white:.1f}%")
    
    return result

mgr._process_face_swap_v21 = patched_process

# Process a frame
result = mgr.process_frame(frame, file_path=tgt_path)
print(f"\n[RESULT] shape={result.shape}")

# Check debug image
debug_dir = 'debug_swap'
raw = cv2.imread(os.path.join(debug_dir, '01_raw_model_f1.png'))
if raw is not None:
    print(f"[DEBUG_PNG] mean(RGB)={raw.mean(axis=(0,1))}")
    white = np.count_nonzero(np.all(raw > 250, axis=2)) / (raw.shape[0]*raw.shape[1]) * 100
    print(f"[DEBUG_PNG] white: {white:.1f}%")

mgr.Release()
print('DONE')
