import sys, os, threading, time, json, glob
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

# Step 1: Import what on_src_changed needs
import roop.globals
import ui.globals
import ui.tabs.faceswap.state as state
from ui.globals import _write_console as _log

# Step 2: Manually call _log like on_src_changed does
_log("[TEST] _write_console direct call from main thread")
print("[TEST] print call (should also work)")

# Step 3: Import logic to call extract_face_images
import ui.tabs.faceswap.logic as logic

# Step 4: Create state like wire_events would
state.current_input_page = 0

# Step 5: Actually find a real image and call on_src_changed logic
# Find any image file in the project or temp
test_images = glob.glob('.autodeep_temp/**/*.jpg', recursive=True) + glob.glob('.autodeep_temp/**/*.png', recursive=True) + glob.glob('assets/**/*.png', recursive=True) + glob.glob('*.jpg')
test_images = [f for f in test_images if os.path.getsize(f) > 5000]

if test_images:
    img_path = test_images[0]
    print(f'[TEST] Using image: {img_path}')
    
    # Simulate what on_src_changed does
    files = [type('FakeFile', (), {'name': img_path})()]
    
    roop.globals.INPUT_FACESETS = []
    ui.globals.ui_input_thumbs = []
    
    for f in files:
        f_path = f.name
        faces_data = logic.extract_face_images(f_path, is_source_face=True)
        if not faces_data:
            print(f'[TEST] No faces in {os.path.basename(f_path)}')
        for face_obj, face_img in faces_data:
            if face_img.shape[0] < 40 or face_img.shape[1] < 40:
                continue
            from roop.types import FaceSet
            face_obj.face_img = face_img
            face_obj.face_img_ref = face_img
            roop.globals.INPUT_FACESETS.append(FaceSet(faces=[face_obj], name=os.path.basename(f_path)))
            ui.globals.ui_input_thumbs.append(ui.globals.util.convert_to_gradio(face_img, is_rgb=True)) if hasattr(ui.globals, 'util') else None
    
    total_faces = len(ui.globals.ui_input_thumbs)
    _log(f'[LOAD] Fuentes: {total_faces} caras en {len(files)} archivo(s)')
    print(f'[TEST] Loaded {total_faces} faces from {len(files)} file(s)')
    
    # Write to desktop
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop', 'debug_autoauto.txt')
    with open(desktop, 'w') as f:
        f.write(f'[LOAD] Fuentes: {total_faces} caras en {len(files)} archivo(s)\n')
    print(f'[TEST] Wrote to {desktop}')
else:
    print('[TEST] No test images found')
    
    # Even without real images, test basic file write
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop', 'debug_autoauto.txt')
    with open(desktop, 'w') as f:
        f.write('[TEST] No test images found but file write works\n')
    _log('[TEST] Write to desktop successful')

print('[TEST] Done')
