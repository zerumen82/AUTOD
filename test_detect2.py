import sys, os, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import glob
imgs = glob.glob('.autodeep_temp/**/*.jpg', recursive=True)
imgs = [f for f in imgs if os.path.getsize(f) > 10000]
if not imgs:
    print('No images')
    os._exit(1)

img = imgs[0]
print(f'Testing: {os.path.basename(img)}', flush=True)

import roop.face_util as fu
print('Import done', flush=True)

try:
    import gc; gc.collect()
    start = time.time()
    faces = fu.extract_face_images(img, (False, 0), is_source_face=True)
    elapsed = time.time() - start
    print(f'Faces: {len(faces)} in {elapsed:.1f}s', flush=True)
    for i, (fo, fi) in enumerate(faces):
        print(f'  Face {i}: bbox={fo.bbox}', flush=True)
except Exception as e:
    print(f'Error: {e}', flush=True)
    import traceback
    traceback.print_exc()

print('Done', flush=True)
os._exit(0)
