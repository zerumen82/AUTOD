import sys, os
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import glob
imgs = glob.glob('.autodeep_temp/**/*.jpg', recursive=True) + glob.glob('.autodeep_temp/**/*.png', recursive=True)
imgs = [f for f in imgs if os.path.getsize(f) > 10000]
print(f'Found {len(imgs)} test images')
if imgs:
    img = imgs[0]
    print(f'Image: {img} ({os.path.getsize(img)} bytes)')
    
    import roop.face_util as fu
    faces = fu.extract_face_images(img, (False, 0), is_source_face=True)
    print(f'Faces detected: {len(faces)}')
    for i, (fo, fi) in enumerate(faces):
        print(f'  Face {i}: bbox={fo.bbox}, img_shape={fi.shape}')
else:
    print('No test images found')
    # List a few files in autodeep_temp
    for root, dirs, files in os.walk('.autodeep_temp'):
        for f in files[:5]:
            fp = os.path.join(root, f)
            print(f'  {fp} ({os.path.getsize(fp)} bytes)')
        break
