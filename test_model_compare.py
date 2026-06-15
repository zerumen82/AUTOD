import os, sys, cv2, numpy as np
sys.path.insert(0, '.')
import roop.globals
roop.globals.execution_providers = ['CPUExecutionProvider']

# USE the SAME detection pipeline as ProcessMgr
from roop.face_util_rotation import get_all_faces_smart
from roop.capturer import get_video_frame
from insightface.utils import face_align
import onnxruntime as ort
from roop.analyser import get_face_single

src_path = r'D:\PROJECTS\uee\FACES\MARETA\mar.JPG'
tgt_path = r'D:\PROJECTS\uee\AUTO-DEEP\input\VBIDE\MORINVIDS\1_4940920829606102276.mp4'
src_img = cv2.imread(src_path)
src_face = get_face_single(src_img)
print(f'src_face: {src_face is not None}')

frame = get_video_frame(tgt_path, 1)
print(f'frame: {frame.shape}')

# Use ProcessMgr's detection pipeline
faces = get_all_faces_smart(frame, min_score=None, for_target=True)
if not faces:
    print('No target faces detected')
    sys.exit(1)

tgt_face = faces[0]
print(f'Target face: bbox={tgt_face.bbox}, kps={tgt_face.kps[:2]}')

src_emb = np.array(src_face.embedding, dtype=np.float32).flatten()
norm = np.linalg.norm(src_emb)
if norm > 0:
    src_emb = src_emb / norm
tgt_kps = np.array(tgt_face.kps, dtype=np.float32)
aimg_tgt, M_tgt = face_align.norm_crop2(frame, tgt_kps, 128)
print(f'aimg_tgt: shape={aimg_tgt.shape}, range=[{aimg_tgt.min()},{aimg_tgt.max()}]')

cv2.imwrite('debug_swap/target_crop.png', aimg_tgt)

ff_path = os.path.join('.', 'models', 'inswapper_128_facefusion.onnx')
orig_path = os.path.join('.', 'models', 'inswapper_128.onnx')

ff_sess = ort.InferenceSession(ff_path, providers=['CPUExecutionProvider'])
ff_inp = ff_sess.get_inputs()
print(f'FaceFusion inputs: {[(i.name, i.shape) for i in ff_inp]}')

orig_sess = ort.InferenceSession(orig_path, providers=['CPUExecutionProvider'])
orig_inp = orig_sess.get_inputs()
print(f'Original inputs: {[(i.name, i.shape) for i in orig_inp]}')

blob_src = src_emb[np.newaxis, :]
blob_255 = aimg_tgt.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
blob_01 = (aimg_tgt.astype(np.float32) / 255.0).transpose(2, 0, 1)[np.newaxis, ...]

tests = [
    ('FF_255', ff_sess, ff_inp, blob_255),
    ('FF_01', ff_sess, ff_inp, blob_01),
    ('ORIG_255', orig_sess, orig_inp, blob_255),
    ('ORIG_01', orig_sess, orig_inp, blob_01),
]

for label, sess, inp, blob in tests:
    res = sess.run(None, {inp[0].name: blob, inp[1].name: blob_src})[0]
    s = res[0].transpose(1, 2, 0)
    if s.max() < 2.0:
        s = s * 255.0
    s = s.clip(0, 255).astype(np.uint8)
    white = np.count_nonzero(np.all(s > 250, axis=2)) / (128 * 128) * 100
    print(f'{label}: mean(RGB)=({s[:,:,0].mean():.0f},{s[:,:,1].mean():.0f},{s[:,:,2].mean():.0f}) white={white:.1f}%')
    cv2.imwrite(f'debug_swap/{label}.png', s)

print('DONE')
