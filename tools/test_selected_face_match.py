import sys
sys.path.insert(0, r'd:\\PROJECTS\\AUTOAUTO')
from roop.ProcessMgr import ProcessMgr
import numpy as np

pm = ProcessMgr()
class F: pass

# Simulate selected face from first frame
s = F()
s.embedding = [1.0, 0.0, 0.0]
s.bbox = (10, 10, 50, 50)
pm._setup_selected_faces_frame_for_video('video1.mp4', [s])

# Candidate faces in later frames
c1 = F()
c1.embedding = [0.9, 0.1, 0.0]
c1.bbox = (12, 12, 48, 48)

c2 = F()
c2.embedding = [0.0, 1.0, 0.0]
c2.bbox = (200, 200, 240, 240)

res = pm._select_deterministic_face(None, [c1, c2], face_swap_mode='selected_faces_frame', video_path='video1.mp4')
print('selected is c1:', res is c1)
# Test bbox fallback when embeddings missing
pm2 = ProcessMgr()
sp = F(); sp.embedding = None; sp.bbox = (100,100,150,150)
pm2._setup_selected_faces_frame_for_video('video2.mp4',[sp])
cc1 = F(); cc1.embedding = None; cc1.bbox = (98,98,152,152)
cc2 = F(); cc2.embedding = None; cc2.bbox = (10,10,20,20)
res2 = pm2._select_deterministic_face(None, [cc1, cc2], face_swap_mode='selected_faces_frame', video_path='video2.mp4')
print('selected bbox match:', res2 is cc1)
