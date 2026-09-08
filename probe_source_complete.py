import os, sys, io
from contextlib import redirect_stdout

_prj = r'D:\PROJECTS\AUTOAUTO'
sys.path.insert(0, _prj)
os.chdir(_prj)

_dll = os.path.join(_prj, 'dll')
_dll_bak = os.path.join(_prj, 'dll_BACKUP')
_renamed = False
if os.path.exists(_dll) and not os.path.exists(_dll_bak):
    os.rename(_dll, _dll_bak)
    _renamed = True

import ui.tabs.faceswap.logic as logic

SRC_DIR = r'D:\PROJECTS\uee\FACES\MIAS'
all_files = sorted(f for f in os.listdir(SRC_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')))
sample = all_files[:12]

stats = {'ok': 0, 'incomplete': 0, 'nogeom': 0, 'nodetect': 0, 'det_low': 0}
buf = io.StringIO()
rejected = []

with redirect_stdout(buf):
    for name in sample:
        faces = logic.extract_face_images(os.path.join(SRC_DIR, name), is_source_face=True)
        if faces:
            stats['ok'] += 1

for line in buf.getvalue().split('\n'):
    if 'INCOMPLETA' in line:
        stats['incomplete'] += 1
        rejected.append(line.split('|')[0].strip())
    elif 'Sin geometr' in line:
        stats['nogeom'] += 1
    elif 'No se detect' in line:
        stats['nodetect'] += 1
    elif 'descartada' in line:
        stats['det_low'] += 1

print(f"MUESTRA {len(sample)} fotos | aceptadas={stats['ok']} | INCOMPLETA={stats['incomplete']} | sin_geom={stats['nogeom']} | det_bajo={stats['det_low']} | sin_det={stats['nodetect']}")
for r in rejected:
    print("  rechazo:", r)

if _renamed and os.path.exists(_dll_bak) and not os.path.exists(_dll):
    os.rename(_dll_bak, _dll)