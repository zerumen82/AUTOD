import shutil
from pathlib import Path
root = Path(__file__).resolve().parents[1] / 'dll' / 'cudnn9'
src_include = root / 'cudnn_cuda12.9' / 'libcudnn_dev' / 'include'
src_libs = root / 'cudnn_cuda12.9' / 'libcudnn_dev' / 'lib' / '12.9' / 'x64'
src_dlls = root / 'cudnn_cuda12.9' / 'libcudnn' / 'bin' / '12.9'
dst = root / 'cmake_layout'
(d := dst / 'include').mkdir(parents=True, exist_ok=True)
(dlib := dst / 'lib' / 'x64').mkdir(parents=True, exist_ok=True)
(dbin := dst / 'bin' / 'x64').mkdir(parents=True, exist_ok=True)

copied = {'include':0,'libs':0,'dlls':0}
for p in src_include.glob('**/*'):
    if p.is_file():
        shutil.copy2(p, d / p.name)
        copied['include']+=1
for p in src_libs.glob('*.lib'):
    shutil.copy2(p, dlib / p.name)
    copied['libs']+=1
for p in src_dlls.glob('*.dll'):
    shutil.copy2(p, dbin / p.name)
    copied['dlls']+=1
print('Copied:', copied)
print('Layout root:', dst)
print('Include files:', list((d).glob('*'))[:5])
print('Lib files:', list((dlib).glob('*'))[:10])
print('Dll files:', list((dbin).glob('*'))[:10])
