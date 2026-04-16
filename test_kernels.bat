@echo off
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin;%PATH%
set PATH=C:\Users\INdaHouse\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib;%PATH%
cd /d D:\PROJECTS\AUTOAUTO
python -c "
import sys
import os
os.environ['PATH'] = r'D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\custom_nodes\hart\hart\kernels\build\lib.win-amd64-cpython-311\hart_backend;' + os.environ.get('PATH', '')
kernels_path = r'D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\custom_nodes\hart\hart\kernels\build\lib.win-amd64-cpython-311'
sys.path.insert(0, kernels_path)

try:
    import hart_backend.fused_kernels as fk
    print('SUCCESS: CUDA kernels loaded!')
    print('Functions:', [x for x in dir(fk) if not x.startswith('_')])
except Exception as e:
    print('FAILED:', e)
    import traceback
    traceback.print_exc()
"
