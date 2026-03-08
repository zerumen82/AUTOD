#!/usr/bin/env python3
import os,sys,ctypes
print('Python:', sys.version.split()[0])
try:
    import onnxruntime as ort
    print('onnxruntime', ort.__version__, 'providers=', ort.get_available_providers())
except Exception as e:
    print('onnxruntime import error:', e)
try:
    import torch
    print('torch', torch.__version__, 'cuda_available=', torch.cuda.is_available(), 'cuda_version=', torch.version.cuda)
    if torch.cuda.is_available():
        try:
            print('GPU:', torch.cuda.get_device_name(0))
        except Exception:
            pass
except Exception as e:
    print('torch import error:', e)

# Buscar cudnn64_*.dll en PATH y en CUDA_DIR
paths = os.environ.get('PATH','').split(os.pathsep)
cudnn_candidates = []
for p in paths:
    try:
        for f in os.listdir(p):
            if f.lower().startswith('cudnn') and f.lower().endswith('.dll'):
                cudnn_candidates.append(os.path.join(p,f))
    except:
        pass
# también buscar en posibles CUDA dirs
possible = [r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA']
for base in possible:
    if os.path.exists(base):
        for root,dirs,files in os.walk(base):
            for f in files:
                if f.lower().startswith('cudnn') and f.lower().endswith('.dll'):
                    candid = os.path.join(root,f)
                    if candid not in cudnn_candidates:
                        cudnn_candidates.append(candid)

print('\nFound cudnn DLL candidates:')
for i,f in enumerate(cudnn_candidates[:20]):
    print(i+1,f)

# Intentar llamar cudnnGetVersion si es posible
def try_get_version(path):
    try:
        dll = ctypes.WinDLL(path)
        try:
            fn = dll.cudnnGetVersion
            fn.restype = ctypes.c_size_t
            v = fn()
            return v
        except Exception as e:
            return f'no cudnnGetVersion symbol ({e})'
    except Exception as e:
        return f'load fail ({e})'

for f in cudnn_candidates[:20]:
    print('\nTesting', f)
    print(' =>', try_get_version(f))
