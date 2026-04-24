#!/usr/bin/env python
"""Verificar que todos los componentes usan GPU"""

import os
import sys

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if os.path.exists(cuda_bin):
    os.environ["PATH"] = cuda_bin + ";" + os.environ.get("PATH", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("VERIFICACIÓN DE USO DE GPU")
print("="*60)

# 1. Torch
import torch
print(f"\n[1] Torch CUDA disponible: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"    Dispositivo: {torch.cuda.get_device_name(0)}")

# 2. ONNX Runtime providers
import onnxruntime as ort
provs = ort.get_available_providers()
print(f"\n[2] ONNX providers: {provs}")

# 3. InsightFace detector
print("\n[3] Cargando Face detector (InsightFace)...")
from roop.face_util import get_face_analyser
analyzer = get_face_analyser()
if analyzer:
    try:
        # Check providers for each model
        if hasattr(analyzer, 'models'):
            for model_name, model in analyzer.models.items():
                if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                    print(f"    Detector '{model_name}': {model.session.get_providers()}")
    except Exception as e:
        print(f"    Error checking providers: {e}")
    print("    [OK] Detector cargado")
else:
    print("    [ERROR] Error cargando detector")

# 4. Face swapper
print("\n[4] Cargando Face swapper (InsightFace)...")
from roop.swapper import get_face_swapper
swapper = get_face_swapper()
if swapper:
    try:
        # The swapper model is an insightface model, check its session
        if hasattr(swapper, 'session') and hasattr(swapper.session, 'get_providers'):
            print(f"    Swapper providers: {swapper.session.get_providers()}")
        else:
            print("    Swapper no expone providers (modelo insightface)")
    except Exception as e:
        print(f"    Error: {e}")
    print("    [OK] Swapper cargado")
else:
    print("    [ERROR] Error cargando swapper")

# 5. Test speed
print("\n[5] Test de velocidad detector (10 frames)...")
import cv2
import numpy as np
test_img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

import time
start = time.time()
for _ in range(10):
    faces = analyzer.get(test_img)
elapsed = time.time() - start
print(f"    Detector: {10/elapsed:.1f} FPS ({elapsed/10*1000:.1f} ms/frame)")

print("\n[6] Test de velocidad swapper (10 swaps)...")
# Crear caras dummy
source_face_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
from roop.analyser import get_face_single
src_face = get_face_single(source_face_img)
tgt_face = get_face_single(test_img)

if src_face and tgt_face:
    start = time.time()
    for _ in range(10):
        result = swapper.get(test_img, tgt_face, src_face, paste_back=True)
    elapsed = time.time() - start
    print(f"    Swapper: {10/elapsed:.1f} FPS ({elapsed/10*1000:.1f} ms/frame)")
else:
    print("    [SKIP] No se pudieron detectar caras para test")

print("\n" + "="*60)
print("RESUMEN")
print("="*60)
print("""
Si tanto detector como swapper usan GPU y están >30 FPS, 
el único bottleneck restante era el enhancer (CodeFormer, 1.2 FPS).

Con enhancer desactivado (use_enhancer=False), se debería alcanzar
tiempo real (~30-60 FPS) en videos 1080p.
""")
