#!/usr/bin/env python
"""Diagnóstico de velocidad de FaceSwap - identifica el bottleneck"""

import sys
import os
import time
import cv2
import numpy as np

# Añadir directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("DIAGNÓSTICO DE VELOCIDAD FACESWAP")
print("=" * 60)

# 1. Verificar CUDA
print("\n[1] Verificando CUDA...")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"  Torch CUDA: {'[OK] Disponible' if cuda_available else '[ERROR] No disponible'}")
    if cuda_available:
        print(f"  Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"  Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
except Exception as e:
    print(f"  ✗ Error: {e}")
    cuda_available = False

# 2. Verificar ONNX Runtime providers
print("\n[2] Verificando ONNX Runtime...")
try:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    print(f"  Providers disponibles: {available_providers}")
    print(f"  CUDA soportado: {'[OK] Sí' if 'CUDAExecutionProvider' in available_providers else '[ERROR] No'}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 3. Verificar configuración de roop.globals
print("\n[3] Configuración actual de roop.globals:")
try:
    import roop.globals as globals
    print(f"  execution_providers: {globals.execution_providers}")
    print(f"  use_enhancer: {getattr(globals, 'use_enhancer', 'No definido')}")
    print(f"  selected_enhancer: {getattr(globals, 'selected_enhancer', 'No definido')}")
    print(f"  enhancer_blend_factor: {getattr(globals, 'enhancer_blend_factor', 'No definido')}")
    print(f"  batch_processing_size: {getattr(globals, 'batch_processing_size', 'No definido')}")
except Exception as e:
    print(f"  ✗ Error: {e}")

# 4. Verificar det_size de InsightFace
print("\n[4] Verificando detector InsightFace...")
try:
    from roop.face_util import get_face_analyser, DETECTION_SIZE
    print(f"  DETECTION_SIZE en face_util.py: {DETECTION_SIZE}")
    
    analyzer = get_face_analyser()
    if analyzer is not None:
        # InsightFace guarda el det_size en el modelo
        print(f"  FaceAnalysis inicializado: [OK]")
        try:
            if hasattr(analyzer, 'models') and 'detection' in analyzer.models:
                providers = analyzer.models['detection'].session.get_providers()
                print(f"  Providers detección: {providers}")
            else:
                print(f"  Providers: No accessible")
        except:
            print(f"  Providers: N/A")
    else:
        print(f"  [ERROR] FaceAnalysis no inicializado")
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 5. Crear imagen de prueba y medir velocidad de detección
print("\n[5] Test de velocidad de detección (100 frames)...")
try:
    from roop.face_util import get_face_analyser
    
    analyzer = get_face_analyser()
    if analyzer is None:
        print("  [ERROR] No se puede probar: analizador no disponible")
    else:
        # Crear imagen de prueba 1920x1080
        test_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        
        # Warmup
        for _ in range(3):
            faces = analyzer.get(test_img)
        
        # Medir tiempo
        start = time.time()
        n_iter = 50
        for _ in range(n_iter):
            faces = analyzer.get(test_img)
        elapsed = time.time() - start
        
        fps = n_iter / elapsed
        print(f"  Detección: {fps:.2f} FPS ({elapsed/n_iter*1000:.1f} ms/frame)")
        
        if fps < 10:
            print(f"  [WARN] ¡LENTO! Debería ser >30 FPS con GPU")
        elif fps < 30:
            print(f"  [WARN] Acceptable pero mejorable")
        else:
            print(f"  [OK] Buena velocidad")
            
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 6. Verificar CodeFormer en GPU
print("\n[6] Verificando CodeFormer...")
try:
    from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
    
    enhancer = Enhance_CodeFormer()
    enhancer.Initialize({"devicename": "cuda" if cuda_available else "cpu"})
    
    if enhancer.model_codeformer is not None:
        providers = enhancer.model_codeformer.get_providers()
        print(f"  CodeFormer providers: {providers}")
        print(f"  Usando CUDA: {'[OK] Sí' if 'CUDAExecutionProvider' in providers else '[ERROR] No'}")
        
        # Test de velocidad
        print("\n  Test de velocidad CodeFormer (10 iteraciones)...")
        test_face = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8).astype(np.float32) / 255.0
        test_face = (test_face - 0.5) / 0.5
        test_face = np.expand_dims(test_face, axis=0).transpose(0, 3, 1, 2)
        
        # Warmup
        class MockFaceSet:
            faces = [None]
            ref_images = [None]
        
        try:
            for _ in range(2):
                result = enhancer.Run(MockFaceSet(), None, test_face[0].transpose(1,2,0))
        except Exception as e:
            print(f"  [WARN] Error en warmup (esperado): {e}")
        
        # Medir
        start = time.time()
        n_iter = 10
        for _ in range(n_iter):
            try:
                result = enhancer.Run(MockFaceSet(), None, test_face[0].transpose(1,2,0))
            except:
                pass
        elapsed = time.time() - start
        
        fps = n_iter / elapsed
        print(f"  CodeFormer: {fps:.2f} FPS ({elapsed/n_iter*1000:.1f} ms/frame)")
        
        if fps < 5:
            print(f"  [CRITICAL] ¡MUY LENTO! Principal bottleneck")
        elif fps < 15:
            print(f"  [WARN] Lento para video realtime")
        else:
            print(f"  [OK] Aceptable")
    else:
        print("  ✗ Modelo no cargado")
        
except Exception as e:
    print(f"  ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# 7. Resumen y recomendaciones
print("\n" + "=" * 60)
print("RESUMEN Y RECOMENDACIONES")
print("=" * 60)

print("""
Si la detección va lenta (>10ms/frame):
  → Asegurar que DETECTION_SIZE = (320, 320) en face_util.py
  → Verificar que InsightFace use CUDA (providers=['CUDAExecutionProvider'])

Si CodeFormer es lento (<5 FPS):
  → Desactivar CodeFormer (use_enhancer=False) o usar GFPGAN
  → Cambiar default_enhancer a 'GFPGAN' en globals.py
  → O implementar procesamiento en批 paralelo

Para procesamiento en paralelo real:
  → Modificar process_frame para ser stateless (sin temporal smoothing)
  → Usar ThreadPoolExecutor para procesar múltiples frames simultáneamente
  → Requiere ajustar tracking de caras sin estado
""")

print("\nDiagnóstico completo.")
