#!/usr/bin/env python3
"""
Script de diagnostico para verificar la configuracion de CUDA
Ejecutar: python tools/check_cuda.py
"""

import sys
import os
import subprocess

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_cuda():
    print_section("VERIFICACION DE CUDA PARA STABLE DIFFUSION")
    
    # 1. Verificar si torch esta instalado
    print("\n1. Verificando PyTorch...")
    try:
        import torch
        print(f"   [OK] PyTorch instalado: {torch.__version__}")
    except ImportError:
        print("   [ERROR] PyTorch NO esta instalado")
        return False
    
    # 2. Verificar CUDA disponible
    print("\n2. Verificando disponibilidad de CUDA...")
    if torch.cuda.is_available():
        print(f"   [OK] CUDA esta disponible")
        print(f"   - Version CUDA: {torch.version.cuda}")
        print(f"   - Dispositivo: {torch.cuda.get_device_name(0)}")
        print(f"   - Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   - Capability: {torch.cuda.get_device_capability(0)}")
    else:
        print("   [ERROR] CUDA NO esta disponible")
        print("   - PyTorch esta ejecutandose en modo CPU")
        print("   - Esto causara generacion muy lenta (341s/it en lugar de 2-5s/it)")
    
    # 3. Verificar variables de entorno
    print_section("VARIABLES DE ENTORNO CUDA")
    cuda_vars = ['CUDA_PATH', 'CUDA_HOME', 'PATH', 'LD_LIBRARY_PATH']
    for var in cuda_vars:
        value = os.environ.get(var, '')
        if 'cuda' in value.lower():
            print(f"   {var}: {value[:80]}...")
    
    # 4. Verificar ubicacion de CUDA toolkit
    print_section("UBICACIONES COMUNES DE CUDA")
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA',
        r'C:\Program Files\NVIDIA Corporation',
        '/usr/local/cuda',
        '/opt/cuda',
    ]
    for path in cuda_paths:
        if os.path.exists(path):
            print(f"   [OK] Existe: {path}")
            # Listar versiones
            try:
                entries = os.listdir(path)
                for entry in entries:
                    full_path = os.path.join(path, entry)
                    if os.path.isdir(full_path):
                        print(f"      - {entry}")
            except:
                pass
        else:
            print(f"   [NO] No existe: {path}")
    
    # 5. Verificar librerias cudnn
    print_section("VERIFICACION DE CUDNN")
    try:
        # Verificar si torch tiene cudnn
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'cudnn'):
            print(f"   [OK] cuDNN disponible: {torch.backends.cudnn.version()}")
            print(f"   - cuDNN habilitado: {torch.backends.cudnn.enabled}")
        else:
            print("   [!] No se pudo verificar cuDNN")
    except Exception as e:
        print(f"   [!] Error verificando cuDNN: {e}")
    
    # 6. Prueba de rendimiento simple
    print_section("PRUEBA DE RENDIMIENTO")
    if torch.cuda.is_available():
        try:
            print("   Ejecutando prueba de matriz en GPU...")
            import time
            
            # Crear tensores grandes
            a = torch.randn(1000, 1000).cuda()
            b = torch.randn(1000, 1000).cuda()
            
            # Sincronizar
            torch.cuda.synchronize()
            
            # Medir tiempo
            start = time.time()
            for _ in range(10):
                c = torch.matmul(a, b)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            print(f"   [OK] Tiempo en GPU: {gpu_time:.4f}s para 10 multiplicaciones 1000x1000")
            print(f"   - Esto indica que CUDA esta funcionando correctamente")
            
        except Exception as e:
            print(f"   [ERROR] Error en prueba GPU: {e}")
    else:
        print("   [!] Saltando prueba de rendimiento (CUDA no disponible)")
    
    # Resumen
    print_section("RESUMEN")
    if torch.cuda.is_available():
        print("   [OK] CONFIGURACION CORRECTA - CUDA esta funcionando")
        print("   - La generacion de imagenes deberia ser rapida (2-5s por iteracion)")
        return True
    else:
        print("   [ERROR] PROBLEMA DETECTADO - CUDA no esta disponible")
        print("   - La generacion sera muy lenta (341s por iteracion)")
        print("\n   POSIBLES SOLUCIONES:")
        print("   1. Reinstalar PyTorch con soporte CUDA:")
        print("      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("   2. Verificar que los drivers de NVIDIA esten instalados")
        print("   3. Asegurar que CUDA toolkit este en el PATH")
        return False

if __name__ == "__main__":
    success = check_cuda()
    sys.exit(0 if success else 1)
