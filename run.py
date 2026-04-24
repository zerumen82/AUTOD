import sys
import os
import shutil
import time
import gc
import ctypes
import traceback
import stat
import tempfile
import glob
import threading

# Añadir CUDA 12.4 bin al PATH para que onnxruntime encuentre cublasLt64_12.dll
_cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
if os.path.exists(_cuda_bin):
    os.environ["PATH"] = _cuda_bin + ";" + os.environ.get("PATH", "")
    print(f"[INFO] CUDA 12.4 bin añadido al PATH: {_cuda_bin}")
else:
    # Intentar encontrar la versión más alta de CUDA disponible
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.exists(cuda_base):
        versions = [d for d in os.listdir(cuda_base) if os.path.isdir(os.path.join(cuda_base, d)) and d.startswith('v')]
        versions.sort(key=lambda x: [int(v) for v in x[1:].split('.')], reverse=True)
        if versions:
            latest = os.path.join(cuda_base, versions[0], "bin")
            if os.path.exists(latest):
                os.environ["PATH"] = latest + ";" + os.environ.get("PATH", "")
                print(f"[INFO] CUDA {versions[0]} bin añadido al PATH: {latest}")


def cleanup_old_temps_async():
    """Limpia los temporales antiguos de C: y del proyecto en un hilo separado."""
    cleaned_count = 0
    
    def _safe_remove_old_items(base_dir, min_age_seconds=6 * 3600):
        """Elimina solo elementos antiguos para evitar carreras con archivos en uso."""
        local_count = 0
        if not os.path.exists(base_dir):
            return local_count
        now = time.time()
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            try:
                age = now - os.path.getmtime(item_path)
            except Exception:
                continue
            # Evitar borrar artefactos recientes que la app podría estar sirviendo.
            if age < min_age_seconds:
                continue
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path, ignore_errors=True)
                else:
                    os.remove(item_path)
                local_count += 1
            except Exception:
                pass  # Ignorar errores de archivos en uso
        return local_count
    
    # 1. Limpiar temporales de Gradio en C:\Users\...\AppData\Local\Temp\gradio
    try:
        default_temp = tempfile._get_default_tempdir() if hasattr(tempfile, '_get_default_tempdir') else None
        if default_temp and os.path.exists(default_temp):
            gradio_temp = os.path.join(default_temp, "gradio")
            active_temp = os.path.abspath(tempfile.gettempdir())
            # Si Gradio usa este mismo directorio activo, no tocarlo en runtime.
            if os.path.abspath(gradio_temp) == active_temp:
                print(f"[CLEANUP] Omitiendo temp activo de Gradio: {gradio_temp}")
            elif os.path.exists(gradio_temp):
                print(f"[CLEANUP] Limpiando temporales antiguos de Gradio: {gradio_temp}")
                cleaned_count += _safe_remove_old_items(gradio_temp)
    except Exception as e:
        print(f"[CLEANUP] Error limpiando temporales de Gradio: {e}")
    
    # 2. Limpiar temporales de roop en C:\Users\...\AppData\Local\Temp\rooptmp
    try:
        default_temp = tempfile._get_default_tempdir() if hasattr(tempfile, '_get_default_tempdir') else None
        if default_temp and os.path.exists(default_temp):
            roop_temp = os.path.join(default_temp, "rooptmp")
            if os.path.exists(roop_temp):
                print(f"[CLEANUP] Limpiando temporales de roop: {roop_temp}")
                shutil.rmtree(roop_temp, ignore_errors=True)
                cleaned_count += 1
    except Exception as e:
        print(f"[CLEANUP] Error limpiando temporales de roop: {e}")
    
    # 3. Limpiar temporales del proyecto (D:\.autodeep_temp) si existe
    try:
        project_temp_path = "D:\\.autodeep_temp"
        active_temp = os.path.abspath(tempfile.gettempdir())
        if os.path.exists(project_temp_path):
            # No limpiar la carpeta temporal activa de esta misma sesión.
            if os.path.abspath(project_temp_path) == active_temp:
                print(f"[CLEANUP] Omitiendo temp activo del proyecto: {project_temp_path}")
            else:
                print(f"[CLEANUP] Limpiando temporales antiguos del proyecto: {project_temp_path}")
                cleaned_count += _safe_remove_old_items(project_temp_path)
    except Exception as e:
        print(f"[CLEANUP] Error limpiando temporales del proyecto: {e}")
    
    if cleaned_count > 0:
        print(f"[CLEANUP] OK Limpiados {cleaned_count} elementos temporales")
    else:
        print("[CLEANUP] No se encontraron temporales para limpiar")


# Ejecutar limpieza en un hilo separado para no bloquear el arranque
cleanup_thread = threading.Thread(target=cleanup_old_temps_async, daemon=True)
cleanup_thread.start()


# --- CONFIGURACION DE CARPETAS TEMPORALES (D: DRIVE Opcional / Forzado) ---
project_temp = "D:\\.autodeep_temp"
try:
    if not os.path.exists("D:\\"):
        project_temp = os.path.abspath(os.path.join(os.getcwd(), "temp_local"))
    os.makedirs(project_temp, exist_ok=True)
    
    # 1. Inyectar en variables de entorno
    os.environ["TEMP"] = project_temp
    os.environ["TMP"] = project_temp
    os.environ["GRADIO_TEMP_DIR"] = project_temp
    
    # 2. Forzar en el modulo tempfile de Python (Gradio lo usa)
    tempfile.tempdir = project_temp
    # Monkey patch para que cualquier llamada a gettempdir devuelva nuestra ruta
    tempfile.gettempdir = lambda: project_temp
    
    print(f" (TEMP) Forzado a: {project_temp}")
except Exception as e:
    print(f" (ERROR) Error configurando carpetas temporales: {e}")

# --- PARCHES GLOBALES CONTRA [WinError 32] ---
original_move = shutil.move
original_remove = os.remove

def resilient_move(src, dst, copy_function=shutil.copy2):
    """Parche ultra-insistente para mover archivos en Windows"""
    import gc
    for i in range(50): # 10 segundos de paciencia
        try:
            if i % 5 == 0: gc.collect()
            
            if os.path.exists(src):
                try: 
                    os.chmod(src, stat.S_IWRITE)
                    os.chmod(src, 0o777)
                except: pass
                
                # Intentar reemplazo rapido si estan en la misma unidad
                if os.path.exists(dst): 
                    try: 
                        os.chmod(dst, stat.S_IWRITE)
                        original_remove(dst)
                    except: pass
                
                try:
                    os.rename(src, dst)
                    return dst
                except: pass
            
            return original_move(src, dst, copy_function)
            
        except (PermissionError, OSError):
            # Si el archivo ya llego al destino, ignoramos el fallo de borrado del origen
            if os.path.exists(dst) and os.path.getsize(dst) > 0:
                return dst
                
            if i == 0:
                print(f"\n (BLOQUEO) Recuperando archivo: {os.path.basename(src)}...")
            if i == 49: raise
            time.sleep(0.2)
    return original_move(src, dst, copy_function)

def resilient_remove(path, *args, **kwargs):
    """Parche para os.remove - solo acepta un argumento posicional."""
    for i in range(20):
        try:
            try: os.chmod(path, stat.S_IWRITE)
            except: pass
            # os.remove solo acepta el path, ignoramos args/kwargs extra
            return original_remove(path)
        except (PermissionError, OSError):
            if i == 19: raise
            time.sleep(0.1)

def resilient_unlink(self, missing_ok=False):
    """Parche para Path.unlink - usa self como path."""
    for i in range(20):
        try:
            try: os.chmod(str(self), stat.S_IWRITE)
            except: pass
            # Llamar al método original de Path
            return Path._original_unlink(self, missing_ok=missing_ok)
        except (PermissionError, OSError):
            if i == 19: raise
            time.sleep(0.1)

# Aplicar a los modulos cargados
shutil.move = resilient_move
os.remove = resilient_remove
if hasattr(os, 'unlink'): os.unlink = resilient_remove

# Parchear Path.unlink específicamente
from pathlib import Path
Path._original_unlink = Path.unlink
Path.unlink = resilient_unlink

def setup_runtime():
    print("\n[STEP 1] Preparando entorno NVIDIA/CUDA...")
    
    # Configurar PyTorch para evitar fragmentación de memoria - solo para el proceso principal
    # NO establecer PYTORCH_CUDA_ALLOC_CONF aqui porque ComfyUI usa su propio entorno
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    
    base_dir = os.getcwd()
    # Forzar uso de venv que es donde está torch instalado
    scripts_dir = os.path.join(base_dir, "venv", "Scripts")
    torch_lib = os.path.join(base_dir, "venv", "Lib", "site-packages", "torch", "lib")
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    
    # Asegurar DLLs
    target_zlib = os.path.join(scripts_dir, "zlibwapi.dll")
    if not os.path.exists(target_zlib):
        src_zlib = os.path.join(torch_lib, "zlibwapi.dll")
        if os.path.exists(src_zlib):
            try: shutil.copy2(src_zlib, target_zlib)
            except: pass

    # Rutas para cargador de DLLs
    if hasattr(os, 'add_dll_directory'):
        for p in [cuda_bin, scripts_dir, torch_lib]:
            if os.path.exists(p):
                try: os.add_dll_directory(p)
                except: pass
    
    os.environ["PATH"] = cuda_bin + os.pathsep + os.environ["PATH"]

    # Precarga de cuDNN - Usar la versión que PyTorch instala
    if os.path.exists(torch_lib):
        os.add_dll_directory(torch_lib)
        os.environ["PATH"] = torch_lib + os.pathsep + os.environ["PATH"]
        dlls = ["cudnn64_9.dll", "cudnn_adv64_9.dll", "cudnn_cnn64_9.dll", "cudnn_ops64_9.dll"]
        for dll in dlls:
            dll_path = os.path.join(torch_lib, dll)
            if os.path.exists(dll_path):
                try:
                    ctypes.WinDLL(dll_path)
                    print(f"CuDNN DLL cargada (PyTorch): {dll}")
                except Exception as e:
                    print(f"Error cargando {dll}: {e}")

if __name__ == "__main__":
    setup_runtime()
    
    # Preparar rutas de Python
    root_dir = os.getcwd()
    if root_dir not in sys.path: sys.path.insert(0, root_dir)
    
    print("\n[STEP 2] Cargando motores de Inteligencia Artificial...")
    try:
        import torch
        print(f"  - PyTorch version: {torch.__version__}")
        
        cuda_ok = torch.cuda.is_available()
        print(f"  - PyTorch CUDA: {'OK' if cuda_ok else 'FALLO'}")
        
        if cuda_ok:
            print(f"  - GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)")
        
        # Verificar ONNX Runtime - FORZAR actualización de providers
        print("  - Verificando ONNX Runtime...")
        import importlib
        import onnxruntime
        importlib.reload(onnxruntime)
        
        providers = onnxruntime.get_available_providers()
        print(f"    Providers disponibles: {providers}")
        
        has_cuda = 'CUDAExecutionProvider' in providers
        has_tensorrt = 'TensorrtExecutionProvider' in providers
        
        if cuda_ok and not has_cuda:
            print("    [WARNING] PyTorch tiene CUDA pero ONNX Runtime no!")
            print("    [WARNING] Posible conflicto de versiones de onnxruntime")
            print("    [INFO] Intentando forzar providers manualmente...")
            try:
                onnxruntime.set_default_session_options(onnxruntime.SessionOptions())
            except:
                pass
        
        provider_status = 'TensorRT+CUDA+CPU' if has_tensorrt else ('CUDA+CPU' if has_cuda else 'CPU only')
        print(f"  - ONNX Runtime: {provider_status}")
        
        print("\n[STEP 3] Lanzando interfaz Gradio...")
        # Importamos roop.core al final para que vea todos los parches anteriores
        import roop.core
        roop.core.run()
        
    except Exception:
        print("\n" + "!"*60)
        print(" [ERROR CRITICO EN EL MOTOR]")
        print("!"*60)
        traceback.print_exc()

    print("\n" + "="*70)
    print(" MANTENIENDO CONSOLA POR SEGURIDAD")
    print("="*70)
    try:
        while True: time.sleep(10)
    except KeyboardInterrupt: pass
