#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Launcher - Maneja el inicio/parada de ComfyUI
"""

import os
import sys
import subprocess
import signal
import time
import requests
import threading
from typing import Tuple, Optional
from roop.utils import get_vram_gb

# Puerto por defecto
DEFAULT_PORTS = [8188, 8189]
COMFYUI_PORT = None  # Se detectará automáticamente

# Detectar ruta de ComfyUI
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFYUI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "tob", "ComfyUI"))

# Usar el venv_flux (GPU) por defecto para ComfyUI
if sys.platform == "win32":
    FLUX_VENV_PYTHON = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "venv_flux", "Scripts", "python.exe"))
    COMFYUI_VENV_PYTHON = os.path.join(COMFYUI_DIR, "venv", "Scripts", "python.exe")
    PROJECT_VENV_PYTHON = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "venv", "Scripts", "python.exe"))
else:
    FLUX_VENV_PYTHON = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "venv_flux", "bin", "python"))
    COMFYUI_VENV_PYTHON = os.path.join(COMFYUI_DIR, "venv", "bin", "python")
    PROJECT_VENV_PYTHON = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "venv", "bin", "python"))

# Determinar cual Python usar - preferir venv de ComfyUI (el mas completo)
if os.path.exists(COMFYUI_VENV_PYTHON):
    COMFYUI_PYTHON = COMFYUI_VENV_PYTHON
    print(f"[ComfyLauncher] Usando venv de ComfyUI (RECOMENDADO): {COMFYUI_PYTHON}")
elif os.path.exists(PROJECT_VENV_PYTHON):
    COMFYUI_PYTHON = PROJECT_VENV_PYTHON
    print(f"[ComfyLauncher] Usando venv del proyecto: {COMFYUI_PYTHON}")
elif os.path.exists(FLUX_VENV_PYTHON):
    COMFYUI_PYTHON = FLUX_VENV_PYTHON
    print(f"[ComfyLauncher] Usando venv_flux: {COMFYUI_PYTHON}")
else:
    COMFYUI_PYTHON = sys.executable
    print(f"[ComfyLauncher] Usando Python del sistema: {COMFYUI_PYTHON}")

# Variable global para rastrear el proceso
comfy_process = None


def is_port_in_use(port: int) -> bool:
    """Check if a port has a LISTENING process - SIMPLIFIED VERSION"""
    import socket
    
    print(f"[DEBUG] Checking port {port}...")
    
    # Direct socket connection test - most reliable method
    # This tests if something is actually listening on the port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('127.0.0.1', port))
            if result == 0:
                print(f"[DEBUG] Puerto {port}: socket conectado exitosamente (127.0.0.1) - retornar True")
                return True
    except Exception as e:
        print(f"[DEBUG] Socket error en puerto {port}: {e}")
    
    # Try localhost as fallback
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            result = s.connect_ex(('localhost', port))
            if result == 0:
                print(f"[DEBUG] Puerto {port}: socket conectado exitosamente (localhost) - retornar True")
                return True
    except Exception as e:
        print(f"[DEBUG] Socket error (localhost) en puerto {port}: {e}")
    
    print(f"[DEBUG] Puerto {port}: no hay proceso escuchando - retornar False")
    return False


def _force_kill_pid(pid, use_tree=True):
    """Force kill a PID and optionally its children tree. Windows-first, psutil preferred."""
    if not pid:
        return False
    pid_str = str(pid)
    killed = False
    try:
        import psutil
        try:
            parent = psutil.Process(int(pid))
            children = parent.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except Exception:
                    pass
            psutil.wait_procs(children, timeout=3)
            try:
                parent.kill()
            except Exception:
                pass
            parent.wait(timeout=3)
            killed = True
        except psutil.NoSuchProcess:
            killed = True  # already gone
        except Exception:
            pass
    except ImportError:
        pass  # psutil not available, fallback below

    if not killed and sys.platform == "win32":
        try:
            flag = "/T /F" if use_tree else "/F"
            res = subprocess.run(
                f"taskkill {flag} /PID {pid_str}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            killed = res.returncode == 0
        except Exception:
            pass

    if sys.platform != "win32" and not killed:
        try:
            os.kill(int(pid), signal.SIGKILL)
            killed = True
        except Exception:
            try:
                subprocess.run(f"kill -9 {pid_str}", shell=True, timeout=5)
                killed = True
            except Exception:
                pass
    return killed


def kill_process_on_port(port: int) -> bool:
    """Kill process running on port - robust tree-kill using psutil or taskkill /T /F"""
    import subprocess
    
    killed = False
    
    print(f"[ComfyLauncher] Verificando procesos en puerto {port}...")
    
    try:
        if sys.platform == "win32":
            result = subprocess.run(
                f"netstat -ano | findstr :{port}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            print(f"[ComfyLauncher] netstat output:\n{result.stdout}")
            
            listening_pids = []
            for line in result.stdout.split('\n'):
                if 'LISTENING' in line:
                    parts = line.split()
                    for part in reversed(parts):
                        if part.strip().isdigit():
                            pid = part.strip()
                            listening_pids.append(pid)
                            break
            
            print(f"[ComfyLauncher] PIDs listening en puerto {port}: {listening_pids}")
            
            if not listening_pids:
                print(f"[ComfyLauncher] No hay proceso LISTENING en puerto {port}")
                if 'SYN_SENT' in result.stdout or 'SYN_RECEIVED' in result.stdout or 'ESTABLISHED' in result.stdout:
                    print(f"[ComfyLauncher] Puerto ocupado por conexión, no por servicio")
                return False
            
            for pid in listening_pids:
                if pid == '0':
                    continue
                print(f"[ComfyLauncher] Matando PID (tree): {pid}")
                if _force_kill_pid(pid, use_tree=True):
                    killed = True
                time.sleep(0.5)
        else:
            result = subprocess.run(
                f"lsof -ti:{port}",
                shell=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip().isdigit():
                        if _force_kill_pid(pid.strip(), use_tree=True):
                            killed = True
    except Exception as e:
        print(f"[ComfyLauncher] Error general en kill_process_on_port: {e}")
    
    if killed:
        time.sleep(1.5)
        if not is_port_in_use(port):
            print(f"[ComfyLauncher] OK Puerto {port} liberado exitosamente")
        else:
            print(f"[ComfyLauncher] ⚠ Puerto {port} aún está ocupado")
            killed = False
    
    return killed


def get_comfy_url() -> str:
    """Obtener la URL de ComfyUI basada en el puerto detectado"""
    global COMFYUI_PORT
    if COMFYUI_PORT is None:
        COMFYUI_PORT = detect_comfyui_port()
        # Si no se detectó puerto (None), usar el primero de defaults
        if COMFYUI_PORT is None:
            COMFYUI_PORT = DEFAULT_PORTS[0]
    return f"http://127.0.0.1:{COMFYUI_PORT}"


def detect_comfyui_port() -> Optional[int]:
    """Detectar si ComfyUI está corriendo en algún puerto - DEBUG VERSION"""
    print(f"[DEBUG] detect_comfyui_port() llamado")
    
    for port in DEFAULT_PORTS:
        print(f"[DEBUG] Verificando puerto {port}...")
        
        # First check if there's a LISTENING process
        if not is_port_in_use(port):
            print(f"[DEBUG] Puerto {port}: no hay LISTENING, saltando")
            continue
        
        # Then verify it's actually ComfyUI
        try:
            print(f"[DEBUG] Haciendo request HTTP a puerto {port}...")
            response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=2)
            print(f"[DEBUG] HTTP response: {response.status_code}")
            if response.status_code == 200:
                print(f"[ComfyLauncher] ✓ ComfyUI detectado en puerto {port}")
                return port
        except Exception as e:
            print(f"[DEBUG] HTTP error en puerto {port}: {e}")
    
    print(f"[DEBUG] No se encontró ComfyUI en puertos {DEFAULT_PORTS}")
    return None


def get_comfyui_executable() -> str:
    """Get the ComfyUI main.py path"""
    main_py = os.path.join(COMFYUI_DIR, "main.py")
    if os.path.exists(main_py):
        return main_py
    return ""


def is_comfyui_running() -> bool:
    """Check if ComfyUI is running"""
    try:
        url = get_comfy_url()
        response = requests.get(f"{url}/system_stats", timeout=2)
        return response.status_code == 200
    except:
        return False


def start(
    port: int = None,
    gpu: int = 0,
    directly_run: bool = False,
    auto_update: bool = False
) -> Tuple[bool, str, int]:
    """Start ComfyUI - VERBOSE"""
    global comfy_process, COMFYUI_PORT
    
    # Primero detectar si ya está corriendo
    detected_port = detect_comfyui_port()
    if detected_port is not None:
        COMFYUI_PORT = detected_port
        print(f"[ComfyLauncher] ComfyUI ya detectado en puerto {COMFYUI_PORT}")
        return True, "ComfyUI ya esta corriendo", COMFYUI_PORT
    
    # Si no está corriendo, usar el puerto especificado o el primero disponible
    if port is None:
        port = int(os.environ.get('COMFYUI_PORT', DEFAULT_PORTS[0]))
    COMFYUI_PORT = port
    
    comfy_url = get_comfy_url()
    
    print(f"\n[ComfyLauncher] ========================================")
    print(f"[ComfyLauncher] INICIANDO ComfyUI en puerto {port}")
    print(f"[ComfyLauncher] ========================================\n")
    
    if is_comfyui_running():
        print(f"[ComfyLauncher] ComfyUI ya esta corriendo en puerto {port}")
        return True, "ComfyUI ya esta corriendo", port
    
    if directly_run:
        os.environ['COMFYUI_PORT'] = str(port)
        print(f"[ComfyLauncher] Puerto establecido: {port}")
        
        exe = get_comfyui_executable()
        if not exe:
            print(f"[ComfyLauncher] ERROR: No se encontro main.py en: {COMFYUI_DIR}")
            return False, "No se encontro main.py de ComfyUI", port
        
        # Usar el Python del venv de ComfyUI
        python_exe = COMFYUI_PYTHON if os.path.exists(COMFYUI_PYTHON) else sys.executable
        
        vram_gb = get_vram_gb()
        if vram_gb <= 8:
            # --normalvram: mantiene el modelo de difusión en GPU, solo offloadea VAE/CLIP.
            # Para Q4_K_S (~5.6GB) cabe sobrado en 8GB, evitando el swap por paso de --lowvram.
            vram_mode = "--normalvram"
        else:
            vram_mode = "--normalvram"
        print(f"[ComfyLauncher] VRAM detectada: {vram_gb}GB → usando {vram_mode}")
        cmd = [python_exe, "-u", exe, "--port", str(port), vram_mode]
        print(f"[ComfyLauncher] Ejecutando: {' '.join(cmd)}")
        print(f"[ComfyLauncher] Directorio: {COMFYUI_DIR}")
        print(f"[ComfyLauncher] Python: {python_exe}")
        print(f"[ComfyLauncher] ---------------------------------------")
        
        if sys.platform == "win32":
            # Windows: capturar stdout y stderr para ver errores
            import threading
            def log_output(stream, prefix):
                for line in iter(stream.readline, ''):
                    if line:
                        print(f"[{prefix}] {line.rstrip()}")
            
            # Agregar variable de entorno para desactivar safety checker y otros ajustes
            env = os.environ.copy()
            env["COMFYUI_DISABLE_SAFETY_CHECKER"] = "true"
            env["SAFETY_CHECKER_DISABLED"] = "true"
            env["DIFFUSERS_SAFETY_CHECKER_DISABLED"] = "true"
            # Desactivar telemetry y tracking
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
            env["DO_NOT_TRACK"] = "1"
            # Aumentar timeout de inicio
            env["COMFYUI_STARTUP_TIMEOUT"] = "300"
            # Fix CUDA allocator - usar backend alternativo
            env["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
            # Evitar offload de modelos entre generaciones
            env["CUDA_MODULE_LOADING"] = "LAZY"
            # Forzar modelos a quedarse en VRAM
            env["COMFYUI_AUTO_CACHE"] = "false"
            
            comfy_process = subprocess.Popen(
                cmd,
                cwd=COMFYUI_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            # Log en hilo separado
            log_thread = threading.Thread(
                target=log_output, 
                args=(comfy_process.stdout, "ComfyUI"),
                daemon=True
            )
            log_thread.start()
            
            print(f"[ComfyLauncher] Proceso creado (PID: {comfy_process.pid})")
        else:
            # Unix/Linux
            # Agregar variable de entorno para desactivar safety checker y otros ajustes
            env = os.environ.copy()
            env["COMFYUI_DISABLE_SAFETY_CHECKER"] = "true"
            env["SAFETY_CHECKER_DISABLED"] = "true"
            env["DIFFUSERS_SAFETY_CHECKER_DISABLED"] = "true"
            # Desactivar telemetry y tracking
            env["HF_HUB_DISABLE_TELEMETRY"] = "1"
            env["DO_NOT_TRACK"] = "1"
            # Aumentar timeout de inicio
            env["COMFYUI_STARTUP_TIMEOUT"] = "300"
            
            comfy_process = subprocess.Popen(
                cmd,
                cwd=COMFYUI_DIR,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=sys.stdin,
                text=True,
                bufsize=1
            )
            
            # Log en hilo separado
            def log_output(stream, prefix):
                for line in iter(stream.readline, ''):
                    if line:
                        print(f"[{prefix}] {line.rstrip()}")
            
            log_thread = threading.Thread(
                target=log_output, 
                args=(comfy_process.stdout, "ComfyUI"),
                daemon=True
            )
            log_thread.start()
            
            print(f"[ComfyLauncher] Proceso Unix creado (PID: {comfy_process.pid})")
        
        # Wait for startup
        print(f"\n[ComfyLauncher] Esperando respuesta de ComfyUI...")
        start_time = time.time()
        timeout = 300  # 5 minutos timeout para ComfyUI
        
        while time.time() - start_time < timeout:
            if is_comfyui_running():
                elapsed = time.time() - start_time
                print(f"\n[ComfyLauncher] OK ComfyUI INICIADO (tiempo: {elapsed:.1f}s)")
                print(f"[ComfyLauncher] URL: {get_comfy_url()}")
                print(f"[ComfyLauncher] ========================================\n")
                return True, "ComfyUI iniciado", port
            
            # Check if process died
            if comfy_process.poll() is not None:
                stdout, _ = comfy_process.communicate(timeout=5)
                error_msg = f"ERROR: El proceso de ComfyUI termino inesperadamente"
                error_msg += f"\nCodigo de salida: {comfy_process.returncode}"
                if stdout:
                    # Show last 50 lines of output
                    lines = stdout.strip().split('\n')
                    if len(lines) > 50:
                        error_msg += f"\n\nUltimas 50 lineas del log:\n" + "\n".join(lines[-50:])
                    else:
                        error_msg += f"\n\nLog completo:\n{stdout}"
                print(f"\n{error_msg}")
                return False, error_msg, port
            
            time.sleep(2)
        
        print(f"\n[ComfyLauncher] ERROR: Timeout esperando ComfyUI")
        return False, "Timeout", port
        
    print(f"[ComfyLauncher] ComfyUI debe iniciarse manualmente")
    return True, "ComfyUI debe iniciarse manualmente", port


def _find_orphan_comfy_pids():
    """Find any python processes that look like they are running ComfyUI (by cmdline)."""
    pids = []
    try:
        if sys.platform == "win32":
            # Get full command line to match ComfyUI specifically
            for exe in ('python.exe', 'pythonw.exe'):
                try:
                    res = subprocess.run(
                        f'wmic process where "name=\'{exe}\'" get ProcessId,CommandLine /format:csv',
                        shell=True, capture_output=True, text=True, timeout=8
                    )
                    for line in (res.stdout or "").splitlines():
                        low = line.lower()
                        if 'comfy' in low or 'ui\\tob\\comfyui' in low.replace('/', '\\') or '\\comfyui\\' in low or 'main.py' in low:
                            # last field is usually PID in csv output from wmic
                            parts = [p.strip().strip('"') for p in line.split(',')]
                            for p in reversed(parts):
                                if p.isdigit() and int(p) > 4:
                                    pids.append(p)
                                    break
                except Exception:
                    continue
        else:
            res = subprocess.run("ps aux | grep -i 'comfy\\|main.py' | grep -v grep", shell=True, capture_output=True, text=True, timeout=5)
            for line in res.stdout.splitlines():
                parts = line.split()
                if len(parts) > 1 and parts[1].isdigit():
                    pids.append(parts[1])
    except Exception:
        pass
    return list(set(pids))


def _sweep_orphans():
    """Last resort: kill known orphan ComfyUI python processes."""
    orphans = _find_orphan_comfy_pids()
    if not orphans:
        return 0
    count = 0
    for pid in orphans:
        # Conservative: only kill if we also have reason to believe it's Comfy (port check or we just do force since stop is only on exit)
        # To be safer we always try on exit sweep because only our launcher starts it this way.
        if _force_kill_pid(pid, use_tree=True):
            count += 1
            print(f"[ComfyLauncher] Orphan sweep killed PID {pid}")
    return count


def stop() -> Tuple[bool, str]:
    """Stop ComfyUI - robust tree-kill + orphan sweep. Always cleans tracked + port + orphans."""
    global comfy_process, COMFYUI_PORT
    
    print(f"\n[DEBUG] stop() llamado - COMFYUI_PORT={COMFYUI_PORT}")
    
    actions = []
    killed_any = False
    
    # Step 1: Kill the tracked Popen (and its whole tree)
    tracked_pid = None
    if comfy_process:
        try:
            tracked_pid = comfy_process.pid
            print(f"[DEBUG] Hay proceso tracked: {tracked_pid}")
            print(f"[ComfyLauncher] Terminando proceso tracked PID (tree): {tracked_pid}")
            
            # Graceful attempt + close pipes (prevents pipe read blocks)
            try:
                comfy_process.terminate()
            except Exception:
                pass
            try:
                comfy_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
            
            # Close stdout pipe used by log thread to release resources
            try:
                if getattr(comfy_process, 'stdout', None):
                    comfy_process.stdout.close()
                if getattr(comfy_process, 'stderr', None):
                    comfy_process.stderr.close()
            except Exception:
                pass
            
            # Force tree kill (psutil or taskkill /T /F)
            if _force_kill_pid(tracked_pid, use_tree=True):
                actions.append(f"tracked {tracked_pid} tree-killed")
                killed_any = True
            
            try:
                comfy_process.kill()
            except Exception:
                pass
        except Exception as e:
            actions.append(f"Error tracked: {e}")
        finally:
            comfy_process = None
    
    # Step 2: Determine port to clean
    current_port = COMFYUI_PORT
    if not current_port:
        print(f"[DEBUG] No hay puerto conocido, escaneando...")
        for p in DEFAULT_PORTS:
            if is_port_in_use(p):
                current_port = p
                print(f"[DEBUG] Encontrado LISTENING en puerto {p}")
                break
    
    # Step 3: Kill whatever is listening (tree kill)
    if current_port:
        print(f"[DEBUG] Puerto a liberar: {current_port}")
        if is_port_in_use(current_port):
            print(f"[ComfyLauncher] Puerto {current_port} ocupado, intentando liberar (tree)...")
            if kill_process_on_port(current_port):
                killed_any = True
                actions.append(f"port {current_port} freed")
            time.sleep(1)
            if is_port_in_use(current_port):
                print(f"[ComfyLauncher] ⚠ Puerto {current_port} aún ocupado después de kill")
            else:
                print(f"[ComfyLauncher] ✓ Puerto {current_port} liberado")
        else:
            print(f"[DEBUG] Puerto {current_port} ya no tiene LISTENING")
        COMFYUI_PORT = None
    else:
        print(f"[DEBUG] No se encontró puerto LISTENING")
    
    # Step 4: Final orphan sweep (catches cases where port var was None or multiple)
    try:
        swept = _sweep_orphans()
        if swept > 0:
            killed_any = True
            actions.append(f"swept {swept} orphan(s)")
    except Exception as e:
        print(f"[ComfyLauncher] Sweep error: {e}")
    
    COMFYUI_PORT = None
    comfy_process = None
    
    if actions or killed_any:
        msg = f"ComfyUI detenido: {'; '.join(actions) if actions else 'tree-killed'}"
        print(f"[ComfyLauncher] Acciones: {actions}")
        return True, msg
    
    print(f"[DEBUG] No había nada que detener (o ya estaba limpio)")
    return False, "No se encontró proceso en puertos ComfyUI (o ya limpio)"


def restart() -> Tuple[bool, str]:
    """Restart ComfyUI - VERBOSE"""
    print(f"\n[ComfyLauncher] ========================================")
    print(f"[ComfyLauncher] REINICIANDO ComfyUI")
    print(f"[ComfyLauncher] ========================================\n")
    
    stop()
    time.sleep(3)
    return start()
