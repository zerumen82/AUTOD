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

# Determinar cual Python usar - preferir venv_flux (GPU)
if os.path.exists(FLUX_VENV_PYTHON):
    COMFYUI_PYTHON = FLUX_VENV_PYTHON
    print(f"[ComfyLauncher] Usando venv_flux (GPU): {COMFYUI_PYTHON}")
elif os.path.exists(COMFYUI_VENV_PYTHON):
    COMFYUI_PYTHON = COMFYUI_VENV_PYTHON
    print(f"[ComfyLauncher] Usando venv de ComfyUI: {COMFYUI_PYTHON}")
elif os.path.exists(PROJECT_VENV_PYTHON):
    COMFYUI_PYTHON = PROJECT_VENV_PYTHON
    print(f"[ComfyLauncher] Usando venv del proyecto: {COMFYUI_PYTHON}")
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


def kill_process_on_port(port: int) -> bool:
    """Kill process running on port - Windows compatible with taskkill"""
    import socket
    import subprocess
    import os
    import signal
    
    killed = False
    
    # First, check if there's actually a LISTENING process on this port
    print(f"[ComfyLauncher] Verificando procesos en puerto {port}...")
    
    try:
        if sys.platform == "win32":
            # Get ALL processes listening on the port
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
                    # PID is the last column in netstat -ano output
                    for part in reversed(parts):
                        if part.strip().isdigit():
                            pid = part.strip()
                            listening_pids.append(pid)
                            break
            
            print(f"[ComfyLauncher] PIDs listening en puerto {port}: {listening_pids}")
            
            if not listening_pids:
                print(f"[ComfyLauncher] No hay proceso LISTENING en puerto {port}")
                # Check if it's just a connection issue
                if 'SYN_SENT' in result.stdout or 'SYN_RECEIVED' in result.stdout or 'ESTABLISHED' in result.stdout:
                    print(f"[ComfyLauncher] Puerto ocupado por conexión, no por servicio")
                    # Try to close TIME_WAIT or established connections
                    try:
                        subprocess.run(
                            f"for /f \"tokens=5\" %a in ('netstat -ano ^| findstr :{port} ^| findstr ESTABLISHED') do @for /f \"tokens=2\" %b in ('echo %a') do @for /f \"tokens=5\" %c in ('echo %b') do @taskkill /F /PID %c",
                            shell=True,
                            timeout=10
                        )
                    except:
                        pass
                return False
            
            # Kill all listening processes
            for pid in listening_pids:
                if pid == '0':  # Skip system processes
                    continue
                print(f"[ComfyLauncher] Matando PID: {pid}")
                try:
                    # First try graceful termination
                    kill_result = subprocess.run(
                        f"taskkill /PID {pid}",
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if kill_result.returncode != 0:
                        # Force kill
                        kill_result = subprocess.run(
                            f"taskkill /F /PID {pid}",
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=10
                        )
                    print(f"[ComfyLauncher] taskkill result: {kill_result.returncode}")
                    if kill_result.returncode == 0:
                        killed = True
                except Exception as e:
                    print(f"[ComfyLauncher] Error matando {pid}: {e}")
                    
                # Give it time to terminate
                time.sleep(1)
        else:
            # Unix/Linux method
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
                        subprocess.run(f"kill -9 {pid}", shell=True, timeout=5)
                        killed = True
    except Exception as e:
        print(f"[ComfyLauncher] Error general: {e}")
    
    # Verify port is free
    if killed:
        time.sleep(2)
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
        
        # Usar --normalvram para mejor velocidad. GGUF ya optimiza la memoria.
        # --disable-smart-memory previene descarga automática de modelos
        cmd = [python_exe, "-u", exe, "--port", str(port), "--normalvram", "--disable-smart-memory"]
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


def stop() -> Tuple[bool, str]:
    """Stop ComfyUI - Force kill with taskkill - DEBUG VERSION"""
    global comfy_process, COMFYUI_PORT
    
    print(f"\n[DEBUG] stop() llamado - COMFYUI_PORT={COMFYUI_PORT}")
    
    # Track what we do
    actions = []
    
    # Step 1: Kill tracked process if exists
    if comfy_process:
        print(f"[DEBUG] Hay proceso tracked: {comfy_process.pid}")
        try:
            pid = comfy_process.pid
            print(f"[ComfyLauncher] Terminando proceso tracked PID: {pid}")
            comfy_process.terminate()
            try:
                comfy_process.wait(timeout=5)
                actions.append(f"Proceso {pid} terminado gracefully")
            except subprocess.TimeoutExpired:
                comfy_process.kill()
                actions.append(f"Proceso {pid} matado (kill)")
            comfy_process = None
        except Exception as e:
            actions.append(f"Error con proceso tracked: {e}")
    
    # Step 2: Check if we have a known port
    current_port = COMFYUI_PORT
    
    # Step 3: If no known port, scan default ports for LISTENING processes
    if not current_port:
        print(f"[DEBUG] No hay puerto conocido, escaneando...")
        for p in DEFAULT_PORTS:
            if is_port_in_use(p):
                current_port = p
                print(f"[DEBUG] Encontrado LISTENING en puerto {p}")
                break
    
    # Step 4: Only try to kill if we found a LISTENING process
    if current_port:
        print(f"[DEBUG] Puerto a liberar: {current_port}")
        
        # Double check it's still listening
        if not is_port_in_use(current_port):
            print(f"[DEBUG] Puerto {current_port} ya no tiene LISTENING - no hay nada que matar")
            COMFYUI_PORT = None
            if actions:
                print(f"[ComfyLauncher] Acciones: {', '.join(actions)}")
                return True, f"ComfyUI detenido: {', '.join(actions[:2])}"
            return False, "No hay proceso LISTENING en puertos ComfyUI"
        
        print(f"[ComfyLauncher] Puerto {current_port} ocupado, intentando liberar...")
        killed = kill_process_on_port(current_port)
        
        if not killed:
            print(f"[DEBUG] kill_process_on_port retornó False")
        
        if killed:
            time.sleep(2)
            # Verify
            if is_port_in_use(current_port):
                print(f"[ComfyLauncher] ⚠ Puerto {current_port} aún ocupado después de kill")
                COMFYUI_PORT = None
            else:
                print(f"[ComfyLauncher] ✓ Puerto {current_port} liberado")
                COMFYUI_PORT = None
                actions.append("Puerto liberado")
    else:
        print(f"[DEBUG] No se encontró ningún puerto LISTENING")
    
    if actions:
        print(f"[ComfyLauncher] Acciones: {', '.join(actions)}")
        return True, f"ComfyUI detenido: {', '.join(actions[:2])}"
    
    print(f"[DEBUG] No había nada que detener")
    COMFYUI_PORT = None
    return False, "No se encontró proceso en puertos ComfyUI"


def restart() -> Tuple[bool, str]:
    """Restart ComfyUI - VERBOSE"""
    print(f"\n[ComfyLauncher] ========================================")
    print(f"[ComfyLauncher] REINICIANDO ComfyUI")
    print(f"[ComfyLauncher] ========================================\n")
    
    stop()
    time.sleep(3)
    return start()
