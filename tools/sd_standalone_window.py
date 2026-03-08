"""
Script independiente para abrir Stable Diffusion en pywebview
=============================================================

Este script abre SD WebUI en una ventana pywebview separada,
independientemente de AUTO-DEEP.

USO:
    python tools/sd_standalone_window.py

Puedes ejecutar este script mientras AUTO-DEEP está corriendo.
"""

import os
import sys
import time
import socket
import threading

# Configurar encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

SD_PORT = 9871
SD_URL = f"http://127.0.0.1:{SD_PORT}"

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def wait_for_server(port, timeout=180):
    """Espera hasta que el servidor esté listo"""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_in_use(port):
            return True
        time.sleep(1)
    return False

def start_sd_if_needed():
    """Inicia SD WebUI si no está corriendo"""
    if is_port_in_use(SD_PORT):
        print(f"[INFO] SD WebUI ya está corriendo en puerto {SD_PORT}")
        return True
    
    print(f"[INFO] SD WebUI no está corriendo. Iniciando...")
    
    args = [
        '--port', str(SD_PORT),
        '--api',
        '--cors-allow-origins', '*',
        '--skip-torch-cuda-test',
        '--opt-sdp-attention',
        '--opt-channelslast',
        '--disable-safe-unpickle',
        '--no-half-vae',
        '--medvram'
    ]
    
    sd_path = "ui/tob/stable-diffusion-webui"
    venv_python = os.path.join(sd_path, "venv", "Scripts", "python.exe")
    launch_py = os.path.join(sd_path, "launch.py")
    
    if not os.path.exists(venv_python):
        print(f"[ERROR] No se encontró Python del venv: {venv_python}")
        return False
    
    if not os.path.exists(launch_py):
        print(f"[ERROR] No se encontró launch.py: {launch_py}")
        return False
    
    cmd = [venv_python, launch_py] + args
    
    try:
        proc = subprocess.Popen(cmd, cwd=sd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"[INFO] SD WebUI iniciado con PID: {proc.pid}")
        
        # Esperar a que esté listo
        if wait_for_server(SD_PORT, timeout=180):
            print(f"[OK] SD WebUI listo en: {SD_URL}")
            return True
        else:
            print("[ERROR] Timeout esperando SD WebUI")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error iniciando SD WebUI: {e}")
        return False

def open_sd_window():
    """Abre SD WebUI en una ventana pywebview"""
    try:
        import webview
        print(f"[INFO] Abriendo ventana pywebview con SD WebUI: {SD_URL}")
        
        window = webview.create_window(
            'Stable Diffusion WebUI',
            SD_URL,
            width=1400,
            height=900,
            resizable=True,
            background_color='#1a1a1a'
        )
        
        # Iniciar webview
        webview.start(debug=False)
        print("[INFO] Ventana SD cerrada")
        
    except Exception as e:
        print(f"[ERROR] Error con pywebview: {e}")
        print(f"[INFO] Puedes abrir SD manualmente en: {SD_URL}")

def main():
    print("=" * 60)
    print("SD WebUI - Ventana Pywebview Independiente")
    print("=" * 60)
    print()
    
    # Verificar si SD está corriendo
    if not is_port_in_use(SD_PORT):
        print("[INFO] SD WebUI no detectado, iniciando...")
        if not start_sd_if_needed():
            print("[ERROR] No se pudo iniciar SD WebUI")
            return
    else:
        print(f"[INFO] SD WebUI detectado en: {SD_URL}")
    
    # Abrir en pywebview
    open_sd_window()
    
    print("[INFO] Saliendo...")

if __name__ == "__main__":
    import subprocess  # Importar aquí para que esté disponible
    main()
