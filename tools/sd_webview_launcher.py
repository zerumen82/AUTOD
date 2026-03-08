"""
Script para abrir Stable Diffusion en pywebview ANTES de iniciar AUTO-DEEP
===========================================================================

Este script inicia SD WebUI y lo abre en una ventana pywebview.
Luego inicia AUTO-DEEP en una ventana separada o en el navegador.

USO:
    python tools/sd_webview_launcher.py
"""

import os
import sys
import time
import threading
import subprocess
import socket
import urllib.request

# Configurar encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

SD_WEBUI_PORT = 9871
SD_WEBUI_PATH = "ui/tob/stable-diffusion-webui"

def wait_for_server(port, timeout=120):
    """Espera hasta que el servidor esté listo"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection(('127.0.0.1', port), timeout=2):
                return True
        except:
            time.sleep(1)
    return False

def start_sd_webui():
    """Inicia SD WebUI en background"""
    print("[INFO] Iniciando Stable Diffusion WebUI...")
    
    # Arguments optimizados para calidad
    args = [
        '--port', str(SD_WEBUI_PORT),
        '--api',
        '--cors-allow-origins', '*',
        '--skip-torch-cuda-test',
        '--opt-sdp-attention',
        '--opt-channelslast',
        '--disable-safe-unpickle',
        '--no-half-vae',
        '--medvram'
    ]
    
    venv_python = os.path.join(SD_WEBUI_PATH, "venv", "Scripts", "python.exe")
    launch_py = os.path.join(SD_WEBUI_PATH, "launch.py")
    
    if os.path.exists(venv_python):
        cmd = [venv_python, launch_py] + args
    else:
        cmd = ['python', launch_py] + args
    
    proc = subprocess.Popen(cmd, cwd=SD_WEBUI_PATH)
    print(f"[INFO] SD WebUI iniciado con PID: {proc.pid}")
    return proc

def open_sd_in_pywebview(sd_url):
    """Abre SD WebUI en una ventana pywebview"""
    try:
        import webview
        print(f"[INFO] Abriendo SD WebUI en pywebview: {sd_url}")
        
        window = webview.create_window(
            'Stable Diffusion WebUI',
            sd_url,
            width=1400,
            height=900,
            resizable=True,
            background_color='#1a1a1a'
        )
        
        # Iniciar webview - esto bloquea hasta que se cierre
        webview.start(debug=False)
        print("[INFO] Ventana de SD cerrada")
        
    except Exception as e:
        print(f"[ERROR] Error con pywebview: {e}")
        print(f"[INFO] Abre manualmente: {sd_url}")

def main():
    """Función principal"""
    print("=" * 60)
    print("AUTO-DEEP v2.2.2 - SD WebUI Launcher")
    print("=" * 60)
    print()
    
    # Verificar si SD ya está corriendo
    if wait_for_server(SD_WEBUI_PORT, timeout=2):
        print("[INFO] SD WebUI ya está corriendo")
        sd_url = f"http://127.0.0.1:{SD_WEBUI_PORT}"
    else:
        # Iniciar SD WebUI
        start_sd_webui()
        
        # Esperar a que esté listo
        print(f"[INFO] Esperando a que SD WebUI esté listo en puerto {SD_WEBUI_PORT}...")
        if wait_for_server(SD_WEBUI_PORT, timeout=180):
            sd_url = f"http://127.0.0.1:{SD_WEBUI_PORT}"
            print(f"[OK] SD WebUI listo en: {sd_url}")
        else:
            print("[ERROR] Timeout esperando SD WebUI")
            return
    
    # Abrir en pywebview
    open_sd_in_pywebview(sd_url)
    
    print("[INFO] Saliendo...")

if __name__ == "__main__":
    main()
