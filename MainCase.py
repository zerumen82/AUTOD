import webview
import threading
import time
import os
import subprocess
import sys
import socket

_public_url = "http://127.0.0.1:7861"
_sd_url = None
_sd_process = None
_sd_window = None
_main_window = None

def set_public_url(url):
    global _public_url
    _public_url = url
    print(f"[DEBUG] MainCase: URL establecida a {_public_url}")

def set_sd_url(url):
    global _sd_url
    _sd_url = url
    print(f"[DEBUG] MainCase: SD URL establecida a {_sd_url}")

def get_sd_url():
    return _sd_url

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_sd_webui():
    """Inicia SD WebUI y retorna la URL"""
    global _sd_url, _sd_process
    
    sd_port = 9871
    
    if is_port_in_use(sd_port):
        _sd_url = f"http://127.0.0.1:{sd_port}"
        return _sd_url
    
    args = [
        '--port', str(sd_port),
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
    
    if os.path.exists(venv_python):
        cmd = [venv_python, launch_py] + args
    else:
        cmd = [sys.executable, launch_py] + args
    
    print(f"[MainCase] Iniciando SD WebUI...")
    _sd_process = subprocess.Popen(cmd, cwd=sd_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Esperar a que esté listo
    max_wait = 180
    waited = 0
    while waited < max_wait:
        if is_port_in_use(sd_port):
            _sd_url = f"http://127.0.0.1:{sd_port}"
            print(f"[MainCase] SD WebUI listo en: {_sd_url}")
            return _sd_url
        time.sleep(1)
        waited += 1
    
    return None

def open_sd_window(url, title="Stable Diffusion"):
    """Abre SD en una nueva ventana pywebview separada"""
    global _sd_url
    
    print(f"[MainCase] Abriendo SD: {url}")
    _sd_url = url
    set_sd_url(url)
    
    # Crear un script temporal para abrir la ventana de SD
    # Esto evita el problema de que pywebview debe correr en el main thread
    script = f'''
import webview
import sys

def open_sd():
    window = webview.create_window(
        "{title}",
        "{url}",
        width=1400,
        height=900,
        resizable=True,
        background_color='#1a1a1a'
    )
    webview.start(debug=False)

if __name__ == "__main__":
    open_sd()
'''
    
    # Guardar script temporal
    temp_script = os.path.join(os.getcwd(), "_sd_webview_temp.py")
    try:
        with open(temp_script, 'w', encoding='utf-8') as f:
            f.write(script)
        
        # Ejecutar en proceso separado SIN ventana de consola
        python_exe = sys.executable
        if sys.platform == 'win32':
            # CREATE_NO_WINDOW = 0x08000000 - ejecuta sin crear ventana de consola
            subprocess.Popen(
                [python_exe, temp_script], 
                creationflags=0x08000000,  # CREATE_NO_WINDOW
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        else:
            subprocess.Popen([python_exe, temp_script])
        
        print(f"[MainCase] SD abierto en ventana pywebview separada: {url}")
        return True
        
    except Exception as e:
        print(f"[MainCase] Error creando ventana SD: {e}")
        # Fallback a navegador
        import webbrowser
        webbrowser.open(url)
        print(f"[MainCase] SD abierto en navegador: {url}")
        return False

def run_gradio_and_load_url():
    """Lanza la ventana de webview"""
    global _public_url, _main_window
    
    print(f"[MainCase] Lanzando ventana webview para: {_public_url}")
    
    try:
        # Crear ventana para AUTO-DEEP
        _main_window = webview.create_window(
            'AUTO-DEEP v2',
            _public_url,
            width=1280,
            height=800,
            resizable=True,
            confirm_close=True,
            background_color='#1a1a1a'
        )
        
        # Iniciar webview
        webview.start(debug=False)
        
    except Exception as e:
        print(f"[ERROR] No se pudo lanzar webview: {e}")
        import webbrowser
        webbrowser.open(_public_url)
        
        while True:
            time.sleep(1)

def cleanup_windows():
    print("Cerrando ventanas de AUTO-DEEP...")
    # Limpiar script temporal si existe
    temp_script = os.path.join(os.getcwd(), "_sd_webview_temp.py")
    try:
        if os.path.exists(temp_script):
            os.remove(temp_script)
    except:
        pass
