#!/usr/bin/env python3
"""
Script para abrir AUTO-DEEP en pywebview esperando a que Gradio esté listo
"""

import sys
import os
import socket
import time
import threading

# Agregar el directorio raíz al path
sys.path.insert(0, os.getcwd())

GRADIO_URL = "http://127.0.0.1:7861"
ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "icon.ico")

def _set_window_icon(uid):
    import time
    import ctypes
    try:
        from webview.platforms.winforms import BrowserView
    except Exception:
        return
    if not os.path.exists(ICON_PATH):
        return
    while uid not in BrowserView.instances:
        time.sleep(0.1)
    form = BrowserView.instances[uid]
    try:
        from System import Action
        from System.Drawing import Icon
        user32 = ctypes.windll.user32
        WM_SETICON = 0x0080
        ICON_SMALL = 0
        ICON_BIG = 1
        IMAGE_ICON = 1
        LR_LOADFROMFILE = 0x0010
        GCLP_HICON = -14
        GCLP_HICONSM = -34
        def _set():
            try:
                try:
                    setAppId = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
                    setAppId.argtypes = [ctypes.c_wchar_p]
                    setAppId.restype = ctypes.c_long
                    setAppId("AutoAuto")
                except Exception:
                    pass
                ico = Icon(ICON_PATH)
                form.Icon = ico
                hWnd_ptr = ctypes.c_void_p(form.Handle.ToInt64())
                shell32 = ctypes.windll.shell32
                shell32.ExtractIconExW.argtypes = [ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
                shell32.ExtractIconExW.restype = ctypes.c_uint
                large = ctypes.c_void_p()
                small = ctypes.c_void_p()
                if shell32.ExtractIconExW(ICON_PATH, 0, ctypes.byref(large), ctypes.byref(small), 1) > 0:
                    h32 = ctypes.c_void_p(large.value)
                    h16 = ctypes.c_void_p(small.value)
                    user32.SendMessageW(hWnd_ptr, WM_SETICON, ICON_BIG, h32)
                    user32.SetClassLongPtrW(hWnd_ptr, GCLP_HICON, h32)
                    user32.SendMessageW(hWnd_ptr, WM_SETICON, ICON_SMALL, h16)
                    user32.SetClassLongPtrW(hWnd_ptr, GCLP_HICONSM, h16)
            except Exception:
                pass
        form.Invoke(Action(_set))
    except Exception:
        pass

def wait_for_server(url, timeout=60):
    """Espera a que el servidor Gradio esté disponible"""
    print(f"[WAIT] Esperando a que {url} esté disponible...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Extraer el puerto de la URL
            port = int(url.split(':')[-1])
            
            # Verificar si el puerto está abierto
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)
            result = sock.connect_ex(('127.0.0.1', port))
            sock.close()
            
            if result == 0:
                print(f"[OK] Servidor detectado en el puerto {port}")
                return True
                
        except Exception as e:
            pass
        
        time.sleep(1)
    
    print(f"[ERROR] Servidor no disponible después de {timeout} segundos")
    return False

def start_gradio_in_background():
    """Inicia Gradio en segundo plano"""
    print("[GRADIO] Iniciando servidor Gradio...")
    try:
        from ui import main as ui_main
        ui_main.run()
    except Exception as e:
        print(f"[ERROR] Error iniciando Gradio: {e}")

def open_webview():
    """Abre la URL en pywebview"""
    print(f"[PYWEBVIEW] Abriendo {GRADIO_URL} en pywebview...")
    
    try:
        import webview
        
        window = webview.create_window(
            'AUTO-DEEP v2',
            GRADIO_URL,
            width=1280,
            height=800,
            resizable=True,
            confirm_close=True,
            background_color='#1a1a1a'
        )
        
        threading.Thread(target=_set_window_icon, args=(window.uid,), daemon=True).start()
        
        print("[PYWEBVIEW] Ventana creada, iniciando bucle...")
        webview.start(debug=False)
        print("[PYWEBVIEW] Ventana cerrada")
        
    except Exception as e:
        print(f"[ERROR] No se pudo abrir pywebview: {e}")
        import webbrowser
        print(f"[INFO] Abriendo en navegador: {GRADIO_URL}")
        webbrowser.open(GRADIO_URL)

def main():
    print("=" * 60)
    print("AUTO-DEEP PyWebView Launcher")
    print("=" * 60)
    
    # Verificar si el servidor ya está ejecutándose
    if not wait_for_server(GRADIO_URL, timeout=5):
        print("[INFO] El servidor no está ejecutándose, iniciando...")
        
        # Iniciar Gradio en un hilo separado
        gradio_thread = threading.Thread(target=start_gradio_in_background, daemon=True)
        gradio_thread.start()
        
        # Esperar a que el servidor esté listo
        if not wait_for_server(GRADIO_URL, timeout=60):
            print("[ERROR] No se pudo iniciar el servidor Gradio")
            sys.exit(1)
    
    # Abrir pywebview
    open_webview()

if __name__ == "__main__":
    main()
