#!/usr/bin/env python3
"""
Script simple para abrir AUTO-DEEP en pywebview
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.getcwd())

# URL del servidor Gradio que ya está ejecutándose
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

print(f"[PYWEBVIEW] Abriendo {GRADIO_URL} en pywebview...")

try:
    import webview
    import threading
    
    # Crear ventana pywebview
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
    
    # Iniciar webview - esto bloqueará hasta que se cierre la ventana
    webview.start(debug=False)
    print("[PYWEBVIEW] Ventana cerrada")
    
except Exception as e:
    print(f"[ERROR] No se pudo abrir pywebview: {e}")
    print(f"[INFO] Abriendo en navegador...")
    
    # Fallback al navegador
    import webbrowser
    webbrowser.open(GRADIO_URL)
