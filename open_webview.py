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
    try:
        from webview.platforms.winforms import BrowserView
        import clr
        clr.AddReference('System.Drawing')
        from System.Drawing import Icon as DotNetIcon
    except Exception:
        return
    if not os.path.exists(ICON_PATH):
        return
    while uid not in BrowserView.instances:
        time.sleep(0.1)
    form = BrowserView.instances[uid]
    try:
        form.Icon = DotNetIcon.CreateFromFile(ICON_PATH)
    except Exception:
        try:
            from System import Func, Type
            def _set():
                form.Icon = DotNetIcon.CreateFromFile(ICON_PATH)
            form.Invoke(Func[Type](_set))
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
