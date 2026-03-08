#!/usr/bin/env python3
"""
Script simple para abrir AUTO-DEEP en pywebview
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.getcwd())

# URL del servidor Gradio que ya está ejecutándose
GRADIO_URL = "http://127.0.0.1:9000"

print(f"[PYWEBVIEW] Abriendo {GRADIO_URL} en pywebview...")

try:
    import webview
    
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
    
    # Iniciar webview - esto bloqueará hasta que se cierre la ventana
    webview.start(debug=False)
    print("[PYWEBVIEW] Ventana cerrada")
    
except Exception as e:
    print(f"[ERROR] No se pudo abrir pywebview: {e}")
    print(f"[INFO] Abriendo en navegador...")
    
    # Fallback al navegador
    import webbrowser
    webbrowser.open(GRADIO_URL)
