#!/usr/bin/env python3
"""
Script para lanzar AUTO-DEEP con pywebview en lugar del navegador
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.getcwd())

# Importar y configurar la URL antes de importar MainCase
from ui.main import find_free_port
import roop.globals

# Encontrar puerto libre
port = find_free_port()
url = f"http://127.0.0.1:{port}"

print(f"[LAUNCHER] Puerto encontrado: {port}")
print(f"[LAUNCHER] URL: {url}")

# Importar MainCase y configurar la URL
import MainCase
MainCase.set_public_url(url)

print(f"[LAUNCHER] URL configurada en MainCase")

# Ahora lanzar la UI de Gradio en un hilo separado y luego abrir pywebview
import threading
import time

def start_gradio():
    """Inicia Gradio en segundo plano"""
    from ui import main as ui_main
    try:
        ui_main.run()
    except Exception as e:
        print(f"[LAUNCHER] Error en Gradio: {e}")

# Iniciar Gradio en un hilo
gradio_thread = threading.Thread(target=start_gradio, daemon=True)
gradio_thread.start()

# Esperar a que Gradio esté listo
print("[LAUNCHER] Esperando a que Gradio esté listo...")
time.sleep(5)

# Verificar que Gradio está ejecutándose
import socket
try:
    with socket.create_connection(('127.0.0.1', port), timeout=5):
        print(f"[LAUNCHER] Gradio está ejecutándose en el puerto {port}")
except Exception as e:
    print(f"[LAUNCHER] Error conectando a Gradio: {e}")
    sys.exit(1)

# Ahora lanzar pywebview
print("[LAUNCHER] Lanzando pywebview...")
MainCase.run_gradio_and_load_url()
