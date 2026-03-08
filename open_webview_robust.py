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

GRADIO_URL = "http://127.0.0.1:9000"

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
