#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main UI - Interfaz Gradio principal
"""

import os
import sys
import gradio as gr
import atexit
import socket

# Puerto base dinamico
GRADIO_BASE_PORT = int(os.environ.get('GRADIO_SERVER_PORT', '7860'))
COMFYUI_PORT = os.environ.get('COMFYUI_PORT', '8188')


def prepare_environment():
    """
    Prepara el entorno de ejecución configurando rutas y variables globales.
    """
    import roop.globals
    import time
    
    # Configurar ruta de salida
    if not hasattr(roop.globals, 'output_path') or roop.globals.output_path is None:
        roop.globals.output_path = os.path.join(os.getcwd(), 'output')
    
    # Asegurar que el directorio de salida existe
    os.makedirs(roop.globals.output_path, exist_ok=True)
    
    # Configurar variables globales por defecto si no están establecidas
    if not hasattr(roop.globals, 'processing'):
        roop.globals.processing = True
    
    if not hasattr(roop.globals, 'execution_threads'):
        roop.globals.execution_threads = getattr(roop.globals.CFG, 'max_threads', 4) if hasattr(roop.globals, 'CFG') else 4
    
    if not hasattr(roop.globals, 'execution_providers'):
        roop.globals.execution_providers = suggest_execution_providers()
    
    # Configurar timestamp para nombres de archivo
    if not hasattr(roop.globals, 'start_time'):
        roop.globals.start_time = time.time()
    
    print(f"[ENV] output_path: {roop.globals.output_path}")
    print(f"[ENV] execution_threads: {roop.globals.execution_threads}")
    print(f"[ENV] execution_providers: {roop.globals.execution_providers}")


def suggest_execution_providers():
    """
    Sugiere los proveedores de ejecución disponibles (CUDA, CPU, etc.)
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        # Priorizar CUDA sobre CPU
        if "CUDAExecutionProvider" in providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    except ImportError:
        return ["CPUExecutionProvider"]


def show_msg(msg: str):
    """Muestra un mensaje al usuario."""
    print(f"[MSG] {msg}")


def find_available_port(start_port: int = 7860, max_attempts: int = 10) -> int:
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError(f"No hay puerto libre")


# Alias for backward compatibility
find_free_port = find_available_port


def cleanup_comfyui():
    from ui.tabs.comfy_launcher import stop
    try:
        stop()
    except:
        pass


def check_comfy_status():
    from ui.tabs.comfy_launcher import is_comfyui_running
    if is_comfyui_running():
        return "🟢 running"
    return "🔴 stopped"


def start_comfyui():
    from ui.tabs.comfy_launcher import start
    start(directly_run=True)
    return check_comfy_status()


def stop_comfyui():
    from ui.tabs.comfy_launcher import stop
    stop()
    return check_comfy_status()


# ============== SD Launcher Functions ==============

def check_sd_status():
    """Verifica si SD WebUI está corriendo"""
    from ui.tabs.sd_launcher import get_last_url
    import socket
    
    # Verificar puertos comunes de SD
    for port in [9871, 9872, 9873, 9874, 9875]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex(('127.0.0.1', port)) == 0:
                    return f"🟢 running (port {port})"
        except:
            pass
    
    url = get_last_url()
    if url:
        return f"🟡 {url}"
    return "🔴 stopped"


def start_sd():
    """Inicia SD WebUI"""
    from ui.tabs.sd_launcher import start as sd_start
    sd_start()
    return check_sd_status()


def stop_sd():
    """Detiene SD WebUI"""
    from ui.tabs.sd_launcher import stop as sd_stop
    sd_stop()
    return check_sd_status()


def open_sd_webview():
    """Abre SD en navegador/webview"""
    from ui.tabs.sd_launcher import get_last_url
    import webbrowser
    
    url = get_last_url()
    if url:
        webbrowser.open(url)
        return f"Abriendo: {url}"
    
    # Intentar puertos comunes
    for port in [9871, 9872, 9873]:
        url = f"http://127.0.0.1:{port}"
        try:
            import urllib.request
            urllib.request.urlopen(url, timeout=2)
            webbrowser.open(url)
            return f"Abriendo: {url}"
        except:
            pass
    
    return "SD no está disponible"


def create_ui():
    atexit.register(cleanup_comfyui)
    
    with gr.Blocks(title="AutoAuto - AI Editor") as demo:
        # Header simple
        gr.Markdown("# AutoAuto")
        
        # Controles ComfyUI - compacto, mitad del ancho
        with gr.Row(elem_id="comfy-controls", variant="compact"):
            # Status pequeno a la izquierda
            comfy_status = gr.Text(
                value=check_comfy_status(),
                interactive=False,
                elem_id="comfy-status",
                show_label=False,
                min_width=100
            )
            # Botones pegados a la derecha
            with gr.Group(elem_id="comfy-buttons"):
                btn_refresh = gr.Button("↻", size="xs", variant="secondary")
                btn_start = gr.Button("▶", size="xs", variant="primary")
                btn_stop = gr.Button("⏹", size="xs", variant="stop")
        
        gr.Markdown("---")
        
        # Tabs
        with gr.Tabs() as tabs:
            # Tab 1: Face Swap
            with gr.Tab("Face Swap"):
                try:
                    from ui.tabs.faceswap_tab import faceswap_tab
                    faceswap_tab()
                except Exception as e:
                    gr.Markdown(f"Error: {e}")
            
            # Tab 2: Animate Image
            with gr.Tab("Animate Image"):
                try:
                    from ui.tabs.animate_photo_tab import animate_photo_tab
                    animate_photo_tab()
                except Exception as e:
                    gr.Markdown(f"Error: {e}")
            
            # Tab 3: SD Editor
            with gr.Tab("SD Editor"):
                try:
                    from ui.tabs.img_editor_tab import create_img_editor_tab
                    create_img_editor_tab()
                except Exception as e:
                    gr.Markdown(f"Error: {e}")
            
            # Tab 4: SD Launcher
            with gr.Tab("SD Launcher"):
                gr.Markdown("### Stable Diffusion WebUI Launcher")
                
                with gr.Row():
                    sd_status = gr.Text(
                        value=check_sd_status(),
                        label="Estado",
                        interactive=False,
                        scale=2
                    )
                    sd_btn_refresh = gr.Button("↻", size="sm", variant="secondary", scale=1)
                
                with gr.Row():
                    sd_btn_start = gr.Button("▶ Iniciar SD", variant="primary", scale=1)
                    sd_btn_stop = gr.Button("⏹ Detener SD", variant="stop", scale=1)
                    sd_btn_open = gr.Button("🌐 Abrir en Navegador", variant="secondary", scale=1)
                
                sd_msg = gr.Textbox(label="Mensaje", interactive=False, visible=False)
                
                gr.Markdown("""
                ---
                **Instrucciones:**
                1. Click en "Iniciar SD" para lanzar Stable Diffusion WebUI
                2. Espera ~1-2 minutos para que cargue el modelo
                3. Click en "Abrir en Navegador" cuando esté listo
                
                **Puerto por defecto:** 9871-9875
                """)
            
            # Tab 5: Settings
            with gr.Tab("Settings"):
                try:
                    from ui.tabs.settings_tab import settings_tab
                    settings_tab()
                except Exception as e:
                    gr.Markdown(f"Error: {e}")
        
        # Eventos de ComfyUI
        btn_refresh.click(fn=check_comfy_status, outputs=[comfy_status])
        btn_start.click(fn=start_comfyui, outputs=[comfy_status])
        btn_stop.click(fn=stop_comfyui, outputs=[comfy_status])
        
        # Eventos de SD Launcher
        sd_btn_refresh.click(fn=check_sd_status, outputs=[sd_status])
        sd_btn_start.click(fn=start_sd, outputs=[sd_status])
        sd_btn_stop.click(fn=stop_sd, outputs=[sd_status])
        sd_btn_open.click(fn=open_sd_webview, outputs=[sd_msg])
    
    return demo


def main():
    available_port = find_available_port()
    demo = create_ui()
    demo.queue()  # Habilitado - necesario para el funcionamiento de los botones
    demo.launch(
        server_name="127.0.0.1",
        server_port=available_port,
        share=False,
        show_error=True,
        quiet=True
    )


def run():
    """Alias for main()"""
    main()


if __name__ == "__main__":
    main()
