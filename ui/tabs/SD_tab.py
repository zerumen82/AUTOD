import os
import gradio as gr
import subprocess
import threading
import time
import socket
import urllib.request
import webbrowser
import json

_sd_url = "http://127.0.0.1:9871"
_script_running = False


def check_server_status():
    """Check if SD server is running"""
    try:
        with socket.create_connection(('127.0.0.1', 9871), timeout=2):
            req = urllib.request.Request(_sd_url)
            req.add_header('User-Agent', 'Mozilla/5.0')
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    content = resp.read(1024).decode('utf-8', errors='ignore')
                    if len(content) > 100:
                        return True
    except Exception:
        pass
    return False


def launch_sd_standalone():
    """Launch SD standalone window script"""
    global _script_running
    
    if _script_running:
        return "Ya se esta ejecutando..."
    
    _script_running = True
    
    try:
        script_path = os.path.abspath("tools/sd_standalone_window.py")
        
        def run_script():
            try:
                subprocess.Popen(
                    ["python", script_path],
                    creationflags=subprocess.CREATE_NEW_CONSOLE
                )
            except Exception as e:
                print(f"Error lanzando script: {e}")
            finally:
                time.sleep(10)
                _script_running = False
        
        threading.Thread(target=run_script, daemon=True).start()
        return "Lanzando ventana SD independiente..."
        
    except Exception as e:
        _script_running = False
        return f"[ERROR] {str(e)}"


def start_browser():
    """Open SD in browser"""
    if check_server_status():
        webbrowser.open(_sd_url)
        return "Navegador abierto!"
    else:
        return "Servidor no disponible. Inicia primero el servidor SD."


def start_sd_server():
    """Start SD WebUI server"""
    if check_server_status():
        return "Servidor ya esta corriendo!"
    
    try:
        import roop.core
        from ui.tabs import sd_launcher
        
        roop.core.update_status('Iniciando Stable Diffusion WebUI...')
        
        thread = sd_launcher.start('webui-user.bat')
        
        if thread is None:
            return "SD ya esta iniciandose..."
        
        return "Iniciando SD WebUI... (10-60 segundos)"
        
    except Exception as e:
        return f"[ERROR] {str(e)}"


def free_gpu_memory():
    """Libera VRAM - reinicia el servidor SD WebUI"""
    # Verificar si el servidor está corriendo
    if not check_server_status():
        return "❌ Servidor SD no está corriendo. Inícialo primero."
    
    try:
        from ui.tabs import sd_launcher
        
        # Detener el servidor
        sd_launcher.stop()
        time.sleep(2)
        
        # Reiniciar
        sd_launcher.start('webui-user.bat')
        
        return "✅ SD WebUI reiniciado. VRAM liberada."
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def SD_tab() -> None:
    with gr.Tab("[SD] Stable Diffusion"):
        gr.Markdown("""
        # Stable Diffusion WebUI
        
        Genera imagenes con IA de alta calidad.
        
        **Mejoras aplicadas:**
        - Modelo: epicrealism_naturalSinRC1VAE (SD 1.5 - más rápido)
        - Face Restoration: CodeFormer activado
        """)
        
        with gr.Row():
            btn_start = gr.Button("[>] Iniciar Servidor SD", variant='primary', size='lg')
            btn_browser = gr.Button("[🌐] Abrir en Navegador", variant='secondary', size='lg')
            btn_free_vram = gr.Button("[🔄] Reiniciar SD", variant='stop', size='lg')
            btn_standalone = gr.Button("[📱] Ventana Indepe.", variant='secondary', size='lg')
        
        status_text = gr.Textbox(
            label="Estado",
            interactive=False,
            value="Servidor detenido - Listo para iniciar",
            lines=3
        )
        
        gr.Markdown("### Vista previa (cuando este corriendo)")
        sd_iframe = gr.HTML(
            value=f'<iframe src="{_sd_url}" width="100%" height="600px" frameborder="0"></iframe>',
            visible=True
        )
        
        btn_start.click(fn=start_sd_server, outputs=[status_text])
        btn_browser.click(fn=start_browser, outputs=[status_text])
        btn_free_vram.click(fn=free_gpu_memory, outputs=[status_text])
        btn_standalone.click(fn=launch_sd_standalone, outputs=[status_text])
