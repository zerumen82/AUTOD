import os
import sys
import time
import requests
import json
import gradio as gr
from PIL import Image

SERVER_IP_FILE = "server_config.json"

def get_server_url():
    if os.path.exists(SERVER_IP_FILE):
        try:
            with open(SERVER_IP_FILE, "r") as f:
                data = json.load(f)
                return data.get("url", "http://192.168.1.50:7861")
        except Exception:
            pass
    return "http://192.168.1.50:7861"

def save_server_url(url):
    url = url.strip().rstrip("/")
    with open(SERVER_IP_FILE, "w") as f:
        json.dump({"url": url}, f)
    return f"Conectado a: {url}"

def build_client_ui():
    with gr.Blocks(title="AutoAuto Image Editor - Mobile App") as demo:
        gr.Markdown("# 🎨 AutoAuto Image Editor (Android Client)")
        
        with gr.Accordion("🔌 Configuración de Conexión al PC", open=False):
            url_input = gr.Textbox(label="IP / URL del PC Servidor", value=get_server_url(), placeholder="http://192.168.1.X:7861 o https://xxx.gradio.live")
            btn_save = gr.Button("💾 Guardar Conexión")
            status_conn = gr.Markdown(f"Estado: Configurado para {get_server_url()}")
            btn_save.click(fn=save_server_url, inputs=[url_input], outputs=[status_conn])

        with gr.Row():
            input_img = gr.Image(label="Imagen a Editar", type="pil")
            output_img = gr.Image(label="Resultado")

        prompt = gr.Textbox(label="Prompt / Instrucción", placeholder="Ej: cambia el vestido a color rojo, añade gafas de sol...", lines=3)
        
        with gr.Row():
            engine = gr.Dropdown(choices=[("LongCat Edit", "longcat"), ("FLUX.2 Klein", "klein_9b")], value="longcat", label="Motor")
            denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.0, label="Fuerza")

        with gr.Row():
            quality_mode = gr.Checkbox(label="🎨 Modo Calidad", value=False)
            autoregressive = gr.Checkbox(label="🔁 Autoregresivo", value=False)

        btn_send = gr.Button("✨ TRANSFORMAR EN EL PC", variant="primary", size="lg")
        status_txt = gr.Markdown("Listo para enviar")

        return demo

if __name__ == "__main__":
    app = build_client_ui()
    app.launch(server_name="0.0.0.0", server_port=7865)
