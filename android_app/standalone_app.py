import os
import sys
import time
import requests
import gradio as gr
from PIL import Image

# Lista de URLs de autoconexión en orden de prioridad
SERVERS_TO_TRY = [
    "http://192.168.1.50:7861",
    "http://192.168.1.1:7861",
    "http://localhost:7861",
    "http://127.0.0.1:7861",
]

def auto_connect_server():
    for url in SERVERS_TO_TRY:
        try:
            r = requests.get(f"{url}/config", timeout=1.5)
            if r.status_code == 200:
                print(f"[AutoConnect] Conectado con éxito a: {url}")
                return url
        except Exception:
            pass
    return SERVERS_TO_TRY[0]

ACTIVE_SERVER_URL = auto_connect_server()

def send_image_edit(image_pil, prompt_text, engine_val, denoise_val, quality_mode,
                    autoregressive, enhance_tier, quality_preserve_faces):
    if image_pil is None:
        return None, "❌ Sube o haz una foto primero"
    if not prompt_text and not quality_mode:
        return None, "❌ Escribe qué quieres cambiar"

    url = auto_connect_server()
    try:
        # Enviar petición directa a la API de Image Editor del servidor PC
        import io
        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        buf.seek(0)

        # Usar la API de Gradio Client remota
        from gradio_client import Client, handle_file
        temp_img_path = os.path.join(os.getcwd(), "temp_upload.png")
        image_pil.save(temp_img_path)

        client = Client(url)
        try:
            info = requests.get(f"{url}/gradio_api/info", timeout=3).json()
            has_options = "/edit_options" in info.get("named_endpoints", {})
        except Exception:
            has_options = False
        args = (
            handle_file(temp_img_path), prompt_text, engine_val, denoise_val,
            autoregressive, 1, quality_mode, enhance_tier,
        )
        res = client.predict(
            *args, *([quality_preserve_faces] if has_options else []),
            api_name="/edit_options" if has_options else "/edit",
        )
        if res and isinstance(res, (tuple, list)):
            out_img_path = res[0]
            msg = res[1]
            if out_img_path and os.path.exists(out_img_path):
                return Image.open(out_img_path), f"✅ Edición completada: {msg}"
        return None, f"Procesado en servidor ({url})"
    except Exception as e:
        return None, f"⚠️ Error al conectar con PC ({url}): {str(e)}"

def build_app():
    with gr.Blocks(title="AutoAuto Image Editor App", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 📱 AutoAuto Image Editor")
        gr.Markdown("_Autoconectado con tu PC. Selecciona foto, escribe prompt y transforma._")

        with gr.Row():
            input_img = gr.Image(label="Imagen / Cámara", type="pil")
            output_img = gr.Image(label="Resultado transformado")

        prompt = gr.Textbox(label="Instrucción de Edición", placeholder="Ej: vestido rojo, cambia el fondo, hazlo más nítido...", lines=3)

        with gr.Row():
            engine = gr.Dropdown(choices=[("LongCat Turbo", "longcat"), ("LongCat Full (lento)", "longcat_full")], value="longcat", label="Motor IA")
            denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.0, label="Fuerza (0 = Auto)")

        with gr.Row():
            quality_mode = gr.Checkbox(label="🎨 Modo Calidad", value=False)
            autoregressive = gr.Checkbox(label="🔁 Autoregresivo", value=False)
        enhance_tier = gr.Radio(choices=[("HD", "hd"), ("4K", "4k"), ("8K", "8k")], value="hd", label="Resolución de salida")
        quality_preserve_faces = gr.Checkbox(label="Restaurar caras (puede alterar rasgos)", value=True)

        btn = gr.Button("✨ TRANSFORMAR IMAGEN", variant="primary", size="lg")
        status = gr.Markdown("🟢 Conectado con el servidor de tu PC")

        btn.click(
            fn=send_image_edit,
            inputs=[input_img, prompt, engine, denoise, quality_mode, autoregressive, enhance_tier, quality_preserve_faces],
            outputs=[output_img, status]
        )

    return demo

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7866, quiet=True)
