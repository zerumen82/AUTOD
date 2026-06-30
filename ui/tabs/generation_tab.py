#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gradio as gr
import os
import queue
import threading
import time

from roop.img_editor.flux_gen_comfy_client import (
    get_flux_gen_client,
    get_default_generation_engine,
    get_installed_generation_engines,
)
from roop.img_editor.gen_prompt_modifiers import get_dropdown_choices, get_compatible_dropdown_choices, preview_modifiers
from roop.img_editor.comfy_progress import build_generation_progress_html, format_duration
from roop.output_paths import get_generation_output_dir
from ui.job_cancel import (
    SCOPE_GENERATION,
    btn_idle,
    btn_running,
    cancel_status_html,
    clear as clear_cancel,
    is_cancelled,
    request as request_cancel,
)


def on_cancel_generation():
    request_cancel(SCOPE_GENERATION, interrupt_comfy=True)
    return cancel_status_html(), *btn_idle()

def _build_modifiers(
    image_style: str,
    shot_framing: str,
    lighting: str = "auto",
    skin_finish: str = "auto",
    *,
    use_rewriter: bool = False,
) -> dict:
    return {
        "image_type": image_style or "auto",
        "lighting": lighting or "auto",
        "skin_finish": skin_finish or "auto",
        "framing": shot_framing or "auto",
        "color_grade": "auto",
        "use_rewriter": use_rewriter,
    }


def set_orientation(orientation: str):
    if orientation == "Vertical (Retrato)":
        return 768, 1152
    return 1152, 768


def _progress_html_from_prog(prog: dict) -> str:
    pct = float(prog.get("progress") or 0) * 100
    elapsed = format_duration(prog.get("elapsed") or 0)
    eta_sec = prog.get("eta") or 0
    eta = format_duration(eta_sec) if eta_sec > 0 else "—"
    return build_generation_progress_html(
        pct=pct,
        step=int(prog.get("step") or 0),
        total_steps=int(prog.get("total") or 0),
        elapsed=elapsed,
        eta=eta,
        phase=prog.get("phase") or "Generando",
        detail=prog.get("detail") or "",
    )


def _success_status_html(res) -> str:
    status = f"<div style='text-align:center;color:#10b981;padding:10px;font-weight:bold;'>✅ Listo ({res.time_taken:.1f}s)</div>"
    if getattr(res, "lora_applied", ""):
        status += f"<div style='text-align:center;color:#64748b;font-size:11px;'>LoRAs: {res.lora_applied}</div>"
    if res.modifier_suffix:
        status += f"<div style='text-align:center;color:#64748b;font-size:11px;'>Estilo: {res.modifier_suffix}</div>"
    if res.user_translated:
        status += f"<div style='text-align:center;color:#94a3b8;font-size:12px;padding-top:6px;'>{res.user_translated}</div>"
    return status


def on_generate_image(
    prompt, width, height, engine_val, image_style, shot_framing, lighting,
    skin_finish, enhance_quality, use_rewriter,
):
    clear_cancel(SCOPE_GENERATION)
    client = get_flux_gen_client()
    if not client.is_available():
        yield None, "<div style='text-align:center;color:#ef4444;padding:10px;'>❌ ComfyUI no está activo.</div>", *btn_idle()
        return

    updates = queue.Queue()
    result = {"image": None, "status": "", "error": None}

    def progress_callback(prog):
        updates.put(_progress_html_from_prog(prog))

    def worker():
        try:
            success, load_msg = client.load(engine_val)
            if not success:
                result["error"] = f"❌ Error cargando modelo: {load_msg}"
                return

            progress_callback({
                "phase": "Modelo listo",
                "detail": load_msg,
                "progress": 0.01,
                "elapsed": 0,
            })

            if is_cancelled(SCOPE_GENERATION):
                result["error"] = "Cancelado"
                return

            res, msg = client.generate_ai(
                prompt=(prompt or "").strip(),
                width=width,
                height=height,
                prompt_modifiers=_build_modifiers(
                    image_style, shot_framing, lighting, skin_finish,
                    use_rewriter=use_rewriter,
                ),
                enhance_quality=bool(enhance_quality),
                progress_callback=progress_callback,
                cancel_check=lambda: is_cancelled(SCOPE_GENERATION),
            )

            if is_cancelled(SCOPE_GENERATION):
                result["error"] = "Cancelado"
                return

            if res:
                output_dir = get_generation_output_dir()
                out_path = os.path.join(output_dir, f"gen_{int(time.time())}.png")
                res.image.save(out_path)
                print(f"[GenFlux] Imagen guardada: {out_path}")
                result["image"] = res.image
                status = _success_status_html(res)
                status += f"<div style='text-align:center;color:#64748b;font-size:11px;'>Guardado en {output_dir}</div>"
                result["status"] = status
            else:
                result["error"] = f"❌ {msg}"
        except Exception as e:
            import traceback
            traceback.print_exc()
            result["error"] = f"❌ Error: {e}"
        finally:
            updates.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    try:
        yield None, build_generation_progress_html(phase="Iniciando", detail="Conectando con ComfyUI…"), *btn_running()

        while True:
            if is_cancelled(SCOPE_GENERATION):
                break
            try:
                item = updates.get(timeout=0.35)
            except queue.Empty:
                yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
                continue
            if item is None:
                break
            yield None, item, gr.skip(), gr.skip()

        if is_cancelled(SCOPE_GENERATION):
            yield None, cancel_status_html(), *btn_idle()
            return

        if result["error"]:
            err = result["error"]
            if err == "Cancelado":
                yield None, cancel_status_html(), *btn_idle()
            else:
                yield None, f"<div style='text-align:center;color:#ef4444;padding:10px;'>{err}</div>", *btn_idle()
            return

        yield result["image"], result["status"], *btn_idle()
    finally:
        clear_cancel(SCOPE_GENERATION)


def open_generation_folder():
    import subprocess
    import sys

    path = get_generation_output_dir()
    try:
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["explorer", path])
    except Exception:
        pass
    return None


def generation_tab():
    gr.HTML("""
        <style>
            .gen-tab-container {
                background: #020617;
                padding: 30px;
                border-radius: 20px;
                border: 1px solid #1e293b;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            .gen-tab-header { text-align: center; margin-bottom: 25px; }
            .gen-tab-header h2 {
                background: linear-gradient(90deg, #a855f7, #22d3ee);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 32px;
                font-weight: 800;
            }
            .prompt-box-gen {
                background: #0f172a !important;
                border: 2px solid #a855f7 !important;
                border-radius: 12px !important;
                font-size: 20px !important;
            }
            .prompt-box-gen textarea {
                font-size: 20px !important;
                line-height: 1.6 !important;
            }
        </style>
    """)

    with gr.Column(elem_classes=["gen-tab-container"]):
        with gr.Group(elem_classes=["gen-tab-header"]):
            gr.Markdown("## 🚀 GENERAR")
            gr.Markdown(
                "_Escribe en español. Tu prompt va primero; estilo solo si lo eliges en los desplegables. "
                "**Modelo**: FLUX arriba, **SDXL ·** abajo. Schnell = más rápido. "
                "Rewriter/ESRGAN opcionales (OFF = más fiel y rápido)._"
            )

        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe la escena que quieres generar...",
            lines=4,
            elem_classes=["prompt-box-gen"],
        )

        with gr.Row():
            image_style = gr.Dropdown(
                choices=get_dropdown_choices("image_type"),
                value="photoreal",
                label="Tipo de imagen",
            )
            lighting = gr.Dropdown(
                choices=get_dropdown_choices("lighting"),
                value="natural",
                label="Iluminación",
            )
            shot_framing = gr.Dropdown(
                choices=get_dropdown_choices("framing"),
                value="auto",
                label="Plano / encuadre",
            )
            skin_finish = gr.Dropdown(
                choices=get_dropdown_choices("skin_finish"),
                value="detailed",
                label="Piel",
            )

        modifier_preview = gr.HTML("")

        with gr.Row():
            orientation = gr.Radio(
                choices=["Horizontal (Paisaje)", "Vertical (Retrato)"],
                value="Horizontal (Paisaje)",
                label="Forma",
            )
            _engines = get_installed_generation_engines()
            _default = get_default_generation_engine()
            engine_model = gr.Dropdown(
                choices=_engines or [("FLUX Abliterated (ultra realista)", "flux_dev_abliterated")],
                value=_default,
                label="Modelo",
            )

        with gr.Row():
            enhance_quality = gr.Checkbox(
                label="Mejorar nitidez (ESRGAN, más lento)",
                value=False,
            )
            use_rewriter = gr.Checkbox(
                label="Rewriter LLM (mejor prompt, +5–15s)",
                value=False,
            )

        width = gr.Slider(minimum=512, maximum=1536, step=64, value=1152, visible=False)
        height = gr.Slider(minimum=512, maximum=1536, step=64, value=768, visible=False)

        with gr.Row():
            gen_btn = gr.Button("GENERAR", variant="primary", size="lg", scale=3)
            btn_cancel = gr.Button("⏹ CANCELAR", variant="stop", interactive=False, scale=1)

        status_html = gr.HTML("<div style='text-align:center;color:#64748b;padding:10px;'>Listo</div>")
        output_img = gr.Image(label="Resultado", height=600)

        bt_open_folder = gr.Button("📂 Abrir carpeta de salidas", size="sm")

    def handle_shape(orient):
        w, h = set_orientation(orient)
        return gr.update(value=w), gr.update(value=h)

    orientation.change(fn=handle_shape, inputs=[orientation], outputs=[width, height])

    def _on_style_change(style, light, frame, skin):
        l_choices, l_safe = get_compatible_dropdown_choices(style, "lighting", light)
        f_choices, f_safe = get_compatible_dropdown_choices(style, "framing", frame)
        preview = preview_modifiers(style, l_safe, skin, f_safe, "auto")
        return gr.update(choices=l_choices, value=l_safe), gr.update(choices=f_choices, value=f_safe), preview

    def _on_modifiers_change(style, light, frame, skin):
        return preview_modifiers(style, light, skin, frame, "auto")

    image_style.change(
        fn=_on_style_change,
        inputs=[image_style, lighting, shot_framing, skin_finish],
        outputs=[lighting, shot_framing, modifier_preview],
    )
    lighting.change(
        fn=_on_modifiers_change,
        inputs=[image_style, lighting, shot_framing, skin_finish],
        outputs=[modifier_preview],
    )
    shot_framing.change(
        fn=_on_modifiers_change,
        inputs=[image_style, lighting, shot_framing, skin_finish],
        outputs=[modifier_preview],
    )
    skin_finish.change(
        fn=_on_modifiers_change,
        inputs=[image_style, lighting, shot_framing, skin_finish],
        outputs=[modifier_preview],
    )

    bt_open_folder.click(fn=open_generation_folder)

    gen_event = gen_btn.click(
        fn=on_generate_image,
        inputs=[
            prompt, width, height, engine_model, image_style, shot_framing, lighting,
            skin_finish, enhance_quality, use_rewriter,
        ],
        outputs=[output_img, status_html, gen_btn, btn_cancel],
    )

    btn_cancel.click(fn=on_cancel_generation, outputs=[status_html, gen_btn, btn_cancel], queue=False)
    btn_cancel.click(fn=None, cancels=[gen_event], queue=False)

    return {
        "prompt": prompt,
        "gen_btn": gen_btn,
        "btn_cancel": btn_cancel,
        "output_img": output_img,
        "status": status_html,
    }