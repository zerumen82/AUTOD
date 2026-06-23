#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gradio as gr
import os
import sys
import time
import queue
import threading
from PIL import Image
from roop.img_editor.img_editor_manager import get_img_editor_manager
from roop.img_editor.comfy_progress import build_generation_progress_html, format_duration
from roop.img_editor.prompt_translator import translate_prompt


def open_output_folder():
    path = os.path.abspath("output/img_editor")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(path)
        else:
            import subprocess
            subprocess.Popen(["explorer", path])
    except Exception:
        pass
    return None


_is_generating = False


def _progress_html_from_prog(prog: dict, detail: str = "") -> str:
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
        phase=prog.get("phase") or "Editando",
        detail=detail or prog.get("detail") or "",
    )


def preview_semantic(user_prompt, engine_val, quality_mode_val):
    """Preview ligero: magnitud, target y params estimados (sin vision models)."""
    try:
        mgr = get_img_editor_manager()
        engine_val = engine_val or "imagine"

        if quality_mode_val:
            analysis = {
                "magnitude": 0.35,
                "mask_target": "subject",
                "is_global": True,
                "quality_only": True,
            }
            params = mgr.auto_detect_params(analysis, engine_val)
            return f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:12px;font-size:12px;">
                <div style="color:#22d3ee;font-weight:bold;margin-bottom:8px;">Modo mejora de calidad</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;color:#cbd5e1;">
                    <div><span style="color:#64748b;">Motor:</span> <b>LongCat Full + ESRGAN 2x + nitidez</b></div>
                    <div><span style="color:#64748b;">Modo:</span> <b>Solo calidad (preserva todo)</b></div>
                    <div><span style="color:#64748b;">Denoise:</span> <b>{params['denoise']:.2f}</b></div>
                    <div><span style="color:#64748b;">Pasos:</span> <b>{params['num_inference_steps']}</b></div>
                </div>
                <div style="color:#94a3b8;margin-top:8px;font-size:11px;">
                    Sin instrucción de usuario — mejora fotográfica automática (nitidez, textura, color).
                </div>
            </div>
            """

        user_prompt = (user_prompt or "").strip()
        if not user_prompt:
            return "<div style='color:#f87171;text-align:center;padding:8px;'>Escribe una instrucción primero</div>"

        translated = translate_prompt(user_prompt)
        nlp = mgr._get_semantic_analyzer(full_ai=False)
        mag = nlp.get_magnitude(translated)
        target = nlp.detect_target(translated)
        analysis = {
            "magnitude": mag,
            "mask_target": target,
            "is_global": target == "subject" and mag < 0.45,
            "quality_only": False,
        }
        params = mgr.auto_detect_params(analysis, engine_val)
        longcat_mode = "LongCat Full (auto)" if (engine_val == "imagine" and mag >= 0.68) else (
            "LongCat Turbo" if engine_val in ("imagine", "longcat") else engine_val
        )
        edit_mode = "Global" if mag > 0.6 else ("Máscara ropa" if target in ("clothes", "subject") and mag >= 0.45 else "Máscara/parcial")

        return f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:12px;font-size:12px;">
            <div style="color:#22d3ee;font-weight:bold;margin-bottom:8px;">Análisis local (sin IA pesada)</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;color:#cbd5e1;">
                <div><span style="color:#64748b;">Magnitud:</span> <b>{mag:.2f}</b></div>
                <div><span style="color:#64748b;">Target:</span> <b>{target}</b></div>
                <div><span style="color:#64748b;">Motor:</span> <b>{longcat_mode}</b></div>
                <div><span style="color:#64748b;">Modo:</span> <b>{edit_mode}</b></div>
                <div><span style="color:#64748b;">Denoise:</span> <b>{params['denoise']:.2f}</b></div>
                <div><span style="color:#64748b;">Pasos:</span> <b>{params['num_inference_steps']}</b></div>
            </div>
            <div style="color:#94a3b8;margin-top:8px;font-size:11px;">
                Traducción: {translated[:120]}{'…' if len(translated) > 120 else ''}
            </div>
            <div style="color:#64748b;margin-top:6px;font-size:10px;">
                Sin FacePreserver — LongCat mantiene parecido vía foto ref. Full (mag≥0.68) sigue mejor las instrucciones fuertes.
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color:#f87171;padding:8px;'>Error preview: {e}</div>"


def on_quality_mode_change(enabled):
    """Deshabilita prompt y rewriter cuando el modo calidad está activo."""
    if enabled:
        return (
            gr.update(interactive=False, placeholder="Desactivado — modo mejora de calidad activo"),
            gr.update(interactive=False, value=False),
        )
    return (
        gr.update(
            interactive=True,
            placeholder="Ej: desnuda y descalza, que estén bailando, cambia el fondo...",
        ),
        gr.update(interactive=True),
    )


def on_generate(img_data, p_text, engine_val, use_ai_val, enhance_val, denoise_val, upscale_val, quality_mode_val):
    global _is_generating

    if _is_generating:
        yield None, "⚠️ Ya hay una transformación en proceso", None
        return

    quality_mode_val = bool(quality_mode_val)
    p_text = (p_text or "").strip()
    if not quality_mode_val and not p_text:
        yield None, "Escribe un prompt o activa «Modo mejora de calidad»", None
        return
    if img_data is None:
        yield None, "Sube una imagen", None
        return

    if isinstance(img_data, dict):
        img = img_data.get("background")
    elif isinstance(img_data, str):
        img = Image.open(img_data).convert("RGB")
    else:
        img = img_data

    if img is None or not isinstance(img, Image.Image):
        yield None, "Imagen inválida", None
        return

    mgr = get_img_editor_manager()
    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    if engine_val in ("imagine", "longcat", "longcat_full", "klein_base", "flux_q2", "flux_dev_abliterated"):
        if not get_flux_edit_comfy_client().is_available():
            yield None, "<div style='color:#ef4444;text-align:center;padding:10px;'>❌ ComfyUI no está activo. Inicia ComfyUI primero.</div>", None
            return

    updates = queue.Queue()
    result = {"image": None, "msg": "", "mask": None, "error": None}

    def progress_callback(prog):
        updates.put(_progress_html_from_prog(prog))

    def worker():
        global _is_generating
        try:
            _is_generating = True
            progress_callback({"phase": "Analizando instrucción", "progress": 0.02, "elapsed": 0})

            res_img, msg, mask_img = mgr.generate_intelligent(
                image=img,
                prompt=p_text,
                use_rewriter=use_ai_val,
                engine=engine_val,
                enhance_faces=enhance_val,
                lora_name=None,
                lora_strength=None,
                denoise=denoise_val if denoise_val > 0 else None,
                progress_callback=progress_callback,
                auto_upscale=upscale_val,
                quality_mode=quality_mode_val,
            )

            if res_img:
                output_dir = os.path.abspath("output/img_editor")
                os.makedirs(output_dir, exist_ok=True)
                ts = int(time.time())
                out_path = os.path.join(output_dir, f"edit_{ts}.png")
                res_img.save(out_path)
                print(f"[ImgEditor] Imagen guardada: {out_path}")
                result["image"] = res_img
                result["msg"] = f"<div style='text-align:center;color:#10b981;padding:10px;font-weight:bold;'>✅ {msg}</div><div style='text-align:center;color:#64748b;font-size:11px;'>Guardado en output/img_editor</div>"
                result["mask"] = mask_img
            else:
                result["error"] = f"<div style='text-align:center;color:#ef4444;padding:10px;'>❌ {msg}</div>"
                result["mask"] = mask_img
        except Exception as e:
            import traceback
            traceback.print_exc()
            result["error"] = f"<div style='text-align:center;color:#ef4444;padding:10px;'>❌ Error: {e}</div>"
        finally:
            _is_generating = False
            updates.put(None)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    yield None, build_generation_progress_html(phase="Iniciando", detail="Preparando edición…"), None

    while True:
        item = updates.get()
        if item is None:
            break
        yield None, item, None

    if result["error"]:
        yield None, result["error"], result.get("mask")
        return

    yield result["image"], result["msg"], result.get("mask")


def create_img_editor_tab():
    gr.HTML("""
        <style>
            .img-editor-container {
                background: #020617;
                padding: 30px;
                border-radius: 20px;
                border: 1px solid #1e293b;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            .img-editor-header { text-align: center; margin-bottom: 25px; }
            .img-editor-header h2 {
                background: linear-gradient(90deg, #22d3ee, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 32px;
                font-weight: 800;
            }
            .btn-transform-main {
                background: linear-gradient(135deg, #06b6d4 0%, #a855f7 100%) !important;
                color: white !important;
                font-weight: 900 !important;
                height: 64px !important;
                border-radius: 14px !important;
                font-size: 18px !important;
                letter-spacing: 1px;
                text-transform: uppercase;
                border: none !important;
            }
            .prompt-box-img {
                background: #0f172a !important;
                border: 2px solid #22d3ee !important;
                border-radius: 12px !important;
                font-size: 20px !important;
            }
            .prompt-box-img textarea {
                font-size: 20px !important;
                line-height: 1.6 !important;
            }
        </style>
    """)

    with gr.Column(elem_classes=["img-editor-container"]):
        with gr.Group(elem_classes=["img-editor-header"]):
            gr.Markdown("## ✨ IMAGE EDITOR AI")
            gr.Markdown("_Misma foto + tu instrucción. LongCat mantiene el parecido — sin pegar caras._")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Imagen de Entrada", type="pil", height=480)

                quality_mode = gr.Checkbox(
                    label="✨ Modo mejora de calidad",
                    value=False,
                    info="Mejora nitidez, textura y realismo sin instrucción. Desactiva el prompt.",
                )

                prompt = gr.Textbox(
                    label="¿Qué quieres cambiar?",
                    placeholder="Ej: desnuda y descalza, que estén bailando, cambia el fondo...",
                    lines=3,
                    elem_classes=["prompt-box-img"]
                )

                with gr.Row():
                    gen_btn = gr.Button("✨ TRANSFORMAR", variant="primary", elem_classes=["btn-transform-main"], scale=3)
                    btn_preview = gr.Button("🔍 PREVIEW", size="sm", scale=1)

                with gr.Accordion("⚙️ Opciones Avanzadas", open=False):
                    gr.Markdown("*Defaults optimizados — sube foto, escribe instrucción y TRANSFORMAR.*")
                    use_ai = gr.Checkbox(
                        label="🧠 Análisis inteligente + rewriter",
                        value=True,
                        info="Análisis local ligero. Rewriter LLM en instrucciones fuertes (mag>0.6).",
                    )
                    with gr.Row():
                        denoise = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                            label="Fuerza de Edición (0 = Auto)",
                            info="0 = automático según magnitud. Turbo ignora denoise (siempre 1.0).",
                        )
                    engine = gr.Dropdown(
                        choices=[
                            ("✨ Grok Imagine (Turbo rápido; Full auto si mag≥0.68)", "imagine"),
                            ("LongCat Turbo (rápido, ~1-2 min)", "longcat"),
                            ("LongCat Full (mejor instrucción, ~3-5 min)", "longcat_full"),
                            ("HART (Autoregressive)", "hart"),
                            ("FLUX.2 Klein", "klein_base"),
                            ("FLUX.1 Dev Q2", "flux_q2"),
                            ("OmniGen 2", "omnigen2"),
                            ("FLUX.1 Dev Abliterated", "flux_dev_abliterated"),
                            ("Qwen Image Edit", "qwen_edit"),
                        ],
                        value="imagine",
                        label="Motor de Generación",
                    )
                    upscale_auto = gr.Checkbox(
                        label="📐 Upscale 2× + nitidez post-edición",
                        value=True,
                        info="ESRGAN x2 + sharpen opcional tras ediciones normales. En modo calidad siempre activo.",
                    )
                    enhance_faces = gr.Checkbox(
                        label="🌟 Mejorar textura facial (CodeFormer suave)",
                        value=False,
                        info="Opcional en edición normal. En modo calidad se aplica automáticamente.",
                    )

                status = gr.HTML("<div style='text-align:center; color:#64748b; padding:10px;'>Listo</div>")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🖼️ RESULTADO"):
                        output_img = gr.Image(label="Resultado", height=500)
                    with gr.TabItem("🎭 MÁSCARA"):
                        mask_preview = gr.Image(label="Máscara generada", height=400)

                with gr.Row():
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA")
                    bt_use_as_input = gr.Button("🔄 USAR COMO ENTRADA")

                    bt_open_folder.click(fn=open_output_folder)
                    bt_use_as_input.click(fn=lambda x: x, inputs=[output_img], outputs=[input_img])

    quality_mode.change(
        on_quality_mode_change,
        inputs=[quality_mode],
        outputs=[prompt, use_ai],
    )

    gen_btn.click(
        on_generate,
        [input_img, prompt, engine, use_ai, enhance_faces, denoise, upscale_auto, quality_mode],
        [output_img, status, mask_preview],
        concurrency_limit=None,
    )

    btn_preview.click(preview_semantic, [prompt, engine, quality_mode], [status])

    return {
        "input_img": input_img,
        "prompt": prompt,
        "quality_mode": quality_mode,
        "gen_btn": gen_btn,
        "output_img": output_img,
        "status": status,
        "mask_preview": mask_preview,
        "enhance_faces": enhance_faces,
    }