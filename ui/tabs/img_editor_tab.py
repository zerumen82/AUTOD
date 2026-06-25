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
from roop.output_paths import get_img_editor_output_dir
from roop.img_editor.comfy_progress import build_generation_progress_html, format_duration
from roop.img_editor.prompt_translator import translate_prompt
from ui.job_cancel import (
    SCOPE_IMG_EDITOR,
    btn_idle,
    btn_running,
    cancel_status_html,
    clear as clear_cancel,
    is_cancelled,
    request as request_cancel,
)


def open_output_folder():
    path = get_img_editor_output_dir()
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
_generate_lock = threading.Lock()


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


ENHANCE_TIER_LABELS = {
    "hd": "HD (~1920px lado largo, 2×)",
    "4k": "4K (~3840px, multi-upscale)",
    "8k": "8K (~7680px, textura+detalle+contraste máx.)",
}

QUALITY_STYLE_LABELS = {
    "auto": "Automático",
    "solo_mejora": "Solo nitidez/resolución (toda la foto)",
    "hibrido": "Editar algo + mejorar (fondo, ropa…)",
}

QUALITY_HELP_HTML = """
<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:10px;font-size:12px;color:#cbd5e1;line-height:1.5;">
<b style="color:#22d3ee;">¿Qué hace «Mejorar imagen»?</b><br>
• <b>Sin prompt</b> → sube nitidez y resolución de <u>toda</u> la foto.<br>
• <b>«ultra realista, más nitidez»</b> → igual, mejora <u>toda</u> la foto (no cambia fondo ni ropa).<br>
• <b>«convierte el fondo en…»</b> → usa modo <b>Editar + mejorar</b> (o Automático). Edita el fondo y luego mejora.<br>
• <b>Nivel HD/4K/8K</b> = tamaño máximo de salida (8K tarda más).
</div>
"""


def preview_semantic(
    user_prompt,
    engine_val,
    quality_mode_val,
    enhance_tier_val,
    quality_style_val,
    quality_longcat_val,
    img_data,
):
    """Preview ligero: magnitud, target y params estimados (sin vision models)."""
    try:
        mgr = get_img_editor_manager()
        engine_val = engine_val or "imagine"

        user_prompt = (user_prompt or "").strip()
        if quality_mode_val:
            tier = (enhance_tier_val or "hd").lower()
            tier_label = ENHANCE_TIER_LABELS.get(tier, tier)
            style = (quality_style_val or "auto").lower()
            style_label = QUALITY_STYLE_LABELS.get(style, style)
            longcat_on = bool(quality_longcat_val)
            prompt_l = user_prompt.lower()
            bg_words = any(w in prompt_l for w in (
                "fondo", "background", "entorno", "escena", "paisaje", "ambiente",
            ))
            if not user_prompt:
                route_note = "Sin prompt → mejora toda la foto (nitidez + resolución)"
            elif bg_words and style == "solo_mejora":
                route_note = "⚠️ Pides FONDO pero «Solo nitidez» no edita escena → usa «Editar + mejorar»"
            elif bg_words:
                route_note = "Editará el FONDO (personas igual) + mejora TOP después"
            elif style == "solo_mejora":
                route_note = "Mejora TODA la foto — no cambia fondo, ropa ni pose"
            elif style == "hibrido":
                route_note = "LongCat edita lo que pidas + mejora TOP después"
            else:
                route_note = "Auto: «ultra realista» → toda la foto; «fondo/desnuda» → edit + TOP"
            hybrid_note = f"<div style='color:#a78bfa;margin-top:6px;'>{route_note}</div>"
            plan_html = "<div style='color:#64748b;'>Sube una imagen para ver el plan adaptativo</div>"
            if img_data is not None:
                try:
                    from roop.img_editor.image_quality_pipeline import analyze_quality_plan
                    if isinstance(img_data, dict):
                        pil = img_data.get("background")
                    else:
                        pil = img_data
                    if pil is not None:
                        plan = analyze_quality_plan(pil, tier=tier)
                        s = plan.get("scores", {})
                        p = plan.get("profile", {})
                        plan_mode = p.get("plan_mode", "medio")
                        if p.get("clean_sharp"):
                            route = "solo upscale"
                        elif p.get("global_restore"):
                            route = "restaurar naturalidad (sin tiles)"
                        else:
                            route = plan_mode
                        plan_html = f"""
                        <div style="color:#22d3ee;margin-top:8px;font-weight:bold;">Plan: {plan_mode.upper()} — {route}</div>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;color:#cbd5e1;margin-top:6px;">
                            <div><span style="color:#64748b;">Posterización:</span> <b>{s.get('poster', 0):.2f}</b></div>
                            <div><span style="color:#64748b;">Restauración:</span> <b>{s.get('restore', 0):.2f}</b></div>
                            <div><span style="color:#64748b;">Nitidez actual:</span> <b>{s.get('sharpness', 0):.2f}</b></div>
                            <div><span style="color:#64748b;">Upscale:</span> <b>{s.get('upscale_need', 0):.2f}</b></div>
                            <div><span style="color:#64748b;">Desposterizar:</span> <b>{'omitido' if p.get('skip_depixelize') else f"{p.get('depixelize_blend', 0):.0%}"}</b></div>
                            <div><span style="color:#64748b;">Textura global:</span> <b>{p.get('texture_blend', 0):.0%}</b></div>
                            <div><span style="color:#64748b;">Tiles locales:</span> <b>{'omitidos' if p.get('skip_tiles') else f"≤{p.get('tile_blend_cap', 0):.0%}"}</b></div>
                            <div><span style="color:#64748b;">Nitidez final:</span> <b>{p.get('sharpen', 1):.2f}</b></div>
                        </div>
                        <div style="color:#94a3b8;margin-top:8px;font-size:11px;">{plan.get('summary', '')}</div>
                        """
                except Exception as e:
                    plan_html = f"<div style='color:#f87171;'>Error análisis: {e}</div>"
            sem_html = ""
            if user_prompt:
                translated = translate_prompt(user_prompt)
                mag = 0.0
                try:
                    nlp = mgr._get_semantic_analyzer(full_ai=False)
                    mag = nlp.get_magnitude(translated)
                    target = nlp.detect_target(translated)
                    sem_html = (
                        f"<div style='color:#22d3ee;margin-top:10px;font-weight:bold;'>Instrucción</div>"
                        f"<div style='color:#cbd5e1;'>Magnitud: <b>{mag:.2f}</b> · Target: <b>{target}</b></div>"
                    )
                except Exception:
                    sem_html = f"<div style='color:#cbd5e1;margin-top:8px;'>Prompt: {user_prompt[:80]}…</div>"

            return f"""
            <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:12px;font-size:12px;">
                <div style="color:#22d3ee;font-weight:bold;margin-bottom:8px;">Modo mejorar — TOP (LongCat + Lanczos)</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;color:#cbd5e1;">
                    <div><span style="color:#64748b;">Objetivo resolución:</span> <b>{tier_label}</b></div>
                    <div><span style="color:#64748b;">Modo:</span> <b>{style_label}</b></div>
                    <div><span style="color:#64748b;">Motor mejora:</span> <b>TOP sin rejilla</b></div>
                    <div><span style="color:#64748b;">LongCat realismo:</span> <b>{'sí' if longcat_on else 'no (solo clásico)'}</b></div>
                </div>
                {hybrid_note}
                {sem_html}
                {plan_html}
                <div style="color:#64748b;margin-top:8px;font-size:10px;">
                    Tier = máximo px; la fuerza de cada paso se adapta a la foto.
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
        axes = nlp.get_axis_scores(translated) if hasattr(nlp, "get_axis_scores") else {}
        structural = float(axes.get("structural", 0.0))
        if hasattr(nlp, "is_structural_dominant"):
            high_structural = nlp.is_structural_dominant(translated)
        else:
            struct_thresh = getattr(nlp, "STRUCTURAL_SIGNAL_THRESHOLD", 0.06)
            high_structural = structural > struct_thresh
        structural_bias = nlp.get_structural_bias(translated) if hasattr(nlp, "get_structural_bias") else "neutral"
        analysis = {
            "magnitude": mag,
            "mask_target": target,
            "is_global": high_structural or (target == "subject" and mag < 0.45),
            "quality_only": False,
            "high_structural": high_structural,
        }
        params = mgr.auto_detect_params(analysis, engine_val)
        longcat_mode = "LongCat Full (auto)" if (engine_val == "imagine" and mag >= 0.62) else (
            "LongCat Turbo" if engine_val in ("imagine", "longcat") else engine_val
        )
        force_global = high_structural or (mag > 0.6 and target in ("subject", "clothes", "face"))
        edit_mode = "Global" if force_global else (
            "Máscara ropa" if target in ("clothes", "subject") and mag >= 0.45 else "Máscara/parcial"
        )

        return f"""
        <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:12px;font-size:12px;">
            <div style="color:#22d3ee;font-weight:bold;margin-bottom:8px;">Análisis local (sin IA pesada)</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;color:#cbd5e1;">
                <div><span style="color:#64748b;">Magnitud:</span> <b>{mag:.2f}</b></div>
                <div><span style="color:#64748b;">Estructural:</span> <b>{structural:.2f}</b></div>
                <div><span style="color:#64748b;">Bias:</span> <b>{structural_bias}</b></div>
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
                LongCat mantiene parecido vía foto ref. Full (mag≥0.68) para instrucciones fuertes. Sin FacePreserver ni rewriter LLM.
            </div>
        </div>
        """
    except Exception as e:
        return f"<div style='color:#f87171;padding:8px;'>Error preview: {e}</div>"


def on_quality_mode_change(enabled):
    """Modo mejorar: muestra opciones al instante (sin tocar el dropdown)."""
    if enabled:
        return (
            gr.update(
                placeholder=(
                    "Vacío = solo nitidez/resolución. "
                    "«ultra realista» = toda la foto. "
                    "«convierte el fondo en…» = cambia fondo (usa Editar + mejorar)."
                ),
            ),
            gr.update(visible=True),
        )
    return (
        gr.update(
            placeholder="Ej: desnuda y descalza, que estén bailando, cambia el fondo...",
        ),
        gr.update(visible=False),
    )


def _drain_progress(updates, scope=SCOPE_IMG_EDITOR, timeout=0.35):
    """Lee progreso sin bloquear el worker de Gradio indefinidamente."""
    while True:
        if is_cancelled(scope):
            break
        try:
            item = updates.get(timeout=timeout)
        except queue.Empty:
            yield gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
            continue
        if item is None:
            break
        yield None, item, None, gr.skip(), gr.skip()


def on_cancel_img_editor():
    request_cancel(SCOPE_IMG_EDITOR, interrupt_comfy=True)
    return cancel_status_html(), *btn_idle()


def on_generate(
    img_data,
    p_text,
    engine_val,
    use_ai_val,
    enhance_val,
    denoise_val,
    upscale_val,
    quality_mode_val,
    enhance_tier_val,
    quality_style_val,
    quality_longcat_val,
    quality_preserve_faces_val,
):
    global _is_generating, _generate_lock

    clear_cancel(SCOPE_IMG_EDITOR)

    with _generate_lock:
        if _is_generating:
            yield None, "⚠️ Ya hay una transformación en proceso", None, *btn_idle()
            return

    quality_mode_val = bool(quality_mode_val)
    p_text = (p_text or "").strip()
    if not quality_mode_val and not p_text:
        yield None, "Escribe un prompt o activa «Mejorar imagen» (con o sin instrucción)", None, *btn_idle()
        return
    if img_data is None:
        yield None, "Sube una imagen", None, *btn_idle()
        return

    if isinstance(img_data, dict):
        img = img_data.get("background")
    elif isinstance(img_data, str):
        img = Image.open(img_data).convert("RGB")
    else:
        img = img_data

    if img is None or not isinstance(img, Image.Image):
        yield None, "Imagen inválida", None, *btn_idle()
        return

    with _generate_lock:
        if _is_generating:
            yield None, "⚠️ Ya hay una transformación en proceso", None, *btn_idle()
            return
        _is_generating = True

    updates = queue.Queue()
    result = {"image": None, "msg": "", "mask": None, "error": None}

    def progress_callback(prog):
        updates.put(_progress_html_from_prog(prog))

    def worker():
        global _is_generating
        try:
            mgr = get_img_editor_manager()
            if engine_val in ("imagine", "longcat", "longcat_full", "klein_base", "flux_q2", "flux_dev_abliterated"):
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if not get_flux_edit_comfy_client().is_available():
                    result["error"] = (
                        "<div style='color:#ef4444;text-align:center;padding:10px;'>"
                        "❌ ComfyUI no está activo. Inicia ComfyUI primero.</div>"
                    )
                    return

            if quality_mode_val and p_text:
                phase = "Mejorar + editar (híbrido)"
            elif quality_mode_val:
                phase = "Mejorando imagen"
            else:
                phase = "Analizando instrucción"
            progress_callback({"phase": phase, "progress": 0.02, "elapsed": 0})

            res_img, msg, mask_img = mgr.generate_intelligent(
                image=img,
                prompt=p_text,
                use_rewriter=False,
                use_semantic=use_ai_val,
                engine=engine_val,
                enhance_faces=enhance_val,
                lora_name=None,
                lora_strength=1.0,
                denoise=denoise_val if denoise_val > 0 else None,
                progress_callback=progress_callback,
                auto_upscale=upscale_val,
                quality_mode=quality_mode_val,
                enhance_tier=enhance_tier_val or "hd",
                quality_enhance_style=quality_style_val or "auto",
                quality_use_generative=bool(quality_longcat_val),
                quality_preserve_faces=bool(quality_preserve_faces_val),
                cancel_check=lambda: is_cancelled(SCOPE_IMG_EDITOR),
            )

            if is_cancelled(SCOPE_IMG_EDITOR):
                result["error"] = cancel_status_html()
                return

            if res_img:
                output_dir = get_img_editor_output_dir()
                ts = int(time.time())
                out_path = os.path.join(output_dir, f"edit_{ts}.png")
                res_img.save(out_path)
                print(f"[ImgEditor] Imagen guardada: {out_path}")
                result["image"] = res_img
                result["msg"] = (
                    f"<div style='text-align:center;color:#10b981;padding:10px;font-weight:bold;'>✅ {msg}</div>"
                    f"<div style='text-align:center;color:#64748b;font-size:11px;'>Guardado en {output_dir}</div>"
                )
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

    try:
        yield (
            None,
            build_generation_progress_html(phase="Iniciando", detail="Preparando edición…"),
            None,
            *btn_running(),
        )

        yield from _drain_progress(updates)

        if is_cancelled(SCOPE_IMG_EDITOR):
            yield None, cancel_status_html(), result.get("mask"), *btn_idle()
            return

        if result["error"]:
            yield None, result["error"], result.get("mask"), *btn_idle()
            return

        yield result["image"], result["msg"], result.get("mask"), *btn_idle()
    finally:
        clear_cancel(SCOPE_IMG_EDITOR)


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
                    label="🖼️ Mejorar imagen (upscale + realismo TOP)",
                    value=False,
                    info="Mejora calidad/resolución. Puedes añadir instrucción en el prompt (fondo, cuerpo, etc.) o dejarlo vacío.",
                )
                with gr.Column(visible=False) as quality_options:
                    gr.HTML(QUALITY_HELP_HTML)
                    enhance_tier = gr.Radio(
                        choices=[
                            ("HD — rápido (~1920px)", "hd"),
                            ("4K — equilibrado (~3840px)", "4k"),
                            ("8K — máximo (~7680px, lento)", "8k"),
                        ],
                        value="hd",
                        label="Tamaño de salida",
                    )
                    quality_style = gr.Radio(
                        choices=[
                            ("🔄 Automático — decide solo", "auto"),
                            ("📸 Solo nitidez/resolución (toda la foto)", "solo_mejora"),
                            ("✏️ Editar algo + mejorar (fondo, ropa…)", "hibrido"),
                        ],
                        value="auto",
                        label="¿Qué quieres hacer?",
                    )
                    quality_longcat = gr.Checkbox(
                        label="Textura fotorrealista (LongCat suave)",
                        value=True,
                        info="Más realismo en piel y detalle. Desmarcar = más rápido.",
                    )
                    quality_preserve_faces = gr.Checkbox(
                        label="Mantener caras originales",
                        value=True,
                        info="Recomendado si las caras salen raras tras el upscale.",
                    )

                prompt = gr.Textbox(
                    label="¿Qué quieres cambiar?",
                    placeholder="Ej: desnuda y descalza, que estén bailando, cambia el fondo...",
                    lines=3,
                    elem_classes=["prompt-box-img"]
                )

                with gr.Row():
                    gen_btn = gr.Button("✨ TRANSFORMAR", variant="primary", elem_classes=["btn-transform-main"], scale=3)
                    btn_cancel = gr.Button("⏹ CANCELAR", variant="stop", interactive=False, scale=1)
                    btn_preview = gr.Button("🔍 PREVIEW", size="sm", scale=1)

                with gr.Accordion("⚙️ Opciones Avanzadas", open=False):
                    gr.Markdown("*Defaults optimizados — sube foto, escribe instrucción y TRANSFORMAR.*")
                    use_ai = gr.Checkbox(
                        label="🧠 Análisis semántico local (auto params)",
                        value=True,
                        info="Magnitud/target/denoise automáticos. Sin rewriter LLM.",
                    )
                    with gr.Row():
                        denoise = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                            label="Fuerza de Edición (0 = Auto)",
                            info="0 = automático según magnitud. Turbo ignora denoise (siempre 1.0).",
                        )
                    engine = gr.Dropdown(
                        choices=[
                            ("✨ Grok Imagine (Turbo rápido; Full auto si mag≥0.62)", "imagine"),
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
                        label="📐 Upscale 2× tras edición con instrucción",
                        value=False,
                        info="Solo para ediciones normales (con prompt). «Solo mejorar imagen» ya incluye upscale según nivel.",
                    )
                    enhance_faces = gr.Checkbox(
                        label="🌟 Mejorar textura facial (CodeFormer suave)",
                        value=True,
                        info="Activo por defecto. En modo calidad se refuerza automáticamente.",
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
        outputs=[prompt, quality_options],
        queue=False,
        show_progress=False,
    )

    gen_event = gen_btn.click(
        on_generate,
        [
            input_img, prompt, engine, use_ai, enhance_faces, denoise, upscale_auto,
            quality_mode, enhance_tier, quality_style, quality_longcat, quality_preserve_faces,
        ],
        [output_img, status, mask_preview, gen_btn, btn_cancel],
    )

    btn_cancel.click(
        fn=on_cancel_img_editor,
        outputs=[status, gen_btn, btn_cancel],
        queue=False,
    )
    btn_cancel.click(fn=None, cancels=[gen_event], queue=False)

    btn_preview.click(
        preview_semantic,
        [prompt, engine, quality_mode, enhance_tier, quality_style, quality_longcat, input_img],
        [status],
    )

    return {
        "input_img": input_img,
        "prompt": prompt,
        "quality_mode": quality_mode,
        "enhance_tier": enhance_tier,
        "quality_style": quality_style,
        "quality_longcat": quality_longcat,
        "quality_preserve_faces": quality_preserve_faces,
        "gen_btn": gen_btn,
        "btn_cancel": btn_cancel,
        "output_img": output_img,
        "status": status,
        "mask_preview": mask_preview,
        "enhance_faces": enhance_faces,
    }