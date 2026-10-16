import os
import os
import queue
import threading
import time
import gradio as gr

from roop.img_editor.flux_gen_comfy_client import get_flux_gen_client
from roop.img_editor.gen_prompt_modifiers import get_dropdown_choices, get_compatible_dropdown_choices, preview_modifiers
from roop.img_editor.comfy_progress import build_generation_progress_html, format_duration
from roop.img_editor.gen_prompt_config import get_generate_ui_config
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
from . import state


def on_cancel_generation():
    request_cancel(SCOPE_GENERATION, interrupt_comfy=True)
    return cancel_status_html(), *btn_idle()


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


def set_orientation(orientation: str):
    if orientation == "Vertical (Retrato)":
        return 768, 1152
    return 1152, 768


def _build_modifiers(
    image_style: str,
    shot_framing: str,
    lighting: str = "auto",
    skin_finish: str = "auto",
    *,
    color_grade: str = "auto",
    use_rewriter: bool = False,
) -> dict:
    return {
        "image_type": image_style or "auto",
        "lighting": lighting or "auto",
        "skin_finish": skin_finish or "auto",
        "framing": shot_framing or "auto",
        "color_grade": color_grade or "auto",
        "use_rewriter": use_rewriter,
    }


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


def _post_generation_unload():
    """Fase 3 — libera la VRAM del servidor tras generar.

    Usa el endpoint síncrono /autoauto/unload del custom node (POST /free de
    ComfyUI es lazy y no descarga con la cola vacía). Si la cola está ocupada
    (otra petaición en marcha) no se descarga nada.
    """
    try:
        from roop.comfy_workflows import get_comfyui_url
        from roop.img_editor.flux_edit_comfy_client import _queue_is_idle, _sync_unload
        url = get_comfyui_url()
        if not _queue_is_idle(url):
            print("[Gen] Unload omitido: la cola de ComfyUI está ocupada")
            return
        _sync_unload(url, "generate")
    except Exception as e:
        print(f"[Gen] Unload post-generación omitido: {e}")


def _route_is_editor(route_val, prompt, ref_image) -> bool:
    """Decide la ruta cuando hay imagen de referencia.

    'editor' / 'img2img' = decisión del usuario; 'auto' = la detecta
    route_prefers_editor (umbrales en config/generation_prompt.json).
    Sin imagen nunca hay ruta de edición.
    """
    if ref_image is None:
        return False
    r = (route_val or "auto").strip().lower()
    if r == "editor":
        return True
    if r == "img2img":
        return False
    try:
        from roop.img_editor.gen_semantic import route_prefers_editor
        return bool(route_prefers_editor((prompt or "").strip()))
    except Exception as e:
        print(f"[Gen] Detección de edición no disponible ({e}); ruta img2img")
        return False


def on_generate_image(
    prompt, width, height, engine_val, image_style, shot_framing, lighting,
    skin_finish, enhance_quality, use_rewriter,
    color_grade="auto", seed=0, steps=0, cfg_scale=0.0, negative_prompt="",
    # Ruta por defecto img2img (config generate_ui.default_route): Generate y
    # el Editor son paralelos e independientes; "editor" es elección manual.
    ref_image=None, route="img2img", enhance_tier="hd", samples=1,
    nsfw_content=None, lora_choice=None,
):
    # Los primeros 10 parámetros son el contrato legacy de la APK
    # (api_name="generate_image"); el resto (modificadores, avanzados,
    # imagen de referencia, NSFW, LoRA) llega solo por generate_image_ui.
    if nsfw_content is None:
        nsfw_content = bool(get_generate_ui_config().get("nsfw_content_default", False))
    if lora_choice is None:
        lora_choice = "Auto"
    with state._generation_lock:
        if state.is_generating:
            yield None, "<div style='text-align:center;color:#ef4444;padding:10px;'>⚠️ Ya hay una generación en curso</div>", *gr.skip(), *gr.skip()
            return
        state.is_generating = True

    worker_started = False
    try:
        clear_cancel(SCOPE_GENERATION)
        client = get_flux_gen_client()
        if not client.is_available():
            yield None, "<div style='text-align:center;color:#ef4444;padding:10px;'>❌ ComfyUI no está activo.</div>", *btn_idle()
            return

        updates = queue.Queue()
        result = {"image": None, "status": "", "error": None}
        use_editor_route = _route_is_editor(route, prompt, ref_image)

        def progress_callback(prog):
            updates.put(_progress_html_from_prog(prog))

        def worker():
            try:
                if use_editor_route:
                    # Ruta de edición: la foto de referencia pasa al Image
                    # Editor (máscara/conceptos/verificación del editor).
                    from roop.img_editor.gen_prompt_config import get_route_editor_config
                    from roop.img_editor.img_editor_manager import get_img_editor_manager

                    rcfg = get_route_editor_config()
                    ed_engine = rcfg["engines"].get(engine_val) or rcfg["default_engine"]
                    progress_callback({
                        "phase": "Image Editor",
                        "detail": f"Ruta de edición · motor {ed_engine}",
                        "progress": 0.02,
                        "elapsed": 0,
                    })
                    manager = get_img_editor_manager()
                    ed_img, ed_msg, _ed_mask = manager.generate_intelligent(
                        image=ref_image,
                        prompt=(prompt or "").strip(),
                        negative_prompt=(negative_prompt or "").strip(),
                        engine=ed_engine,
                        seed=int(seed) if seed else None,
                        num_inference_steps=int(steps) if steps else None,
                        guidance_scale=float(cfg_scale) if cfg_scale else None,
                        use_rewriter=bool(use_rewriter),
                        mask_mode="global",
                        enhance_faces=True,
                        quality_mode=False,
                        enhance_tier=enhance_tier or "hd",
                        auto_upscale=False,
                        progress_callback=progress_callback,
                        cancel_check=lambda: is_cancelled(SCOPE_GENERATION),
                    )
                    if is_cancelled(SCOPE_GENERATION):
                        result["error"] = "Cancelado"
                        return
                    if ed_img is None:
                        result["error"] = f"❌ {ed_msg}"
                        return
                    output_dir = get_generation_output_dir()
                    out_path = os.path.join(output_dir, f"gen_{int(time.time())}.png")
                    ed_img.save(out_path)
                    print(f"[Gen] Editado en el Editor y guardado: {out_path}")
                    state.last_result_image = ed_img
                    result["image"] = ed_img
                    result["status"] = (
                        "<div style='text-align:center;color:#a855f7;padding:6px;'>"
                        f"✏️ Editado en Image Editor — {ed_msg}</div>"
                        f"<div style='text-align:center;color:#64748b;font-size:11px;'>Guardado en {output_dir}</div>"
                    )
                    return

                if not engine_val:
                    result["error"] = "❌ No hay un modelo normal disponible. Elige un modelo instalado."
                    return
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
                        color_grade=color_grade,
                        use_rewriter=use_rewriter,
                    ),
                    enhance_quality=bool(enhance_quality),
                    seed=int(seed) if seed else None,
                    steps=int(steps) if steps else None,
                    guidance_scale=float(cfg_scale) if cfg_scale else None,
                    negative_prompt=(negative_prompt or "").strip(),
                    init_image=ref_image,
                    progress_callback=progress_callback,
                    cancel_check=lambda: is_cancelled(SCOPE_GENERATION),
                    enhance_tier=enhance_tier or "hd",
                    batch_size=int(samples or 1),
                    inject_nsfw=bool(nsfw_content),
                    lora_name=(lora_choice or "Auto"),
                )

                if is_cancelled(SCOPE_GENERATION):
                    result["error"] = "Cancelado"
                    return

                if res:
                    output_dir = get_generation_output_dir()
                    ts = int(time.time())
                    out_path = os.path.join(output_dir, f"gen_{ts}.png")
                    res.image.save(out_path)
                    extras = list(getattr(res, "extra_images", None) or [])
                    for i, im in enumerate(extras, start=2):
                        try:
                            im.save(os.path.join(output_dir, f"gen_{ts}_{i}.png"))
                        except Exception as e:
                            print(f"[Gen] No se pudo guardar la muestra {i}: {e}")
                    print(f"[Gen] Imagen guardada: {out_path}")
                    result["image"] = res.image
                    status = _success_status_html(res)
                    status += f"<div style='text-align:center;color:#64748b;font-size:11px;'>Guardado en {output_dir}</div>"
                    if extras:
                        status += (
                            "<div style='text-align:center;color:#22d3ee;font-size:11px;'>"
                            f"{1 + len(extras)} muestras guardadas (gen_{ts}*.png)</div>"
                        )
                    result["status"] = status
                    state.last_result_image = res.image
                else:
                    result["error"] = f"❌ {msg}"
            except Exception as e:
                import traceback
                traceback.print_exc()
                result["error"] = f"❌ Error: {e}"
            finally:
                _post_generation_unload()
                # El worker es dueño de las banderas: limpiarlas con el hilo
                # vivo reviviría el trabajo cancelado (job zombie solapado
                # con la siguiente pulsación).
                clear_cancel(SCOPE_GENERATION)
                state.is_generating = False
                updates.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        worker_started = True

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
            if not worker_started:
                clear_cancel(SCOPE_GENERATION)
    finally:
        if not worker_started:
            state.is_generating = False


def wire_events(ui_comp):
    def handle_shape(orient):
        w, h = set_orientation(orient)
        return gr.update(value=w), gr.update(value=h)

    ui_comp["orientation"].change(
        fn=handle_shape,
        inputs=[ui_comp["orientation"]],
        outputs=[ui_comp["width"], ui_comp["height"]],
    )

    def _on_style_change(style, light, frame, skin, grade):
        l_choices, l_safe = get_compatible_dropdown_choices(style, "lighting", light)
        f_choices, f_safe = get_compatible_dropdown_choices(style, "framing", frame)
        preview = preview_modifiers(style, l_safe, skin, f_safe, grade or "auto")
        return gr.update(choices=l_choices, value=l_safe), gr.update(choices=f_choices, value=f_safe), preview

    def _on_modifiers_change(style, light, frame, skin, grade):
        return preview_modifiers(style, light, skin, frame, grade or "auto")

    _modifier_inputs = [
        ui_comp["image_style"], ui_comp["lighting"], ui_comp["shot_framing"],
        ui_comp["skin_finish"], ui_comp["color_grade"],
    ]

    ui_comp["image_style"].change(
        fn=_on_style_change,
        inputs=_modifier_inputs,
        outputs=[ui_comp["lighting"], ui_comp["shot_framing"], ui_comp["modifier_preview"]],
    )
    for _comp in ("lighting", "shot_framing", "skin_finish", "color_grade"):
        ui_comp[_comp].change(
            fn=_on_modifiers_change,
            inputs=_modifier_inputs,
            outputs=[ui_comp["modifier_preview"]],
        )

    ui_comp["bt_open_folder"].click(fn=open_generation_folder)

    _legacy_inputs = [
        ui_comp["prompt"], ui_comp["width"], ui_comp["height"], ui_comp["engine_model"],
        ui_comp["image_style"], ui_comp["shot_framing"], ui_comp["lighting"],
        ui_comp["skin_finish"], ui_comp["enhance_quality"], ui_comp["use_rewriter"],
    ]
    _outputs = [ui_comp["output_img"], ui_comp["status_html"], ui_comp["gen_btn"], ui_comp["btn_cancel"]]

    gen_event = ui_comp["gen_btn"].click(
        fn=on_generate_image,
        inputs=_legacy_inputs + [
            ui_comp["color_grade"], ui_comp["seed"], ui_comp["steps"],
            ui_comp["cfg_scale"], ui_comp["negative_prompt"],
            ui_comp["ref_image"], ui_comp["route"],
            ui_comp["enhance_tier"], ui_comp["samples"],
            ui_comp["nsfw_content"],
            ui_comp["lora_choice"],
        ],
        outputs=_outputs,
        api_name="generate_image_ui",
    )

    # Contrato legacy de la APK: mismos 10 argumentos posicionales de siempre.
    gen_event_legacy = ui_comp["api_legacy_btn"].click(
        fn=on_generate_image,
        inputs=_legacy_inputs,
        outputs=_outputs,
        api_name="generate_image",
    )

    ui_comp["btn_cancel"].click(
        fn=on_cancel_generation,
        outputs=[ui_comp["status_html"], ui_comp["gen_btn"], ui_comp["btn_cancel"]],
        queue=False,
    )
    ui_comp["btn_cancel"].click(
        fn=None, cancels=[gen_event, gen_event_legacy], queue=False,
    )

    print("[GenTab] Eventos conectados (modo modular, sin censura, máxima calidad)")


def wire_editor_bridge(gen_comp, edit_comp):
    """Puente Generate → Image Editor: carga el resultado como entrada del Editor."""

    def _send_to_editor(image):
        ok_html = (
            "<div style='text-align:center; color:#22c55e; padding:10px;'>"
            "Imagen de Generate cargada en el Editor. Ve a la pestaña 🎨 Image Editor "
            "y escribe qué quieres cambiar.</div>"
        )
        if image is None:
            return gr.update(), (
                "<div style='text-align:center; color:#f59e0b; padding:10px;'>"
                "Genera una imagen primero y vuelve a pulsar el botón.</div>"
            )
        try:
            import numpy as np
            from PIL import Image
            if isinstance(image, np.ndarray):
                arr = image
                if arr.dtype != "uint8":
                    arr = (np.clip(arr, 0, 1) * 255).astype("uint8") if float(arr.max()) <= 1.0 else arr.astype("uint8")
                image = Image.fromarray(arr).convert("RGB")
            elif not isinstance(image, Image.Image):
                image = Image.fromarray(np.asarray(image)).convert("RGB")
        except Exception:
            return gr.update(), (
                "<div style='text-align:center; color:#ef4444; padding:10px;'>"
                "No se pudo pasar la imagen al Editor.</div>"
            )
        return image, ok_html

    gen_comp["bt_to_editor"].click(
        fn=_send_to_editor,
        inputs=[gen_comp["output_img"]],
        outputs=[edit_comp["input_img"], edit_comp["status"]],
    )
