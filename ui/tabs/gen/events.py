import os
import queue
import threading
import time
import gradio as gr

from roop.img_editor.flux_gen_comfy_client import get_flux_gen_client
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
    with state._generation_lock:
        if state.is_generating:
            yield None, "<div style='text-align:center;color:#ef4444;padding:10px;'>⚠️ Ya hay una generación en curso</div>", *gr.skip(), *gr.skip()
            return
        state.is_generating = True

    try:
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
                    print(f"[Gen] Imagen guardada: {out_path}")
                    result["image"] = res.image
                    status = _success_status_html(res)
                    status += f"<div style='text-align:center;color:#64748b;font-size:11px;'>Guardado en {output_dir}</div>"
                    result["status"] = status
                    state.last_result_image = res.image
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
    finally:
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

    def _on_style_change(style, light, frame, skin):
        l_choices, l_safe = get_compatible_dropdown_choices(style, "lighting", light)
        f_choices, f_safe = get_compatible_dropdown_choices(style, "framing", frame)
        preview = preview_modifiers(style, l_safe, skin, f_safe, "auto")
        return gr.update(choices=l_choices, value=l_safe), gr.update(choices=f_choices, value=f_safe), preview

    def _on_modifiers_change(style, light, frame, skin):
        return preview_modifiers(style, light, skin, frame, "auto")

    ui_comp["image_style"].change(
        fn=_on_style_change,
        inputs=[ui_comp["image_style"], ui_comp["lighting"], ui_comp["shot_framing"], ui_comp["skin_finish"]],
        outputs=[ui_comp["lighting"], ui_comp["shot_framing"], ui_comp["modifier_preview"]],
    )
    ui_comp["lighting"].change(
        fn=_on_modifiers_change,
        inputs=[ui_comp["image_style"], ui_comp["lighting"], ui_comp["shot_framing"], ui_comp["skin_finish"]],
        outputs=[ui_comp["modifier_preview"]],
    )
    ui_comp["shot_framing"].change(
        fn=_on_modifiers_change,
        inputs=[ui_comp["image_style"], ui_comp["lighting"], ui_comp["shot_framing"], ui_comp["skin_finish"]],
        outputs=[ui_comp["modifier_preview"]],
    )
    ui_comp["skin_finish"].change(
        fn=_on_modifiers_change,
        inputs=[ui_comp["image_style"], ui_comp["lighting"], ui_comp["shot_framing"], ui_comp["skin_finish"]],
        outputs=[ui_comp["modifier_preview"]],
    )

    ui_comp["bt_open_folder"].click(fn=open_generation_folder)

    gen_event = ui_comp["gen_btn"].click(
        fn=on_generate_image,
        inputs=[
            ui_comp["prompt"], ui_comp["width"], ui_comp["height"], ui_comp["engine_model"],
            ui_comp["image_style"], ui_comp["shot_framing"], ui_comp["lighting"],
            ui_comp["skin_finish"], ui_comp["enhance_quality"], ui_comp["use_rewriter"],
        ],
        outputs=[ui_comp["output_img"], ui_comp["status_html"], ui_comp["gen_btn"], ui_comp["btn_cancel"]],
        api_name="generate_image",
    )

    ui_comp["btn_cancel"].click(
        fn=on_cancel_generation,
        outputs=[ui_comp["status_html"], ui_comp["gen_btn"], ui_comp["btn_cancel"]],
        queue=False,
    )
    ui_comp["btn_cancel"].click(fn=None, cancels=[gen_event], queue=False)

    print("[GenTab] Eventos conectados (modo modular, sin censura, máxima calidad)")
