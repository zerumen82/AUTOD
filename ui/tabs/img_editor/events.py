import os
import sys
import time
import gc
import traceback
import numpy as np
from PIL import Image

import gradio as gr

from roop.img_editor.img_editor_manager import get_img_editor_manager
from roop.output_paths import get_img_editor_output_dir
from . import state
from .selection_events import (
    on_analyze, on_new_image, on_point_select, on_select,
    use_result_as_input as _use_result_as_input,
)

RETRY_ATTEMPTS = 2


def _extract_painted_mask(editor_value, background):
    """Extract an optional alpha selection painted in Gradio's ImageEditor."""
    if not isinstance(editor_value, dict) or background is None:
        return None
    layers = editor_value.get("layers") or []
    if not layers:
        return None
    try:
        width, height = background.size
        combined = np.zeros((height, width), dtype=np.uint8)
        for layer in layers:
            if not isinstance(layer, Image.Image):
                continue
            rgba = layer.convert("RGBA")
            if rgba.size != (width, height):
                rgba = rgba.resize((width, height), Image.Resampling.NEAREST)
            combined = np.maximum(combined, np.asarray(rgba, dtype=np.uint8)[:, :, 3])
        coverage = float((combined > 0).mean()) if combined.size else 0.0
        if coverage <= 0.0001:
            return None
        print(f"[ImgEditor] Selección manual recibida (cobertura={coverage:.1%})", flush=True)
        return Image.fromarray(combined).convert("L")
    except Exception as exc:
        print(f"[ImgEditor] Selección manual omitida: {exc}", flush=True)
        return None


def open_output_folder():
    try:
        path = get_img_editor_output_dir()
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(path)
        else:
            import subprocess
            subprocess.Popen(["explorer", path])
    except Exception as e:
        print(f"[ImgEditor] Error al abrir carpeta: {e}")


def _cleanup_vram():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass
    gc.collect()


def on_free_memory():
    """v5.1137 — botón LIBERAR MEMORIA: purga a FULL.
    1) Caches ONNX del proceso principal (CodeFormer/ESRGAN/rewriter/CLIPSeg).
    2) Unload SINCRONO del modelo residente en ComfyUI (el cache de reutilización
       se sacrifica voluntariamente — esto es una purga manual, no automática).
    Devuelve un HTML de status con la VRAM libre antes/después para feedback visual."""
    def _vram_mb():
        try:
            import requests
            from roop.comfy_workflows import get_comfyui_url
            r = requests.get(f"{get_comfyui_url()}/system_stats", timeout=5).json()
            d = r.get("devices", [{}])[0]
            return int(d.get("vram_free", 0)) / (1024 * 1024), int(d.get("vram_total", 0)) / (1024 * 1024)
        except Exception:
            return None, None

    _before, _total = _vram_mb()

    # 1) Proceso principal (idempotente: recrea bajo demanda en el próximo uso)
    try:
        get_img_editor_manager()._release_local_gpu_before_comfy()
    except Exception as e:
        print(f"[ImgEditor] FreeMemory: release local falló: {e}", flush=True)

    # 2) ComfyUI: unload sincrono del modelo residente
    _unloaded = False
    try:
        import requests
        from roop.comfy_workflows import get_comfyui_url
        r = requests.post(f"{get_comfyui_url()}/autoauto/unload", json={}, timeout=60)
        if r.status_code == 200 and r.json().get("ok"):
            _unloaded = True
    except Exception as e:
        print(f"[ImgEditor] FreeMemory: unload ComfyUI falló: {e}", flush=True)

    _cleanup_vram()
    _after, _total = _vram_mb()

    if _after is not None:
        _msg = (
            f"Memoria liberada{' (modelo ComfyUI descargado)' if _unloaded else ''} — "
            f"VRAM: {_after:.0f}MB libres de {_total:.0f}MB"
            + (f" (antes: {_before:.0f}MB)" if _before is not None else "")
        )
    else:
        _msg = "Memoria liberada (VRAM del servidor no disponible)"
    print(f"[ImgEditor] FreeMemory: {_msg}", flush=True)
    return f"<div style='text-align:center; color:#22d3ee; padding:10px;'>{_msg}</div>"


def on_generate(img_data, p_text, engine_val, denoise_val,
                autoregressive, auto_iterations, quality_mode, enhance_tier,
                quality_preserve_faces=True,
                quality_deposter=True, quality_upscale=True,
                quality_color=True, quality_realism=True,
                progress=gr.Progress(), source="?", selection_state=None):
    with state._generation_lock:
        if state.is_generating:
            _txt = "Ya hay una transformacion en proceso"
            if state.generation_started_at:
                _txt += f" (lleva {int(time.time() - state.generation_started_at)} s)"
            print(f"[ImgEditor] Clic rechazado: {_txt}", flush=True)
            return None, _txt, None
        state.is_generating = True
        state.generation_started_at = time.time()

    def _end_generation():
        with state._generation_lock:
            state.is_generating = False
            state.generation_started_at = None

    # Las validaciones tempranas también viven dentro de try: una excepción
    # aquí (p.ej. Image.open de una ruta temporal borrada) dejaba
    # is_generating clavado en True y cada edicion posterior moria en
    # silencio con "ya hay una transformacion".
    try:
        p_text = (p_text or "").strip()
        # Primer print del handler: si la consola no muestra esta línea, el
        # clic no llegó a Python (cola/conexión de Gradio) y el problema es
        # de despacho, no del pipeline.
        print(
            f"[ImgEditor] Solicitud recibida prompt={p_text[:60]!r} "
            f"engine={engine_val} calidad={bool(quality_mode)} "
            f"origen={source}",
            flush=True,
        )
        progress(0, desc="Preparando")
        if not p_text and not quality_mode:
            _end_generation()
            return None, "Escribe un prompt", None
        if img_data is None:
            _end_generation()
            return None, "Sube una imagen", None

        if isinstance(img_data, dict):
            img = img_data.get("background")
        elif isinstance(img_data, str):
            img = Image.open(img_data).convert("RGB")
        else:
            img = img_data

        if img is None:
            _end_generation()
            return None, "Imagen invalida", None
        if not isinstance(img, Image.Image):
            _end_generation()
            return None, "Formato de imagen no soportado", None

        manual_mask = _extract_painted_mask(img_data, img)
        # ImageEditor entrega un fondo RGBA. El gestor y FaceAnalysis reciben RGB;
        # conservar alpha aquí haría que el detector omitiera las caras antes del
        # preflight, aunque la selección manual fuese válida.
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = img.copy()

        # Selección visual explícita (H-EDIT-SELECT): gana a la inferencia
        # textual. Sin selección, el flujo automático se conserva intacto.
        selection_analysis = None
        selected_region_ids = None
        print(f"[ImgEditor] DEBUG selection_state type={type(selection_state)}", flush=True)
        if isinstance(selection_state, state.SelectionSession):
            selection_analysis, selected_region_ids = selection_state.snapshot()
            print(f"[ImgEditor] DEBUG snapshot: analysis={selection_analysis is not None}, ids={selected_region_ids}", flush=True)
        else:
            print(f"[ImgEditor] DEBUG selection_state is NOT SelectionSession: {selection_state}", flush=True)
        selection_kwargs = {}
        if selected_region_ids:
            from roop.img_editor.element_selection import validate_selection
            valid, reason = validate_selection(img, selection_analysis, selected_region_ids)
            print(f"[ImgEditor] DEBUG validate_selection: valid={valid}, reason={reason}", flush=True)
            if not valid:
                _end_generation()
                return None, f"Selección inválida: {reason}", None
            selection_kwargs = {"selection_analysis": selection_analysis,
                                "selected_region_ids": selected_region_ids}
    except Exception as e:
        traceback.print_exc()
        _end_generation()
        return None, f"Error: {str(e)}", None

    def cancel_check():
        return not state.is_generating

    # Barra de progreso en la propia UI: sin esto, la fase local y la espera
    # de ComfyUI no pintan NADA hasta el final y "arrancó pero trabaja en
    # silencio" es indistinguible de "no arrancó". Acepta los dicts del
    # manager (phase/progress/detail) y los de ComfyUI (phase/step/total).
    # Un fallo aquí jamás debe romper la edición.
    def _ui_progress(prog):
        try:
            frac = max(0.0, min(1.0, float(prog.get("progress") or 0.0)))
            desc = str(prog.get("phase") or "")
            detail = prog.get("detail")
            if detail and str(detail) not in desc:
                desc = f"{desc}: {detail}" if desc else str(detail)
            step = int(prog.get("step") or 0)
            total = int(prog.get("total") or 0)
            if step and total:
                desc = f"{desc} (paso {step}/{total})".strip()
            progress(frac, desc=desc or "Trabajando")
        except Exception:
            pass

    try:
        manager = get_img_editor_manager()
        try:
            iterations = max(1, min(int(auto_iterations), 20)) if autoregressive else 1
        except (ValueError, TypeError):
            iterations = 1
        if selected_region_ids:
            iterations = 1
        current_img = img
        final_mask = None
        last_msg = ""

        for i in range(iterations):
            if cancel_check():
                return None, "Cancelado por el usuario", final_mask

            print(f"[ImgEditor] Iteracion autoregresiva {i+1}/{iterations}")

            best_hair_candidate = None
            attempts = 1 if selected_region_ids else RETRY_ATTEMPTS
            for attempt in range(attempts):
                res_img, msg, mask_img = manager.generate_intelligent(
                    image=current_img, prompt=p_text,
                    engine=engine_val,
                    mask_image=manual_mask,
                    mask_mode="manual" if manual_mask is not None else "global",
                    enhance_faces=True,
                    quality_mode=quality_mode,
                    enhance_tier=enhance_tier,
                    quality_preserve_faces=quality_preserve_faces,
                    quality_steps=(
                        {
                            "deposter": bool(quality_deposter),
                            "upscale": bool(quality_upscale),
                            "color": bool(quality_color),
                            "realism": bool(quality_realism),
                        }
                        if quality_mode else None
                    ),
                    # Rewriter real: el manager lo activa solo si mag/pose/structural/recolor
                    # lo piden (sin hardcode de intención). use_rewriter=True = permitir.
                    use_rewriter=True,
                    lora_name=None,
                    lora_strength=1.0,
                    denoise=denoise_val if denoise_val and denoise_val > 0 else None,
                    auto_upscale=False,
                    cancel_check=cancel_check,
                    progress_callback=_ui_progress,
                    **selection_kwargs,
                )
                compliance = getattr(manager, "_last_instruction_compliance", None) or {}
                hair_unmet = (res_img is not None
                              and compliance.get("source") == "hair_mask"
                              and compliance.get("compliant") is False)
                if hair_unmet:
                    score = float(compliance.get("hair_after", 1.0))
                    if best_hair_candidate is None or score < best_hair_candidate[4]:
                        best_hair_candidate = (res_img, msg, mask_img, compliance, score)
                    if attempt + 1 < attempts:
                        # La receta anclada conserva el pelo que el modelo
                        # todavía ve, así que remuestrear la misma foto solo
                        # afeita otro parche. La siguiente muestra parte del
                        # resultado medido: con menos pelo visible la
                        # eliminación se acumula, y el pegado sigue midiendo
                        # el cambio sobre esa nueva fuente.
                        print(
                            "[ImgEditor] Pelo residual medido; la siguiente "
                            "muestra completa la eliminación sobre el resultado",
                            flush=True,
                        )
                        current_img = best_hair_candidate[0]
                        continue
                if (res_img is not None and compliance.get("compliant") is False
                        and not hair_unmet):
                    # Un VLM pequeño puede no resolver un sujeto lejano o una
                    # composición con varias zonas. Repetir el MISMO prompt,
                    # modelo y parámetros no es una alternativa independiente:
                    # duplica minutos de GPU sin nueva evidencia. El veredicto
                    # se conserva como aviso en el resultado, pero el reintento
                    # queda reservado a fallos reales de generación.
                    print(
                        "[ImgEditor] Verificación visual no confirmada; "
                        "se conserva el candidato y no se repite ciegamente.",
                        flush=True,
                    )
                if res_img is not None:
                    break
                if "no se reintenta" in str(msg).lower() and best_hair_candidate is None:
                    return None, msg, final_mask
                if attempt == 0:
                    print(f"[ImgEditor] Reintentando iteracion {i+1}...")

            if best_hair_candidate is not None and (
                    res_img is None or (
                        compliance.get("source") == "hair_mask"
                        and compliance.get("compliant") is False
                        and best_hair_candidate[4] < float(compliance.get("hair_after", 1.0))
                    )):
                res_img, msg, mask_img, best_compliance, _ = best_hair_candidate
                manager._last_instruction_compliance = best_compliance
                print("[ImgEditor] Se conserva la muestra con menos pelo residual", flush=True)

            if res_img is None:
                return None, f"Error iteracion {i+1}: {msg}", final_mask

            if mask_img is not None:
                final_mask = mask_img
            last_msg = msg
            current_img = res_img

            _cleanup_vram()
            if str(msg).startswith("Omitida:"):
                # La imagen sin cambios es una advertencia, no una edición
                # terminada. Tampoco repitas una localización imposible en las
                # iteraciones autorregresivas ni guardes un falso resultado.
                try:
                    manager._release_local_gpu_before_comfy()
                except Exception:
                    pass
                return current_img, msg, final_mask

        output_dir = get_img_editor_output_dir()
        ts = int(time.time())
        out_path = os.path.join(output_dir, f"edit_{ts}.png")
        current_img.save(out_path)

        state.last_result_image = current_img
        print(f"[ImgEditor] Imagen guardada: {out_path}")

        # v5.1136 — liberación POST-petición: CodeFormer/ESRGAN/rewriter del
        # proceso principal quedaban residentes entre edits y el pool de VRAM
        # menguaba run a run (4987MB → 1624MB libres en el log 07:17) → la 2ª/3ª
        # edición pagaba offload dance al cargar ComfyUI. Liberando al FINAL, cada
        # petición nueva arranca con el pool máximo y el modelo ComfyUI residente
        # (que SÍ conservamos para reutilización rápida) tiene holgura.
        try:
            manager._release_local_gpu_before_comfy()
        except Exception as _e:
            print(f"[ImgEditor] Release post-petición omitido: {_e}", flush=True)

        label = f"{last_msg}"
        if iterations > 1:
            label += f" ({iterations} iteraciones autoregresivas)"
        label += f" - Guardado en {output_dir}"

        return current_img, label, final_mask

    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {str(e)}", None
    finally:
        _end_generation()


def _show_quality_options(enabled):
    return gr.update(visible=bool(enabled))


def _on_generate_desktop(*args, **kwargs):
    """Botón ✨ TRANSFORMAR del escritorio; el origen va al log."""
    print(f"[ImgEditor] DEBUG _on_generate_desktop: len(args)={len(args)}, kwargs={list(kwargs.keys())}", flush=True)
    if len(args) == 14:
        kwargs["selection_state"] = args[-1]
        args = args[:-1]
    kwargs.setdefault("source", "desktop")
    return on_generate(*args, **kwargs)


def _on_generate_api_select(*args, **kwargs):
    """Endpoint /edit_select conservado para clientes de 13 controles."""
    kwargs.setdefault("source", "api/edit_select")
    return on_generate(*args, **kwargs)


def _on_generate_api_options(*args, **kwargs):
    """Endpoint /edit_options (app Android)."""
    kwargs.setdefault("source", "api/edit_options")
    return on_generate(*args, **kwargs)


def _on_generate_api_legacy(*args, **kwargs):
    """Endpoint /edit (apps antiguas instaladas)."""
    kwargs.setdefault("source", "api/edit")
    return on_generate(*args, **kwargs)


def wire_events(ui_comp):
    desktop_inputs = [
        ui_comp["input_img"], ui_comp["prompt"],
        ui_comp["engine"], ui_comp["denoise"],
        ui_comp["autoregressive"], ui_comp["auto_iterations"],
        ui_comp["quality_mode"], ui_comp["enhance_tier"],
        ui_comp["quality_preserve_faces"],
        ui_comp["quality_deposter"], ui_comp["quality_upscale"],
        ui_comp["quality_color"], ui_comp["quality_realism"],
    ]
    outputs = [ui_comp["output_img"], ui_comp["status"], ui_comp["mask_preview"]]

    # Concurrencia ilimitada en TODOS los listeners de on_generate. Medido
    # 2026-10-16 (gradio 6.13.0): con la concurrencia por defecto (1) un
    # segundo envío mientras corre quedaba ENCOLADO y se ejecutaba tras el
    # primero sin rechazo — el «procesado dos veces» de 18:56/19:00 (dos
    # pasadas de ~3 min para un solo pedido). Con None el envío en caliente
    # corre en paralelo y la guarda de on_generate lo rechaza en voz alta
    # («Ya hay una transformacion en proceso»). Los listeners de API del
    # móvil se comportan igual: un envío mientras corre responde, no cola.
    def _bind_generate(btn, inputs, fn, api_name):
        try:
            return btn.click(
                fn, inputs, outputs,
                api_name=api_name, concurrency_limit=None,
            )
        except TypeError:
            # Gradio 3.x no acepta concurrency_limit y pasarlo rompe el
            # cableado de toda la pestaña; ahí la guarda del handler es la
            # única protección.
            return btn.click(fn, inputs, outputs, api_name=api_name)

    if "selection_state" in ui_comp:
        _bind_generate(
            ui_comp["gen_btn"], desktop_inputs + [ui_comp["selection_state"]],
            _on_generate_desktop, "edit_region",
        )
        _bind_generate(
            ui_comp["legacy_select_btn"], desktop_inputs,
            _on_generate_api_select, "edit_select",
        )
    else:
        _bind_generate(
            ui_comp["gen_btn"], desktop_inputs, _on_generate_desktop, "edit_select",
        )
    # Nine arguments, ending in Restaurar caras. The phone app posts this list.
    # The four step checkboxes stay off this endpoint so that call still works;
    # missing steps mean every finish step stays on.
    _bind_generate(
        ui_comp["legacy_options_btn"], desktop_inputs[:9],
        _on_generate_api_options, "edit_options",
    )
    # Eight arguments, without Restaurar caras. Older installed apps use this.
    _bind_generate(
        ui_comp["legacy_api_btn"], desktop_inputs[:8],
        _on_generate_api_legacy, "edit",
    )
    ui_comp["quality_mode"].change(
        fn=_show_quality_options,
        inputs=[ui_comp["quality_mode"]],
        outputs=[ui_comp["quality_options"]],
    )

    _use_outputs = [ui_comp["input_img"]]
    if "overlay_img" in ui_comp:
        _use_outputs += [
            ui_comp["overlay_img"], ui_comp["region_choices"],
            ui_comp["selection_state"], ui_comp["status"],
        ]
        ui_comp["bt_use_as_input"].click(
            fn=_use_result_as_input,
            inputs=[ui_comp["output_img"], ui_comp["selection_state"]],
            outputs=_use_outputs,
            queue=False, api_name="reuse_editor_result",
        )
    else:
        ui_comp["bt_use_as_input"].click(
            fn=lambda x: x,
            inputs=[ui_comp["output_img"]],
            outputs=_use_outputs,
        )

    # --- Selector visual (H-EDIT-SELECT) ---------------------------------
    # Analizar corre en CPU (no toca la GPU); la selección es múltiple y se
    # aplica como una sola región focal en la generación. Se cablea solo si la
    # UI incluye los componentes nuevos (las pruebas antiguas no los pasan).
    if "analyze_btn" in ui_comp and "overlay_img" in ui_comp and "region_choices" in ui_comp:
        ui_comp["analyze_btn"].click(
            fn=on_analyze,
            inputs=[ui_comp["input_img"], ui_comp["selection_state"]],
            outputs=[
                ui_comp["overlay_img"], ui_comp["region_choices"],
                ui_comp["selection_state"], ui_comp["status"],
            ],
            concurrency_limit=None, api_name="analyze_regions",
        )
        ui_comp["region_choices"].input(
            fn=on_select,
            inputs=[ui_comp["region_choices"], ui_comp["input_img"], ui_comp["selection_state"]],
            outputs=[ui_comp["overlay_img"], ui_comp["selection_state"], ui_comp["status"]],
            queue=False, api_name="select_regions",
        )
        ui_comp["overlay_img"].select(
            fn=on_point_select,
            inputs=[ui_comp["input_img"], ui_comp["selection_state"]],
            outputs=[ui_comp["overlay_img"], ui_comp["region_choices"],
                     ui_comp["selection_state"], ui_comp["status"]],
            queue=False, api_name=False,
        )
        ui_comp["input_img"].change(
            fn=on_new_image,
            inputs=[ui_comp["input_img"], ui_comp["selection_state"]],
            outputs=[
                ui_comp["overlay_img"], ui_comp["region_choices"],
                ui_comp["selection_state"], ui_comp["status"],
            ],
            queue=False, api_name="reset_regions",
        )

    ui_comp["bt_open_folder"].click(
        fn=open_output_folder,
        inputs=[],
        outputs=[],
    )

    # La UI restaurada no muestra la purga manual. Mantener el handler opcional
    # permite abrir tanto esa versión como la variante que sí incluía el botón.
    free_memory_button = ui_comp.get("bt_free_memory")
    if free_memory_button is not None:
        free_memory_button.click(
            fn=on_free_memory,
            inputs=[],
            outputs=[ui_comp["status"]],
        )

    print("[ImgEditor] Eventos conectados (versátil: color/pose/add-remove, rewriter mag-driven)")
