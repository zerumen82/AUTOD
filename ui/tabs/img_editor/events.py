import os
import sys
import time
import gc
import traceback
from PIL import Image

import gradio as gr

from roop.img_editor.img_editor_manager import get_img_editor_manager
from roop.output_paths import get_img_editor_output_dir
from . import state

RETRY_ATTEMPTS = 2


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


def on_generate(img_data, p_text, engine_val, denoise_val,
                autoregressive, auto_iterations, quality_mode, enhance_tier):
    with state._generation_lock:
        if state.is_generating:
            return None, "Ya hay una transformacion en proceso", None
        state.is_generating = True

    p_text = (p_text or "").strip()
    if not p_text and not quality_mode:
        state.is_generating = False
        return None, "Escribe un prompt", None
    if img_data is None:
        state.is_generating = False
        return None, "Sube una imagen", None

    if isinstance(img_data, dict):
        img = img_data.get("background")
    elif isinstance(img_data, str):
        img = Image.open(img_data).convert("RGB")
    else:
        img = img_data

    if img is None:
        state.is_generating = False
        return None, "Imagen invalida", None
    if not isinstance(img, Image.Image):
        state.is_generating = False
        return None, "Formato de imagen no soportado", None

    cancel_check = lambda: not state.is_generating

    try:
        manager = get_img_editor_manager()
        try:
            iterations = max(1, min(int(auto_iterations), 20)) if autoregressive else 1
        except (ValueError, TypeError):
            iterations = 1
        current_img = img
        final_mask = None
        last_msg = ""

        for i in range(iterations):
            if cancel_check():
                return None, "Cancelado por el usuario", final_mask

            print(f"[ImgEditor] Iteracion autoregresiva {i+1}/{iterations}")

            for attempt in range(RETRY_ATTEMPTS):
                res_img, msg, mask_img = manager.generate_intelligent(
                    image=current_img, prompt=p_text,
                    engine=engine_val,
                    enhance_faces=True,
                    quality_mode=quality_mode,
                    enhance_tier=enhance_tier,
                    # Rewriter real: el manager lo activa solo si mag/pose/structural/recolor
                    # lo piden (sin hardcode de intención). use_rewriter=True = permitir.
                    use_rewriter=True,
                    lora_name=None,
                    lora_strength=1.0,
                    denoise=denoise_val if denoise_val and denoise_val > 0 else None,
                    auto_upscale=False,
                    cancel_check=cancel_check,
                )
                if res_img is not None:
                    break
                if attempt == 0:
                    print(f"[ImgEditor] Reintentando iteracion {i+1}...")

            if res_img is None:
                return None, f"Error iteracion {i+1}: {msg}", final_mask

            if mask_img is not None:
                final_mask = mask_img
            last_msg = msg
            current_img = res_img

            _cleanup_vram()

        output_dir = get_img_editor_output_dir()
        ts = int(time.time())
        out_path = os.path.join(output_dir, f"edit_{ts}.png")
        current_img.save(out_path)

        state.last_result_image = current_img
        print(f"[ImgEditor] Imagen guardada: {out_path}")

        label = f"{last_msg}"
        if iterations > 1:
            label += f" ({iterations} iteraciones autoregresivas)"
        label += f" - Guardado en {output_dir}"

        return current_img, label, final_mask

    except Exception as e:
        traceback.print_exc()
        return None, f"Error: {str(e)}", None
    finally:
        state.is_generating = False


def wire_events(ui_comp):
    ui_comp["gen_btn"].click(
        on_generate,
        [ui_comp["input_img"], ui_comp["prompt"],
         ui_comp["engine"], ui_comp["denoise"],
         ui_comp["autoregressive"], ui_comp["auto_iterations"],
         ui_comp["quality_mode"], ui_comp["enhance_tier"]],
        [ui_comp["output_img"], ui_comp["status"], ui_comp["mask_preview"]],
        concurrency_limit=None,
        api_name="edit",
    )

    ui_comp["bt_use_as_input"].click(
        fn=lambda x: x,
        inputs=[ui_comp["output_img"]],
        outputs=[ui_comp["input_img"]],
    )

    ui_comp["bt_open_folder"].click(
        fn=open_output_folder,
        inputs=[],
        outputs=[],
    )

    print("[ImgEditor] Eventos conectados (versátil: color/pose/add-remove, rewriter mag-driven)")
