"""Eventos de la pestaña «✨ Mejorar fotos» (estilo local).

Reglas Gradio 6:
- No se escribe `.value` sobre un componente desde un handler; se devuelven
  valores y se declaran en `outputs`.
- `gr.Files` devuelve lista de rutas (str) u objetos con `.name`; se normaliza.
"""

import json
import tempfile
import time
from pathlib import Path

from PIL import Image

from roop.output_paths import get_enhance_output_dir

from . import logic as enh_logic
from .logic import process_batch, process_single_image


PRESET_DEFAULTS = {
    "Retrato": {
        "denoise": 30, "sharpen": 20, "recover": 70, "upscale": 2,
        "exposure": 1.0, "shadow": 0.8, "highlight": 1.2, "temperature": 45,
        "grain_strength": 10, "grain_size": 5,
    },
    "Paisaje": {
        "denoise": 50, "sharpen": 60, "recover": 60, "upscale": 4,
        "exposure": 1.0, "shadow": 0.5, "highlight": 1.2, "temperature": 55,
        "grain_strength": 15, "grain_size": 15,
    },
    "Nocturna": {
        "denoise": 80, "sharpen": 30, "recover": 50, "upscale": 2,
        "exposure": 1.5, "shadow": 1.5, "highlight": 0.5, "temperature": 30,
        "grain_strength": 25, "grain_size": 8,
    },
    "Producto": {
        "denoise": 40, "sharpen": 50, "recover": 80, "upscale": 4,
        "exposure": 1.0, "shadow": 0.8, "highlight": 1.0, "temperature": 50,
        "grain_strength": 15, "grain_size": 12,
    },
    "Archivo": {
        "denoise": 20, "sharpen": 20, "recover": 50, "upscale": 2,
        "exposure": 1.0, "shadow": 1.0, "highlight": 1.0, "temperature": 50,
        "grain_strength": 5, "grain_size": 10,
    },
}


def _resolve_path(item) -> str:
    """gr.Files puede dar str, dict o objeto con .name/.path."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        return item.get("path") or item.get("name") or ""
    return getattr(item, "path", None) or getattr(item, "name", None) or str(item)


def _normalize_files(raw) -> list:
    if not raw:
        return []
    if isinstance(raw, (list, tuple)):
        return [_resolve_path(x) for x in raw if _resolve_path(x)]
    single = _resolve_path(raw)
    return [single] if single else []


def wire_events(ui_components):
    """Conecta todos los eventos de la pestaña de mejora de fotos."""

    input_files = ui_components["input_files"]
    input_folder = ui_components["input_folder"]
    queue_table = ui_components["queue_table"]
    preview_slider = ui_components["preview_slider"]
    btn_preview = ui_components["btn_preview"]
    preview_status = ui_components["preview_status"]
    denoise_slider = ui_components["denoise_slider"]
    sharpen_slider = ui_components["sharpen_slider"]
    recover_slider = ui_components["recover_slider"]
    upscale_factor = ui_components["upscale_factor"]
    exposure_slider = ui_components["exposure_slider"]
    shadow_slider = ui_components["shadow_slider"]
    highlight_slider = ui_components["highlight_slider"]
    temperature_slider = ui_components["temperature_slider"]
    grain_strength_slider = ui_components["grain_strength_slider"]
    grain_size_slider = ui_components["grain_size_slider"]
    preset_dropdown = ui_components["preset_dropdown"]
    preset_apply_btn = ui_components["preset_apply_btn"]
    preset_save_name = ui_components["preset_save_name"]
    preset_save_btn = ui_components["preset_save_btn"]
    preset_status = ui_components["preset_status"]
    auto_mode = ui_components["auto_mode"]
    format_dropdown = ui_components["format_dropdown"]
    apply_filters_ck = ui_components["apply_filters_to_filename"]
    batch_btn = ui_components["batch_btn"]
    cancel_btn = ui_components["cancel_btn"]
    status_msg = ui_components["status_msg"]
    result_table = ui_components["result_table"]

    # ------------------------------------------------------------------
    # Cola: cuando llegan archivos (sueltos o carpeta), llenar la tabla
    # ------------------------------------------------------------------
    def refresh_queue(files_val, folder_val):
        paths = _normalize_files(files_val) + _normalize_files(folder_val)
        rows = []
        for p in paths:
            fp = Path(p)
            try:
                size_kb = round(fp.stat().st_size / 1024, 1)
            except Exception:
                size_kb = 0
            rows.append([fp.name, "en cola", f"{size_kb} KB"])
        return rows[:50], f"{len(rows)} foto(s) en cola"

    for comp in (input_files, input_folder):
        comp.upload(fn=refresh_queue, inputs=[input_files, input_folder],
                    outputs=[queue_table, preview_status], queue=False)
        comp.change(fn=refresh_queue, inputs=[input_files, input_folder],
                    outputs=[queue_table, preview_status], queue=False)

    # ------------------------------------------------------------------
    # Aplicar preset de fábrica a los sliders
    # ------------------------------------------------------------------
    def apply_preset(name: str):
        vals = PRESET_DEFAULTS.get(name, PRESET_DEFAULTS["Retrato"])
        return (
            vals["denoise"], vals["sharpen"], vals["recover"], vals["upscale"],
            vals["exposure"], vals["shadow"], vals["highlight"], vals["temperature"],
            vals["grain_strength"], vals["grain_size"],
            f"Preset «{name}» aplicado a los sliders.",
        )

    preset_apply_btn.click(
        fn=apply_preset,
        inputs=preset_dropdown,
        outputs=[
            denoise_slider, sharpen_slider, recover_slider, upscale_factor,
            exposure_slider, shadow_slider, highlight_slider, temperature_slider,
            grain_strength_slider, grain_size_slider, preset_status,
        ],
        queue=False,
    )

    # ------------------------------------------------------------------
    # Guardar preset personalizado en config/enhance/presets/
    # ------------------------------------------------------------------
    def save_preset(
        name, denoise, sharpen, recover, upscale, exposure, shadow, highlight,
        temperature, grain_strength, grain_size,
    ):
        name = (name or "").strip()
        if not name:
            return "Escribe un nombre para el preset."
        data = {
            "name": name,
            "description": "Preset manual (pestaña Mejorar fotos)",
            "sliders": {
                "denoise_strength": denoise,
                "sharpen_strength": sharpen,
                "recover_faces_strength": recover / 100.0,
                "upscale_factor": upscale,
                "exposure": exposure,
                "shadow": shadow,
                "highlight": highlight,
                "temperature": temperature,
                "grain_strength": grain_strength,
                "grain_size": grain_size,
            },
            "models": {},
        }
        try:
            preset_dir = Path(__file__).resolve().parents[2] / "config" / "enhance" / "presets"
            preset_dir.mkdir(parents=True, exist_ok=True)
            preset_path = preset_dir / f"{name.lower().replace(' ', '_')}.json"
            with open(preset_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return f"Preset «{name}» guardado en {preset_path.name}."
        except Exception as e:
            return f"Error al guardar preset: {e}"

    preset_save_btn.click(
        fn=save_preset,
        inputs=[
            preset_save_name, denoise_slider, sharpen_slider, recover_slider,
            upscale_factor, exposure_slider, shadow_slider, highlight_slider,
            temperature_slider, grain_strength_slider, grain_size_slider,
        ],
        outputs=preset_status,
        queue=False,
    )

    # ------------------------------------------------------------------
    # Vista previa antes/después de la PRIMERA foto de la cola
    # ------------------------------------------------------------------
    def make_preview(
        files, folder, denoise, sharpen, recover, upscale, exposure, shadow,
        highlight, temperature, grain_strength, grain_size,
    ):
        paths = _normalize_files(files) + _normalize_files(folder)
        if not paths:
            return None, "_Sube al menos una foto para ver el antes/después._"
        path = Path(paths[0])
        try:
            settings = {
                "denoise_strength": denoise,
                "sharpen_strength": sharpen,
                "recover_faces_strength": recover / 100.0,
                "upscale_factor": int(upscale),
                "exposure": exposure,
                "shadow": shadow,
                "highlight": highlight,
                "temperature": temperature,
                "grain_strength": grain_strength,
                "grain_size": grain_size,
            }
            # Vista previa a tamaño seguro para UI (máx 1024 de lado)
            _max_side = 1024
            before = Image.open(path).convert("RGB")
            w, h = before.size
            if max(w, h) > _max_side:
                s = _max_side / max(w, h)
                before = before.resize((int(w * s), int(h * s)), Image.LANCZOS)
            # process_single_image abre el archivo otra vez; para preview usamos
            # la imagen ya redimensionada como entrada temporal.
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                before.save(tf.name, "PNG")
                tmp_path = Path(tf.name)
            after, _notes, _m = process_single_image(tmp_path, settings)
            return (before, after), f"Vista previa: {path.name}"
        except Exception as e:
            return None, f"Vista previa falló: {e}"

    btn_preview.click(
        fn=make_preview,
        inputs=[
            input_files, input_folder, denoise_slider, sharpen_slider, recover_slider,
            upscale_factor, exposure_slider, shadow_slider, highlight_slider,
            temperature_slider, grain_strength_slider, grain_size_slider,
        ],
        outputs=[preview_slider, preview_status],
        queue=True,
    )

    # ------------------------------------------------------------------
    # Previsualización en tiempo real al mover cualquier slider
    # (denoise, sharpen, luz, grano, etc.) sin necesidad de pulsar el botón
    # ------------------------------------------------------------------
    preview_inputs = [
        input_files, input_folder, denoise_slider, sharpen_slider, recover_slider,
        upscale_factor, exposure_slider, shadow_slider, highlight_slider,
        temperature_slider, grain_strength_slider, grain_size_slider,
    ]

    for slider in (
        denoise_slider, sharpen_slider, recover_slider, upscale_factor,
        exposure_slider, shadow_slider, highlight_slider,
        temperature_slider, grain_strength_slider, grain_size_slider,
    ):
        slider.change(
            fn=make_preview,
            inputs=preview_inputs,
            outputs=[preview_slider, preview_status],
            queue=False,
        )

    # ------------------------------------------------------------------
    # Procesar lote completo
    # ------------------------------------------------------------------
    def do_batch(
        files, folder, preset_name,
        denoise, sharpen, recover, upscale, exposure, shadow, highlight,
        temperature, grain_strength, grain_size,
        export_format, apply_filters_to_filename, auto_mode,
    ):
        t0 = time.time()
        paths = _normalize_files(files) + _normalize_files(folder)
        if not paths:
            return "No se seleccionaron archivos.", []

        enh_logic.clear_cancel()

        settings = {
            "denoise_strength": denoise,
            "sharpen_strength": sharpen,
            "recover_faces_strength": recover / 100.0,
            "upscale_factor": upscale,
            "exposure": exposure,
            "shadow": shadow,
            "highlight": highlight,
            "temperature": temperature,
            "grain_strength": grain_strength,
            "grain_size": grain_size,
        }

        output_dir = Path(get_enhance_output_dir())

        results = process_batch(
            file_paths=[Path(p) for p in paths],
            settings=settings,
            output_dir=output_dir,
            export_format=str(export_format or "png"),
            apply_filters_to_filename=bool(apply_filters_to_filename),
            allow_overwrite=True,
            progress_cb=None,
            cancel_check=enh_logic.is_cancel_requested,
            autopilot=bool(auto_mode),
        )

        table_data = []
        ok_count = 0
        for res in results:
            entry = Path(res.get("input", "")).name
            output = Path(res.get("output", "")).name if res.get("output") else "—"
            notes = res.get("notes", "")
            if isinstance(notes, (list, tuple)):
                notes = "; ".join(str(n) for n in notes)
            notes = str(notes) if notes else "OK"
            ok = "✅" if res.get("success") else "❌"
            ok_count += 1 if res.get("success") else 0
            table_data.append([entry, output, notes, ok])

        elapsed = time.time() - t0
        msg = f"Procesados {ok_count}/{len(results)} archivos en {elapsed:.1f} s."
        return msg, table_data

    batch_btn.click(
        fn=do_batch,
        inputs=[
            input_files, input_folder, preset_dropdown,
            denoise_slider, sharpen_slider, recover_slider, upscale_factor,
            exposure_slider, shadow_slider, highlight_slider, temperature_slider,
            grain_strength_slider, grain_size_slider,
            format_dropdown, apply_filters_ck, auto_mode,
        ],
        outputs=[status_msg, result_table],
        queue=True,
    )

    # ------------------------------------------------------------------
    # Cancelar
    # ------------------------------------------------------------------
    def do_cancel():
        enh_logic.request_cancel()
        return "Cancelado por el usuario."

    cancel_btn.click(fn=do_cancel, inputs=[], outputs=status_msg, queue=False)
