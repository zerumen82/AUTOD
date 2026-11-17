"""Eventos de la pestaña «Mejorar fotos» (estilo local).

Reglas Gradio 6:
- No se escribe `.value` sobre un componente desde un handler; se devuelven
  valores y se declaran en `outputs`.
- `gr.Files` devuelve lista de rutas (str) u objetos con `.name`; se normaliza.
- La primera foto de la cola se selecciona y renderiza sola al importar.
"""

import json
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path

import gradio as gr
from PIL import Image, ImageOps

from roop.output_paths import get_enhance_output_dir

from . import logic as enh_logic
from .logic import process_batch, process_single_image


PRESET_DEFAULTS = {
    "Realista": {
        "denoise": 25, "sharpen": 30, "recover": 60, "upscale": 1,
        "exposure": 1.0, "shadow": 0.95, "highlight": 1.03, "temperature": 50,
        "grain_strength": 6, "grain_size": 8, "deband": 20, "vibrance": 104,
        "face_model": "GFPGAN",
    },
    "Ultra realista": {
        "denoise": 30, "sharpen": 45, "recover": 75, "upscale": 2,
        "exposure": 1.02, "shadow": 0.9, "highlight": 1.05, "temperature": 51,
        "grain_strength": 10, "grain_size": 6, "deband": 15, "vibrance": 108,
        "face_model": "CodeFormer",
    },
    "Sin ruido": {
        "denoise": 75, "sharpen": 18, "recover": 30, "upscale": 1,
        "exposure": 1.0, "shadow": 1.0, "highlight": 1.0, "temperature": 50,
        "grain_strength": 0, "grain_size": 10, "deband": 40, "vibrance": 100,
        "face_model": "GFPGAN",
    },
    "Sin bandas": {
        "denoise": 55, "sharpen": 20, "recover": 0, "upscale": 1,
        "exposure": 1.0, "shadow": 1.0, "highlight": 1.0, "temperature": 50,
        "grain_strength": 4, "grain_size": 12, "deband": 75, "vibrance": 100,
        "face_model": "GFPGAN",
    },
    "Color vivo": {
        "denoise": 20, "sharpen": 28, "recover": 40, "upscale": 1,
        "exposure": 1.05, "shadow": 1.08, "highlight": 0.96, "temperature": 53,
        "grain_strength": 4, "grain_size": 8, "deband": 10, "vibrance": 115,
        "face_model": "GFPGAN",
    },
    "Retrato": {
        "denoise": 30, "sharpen": 25, "recover": 65, "upscale": 1,
        "exposure": 1.02, "shadow": 0.95, "highlight": 1.02, "temperature": 51,
        "grain_strength": 5, "grain_size": 10, "deband": 20, "vibrance": 104,
        "face_model": "GFPGAN",
    },
    "Paisaje": {
        "denoise": 45, "sharpen": 55, "recover": 0, "upscale": 2,
        "exposure": 1.0, "shadow": 0.7, "highlight": 1.1, "temperature": 54,
        "grain_strength": 8, "grain_size": 12, "deband": 30, "vibrance": 112,
        "face_model": "GFPGAN",
    },
    "Nocturna": {
        "denoise": 80, "sharpen": 25, "recover": 40, "upscale": 2,
        "exposure": 1.4, "shadow": 1.4, "highlight": 0.6, "temperature": 35,
        "grain_strength": 20, "grain_size": 8, "deband": 50, "vibrance": 105,
        "face_model": "GFPGAN",
    },
    "Producto": {
        "denoise": 35, "sharpen": 50, "recover": 0, "upscale": 2,
        "exposure": 1.0, "shadow": 0.85, "highlight": 1.0, "temperature": 50,
        "grain_strength": 8, "grain_size": 10, "deband": 25, "vibrance": 106,
        "face_model": "GFPGAN",
    },
    "Archivo": {
        "denoise": 20, "sharpen": 20, "recover": 30, "upscale": 1,
        "exposure": 1.0, "shadow": 1.0, "highlight": 1.0, "temperature": 50,
        "grain_strength": 3, "grain_size": 10, "deband": 15, "vibrance": 100,
        "face_model": "GFPGAN",
    },
    "Blanco y negro": {
        "denoise": 30, "sharpen": 25, "recover": 0, "upscale": 1,
        "exposure": 1.0, "shadow": 1.0, "highlight": 1.0, "temperature": 50,
        "grain_strength": 15, "grain_size": 10, "deband": 20, "vibrance": 0,
        "face_model": "GFPGAN",
    },
    "HDR": {
        "denoise": 15, "sharpen": 45, "recover": 30, "upscale": 1,
        "exposure": 1.1, "shadow": 0.7, "highlight": 1.2, "temperature": 50,
        "grain_strength": 5, "grain_size": 8, "deband": 10, "vibrance": 115,
        "face_model": "GFPGAN",
    },
    "Selfie": {
        "denoise": 40, "sharpen": 20, "recover": 70, "upscale": 1,
        "exposure": 1.05, "shadow": 0.9, "highlight": 1.0, "temperature": 52,
        "grain_strength": 0, "grain_size": 10, "deband": 15, "vibrance": 108,
        "face_model": "GFPGAN",
    },
    "Vintage": {
        "denoise": 25, "sharpen": 15, "recover": 20, "upscale": 1,
        "exposure": 0.95, "shadow": 1.1, "highlight": 0.9, "temperature": 60,
        "grain_strength": 25, "grain_size": 12, "deband": 10, "vibrance": 90,
        "face_model": "GFPGAN",
    },
    "Noche clara": {
        "denoise": 70, "sharpen": 30, "recover": 40, "upscale": 2,
        "exposure": 1.5, "shadow": 1.5, "highlight": 0.5, "temperature": 38,
        "grain_strength": 12, "grain_size": 8, "deband": 45, "vibrance": 102,
        "face_model": "GFPGAN",
    },
}

# Vista previa: el original se guarda ya escalado y el resultado se acota para
# que mover un slider sea inmediato; el lote sigue a resolución completa.
_PREVIEW_BEFORE_MAX = 1024
_PREVIEW_AFTER_MAX = 2048
_BEFORE_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_BEFORE_CACHE_MAX = 12

# Fotografías de ejemplo locales (sin contenido sensible).
_SAMPLE_CANDIDATES = (
    "diagnostics/out/editor_categories_20261115/remove.png",
    "diagnostics/out/office/window.png",
    "diagnostics/out/couch_check/window.png",
)


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


def _queue_paths(files_val, folder_val) -> list:
    """Rutas de la cola, sin duplicados (misma foto en archivos y carpeta)."""
    out = []
    seen = set()
    for p in _normalize_files(files_val) + _normalize_files(folder_val):
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _pick_path(paths: list, selected) -> Path:
    """Ruta de la foto elegida; si no hay selección, la primera de la cola."""
    if selected:
        target = os.path.normcase(os.path.abspath(_resolve_path(selected)))
        for p in paths:
            if os.path.normcase(os.path.abspath(p)) == target:
                return Path(p)
    return Path(paths[0])


def _load_before(path: Path) -> Image.Image:
    """Original ya escalado, cacheado por ruta+mtime (no re-decodifica)."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    key = str(path)
    cached = _BEFORE_CACHE.get(key)
    if cached and cached.get("mtime") == mtime:
        _BEFORE_CACHE.move_to_end(key)
        return cached["image"]
    with Image.open(path) as _img:
        image = ImageOps.exif_transpose(_img).convert("RGB")
    w, h = image.size
    if max(w, h) > _PREVIEW_BEFORE_MAX:
        s = _PREVIEW_BEFORE_MAX / max(w, h)
        image = image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
    _BEFORE_CACHE[key] = {"mtime": mtime, "image": image}
    _BEFORE_CACHE.move_to_end(key)
    while len(_BEFORE_CACHE) > _BEFORE_CACHE_MAX:
        _BEFORE_CACHE.popitem(last=False)
    return image


def _cap_size(image: Image.Image, max_side: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_side:
        return image
    s = max_side / max(w, h)
    return image.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)


def _settings(denoise, sharpen, recover, upscale, exposure, shadow, highlight,
              temperature, grain_strength, grain_size, deband=0, vibrance=100,
              face_model="GFPGAN") -> dict:
    return {
        "denoise_strength": denoise,
        "sharpen_strength": sharpen,
        "recover_faces_strength": recover / 100.0,
        "upscale_factor": int(upscale or 1),
        "exposure": exposure,
        "shadow": shadow,
        "highlight": highlight,
        "temperature": temperature,
        "grain_strength": grain_strength,
        "grain_size": grain_size,
        "deband_strength": float(deband) if deband is not None else 0.0,
        "vibrance": float(vibrance) if vibrance is not None else 100.0,
        "face_model": str(face_model or "GFPGAN"),
    }


def _render_after(before: Image.Image, settings: dict, return_notes: bool = False):
    """Procesa la vista previa sobre el original ya escalado.

    El upscale se omite en la vista previa: el panel se muestra reducido, no
    aporta nada visual y es el paso más lento. Se aplica íntegro en el lote.
    """
    preview_settings = dict(settings)
    preview_settings["upscale_factor"] = 1
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
        before.save(tf.name, "PNG")
        tmp_path = Path(tf.name)
    try:
        after, notes, _metrics = process_single_image(tmp_path, preview_settings)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    after = _cap_size(after, _PREVIEW_AFTER_MAX)
    if return_notes:
        return after, notes
    return after


def _status_line(name: str, s: dict) -> str:
    return (
        f"Vista previa: **{name}** · denoise {float(s['denoise_strength']):g}"
        f" · sharpen {float(s['sharpen_strength']):g}"
        f" · caras {s.get('face_model', 'GFPGAN')} {float(s['recover_faces_strength']) * 100:g}"
        f" · upscale {s['upscale_factor']}x al exportar (preview 1x)"
    )


def wire_events(ui_components):
    """Conecta todos los eventos de la pestaña de mejora de fotos."""

    input_files = ui_components["input_files"]
    input_folder = ui_components["input_folder"]
    btn_load_sample = ui_components["btn_load_sample"]
    queue_summary = ui_components["queue_summary"]
    preview_before = ui_components["preview_before"]
    preview_after = ui_components["preview_after"]
    preview_photo = ui_components["preview_photo"]
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
    deband_slider = ui_components["deband_slider"]
    vibrance_slider = ui_components["vibrance_slider"]
    face_model = ui_components["face_model"]
    preset_dropdown = ui_components["preset_dropdown"]
    preset_apply_btn = ui_components["preset_apply_btn"]
    preset_save_name = ui_components["preset_save_name"]
    preset_save_btn = ui_components["preset_save_btn"]
    preset_status = ui_components["preset_status"]
    auto_mode = ui_components["auto_mode"]
    output_path = ui_components["output_path"]
    btn_open_folder = ui_components["btn_open_folder"]
    folder_status = ui_components["folder_status"]
    format_dropdown = ui_components["format_dropdown"]
    apply_filters_ck = ui_components["apply_filters_to_filename"]
    batch_btn = ui_components["batch_btn"]
    cancel_btn = ui_components["cancel_btn"]
    status_msg = ui_components["status_msg"]
    result_summary = ui_components["result_summary"]

    slider_inputs = [
        denoise_slider, sharpen_slider, recover_slider,
        upscale_factor, exposure_slider, shadow_slider, highlight_slider,
        temperature_slider, grain_strength_slider, grain_size_slider,
        deband_slider, vibrance_slider,
    ]

    # ------------------------------------------------------------------
    # Cola: cuando llegan archivos (sueltos o carpeta), llenar la tabla y
    # seleccionar la PRIMERA foto para la vista previa.
    # ------------------------------------------------------------------
    def refresh_queue(files_val, folder_val):
        paths = _queue_paths(files_val, folder_val)
        if not paths:
            return (
                "_Sin fotos en cola._",
                gr.update(choices=[], value=None),
                "_Importa fotos para ver el antes/después._",
            )
        first_name = Path(paths[0]).name
        summary = (
            f"**{len(paths)} foto(s) en cola** · vista previa: `{first_name}`"
        )
        choices = [(Path(p).name, p) for p in paths]
        return (
            summary,
            gr.update(choices=choices, value=paths[0]),
            f"Generando vista previa de {len(paths)} foto(s)…",
        )

    # ------------------------------------------------------------------
    # Vista previa ANTES / DESPUÉS
    # ------------------------------------------------------------------
    def make_pair(files_val, folder_val, selected, *slider_vals):
        paths = _queue_paths(files_val, folder_val)
        if not paths:
            return None, None, "_Sube al menos una foto para ver el antes/después._"
        path = _pick_path(paths, selected)
        settings = _settings(*slider_vals)
        try:
            before = _load_before(path)
            after, notes = _render_after(before, settings, return_notes=True)
            status = _status_line(path.name, settings)
            if notes:
                status = f"{status}\n\n{notes}"
            return before, after, status
        except Exception as e:
            return None, None, f"Vista previa falló: {e}"

    def make_after(files_val, folder_val, selected, *slider_vals):
        paths = _queue_paths(files_val, folder_val)
        if not paths:
            return None, "_Sube al menos una foto para ver el antes/después._"
        path = _pick_path(paths, selected)
        settings = _settings(*slider_vals)
        try:
            before = _load_before(path)
            after, notes = _render_after(before, settings, return_notes=True)
            status = _status_line(path.name, settings)
            if notes:
                status = f"{status}\n\n{notes}"
            return after, status
        except Exception as e:
            return None, f"Vista previa falló: {e}"

    pair_inputs = [input_files, input_folder, preview_photo] + slider_inputs + [face_model]

    # Gradio 6.x soporta concurrency_limit y trigger_mode: se re-añaden para
    # evitar renders encolados al mover sliders (always_last = solo corre el último).
    for comp in (input_files, input_folder):
        event = comp.change(
            fn=refresh_queue,
            inputs=[input_files, input_folder],
            outputs=[queue_summary, preview_photo, preview_status],
            queue=False,
        )
        event.then(
            fn=make_pair,
            inputs=pair_inputs,
            outputs=[preview_before, preview_after, preview_status],
            queue=True,
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="default",
        )

    preview_photo.change(
        fn=make_pair,
        inputs=pair_inputs,
        outputs=[preview_before, preview_after, preview_status],
        queue=True,
        concurrency_limit=1,
        trigger_mode="always_last",
        show_progress="default",
    )

    btn_preview.click(
        fn=make_pair,
        inputs=pair_inputs,
        outputs=[preview_before, preview_after, preview_status],
        queue=True,
        concurrency_limit=1,
        show_progress="minimal",
    )

    # Cambiar cualquier ajuste re-renderiza SOLO el panel «Después».
    for slider in slider_inputs:
        slider.change(
            fn=make_after,
            inputs=pair_inputs,
            outputs=[preview_after, preview_status],
            queue=True,
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="default",
        )

    face_model.change(
        fn=make_after,
        inputs=pair_inputs,
        outputs=[preview_after, preview_status],
        queue=True,
        concurrency_limit=1,
        trigger_mode="always_last",
        show_progress="default",
    )

    # ------------------------------------------------------------------
    # Cargar fotos de ejemplo (evita un botón sin función)
    # ------------------------------------------------------------------
    def load_sample():
        root = Path(__file__).resolve().parents[3]
        found = [str(root / rel) for rel in _SAMPLE_CANDIDATES if (root / rel).exists()]
        if not found:
            return gr.update(), "No hay fotos de ejemplo disponibles en diagnostics/out."
        return gr.update(value=found), f"Ejemplo cargado: {len(found)} foto(s)."

    btn_load_sample.click(
        fn=load_sample,
        inputs=[],
        outputs=[input_files, preview_status],
        queue=False,
    )

    # ------------------------------------------------------------------
    # Aplicar preset de fábrica a los sliders
    # ------------------------------------------------------------------
    def apply_preset(name: str):
        vals = PRESET_DEFAULTS.get(name, PRESET_DEFAULTS["Retrato"])
        return (
            vals["denoise"], vals["sharpen"], vals["recover"], vals["upscale"],
            vals["exposure"], vals["shadow"], vals["highlight"], vals["temperature"],
            vals["grain_strength"], vals["grain_size"],
            vals.get("deband", 0), vals.get("vibrance", 100), vals.get("face_model", "GFPGAN"),
            f"Preset «{name}» aplicado a los sliders.",
        )

    preset_apply_btn.click(
        fn=apply_preset,
        inputs=preset_dropdown,
        outputs=[
            denoise_slider, sharpen_slider, recover_slider, upscale_factor,
            exposure_slider, shadow_slider, highlight_slider, temperature_slider,
            grain_strength_slider, grain_size_slider,
            deband_slider, vibrance_slider, face_model, preset_status,
        ],
        queue=False,
    )

    # Reset sliders a valores por defecto
    def reset_sliders():
        from .logic import load_enhance_config
        d = load_enhance_config().get("default_slider_values", {})
        recover = d.get("recover_faces_strength", 0.8)
        recover_pct = (recover * 100) if isinstance(recover, (int, float)) and recover <= 1.0 else float(recover)
        return (
            float(d.get("denoise_strength", 20)),
            float(d.get("sharpen_strength", 10)),
            recover_pct,
            int(d.get("upscale_factor", 1)),
            float(d.get("exposure", 1.0)),
            float(d.get("shadow", 1.0)),
            float(d.get("highlight", 1.0)),
            float(d.get("temperature", 50)),
            float(d.get("grain_strength", 0)),
            float(d.get("grain_size", 10)),
            float(d.get("deband_strength", 0)),
            float(d.get("vibrance", 100)),
            str(d.get("face_model", "GFPGAN")),
            "Sliders restablecidos a valores por defecto.",
        )

    btn_reset_sliders = ui_components.get("btn_reset_sliders")
    if btn_reset_sliders is not None:
        btn_reset_sliders.click(
            fn=reset_sliders,
            outputs=[
                denoise_slider, sharpen_slider, recover_slider, upscale_factor,
                exposure_slider, shadow_slider, highlight_slider, temperature_slider,
                grain_strength_slider, grain_size_slider,
                deband_slider, vibrance_slider, face_model, preset_status,
            ],
            queue=False,
        )

    # ------------------------------------------------------------------
    # Guardar preset personalizado en config/enhance/presets/
    # ------------------------------------------------------------------
    def save_preset(
        name, denoise, sharpen, recover, upscale, exposure, shadow, highlight,
        temperature, grain_strength, grain_size, deband, vibrance, face_model,
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
                "deband_strength": deband,
                "vibrance": vibrance,
                "face_model": face_model,
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
            deband_slider, vibrance_slider, face_model,
        ],
        outputs=preset_status,
        queue=False,
    )

    # ------------------------------------------------------------------
    # Abrir la carpeta de destino en el explorador
    # ------------------------------------------------------------------
    def open_output_folder():
        path = get_enhance_output_dir()
        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                subprocess.Popen(["xdg-open", path])
            msg = f"Carpeta abierta: {path}"
        except Exception as e:
            msg = f"No se pudo abrir la carpeta: {e}"
        return msg

    btn_open_folder.click(
        fn=open_output_folder,
        inputs=[],
        outputs=folder_status,
        queue=False,
    )

    # ------------------------------------------------------------------
    # Procesar lote completo
    # ------------------------------------------------------------------
    def do_batch(
        files, folder, preset_name,
        denoise, sharpen, recover, upscale, exposure, shadow, highlight,
        temperature, grain_strength, grain_size, deband, vibrance, face_model,
        export_format, apply_filters_to_filename, auto_mode,
    ):
        paths = _queue_paths(files, folder)
        if not paths:
            yield "<div class='enhance-status'>No se seleccionaron archivos.</div>", ""
            return

        t0 = time.time()
        enh_logic.clear_cancel()

        settings = _settings(
            denoise, sharpen, recover, upscale, exposure, shadow, highlight,
            temperature, grain_strength, grain_size, deband, vibrance, face_model,
        )
        output_dir = Path(get_enhance_output_dir())
        total = len(paths)
        updates: "queue.Queue" = queue.Queue()
        result: dict = {"results": None, "error": None}

        def progress_cb(payload):
            updates.put(payload)

        def worker():
            try:
                result["results"] = process_batch(
                    file_paths=[Path(p) for p in paths],
                    settings=settings,
                    output_dir=output_dir,
                    export_format=str(export_format or "png"),
                    apply_filters_to_filename=bool(apply_filters_to_filename),
                    allow_overwrite=True,
                    progress_cb=progress_cb,
                    cancel_check=enh_logic.is_cancel_requested,
                    autopilot=bool(auto_mode),
                )
            except Exception as exc:
                result["error"] = exc
            finally:
                updates.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        yield (
            f"<div class='enhance-status'>Procesando lote (0/{total})…</div>",
            "",
        )

        try:
            while True:
                try:
                    payload = updates.get(timeout=0.5)
                except queue.Empty:
                    yield gr.skip(), gr.skip()
                    continue
                if payload is None:
                    break
                phase = str(payload.get("phase", ""))
                detail = str(payload.get("detail", ""))
                if phase == "batch":
                    pct = int(float(payload.get("progress", 0.0)) * 100)
                    status = f"<div class='enhance-status'>Procesando {pct}% · {detail}</div>"
                else:
                    status = f"<div class='enhance-status'>{phase}: {detail}</div>"
                yield status, gr.skip()
        finally:
            if result["results"] is None:
                enh_logic.request_cancel()

        if result["error"] is not None:
            yield (
                f"<div class='enhance-status'>Error en el lote: {result['error']}</div>",
                "",
            )
            return

        results = result["results"] or []
        ok_count = 0
        errors = []
        reverted = 0
        for res in results:
            notes = str(res.get("notes", "") or "")
            if "revertido" in notes:
                reverted += 1
            if res.get("success"):
                ok_count += 1
                continue
            entry = Path(res.get("input", "")).name
            errors.append(f"{entry}: {notes or 'error'}")

        elapsed = time.time() - t0
        msg = f"Procesados {ok_count}/{len(results)} archivos en {elapsed:.1f} s."
        summary = (
            f"**{ok_count}/{len(results)} fotos procesadas** en {elapsed:.1f} s"
            f" · salida: `{output_dir.name}/`"
        )
        if reverted:
            summary += f" · {reverted} foto(s) con pasos revertidos"
        if errors:
            summary += " · errores: " + "; ".join(errors[:5])
        yield f"<div class='enhance-status'>{msg}</div>", summary

    batch_btn.click(
        fn=do_batch,
        inputs=[
            input_files, input_folder, preset_dropdown,
            denoise_slider, sharpen_slider, recover_slider, upscale_factor,
            exposure_slider, shadow_slider, highlight_slider, temperature_slider,
            grain_strength_slider, grain_size_slider,
            deband_slider, vibrance_slider, face_model,
            format_dropdown, apply_filters_ck, auto_mode,
        ],
        outputs=[status_msg, result_summary],
        queue=True,
    )

    # ------------------------------------------------------------------
    # Cancelar
    # ------------------------------------------------------------------
    def do_cancel():
        enh_logic.request_cancel()
        return "<div class='enhance-status'>Cancelado por el usuario.</div>"

    cancel_btn.click(fn=do_cancel, inputs=[], outputs=status_msg, queue=False)
