"""Eventos de la pestaña «Mejorar fotos» (estilo local).

Reglas Gradio 6:
- No se escribe `.value` sobre un componente desde un handler; se devuelven
  valores y se declaran en `outputs`.
- `gr.Files` devuelve lista de rutas (str) u objetos con `.name`; se normaliza.
- La primera foto de la cola se selecciona y renderiza sola al importar.
"""

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
from .presets import icon_for_preset, load_presets, save_custom_preset

Image.MAX_IMAGE_PIXELS = None


# Presets/ajustes por foto: ruta -> settings dict
_PER_IMAGE_SETTINGS: dict = {}
# Cola resuelta por refresh_queue; el .then posterior no recibe los FileData.
_LAST_QUEUE_PATHS: list = []


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


def _queue_paths(files_val, folder_val=None) -> list:
    """Rutas de la cola: sin duplicados y solo imágenes (modo carpeta).

    El selector de carpeta de Gradio entrega TODOS los archivos del directorio;
    el registro de formatos de PIL decide qué es imagen sin listas escritas a
    mano. Un archivo sin extensión se conserva para que el procesador lo juzgue.
    """
    image_exts = {ext.lower() for ext in Image.registered_extensions()}
    out = []
    seen = set()
    sources = _normalize_files(files_val) + (_normalize_files(folder_val) if folder_val else [])
    for p in sources:
        suffix = Path(p).suffix.lower()
        if suffix and suffix not in image_exts:
            continue
        key = os.path.normcase(os.path.abspath(p))
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _path_index(paths: list, selected) -> int:
    """Índice de la ruta seleccionada, comparando rutas equivalentes."""
    if selected:
        target = os.path.normcase(os.path.abspath(_resolve_path(selected)))
        for index, p in enumerate(paths):
            if os.path.normcase(os.path.abspath(p)) == target:
                return index
    return 0


def _pick_path(paths: list, selected) -> Path:
    """Ruta de la foto elegida; si no hay selección, la primera de la cola."""
    return Path(paths[_path_index(paths, selected)])


def _photo_position(paths: list, selected) -> str:
    if not paths:
        return "_0 / 0 fotos_"
    index = _path_index(paths, selected) + 1
    chosen = Path(paths[index - 1])
    return f"**{index} / {len(paths)}** · `{chosen.name}`"


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
        resample = Image.Resampling.BILINEAR if max(w, h) > 2500 else Image.Resampling.LANCZOS
        image = image.resize((max(1, int(w * s)), max(1, int(h * s))), resample)
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
        "accumulated_presets": [],
    }


def _render_after(before: Image.Image, settings: dict, return_notes: bool = False):
    """Procesa la vista previa sobre el original ya escalado.

    El upscale se omite en la vista previa: el panel se muestra reducido, no
    aporta nada visual y es el paso más lento. Se aplica íntegro en el lote.
    """
    preview_settings = dict(settings)
    preview_settings["upscale_factor"] = 1
    preview_settings["recover_faces_strength"] = 0
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


def _neutral_settings() -> dict:
    d = enh_logic.load_enhance_config().get("default_slider_values", {})
    recover = d.get("recover_faces_strength", 0.8)
    recover_pct = (recover * 100) if isinstance(recover, (int, float)) and recover <= 1.0 else float(recover)
    s = _settings(
        float(d.get("denoise_strength", 0)),
        float(d.get("sharpen_strength", 0)),
        recover_pct,
        int(d.get("upscale_factor", 1)),
        float(d.get("exposure", 1.0)),
        float(d.get("shadow", 1.0)),
        float(d.get("highlight", 1.0)),
        float(d.get("temperature", 50)),
        float(d.get("grain_strength", 0)),
        float(d.get("grain_size", 10)),
        deband=float(d.get("deband_strength", 0)),
        vibrance=float(d.get("vibrance", 100)),
        face_model=str(d.get("face_model", "GFPGAN")),
    )
    s["accumulated_presets"] = []
    return s


# v5.1437 — la pila no puede sumar hasta la piel de cera («muy muñeco»):
# por encima de estos topes el poro se pierde igual, así que satura ahí. El
# slider manual sigue llegando a 100 para quien lo quiera forzar.
_STACK_CAPS = {
    "denoise_strength": 60.0,
    "sharpen_strength": 45.0,
    "deband_strength": 60.0,
    "grain_strength": 30.0,
}


def _merge_settings(base: dict, add: dict) -> dict:
    out = dict(base)
    out.setdefault("accumulated_presets", list(base.get("accumulated_presets", [])))
    limits = enh_logic.load_enhance_config().get("control_limits", {})

    def _clamp(key, value):
        rule = limits.get(key, {})
        if "choices" in rule:
            return value if value in rule["choices"] else base.get(key, value)
        lo = float(rule.get("minimum", 0))
        hi = float(rule.get("maximum", 100))
        return float(min(hi, max(lo, value)))

    capped = []
    for key in ("denoise_strength", "sharpen_strength", "deband_strength", "grain_strength"):
        total = float(base.get(key, 0)) + float(add.get(key, 0))
        limit = _STACK_CAPS[key]
        if total > limit:
            capped.append(key)
        out[key] = _clamp(key, min(total, limit))
    if capped:
        out["stack_capped"] = capped
    else:
        out.pop("stack_capped", None)
    out["recover_faces_strength"] = min(
        0.8, float(base.get("recover_faces_strength", 0)) + float(add.get("recover_faces_strength", 0))
    )
    out["upscale_factor"] = max(int(base.get("upscale_factor", 1)), int(add.get("upscale_factor", 1)))
    for key in ("exposure", "shadow", "highlight"):
        out[key] = _clamp(key, float(base.get(key, 1.0)) * float(add.get(key, 1.0)))
    if float(add.get("temperature", 50)) != 50:
        out["temperature"] = _clamp("temperature", float(add.get("temperature", 50)))
    out["grain_size"] = max(float(base.get("grain_size", 10)), float(add.get("grain_size", 10)))
    new_factor = (1.0 + (float(base.get("vibrance", 100)) - 100.0) / 100.0) * (
        1.0 + (float(add.get("vibrance", 100)) - 100.0) / 100.0
    )
    out["vibrance"] = _clamp("vibrance", 100.0 + (new_factor - 1.0) * 100.0)
    if str(add.get("face_model", "GFPGAN")) != "GFPGAN":
        out["face_model"] = str(add.get("face_model", base.get("face_model", "GFPGAN")))
    ap_base = list(base.get("accumulated_presets", []))
    for n in add.get("accumulated_presets", []):
        if n not in ap_base:
            ap_base.append(n)
    out["accumulated_presets"] = ap_base
    return out



def _stack_html(names):
    names = list(names or [])
    if not names:
        return "<div class='enhance-stack'>Pila: (ninguna)</div>"
    parts = []
    for name in names:
        data = load_presets().get(name)
        svg = ""
        if data is not None:
            try:
                svg = icon_for_preset(data).read_text(encoding="utf-8")
            except Exception:
                svg = ""
        parts.append(
            "<span class='enhance-stack-item' title='{}'>{}<small>{}</small></span>".format(
                name, svg, name
            )
        )
    return "<div class='enhance-stack'>Pila: " + "".join(parts) + "</div>"


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
    btn_browse_pc = ui_components.get("btn_browse_pc")
    input_folder = ui_components["input_folder"]
    select_all = ui_components["select_all"]
    btn_load_sample = ui_components["btn_load_sample"]
    queue_summary = ui_components["queue_summary"]
    preview_before = ui_components["preview_before"]
    preview_after = ui_components["preview_after"]
    preview_photo = ui_components["preview_photo"]
    btn_prev_photo = ui_components["btn_prev_photo"]
    btn_next_photo = ui_components["btn_next_photo"]
    preview_counter = ui_components["preview_counter"]
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
    preset_stack = ui_components.get("preset_stack")
    btn_clear_stack = ui_components.get("btn_clear_stack")
    preset_buttons = ui_components["preset_buttons"]
    auto_mode = ui_components["auto_mode"]
    output_path = ui_components["output_path"]
    btn_browse_folder = ui_components.get("btn_browse_folder")
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

    def _apply_settings_to_all(paths, settings):
        """«Seleccionar todas»: la misma receta para cada foto de la cola."""
        for p in paths:
            _PER_IMAGE_SETTINGS[str(p)] = dict(settings)

    def _resolve_output_dir(raw):
        """Carpeta de destino elegida; se crea si falta y usa enhance/ si falla."""
        default = get_enhance_output_dir()
        text = str(raw or "").strip()
        if not text:
            try:
                enh_logic.save_persisted_output_dir(default)
            except Exception:
                pass
            return default, f"Destino: carpeta por defecto `{default}`."
        try:
            path = Path(os.path.abspath(os.path.expandvars(os.path.expanduser(text))))
            path.mkdir(parents=True, exist_ok=True)
        except (OSError, ValueError) as exc:
            try:
                enh_logic.save_persisted_output_dir(default)
            except Exception:
                pass
            return default, f"No se pudo usar `{text}` ({exc}). Se mantiene `{default}`."
        try:
            enh_logic.save_persisted_output_dir(str(path))
        except Exception:
            pass
        return str(path), f"Destino: `{path}`."

    # ------------------------------------------------------------------
    # Cola: cuando llegan archivos (sueltos o carpeta), llenar la tabla y
    # seleccionar la PRIMERA foto para la vista previa.
    # ------------------------------------------------------------------
    def _nav_updates(paths, selected):
        index = _path_index(paths, selected) if paths else 0
        return (
            gr.update(interactive=bool(paths) and index > 0),
            gr.update(interactive=bool(paths) and index < len(paths) - 1),
        )

    def refresh_queue(files_val=None, folder_val=None):
        global _LAST_QUEUE_PATHS
        _PER_IMAGE_SETTINGS.clear()
        paths = _queue_paths(files_val, folder_val)
        _LAST_QUEUE_PATHS = paths
        if not paths:
            return (
                "_Sin fotos en cola._",
                gr.update(choices=[], value=None),
                "_Importa fotos para ver el antes/después._",
                _photo_position(paths, None),
                *_nav_updates(paths, None),
            )
        summary = f"**{len(paths)} foto(s) en cola**"
        choices = [(f"{index}. {Path(p).name}", p) for index, p in enumerate(paths, start=1)]
        max_visible = int(
            enh_logic.load_enhance_config().get("large_batch", {}).get("max_visible_choices", 100)
        )
        visible = choices[:max_visible]
        if len(choices) > max_visible:
            summary += f" · desplegable: {max_visible} de {len(choices)} (usa Anterior/Siguiente)"
        return (
            summary,
            gr.update(choices=visible, value=paths[0]),
            f"Generando vista previa de {len(paths)} foto(s)…",
            _photo_position(paths, paths[0]),
            *_nav_updates(paths, paths[0]),
        )

    # ------------------------------------------------------------------
    # Vista previa ANTES / DESPUÉS
    # ------------------------------------------------------------------
    _N_SLIDERS = 13

    def _slider_updates_from_settings(s: dict):
        return (
            gr.update(value=s["denoise_strength"]),
            gr.update(value=s["sharpen_strength"]),
            gr.update(value=float(s["recover_faces_strength"]) * 100.0),
            gr.update(value=int(s["upscale_factor"])),
            gr.update(value=s["exposure"]),
            gr.update(value=s["shadow"]),
            gr.update(value=s["highlight"]),
            gr.update(value=s["temperature"]),
            gr.update(value=s["grain_strength"]),
            gr.update(value=s["grain_size"]),
            gr.update(value=s["deband_strength"]),
            gr.update(value=s["vibrance"]),
            gr.update(value=s["face_model"]),
        )

    def make_pair(*args, **kwargs):
        files_val = kwargs.get("files_val")
        folder_val = kwargs.get("folder_val")
        selected = kwargs.get("selected")
        slider_vals = []

        if len(args) >= _N_SLIDERS + 3:
            files_val, folder_val, selected = args[0], args[1], args[2]
            slider_vals = list(args[3:])
        elif len(args) >= _N_SLIDERS + 1:
            selected = args[0]
            slider_vals = list(args[1:])
        elif len(args) > 0:
            selected = args[0]
            slider_vals = list(args[1:])

        paths = _queue_paths(files_val, folder_val) if (files_val or folder_val) else _LAST_QUEUE_PATHS
        if not paths:
            return (None, None, "_Sube al menos una foto para ver el antes/después._", _photo_position(paths, None), *_nav_updates(paths, None)) + tuple(gr.update() for _ in range(_N_SLIDERS))
        path = _pick_path(paths, selected)
        if str(path) in _PER_IMAGE_SETTINGS:
            settings = dict(_PER_IMAGE_SETTINGS[str(path)])
            updates = _slider_updates_from_settings(settings)
        else:
            settings = _settings(*slider_vals) if slider_vals else _neutral_settings()
            _PER_IMAGE_SETTINGS[str(path)] = settings
            updates = tuple(gr.update() for _ in range(_N_SLIDERS))
        try:
            before = _load_before(path)
            after, notes = _render_after(before, settings, return_notes=True)
            status = _status_line(path.name, settings)
            if notes:
                status = f"{status}\n\n{notes}"
            return (before, after, status, _photo_position(paths, path), *_nav_updates(paths, path)) + updates
        except Exception as e:
            return (None, None, f"Vista previa falló: {e}", _photo_position(paths, path), *_nav_updates(paths, path)) + tuple(gr.update() for _ in range(_N_SLIDERS))

    def make_pair_on_import(selected, *slider_vals):
        """Tras importar, omite el render en lotes grandes (congela pywebview).

        Las rutas ya las resolvió refresh_queue en el servidor: el navegador NO
        reenvía los cientos de FileData del drag&drop a este segundo evento.
        """
        paths = _LAST_QUEUE_PATHS
        limit = int(
            enh_logic.load_enhance_config().get("large_batch", {}).get("auto_preview_max_files", 60)
        )
        if len(paths) > limit:
            first = paths[0] if paths else None
            # En lotes grandes se renderiza SOLO la primera foto: el usuario
            # ve el antes/después, pero no se procesan las 400 ni se repintan
            # los 13 sliders (gr.skip).
            if first:
                try:
                    before = _load_before(Path(first))
                    after, notes = _render_after(
                        before, _settings(*slider_vals), return_notes=True
                    )
                    message = (
                        f"**{len(paths)} fotos en cola.** Vista previa de la primera: "
                        f"`{Path(first).name}`. Elige un preset y pulsa PROCESAR LOTE."
                    )
                    if notes:
                        message = f"{message}\n\n{notes}"
                    return (
                        before, after, message,
                        _photo_position(paths, first),
                        *_nav_updates(paths, first),
                    ) + tuple(gr.skip() for _ in range(_N_SLIDERS))
                except Exception as exc:
                    message = f"Vista previa no disponible: {exc}"
            else:
                message = "No hay fotos en cola."
            return (
                gr.skip(), gr.skip(), message,
                _photo_position(paths, first),
                *_nav_updates(paths, first),
            ) + tuple(gr.skip() for _ in range(_N_SLIDERS))
        return make_pair([selected] if selected else paths, None, selected, *slider_vals)

    def make_after(*args, **kwargs):
        files_val = kwargs.get("files_val")
        folder_val = kwargs.get("folder_val")
        selected = kwargs.get("selected")
        slider_vals = []

        if len(args) >= _N_SLIDERS + 3:
            files_val, folder_val, selected = args[0], args[1], args[2]
            slider_vals = list(args[3:])
        elif len(args) >= _N_SLIDERS + 1:
            selected = args[0]
            slider_vals = list(args[1:])
        elif len(args) > 0:
            selected = args[0]
            slider_vals = list(args[1:])

        paths = _queue_paths(files_val, folder_val) if (files_val or folder_val) else _LAST_QUEUE_PATHS
        if not paths:
            return None, "_Sube al menos una foto para ver el antes/después._"
        path = _pick_path(paths, selected)
        settings = _settings(*slider_vals) if slider_vals else _neutral_settings()
        if path is not None and str(path) in _PER_IMAGE_SETTINGS:
            settings["accumulated_presets"] = list(
                _PER_IMAGE_SETTINGS[str(path)].get("accumulated_presets", [])
            )
        if path is not None:
            _PER_IMAGE_SETTINGS[str(path)] = settings
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
    # Igual que FaceSwap: `upload` dispara una sola vez al terminar la subida.
    _queue_inputs = [input_files, input_folder]
    _refresh_outputs = [
        queue_summary, preview_photo, preview_status, preview_counter,
        btn_prev_photo, btn_next_photo,
    ]
    _mp_outputs = [preview_before, preview_after, preview_status, preview_counter, btn_prev_photo, btn_next_photo, denoise_slider, sharpen_slider, recover_slider, upscale_factor, exposure_slider, shadow_slider, highlight_slider, temperature_slider, grain_strength_slider, grain_size_slider, deband_slider, vibrance_slider, face_model]

    def _browse_pc_paths():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            paths = filedialog.askopenfilenames(
                title="Seleccionar fotos",
                filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff *.tif")],
            )
            root.destroy()
            return list(paths) if paths else None
        except Exception:
            return None

    if btn_browse_pc is not None:
        def _on_browse(selected, *slider_vals):
            paths = _browse_pc_paths()
            if not paths:
                return [gr.update()] * (len(_refresh_outputs) + len(_mp_outputs))
            try:
                refresh_result = list(refresh_queue(paths, None))
            except Exception:
                return [gr.update()] * (len(_refresh_outputs) + len(_mp_outputs))
            try:
                mp_result = list(make_pair_on_import(paths[0], *slider_vals))
            except Exception:
                mp_result = [gr.update()] * len(_mp_outputs)
            if len(refresh_result) < len(_refresh_outputs):
                refresh_result += [gr.update()] * (len(_refresh_outputs) - len(refresh_result))
            if len(mp_result) < len(_mp_outputs):
                mp_result += [gr.update()] * (len(_mp_outputs) - len(mp_result))
            return refresh_result + mp_result

        btn_browse_pc.click(
            fn=_on_browse,
            inputs=[preview_photo] + slider_inputs + [face_model],
            outputs=_refresh_outputs + _mp_outputs,
            queue=True,
            concurrency_limit=1,
            show_progress="full",
        )

    _mp_outputs = [preview_before, preview_after, preview_status, preview_counter, btn_prev_photo, btn_next_photo, denoise_slider, sharpen_slider, recover_slider, upscale_factor, exposure_slider, shadow_slider, highlight_slider, temperature_slider, grain_strength_slider, grain_size_slider, deband_slider, vibrance_slider, face_model]

    # Drop nativo (pywebview): recoge rutas completas sin subir File objects.
    native_drop_timer = gr.Timer(1.0)

    def poll_native_drop(selected, *slider_vals):
        import roop.globals as _g
        with _g.NATIVE_DROP_LOCK:
            paths = list(_g.NATIVE_DROP_QUEUES.get("enhance", []))
            _g.NATIVE_DROP_QUEUES["enhance"].clear()
        if not paths:
            return [gr.update()] * (len(_refresh_outputs) + len(_mp_outputs))
        print(f"[NATIVE-DROP] enhance: cargando {len(paths)} foto(s) por ruta")
        print(f"[NATIVE-DROP] enhance: primeras: {[Path(p).name for p in paths[:3]]}")
        try:
            refresh_result = list(refresh_queue(paths, None))
        except Exception as exc:
            print(f"[NATIVE-DROP] enhance: refresh_queue falló: {exc}")
            return [gr.update()] * (len(_refresh_outputs) + len(_mp_outputs))
        try:
            mp_result = list(make_pair_on_import(paths[0], *slider_vals))
        except Exception as exc:
            print(f"[NATIVE-DROP] enhance: vista previa falló: {exc}")
            mp_result = [gr.update()] * len(_mp_outputs)
        accepted = len(_LAST_QUEUE_PATHS)
        print(
            f"[NATIVE-DROP] enhance: aceptadas {accepted}/{len(paths)} · "
            f"{refresh_result[0]!r} · {mp_result[2]!r}"
        )
        try:
            if accepted:
                gr.Info(f"{accepted} foto(s) cargadas por arrastre")
            else:
                gr.Warning("El arrastre no trajo fotos compatibles")
        except Exception:
            pass
        # rellenar si falta
        if len(refresh_result) < len(_refresh_outputs):
            refresh_result += [gr.update()] * (len(_refresh_outputs) - len(refresh_result))
        if len(mp_result) < len(_mp_outputs):
            mp_result += [gr.update()] * (len(_mp_outputs) - len(mp_result))
        return refresh_result + mp_result

    native_drop_timer.tick(
        fn=poll_native_drop,
        inputs=[preview_photo] + slider_inputs + [face_model],
        outputs=_refresh_outputs + _mp_outputs,
        queue=True,
        concurrency_limit=1,
    )

    for comp in (input_files, input_folder):
        event = comp.upload(
            fn=refresh_queue,
            inputs=_queue_inputs,
            outputs=_refresh_outputs,
            show_progress="full",
        )
        # El segundo evento solo recibe rutas (State), NO los FileData del
        # drag&drop: el navegador no vuelve a procesar los 400 archivos.
        event.then(
            fn=make_pair_on_import,
            inputs=[preview_photo] + slider_inputs + [face_model],
            outputs=_mp_outputs,
            queue=True,
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="minimal",
        )

    preview_photo.input(
        fn=make_pair,
        inputs=pair_inputs,
        outputs=_mp_outputs,
        queue=True,
        concurrency_limit=1,
        trigger_mode="always_last",
        show_progress="minimal",
    )

    def move_photo(files_val, folder_val, selected, direction):
        paths = _LAST_QUEUE_PATHS if _LAST_QUEUE_PATHS else _queue_paths(files_val, folder_val)
        if not paths:
            return gr.update(), _photo_position(paths, None), *_nav_updates(paths, None)
        index = max(0, min(len(paths) - 1, _path_index(paths, selected) + direction))
        return gr.update(value=paths[index]), _photo_position(paths, paths[index]), *_nav_updates(paths, paths[index])

    def move_previous(files_val, folder_val, selected):
        return move_photo(files_val, folder_val, selected, -1)

    def move_next(files_val, folder_val, selected):
        return move_photo(files_val, folder_val, selected, 1)

    for button, handler in ((btn_prev_photo, move_previous), (btn_next_photo, move_next)):
        button.click(
            fn=handler,
            inputs=[input_files, input_folder, preview_photo],
            outputs=[preview_photo, preview_counter, btn_prev_photo, btn_next_photo],
            queue=False,
        ).then(
            fn=make_pair,
            inputs=pair_inputs,
            outputs=_mp_outputs,
            queue=True,
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="minimal",
        )

    btn_preview.click(
        fn=make_pair,
        inputs=pair_inputs,
        outputs=_mp_outputs,
        queue=True,
        concurrency_limit=1,
        show_progress="minimal",
    )

    # Cambiar cualquier ajuste re-renderiza SOLO el panel «Después».
    for slider in slider_inputs:
        slider.input(
            fn=make_after,
            inputs=pair_inputs,
            outputs=[preview_after, preview_status],
            queue=True,
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="minimal",
        )

    face_model.input(
        fn=make_after,
        inputs=pair_inputs,
        outputs=[preview_after, preview_status],
        queue=True,
        concurrency_limit=1,
        trigger_mode="always_last",
        show_progress="minimal",
    )

    # «Seleccionar todas»: la receta actual pasa a ser la de todas las fotos.
    def on_select_all_changed(enabled, files_val, folder_val, *slider_vals):
        paths = _queue_paths(files_val, folder_val)
        if not paths:
            return gr.update()
        settings = _settings(*slider_vals)
        if enabled:
            _apply_settings_to_all(paths, settings)
            return gr.update(
                value=f"**Seleccionar todas** activo: el ajuste actual se aplicará a las {len(paths)} fotos."
            )
        return gr.update(value="**Seleccionar todas** desactivado: cada foto conserva su ajuste.")

    select_all.change(
        fn=on_select_all_changed,
        inputs=[select_all, input_files, input_folder] + slider_inputs + [face_model],
        outputs=[preset_status],
        queue=False,
    )

    # ------------------------------------------------------------------
    # Cargar fotos de ejemplo (evita un botón sin función)
    # ------------------------------------------------------------------
    def load_sample():
        root = Path(__file__).resolve().parents[3]
        found = [str(root / rel) for rel in _SAMPLE_CANDIDATES if (root / rel).exists()]
        if not found:
            return gr.update(), "No hay fotos de ejemplo disponibles en diagnostics/out."
        return gr.update(value=found), gr.skip()

    btn_load_sample.click(
        fn=load_sample,
        inputs=[],
        outputs=[input_files, preview_status],
        queue=False,
    )

    # ------------------------------------------------------------------
    # Aplicar preset de fábrica a los sliders
    # ------------------------------------------------------------------
    def apply_preset(name: str, files_val, folder_val, selected, select_all=False, *current_vals):
        if not name:
            return tuple(gr.update() for _ in range(_N_SLIDERS)) + ("Selecciona uno de tus presets guardados.",)
        preset = load_presets().get(name)
        if preset is None:
            return tuple(gr.update() for _ in range(_N_SLIDERS)) + (f"Preset desconocido: {name}",)
        vals = preset["sliders"]
        preset_settings = _settings(
            vals["denoise_strength"], vals["sharpen_strength"],
            float(vals["recover_faces_strength"]) * 100.0, vals["upscale_factor"],
            vals["exposure"], vals["shadow"], vals["highlight"], vals["temperature"],
            vals["grain_strength"], vals["grain_size"], vals["deband_strength"],
            vals["vibrance"], vals["face_model"],
        )
        preset_settings["accumulated_presets"] = [name]
        if current_vals:
            base_settings = _settings(*current_vals)
        else:
            base_settings = _neutral_settings()
        paths = _queue_paths(files_val, folder_val)
        path = _pick_path(paths, selected) if paths else None
        if path is not None and str(path) in _PER_IMAGE_SETTINGS:
            base_settings["accumulated_presets"] = list(
                _PER_IMAGE_SETTINGS[str(path)].get("accumulated_presets", [])
            )
        if name in (base_settings.get("accumulated_presets") or []):
            settings = base_settings
        else:
            settings = _merge_settings(base_settings, preset_settings)
        paths = _queue_paths(files_val, folder_val)
        path = _pick_path(paths, selected) if paths else None
        if paths:
            if select_all:
                _apply_settings_to_all(paths, settings)
            elif path is not None:
                _PER_IMAGE_SETTINGS[str(path)] = settings
        updates = _slider_updates_from_settings(settings)
        suffix = " · mismo ajuste para todas" if select_all else ""
        pila = settings.get("accumulated_presets") or []
        pila_txt = (" · pila: " + " + ".join(pila)) if pila else ""
        if settings.get("stack_capped"):
            pila_txt += " · tope de pila en " + ", ".join(
                key.replace("_strength", "") for key in settings["stack_capped"]
            )
        if preset_stack is not None:
            return updates + (f"Preset «{name}» acumulado{suffix}. {preset.get('description', '')}{pila_txt}", gr.update(value=_stack_html(pila)))
        return updates + (f"Preset «{name}» acumulado{suffix}. {preset.get('description', '')}{pila_txt}",)

    preset_apply_btn.click(
        fn=apply_preset,
        inputs=[preset_dropdown, input_files, input_folder, preview_photo, select_all] + slider_inputs + [face_model],
        outputs=(
            [denoise_slider, sharpen_slider, recover_slider, upscale_factor,
             exposure_slider, shadow_slider, highlight_slider, temperature_slider,
             grain_strength_slider, grain_size_slider,
             deband_slider, vibrance_slider, face_model, preset_status]
            + ([preset_stack] if preset_stack is not None else [])
        ),
        queue=False,
    ).then(
        # Solo se repinta «DESPUÉS»: el «ANTES» es la foto original y no cambia.
        fn=make_after,
        inputs=pair_inputs,
        outputs=[preview_after, preview_status],
        queue=True,
        concurrency_limit=1,
        trigger_mode="always_last",
        show_progress="minimal",
    )

    def _button_handler(name):
        def select_and_apply(files_val, folder_val, selected, select_all_value, *slider_vals):
            return apply_preset(name, files_val, folder_val, selected, select_all_value, *slider_vals)
        return select_and_apply

    preset_outputs = [
        denoise_slider, sharpen_slider, recover_slider, upscale_factor,
        exposure_slider, shadow_slider, highlight_slider, temperature_slider,
        grain_strength_slider, grain_size_slider, deband_slider, vibrance_slider,
        face_model, preset_status,
    ]
    if preset_stack is not None:
        preset_outputs = preset_outputs + [preset_stack]
    pair_outputs = [preview_before, preview_after, preview_status, preview_counter, btn_prev_photo, btn_next_photo] + slider_inputs + [face_model]
    for name, button in preset_buttons.items():
        button.click(
            fn=_button_handler(name),
            inputs=[input_files, input_folder, preview_photo, select_all] + slider_inputs + [face_model],
            outputs=preset_outputs,
            queue=False,
        ).then(
            # Al pulsar un preset solo cambia el preview de «cómo quedará».
            fn=make_after,
            inputs=pair_inputs,
            outputs=[preview_after, preview_status],
            queue=True,
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="minimal",
        )

    # Reset sliders a valores por defecto
    def reset_sliders(files_val, folder_val, selected):
        from .logic import load_enhance_config
        d = load_enhance_config().get("default_slider_values", {})
        recover = d.get("recover_faces_strength", 0.8)
        recover_pct = (recover * 100) if isinstance(recover, (int, float)) and recover <= 1.0 else float(recover)
        values = (
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
        )
        paths = _queue_paths(files_val, folder_val)
        if paths:
            _PER_IMAGE_SETTINGS[str(_pick_path(paths, selected))] = _neutral_settings()
        if preset_stack is not None:
            return values + ("Sliders restablecidos a valores por defecto.", gr.update(value=_stack_html([])))
        return values + ("Sliders restablecidos a valores por defecto.",)

    if btn_clear_stack is not None:
        btn_clear_stack.click(
            fn=reset_sliders,
            inputs=[input_files, input_folder, preview_photo],
            outputs=(
                [denoise_slider, sharpen_slider, recover_slider, upscale_factor,
                 exposure_slider, shadow_slider, highlight_slider, temperature_slider,
                 grain_strength_slider, grain_size_slider,
                 deband_slider, vibrance_slider, face_model, preset_status]
                + ([preset_stack] if preset_stack is not None else [])
            ),
            queue=False,
        ).then(
            fn=make_pair,
            inputs=pair_inputs,
            outputs=pair_outputs,
            queue=True,
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="minimal",
        )

    btn_reset_sliders = ui_components.get("btn_reset_sliders")
    if btn_reset_sliders is not None:
        btn_reset_sliders.click(
            fn=reset_sliders,
            inputs=[input_files, input_folder, preview_photo],
            outputs=(
                [denoise_slider, sharpen_slider, recover_slider, upscale_factor,
                 exposure_slider, shadow_slider, highlight_slider, temperature_slider,
                 grain_strength_slider, grain_size_slider,
                 deband_slider, vibrance_slider, face_model, preset_status]
                + ([preset_stack] if preset_stack is not None else [])
            ),
            queue=False,
        ).then(
            fn=make_pair,
            inputs=pair_inputs,
            outputs=pair_outputs,
            queue=True,
            concurrency_limit=1,
            trigger_mode="always_last",
            show_progress="minimal",
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
            return gr.update(), "Escribe un nombre para el preset."
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
            preset_file = save_custom_preset(name, data["sliders"])
            customs = [key for key, preset in load_presets().items() if not preset["_factory"]]
            return gr.update(choices=customs, value=name), f"Preset «{name}» guardado en {preset_file}."
        except Exception as e:
            return gr.update(), f"Error al guardar preset: {e}"

    preset_save_btn.click(
        fn=save_preset,
        inputs=[
            preset_save_name, denoise_slider, sharpen_slider, recover_slider,
            upscale_factor, exposure_slider, shadow_slider, highlight_slider,
            temperature_slider, grain_strength_slider, grain_size_slider,
            deband_slider, vibrance_slider, face_model,
        ],
        outputs=[preset_dropdown, preset_status],
        queue=False,
    )

    # ------------------------------------------------------------------
    # Botón nativo "Buscar fotos en el PC"
    # ------------------------------------------------------------------
    def _browse_files_native():
        """Diálogo nativo Windows para seleccionar múltiples archivos."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            files = filedialog.askopenfilenames(
                title="Seleccionar fotos (máx. 30 recomendado)",
                filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("Todos", "*.*")]
            )
            root.destroy()
        except Exception as exc:
            return gr.update(), f"Selector no disponible ({exc}). Usa arrastrar o carpeta."
        if not files:
            return gr.update(), "Selección cancelada."
        paths = list(files)
        if len(paths) > 30:
            return gr.update(), f"Seleccionadas {len(paths)} fotos; se recomienda máx. 30. Usa 'Carpeta' para lotes grandes."
        return gr.update(value=paths), f"{len(paths)} fotos seleccionadas nativamente."

    if "btn_browse_files" in ui_components:
        btn_browse_files = ui_components["btn_browse_files"]
        btn_browse_files.click(
            fn=_browse_files_native,
            inputs=[],
            outputs=[input_files, preview_status],
            queue=False,
        )

    # ------------------------------------------------------------------
    # Elegir / abrir la carpeta de destino
    # ------------------------------------------------------------------
    def browse_output_folder():
        """Diálogo nativo de Windows, mismo patrón que el selector de FaceSwap."""
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            chosen = filedialog.askdirectory(
                title="Selecciona la carpeta de destino de las fotos mejoradas"
            )
            root.destroy()
        except Exception as exc:
            return gr.update(), f"Selector no disponible ({exc}). Escribe la ruta en el campo."
        if not chosen:
            return gr.update(), "Selección de carpeta cancelada."
        try:
            enh_logic.save_persisted_output_dir(chosen)
        except Exception:
            pass
        return gr.update(value=chosen), f"Carpeta elegida: `{chosen}`."

    if btn_browse_folder is not None:
        btn_browse_folder.click(
            fn=browse_output_folder,
            inputs=[],
            outputs=[output_path, folder_status],
            queue=False,
        )

    def open_output_folder(raw_path):
        target, _ = _resolve_output_dir(raw_path)
        try:
            if os.name == "nt":
                os.startfile(target)
            else:
                subprocess.Popen(["xdg-open", target])
            msg = f"Carpeta abierta: {target}"
        except Exception as e:
            msg = f"No se pudo abrir la carpeta: {e}"
        return msg

    btn_open_folder.click(
        fn=open_output_folder,
        inputs=[output_path],
        outputs=folder_status,
        queue=False,
    )


    # Persistir la carpeta de destino al cambiarla (de sesión en sesión).
    def _persist_output_path(raw):
        chosen_dir, _ = _resolve_output_dir(raw)
        try:
            enh_logic.save_persisted_output_dir(chosen_dir)
        except Exception:
            pass
        return gr.update(value=chosen_dir)

    if output_path is not None:
        output_path.change(
            fn=_persist_output_path,
            inputs=[output_path],
            outputs=[output_path],
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
        select_all=False, output_path=None,
    ):
        paths = _queue_paths(files, folder) if (files or folder) else _LAST_QUEUE_PATHS
        if not paths:
            yield "<div class='enhance-status'>No se seleccionaron archivos. Arrastra fotos o usa «Buscar fotos en el PC».</div>", ""
            return

        t0 = time.time()
        enh_logic.clear_cancel()

        settings = _settings(
            denoise, sharpen, recover, upscale, exposure, shadow, highlight,
            temperature, grain_strength, grain_size, deband, vibrance, face_model,
        )
        chosen_dir, dest_msg = _resolve_output_dir(output_path)
        try:
            enh_logic.save_persisted_output_dir(chosen_dir)
        except Exception:
            pass
        output_dir = Path(chosen_dir)
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
                    # «Seleccionar todas» gana: una sola receta para todo el lote.
                    autopilot=bool(auto_mode) and not bool(select_all),
                    settings_by_path=None if select_all else {
                        str(p): _PER_IMAGE_SETTINGS.get(str(p)) for p in paths
                    },
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

        _last_pct, _last_phase = -1, None
        try:
            last_heartbeat = time.time()
            while True:
                try:
                    payload = updates.get(timeout=0.5)
                except queue.Empty:
                    # Heartbeat: yield periodically to keep UI alive
                    if time.time() - last_heartbeat > 2.0:
                        last_heartbeat = time.time()
                        # Small yield to process events
                        yield gr.skip(), gr.skip()
                    else:
                        yield gr.skip(), gr.skip()
                    continue
                if payload is None:
                    break
                phase = str(payload.get("phase", ""))
                detail = str(payload.get("detail", ""))
                pct = int(float(payload.get("progress", 0.0)) * 100)
                if 0 < pct < 100:
                    pct = max(1, pct)
                # Evita inundar la UI con un yield por cada paso; solo repinta
                # cuando cambia el porcentaje o la fase.
                if pct == _last_pct and phase == _last_phase:
                    yield gr.skip(), gr.skip()
                    continue
                _last_pct, _last_phase = pct, phase
                status = f"<div class='enhance-status'>Procesando {pct}% · {phase}: {detail}</div>"
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
            f" · salida: `{output_dir}`"
        )
        if "Se mantiene" in dest_msg:
            summary += f" · aviso: {dest_msg}"
        if select_all:
            summary += " · mismo ajuste para todas"
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
            format_dropdown, apply_filters_ck, auto_mode, select_all, output_path,
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
