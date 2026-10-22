"""Session-scoped visual selection callbacks, independent of generation."""
import traceback

import gradio as gr
import numpy as np
from PIL import Image

from . import state


def normalize_image(value):
    if isinstance(value, dict):
        value = value.get("background")
    if isinstance(value, str):
        with Image.open(value) as image:
            return image.convert("RGB")
    if isinstance(value, np.ndarray):
        return Image.fromarray(value).convert("RGB")
    return value.convert("RGB") if isinstance(value, Image.Image) else None


def _session(value):
    return value if isinstance(value, state.SelectionSession) else state.SelectionSession()


def _empty_choices():
    return gr.update(choices=[], value=[], visible=False)


def _skip(count):
    return tuple(gr.skip() for _ in range(count))


def _matches(image, analysis):
    from roop.img_editor import element_selection as es
    return (image is not None and analysis is not None
            and tuple(image.size) == tuple(analysis.size)
            and es.image_fingerprint(image) == analysis.fingerprint)


def on_analyze(img_data, selection_state):
    session = _session(selection_state)
    image = normalize_image(img_data)
    if image is None:
        with session.lock:
            if session.image_bound and session.current_fingerprint is not None:
                return _skip(4)
        return None, _empty_choices(), session, "Sube una imagen para analizar"
    from roop.img_editor import element_selection as es
    fingerprint = es.image_fingerprint(image)

    # Share the reservation with generation before importing/loading any model.
    with state._generation_lock:
        if state.is_generating:
            return (*_skip(3), "Hay una transformación o análisis en proceso")
        with session.lock:
            if session.image_bound and session.current_fingerprint != fingerprint:
                return _skip(4)
            if not session.image_bound:
                # First API request may have no preceding frontend change
                # event. Subsequent requests cannot overwrite that binding.
                session.current_fingerprint = fingerprint
                session.image_bound = True
            revision = session.invalidate()
        state.is_generating = True
    try:
        from roop.comfy_workflows import get_comfyui_url
        from roop.img_editor.flux_edit_comfy_client import _queue_is_idle
        if not _queue_is_idle(get_comfyui_url()):
            with session.lock:
                if session.revision != revision:
                    return _skip(4)
                return (None, _empty_choices(), session,
                        "No se puede analizar: ComfyUI está ocupado o no se pudo comprobar su cola")
        analysis = es.analyze_elements(image)
        with session.lock:
            if session.revision != revision:
                return _skip(4)
            if not _matches(image, analysis):
                raise ValueError("el análisis no corresponde a la imagen")
            overlay = es.render_selection(image, analysis, [])
            session.analysis = analysis
            session.selected_ids = []
            choices = [(f"{region.label} ({region.coverage:.1%})", region.id)
                       for region in analysis.regions]
            message = (f"{analysis.message}. Marca una o varias zonas."
                       if choices else analysis.message or "No se detectaron zonas")
            return (overlay, gr.update(choices=choices, value=[], visible=bool(choices)),
                    session, message)
    except Exception as exc:
        traceback.print_exc()
        with session.lock:
            if session.revision != revision:
                return _skip(4)
            return None, _empty_choices(), session, f"Error analizando: {exc}"
    finally:
        with state._generation_lock:
            state.is_generating = False
            state.generation_started_at = None


def on_select(selected_ids, img_data, selection_state):
    session = _session(selection_state)
    image = normalize_image(img_data)
    from roop.img_editor import element_selection as es
    with session.lock:
        analysis = session.analysis
        if not _matches(image, analysis):
            session.invalidate()
            return None, session, "La foto cambió; vuelve a analizarla"
        ids = list(dict.fromkeys(str(value) for value in selected_ids or []))
        if ids:
            valid, reason = es.validate_selection(image, analysis, ids)
            if not valid:
                return gr.skip(), session, f"Selección inválida: {reason}"
        overlay = es.render_selection(image, analysis, ids)
        session.selected_ids = ids
        return overlay, session, f"{len(ids)} zona(s) seleccionada(s)"


def on_point_select(img_data, selection_state, evt: gr.SelectData):
    session = _session(selection_state)
    image = normalize_image(img_data)
    from roop.img_editor import element_selection as es
    with session.lock:
        analysis = session.analysis
        if not _matches(image, analysis):
            session.invalidate()
            return None, _empty_choices(), session, "La foto cambió; vuelve a analizarla"
        try:
            x, y = evt.index[:2]
            region = es.choose_at_point(analysis, int(x), int(y))
        except (TypeError, ValueError, IndexError):
            return (*_skip(3), "Punto de selección inválido")
        if region is None:
            candidates = es.regions_at_point(analysis, int(x), int(y))
            message = ("Hay varias zonas superpuestas: "
                       + ", ".join(candidate.label for candidate in candidates)
                       + ". Elige en la lista."
                       if candidates else "No hay una región detectada en ese punto")
            return (*_skip(3), message)
        ids = list(session.selected_ids)
        if region.id in ids:
            ids.remove(region.id)
        else:
            ids.append(region.id)
        valid, reason = es.validate_selection(image, analysis, ids) if ids else (True, "")
        if not valid:
            return (*_skip(3), f"Selección inválida: {reason}")
        overlay = es.render_selection(image, analysis, ids)
        session.selected_ids = ids
        return (overlay, gr.update(value=ids), session,
                f"{len(ids)} zona(s) seleccionada(s)")


def on_new_image(img_data, selection_state):
    session = _session(selection_state)
    from roop.img_editor.element_selection import image_fingerprint
    image = normalize_image(img_data)
    session.replace_image(image_fingerprint(image) if image is not None else None)
    return None, _empty_choices(), session, "Selección limpia (foto nueva)"


def use_result_as_input(image, selection_state):
    session = _session(selection_state)
    normalized = normalize_image(image)
    if normalized is None:
        return (*_skip(4), "Primero genera un resultado")
    from roop.img_editor.element_selection import image_fingerprint
    session.replace_image(image_fingerprint(normalized))
    return normalized, None, _empty_choices(), session, "Selección limpia (foto reutilizada)"
