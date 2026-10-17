import gradio as gr
import os
import sys
import cv2
import numpy as np
import time
import threading
import json
import roop.globals
import ui.globals
import ui.tabs.faceswap.state as state
import ui.tabs.faceswap.logic as logic
import roop.utilities as util
import roop.face_util as face_util

DETECTED_FACES_GALLERY_COLUMNS = 10
is_deleting_input_face = False
is_deleting_target_face = False

_target_load_lock = threading.Lock()

def _write_console(msg: str):
    for fd in (1, 2):
        try:
            os.write(fd, (msg + "\n").encode("utf-8", errors="replace"))
            return
        except Exception:
            pass
    try:
        sys.__stdout__.write(msg + "\n")
        sys.__stdout__.flush()
        return
    except Exception:
        pass

_log = _write_console
_log("[TEST_CONSOLE] _write_console desde module-level de events.py OK")


def _resolve_gallery_index(raw_index, total_items, columns=DETECTED_FACES_GALLERY_COLUMNS):
    """Convierte el índice de Gradio (int o fila/columna) a índice lineal."""
    if raw_index is None:
        return None

    if isinstance(raw_index, int):
        return raw_index if 0 <= raw_index < total_items else None

    if isinstance(raw_index, (tuple, list)):
        if len(raw_index) == 1 and isinstance(raw_index[0], int):
            idx = raw_index[0]
            return idx if 0 <= idx < total_items else None
        if len(raw_index) >= 2 and all(isinstance(x, int) for x in raw_index[:2]):
            row, col = raw_index[0], raw_index[1]
            idx = (row * columns) + col
            return idx if 0 <= idx < total_items else None

    return None


def _normalize_face_thumb(image, size=160):
    if image is None:
        return None
    try:
        arr = np.array(image)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        elif arr.ndim != 3:
            return None

        h, w = arr.shape[:2]
        if h <= 0 or w <= 0:
            return None

        scale = min(size / w, size / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.full((size, size, 3), 24, dtype=np.uint8)
        x = (size - new_w) // 2
        y = (size - new_h) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        return canvas
    except Exception as e:
        print(f"[DEBUG] _normalize_face_thumb error: {e}")
        return None


def _warmup_face_analyser_async():
    """Precarga el analizador para evitar multi-click en la primera detección."""
    try:
        if face_util.FACE_ANALYSER is None:
            print("[FaceWarmup] Precargando FaceAnalysis...")
            face_util.get_face_analyser()
    except Exception as e:
        print(f"[FaceWarmup] Warmup falló: {e}")


# Banco de caras destino por archivo (máx 3): frontal + perfil + ... El tracker
# usa max-sim contra el banco, así un corte a otro ángulo re-engancha con su ref.
_MAX_REFS_PER_FILE = 3

def _get_file_bank(filename, source_path=None):
    """Lista de refs destino de un archivo (migra forma legacy dict → lista)."""
    try:
        refs = getattr(roop.globals, 'selected_face_references', {}) or {}
        bank = refs.get(f"selected_face_ref_{filename}")
        if isinstance(bank, dict):
            bank = [bank]
        if isinstance(bank, (list, tuple)):
            valid = [r for r in bank if isinstance(r, dict) and r.get('embedding') is not None]
            if source_path:
                exact = os.path.normcase(os.path.abspath(source_path))
                return [r for r in valid if r.get('source_path') == exact]
            return valid
    except Exception:
        pass
    return []

def _store_destination_reference(filename, face_obj, source_path=None):
    """Acumula ref destino para un archivo. Devuelve (n, estado):
    ok / dup (ya estaba) / full (3/3)."""
    if not hasattr(roop.globals, 'selected_face_references'):
        roop.globals.selected_face_references = {}
    refs = roop.globals.selected_face_references
    key = f"selected_face_ref_{filename}"
    bank = _get_file_bank(filename)
    try:
        new_bbox = np.asarray(face_obj.bbox, dtype=np.float32).reshape(-1)
        if new_bbox.size != 4 or not np.isfinite(new_bbox).all():
            new_bbox = None
    except (TypeError, ValueError):
        new_bbox = None
    normalized_path = os.path.normcase(os.path.abspath(source_path)) if source_path else None
    file_bank = _get_file_bank(filename, source_path) if source_path else bank
    for r in file_bank:
        try:
            old_bbox = np.asarray(r.get('bbox'), dtype=np.float32).reshape(-1)
            if (new_bbox is not None and old_bbox.size == 4
                    and r.get('source_path') == normalized_path
                    and np.array_equal(old_bbox, new_bbox)):
                return len(file_bank), 'dup'
        except (TypeError, ValueError):
            continue
    if len(file_bank) >= _MAX_REFS_PER_FILE:
        return len(file_bank), 'full'
    bank.append({
        'face_obj': face_obj,
        'embedding': face_obj.embedding,
        'bbox': face_obj.bbox,
        'source_path': normalized_path,
    })
    refs[key] = bank
    return len(file_bank) + 1, 'ok'


def _register_destination_face(entry, face_obj, face_thumb):
    """Persist the file selection before updating the display gallery."""
    filename = os.path.basename(entry.filename)
    count, status = _store_destination_reference(filename, face_obj, entry.filename)
    if status == 'ok':
        roop.globals.TARGET_FACES.append(face_obj)
        thumb = util.convert_to_gradio(face_thumb, is_rgb=True)
        if thumb is not None:
            ui.globals.ui_target_thumbs.append(thumb)
    return filename, count, status


def _face_detection_key(file_path, frame_idx):
    """A detection belongs to one destination file and one frame."""
    return os.path.normcase(os.path.abspath(file_path)), int(frame_idx)

def _drop_source_file_faces(global_idx):
    """Quita de la galería la cara elegida y las demás del mismo archivo."""
    facesets = roop.globals.INPUT_FACESETS
    thumbs = ui.globals.ui_input_thumbs
    if not (0 <= global_idx < len(facesets)):
        return 0
    name = getattr(facesets[global_idx], 'name', None)
    if not name:
        facesets.pop(global_idx)
        if global_idx < len(thumbs):
            thumbs.pop(global_idx)
        return 1
    kept_faces, kept_thumbs = [], []
    removed = 0
    for index, faceset in enumerate(facesets):
        if getattr(faceset, 'name', None) == name:
            removed += 1
            continue
        kept_faces.append(faceset)
        if index < len(thumbs):
            kept_thumbs.append(thumbs[index])
    facesets[:] = kept_faces
    thumbs[:] = kept_thumbs
    print(f"[UI] Origen eliminado: {name} ({removed} cara(s))", flush=True)
    return removed


def _purge_bank_by_face(face_obj):
    """Remove only the gallery selection being deleted, even across equal embeddings."""
    try:
        refs = getattr(roop.globals, 'selected_face_references', {}) or {}
        for key in list(refs.keys()):
            bank = refs.get(key)
            items = bank if isinstance(bank, (list, tuple)) else ([bank] if isinstance(bank, dict) else [])
            kept = [r for r in items if r.get('face_obj') is not face_obj]
            if len(kept) != len(items):
                if kept:
                    refs[key] = kept
                else:
                    refs.pop(key, None)
    except Exception as e:
        print(f"[UI] Purga de banco omitida: {e}")

def _file_indicator_text():
    """Texto del indicador de archivo + nº de caras destino (🎯 k/3)."""
    try:
        if not state.list_files_process:
            return "📂 **Archivo:** Ninguno seleccionado"
        entry = state.list_files_process[state.selected_preview_index]
        base = f"📂 **Archivo ({state.selected_preview_index+1}/{len(state.list_files_process)}):** {os.path.basename(entry.filename)}"
        n = len(_get_file_bank(os.path.basename(entry.filename), entry.filename))
        return f"{base} — 🎯 destino: {n}/{_MAX_REFS_PER_FILE}"
    except Exception:
        return "📂 **Archivo:** Ninguno seleccionado"


def on_face_selection_click(evt: gr.SelectData, frame_num):
    """Maneja el clic en galería, sincroniza el slider, añade la cara y actualiza el feedback visual."""
    if not evt or evt.index is None:
        return gr.update(), gr.update()

    total_faces = len(state.SELECTION_FACES_DATA) if state.SELECTION_FACES_DATA else 0
    face_index = _resolve_gallery_index(evt.index, total_faces, columns=4)
    
    if face_index is None or state.SELECTION_FACES_DATA is None:
        return gr.update(), gr.update()

    state.TEMP_SELECTED_FACE_INDEX = face_index

    try:
        face_data = state.SELECTION_FACES_DATA[face_index]
        face_obj = face_data[0]
        
        entry = state.list_files_process[state.selected_preview_index]
        filename, n, st = _register_destination_face(entry, face_obj, face_data[1])
        if st == 'ok':
            print(f"[UI] ✅ Cara añadida — destino {filename}: {n}/{_MAX_REFS_PER_FILE}" + (" (tip: añade frontal + perfil para vídeos con cortes)" if n == 1 else ""))
        elif st == 'dup':
            print(f"[UI] ℹ️ Esa cara ya es destino de {filename} ({n}/{_MAX_REFS_PER_FILE})")
        else:
            print(f"[UI] ℹ️ {filename} ya tiene {n}/{_MAX_REFS_PER_FILE} caras destino")

        # Actualizar feedback visual con caja verde
        entry = state.list_files_process[state.selected_preview_index]
        from roop.capturer import get_video_frame, get_image_frame
        frame = get_video_frame(entry.filename, int(frame_num or 1)) if util.is_video(entry.filename) else get_image_frame(entry.filename)
        boxed_frame = logic.draw_face_boxes(frame, state.SELECTION_FACES_DATA, selected_idx=face_index)
        boxed_frame_gradio = util.convert_to_gradio(boxed_frame)
        
    except Exception as e:
        print(f"[FaceSelection] Error: {e}")
        boxed_frame_gradio = gr.update()

    return gr.update(maximum=max(1, total_faces), value=face_index + 1), boxed_frame_gradio


def on_preview_click(evt: gr.SelectData, frame_num):
    """Maneja el clic directo en la imagen de previsualización para seleccionar caras."""
    if not state.list_files_process:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    entry = state.list_files_process[state.selected_preview_index]
    frame_idx = max(1, int(frame_num or 1))

    # 1. AUTO-DETECCIÓN: Si no hay caras o cambiamos de frame, detectamos primero
    detection_key = _face_detection_key(entry.filename, frame_idx)
    is_new_frame = getattr(state, 'CURRENT_DETECTED_KEY', None) != detection_key
    if not state.SELECTION_FACES_DATA or is_new_frame:
        print(f"[FaceSelection] Auto-detectando caras para frame {frame_idx} (Nuevo={is_new_frame})...")
        detected_faces = logic.extract_face_images(entry.filename, (True, frame_idx), target_face_detection=True, ui_padding=0.5)
        state.ALL_DETECTED_FACES_DATA = detected_faces
        state.SELECTION_FACES_DATA = detected_faces
        state.CURRENT_DETECTED_FRAME = frame_idx
        state.CURRENT_DETECTED_KEY = detection_key

    if not state.SELECTION_FACES_DATA:
        print("[UI] ⚠️ No se detectaron caras.")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # 2. Buscar cara que coincida con las coordenadas del clic [x, y]
    # Gradio proporciona coordenadas relativas a la visualización, necesitamos mapearlas o usar distancias relativas
    click_x, click_y = evt.index[0], evt.index[1]
    
    # Obtener dimensiones reales de la imagen para normalizar distancias si es necesario
    from roop.capturer import get_image_frame, get_video_frame
    frame_orig = get_video_frame(entry.filename, frame_idx) if util.is_video(entry.filename) else get_image_frame(entry.filename)
    if frame_orig is None:
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    h_orig, w_orig = frame_orig.shape[:2]
    
    best_face_idx = -1
    min_dist_norm = float('inf')

    for idx, face_data in enumerate(state.SELECTION_FACES_DATA):
        face_obj = face_data[0]
        x1, y1, x2, y2 = face_obj.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Distancia normalizada para ignorar diferencias de escala UI vs Original
        dist_norm = np.sqrt(((click_x - cx)/w_orig)**2 + ((click_y - cy)/h_orig)**2)
        
        # Si el clic está dentro del bbox (con margen) o es el centro más cercano
        margin = max(w_orig, h_orig) * 0.05
        is_inside = (x1 - margin) <= click_x <= (x2 + margin) and (y1 - margin) <= click_y <= (y2 + margin)
        
        if is_inside and dist_norm < min_dist_norm:
            min_dist_norm = dist_norm
            best_face_idx = idx

    # Fallback: si no está "dentro", coger el centro más cercano en un radio razonable
    if best_face_idx == -1:
        for idx, face_data in enumerate(state.SELECTION_FACES_DATA):
            face_obj = face_data[0]
            x1, y1, x2, y2 = face_obj.bbox
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            dist_norm = np.sqrt(((click_x - cx)/w_orig)**2 + ((click_y - cy)/h_orig)**2)
            if dist_norm < 0.15 and dist_norm < min_dist_norm: # Radio del 15% de la imagen
                min_dist_norm = dist_norm
                best_face_idx = idx

    if best_face_idx != -1:
        face_data = state.SELECTION_FACES_DATA[best_face_idx]
        face_obj = face_data[0]
        
        # Sincronizar índice para preview del enhancer
        state.TEMP_SELECTED_FACE_INDEX = best_face_idx

        filename, n, st = _register_destination_face(entry, face_obj, face_data[1])
        if st == 'ok':
            print(f"[UI] ✅ Cara añadida a Destino ({filename}: {n}/{_MAX_REFS_PER_FILE})" + (" (tip: añade frontal + perfil para vídeos con cortes)" if n == 1 else ""))
        elif st == 'dup':
            print(f"[UI] ℹ️ Esa cara ya es destino de {filename} ({n}/{_MAX_REFS_PER_FILE})")
        else:
            print(f"[UI] ℹ️ {filename} ya tiene {n}/{_MAX_REFS_PER_FILE} caras destino")

        total_faces = len(ui.globals.ui_target_thumbs)
        
        # Feedback visual con caja verde
        from roop.capturer import get_video_frame, get_image_frame
        frame = get_video_frame(entry.filename, frame_idx) if util.is_video(entry.filename) else get_image_frame(entry.filename)
        boxed_frame = logic.draw_face_boxes(frame, state.SELECTION_FACES_DATA, selected_idx=best_face_idx)
        
        return (
            logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"),
            logic.update_pagination_info(ui.globals.ui_target_thumbs, "target"),
            *logic.update_pagination_buttons(total_faces, "target"),
            util.convert_to_gradio(boxed_frame)
        )

    return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()


# Variables de estado locales para borrado
selected_input_face_idx = None
selected_target_face_idx = None

def wire_events(ui_comp):
    """Conecta los componentes con la lógica"""
    _log("[TEST_CONSOLE] wire_events() llamado - _write_console funciona desde aqui")

    logic.apply_optimal_faceswap_defaults()

    # Warmup: precargar FaceAnalysis al iniciar la pestaña
    try:
        _warmup_face_analyser_async()
    except Exception as e:
        print(f"[FaceSwap] Warmup error: {e}")
    
    # --- FILTRADO ---
    def on_filter_change(gender, size):
        if not hasattr(state, 'ALL_DETECTED_FACES_DATA'):
            return gr.update()
        filtered = logic.get_filtered_faces(state.ALL_DETECTED_FACES_DATA, gender, size)
        state.SELECTION_FACES_DATA = filtered
        thumbs = [_normalize_face_thumb(util.convert_to_gradio(img, is_rgb=True), 300) for _, img in filtered]
        return gr.Gallery(value=thumbs, columns=min(4, len(thumbs)) if thumbs else 1)

    ui_comp["gender_filter"].change(on_filter_change, [ui_comp["gender_filter"], ui_comp["size_filter"]], [ui_comp["face_selection"]])
    ui_comp["size_filter"].change(on_filter_change, [ui_comp["gender_filter"], ui_comp["size_filter"]], [ui_comp["face_selection"]])

    # --- ENHANCER PREVIEW ---
    def on_enhancer_preview_click(frame_num, enhancer, blend):
        if not state.list_files_process:
            return gr.update(), gr.update()
        
        face_idx = getattr(state, 'TEMP_SELECTED_FACE_INDEX', None)
        if face_idx is None: 
            return gr.update(), gr.Markdown("### ⚠️ Selecciona una cara primero en la galería superior")
        
        entry = state.list_files_process[state.selected_preview_index]
        preview_img, msg = logic.get_enhancer_preview(entry.filename, int(frame_num or 1), face_idx, enhancer, blend)
        if preview_img is not None:
            return gr.update(value=preview_img, visible=True), gr.update(value=msg)
        return gr.update(visible=False), gr.update(value=msg)

    ui_comp["bt_enhancer_preview"].click(
        on_enhancer_preview_click, 
        [ui_comp["preview_frame_num"], ui_comp["enhancer"], ui_comp["enhancer_blend"]],
        [ui_comp["preview_result"], ui_comp["face_detection_title"]]
    )

    # Tracking de clics en galerías finales (para borrar)
    def on_input_face_select(evt: gr.SelectData):
        global selected_input_face_idx
        total_items = len(logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"))
        rel_index = _resolve_gallery_index(evt.index, total_items, columns=3)
        if rel_index is not None:
            selected_input_face_idx = state.current_input_page * state.FACES_PER_PAGE + rel_index
            # Solo marca la miniatura para inspección o eliminación.
            # La source del swap se elige automáticamente frente al destino.
        return logic.update_pagination_info(ui.globals.ui_input_thumbs, "input")

    def on_input_face_delete(evt=None):
        global is_deleting_input_face, selected_input_face_idx
        if is_deleting_input_face:
            return gr.update(), gr.update(), gr.update(), gr.update()
        
        raw_idx = None
        if evt is not None:
            raw_idx = evt.index if hasattr(evt, "index") else (evt["index"] if isinstance(evt, dict) and "index" in evt else evt)
        
        total_items = len(logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"))
        rel_index = _resolve_gallery_index(raw_idx, total_items, columns=3)
        
        if rel_index is None:
            if selected_input_face_idx is not None:
                global_idx = selected_input_face_idx
            else:
                return gr.update(), gr.update(), gr.update(), gr.update()
        else:
            global_idx = state.current_input_page * state.FACES_PER_PAGE + rel_index
    
        is_deleting_input_face = True
        try:
            if _drop_source_file_faces(global_idx):
                selected_input_face_idx = None
                total_faces = len(ui.globals.ui_input_thumbs)
                max_pages = max(0, (total_faces - 1) // state.FACES_PER_PAGE)
                if state.current_input_page > max_pages:
                    state.current_input_page = max_pages
        finally:
            is_deleting_input_face = False
        return (
            logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"), 
            logic.update_pagination_info(ui.globals.ui_input_thumbs, "input"), 
            *logic.update_pagination_buttons(total_faces, "input")
        )

    ui_comp["input_faces"].select(fn=on_input_face_select, outputs=[ui_comp["input_page_info"]])

    def on_target_face_delete(evt=None):
        global is_deleting_target_face, selected_target_face_idx
        if is_deleting_target_face:
            return gr.update(), gr.update(), gr.update(), gr.update()

        raw_idx = None
        if evt is not None:
            raw_idx = evt.index if hasattr(evt, "index") else (evt["index"] if isinstance(evt, dict) and "index" in evt else evt)
            
        total_items = len(logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"))
        rel_index = _resolve_gallery_index(raw_idx, total_items, columns=3)
        
        if rel_index is None:
            if selected_target_face_idx is not None:
                global_idx = selected_target_face_idx
            else:
                return gr.update(), gr.update(), gr.update(), gr.update()
        else:
            global_idx = state.current_target_page * state.FACES_PER_PAGE + rel_index
            
        is_deleting_target_face = True
        try:
            if 0 <= global_idx < len(roop.globals.TARGET_FACES):
                _purged = roop.globals.TARGET_FACES[global_idx]
                roop.globals.TARGET_FACES.pop(global_idx)
                ui.globals.ui_target_thumbs.pop(global_idx)
                selected_target_face_idx = None
                _purge_bank_by_face(_purged)
                total_faces = len(ui.globals.ui_target_thumbs)
                max_pages = max(0, (total_faces - 1) // state.FACES_PER_PAGE)
                if state.current_target_page > max_pages:
                    state.current_target_page = max_pages
        finally:
            is_deleting_target_face = False
        return (
            logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), 
            logic.update_pagination_info(ui.globals.ui_target_thumbs, "target"), 
            *logic.update_pagination_buttons(total_faces, "target")
        )

    def on_target_face_select(evt: gr.SelectData):
        global selected_target_face_idx
        total_items = len(logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"))
        rel_index = _resolve_gallery_index(evt.index, total_items, columns=3)
        
        if rel_index is not None:
            global_idx = state.current_target_page * state.FACES_PER_PAGE + rel_index
            selected_target_face_idx = global_idx
            
        
        return gr.update(), gr.update(), gr.update(), gr.update()

    ui_comp["target_faces"].select(
        fn=on_target_face_select, 
        outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]]
    )

    ui_comp["previewimage"].select(
        fn=on_preview_click,
        inputs=[ui_comp["preview_frame_num"]],
        outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"], ui_comp["previewimage"]]
    ).then(
        fn=lambda: _file_indicator_text(),
        outputs=[ui_comp["file_indicator"]]
    )

    def detect_faces_in_frame(frame_num):
        if not state.list_files_process: 
            return gr.update(visible=False), gr.Markdown("### ❌ No hay archivo"), gr.update(value=[]), gr.update(), gr.update()
        
        entry = state.list_files_process[state.selected_preview_index]
        frame_idx = max(1, int(frame_num or 1))
        
        detected_faces = logic.extract_face_images(entry.filename, (True, frame_idx), target_face_detection=True, ui_padding=0.5)
        state.ALL_DETECTED_FACES_DATA = detected_faces
        state.SELECTION_FACES_DATA = detected_faces
        state.CURRENT_DETECTED_FRAME = frame_idx
        state.CURRENT_DETECTED_KEY = _face_detection_key(entry.filename, frame_idx)
        
        if not detected_faces:
            print(f"[FACES] ⚠️ No se detectaron caras en frame {frame_idx}")
            print(f"[UI] ⚠️ No se detectaron caras en frame {frame_idx}.")
        
        thumbs = [_normalize_face_thumb(util.convert_to_gradio(f[1], is_rgb=True), 300) for f in detected_faces]
        
        # Feedback visual con cajas
        from roop.capturer import get_video_frame, get_image_frame
        frame = get_video_frame(entry.filename, frame_idx) if util.is_video(entry.filename) else get_image_frame(entry.filename)
        boxed_frame = logic.draw_face_boxes(frame, detected_faces)
        
        detect_stamp = int(time.time() * 1000)
        return (
            gr.update(visible=True), 
            gr.Markdown(f"### ✅ {len(thumbs)} cara(s) detectada(s)"), 
            gr.Gallery(value=thumbs, columns=4, key=f"fs_{detect_stamp}"), 
            util.convert_to_gradio(boxed_frame),
            gr.update(value=f"🔍 BUSCAR OTRA ({len(thumbs)})")
        )

    ui_comp["bt_use_face_from_preview"].click(
        fn=detect_faces_in_frame, 
        inputs=[ui_comp["preview_frame_num"]], 
        outputs=[ui_comp["dynamic_face_selection"], ui_comp["face_detection_title"], ui_comp["face_selection"], ui_comp["previewimage"], ui_comp["bt_use_face_from_preview"]]
    )
    
    ui_comp["face_selection"].select(
        fn=on_face_selection_click, 
        inputs=[ui_comp["preview_frame_num"]],
        outputs=[ui_comp["face_selector_slider"], ui_comp["previewimage"]], 
        queue=False
    ).then(
        fn=lambda: (gr.update(visible=True), logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), logic.update_pagination_info(ui.globals.ui_target_thumbs, "target"), *logic.update_pagination_buttons(len(ui.globals.ui_target_thumbs), "target")),
        outputs=[ui_comp["dynamic_face_selection"], ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]]
    ).then(
        fn=lambda: _file_indicator_text(),
        outputs=[ui_comp["file_indicator"]]
    )
    
    # ARCHIVOS ORIGEN/DESTINO
    def _resolve_src_path(raw):
        try:
            if isinstance(raw, str):
                return raw
            if isinstance(raw, dict):
                name = raw.get("name") or raw.get("path") or raw.get("filename")
                if name:
                    return name
            if hasattr(raw, "name"):
                return raw.name
        except Exception:
            pass
        return None

    def _is_image_path(path: str) -> bool:
        try:
            ext = os.path.splitext(path)[1].lower()
            return ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".avif", ".gif"}
        except Exception:
            return False

    def on_src_changed(files):
        global selected_input_face_idx
        files = files or []
        _log(f"[TEST_CONSOLE] on_src_changed INICIO - {len(files)} archivo(s)")
        state.current_input_page = 0
        state.source_load_summary = ""
        selected_input_face_idx = None
        if not files:
            roop.globals.INPUT_FACESETS, ui.globals.ui_input_thumbs = [], []
            return [], "📄 Página 1 de 1 (0 caras)", gr.update(interactive=False), gr.update(interactive=False)

        roop.globals.INPUT_FACESETS, ui.globals.ui_input_thumbs = [], []
        handled = 0
        no_face = 0
        invalid_crop = 0
        for raw in files:
            f_path = _resolve_src_path(raw)
            if not f_path or not _is_image_path(f_path):
                continue
            handled += 1
            faces_data = logic.extract_face_images(f_path, is_source_face=True)
            if not faces_data:
                no_face += 1
                print(f"[FACES] ⚠️ No se detectaron caras en: {os.path.basename(f_path)}")
                print(f"[UI] ⚠️ No se detectaron caras en {os.path.basename(f_path)}")
            for face_obj, face_img in faces_data:
                # El swap usa el embedding detectado. El tamaño de la
                # miniatura no debe quitar una cara válida del banco.
                if face_img is None or face_img.size == 0:
                    invalid_crop += 1
                    continue
                face_img = np.ascontiguousarray(face_img)
                from roop.types import FaceSet
                face_obj.face_img = face_img
                face_obj.face_img_ref = face_img
                gradio_img = util.convert_to_gradio(face_img, is_rgb=True)
                if gradio_img is None:
                    invalid_crop += 1
                    continue
                roop.globals.INPUT_FACESETS.append(FaceSet(faces=[face_obj], name=os.path.basename(f_path)))
                ui.globals.ui_input_thumbs.append(gradio_img)

        total_faces = len(ui.globals.ui_input_thumbs)
        state.source_load_summary = (
            f"{total_faces} caras cargadas de {len(files)} archivos recibidos "
            f"({handled} imágenes válidas, {no_face} sin cara, {invalid_crop} recortes inválidos)"
        )
        _log(f"[LOAD] Fuentes: {state.source_load_summary}")
        return (logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"), logic.update_pagination_info(ui.globals.ui_input_thumbs, "input"), *logic.update_pagination_buttons(total_faces, "input"))
    
    _src_processing = False
    _src_last_key = None
    _src_last_finished_at = 0.0

    def _src_files_key(files):
        if not files:
            return ()
        paths = []
        for f in files:
            p = f.name if hasattr(f, "name") else (f["name"] if isinstance(f, dict) else str(f))
            paths.append(os.path.normpath(p))
        return tuple(sorted(paths))

    def _src_pagination_outputs():
        total_faces = len(ui.globals.ui_input_thumbs)
        return (
            logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"),
            logic.update_pagination_info(ui.globals.ui_input_thumbs, "input"),
            *logic.update_pagination_buttons(total_faces, "input"),
        )

    def _src_loading_outputs():
        if ui.globals.ui_input_thumbs:
            return _src_pagination_outputs()
        return (
            [],
            "⏳ Cargando caras...",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )

    def _safe_on_src_changed(files):
        nonlocal _src_processing, _src_last_key, _src_last_finished_at
        files_key = _src_files_key(files)
        now = time.time()

        if _src_processing:
            return _src_loading_outputs()

        if files_key and files_key == _src_last_key and (now - _src_last_finished_at) < 3.0:
            return _src_pagination_outputs()

        _src_processing = True
        try:
            result = on_src_changed(files)
            _src_last_key = files_key
            _src_last_finished_at = time.time()
            return result
        except Exception as exc:
            print(f"[UI] Error cargando sources: {exc}")
            return (
                [],
                f"❌ Error cargando caras: {exc}",
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        finally:
            _src_processing = False

    _src_outputs = [
        ui_comp["input_faces"],
        ui_comp["input_page_info"],
        ui_comp["bt_input_prev"],
        ui_comp["bt_input_next"],
    ]
    ui_comp["bt_srcfiles"].upload(
        fn=_safe_on_src_changed,
        inputs=[ui_comp["bt_srcfiles"]],
        outputs=_src_outputs,
        show_progress="full",
    )

    def _direct_paths(value):
        value = (value or "").strip()
        try:
            parsed = json.loads(value)
            paths = parsed if isinstance(parsed, list) else [value]
        except (ValueError, TypeError):
            paths = [value.strip('"').strip("'")]
        return [p for p in paths if isinstance(p, str) and os.path.isfile(p)]

    def on_add_source_paths(value):
        paths = [p for p in _direct_paths(value) if _is_image_path(p)]
        if not paths:
            return (gr.update(value=""), *_src_pagination_outputs())
        return (gr.update(value=""), *_safe_on_src_changed(paths))

    ui_comp["bt_add_source_paths"].click(
        fn=on_add_source_paths,
        inputs=[ui_comp["source_path_input"]],
        outputs=[ui_comp["source_path_input"], *_src_outputs],
        show_progress="hidden",
    )

    def _resolve_target_path(f):
        return f.name if hasattr(f, "name") else (f["name"] if isinstance(f, dict) else str(f))

    def _target_empty_outputs():
        return (gr.update(maximum=1, value=1), None, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), [], [], "📄 Página 1 de 1 (0 caras)", gr.update(interactive=False), gr.update(interactive=False))

    def _target_unchanged_outputs():
        return tuple(gr.update() for _ in range(10))

    def _load_target_files(paths):
        loaded_entries = []
        first_preview = None
        logic.initialize_thread_pools()
        pool = state._target_thread_pool

        t0 = time.time()
        futures = [pool.submit(logic.process_target_file_async, p) for p in paths]
        for fut in futures:
            entry, preview_img, _ = fut.result()
            if entry:
                loaded_entries.append(entry)
                if preview_img is not None and first_preview is None:
                    first_preview = preview_img
        _log(f"[LOAD] {len(loaded_entries)} archivo(s) cargado(s) en {time.time() - t0:.2f}s")

        if not loaded_entries:
            gr.Warning("No se pudo abrir ningún archivo de destino; se conserva la selección anterior.")
            return _target_unchanged_outputs(), 0

        state.current_target_page = 0
        state.SELECTION_FACES_DATA = []
        state.ALL_DETECTED_FACES_DATA = []
        state.CURRENT_DETECTED_FRAME = -1
        state.CURRENT_DETECTED_KEY = None
        roop.globals.TARGET_FACES.clear()
        ui.globals.ui_target_thumbs.clear()
        if hasattr(roop.globals, 'selected_face_references'):
            roop.globals.selected_face_references.clear()
        state.list_files_process = loaded_entries
        state.selected_preview_index = 0
        roop.globals.face_swap_mode = 'selected'
        first_entry = state.list_files_process[0]
        roop.globals.target_path = first_entry.filename
        _log(f"[LOAD] Destino: {len(state.list_files_process)} archivo(s) - {os.path.basename(first_entry.filename)}")
        
        return (
            gr.update(maximum=first_entry.endframe, value=1), 
            first_preview, 
            gr.update(interactive=False), 
            gr.update(interactive=len(state.list_files_process)>1), 
            gr.update(visible=False), 
            [], 
            logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), 
            logic.update_pagination_info(ui.globals.ui_target_thumbs, "target"), 
            *logic.update_pagination_buttons(0, "target")
        ), len(loaded_entries)

    def on_dest_changed(files):
        if not files:
            return _target_empty_outputs()
        with _target_load_lock:
            outputs, _ = _load_target_files([_resolve_target_path(f) for f in files])
            return outputs

    def on_add_target_path(ruta):
        paths = _direct_paths(ruta)
        if not paths:
            _log(f"[LOAD] Ruta no existe en el servidor: {ruta}")
            gr.Warning("La ruta no existe en el servidor; se conserva el destino cargado.")
            return (gr.update(), *_target_unchanged_outputs(), gr.update())
        _log(f"[LOAD] Cargando por ruta (sin subida por red): {len(paths)} archivo(s)")
        with _target_load_lock:
            outs, loaded_count = _load_target_files(paths)
        if not loaded_count:
            return (gr.update(), *outs, gr.update())
        return (gr.update(value=f"✅ {loaded_count} archivo(s) añadido(s) por ruta directa", label="⚡ Ruta directa en el servidor (recomendado si el video es grande/remoto)"), *outs, _file_indicator_text())

    def on_browse_and_load():
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            path = filedialog.askopenfilename(
                title="Selecciona el video/imagen en este PC (se carga por ruta, sin subida)",
                filetypes=[("Videos e imágenes", "*.mp4 *.mkv *.avi *.mov *.webm *.gif *.m4v *.wmv *.flv *.mpeg *.mpg *.jpg *.jpeg *.png *.bmp *.webp *.heic"), ("Todos los archivos", "*.*")]
            )
            root.destroy()
            if not path:
                return (gr.update(), *_target_unchanged_outputs(), gr.update())
            _log(f"[LOAD] Diálogo nativo → ruta: {path}")
            with _target_load_lock:
                outs, loaded_count = _load_target_files([path])
            if not loaded_count:
                return (gr.update(), *outs, gr.update())
            return (gr.update(value=f"✅ {os.path.basename(path)} cargado por ruta directa (sin subida)", label="⚡ Ruta directa en el servidor (recomendado si el video es grande/remoto)"), *outs, f"📂 **Archivo ({state.selected_preview_index+1}/{len(state.list_files_process)}):** {os.path.basename(state.list_files_process[0].filename)}")
        except Exception as e:
            _log(f"[BROWSE] Error abriendo diálogo: {e}")
            gr.Warning(f"No se pudo abrir el diálogo: {e}")
            return (gr.update(), *_target_unchanged_outputs(), gr.update())

    # NAVEGACIÓN Y INDICADORES
    def update_file_indicator():
        return _file_indicator_text()

    def update_source_random_state():
        has_video = any(util.is_video(entry.filename) for entry in state.list_files_process) if state.list_files_process else False
        if has_video:
            return gr.update(interactive=False, value=False)
        return gr.update(interactive=True)

    _path_outputs = [ui_comp["target_path_input"], ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"], ui_comp["dynamic_face_selection"], ui_comp["face_selection"], ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"], ui_comp["file_indicator"]]
    ui_comp["bt_destfiles"].change(on_dest_changed, [ui_comp["bt_destfiles"]], [ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"], ui_comp["dynamic_face_selection"], ui_comp["face_selection"], ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]], queue=False).then(update_file_indicator, None, ui_comp["file_indicator"]).then(update_source_random_state, None, ui_comp["source_random"])
    ui_comp["bt_add_target_path"].click(fn=on_add_target_path, inputs=[ui_comp["target_path_input"]], outputs=_path_outputs, queue=False).then(update_source_random_state, None, ui_comp["source_random"])
    ui_comp["target_path_input"].submit(fn=on_add_target_path, inputs=[ui_comp["target_path_input"]], outputs=_path_outputs, queue=False).then(update_source_random_state, None, ui_comp["source_random"])
    ui_comp["bt_browse_path"].click(fn=on_browse_and_load, outputs=_path_outputs, queue=False).then(update_source_random_state, None, ui_comp["source_random"])

    def jump_frame(current_f, delta):
        if not state.list_files_process:
            return gr.update(), None
        entry = state.list_files_process[state.selected_preview_index]
        new_f = max(1, min(entry.endframe, int(current_f or 1) + delta))
        from roop.capturer import get_video_frame
        frame = get_video_frame(entry.filename, new_f)
        return gr.update(value=new_f), util.convert_to_gradio(frame)

    ui_comp["bt_prev_frame"].click(fn=lambda f: jump_frame(f, -1), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]], queue=False)
    ui_comp["bt_next_frame"].click(fn=lambda f: jump_frame(f, 1), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]], queue=False)
    ui_comp["bt_jump_back_10"].click(fn=lambda f: jump_frame(f, -10), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]], queue=False)
    ui_comp["bt_jump_fwd_10"].click(fn=lambda f: jump_frame(f, 10), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]], queue=False)
    ui_comp["bt_jump_back_100"].click(fn=lambda f: jump_frame(f, -100), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]], queue=False)
    ui_comp["bt_jump_fwd_100"].click(fn=lambda f: jump_frame(f, 100), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]], queue=False)

    def on_frame_change(frame_num):
        if not state.list_files_process:
            return None, gr.update()
        filename = state.list_files_process[state.selected_preview_index].filename
        from roop.capturer import get_video_frame
        frame = get_video_frame(filename, int(frame_num or 1))
        return util.convert_to_gradio(frame), gr.update(info=f"Frame {int(frame_num or 1)}")

    ui_comp["preview_frame_num"].release(fn=on_frame_change, inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["previewimage"], ui_comp["preview_frame_num"]])

    def navigate_file(direction):
        if not state.list_files_process:
            return gr.update(), None, gr.update(), gr.update()
        state.selected_preview_index = min(len(state.list_files_process)-1, state.selected_preview_index + 1) if direction == "next" else max(0, state.selected_preview_index - 1)
        entry = state.list_files_process[state.selected_preview_index]
        roop.globals.target_path = entry.filename
        state.CURRENT_DETECTED_FRAME = -1
        state.CURRENT_DETECTED_KEY = None
        state.SELECTION_FACES_DATA = None
        from roop.capturer import get_video_frame, get_image_frame
        frame = get_video_frame(entry.filename, 1) if util.is_video(entry.filename) else get_image_frame(entry.filename)
        return gr.update(maximum=entry.endframe, value=1), util.convert_to_gradio(frame), gr.update(interactive=state.selected_preview_index>0), gr.update(interactive=state.selected_preview_index<len(state.list_files_process)-1)

    ui_comp["bt_prev_file"].click(fn=lambda: navigate_file("prev"), outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]], queue=False).then(update_file_indicator, None, ui_comp["file_indicator"])
    ui_comp["bt_next_file"].click(fn=lambda: navigate_file("next"), outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]], queue=False).then(update_file_indicator, None, ui_comp["file_indicator"])

    # PAGINACIÓN
    ui_comp["bt_input_prev"].click(fn=lambda: on_input_page_change(-1), outputs=[ui_comp["input_faces"], ui_comp["input_page_info"], ui_comp["bt_input_prev"], ui_comp["bt_input_next"]])
    ui_comp["bt_input_next"].click(fn=lambda: on_input_page_change(1), outputs=[ui_comp["input_faces"], ui_comp["input_page_info"], ui_comp["bt_input_prev"], ui_comp["bt_input_next"]])
    ui_comp["bt_target_prev"].click(fn=lambda: on_target_page_change(-1), outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]])
    ui_comp["bt_target_next"].click(fn=lambda: on_target_page_change(1), outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]])

    # BORRADO Y OTROS
    def clear_source():
        global selected_input_face_idx
        nonlocal _src_last_key, _src_last_finished_at
        roop.globals.INPUT_FACESETS.clear()
        ui.globals.ui_input_thumbs.clear()
        state.source_filenames = []
        state.source_load_summary = ""
        state.current_input_page = 0
        state.SELECTED_FACE_INDEX = 0
        selected_input_face_idx = None
        _src_last_key = None
        _src_last_finished_at = 0.0
        _log("[CLEAR] Origen: todos los archivos y caras retirados de la UI")
        return (None, gr.update(value=""), *_src_pagination_outputs(),
                gr.update(value=None, visible=False))

    def clear_target():
        global selected_target_face_idx
        with _target_load_lock:
            state.list_files_process = []
            state.selected_preview_index = 0
            state.current_target_page = 0
            state.SELECTION_FACES_DATA = []
            state.ALL_DETECTED_FACES_DATA = []
            state.CURRENT_DETECTED_FACES = []
            state.CURRENT_DETECTED_FRAME = -1
            state.CURRENT_DETECTED_KEY = None
            state.TEMP_SELECTED_FACE_INDEX = None
            state.SELECTED_TARGET_FACE_INDEX = 0
            state.selected_face_references.clear()
            state.FACE_DETECTION_CACHE.clear()
            state.is_video_processing = False
            state.is_image_processing = False
            selected_target_face_idx = None
            roop.globals.TARGET_FACES.clear()
            ui.globals.ui_target_thumbs.clear()
            if hasattr(roop.globals, 'selected_face_references'):
                roop.globals.selected_face_references.clear()
            roop.globals.target_path = None
        _log("[CLEAR] Destino: todos los archivos, caras y previews retirados de la UI")
        return (None, gr.update(value=""), *_target_empty_outputs(),
                "📂 **Archivo:** Ninguno seleccionado",
                gr.update(value=None, visible=False),
                "### ✨ Busca caras en el frame",
                gr.update(value="🔍 BUSCAR CARAS EN ESTE FRAME"),
                gr.update(maximum=1, value=1))

    ui_comp["bt_clear_source"].click(
        fn=clear_source,
        outputs=[ui_comp[k] for k in ("bt_srcfiles", "source_path_input", "input_faces",
                 "input_page_info", "bt_input_prev", "bt_input_next", "preview_result")],
    )
    ui_comp["bt_clear_target"].click(
        fn=clear_target,
        outputs=[ui_comp[k] for k in ("bt_destfiles", "target_path_input", "preview_frame_num",
                 "previewimage", "bt_prev_file", "bt_next_file", "dynamic_face_selection",
                 "face_selection", "target_faces", "target_page_info", "bt_target_prev",
                 "bt_target_next", "file_indicator", "preview_result", "face_detection_title",
                 "bt_use_face_from_preview", "face_selector_slider")],
    ).then(update_source_random_state, None, ui_comp["source_random"])

    # AJUSTES EXPERTOS
    ui_comp["enhancer"].change(fn=lambda v: setattr(roop.globals, 'selected_enhancer', v) or setattr(roop.globals, 'use_enhancer', v != "None"), inputs=[ui_comp["enhancer"]])
    ui_comp["enhancer_blend"].change(fn=lambda v: setattr(roop.globals, 'enhancer_blend_factor', v), inputs=[ui_comp["enhancer_blend"]])

    # PROCESAMIENTO
    start_event = ui_comp["bt_start"].click(
        fn=on_start_process,
        inputs=[
            ui_comp["autorotate"], ui_comp["smoothing"],
            ui_comp["face_distance"], ui_comp["blend_ratio"], ui_comp["enhancer_blend"], ui_comp["enhancer"],
            ui_comp["source_random"],
        ],
        outputs=[ui_comp["bt_start"], ui_comp["bt_stop"], ui_comp["metrics_display"]],
    )
    ui_comp["bt_stop"].click(
        fn=on_stop_process,
        outputs=[ui_comp["bt_start"], ui_comp["bt_stop"], ui_comp["metrics_display"]],
        queue=False,
    )
    ui_comp["bt_stop"].click(fn=None, cancels=[start_event], queue=False)
    if "bt_open_output" in ui_comp:
        ui_comp["bt_open_output"].click(fn=lambda: ui.globals.open_output_folder())

def on_input_page_change(delta):
    total = len(ui.globals.ui_input_thumbs)
    state.current_input_page = max(0, min(max(0, (total-1)//state.FACES_PER_PAGE), state.current_input_page + delta))
    return (logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"), logic.update_pagination_info(ui.globals.ui_input_thumbs, "input"), *logic.update_pagination_buttons(total, "input"))

def on_target_page_change(delta):
    total = len(ui.globals.ui_target_thumbs)
    state.current_target_page = max(0, min(max(0, (total-1)//state.FACES_PER_PAGE), state.current_target_page + delta))
    return (logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), logic.update_pagination_info(ui.globals.ui_target_thumbs, "target"), *logic.update_pagination_buttons(total, "target"))

def on_start_process(auto_rot, temp_smooth, dist, blend, enhancer_blend, enhancer, source_random):
    from ui.tabs.faceswap.logic import start_swap
    from ui.job_cancel import clear as clear_cancel, SCOPE_FACESWAP
    clear_cancel(SCOPE_FACESWAP)
    import roop.globals
    roop.globals.processing = True
    roop.globals.enhancer_blend_factor = enhancer_blend
    roop.globals.blend_ratio = blend
    roop.globals.distance_threshold = dist
    roop.globals.selected_enhancer = enhancer or 'GFPGAN'
    roop.globals.use_enhancer = enhancer not in (None, "", "None")
    roop.globals.autorotate_faces = auto_rot
    roop.globals.temporal_smoothing = temp_smooth
    roop.globals.source_random = source_random
    print(f"[SOURCE_RANDOM] {'ACTIVADO' if source_random else 'desactivado'}")
    gen = start_swap(enhancer=enhancer, keep_frames=False, wait_after_extraction=False, skip_audio=False, face_distance=dist, blend_ratio=blend, blend_mode="blend", selected_mask_engine="None", processing_method="Inswapper 128", no_face_action="skip", vr_mode=False, use_single_source_all=False, autorotate=auto_rot, temporal_smoothing=temp_smooth, num_swap_steps=1, imagemask=None)
    for value in gen:
        yield value

def on_stop_process():
    from ui.tabs.faceswap.logic import stop_swap
    from ui.job_cancel import request as request_cancel, SCOPE_FACESWAP
    request_cancel(SCOPE_FACESWAP)
    return stop_swap()
