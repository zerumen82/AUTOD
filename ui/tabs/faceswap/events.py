import gradio as gr
import os
import cv2
import numpy as np
import time
import threading
import roop.globals
import ui.globals
import ui.tabs.faceswap.state as state
import ui.tabs.faceswap.logic as logic
import roop.utilities as util
import roop.face_util as face_util

DETECTED_FACES_GALLERY_COLUMNS = 10
is_deleting_input_face = False
is_deleting_target_face = False


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


def on_face_selection_click(evt: gr.SelectData, frame_num):
    """Maneja el clic en galería, sincroniza el slider, añade la cara y actualiza el feedback visual."""
    if not evt or evt.index is None:
        return gr.update(), gr.update()

    total_faces = len(state.SELECTION_FACES_DATA) if state.SELECTION_FACES_DATA else 0
    # Arreglado: La galería de selección tiene 4 columnas en ui.py
    face_index = _resolve_gallery_index(evt.index, total_faces, columns=4)
    
    if face_index is None or state.SELECTION_FACES_DATA is None:
        return gr.update(), gr.update()

    # Guardar índice para preview del enhancer
    state.TEMP_SELECTED_FACE_INDEX = face_index

    try:
        face_data = state.SELECTION_FACES_DATA[face_index]
        face_obj = face_data[0]
        
        # Verificar duplicados (solo si es exactamente la misma cara en el mismo archivo/frame)
        is_duplicate = False
        current_file = os.path.basename(state.list_files_process[state.selected_preview_index].filename) if state.list_files_process else ""
        current_frame = int(frame_num or 1)
        
        for existing in roop.globals.TARGET_FACES:
            # Para ser duplicado, debe venir de la misma fuente O tener un embedding virtualmente idéntico
            existing_source = getattr(existing, 'source_image', '')
            
            # Comparación por embedding (muy estricta: 0.9999)
            # Solo bloqueamos por embedding si sospechamos que es el mismo archivo (o muy muy parecido)
            if hasattr(existing, 'embedding') and hasattr(face_obj, 'embedding') and existing.embedding is not None and face_obj.embedding is not None:
                # Usar dot product sobre embeddings normalizados (si están disponibles)
                emb1 = getattr(existing, 'normed_embedding', None)
                emb2 = getattr(face_obj, 'normed_embedding', None)
                
                if emb1 is not None and emb2 is not None:
                    similarity = np.dot(emb1, emb2)
                else:
                    # Fallback a dot product normal (asumiendo que están normalizados)
                    similarity = np.dot(existing.embedding, face_obj.embedding)
                
                # Si son idénticos y vienen del mismo archivo -> DUPLICADO
                if similarity > 0.9999 and existing_source == str(image_path):
                    is_duplicate = True; break
                
                # Si son idénticos (0.99999) incluso si no sabemos el origen -> PROBABLE DUPLICADO
                if similarity > 0.99999:
                    is_duplicate = True; break
            
            # Comparación por BBox (solo si es el mismo archivo)
            if existing.bbox == face_obj.bbox and existing_source == str(image_path):
                is_duplicate = True; break

        if not is_duplicate:
            roop.globals.TARGET_FACES.append(face_obj)
            ui.globals.ui_target_thumbs.append(util.convert_to_gradio(face_data[1], is_rgb=True))
            
            if state.list_files_process and state.selected_preview_index < len(state.list_files_process):
                filename = os.path.basename(state.list_files_process[state.selected_preview_index].filename)
                video_key = f"selected_face_ref_{filename}"
                if not hasattr(roop.globals, 'selected_face_references'): roop.globals.selected_face_references = {}
                roop.globals.selected_face_references[video_key] = {'face_obj': face_obj, 'embedding': face_obj.embedding, 'bbox': face_obj.bbox}
            
            gr.Info("✅ Cara añadida")
        else:
            gr.Info("ℹ️ Ya está en Destino")

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
    is_new_frame = getattr(state, 'CURRENT_DETECTED_FRAME', -1) != frame_idx
    if not state.SELECTION_FACES_DATA or is_new_frame:
        print(f"[FaceSelection] Auto-detectando caras para frame {frame_idx} (Nuevo={is_new_frame})...")
        detected_faces = logic.extract_face_images(entry.filename, (True, frame_idx), target_face_detection=True, ui_padding=1.5)
        state.ALL_DETECTED_FACES_DATA = detected_faces
        state.SELECTION_FACES_DATA = detected_faces
        state.CURRENT_DETECTED_FRAME = frame_idx

    if not state.SELECTION_FACES_DATA:
        gr.Warning("⚠️ No se detectaron caras. Intenta buscar caras primero.")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

    # 2. Buscar cara que coincida con las coordenadas del clic [x, y]
    click_x, click_y = evt.index[0], evt.index[1]
    best_face_idx = -1
    min_dist = float('inf')

    for idx, face_data in enumerate(state.SELECTION_FACES_DATA):
        face_obj = face_data[0]
        x1, y1, x2, y2 = face_obj.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        margin = 30
        if (x1 - margin) <= click_x <= (x2 + margin) and (y1 - margin) <= click_y <= (y2 + margin):
            dist = np.sqrt((click_x - cx)**2 + (click_y - cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_face_idx = idx

    if best_face_idx != -1:
        face_data = state.SELECTION_FACES_DATA[best_face_idx]
        face_obj = face_data[0]
        
        # Sincronizar índice para preview del enhancer
        state.TEMP_SELECTED_FACE_INDEX = best_face_idx

        # Verificar duplicados (solo si es exactamente la misma cara en el mismo archivo/frame)
        is_duplicate = False
        current_file = os.path.basename(state.list_files_process[state.selected_preview_index].filename) if state.list_files_process else ""
        current_frame = int(frame_num or 1)
        
        for existing in roop.globals.TARGET_FACES:
            # Para ser duplicado, debe venir de la misma fuente O tener un embedding virtualmente idéntico
            existing_source = getattr(existing, 'source_image', '')
            
            # Comparación por embedding (muy estricta: 0.9999)
            # Solo bloqueamos por embedding si sospechamos que es el mismo archivo (o muy muy parecido)
            if hasattr(existing, 'embedding') and hasattr(face_obj, 'embedding') and existing.embedding is not None and face_obj.embedding is not None:
                # Usar dot product sobre embeddings normalizados (si están disponibles)
                emb1 = getattr(existing, 'normed_embedding', None)
                emb2 = getattr(face_obj, 'normed_embedding', None)
                
                if emb1 is not None and emb2 is not None:
                    similarity = np.dot(emb1, emb2)
                else:
                    # Fallback a dot product normal (asumiendo que están normalizados)
                    similarity = np.dot(existing.embedding, face_obj.embedding)
                
                # Si son idénticos y vienen del mismo archivo -> DUPLICADO
                if similarity > 0.9999 and existing_source == str(image_path):
                    is_duplicate = True; break
                
                # Si son idénticos (0.99999) incluso si no sabemos el origen -> PROBABLE DUPLICADO
                if similarity > 0.99999:
                    is_duplicate = True; break
            
            # Comparación por BBox (solo si es el mismo archivo)
            if existing.bbox == face_obj.bbox and existing_source == str(image_path):
                is_duplicate = True; break

        if not is_duplicate:
            roop.globals.TARGET_FACES.append(face_obj)
            ui.globals.ui_target_thumbs.append(util.convert_to_gradio(face_data[1], is_rgb=True))
            
            filename = os.path.basename(entry.filename)
            video_key = f"selected_face_ref_{filename}"
            if not hasattr(roop.globals, 'selected_face_references'): roop.globals.selected_face_references = {}
            roop.globals.selected_face_references[video_key] = {'face_obj': face_obj, 'embedding': face_obj.embedding, 'bbox': face_obj.bbox}
            
            gr.Info(f"✅ Cara añadida a Destino ({filename})")
        else:
            gr.Info("ℹ️ Esta cara ya está en la lista de Destino")

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
    
    # --- FILTRADO ---
    def on_filter_change(gender, size):
        if not hasattr(state, 'ALL_DETECTED_FACES_DATA'): return gr.update()
        filtered = logic.get_filtered_faces(state.ALL_DETECTED_FACES_DATA, gender, size)
        state.SELECTION_FACES_DATA = filtered
        thumbs = [_normalize_face_thumb(util.convert_to_gradio(img, is_rgb=True), 300) for _, img in filtered]
        return gr.Gallery(value=thumbs, columns=min(4, len(thumbs)) if thumbs else 1)

    ui_comp["gender_filter"].change(on_filter_change, [ui_comp["gender_filter"], ui_comp["size_filter"]], [ui_comp["face_selection"]])
    ui_comp["size_filter"].change(on_filter_change, [ui_comp["gender_filter"], ui_comp["size_filter"]], [ui_comp["face_selection"]])

    # --- ENHANCER PREVIEW ---
    def on_enhancer_preview_click(frame_num, enhancer, blend):
        if not state.list_files_process: return gr.update(), gr.update()
        
        face_idx = getattr(state, 'TEMP_SELECTED_FACE_INDEX', None)
        if face_idx is None: 
            return gr.update(), gr.Markdown("### ⚠️ Selecciona una cara primero en la galería superior")
        
        entry = state.list_files_process[state.selected_preview_index]
        preview_img, msg = logic.get_enhancer_preview(entry.filename, int(frame_num or 1), face_idx, enhancer, blend)
        if preview_img:
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
            # NUEVO: Marcar esta cara como la identidad origen seleccionada
            import roop.globals as roop_globals
            roop_globals.source_face_index = selected_input_face_idx
            print(f"[SOURCE_SELECT] Cara origen seleccionada: índice #{selected_input_face_idx}")

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
        if 0 <= global_idx < len(roop.globals.INPUT_FACESETS):
            roop.globals.INPUT_FACESETS.pop(global_idx)
            ui.globals.ui_input_thumbs.pop(global_idx)
            selected_input_face_idx = None
            
            total_faces = len(ui.globals.ui_input_thumbs)
            max_pages = max(0, (total_faces - 1) // state.FACES_PER_PAGE)
            if state.current_input_page > max_pages: 
                state.current_input_page = max_pages
        
        total_faces = len(ui.globals.ui_input_thumbs)
        is_deleting_input_face = False
        return (
            logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"), 
            logic.update_pagination_info(ui.globals.ui_input_thumbs, "input"), 
            *logic.update_pagination_buttons(total_faces, "input")
        )

    ui_comp["input_faces"].select(fn=on_input_face_select)

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
        if 0 <= global_idx < len(roop.globals.TARGET_FACES):
            roop.globals.TARGET_FACES.pop(global_idx)
            ui.globals.ui_target_thumbs.pop(global_idx)
            selected_target_face_idx = None
            
            total_faces = len(ui.globals.ui_target_thumbs)
            max_pages = max(0, (total_faces - 1) // state.FACES_PER_PAGE)
            if state.current_target_page > max_pages: 
                state.current_target_page = max_pages
        
        total_faces = len(ui.globals.ui_target_thumbs)
        is_deleting_target_face = False
        return (
            logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), 
            logic.update_pagination_info(ui.globals.ui_target_thumbs, "target"), 
            *logic.update_pagination_buttons(total_faces, "target")
        )

    def on_target_face_select(evt: gr.SelectData, is_delete_mode):
        global selected_target_face_idx
        total_items = len(logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"))
        rel_index = _resolve_gallery_index(evt.index, total_items, columns=3)
        
        if rel_index is not None:
            global_idx = state.current_target_page * state.FACES_PER_PAGE + rel_index
            selected_target_face_idx = global_idx
            
            # Si el modo borrado está activo, borrar inmediatamente
            if is_delete_mode:
                return on_target_face_delete(evt)
        
        return gr.update(), gr.update(), gr.update(), gr.update()

    ui_comp["target_faces"].select(
        fn=on_target_face_select, 
        inputs=[ui_comp["delete_mode"]], 
        outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]]
    )
    ui_comp["target_faces"].delete(fn=on_target_face_delete, outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]])

    ui_comp["previewimage"].select(
        fn=on_preview_click,
        inputs=[ui_comp["preview_frame_num"]],
        outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"], ui_comp["previewimage"]]
    )

    def detect_faces_in_frame(frame_num):
        if not state.list_files_process: 
            return gr.update(visible=False), gr.Markdown("### ❌ No hay archivo"), gr.update(value=[]), gr.update(), gr.update()
        
        entry = state.list_files_process[state.selected_preview_index]
        frame_idx = max(1, int(frame_num or 1))
        
        detected_faces = logic.extract_face_images(entry.filename, (True, frame_idx), target_face_detection=True, ui_padding=1.5)
        state.ALL_DETECTED_FACES_DATA = detected_faces # Guardar todo para filtros
        state.SELECTION_FACES_DATA = detected_faces
        
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
    )
    
    # ARCHIVOS ORIGEN/DESTINO
    def on_src_changed(files):
        state.current_input_page = 0
        if not files: 
            roop.globals.INPUT_FACESETS, ui.globals.ui_input_thumbs = [], []
            return [], "📄 Página 1 de 1 (0 caras)", gr.update(interactive=False), gr.update(interactive=False)
        
        roop.globals.INPUT_FACESETS, ui.globals.ui_input_thumbs = [], []
        MIN_SRC_SIZE = 40 
        for f in files:
            f_path = f.name if hasattr(f, "name") else (f["name"] if isinstance(f, dict) else str(f))
            faces_data = logic.extract_face_images(f_path, is_source_face=True)
            for face_obj, face_img in faces_data:
                if face_img.shape[0] < MIN_SRC_SIZE or face_img.shape[1] < MIN_SRC_SIZE: continue
                from roop.types import FaceSet
                face_obj.face_img = face_img
                face_obj.face_img_ref = face_img
                roop.globals.INPUT_FACESETS.append(FaceSet(faces=[face_obj], name=os.path.basename(f_path)))
                ui.globals.ui_input_thumbs.append(util.convert_to_gradio(face_img, is_rgb=True))
        
        total_faces = len(ui.globals.ui_input_thumbs)
        return (logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"), logic.update_pagination_info(ui.globals.ui_input_thumbs, "input"), *logic.update_pagination_buttons(total_faces, "input"))
    
    ui_comp["bt_srcfiles"].change(fn=on_src_changed, inputs=[ui_comp["bt_srcfiles"]], outputs=[ui_comp["input_faces"], ui_comp["input_page_info"], ui_comp["bt_input_prev"], ui_comp["bt_input_next"]])

    def on_dest_changed(files):
        state.current_target_page = 0
        roop.globals.TARGET_FACES.clear()
        ui.globals.ui_target_thumbs.clear()
        if hasattr(roop.globals, 'selected_face_references'): roop.globals.selected_face_references.clear()
        
        if not files: 
            return gr.update(maximum=1), None, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), [], [], "📄 Página 1 de 1 (0 caras)", gr.update(interactive=False), gr.update(interactive=False)
        
        state.list_files_process = []
        first_preview = None
        for i, f in enumerate(files):
            f_path = f.name if hasattr(f, "name") else (f["name"] if isinstance(f, dict) else str(f))
            entry, preview_img, _ = logic.process_target_file_async(f_path)
            if entry:
                state.list_files_process.append(entry)
                if i == 0: first_preview = preview_img
        
        state.selected_preview_index = 0
        roop.globals.face_swap_mode = 'selected'
        first_entry = state.list_files_process[0]
        roop.globals.target_path = first_entry.filename
        
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
        )

    # NAVEGACIÓN Y INDICADORES
    def update_file_indicator():
        if not state.list_files_process: return "📂 **Archivo:** Ninguno"
        entry = state.list_files_process[state.selected_preview_index]
        return f"📂 **Archivo ({state.selected_preview_index+1}/{len(state.list_files_process)}):** {os.path.basename(entry.filename)}"

    ui_comp["bt_destfiles"].change(on_dest_changed, [ui_comp["bt_destfiles"]], [ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"], ui_comp["dynamic_face_selection"], ui_comp["face_selection"], ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]]).then(update_file_indicator, None, ui_comp["file_indicator"])

    def jump_frame(current_f, delta):
        if not state.list_files_process: return gr.update(), None
        entry = state.list_files_process[state.selected_preview_index]
        new_f = max(1, min(entry.endframe, int(current_f or 1) + delta))
        from roop.capturer import get_video_frame
        frame = get_video_frame(entry.filename, new_f)
        return gr.update(value=new_f), util.convert_to_gradio(frame)

    ui_comp["bt_prev_frame"].click(fn=lambda f: jump_frame(f, -1), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]])
    ui_comp["bt_next_frame"].click(fn=lambda f: jump_frame(f, 1), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]])
    ui_comp["bt_jump_back_10"].click(fn=lambda f: jump_frame(f, -10), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]])
    ui_comp["bt_jump_fwd_10"].click(fn=lambda f: jump_frame(f, 10), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]])
    ui_comp["bt_jump_back_100"].click(fn=lambda f: jump_frame(f, -100), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]])
    ui_comp["bt_jump_fwd_100"].click(fn=lambda f: jump_frame(f, 100), inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"]])

    def on_frame_change(frame_num):
        if not state.list_files_process: return None, gr.update()
        filename = state.list_files_process[state.selected_preview_index].filename
        from roop.capturer import get_video_frame
        frame = get_video_frame(filename, int(frame_num or 1))
        return util.convert_to_gradio(frame), gr.update(info=f"Frame {int(frame_num or 1)}")

    ui_comp["preview_frame_num"].release(fn=on_frame_change, inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["previewimage"], ui_comp["preview_frame_num"]])

    def navigate_file(direction):
        if not state.list_files_process: return gr.update(), None, gr.update(), gr.update()
        state.selected_preview_index = min(len(state.list_files_process)-1, state.selected_preview_index + 1) if direction == "next" else max(0, state.selected_preview_index - 1)
        entry = state.list_files_process[state.selected_preview_index]
        roop.globals.target_path = entry.filename
        from roop.capturer import get_video_frame, get_image_frame
        frame = get_video_frame(entry.filename, 1) if util.is_video(entry.filename) else get_image_frame(entry.filename)
        return gr.update(maximum=entry.endframe, value=1), util.convert_to_gradio(frame), gr.update(interactive=state.selected_preview_index>0), gr.update(interactive=state.selected_preview_index<len(state.list_files_process)-1)

    ui_comp["bt_prev_file"].click(fn=lambda: navigate_file("prev"), outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]]).then(update_file_indicator, None, ui_comp["file_indicator"])
    ui_comp["bt_next_file"].click(fn=lambda: navigate_file("next"), outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]]).then(update_file_indicator, None, ui_comp["file_indicator"])

    # PAGINACIÓN
    ui_comp["bt_input_prev"].click(fn=lambda: on_input_page_change(-1), outputs=[ui_comp["input_faces"], ui_comp["input_page_info"], ui_comp["bt_input_prev"], ui_comp["bt_input_next"]])
    ui_comp["bt_input_next"].click(fn=lambda: on_input_page_change(1), outputs=[ui_comp["input_faces"], ui_comp["input_page_info"], ui_comp["bt_input_prev"], ui_comp["bt_input_next"]])
    ui_comp["bt_target_prev"].click(fn=lambda: on_target_page_change(-1), outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]])
    ui_comp["bt_target_next"].click(fn=lambda: on_target_page_change(1), outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]])

    # BORRADO Y OTROS
    def clear_all_target_faces():
        roop.globals.TARGET_FACES.clear()
        ui.globals.ui_target_thumbs.clear()
        total = 0
        return ([], "📄 Página 1 de 1 (0 caras)", gr.update(interactive=False), gr.update(interactive=False))
    
    def remove_selected_input_face(evt=None):
        if state.selected_input_face_index is not None:
            idx = state.selected_input_face_index
            if 0 <= idx < len(roop.globals.INPUT_FACES):
                roop.globals.INPUT_FACES.pop(idx)
                ui.globals.ui_input_thumbs.pop(idx)
                state.selected_input_face_index = None
        total = len(ui.globals.ui_input_thumbs)
        return (logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"), logic.update_pagination_info(ui.globals.ui_input_thumbs, "input"), *logic.update_pagination_buttons(total, "input"))
    
    def remove_selected_target_face(evt=None):
        if state.selected_target_face_index is not None:
            idx = state.selected_target_face_index
            if 0 <= idx < len(roop.globals.TARGET_FACES):
                roop.globals.TARGET_FACES.pop(idx)
                ui.globals.ui_target_thumbs.pop(idx)
                state.selected_target_face_index = None
        total = len(ui.globals.ui_target_thumbs)
        return (logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), logic.update_pagination_info(ui.globals.ui_target_thumbs, "target"), *logic.update_pagination_buttons(total, "target"))
    
    ui_comp["bt_clear_all_target"].click(fn=clear_all_target_faces, outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]])
    ui_comp["bt_remove_selected_input_face"].click(fn=remove_selected_input_face, outputs=[ui_comp["input_faces"], ui_comp["input_page_info"], ui_comp["bt_input_prev"], ui_comp["bt_input_next"]])
    ui_comp["bt_remove_selected_target_face"].click(fn=remove_selected_target_face, outputs=[ui_comp["target_faces"], ui_comp["target_page_info"], ui_comp["bt_target_prev"], ui_comp["bt_target_next"]])

    # AJUSTES EXPERTOS
    ui_comp["enhancer"].change(fn=lambda v: setattr(roop.globals, 'selected_enhancer', v) or setattr(roop.globals, 'use_enhancer', v != "None"), inputs=[ui_comp["enhancer"]])
    ui_comp["enhancer_blend"].change(fn=lambda v: setattr(roop.globals, 'enhancer_blend_factor', v), inputs=[ui_comp["enhancer_blend"]])

    # PROCESAMIENTO
    ui_comp["bt_start"].click(fn=on_start_process, inputs=[ui_comp["fake_preview"], ui_comp["autorotate"], ui_comp["smoothing"], ui_comp["face_distance"], ui_comp["blend_ratio"], ui_comp["enhancer_blend"], ui_comp["enhancer"]], outputs=[ui_comp["bt_start"], ui_comp["bt_stop"], ui_comp["metrics_display"]])
    ui_comp["bt_stop"].click(fn=on_stop_process, outputs=[ui_comp["bt_start"], ui_comp["bt_stop"], ui_comp["metrics_display"]])
    if "bt_open_output" in ui_comp: ui_comp["bt_open_output"].click(fn=lambda: ui.globals.open_output_folder())

def on_input_page_change(delta):
    total = len(ui.globals.ui_input_thumbs)
    state.current_input_page = max(0, min(max(0, (total-1)//state.FACES_PER_PAGE), state.current_input_page + delta))
    return (logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"), logic.update_pagination_info(ui.globals.ui_input_thumbs, "input"), *logic.update_pagination_buttons(total, "input"))

def on_target_page_change(delta):
    total = len(ui.globals.ui_target_thumbs)
    state.current_target_page = max(0, min(max(0, (total-1)//state.FACES_PER_PAGE), state.current_target_page + delta))
    return (logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), logic.update_pagination_info(ui.globals.ui_target_thumbs, "target"), *logic.update_pagination_buttons(total, "target"))

async def on_start_process(fake_preview, auto_rot, temp_smooth, dist, blend, enhancer_blend, enhancer):
    from ui.tabs.faceswap.logic import start_swap
    # Configurar enhancer_blend_factor en globals antes del procesamiento
    roop.globals.enhancer_blend_factor = enhancer_blend
    gen = start_swap(enhancer=enhancer, keep_frames=False, wait_after_extraction=False, skip_audio=False, face_distance=dist, blend_ratio=blend, blend_mode="blend", selected_mask_engine="None", processing_method="Inswapper 128", no_face_action="skip", vr_mode=False, use_single_source_all=False, autorotate=auto_rot, temporal_smoothing=temp_smooth, num_swap_steps=1, imagemask=None)
    for value in gen:
        yield value

def on_stop_process():
    from ui.tabs.faceswap.logic import stop_swap
    return stop_swap()
