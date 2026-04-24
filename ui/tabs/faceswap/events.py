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

DETECTED_FACES_GALLERY_COLUMNS = 8
_inferred_gallery_columns = None


def _resolve_gallery_index(raw_index, total_items, columns=DETECTED_FACES_GALLERY_COLUMNS):
    """Convierte el índice de Gradio (int o fila/columna) a índice lineal."""
    global _inferred_gallery_columns
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
            # Gradio puede renderizar menos columnas reales que las configuradas.
            # Inferimos dinámicamente a partir de la columna observada.
            if _inferred_gallery_columns is None:
                _inferred_gallery_columns = max(1, col + 1)
            else:
                _inferred_gallery_columns = max(_inferred_gallery_columns, col + 1)

            idx = (row * _inferred_gallery_columns) + col
            if 0 <= idx < total_items:
                return idx
            # Fallback: intentar con el valor configurado por UI.
            idx_cfg = (row * columns) + col
            if 0 <= idx_cfg < total_items:
                return idx_cfg
            # Fallback extra para layouts de 1 columna efectiva.
            idx_single = row + col
            return idx_single if 0 <= idx_single < total_items else None

    return None


def _normalize_face_thumb(image, size=160):
    """
    Normaliza miniaturas a tamaño fijo (canvas cuadrado) para evitar
    layouts inestables de la Gallery cuando las caras vienen con tamaños distintos.
    """
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
    except Exception:
        return None


def on_face_selection_click(evt: gr.SelectData):
    """Maneja el clic en galería y sincroniza el slider."""
    if not evt or evt.index is None:
        return gr.update()

    total_faces = len(state.SELECTION_FACES_DATA) if state.SELECTION_FACES_DATA else 0
    face_index = _resolve_gallery_index(evt.index, total_faces)
    print(f"[FaceSelection] raw_index={evt.index} -> resolved_index={face_index} (total={total_faces})")
    if face_index is None or state.SELECTION_FACES_DATA is None:
        return gr.update()

    return gr.update(
        maximum=max(1, len(state.SELECTION_FACES_DATA)),
        value=min(face_index + 1, max(1, len(state.SELECTION_FACES_DATA)))
    )

def wire_events(ui_comp):
    """Conecta los componentes con la lógica"""
    
    # --- 0. DETECTAR CARAS ---
    def _warmup_face_analyser_async():
        """Precarga el analizador para evitar multi-click en la primera detección."""
        try:
            if face_util.FACE_ANALYSER is None:
                print("[FaceWarmup] Precargando FaceAnalysis...")
                face_util.get_face_analyser()
        except Exception as e:
            print(f"[FaceWarmup] Warmup falló: {e}")

    def detect_faces_in_frame(frame_num):
        global _inferred_gallery_columns
        
        if not state.list_files_process:
            return (
                gr.update(visible=False),
                gr.update(value=[], selected_index=None, interactive=True),
                gr.update(maximum=1, value=1),
                gr.update(value="🔍 BUSCAR CARAS EN ESTE FRAME", interactive=True),
            )

        analyser = face_util.get_face_analyser()
        if analyser is None:
            return (
                gr.update(visible=False),
                gr.update(value=[], selected_index=None, interactive=True),
                gr.update(maximum=1, value=1),
                gr.update(value="🔍 BUSCAR CARAS EN ESTE FRAME", interactive=True),
            )

        if state.selected_preview_index < 0 or state.selected_preview_index >= len(state.list_files_process):
            state.selected_preview_index = 0

        filename = state.list_files_process[state.selected_preview_index].filename
        roop.globals.target_path = filename
        frame_idx = max(1, int(frame_num or 1))
        
        detected_faces = logic.extract_face_images(filename, (True, frame_idx), target_face_detection=True)

        valid_faces = []
        thumbs = []
        for face_data in detected_faces:
            face_obj, face_img = face_data[0], face_data[1]
            thumb = util.convert_to_gradio(face_img, is_rgb=True)
            thumb = _normalize_face_thumb(thumb, size=160)
            if thumb is None:
                continue
            valid_faces.append((face_obj, face_img))
            thumbs.append(thumb)

        state.SELECTION_FACES_DATA = valid_faces
        # Reiniciar inferencia de columnas para cada nueva detección.
        _inferred_gallery_columns = None
        if not thumbs:
            return (
                gr.update(visible=False),
                gr.update(value=[], selected_index=None, interactive=True),
                gr.update(maximum=1, value=1),
                gr.update(value="🔍 BUSCAR CARAS EN ESTE FRAME", interactive=True),
            )

        # Remontar la galería en cada búsqueda para resetear handlers de select
        # cuando Gradio se queda en estado inconsistente.
        detect_stamp = int(time.time() * 1000)
        return (
            gr.update(visible=True),
            gr.Gallery(
                value=thumbs,
                selected_index=None,
                interactive=True,
                allow_preview=False,
                preview=False,
                columns=DETECTED_FACES_GALLERY_COLUMNS,
                height=200,
                object_fit="cover",
                key=f"face_selection_{detect_stamp}",
            ),
            gr.update(maximum=len(thumbs), value=1),
            gr.update(value="🔍 BUSCAR CARAS EN ESTE FRAME", interactive=True),
        )

    ui_comp["bt_use_face_from_preview"].click(
        fn=detect_faces_in_frame,
        inputs=[ui_comp["preview_frame_num"]],
        outputs=[ui_comp["dynamic_face_selection"], ui_comp["face_selection"], ui_comp["face_selector_slider"], ui_comp["bt_use_face_from_preview"]],
        queue=False,
        show_progress="hidden",
    )

    # --- 1. SELECCIÓN ---
    select_evt = ui_comp["face_selection"].select(
        fn=on_face_selection_click,
        outputs=[ui_comp["face_selector_slider"]],
        queue=False,
        show_progress="hidden",
        trigger_mode="always_last",
    )
    
    def add_from_slider(idx):
        if state.SELECTION_FACES_DATA and 0 <= (idx-1) < len(state.SELECTION_FACES_DATA):
            face_data = state.SELECTION_FACES_DATA[idx-1]
            roop.globals.TARGET_FACES.append(face_data[0])
            ui.globals.ui_target_thumbs.append(util.convert_to_gradio(face_data[1], is_rgb=True))
        return gr.update(visible=True), logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), logic.update_pagination_info(ui.globals.ui_target_thumbs, "target")

    # Auto-add tras click en miniatura (manteniendo click directo).
    select_evt.then(
        fn=add_from_slider,
        inputs=[ui_comp["face_selector_slider"]],
        outputs=[ui_comp["dynamic_face_selection"], ui_comp["target_faces"], ui_comp["target_page_info"]],
        queue=False,
        show_progress="hidden",
    )

    ui_comp["bt_use_selected_face"].click(fn=add_from_slider, inputs=[ui_comp["face_selector_slider"]], outputs=[ui_comp["dynamic_face_selection"], ui_comp["target_faces"], ui_comp["target_page_info"]])

    # --- 2. ARCHIVOS ORIGEN ---
    def on_src_changed(files):
        if not files: return [], "📄 Página 1 de 1 (0 caras)"
        roop.globals.INPUT_FACESETS, ui.globals.ui_input_thumbs = [], []
        for f in files:
            faces_data = logic.extract_face_images(f.name, is_source_face=True)
            for face_obj, face_img in faces_data:
                from roop.types import FaceSet
                roop.globals.INPUT_FACESETS.append(FaceSet(faces=[face_obj], name=os.path.basename(f.name)))
                ui.globals.ui_input_thumbs.append(util.convert_to_gradio(face_img, is_rgb=True))
        return logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input"), logic.update_pagination_info(ui.globals.ui_input_thumbs, "input")
    
    ui_comp["bt_srcfiles"].change(fn=on_src_changed, inputs=[ui_comp["bt_srcfiles"]], outputs=[ui_comp["input_faces"], ui_comp["input_page_info"]])

    # --- 3. ARCHIVOS DESTINO (FIX PREVIEW) ---
    def on_dest_changed(files):
        if not files: return gr.update(maximum=1), None, gr.update(interactive=False), gr.update(interactive=False), gr.update(visible=False), [], [], "📄 Página 1 de 1 (0 caras)"
        # Warmup temprano del detector para que el primer "Buscar caras" responda a la primera.
        threading.Thread(target=_warmup_face_analyser_async, daemon=True).start()
        state.list_files_process = []
        first_preview = None
        for i, f in enumerate(files):
            entry, preview_img, _ = logic.process_target_file_async(f.name)
            if entry:
                state.list_files_process.append(entry)
                if i == 0: first_preview = preview_img
        state.selected_preview_index = 0
        first_entry = state.list_files_process[0]
        roop.globals.target_path = first_entry.filename
        return gr.update(maximum=first_entry.endframe, value=1), first_preview, gr.update(interactive=False), gr.update(interactive=len(state.list_files_process)>1), gr.update(visible=False), [], logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target"), logic.update_pagination_info(ui.globals.ui_target_thumbs, "target")

    ui_comp["bt_destfiles"].change(fn=on_dest_changed, inputs=[ui_comp["bt_destfiles"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"], ui_comp["dynamic_face_selection"], ui_comp["face_selection"], ui_comp["target_faces"], ui_comp["target_page_info"]])

    # --- 4. NAVEGACIÓN ---
    def on_frame_change(frame_num):
        if not state.list_files_process: return None, gr.update()
        filename = state.list_files_process[state.selected_preview_index].filename
        from roop.capturer import get_video_frame
        frame = get_video_frame(filename, frame_num)
        return util.convert_to_gradio(frame), gr.update(info=f"Frame {int(frame_num)}")

    ui_comp["preview_frame_num"].release(fn=on_frame_change, inputs=[ui_comp["preview_frame_num"]], outputs=[ui_comp["previewimage"], ui_comp["preview_frame_num"]])

    def navigate_file(direction):
        if not state.list_files_process: return gr.update(), None, gr.update(), gr.update()
        state.selected_preview_index = min(len(state.list_files_process)-1, state.selected_preview_index + 1) if direction == "next" else max(0, state.selected_preview_index - 1)
        entry = state.list_files_process[state.selected_preview_index]
        roop.globals.target_path = entry.filename
        from roop.capturer import get_video_frame, get_image_frame
        frame = get_video_frame(entry.filename, 1) if util.is_video(entry.filename) else get_image_frame(entry.filename)
        return gr.update(maximum=entry.endframe, value=1), util.convert_to_gradio(frame), gr.update(interactive=state.selected_preview_index>0), gr.update(interactive=state.selected_preview_index<len(state.list_files_process)-1)

    ui_comp["bt_prev_file"].click(fn=lambda: navigate_file("prev"), outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]])
    ui_comp["bt_next_file"].click(fn=lambda: navigate_file("next"), outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]])

    # --- 5. BORRADO ---
    def on_clear_target():
        roop.globals.TARGET_FACES, ui.globals.ui_target_thumbs = [], []
        return [], "📄 Página 1 de 1 (0 caras)"
    ui_comp["bt_remove_selected_target_face"].click(fn=on_clear_target, outputs=[ui_comp["target_faces"], ui_comp["target_page_info"]])
