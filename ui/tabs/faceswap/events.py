import gradio as gr
import os
import roop.globals
import ui.globals
import ui.tabs.faceswap.state as state
import ui.tabs.faceswap.logic as logic
import roop.utilities as util

def on_face_selection_click(evt: gr.SelectData):
    """
    Maneja el clic directo en una cara detectada para añadirla a Destino.
    MODIFICACIÓN: Mantiene el panel visible para permitir selección múltiple.
    """
    if state._IS_UPDATING_GALLERY or not evt or evt.index is None:
        return gr.update(), gr.update(), gr.update()

    face_index = evt.index
    print(f"[UI] Cara seleccionada por clic: {face_index}")
    
    if state.SELECTION_FACES_DATA is None or face_index >= len(state.SELECTION_FACES_DATA):
        return gr.update(), gr.update(), gr.update()
        
    face_data = state.SELECTION_FACES_DATA[face_index]
    face_obj = face_data[0]
    face_img = face_data[1]
    
    # Smart Tracking: Solo en modo video 'selected_faces_frame'
    if roop.globals.face_swap_mode == 'selected_faces_frame' and roop.globals.target_path:
        video_key = os.path.basename(roop.globals.target_path)
        state.selected_face_references[video_key] = {
            'bbox': face_obj.bbox,
            'embedding': face_obj.embedding
        }
        print(f"[SmartTracking] Referencia guardada para {video_key}")

    # Añadir a destino
    image = util.convert_to_gradio(face_img, is_rgb=True)
    roop.globals.TARGET_FACES.append(face_obj)
    ui.globals.ui_target_thumbs.append(image)
    
    state._IS_UPDATING_TARGET = True
    target_page = logic.get_faces_for_page(ui.globals.ui_target_thumbs, "target")
    target_info = logic.update_pagination_info(ui.globals.ui_target_thumbs, "target")
    
    print(f"[UI] Cara añadida. Total en destino: {len(ui.globals.ui_target_thumbs)}")
    
    # IMPORTANTE: No cerramos el panel (visible=True)
    return gr.update(visible=True), target_page, target_info

def wire_events(ui_comp):
    """Conecta los componentes con la lógica"""
    
    # 1. ARCHIVOS ORIGEN
    def on_src_changed(files):
        if not files: return []
        # Limpiar selecciones previas de origen
        roop.globals.INPUT_FACESETS = []
        ui.globals.ui_input_thumbs = []
        
        for f in files:
            # Detección de caras en archivos de origen
            faces_data = logic.extract_face_images(f.name, is_source_face=True)
            for face_obj, face_img in faces_data:
                from roop.types import FaceSet
                roop.globals.INPUT_FACESETS.append(FaceSet(faces=[face_obj], name=os.path.basename(f.name)))
                ui.globals.ui_input_thumbs.append(util.convert_to_gradio(face_img, is_rgb=True))
        
        return logic.get_faces_for_page(ui.globals.ui_input_thumbs, "input")
    
    ui_comp["bt_srcfiles"].change(fn=on_src_changed, inputs=[ui_comp["bt_srcfiles"]], outputs=[ui_comp["input_faces"]])

    # 2. ARCHIVOS DESTINO
    def on_dest_changed(files):
        if not files: return gr.update(maximum=1), None, gr.update(interactive=False), gr.update(interactive=False)
        state.list_files_process = []
        for f in files:
            entry, _, _ = logic.process_target_file_async(f.name)
            if entry: state.list_files_process.append(entry)
        
        if not state.list_files_process: return gr.update(maximum=1), None, gr.update(interactive=False), gr.update(interactive=False)
        
        state.selected_preview_index = 0
        first_entry = state.list_files_process[0]
        roop.globals.target_path = first_entry.filename
        
        from roop.capturer import get_video_frame
        frame = get_video_frame(first_entry.filename, 1)
        
        return gr.update(maximum=first_entry.endframe, value=1), util.convert_to_gradio(frame), gr.update(interactive=False), gr.update(interactive=len(files)>1)

    ui_comp["bt_destfiles"].change(fn=on_dest_changed, inputs=[ui_comp["bt_destfiles"]], outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]])

    # 3. NAVEGACIÓN DE VIDEO (BLINDADA)
    def on_frame_change(frame_num):
        if not state.list_files_process: return None, gr.update(), gr.update(visible=False), []
        filename = state.list_files_process[state.selected_preview_index].filename
        from roop.capturer import get_video_frame
        frame = get_video_frame(filename, frame_num)
        
        # Al cambiar frame en video, detectamos TODAS las caras para permitir selección
        state.SELECTION_FACES_DATA = logic.extract_face_images(filename, (True, frame_num), target_face_detection=True)
        thumbs = [util.convert_to_gradio(f[1], is_rgb=True) for f in state.SELECTION_FACES_DATA]
        
        return util.convert_to_gradio(frame), gr.update(info=f"Frame {int(frame_num)}"), gr.update(visible=len(thumbs)>0), thumbs

    ui_comp["preview_frame_num"].release(
        fn=on_frame_change, 
        inputs=[ui_comp["preview_frame_num"]], 
        outputs=[ui_comp["previewimage"], ui_comp["preview_frame_num"], ui_comp["dynamic_face_selection"], ui_comp["face_selection"]]
    )

    # 4. SELECCIÓN DE CARAS (FIX ONCLICK + MULTISELECT)
    ui_comp["face_selection"].select(
        fn=on_face_selection_click,
        outputs=[ui_comp["dynamic_face_selection"], ui_comp["target_faces"], ui_comp["target_page_info"]]
    )

    # 5. NAVEGACIÓN DE ARCHIVOS
    def navigate_file(direction):
        if not state.list_files_process: return gr.update(), None, gr.update(), gr.update()
        
        if direction == "next":
            state.selected_preview_index = min(len(state.list_files_process)-1, state.selected_preview_index + 1)
        else:
            state.selected_preview_index = max(0, state.selected_preview_index - 1)
            
        entry = state.list_files_process[state.selected_preview_index]
        roop.globals.target_path = entry.filename
        
        from roop.capturer import get_video_frame
        frame = get_video_frame(entry.filename, 1)
        
        return gr.update(maximum=entry.endframe, value=1), util.convert_to_gradio(frame), gr.update(interactive=state.selected_preview_index>0), gr.update(interactive=state.selected_preview_index<len(state.list_files_process)-1)

    ui_comp["bt_prev_file"].click(fn=lambda: navigate_file("prev"), outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]])
    ui_comp["bt_next_file"].click(fn=lambda: navigate_file("next"), outputs=[ui_comp["preview_frame_num"], ui_comp["previewimage"], ui_comp["bt_prev_file"], ui_comp["bt_next_file"]])

    # 6. BORRADO
    def on_clear_target():
        roop.globals.TARGET_FACES = []
        ui.globals.ui_target_thumbs = []
        return [], "📄 Página 1 de 1 (0 caras)"
        
    ui_comp["bt_remove_selected_target_face"].click(
        fn=on_clear_target,
        outputs=[ui_comp["target_faces"], ui_comp["target_page_info"]]
    )
