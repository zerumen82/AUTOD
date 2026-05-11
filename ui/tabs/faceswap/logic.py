import os
import json
import time
import cv2
import numpy as np
import gradio as gr
import roop.globals
import roop.utilities as util
from concurrent.futures import ThreadPoolExecutor
from roop.ProcessEntry import ProcessEntry
import ui.tabs.faceswap.state as state
import ui.globals
from roop.core import get_processing_plugins, live_swap
from roop.ProcessOptions import ProcessOptions
import roop.face_util as face_util

# ============================================================
# Funciones de Utilidad y Lógica
# ============================================================

def get_metrics_html(percent, processed, total, time_elapsed, time_remaining, status):
    """Genera HTML de métricas profesional para FaceSwap"""
    bar_color = "linear-gradient(90deg, #3b82f6, #10b981)" if status != "Error" else "linear-gradient(90deg, #ef4444, #f59e0b)"
    safe_percent = max(0, min(100, percent))
    return f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #334155;">
        <h3 style="color: #3b82f6; margin-top: 0; font-size: 16px; border-bottom: 1px solid #334155; padding-bottom: 5px;">🔄 Progreso en Tiempo Real</h3>
        <div style="margin-bottom: 12px; background: rgba(255,255,255,0.05); border-radius: 8px; height: 24px; overflow: hidden; position: relative;">
            <div style="width: {safe_percent}%; height: 100%; background: {bar_color}; transition: width 0.4s ease-out; display: flex; align-items: center; justify-content: center;">
                <span style="color: white; font-size: 12px; font-weight: bold; text-shadow: 0 1px 2px rgba(0,0,0,0.5);">{safe_percent:.1f}%</span>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
            <div style="text-align: center;"><div style="color: #94a3b8; font-size: 11px;">ARCHIVOS</div><div style="color: #10b981; font-size: 20px; font-weight: bold;">{processed}/{total}</div></div>
            <div style="text-align: center;"><div style="color: #94a3b8; font-size: 11px;">TRANSCURRIDO</div><div style="color: #f59e0b; font-size: 18px; font-weight: bold;">{time_elapsed}</div></div>
            <div style="text-align: center;"><div style="color: #94a3b8; font-size: 11px;">RESTANTE</div><div style="color: #06b6d4; font-size: 18px; font-weight: bold;">{time_remaining}</div></div>
            <div style="text-align: center;"><div style="color: #94a3b8; font-size: 11px;">ESTADO</div><div style="color: #8b5cf6; font-size: 13px; font-weight: bold;">{status}</div></div>
        </div>
    </div>
    """

def validate_image_file(file_path):
    try:
        img_data = np.fromfile(file_path, dtype=np.uint8)
        if len(img_data) == 0: return False
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        return img is not None
    except: return False

def cleanup_temp_files():
    """Limpieza completa de archivos temporales al inicio y final"""
    import tempfile
    import shutil
    patterns = ("temp_frame_", "faceset_", "faceswap_", "roop_", "gradio_")
    
    # 1. Limpiar temp del sistema
    temp_dir = tempfile.gettempdir()
    try:
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if any(item.startswith(p) for p in patterns):
                try:
                    if os.path.isdir(item_path):
                        shutil.rmtree(item_path, ignore_errors=True)
                    else:
                        os.remove(item_path)
                except: pass
    except: pass
    
    # 2. Limpiar temp del proyecto (D:\.autodeep_temp)
    project_temp = "D:\\.autodeep_temp"
    try:
        if os.path.exists(project_temp):
            for item in os.listdir(project_temp):
                item_path = os.path.join(project_temp, item)
                if any(item.startswith(p) for p in patterns):
                    try:
                        if os.path.isdir(item_path):
                            shutil.rmtree(item_path, ignore_errors=True)
                        else:
                            os.remove(item_path)
                    except: pass
    except: pass

def load_folder_history():
    try:
        if os.path.exists("source_folder_history.json"):
            with open("source_folder_history.json", "r") as f:
                state.last_source_folder = json.load(f).get("last_source_folder", "")
        if os.path.exists("dest_folder_history.json"):
            with open("dest_folder_history.json", "r") as f:
                state.last_target_folder = json.load(f).get("last_target_folder", "")
    except: pass

def initialize_thread_pools():
    if state._source_thread_pool is None or state._source_thread_pool._shutdown:
        state._source_thread_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="SourceLoader")
    if state._target_thread_pool is None or state._target_thread_pool._shutdown:
        state._target_thread_pool = ThreadPoolExecutor(max_workers=16, thread_name_prefix="TargetLoader")

def process_target_file_async(file_path):
    try:
        filename = file_path
        list_entry = ProcessEntry(filename, 0, 0, 0)
        if util.is_video(filename) or filename.lower().endswith("gif"):
            from roop.capturer import get_video_frame, get_video_frame_total
            total_frames = get_video_frame_total(filename)
            current_frame = get_video_frame(filename, 1)
            list_entry.endframe = total_frames
        else:
            from roop.capturer import get_image_frame
            total_frames = 1
            current_frame = get_image_frame(filename)
            if current_frame is None: return None, None, 0
            list_entry.endframe = total_frames
        preview_img = util.convert_to_gradio(current_frame)
        return list_entry, preview_img, total_frames
    except: return None, None, 0

def extract_face_images(file_path, face_detection_params=(False, 0), target_face_detection=True, is_source_face=False, ui_padding=None):
    """
    Extrae imágenes de caras de un archivo usando el módulo face_util.
    Ahora incluye metadatos de género y tamaño para filtrado.
    """
    try:
        faces_data = face_util.extract_face_images(
            file_path, 
            face_detection_params, 
            target_face_detection=target_face_detection, 
            is_source_face=is_source_face,
            ui_padding=ui_padding
        )
        
        # Enriquecer con metadatos si no los tiene
        enriched_data = []
        for face_obj, face_img in faces_data:
            if not hasattr(face_obj, 'gender') or face_obj.gender is None:
                from roop.ProcessMgr import get_gender
                face_obj.gender = get_gender(face_obj)
            
            # Calcular tamaño relativo
            if hasattr(face_obj, 'bbox'):
                w = face_obj.bbox[2] - face_obj.bbox[0]
                h = face_obj.bbox[3] - face_obj.bbox[1]
                face_obj.area = w * h
            
            enriched_data.append((face_obj, face_img))
            
        return enriched_data
    except Exception as e:
        print(f"Error extrayendo caras: {e}")
        return []

def get_filtered_faces(faces_data, gender_filter="Todos", size_filter="Todos"):
    """Filtra los datos de caras según los criterios de la UI"""
    if not faces_data: return []
    
    filtered = []
    for face_obj, face_img in faces_data:
        # Filtro de género
        if gender_filter != "Todos":
            gender = getattr(face_obj, 'gender', None)
            if gender_filter == "Hombres" and gender != "male": continue
            if gender_filter == "Mujeres" and gender != "female": continue
            
        # Filtro de tamaño
        if size_filter != "Todos":
            area = getattr(face_obj, 'area', 0)
            if size_filter == "Grandes" and area < 20000: continue
            if size_filter == "Pequeñas" and area > 10000: continue
            if size_filter == "Medianas" and (area <= 10000 or area >= 20000): continue
            
        filtered.append((face_obj, face_img))
    return filtered

def draw_face_boxes(frame, faces_data, selected_idx=None):
    """Dibuja cajas alrededor de las caras detectadas para feedback visual"""
    if frame is None: return None
    
    canvas = frame.copy()
    for i, (face_obj, _) in enumerate(faces_data):
        if not hasattr(face_obj, 'bbox'): continue
        
        x1, y1, x2, y2 = [int(v) for v in face_obj.bbox]
        color = (59, 130, 246) # Azul
        thickness = 2
        
        if i == selected_idx:
            color = (16, 185, 129) # Verde para la seleccionada
            thickness = 4
            
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        
        # Etiqueta con ID o género
        label = f"#{i+1}"
        gender = getattr(face_obj, 'gender', None)
        if gender: label += f" ({'M' if gender=='male' else 'F'})"
        
        cv2.putText(canvas, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
    return canvas

def get_enhancer_preview(image_path, frame_num, face_idx, enhancer_name, blend_factor):
    """Genera una vista previa rápida (crop) del enhancer aplicado"""
    try:
        from roop.capturer import get_video_frame, get_image_frame
        from roop.ProcessMgr import ProcessMgr
        from roop.ProcessOptions import ProcessOptions
        
        # 1. Obtener frame
        if util.is_video(image_path):
            frame = get_video_frame(image_path, frame_num)
        else:
            frame = get_image_frame(image_path)
            
        if frame is None: return None, "No se pudo leer el frame"
        
        # 2. Detectar caras
        faces_data = extract_face_images(image_path, (True, frame_num) if util.is_video(image_path) else (False, 0))
        if not faces_data or face_idx >= len(faces_data):
            return None, "Cara no encontrada"
            
        target_face = faces_data[face_idx][0]
        
        # 3. Obtener cara origen
        if not roop.globals.INPUT_FACESETS:
            return None, "Carga una cara de origen primero"
            
        source_face = roop.globals.INPUT_FACESETS[0].faces[0] # Usar la primera por defecto para preview
        
        # 4. Configurar ProcessMgr para swap rápido
        mgr = ProcessMgr()
        # Mock options
        roop.globals.selected_enhancer = enhancer_name
        roop.globals.use_enhancer = enhancer_name != "None"
        roop.globals.enhancer_blend_factor = blend_factor
        
        mgr.initialize(roop.globals.INPUT_FACESETS, [target_face], None)
        
        # 5. Procesar solo la cara
        # Hacemos un crop generoso alrededor de la cara para el preview
        x1, y1, x2, y2 = [int(v) for v in target_face.bbox]
        h, w = frame.shape[:2]
        pad = int(max(x2-x1, y2-y1) * 0.5)
        cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
        cx2, cy2 = min(w, x2+pad), min(h, y2+pad)
        
        face_crop = frame[cy1:cy2, cx1:cx2].copy()
        
        # Ajustar bbox de target_face al crop
        original_bbox = target_face.bbox
        target_face.bbox = (x1-cx1, y1-cy1, x2-cx1, y2-cy1)
        
        processed_crop = mgr._process_face_swap_v21(source_face, target_face, face_crop.copy(), face_crop)
        
        # Restaurar bbox original
        target_face.bbox = original_bbox
        
        if processed_crop is None: return None, "Error procesando preview"
        
        # 6. Crear imagen comparativa (Original | Procesado)
        comparison = np.hstack((face_crop, processed_crop))
        return util.convert_to_gradio(comparison), "Preview generado"
        
    except Exception as e:
        print(f"[PREVIEW_ERROR] {e}")
        return None, f"Error: {str(e)}"

def find_best_matching_face(detected_faces, reference_embedding):
    if not reference_embedding or not detected_faces:
        return 0
    best_idx = 0
    max_similarity = -1
    for i, face_data in enumerate(detected_faces):
        face_obj = face_data[0]
        if hasattr(face_obj, 'embedding'):
            sim = np.dot(face_obj.embedding, reference_embedding) / (np.linalg.norm(face_obj.embedding) * np.linalg.norm(reference_embedding))
            if sim > max_similarity:
                max_similarity = sim
                best_idx = i
    return best_idx

def translate_swap_mode(mode_text):
    if mode_text == "All faces": return "all"
    return "selected"

def get_mode_text(mode):
    if mode == "all": return "All faces"
    return "Selected faces"

def start_swap_process(files, detection, enhancer, face_distance, blend_ratio):
    # Limpieza de temporales al inicio
    cleanup_temp_files()
    from roop.core import batch_process_regular
    state.is_processing = True
    total_files = len(state.list_files_process)
    start_t = time.time()
    roop.globals.face_swap_mode = translate_swap_mode(detection)
    roop.globals.selected_enhancer = enhancer
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    print(f"[Logic] Iniciando swap en modo: {roop.globals.face_swap_mode}")
    return gr.update(interactive=False), gr.update(interactive=True), "Procesando..."

def get_faces_for_page(thumbs, gallery_type):
    current_page = state.current_input_page if gallery_type == "input" else state.current_target_page
    start_idx = current_page * state.FACES_PER_PAGE
    end_idx = start_idx + state.FACES_PER_PAGE
    return thumbs[start_idx:end_idx]

def update_pagination_info(thumbs, gallery_type):
    current_page = state.current_input_page if gallery_type == "input" else state.current_target_page
    total_faces = len(thumbs)
    total_pages = max(1, (total_faces + state.FACES_PER_PAGE - 1) // state.FACES_PER_PAGE)
    start_face = current_page * state.FACES_PER_PAGE + 1
    end_face = min((current_page + 1) * state.FACES_PER_PAGE, total_faces)
    return f"📄 Página {current_page + 1} de {total_pages} ({start_face}-{end_face} de {total_faces} caras)"

def update_pagination_buttons(total_faces, gallery_type):
    current_page = state.current_input_page if gallery_type == "input" else state.current_target_page
    total_pages = max(1, (total_faces + state.FACES_PER_PAGE - 1) // state.FACES_PER_PAGE)
    prev_enabled = current_page > 0
    next_enabled = current_page < total_pages - 1
    return gr.update(interactive=prev_enabled), gr.update(interactive=next_enabled)


def start_swap(enhancer, keep_frames, wait_after_extraction, skip_audio, face_distance, blend_ratio, blend_mode, selected_mask_engine, processing_method, no_face_action, vr_mode, use_single_source_all, autorotate, temporal_smoothing, num_swap_steps, imagemask=None):

    """Inicia el procesamiento de face swap - Versión modular"""
    # LIMPIEZA DE TEMPORALES AL INICIO
    cleanup_temp_files()

    print(f"[DIAG] start_swap() called. INPUT_FACESETS has {len(getattr(roop.globals, 'INPUT_FACESETS', []))} facesets")

    from roop.core import batch_process_regular
    import time
    import shutil
    from ui.main import prepare_environment
    
    # EMITIR ESTADO INICIAL INMEDIATAMENTE
    yield (
        gr.update(variant="secondary", interactive=False),
        gr.update(variant="primary", interactive=True),
        get_metrics_html(0, 0, 0, "00:00", "--:--", "Validando..."),
    )

    # Verificar archivos de destino
    if not state.list_files_process or len(state.list_files_process) <= 0:
        error_msg = "[ERROR] No hay archivos de destino configurados."
        print(f"[DIAGNÓSTICO] {error_msg}")
        gr.Error(error_msg)
        yield (gr.update(variant="primary", interactive=True), gr.update(variant="stop", interactive=False), get_metrics_html(0, 0, 0, "00:00", "--:--", "Error: Sin archivos"))
        return
    
    # Verificar caras de entrada
    if not hasattr(roop.globals, "INPUT_FACESETS") or len(roop.globals.INPUT_FACESETS) <= 0:
        error_msg = "[ERROR] ERROR: No hay caras de origen configuradas. Carga imágenes con caras en 'Archivos Origen'."
        print(f"[DIAGNÓSTICO] {error_msg}")
        gr.Error(error_msg)
        yield (gr.update(variant="primary", interactive=True), gr.update(variant="stop", interactive=False), get_metrics_html(0, 0, 0, "00:00", "--:--", "Error: Sin caras origen"))
        return
    
    # Configurar parámetros
    roop.globals.face_swap_mode = 'selected'
    roop.globals.selected_enhancer = enhancer if enhancer else "None"
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    roop.globals.blend_mode = blend_mode
    roop.globals.keep_frames = keep_frames

    # Verificar caras de destino para fotos (en video el fallback es automático)
    is_video_present = any(util.is_video(entry.filename) for entry in state.list_files_process)
    if not is_video_present and (not hasattr(roop.globals, 'TARGET_FACES') or len(roop.globals.TARGET_FACES) <= 0):
        error_msg = "[ERROR] ERROR: No hay caras de destino seleccionadas. En fotos, debes elegir al menos una cara para cambiar."
        print(f"[DIAGNÓSTICO] {error_msg}")
        gr.Error(error_msg)
        yield (gr.update(variant="primary", interactive=True), gr.update(variant="stop", interactive=False), get_metrics_html(0, 0, 0, "00:00", "--:--", "Error: Sin selección"))
        return
    roop.globals.skip_audio = skip_audio
    roop.globals.temporal_smoothing = temporal_smoothing
    roop.globals.num_swap_steps = num_swap_steps
    roop.globals.autorotate_faces = autorotate
    
    print(f"[OK] Iniciando swap en modo: {roop.globals.face_swap_mode}")
    print(f"[OK] Archivos destino: {len(state.list_files_process)}")
    print(f"[OK] Caras origen: {len(roop.globals.INPUT_FACESETS)}")
    
    # Preparar entorno (configura output_path, etc.)
    try:
        prepare_environment()
        if not os.path.exists(roop.globals.output_path):
            os.makedirs(roop.globals.output_path, exist_ok=True)
        print(f"[OK] Entorno preparado. output_path: {roop.globals.output_path}")
    except Exception as e:
        print(f"[WARNING] Error preparando entorno: {e}")
    
    # Limpiar directorio de salida si está configurado
    if getattr(roop.globals, 'CFG', None) and roop.globals.CFG.clear_output:
        if hasattr(roop.globals, 'output_path') and roop.globals.output_path:
            try:
                shutil.rmtree(roop.globals.output_path)
                print(f"[OK] Directorio de salida limpiado: {roop.globals.output_path}")
            except Exception as e:
                print(f"[WARNING] No se pudo limpiar output_path: {e}")
    
    # Configurar parámetros adicionales que espera batch_process_regular
    if hasattr(roop.globals, 'CFG'):
        roop.globals.execution_threads = roop.globals.CFG.max_threads
        roop.globals.video_encoder = roop.globals.CFG.output_video_codec
        roop.globals.video_quality = roop.globals.CFG.video_quality
        roop.globals.max_memory = roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    
    # Iniciar procesamiento
    state.is_processing = True
    start_t = time.time()
    total_files = len(state.list_files_process)
    
    print(f"[DIAG] About to call batch_process_regular. INPUT_FACESETS has {len(getattr(roop.globals, 'INPUT_FACESETS', []))} facesets")
    
    # Preparar mask_engine
    mask_engine = selected_mask_engine if selected_mask_engine != "None" else None
    
    # DIAGNÓSTICO DETALLADO DE TARGET_FACES
    target_faces_list = getattr(roop.globals, 'TARGET_FACES', [])
    selected_refs = getattr(roop.globals, 'selected_face_references', {})
    
    print(f"[DIAGNÓSTICO BATCH] Iniciando FaceSwap...")
    print(f" - Modo traducido: {roop.globals.face_swap_mode}")
    print(f" - Caras de destino (TARGET_FACES): {len(target_faces_list)}")
    print(f" - Referencias guardadas (selected_face_references): {len(selected_refs)}")
    
    if roop.globals.face_swap_mode == 'selected':
        for i, ref_key in enumerate(selected_refs.keys()):
            print(f"   [Ref {i+1}] Clave: {ref_key}")
    
    # Yield estado inicial (botones)
    yield (
        gr.update(variant="secondary", interactive=False),
        gr.update(variant="primary", interactive=True),
        get_metrics_html(0, 0, total_files, "00:00", "--:--", "Iniciando..."),
    )
    
    try:
        for progress_percent, progress_message in batch_process_regular(
            state.list_files_process,
            mask_engine,
            "",
            processing_method == "In-Memory processing",
            {"layers": []},
            roop.globals.num_swap_steps,
            None,
            getattr(state, "SELECTED_FACE_INDEX", 0),
            temporal_smoothing,
        ):
            elapsed = time.time() - start_t
            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
            files_done = int(progress_percent / 100.0 * total_files) if total_files > 0 else 0
            time_remaining = "--:--"
            if progress_percent > 0 and files_done > 0:
                rate = files_done / elapsed
                remaining = (total_files - files_done) / rate if rate > 0 else 0
                time_remaining = time.strftime("%H:%M:%S", time.gmtime(remaining))
            
            print(f"[UI] Progreso global: {progress_percent:.1f}% - {progress_message}")
            
            yield (
                gr.update(variant="secondary", interactive=False),
                gr.update(variant="primary", interactive=True),
                get_metrics_html(progress_percent, files_done, total_files, elapsed_str, time_remaining, progress_message),
            )
        
        # Completado
        elapsed = time.time() - start_t
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        gr.Info(f"✅ Procesamiento completado. Archivos gestionados: {total_files}")
        print(f"[OK] Procesamiento completado en {elapsed_str}")
        
        yield (
            gr.update(variant="primary", interactive=True),
            gr.update(variant="secondary", interactive=False),
            get_metrics_html(100, total_files, total_files, elapsed_str, "--:--", "Completado"),
        )
        
    except Exception as e:
        error_msg = f"[ERROR] ERROR durante el procesamiento: {str(e)}"
        print(f"[DIAGNÓSTICO] {error_msg}")
        import traceback
        traceback.print_exc()
        gr.Error(error_msg)
        
        yield (
            gr.update(variant="primary", interactive=True),
            gr.update(variant="secondary", interactive=False),
            get_metrics_html(0, 0, total_files, "--:--", "--:--", "Error"),
        )
    finally:
        state.is_processing = False
        # Limpiar archivos temporales
        cleanup_temp_files()


def stop_swap():
    """Detiene el procesamiento"""
    roop.globals.processing = False
    gr.Info("Abortando el procesamiento: espere a que se detengan los subprocesos restantes")
    return (
        gr.update(variant="primary", interactive=True),
        gr.update(variant="secondary", interactive=False),
        None,
    )
