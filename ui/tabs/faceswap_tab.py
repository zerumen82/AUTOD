from concurrent.futures import ThreadPoolExecutor
import concurrent
import json
import os
import pathlib
import queue
import shutil
import sys
import time
from PIL import Image

# Configurar UTF-8 para stdout/stderr en Windows para evitar errores de charmap
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import gradio as gr

import roop.globals


# ============================================================
# Métricas en tiempo real (igual que img_editor_tab.py)
# ============================================================

def get_metrics_html(percent, processed, total, time_elapsed, time_remaining, status):
    """Genera HTML de métricas profesional para FaceSwap"""
    progress_color = "#3b82f6" if status not in ["Error", "Completado"] else ("#10b981" if status == "Completado" else "#ef4444")
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
import roop.utilities as util
import ui.globals
from roop.types import FaceSet
from roop.ProcessEntry import ProcessEntry
from roop.ProcessOptions import ProcessOptions

# Función de validación movida al inicio para evitar NameError
def validate_image_file(file_path):
    """Valida si un archivo es una imagen válida - versión rápida usando cv2"""
    try:
        # Usar cv2 que es más rápido que PIL para validación
        import cv2
        import numpy as np
        img_data = np.fromfile(file_path, dtype=np.uint8)
        if len(img_data) == 0:
            return False
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        return img is not None
    except Exception:
        return False

# Variables globales para control de hilos
_source_thread_pool = None
_target_thread_pool = None
_processing_queue = queue.Queue()
MAX_CONCURRENT_THREADS = 8  # Aumentado para mejor rendimiento
FACE_DETECTION_CACHE = {}  # Caché para resultados de detección de caras
MAX_CACHE_SIZE = 100  # Límite de cache

# Variables para recordar últimas carpetas
last_source_folder = None
last_target_folder = None

# Variables para almacenar archivos seleccionados y prevenir duplicados
source_filenames = []

# Variables para trackear tipo de procesamiento y limpieza
is_video_processing = False
is_image_processing = False

# Variables para paginación
current_input_page = 0
current_target_page = 0
FACES_PER_PAGE = 32

# Variables para selección de caras
SELECTED_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0
IS_INPUT = True
SELECTION_FACES_DATA = []
CURRENT_DETECTED_FACES = []  # Almacena las caras detectadas actualmente para mantener en la galería
TEMP_SELECTED_FACE_INDEX = 0  # Índice temporal para selección en galería (antes de confirmar)

# Flag para prevenir procesamiento de eventos de selección automáticos de Gradio
# Flag para prevenir eventos de selección automáticos durante actualización de galería
# _IS_UPDATING_GALLERY = True durante actualizaciones de target_faces (para ignorar select de target_faces)
# _IS_UPDATING_TARGET = True durante actualizaciones de target_faces específicas
_IS_UPDATING_GALLERY = False
_IS_UPDATING_TARGET = False  # Flag específico para actualizar target_faces

# Variables para procesamiento de videos
selected_preview_index = 0
list_files_process = []
current_video_fps = 30

# Variables de control de estado
is_processing = False
manual_masking = False


def save_source_folder_history():
    """Guarda el historial de carpetas de origen en Settings y JSON"""
    global last_source_folder
    if not last_source_folder:
        return
    try:
        with open("source_folder_history.json", "w") as f:
            json.dump({"last_source_folder": last_source_folder}, f)
        if hasattr(roop.globals, 'CFG') and roop.globals.CFG:
            roop.globals.CFG.last_source_folder = last_source_folder
            roop.globals.CFG.save()
    except:
        pass


def save_target_folder_history():
    """Guarda el historial de carpetas de destino en Settings y JSON"""
    global last_target_folder
    if not last_target_folder:
        return
    try:
        with open("dest_folder_history.json", "w") as f:
            json.dump({"last_target_folder": last_target_folder}, f)
        if hasattr(roop.globals, 'CFG') and roop.globals.CFG:
            roop.globals.CFG.last_target_folder = last_target_folder
            roop.globals.CFG.save()
    except:
        pass


def load_source_folder_history():
    """Carga el historial de carpetas de origen desde Settings o JSON"""
    global last_source_folder
    if roop.globals.CFG and getattr(roop.globals.CFG, 'last_source_folder', None):
        last_source_folder = roop.globals.CFG.last_source_folder
    if not last_source_folder:
        try:
            if os.path.exists("source_folder_history.json"):
                with open("source_folder_history.json", "r") as f:
                    last_source_folder = json.load(f).get("last_source_folder", "")
        except:
            pass
    if not last_source_folder:
        desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
        last_source_folder = os.path.join(desktop, "AutoDeep_Origen")
        os.makedirs(last_source_folder, exist_ok=True)


def load_target_folder_history():
    """Carga el historial de carpetas de destino desde Settings o JSON"""
    global last_target_folder
    if roop.globals.CFG and getattr(roop.globals.CFG, 'last_target_folder', None):
        last_target_folder = roop.globals.CFG.last_target_folder
    if not last_target_folder:
        try:
            if os.path.exists("dest_folder_history.json"):
                with open("dest_folder_history.json", "r") as f:
                    last_target_folder = json.load(f).get("last_target_folder", "")
        except:
            pass
    if not last_target_folder:
        desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
        last_target_folder = os.path.join(desktop, "AutoDeep_Destino")
        os.makedirs(last_target_folder, exist_ok=True)


def load_folder_history():
    """Carga ambos historiales de carpetas"""
    load_source_folder_history()
    load_target_folder_history()


def open_folder_dialog(initial_dir, title="Seleccionar Carpeta"):
    """Abre un diálogo nativo de selección de carpeta usando tkinter"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(initialdir=initial_dir, title=title)
        root.destroy()
        return folder if folder else None
    except Exception as e:
        print(f"[ERROR] No se pudo abrir el diálogo de carpeta: {e}")
        return None


class GradioFileShim:
    """Clase simple para emular el objeto de archivo de Gradio"""
    def __init__(self, path):
        self.name = path


def cleanup_thread_pools():
    """Limpia los pools de hilos al cerrar"""
    global _source_thread_pool, _target_thread_pool

    if _source_thread_pool:
        _source_thread_pool.shutdown(wait=False)
    if _target_thread_pool:
        _target_thread_pool.shutdown(wait=False)


def cleanup_temp_files():
    """Limpia archivos temporales generados durante el procesamiento"""
    import tempfile
    import shutil

    temp_dir = tempfile.gettempdir()
    cleaned_files = 0

    try:
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path) and item.startswith("faceset_"):
                shutil.rmtree(item_path, ignore_errors=True)
                cleaned_files += 1
            elif os.path.isfile(item_path) and item.startswith("temp_frame_") and item.endswith(".png"):
                os.remove(item_path)
                cleaned_files += 1

        if cleaned_files > 0:
            print(f"[LIMPIEZA] Se limpiaron {cleaned_files} archivos temporales")
    except Exception as e:
        print(f"[WARNING] Error en cleanup_temp_files: {e}")


def get_default_target_state():
    """Función auxiliar para obtener el estado por defecto de las caras destino"""
    faces_page = get_faces_for_page(ui.globals.ui_target_thumbs, "target")
    target_page_info = update_pagination_info(ui.globals.ui_target_thumbs, "target")
    bt_target_prev, bt_target_next = update_pagination_buttons(
        len(ui.globals.ui_target_thumbs), "target"
    )
    return faces_page, target_page_info, bt_target_prev, bt_target_next


def get_error_target_state(error_msg="**Error al procesar**"):
    """Función auxiliar para obtener el estado de error de las caras destino"""
    faces_page, target_page_info, bt_target_prev, bt_target_next = get_default_target_state()
    # CORRECCIÓN: Preservar el modo actual en estados de error
    current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
    if current_mode == 'selected_faces_frame':
        error_target_mode = "Selected faces frame"
    elif current_mode == 'selected_faces':
        error_target_mode = "Selected faces"
    elif current_mode == 'selected':
        error_target_mode = "Selected faces"
    else:
        error_target_mode = "All faces"
    
    return (
        gr.update(visible=False),  # Para dynamic_face_selection
        [],  # Para face_selection (lista vacía, no None)
        gr.update(),  # Para face_selector_slider (sin cambios)
        faces_page,
        error_target_mode,  # Usar modo preservado
        error_msg,
        target_page_info,
        bt_target_prev,
        bt_target_next,
    )


def gen_processing_text(start, end):
    return f"Processing frame range [{start} - {end}]"


def initialize_thread_pools():
    """Inicializa los pools de hilos para carga paralela"""
    global _source_thread_pool, _target_thread_pool

    if _source_thread_pool is None or _source_thread_pool._shutdown:
        # Aumentado workers de 10 a 16 para mejor paralelización en carga masiva
        _source_thread_pool = ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="SourceLoader"
        )

    if _target_thread_pool is None or _target_thread_pool._shutdown:
        _target_thread_pool = ThreadPoolExecutor(
            max_workers=16, thread_name_prefix="TargetLoader"
        )


def cleanup_cache():
    """Limpia el cache si es demasiado grande"""
    global FACE_DETECTION_CACHE
    if len(FACE_DETECTION_CACHE) > MAX_CACHE_SIZE:
        # Mantener solo los 50 más recientes
        keys_to_remove = list(FACE_DETECTION_CACHE.keys())[:-50]
        for key in keys_to_remove:
            del FACE_DETECTION_CACHE[key]


def process_target_file_async(file_path, progress_callback=None):
    """Procesa un archivo de destino de forma asíncrona"""
    try:
        filename = file_path

        list_entry = ProcessEntry(filename, 0, 0, 0)

        if util.is_video(filename) or filename.lower().endswith("gif"):
            from roop.capturer import get_video_frame, get_video_frame_total

            total_frames = get_video_frame_total(filename)
            current_video_fps = util.detect_fps(filename)
            current_frame = get_video_frame(filename, 1)
            list_entry.endframe = total_frames
            list_entry.fps = current_video_fps
        else:
            # Para imágenes, leer directamente sin validación previa
            # La validación ocurrirá implícitamente al leer
            from roop.capturer import get_image_frame

            total_frames = 1
            current_frame = get_image_frame(filename)
            
            # Si el frame es None, el archivo no es válido
            if current_frame is None:
                print(f"[WARNING] Archivo de destino inválido: {os.path.basename(filename)}")
                return None, None, 0
                
            list_entry.endframe = total_frames

        # convert_to_gradio maneja la conversión BGR->RGB automáticamente
        preview_img = util.convert_to_gradio(current_frame)

        return list_entry, preview_img, total_frames

    except Exception as e:
        print(f"[ERROR] Error procesando archivo destino {file_path}: {str(e)}")
        return None, None, 0


def on_destfiles_changed_async(destfiles, progress=gr.Progress()):
    """Maneja la carga de archivos de destino de forma asíncrona"""
    global \
        _target_thread_pool, \
        selected_preview_index, \
        list_files_process, \
        current_video_fps

    # print(f"[DEBUG] 📁 on_destfiles_changed_async llamado con {len(destfiles) if destfiles else 0} archivos")

    initialize_thread_pools()
    
    if destfiles is None or len(destfiles) < 1:
        list_files_process.clear()
        roop.globals.TARGET_FACES.clear()
        ui.globals.ui_target_thumbs.clear()
        # Reset mode to default when no files
        initial_target_mode = "Selected faces"
        roop.globals.face_swap_mode = 'selected_faces'
        return (
            gr.Slider(value=1, maximum=1, info="0:00:00"),
            "",
            None,
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            initial_target_mode,
            gr.Button(interactive=False),
            gr.Button(interactive=False),
        )
    
    # ORDENAR ARCHIVOS ALFABÉTICAMENTE para coincidir con preview
    destfiles.sort(key=lambda x: x.name.lower())
    print(f"[DEBUG] Folder Archivos ordenados: {[os.path.basename(f.name) for f in destfiles]}")

    # Limpiar lista existente y caras detectadas del archivo anterior
    list_files_process.clear()
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs.clear()
    last_target_folder = os.path.dirname(destfiles[0].name)
    save_target_folder_history()

    try:
        # print(f"[DEBUG] 🔄 Iniciando carga paralela de {len(destfiles)} archivos...")
        progress(0, desc="Iniciando carga paralela de archivos destino...")

        # Procesar archivos en paralelo
        futures = []
        for f in destfiles:
            # print(f"[DEBUG] 📄 Enviando a thread pool: {os.path.basename(f.name)}")
            future = _target_thread_pool.submit(process_target_file_async, f.name)
            futures.append(future)

        # Recopilar resultados
        results = []
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                result = future.result()
                if result[0] is not None:  # Si el procesamiento fue exitoso
                    results.append(result)
                    # print(f"[DEBUG] [OK] Archivo procesado {i + 1}/{len(futures)}")

                # Actualizar progreso
                progress(
                    (i + 1) / len(futures),
                    desc=f"Procesando archivo destino {i + 1}/{len(futures)}",
                )

            except Exception as e:
                print(f"[ERROR] [ERROR] Error en hilo de procesamiento destino: {str(e)}")
                # Si es un error de imagen inválida, mostrar warning
                if "UnidentifiedImageError" in str(e) or "cannot identify image file" in str(e):
                    filename = futures[i-1] if i > 0 else "desconocido"
                    msg = f"⚠️ ARCHIVO INVÁLIDO: Un archivo de destino no es una imagen válida o está corrupto. Ignorando archivo."
                    print(f"[WARNING] {msg}")
                    gr.Warning(msg)
                continue

        progress(1.0, desc="Completado")
        # print(f"[DEBUG] [OK] Procesamiento completado: {len(results)} archivos listos")

        # Actualizar listas y UI
        for list_entry, preview_img, total_frames in results:
            list_files_process.append(list_entry)

        # print(f"[DEBUG] 📋 list_files_process actualizado: {len(list_files_process)} entradas")

        if list_files_process:
            selected_preview_index = 0
            filename = list_files_process[0].filename
            # Obtener la primera imagen de preview
            preview_img = None
            for list_entry, img, _ in results:
                if list_entry.filename == filename:
                    preview_img = img
                    break

            # Verificar si hay videos para activar navegación de frames
            has_videos = any(util.is_video(entry.filename) or entry.filename.lower().endswith("gif") for entry in list_files_process)
            bt_frame_prev_enabled = has_videos
            bt_frame_next_enabled = has_videos
            
            # Verificar si hay múltiples archivos para activar navegación de archivos
            has_multiple_files = len(list_files_process) > 1
            bt_file_prev_enabled = has_multiple_files
            bt_file_next_enabled = has_multiple_files
            
            # CORRECCIÓN: Automatically set mode based on file type:
            # - If ALL files are images (JPG, PNG): Selected faces
            # - If ANY file is a video: Selected faces frame
            first_file = list_files_process[0].filename if list_files_process else None
            has_any_video = any(util.is_video(entry.filename) or entry.filename.lower().endswith("gif") for entry in list_files_process)
            
            if has_any_video:
                initial_target_mode = "Selected faces frame"
                # Also update the global mode for consistency
                roop.globals.face_swap_mode = 'selected_faces_frame'
            else:
                initial_target_mode = "Selected faces"
                # Also update the global mode for consistency
                roop.globals.face_swap_mode = 'selected_faces'
            
            if list_files_process[0].endframe > 1:
                # print(f"[DEBUG] 🎬 Video detectado: {list_files_process[0].endframe} frames")
                return (
                    gr.Slider(
                        value=1, maximum=list_files_process[0].endframe, info="0:00:00"
                    ),
                    gen_processing_text(
                        list_files_process[0].startframe, list_files_process[0].endframe
                    ),
                    preview_img,
                    gr.Button(interactive=bt_frame_prev_enabled),
                    gr.Button(interactive=bt_frame_next_enabled),
                    initial_target_mode,  # Usar modo preservado
                    gr.Button(interactive=bt_file_prev_enabled),
                    gr.Button(interactive=bt_file_next_enabled),
                )
            # print(f"[DEBUG] [IMG] Imagen detectada")
            return (
                gr.Slider(value=1, maximum=1, info="0:00:00"),
                "",
                preview_img,
                gr.Button(interactive=bt_frame_prev_enabled),
                gr.Button(interactive=bt_frame_next_enabled),
                initial_target_mode,  # Usar modo preservado
                gr.Button(interactive=bt_file_prev_enabled),
                gr.Button(interactive=bt_file_next_enabled),
            )
        
        # Caso de que list_files_process esté vacío
        return (
            gr.Slider(value=1, maximum=1, info="0:00:00"),
            "",
            None,
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            "Selected faces",
            gr.Button(interactive=False),
            gr.Button(interactive=False),
        )

    except Exception as e:
        print(f"[ERROR] [ERROR] Error en carga asíncrona de archivos destino: {str(e)}")
        import traceback

        traceback.print_exc()
        # CORRECCIÓN: Preservar el modo actual en caso de error
        current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
        if current_mode == 'selected_faces_frame':
            error_target_mode = "Selected faces frame"
        elif current_mode == 'selected_faces':
            error_target_mode = "Selected faces"
        elif current_mode == 'selected':
            error_target_mode = "Selected faces"
        else:
            error_target_mode = "All faces"
        return (
            gr.Slider(value=1, maximum=1, info="0:00:00"),
            "",
            None,
            gr.Button(interactive=False),
            gr.Button(interactive=False),
            error_target_mode,
            gr.Button(interactive=False),
            gr.Button(interactive=False),
        )


def cleanup_thread_pools():
    """Limpia los pools de hilos al cerrar"""
    global _source_thread_pool, _target_thread_pool

    if _source_thread_pool:
        _source_thread_pool.shutdown(wait=False)
    if _target_thread_pool:
        _target_thread_pool.shutdown(wait=False)


def cleanup_temp_files():
    """Limpia archivos temporales generados durante el procesamiento"""
    import tempfile
    import shutil

    temp_dir = tempfile.gettempdir()
    cleaned_files = 0

    try:
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            if os.path.isdir(item_path) and item.startswith("faceset_"):
                shutil.rmtree(item_path, ignore_errors=True)
                cleaned_files += 1
            elif os.path.isfile(item_path) and item.startswith("temp_frame_") and item.endswith(".png"):
                os.remove(item_path)
                cleaned_files += 1

        # Mostrar aviso
        if is_video_processing and is_image_processing:
            gr.Info("🧹 [LIMPIEZA] Se limpiaron archivos temporales generados durante el procesamiento de videos e imágenes.")
        elif is_video_processing:
            gr.Info("🧹 [LIMPIEZA] Se limpiaron archivos temporales generados durante el procesamiento de videos.")
        elif is_image_processing:
            gr.Info("🧹 [LIMPIEZA] Se limpiaron archivos temporales generados durante el procesamiento de imágenes.")
        else:
            gr.Info("🧹 [LIMPIEZA] No se encontraron archivos temporales para limpiar.")

        if cleaned_files > 0:
            gr.Info(f"   - Archivos/carpetas eliminados: {cleaned_files}")

    except Exception as e:
        gr.Info(f"[ERROR] Error durante la limpieza de temporales: {e}")


# Estado de la aplicación
selected_preview_index = 0
is_processing = False
list_files_process: list[ProcessEntry] = []
SELECTED_INPUT_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0
IS_INPUT = True
SELECTION_FACES_DATA = None

# Configuración de detección facial
no_face_choices = [
    "Use untouched original frame",
    "Retry rotated",
    "Skip Frame",
    "Skip Frame if no similar face",
]
current_video_fps = 50

# Configuración de paginación
FACES_PER_PAGE = 50
current_input_page = 0
current_target_page = 0


def faceswap_tab():
    global no_face_choices, previewimage

    # Cargar historial de carpetas al iniciar
    load_folder_history()

    roop.globals.distance_threshold = 0.6  # Más permisivo para mejor matching de caras similares
    roop.globals.blend_ratio = 1.0  # MÁXIMO PARECIDO: 100% cara de origen (siemppre máximo posible)
    # Configuración de CALIDAD MÁXIMA para el swap
    roop.globals.num_swap_steps = 10  # MÁXIMA CALIDAD: 10 pasos de intercambio
    if not hasattr(roop.globals, "face_swap_mode"):
        roop.globals.face_swap_mode = "selected"  # Modo por defecto más seguro

    # NOTA: El gr.Tab() ahora está en ui/main.py
    # CSS personalizado para mejor visual con Google Fonts
    gr.HTML("""
        <link href='https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap' rel='stylesheet'>
        <style>
            * { font-family: 'Inter', sans-serif; }
            .origin-gallery { border: 3px solid #2196F3 !important; border-radius: 10px; }
            .target-gallery { border: 3px solid #FF9800 !important; border-radius: 10px; }
            .face-counter {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                margin: 5px 0;
            }
            .dynamic-panel {
                background-color: rgba(33, 150, 243, 0.1);
                border: 2px dashed #2196F3;
                padding: 15px !important;
                margin-bottom: 15px !important;
                border-radius: 12px;
            }
        </style>

    """)

    with gr.Row(variant="panel"):
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(min_width=160):
                        gr.HTML('<div class="face-counter">🎨 CARAS DE ORIGEN</div>')
                        input_faces = gr.Gallery(
                            label="",
                            allow_preview=False,
                            preview=False,
                            height=400,  # Aumentado para mejor visualización
                            object_fit="cover",
                            columns=4,  # Reducido para mejor visualización
                            elem_classes=["origin-gallery"],
                        )

                        input_page_info = gr.Markdown("📄 Página 1 de 1 (0 caras)")

                        with gr.Row():
                            bt_input_prev = gr.Button(
                                "⬅ Anterior", size="sm", interactive=False
                            )
                            bt_input_next = gr.Button(
                                "Siguiente ➡", size="sm", interactive=False
                            )
                        # Parámetros optimizados automáticamente para máxima calidad
                        # No se requiere configuración manual
                        bt_remove_selected_input_face = gr.Button(
                            "❌ Remove selected", size="sm"
                        )
                        bt_clear_input_faces = gr.Button(
                            "💥 Clear all", variant="stop", size="sm"
                        )
                    with gr.Column(min_width=160):
                        gr.HTML('<div class="face-counter">🎯 CARAS DE DESTINO</div>')
                        target_faces = gr.Gallery(
                            label="",
                            allow_preview=True,
                            preview=False,
                            height=400,  # Aumentado para mejor visualización
                            object_fit="cover",
                            columns=4,  # Reducido para mejor visualización
                            elem_classes=["target-gallery"],
                        )

                        target_page_info = gr.Markdown("📄 Página 1 de 1 (0 caras)")
                        
                        with gr.Row():
                            bt_target_prev = gr.Button(
                                "⬅ Anterior", size="sm", interactive=False
                            )
                            bt_target_next = gr.Button(
                                "Siguiente ➡", size="sm", interactive=False
                            )

                        bt_remove_selected_target_face = gr.Button(
                            "❌ Remove selected", size="sm"
                        )
                        selected_target_text = gr.Markdown(
                            "**Cara de destino seleccionada:** Ninguna"
                        )
                with gr.Row(variant="panel"):
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown("**1️⃣ CARAS ORIGEN")
                            # Botón de carpeta eliminado
                            
                        bt_srcfiles = gr.Files(
                            label="Arrastra archivos o haz clic",
                            file_count="multiple",
                            file_types=["image", ".fsz"],
                            elem_id="filelist_src",
                            height=280,
                        )
                    with gr.Column():
                        with gr.Row():
                            gr.Markdown("**2️⃣ ARCHIVOS DESTINO")
                            # Botón de carpeta eliminado
                            
                        bt_destfiles = gr.Files(
                            label="Arrastra archivos o haz clic",
                            file_count="multiple",
                            file_types=["image", "video"],
                            elem_id="filelist_dest",
                            height=280,
                        )
            with gr.Column(scale=2):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            bt_prev_frame = gr.Button("⬅️ Frame Anterior", size="sm", interactive=False)
                            bt_next_frame = gr.Button("Frame Siguiente ➡️", size="sm", interactive=False)
                    with gr.Column(scale=3):
                        preview_frame_num = gr.Slider(
                            1,
                            1,
                            value=1,
                            label="🎬 Frame del Video",
                            info="0:00:00",
                            step=1.0,
                            interactive=True,
                        )
                        gr.HTML('<span title="🎬 Navegación de Frames:\n• Usa los botones ◀️ ▶️ para navegar frame a frame\n• O arrastra el slider para saltar a cualquier frame\n• Solo funciona cuando hay un video cargado">ℹ️</span>')
                with gr.Row():
                    text_frame_clip = gr.Markdown("Processing frame range [0 - 0]")
                    set_frame_start = gr.Button("⬅ Set as Start", size="sm")
                    set_frame_end = gr.Button("➡ Set as End", size="sm")
                
                previewimage = gr.Image(
                    label="Previsualización de la Imagen",
                    height=576,
                    interactive=True,
                    visible=True,
                    type="filepath",
                )
                
                with gr.Row(variant="panel"):
                    fake_preview = gr.Checkbox(
                        label="Frames de intercambio", value=False
                    )
                    bt_refresh_preview = gr.Button(
                        "🔄 Refresh", variant="secondary", size="sm"
                    )
                    bt_use_face_from_preview = gr.Button(
                        "✅ Use Face from this Frame", variant="primary", size="sm"
                    )
                
                # Botones de navegación para archivos de destino (horizontal)
                with gr.Row():
                    bt_prev_file = gr.Button("⬅️ Archivo Anterior", size="sm", interactive=False)
                    bt_next_file = gr.Button("Archivo Siguiente ➡️", size="sm", interactive=False)
                
                # ============================================================================
                # SECCIÓN DE PROCESAMIENTO Y RESULTADOS
                # ============================================================================
                
                # Panel de selección de caras (justo debajo de botones de navegación)
                with gr.Row(visible=False, variant="panel", elem_classes=["dynamic-panel"]) as dynamic_face_selection:
                    with gr.Column(scale=2):
                        face_selection = gr.Gallery(
                            elem_id="face_selection_gallery",
                            columns=6,  # Más columnas para mostrar más caras
                            height=250,  # Altura más grande para mostrar todas las caras con scroll
                            object_fit="contain",  # Mantener proporción de la cara
                            label="Caras detectadas",
                            allow_preview=False,  # Desactivar modo preview (imagen grande + miniaturas)
                            preview=False,  # No mostrar preview
                            interactive=False,  # No interactivo para evitar eventos de selección problemáticos
                        )
                        
                        # Slider para seleccionar cara (más confiable que el click en galería)
                        face_selector_slider = gr.Slider(
                            minimum=1,
                            maximum=1,
                            value=1,
                            step=1,
                            label="Selecciona el numero de cara",
                            info="Mueve el slider para seleccionar una cara",
                            interactive=True,
                        )
                        
                        # Botón para confirmar selección
                        bt_use_selected_face = gr.Button("[OK] Usar cara seleccionada", variant="primary", size="sm")
                
                # Dropdown de detección de caras (debajo del panel de selección)
                selected_face_detection = gr.Dropdown(
                    choices=["First found", "Selected faces", "Selected faces frame", "All faces"],
                    value="Selected faces",
                    label="🎭 Face Detection Mode",
                    info="First: mejor velocidad | Selected: usa caras elegido por usuario | All: procesa todas",
                    visible=True
                )
                
                # Panel horizontal completo con Advanced Settings
                with gr.Row(variant="panel"):
                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Mejoras de Calidad**")
                                
                                ui_selected_enhancer = gr.Dropdown(
                                    choices=["None", "GFPGAN", "CodeFormer", "Restoreformer++", "GPEN"],
                                    value="CodeFormer",
                                    label="Face Enhancer",
                                    info="CodeFormer=Mejor identidad | GPEN=Mejor calidad | GFPGAN=Fuerte | None=Máximo parecido"
                                )
                                
                                ui_enhancer_blend = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.15,
                                    step=0.05,
                                    label="Enhancer Blend",
                                    info="0=Máximo parecido, 0.15=Óptimo, 1=Más calidad"
                                )
                                
                                ui_color_match = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.1,
                                    step=0.05,
                                    label="Color Match",
                                    info="0=Máximo parecido, 0.1=Óptimo, 1=Más ajuste"
                                )
                                
                                ui_brightness = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=0.1,
                                    step=0.05,
                                    label="Brightness Adjust",
                                    info="0=Máximo parecido, 0.1=Óptimo, 1=Más ajuste"
                                )
                                
                                ui_blend_ratio = gr.Slider(
                                    minimum=0.5,
                                    maximum=1.0,
                                    value=0.95,
                                    step=0.05,
                                    label="Blend Ratio",
                                    info="1.0 = Máximo parecido a cara origen"
                                )
                                
                                ui_num_swap_steps = gr.Slider(
                                    minimum=1,
                                    maximum=10,
                                    value=10,
                                    step=1,
                                    label="Swap Steps",
                                    info="Más pasos = mejor calidad (más lento)"
                                )
                                
                                ui_face_distance = gr.Slider(
                                    minimum=0.1,
                                    maximum=1.0,
                                    value=0.6,
                                    step=0.05,
                                    label="Face Distance Threshold",
                                    info="Menor = más estricto en matching"
                                )
                            
                            with gr.Column():
                                gr.Markdown("**Máscara y Post-procesado**")
                                
                                ui_selected_mask_engine = gr.Dropdown(
                                    choices=["None", "Clip2Seg", "DFL XSeg"],
                                    value="DFL XSeg",
                                    label="Mask Engine",
                                    info="Segmentación facial avanzada"
                                )
                                
                                ui_mask_blur = gr.Slider(
                                    minimum=0,
                                    maximum=100,
                                    value=45,
                                    step=5,
                                    label="Mask Blur",
                                    info="Suavizado de bordes de máscara"
                                )
                                
                                ui_similarity_threshold = gr.Slider(
                                    minimum=0.1,
                                    maximum=0.8,
                                    value=0.3,
                                    step=0.05,
                                    label="Similarity Threshold",
                                    info="Umbral para matching de caras (menor = más permisivo)"
                                )
                            
                            with gr.Column():
                                gr.Markdown("**Detección y Tracking**")
                                
                                ui_gender_strictness = gr.Dropdown(
                                    choices=["permissive", "balanced", "strict"],
                                    value="permissive",
                                    label="Gender Detection Mode",
                                    info="Permissive: más caras detectadas, Strict: solo caras claras"
                                )
                                
                                ui_auto_rotate = gr.Checkbox(
                                    value=True,
                                    label="Auto-rotate Faces",
                                    info="Corregir orientación de caras automáticamente"
                                )
                                
                                ui_temporal_smoothing = gr.Checkbox(
                                    value=True,
                                    label="Temporal Smoothing",
                                    info="Reduce parpadeo en videos (suavizado temporal)"
                                )
                                
                                ui_color_correction = gr.Checkbox(
                                    value=True,
                                    label="Color Correction",
                                    info="Ajustar color de la cara al target"
                                )
                
                # Botones de acción y Preview (en fila separada)
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            bt_start = gr.Button("▶️ Generate", variant="primary", size="lg")
                            bt_stop = gr.Button("⏹️ Stop", variant="stop", size="lg", interactive=False)
                        with gr.Row():
                            bt_open_output = gr.Button("📂 Open Output Folder", size="sm")
                            bt_clear_destfiles = gr.Button("🗑️ Clear All", size="sm")
                    
                    with gr.Column(scale=2):
                        # Preview del resultado
                        result_image = gr.Image(
                            label="🖼️ Resultado",
                            height=300,
                            interactive=False,
                            type="filepath",
                        )
                        
                        # Panel de métricas en tiempo real (encima de resultados)
                        swap_metrics = gr.HTML(value=get_metrics_html(0, 0, 0, "--:--", "--:--", "Listo para empezar"))

                        # Galería de resultados para múltiples archivos
                        result_gallery = gr.Gallery(
                            label="📁 Resultados",
                            height=200,
                            columns=4,
                            visible=False,
                            allow_preview=True,
                        )
    
    def update_display(files):
        if files:
            return on_resultfiles_finished(files)
        return None, None, None
    
    # ============================================================================
    # COMPONENTES FALTANTES PARA CALLBACKS
    # ============================================================================
    
    # Crear componentes faltantes para los callbacks de preview
    # Estos son necesarios para el funcionamiento del slider de preview
    
    # Placeholder components - no son visibles, solo paraReferencia en callbacks
    # NOTA: selected_face_detection ya está definido en la UI principal (línea 749)
    
    _enhancer_placeholder = gr.Dropdown(
        choices=["None", "GFPGAN", "CodeFormer", "Restoreformer++", "GPEN"],
        value="None",
        visible=False
    )
    _mask_engine_placeholder = gr.Dropdown(
        choices=["None", "Clip2Seg", "DFL XSeg"],
        value="None",
        visible=False
    )
    _no_face_placeholder = gr.Dropdown(
        choices=["Use untouched original frame", "Retry rotated", "Skip Frame", "Skip Frame if no similar face"],
        value="Skip Frame",
        visible=False
    )
    _vr_mode_placeholder = gr.Checkbox(value=False, visible=False)
    _auto_rotate_placeholder = gr.Checkbox(value=True, visible=False)
    _maskimage_placeholder = gr.Image(visible=False)
    _num_steps_placeholder = gr.Slider(value=10, minimum=1, maximum=20, visible=False)
    
    # Listas de inputs/outputs para preview_frame_num.release
    previewinputs = [
        preview_frame_num,
        bt_destfiles,
        fake_preview,
        _enhancer_placeholder,
        selected_face_detection,
        _mask_engine_placeholder,  # face_distance
        _mask_engine_placeholder,  # selected_mask_engine
        _no_face_placeholder,      # no_face_action
        _vr_mode_placeholder,      # vr_mode
        _auto_rotate_placeholder,  # auto_rotate
        _maskimage_placeholder,    # maskimage
        _num_steps_placeholder,    # num_steps
    ]
    
    previewoutputs = [
        previewimage,
        previewimage,  # Esta es la imagen de swap (visible cuando fake_preview=True)
        preview_frame_num,
    ]
    
    # ============================================================================
    # EVENT HANDLERS - Conexiones de eventos para la UI
    # ============================================================================
    
    # Manejo de archivos de destino - CRÍTICO para cargar preview
    bt_destfiles.change(
        fn=on_destfiles_changed,
        inputs=[bt_destfiles],
        outputs=[
            preview_frame_num,
            text_frame_clip,
            previewimage,
            bt_prev_frame,
            bt_next_frame,
            selected_face_detection,
            bt_prev_file,
            bt_next_file,
        ],
        show_progress="hidden",
    )
    
    # Manejo de archivos de origen - ACTUALIZADO para devolver correctamente las caras
    bt_srcfiles.change(
        fn=on_srcfile_changed,
        inputs=[bt_srcfiles],
        outputs=[
            dynamic_face_selection,  # Primer output
            face_selection,  # Segundo output
            input_faces,  # Tercer output - caras de origen
            input_page_info,
            bt_input_prev,
            bt_input_next,
        ],
        show_progress="full",
    )
    
    # Selección de caras en galerías
    # Usamos JavaScript para capturar el índice y evitar errores con SelectData
    input_faces.select(
        fn=on_select_input_face_js,
        inputs=[],
        outputs=[],
        js="(evt) => { return [evt.index !== undefined ? evt.index : -1]; }"
    )
    
    # Nota: El evento select en target_faces puede dispararse cuando se actualiza la galería
    # Usamos JavaScript para pasar el índice directamente y evitar errores con SelectData
    target_faces.select(
        fn=on_select_target_face_js,
        inputs=[],
        outputs=[selected_target_text],
        js="(evt) => { return evt && evt.index !== undefined ? [evt.index] : [-1]; }"
    )
    
    # Slider de selección de cara eliminado
    # La selección se hace haciendo clic en las caras de la galería

    # Slider para seleccionar cara - evento change
    face_selector_slider.change(
        fn=on_face_slider_changed,
        inputs=[face_selector_slider],
        outputs=[
            dynamic_face_selection,
            face_selection,
            target_faces,
            selected_face_detection,
            selected_target_text,
            target_page_info,
            bt_target_prev,
            bt_target_next,
        ],
    )
    
    # Botón para confirmar selección de cara
    bt_use_selected_face.click(
        fn=on_confirm_use_selected_face,
        inputs=[],
        outputs=[
            dynamic_face_selection,
            face_selection,
            target_faces,
            selected_face_detection,
            selected_target_text,
            target_page_info,
            bt_target_prev,
            bt_target_next,
        ],
    )
    
    # Navegación de archivos
    bt_prev_file.click(
        fn=on_prev_file,
        inputs=[preview_frame_num],
        outputs=[preview_frame_num, previewimage, text_frame_clip, bt_prev_file, bt_next_file],
        api_name="prev_file",
    )
    
    bt_next_file.click(
        fn=on_next_file,
        inputs=[preview_frame_num],
        outputs=[preview_frame_num, previewimage, text_frame_clip, bt_prev_file, bt_next_file],
        api_name="next_file",
    )
    
    # Navegación de frames
    bt_prev_frame.click(
        fn=on_prev_frame,
        inputs=[preview_frame_num],
        outputs=[preview_frame_num, previewimage, text_frame_clip],
        api_name="prev_frame",
    )
    
    bt_next_frame.click(
        fn=on_next_frame,
        inputs=[preview_frame_num],
        outputs=[preview_frame_num, previewimage, text_frame_clip],
        api_name="next_frame",
    )
    
    # Limpiar botones
    bt_clear_input_faces.click(
        fn=on_clear_input_faces,
        outputs=[input_faces, input_page_info, bt_input_prev, bt_input_next],
    )
    
    # Botones de eliminar cara seleccionada
    bt_remove_selected_input_face.click(
        fn=remove_selected_input_face,
        outputs=[input_faces, input_page_info, bt_input_prev, bt_input_next],
    )
    
    bt_remove_selected_target_face.click(
        fn=remove_selected_target_face,
        outputs=[target_faces, selected_target_text, target_page_info, bt_target_prev, bt_target_next],
    )
    
    # Evento de cambio de modo de detección - handler vacío para ignorar cambios
    selected_face_detection.change(
        fn=lambda x: None,
        inputs=selected_face_detection,
        outputs=[],
    )
    
    # ============================================================================
    # PROCESAMIENTO - Botones de inicio/parada (componentes definidos en otra parte)
    # ============================================================================
    # Nota: Los componentes bt_start, bt_stop, start_event, resultfiles, etc.
    # deben definirse en otra parte del UI o pasarse como parámetros
    # Por ahora, comentamos estas líneas ya que los componentes no están定义
    
    # after_swap_event = start_event.then(
    #     fn=update_display,
    #     inputs=[resultfiles],
    #     outputs=[resultfiles_display, resultimage_display, resultvideo_display],
    # )
    
    # bt_stop.click(
    #     fn=stop_swap,
    #     cancels=[start_event, after_swap_event],
    #     outputs=[bt_start, bt_stop],
    #     queue=False,
    # )

    preview_frame_num.release(
        fn=on_preview_frame_changed,
        inputs=previewinputs,
        outputs=previewoutputs,
        show_progress="hidden",
    )
    bt_use_face_from_preview.click(
        fn=on_use_face_from_selected,
        show_progress="full",
        inputs=[],
        outputs=[
            dynamic_face_selection,
            face_selection,
            face_selector_slider,
            target_faces,
            selected_face_detection,
            selected_target_text,
            target_page_info,
            bt_target_prev,
            bt_target_next,
        ],
        api_name="use_face_from_frame",
    )
    set_frame_start.click(
        fn=on_set_frame,
        inputs=[set_frame_start, preview_frame_num],
        outputs=[text_frame_clip],
        api_name="set_frame_start",
    )
    bt_refresh_preview.click(
        fn=on_preview_frame_changed,
        inputs=previewinputs,
        outputs=previewoutputs,
        show_progress="hidden",
        api_name="refresh_preview",
    )

    set_frame_end.click(
        fn=on_set_frame,
        inputs=[set_frame_end, preview_frame_num],
        outputs=[text_frame_clip],
        api_name="set_frame_end",
    )
    
    # ============================================================================
    # EVENT HANDLERS - Botones de procesamiento
    # ============================================================================
    
    # Botón Generate - iniciar procesamiento
    bt_start.click(
        fn=start_swap,
        inputs=[
            ui_selected_enhancer,      # enhancer - desde UI
            selected_face_detection,
            gr.Checkbox(value=False, visible=False),  # keep_frames
            gr.Checkbox(value=False, visible=False),  # wait_after_extraction
            gr.Checkbox(value=False, visible=False),  # skip_audio
            gr.Checkbox(value=True, visible=False),   # use_enhancer
            ui_face_distance,          # face_distance - desde UI
            ui_blend_ratio,            # blend_ratio - desde UI
            gr.Dropdown(value="blend", choices=["blend"], visible=False), # blend_mode
            ui_selected_mask_engine,   # selected_mask_engine - desde UI
            gr.Dropdown(value="Inswapper 128", choices=["Inswapper 128"], visible=False), # processing_method
            _no_face_placeholder,                      # no_face_action
            _vr_mode_placeholder,                      # vr_mode
            gr.Checkbox(value=False, visible=False),  # use_single_source_all
            ui_auto_rotate,            # autorotate - desde UI
            ui_temporal_smoothing,     # temporal_smoothing - desde UI
            ui_num_swap_steps,         # num_swap_steps - desde UI
            _maskimage_placeholder,    # imagemask
            ui_similarity_threshold,   # similarity_threshold - desde UI
            ui_gender_strictness,      # gender_strictness - desde UI
            ui_color_correction,       # color_correction - desde UI
            # Nuevos controles de calidad
            ui_enhancer_blend,        # enhancer_blend_factor
            ui_color_match,            # color_match_strength
            ui_brightness,            # brightness_strength
        ],
        outputs=[bt_start, bt_stop, result_gallery, swap_metrics],
    )
    
    # Botón Stop - detener procesamiento
    bt_stop.click(
        fn=stop_swap,
        outputs=[bt_start, bt_stop],
    )
    
    # Botón Open Output Folder
    def open_output_folder():
        """Abre la carpeta de salida"""
        output_path = getattr(roop.globals, 'output_path', 'output')
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        import subprocess
        if os.name == 'nt':  # Windows
            os.startfile(output_path)
        elif os.name == 'posix':  # Linux/Mac
            subprocess.run(['xdg-open' if os.sys.platform == 'linux' else 'open', output_path])
        # No retornar nada si no hay outputs
    
    bt_open_output.click(
        fn=open_output_folder,
        outputs=[],
    )
    
    # Botón Clear All - limpiar archivos de destino
    bt_clear_destfiles.click(
        fn=on_clear_destfiles,
        outputs=[preview_frame_num, text_frame_clip, previewimage, bt_prev_frame, bt_next_frame],
    )


def on_content_type_changed(content_type):
    """Configura automáticamente los parámetros óptimos según el tipo de contenido"""
    if content_type == "Imágenes":
        # Configuración óptima para imágenes
        return [
            gr.update(value=0.25),  # max_face_distance - CONSERVADOR
            gr.update(value=1),  # num_swap_steps - CONSERVADOR
            gr.update(value=0.65),  # ui_blend_ratio - CONSERVADOR
            gr.update(value=45),  # mask_blur
            gr.update(value="DFL XSeg"),  # selected_mask_engine
            gr.update(value="CodeFormer"),  # ui_selected_enhancer
            gr.update(value="General", visible=True),  # sub_category
            gr.update(value="Selected faces"),  # selected_face_detection
        ]
    else:  # Videos
        # Configuración óptima para videos
        return [
            gr.update(value=0.25),  # max_face_distance - CONSERVADOR
            gr.update(value=1),  # num_swap_steps - CONSERVADOR
            gr.update(value=0.65),  # ui_blend_ratio - CONSERVADOR
            gr.update(value=45),  # mask_blur (más suave para boca)
            gr.update(value="DFL XSeg"),  # selected_mask_engine
            gr.update(value="CodeFormer"),  # ui_selected_enhancer
            gr.update(value="General", visible=True),  # sub_category
            gr.update(value="First found"),  # selected_face_detection
        ]


def on_sub_category_changed(sub_cat):
    """Configura parámetros según subcategoría seleccionada"""
    if sub_cat == "General":
        return [
            gr.update(value=0.015),
            gr.update(value=1),
            gr.update(value=0.99),
            gr.update(value=45),
            gr.update(value="DFL XSeg"),
            gr.update(value="CodeFormer"),
            gr.update(
                value="**General:** Configuración equilibrada - CodeFormer para mejor preservación de identidad."
            ),
        ]
    elif sub_cat == "Acciones de Boca y Objetos":
        return [
            gr.update(value=0.01),
            gr.update(value=2),
            gr.update(value=0.98),
            gr.update(value=40),
            gr.update(value="DFL XSeg"),
            gr.update(value="CodeFormer"),
            gr.update(
                value="**Acciones de Boca y Objetos:** Optimizado para escenas con apertura de boca, dientes o inserción de objetos. Mayor padding y pasos para mejor detalle."
            ),
        ]
    elif sub_cat == "Expresiones Faciales":
        return [
            gr.update(value=0.015),
            gr.update(value=1),
            gr.update(value=0.99),
            gr.update(value=40),
            gr.update(value="DFL XSeg"),
            gr.update(value="CodeFormer"),
            gr.update(
                value="**Expresiones Faciales:** Ideal para expresiones faciales dinámicas como sonrisas o gestos. CodeFormer para mejor identidad."
            ),
        ]
    elif sub_cat == "Modo Rápido":
        return [
            gr.update(value=0.02),
            gr.update(value=1),
            gr.update(value=0.99),
            gr.update(value=50),
            gr.update(value="None"),
            gr.update(value="None"),
            gr.update(
                value="**Modo Rápido:** Procesamiento acelerado con menos calidad. Útil para pruebas rápidas o videos largos donde la velocidad es prioritaria."
            ),
        ]
    elif sub_cat == "Modo Alta Calidad":
        return [
            gr.update(value=0.005),
            gr.update(value=3),
            gr.update(value=0.95),
            gr.update(value=25),
            gr.update(value="DFL XSeg"),
            gr.update(value="Restoreformer++"),
            gr.update(
                value="**Modo Alta Calidad:** Máxima calidad con detección precisa y mejor post-procesado. Más lento pero resultados superiores."
            ),
        ]



def on_srcfile_changed(srcfiles, progress=gr.Progress()):
    global SELECTION_FACES_DATA, IS_INPUT, current_input_page, _source_thread_pool, last_source_folder
    
    current_input_page = 0  # Reset pagination
    IS_INPUT = True
    MAX_TOTAL_FACES = 1000
    # Aumentar tamaño de lote para procesar más archivos por iteración
    BATCH_SIZE = 32  # Aumentado de 8 a 32 para mejor rendimiento con muchos archivos

    print(f"[DEBUG] on_srcfile_changed paralelo llamado con {len(srcfiles) if srcfiles else 0} archivos")
    
    # Guardar historial de carpeta
    if srcfiles and len(srcfiles) > 0:
        try:
            last_source_folder = os.path.dirname(srcfiles[0].name)
            save_source_folder_history()
        except Exception as e:
            print(f"[WARNING] No se pudo guardar historial de carpeta origen: {e}")

    if srcfiles is None or len(srcfiles) < 1:
        prev_btn, next_btn = update_pagination_buttons(0, "input")
        yield (
            gr.update(visible=False),
            [],
            [],
            "📄 Página 1 de 1 (0 caras)",
            prev_btn,
            next_btn,
        )
        return

    # Iniciar pools de hilos
    initialize_thread_pools()
    
    # Limpiar estado anterior
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    
    yield (
        gr.update(visible=False),  # dynamic_face_selection - oculto inicialmente
        [],  # face_selection - vacío inicialmente
        [],  # input_faces - vacío inicialmente
        "⌛ Iniciando cargador paralelo...",
        gr.update(interactive=False),
        gr.update(interactive=False),
    )
    
    total_faces_processed = 0
    total_files = len(srcfiles)
    
    def process_file_worker(file_info):
        """Worker para procesar un solo archivo en un hilo"""
        source_path = file_info.name
        results = []
        try:
            if source_path.lower().endswith("fsz"):
                # Los fsz son complejos, mejor procesarlos síncronamente o con su propia lógica
                # pero por simplicidad los llamamos aquí
                unzipfolder = os.path.join(os.environ["TEMP"], f"fs_{os.getpid()}_{int(time.time()*1000)}")
                from roop.face_util import extract_face_images_fast
                os.makedirs(unzipfolder, exist_ok=True)
                util.unzip(source_path, unzipfolder)
                for img_file in [f for f in os.listdir(unzipfolder) if f.endswith(".png")]:
                    res = extract_face_images_fast(os.path.join(unzipfolder, img_file), (False, 0), is_source_face=True)
                    if res: results.extend(res)
                shutil.rmtree(unzipfolder, ignore_errors=True)
            elif util.has_image_extension(source_path):
                from roop.face_util import extract_face_images_fast
                res = extract_face_images_fast(source_path, (False, 0), is_source_face=True)
                if res: results.extend(res)
        except Exception as e:
            print(f"[ERROR] Worker error: {e}")
        return results

    try:
        progress(0, desc=f"Preparando {total_files} archivos...")
        
        # Procesar en lotes para mantener la UI reactiva
        for i in range(0, len(srcfiles), BATCH_SIZE):
            batch = srcfiles[i:i + BATCH_SIZE]
            futures = [_source_thread_pool.submit(process_file_worker, f) for f in batch]
            
            # Recopilar resultados del lote actual
            batch_faces = 0
            for future in concurrent.futures.as_completed(futures):
                faces_data = future.result()
                if faces_data:
                    for face_obj, face_img in faces_data:
                        if total_faces_processed >= MAX_TOTAL_FACES:
                            break
                        
                        # Post-procesamiento ligero (thumbnail reducido a 128px para velocidad)
                        face_obj.mask_offsets = (0, 0.15, 0, 0, 1, 15)
                        # Reducir thumbnail a 128px para carga más rápida con muchos archivos
                        thumbnail = util.convert_to_gradio(face_img, is_rgb=True)
                        if thumbnail is not None and hasattr(thumbnail, 'shape'):
                            try:
                                import numpy as np
                                if len(thumbnail.shape) == 3:
                                    h, w = thumbnail.shape[:2]
                                    if h > 128 or w > 128:
                                        # Redimensionar a 128px máximo
                                        scale = 128 / max(h, w)
                                        new_h, new_w = int(h * scale), int(w * scale)
                                        thumbnail = cv2.resize(thumbnail, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            except:
                                pass
                        
                        ui.globals.ui_input_thumbs.append(thumbnail)
                        face_set = FaceSet()
                        face_set.faces.append(face_obj)
                        roop.globals.INPUT_FACESETS.append(face_set)
                        total_faces_processed += 1
                        batch_faces += 1

            # Actualizar progreso y UI después de cada lote
            files_processed = min(i + BATCH_SIZE, total_files)
            progress_val = files_processed / total_files
            progress(progress_val, desc=f"Cargadas {total_faces_processed} caras ({files_processed}/{total_files} archivos)")
            
            # Solo actualizar UI cada 3 lotes para reducir overhead
            if (i // BATCH_SIZE) % 3 == 0 or files_processed >= total_files:
                faces_page = get_faces_for_page(ui.globals.ui_input_thumbs, "input")
                page_info = update_pagination_info(ui.globals.ui_input_thumbs, "input")
                p_prev, p_next = update_pagination_buttons(len(ui.globals.ui_input_thumbs), "input")
                
                yield (
                    gr.update(visible=False),  # dynamic_face_selection
                    [],  # face_selection
                    faces_page,  # input_faces - las caras de la página actual
                    page_info,
                    p_prev,
                    p_next,
                )
            else:
                # Yield mínimo para mantener la UI viva
                yield (
                    gr.update(visible=False),
                    [],
                    gr.update(),  # No actualizar gallery
                    f"⌛ Procesando... {total_faces_processed} caras ({files_processed}/{total_files})",
                    gr.update(),
                    gr.update(),
                )

        if total_faces_processed > 0:
            gr.Info(f"✅ Finalizado: {total_faces_processed} rostros detectados")
        else:
            gr.Warning("⚠️ No se detectaron rostros")
            
    except Exception as e:
        print(f"[ERROR] Error general en carga paralela: {e}")
        import traceback
        traceback.print_exc()
        gr.Error(f"❌ Error: {str(e)}")

    # Rendimiento final garantizado
    faces_page = get_faces_for_page(ui.globals.ui_input_thumbs, "input")
    page_info = update_pagination_info(ui.globals.ui_input_thumbs, "input")
    p_prev, p_next = update_pagination_buttons(len(ui.globals.ui_input_thumbs), "input")
    yield (gr.update(visible=False), [], faces_page, page_info, p_prev, p_next)


def process_faceset_file(source_path, progress):
    """Procesa archivos .fsz de forma optimizada"""
    unzipfolder = os.path.join(os.environ["TEMP"], f"faceset_{os.getpid()}_{int(time.time())}")
    face_set = FaceSet()
    faces_count = 0
    
    try:
        os.makedirs(unzipfolder, exist_ok=True)
        util.unzip(source_path, unzipfolder)
        
        image_files = [f for f in os.listdir(unzipfolder) if f.endswith(".png")]
        
        for img_file in image_files:
            img_path = os.path.join(unzipfolder, img_file)
            try:
                from roop.face_util import extract_face_images_fast
                faces_data = extract_face_images_fast(img_path, (False, 0), is_source_face=True)
                
                if faces_data:
                    for face_data in faces_data:
                        if len(face_data) >= 2:
                            face_obj = face_data[0]
                            face_img = face_data[1]
                            
                            face_obj.mask_offsets = (0, 0.15, 0, 0, 1, 15)
                            face_set.faces.append(face_obj)
                            
                            # Crear thumbnail para cada cara del faceset si queremos que sea dinámico
                            if face_img is not None:
                                thumbnail = util.convert_to_gradio(face_img, is_rgb=True)
                                ui.globals.ui_input_thumbs.append(thumbnail)
                            
                            faces_count += 1
            except Exception as e:
                print(f"[WARNING] Error procesando {img_file}: {e}")
                continue
                
    finally:
        # Limpiar carpeta temporal
        if os.path.exists(unzipfolder):
            try:
                import shutil
                shutil.rmtree(unzipfolder)
            except:
                pass
    
    if faces_count > 0:
        if len(face_set.faces) > 1:
            face_set.AverageEmbeddings()
        roop.globals.INPUT_FACESETS.append(face_set)
    
    return faces_count


def process_image_file(source_path, progress):
    """Procesa archivos de imagen individuales de forma optimizada"""
    from roop.face_util import extract_face_images_fast
    
    faces_count = 0
    
    try:
        # Usar versión rápida de extracción
        faces_data = extract_face_images_fast(source_path, (False, 0), is_source_face=True)
        
        if faces_data:
            for face_data in faces_data:
                if len(face_data) >= 2:
                    face_obj = face_data[0]
                    face_img = face_data[1]
                    
                    face_obj.mask_offsets = (0, 0.15, 0, 0, 1, 15)
                    
                    # Crear thumbnail
                    if face_img is not None:
                        thumbnail = util.convert_to_gradio(face_img, is_rgb=True)
                        ui.globals.ui_input_thumbs.append(thumbnail)
                    
                    # Crear FaceSet individual para cada cara
                    face_set = FaceSet()
                    face_set.faces.append(face_obj)
                    roop.globals.INPUT_FACESETS.append(face_set)
                    
                    faces_count += 1
                    
    except Exception as e:
        print(f"[ERROR] Error en process_image_file: {e}")
        raise
    
    return faces_count



def on_select_input_face_js(face_index=-1):
    """Versión de on_select_input_face que recibe el índice desde JavaScript."""
    global SELECTED_INPUT_FACE_INDEX, _IS_UPDATING_GALLERY
    
    # DEBUG: Verificar el estado del flag cuando se llama al handler
    print(f"[DEBUG] on_select_input_face_js - face_index: {face_index}, _IS_UPDATING_TARGET: {_IS_UPDATING_TARGET}")
    
    # PROTEGER: si estamos actualizando la galería programáticamente, ignorar el evento
    if _IS_UPDATING_TARGET:
        print("[DEBUG] on_select_input_face_js - Ignorando evento durante actualización de target_faces")
        return
    
    # Actualizar el índice de cara de entrada seleccionada
    if face_index is not None and face_index >= 0:
        SELECTED_INPUT_FACE_INDEX = int(face_index)
        print(f"[DEBUG] on_select_input_face_js - SELECTED_INPUT_FACE_INDEX actualizado a: {SELECTED_INPUT_FACE_INDEX}")


def on_select_input_face(evt: gr.SelectData):
    global SELECTED_INPUT_FACE_INDEX, _IS_UPDATING_GALLERY
    
    # DEBUG: Verificar el estado del flag cuando se llama al handler
    print(f"[DEBUG] on_select_input_face - evt type: {type(evt)}, _IS_UPDATING_GALLERY: {_IS_UPDATING_GALLERY}")
    
    # PROTEGER: si estamos actualizando la galería programáticamente, ignorar el evento
    # NOTA: Solo ignorar si _IS_UPDATING_TARGET está activo (actualización de target_faces)
    if _IS_UPDATING_TARGET:
        print("[DEBUG] on_select_input_face - Ignorando evento durante actualización de target_faces")
        return
    
    # Cuando se llama sin evento (programáticamente), no necesitamos hacer nada especial
    # La selección de caras de entrada se maneja de otra manera


def remove_selected_input_face():
    global SELECTED_INPUT_FACE_INDEX

    # Check if there are any faces to remove
    if not roop.globals.INPUT_FACESETS or not ui.globals.ui_input_thumbs:
        pagination_buttons = update_pagination_buttons(0, "input")
        return (
            gr.update(value=[]),
            update_pagination_info([], "input"),
            pagination_buttons[0],
            pagination_buttons[1],
        )

    # Ensure the selected index is within bounds
    if SELECTED_INPUT_FACE_INDEX < 0 or SELECTED_INPUT_FACE_INDEX >= len(
        roop.globals.INPUT_FACESETS
    ):
        SELECTED_INPUT_FACE_INDEX = 0  # Reset to first face if out of bounds

    # Remove the selected face
    if 0 <= SELECTED_INPUT_FACE_INDEX < len(roop.globals.INPUT_FACESETS):
        roop.globals.INPUT_FACESETS.pop(SELECTED_INPUT_FACE_INDEX)

    # Remove the corresponding thumbnail if it exists
    if 0 <= SELECTED_INPUT_FACE_INDEX < len(ui.globals.ui_input_thumbs):
        ui.globals.ui_input_thumbs.pop(SELECTED_INPUT_FACE_INDEX)

    # Update the selected index if we removed the last face
    if SELECTED_INPUT_FACE_INDEX > 0 and SELECTED_INPUT_FACE_INDEX >= len(
        roop.globals.INPUT_FACESETS
    ):
        SELECTED_INPUT_FACE_INDEX = max(0, len(roop.globals.INPUT_FACESETS) - 1)

    # Get the updated faces for the current page
    faces_page = get_faces_for_page(ui.globals.ui_input_thumbs, "input")

    # Get pagination info and buttons
    pagination_info = update_pagination_info(ui.globals.ui_input_thumbs, "input")
    prev_btn, next_btn = update_pagination_buttons(
        len(ui.globals.ui_input_thumbs), "input"
    )

    return faces_page, pagination_info, prev_btn, next_btn


def on_select_target_face_js(face_index=-1):
    """Versión de on_select_target_face que recibe el índice desde JavaScript."""
    global SELECTED_TARGET_FACE_INDEX, _IS_UPDATING_TARGET
    
    # DEBUG: Verificar el estado del flag cuando se llama al handler
    print(f"[DEBUG] on_select_target_face_js - face_index: {face_index}, _IS_UPDATING_TARGET: {_IS_UPDATING_TARGET}")
    
    # PROTEGER: si estamos actualizando target_faces programáticamente, ignorar el evento
    if _IS_UPDATING_TARGET:
        print("[DEBUG] on_select_target_face_js - Ignorando evento durante actualización de target_faces")
        return "**Cara de destino seleccionada:** Ninguna"
    
    # CORRECCIÓN: Actualizar SELECTED_TARGET_FACE_INDEX con el índice recibido
    if face_index is not None and face_index >= 0:
        SELECTED_TARGET_FACE_INDEX = int(face_index)
        print(f"[DEBUG] on_select_target_face_js - SELECTED_TARGET_FACE_INDEX actualizado a: {SELECTED_TARGET_FACE_INDEX}")
    
    filename = roop.globals.target_path if hasattr(roop.globals, 'target_path') and roop.globals.target_path else "Ninguno"
    if SELECTED_TARGET_FACE_INDEX >= 0:
        return f"**Cara de destino seleccionada:** {SELECTED_TARGET_FACE_INDEX + 1} - {filename}"
    return "**Cara de destino seleccionada:** Ninguna"


def on_select_target_face(evt: gr.SelectData):
    global SELECTED_TARGET_FACE_INDEX, _IS_UPDATING_TARGET
    
    # DEBUG: Verificar el estado del flag cuando se llama al handler
    print(f"[DEBUG] on_select_target_face - evt type: {type(evt)}, _IS_UPDATING_TARGET: {_IS_UPDATING_TARGET}")
    
    # PROTEGER: si estamos actualizando target_faces programáticamente, ignorar el evento
    if _IS_UPDATING_TARGET:
        print("[DEBUG] on_select_target_face - Ignorando evento durante actualización de target_faces")
        return gr.update()
    
    # Protección contra eventos sin datos completos
    try:
        if evt is None:
            print("[DEBUG] on_select_target_face - evt es None")
            return gr.update()
        
        # CORRECCIÓN: Actualizar SELECTED_TARGET_FACE_INDEX con el índice del evento
        if hasattr(evt, 'index') and evt.index is not None:
            SELECTED_TARGET_FACE_INDEX = evt.index
            print(f"[DEBUG] on_select_target_face - SELECTED_TARGET_FACE_INDEX actualizado a: {SELECTED_TARGET_FACE_INDEX}")
    except Exception as e:
        print(f"[DEBUG] on_select_target_face - Error accediendo a evt: {e}")
        return gr.update()
    
    filename = roop.globals.target_path if hasattr(roop.globals, 'target_path') and roop.globals.target_path else "Ninguno"
    if SELECTED_TARGET_FACE_INDEX >= 0:
        return f"**Cara de destino seleccionada:** {SELECTED_TARGET_FACE_INDEX + 1} - {filename}"
    return "**Cara de destino seleccionada:** Ninguna"


def remove_selected_target_face():
    global SELECTED_TARGET_FACE_INDEX

    # Check if there are any faces to remove
    if not roop.globals.TARGET_FACES or not ui.globals.ui_target_thumbs:
        prev_btn, next_btn = update_pagination_buttons(0, "target")
        return (
            gr.update(value=[]),
            "**Cara de destino seleccionada:** Ninguna",
            update_pagination_info([], "target"),
            prev_btn,
            next_btn,
        )

    # Ensure the selected index is within bounds
    if SELECTED_TARGET_FACE_INDEX < 0 or SELECTED_TARGET_FACE_INDEX >= len(
        roop.globals.TARGET_FACES
    ):
        SELECTED_TARGET_FACE_INDEX = 0  # Reset to first face if out of bounds

    # Remove the selected face
    if 0 <= SELECTED_TARGET_FACE_INDEX < len(roop.globals.TARGET_FACES):
        roop.globals.TARGET_FACES.pop(SELECTED_TARGET_FACE_INDEX)

    # Remove the corresponding thumbnail if it exists
    if 0 <= SELECTED_TARGET_FACE_INDEX < len(ui.globals.ui_target_thumbs):
        ui.globals.ui_target_thumbs.pop(SELECTED_TARGET_FACE_INDEX)

    # Update the selected index if we removed the last face
    if SELECTED_TARGET_FACE_INDEX > 0 and SELECTED_TARGET_FACE_INDEX >= len(
        roop.globals.TARGET_FACES
    ):
        SELECTED_TARGET_FACE_INDEX = max(0, len(roop.globals.TARGET_FACES) - 1)

    # Get the updated faces for the current page
    faces_page = get_faces_for_page(ui.globals.ui_target_thumbs, "target")

    # Update the status message
    status_msg = "**Cara de destino seleccionada:** Ninguna"
    if roop.globals.TARGET_FACES:
        filename = (
            roop.globals.target_path
            if hasattr(roop.globals, "target_path") and roop.globals.target_path
            else "Ninguno"
        )
        status_msg = f"**Cara de destino seleccionada:** {SELECTED_TARGET_FACE_INDEX + 1} - {filename}"

    # Get pagination buttons
    prev_btn, next_btn = update_pagination_buttons(
        len(ui.globals.ui_target_thumbs), "target"
    )

    return (
        faces_page,
        status_msg,
        update_pagination_info(ui.globals.ui_target_thumbs, "target"),
        prev_btn,
        next_btn,
    )


def reset_updating_flags():
    """Función para resetear los flags de actualización después de que Gradio procese los eventos."""
    global _IS_UPDATING_GALLERY, _IS_UPDATING_TARGET
    _IS_UPDATING_TARGET = False
    _IS_UPDATING_GALLERY = False
    print(f"[DEBUG] reset_updating_flags - Flags reseteados")
    return None


def on_use_face_from_selected(frame_num=None):
    """Función para usar la cara del frame seleccionado.
    
    Args:
        frame_num: Número de frame seleccionado en el slider (pasado como input desde Gradio)
    """
    global IS_INPUT, SELECTION_FACES_DATA, SELECTED_FACE_INDEX, selected_preview_index, _IS_UPDATING_GALLERY, _IS_UPDATING_TARGET, CURRENT_DETECTED_FACES, list_files_process
    import os
    import tempfile
    import traceback

    from roop.face_util import extract_face_images
    from roop.capturer import get_video_frame
    
    # Asegurarse de que frame_num sea un número válido
    if frame_num is None or not isinstance(frame_num, (int, float)):
        frame_num = 1
    else:
        frame_num = int(frame_num)
    
    # DEBUG - Mensaje inicial para verificar que el botón funciona
    print(f"[DEBUG] =======================================")
    print(f"[DEBUG] BOTÓN PRESIONADO - frame_num: {frame_num}")
    print(f"[DEBUG] selected_preview_index: {selected_preview_index}")
    print(f"[DEBUG] list_files_process length: {len(list_files_process) if list_files_process else 0}")
    print(f"[DEBUG] =======================================")
    
    # Verificar si hay archivos cargados
    if not list_files_process or len(list_files_process) == 0:
        print("[DEBUG] ERROR: No hay archivos cargados")
        gr.Warning("⚠️ No hay archivos de destino cargados")
        return get_error_target_state("**Error:** No hay archivos de destino")
    
    # Verificar el índice
    if selected_preview_index < 0 or selected_preview_index >= len(list_files_process):
        print(f"[DEBUG] ERROR: Índice inválido {selected_preview_index}")
        gr.Warning("⚠️ Archivo no válido seleccionado")
        return get_error_target_state("**Error:** Archivo no válido")
    
    # Activar flag para prevenir eventos de selección automáticos durante actualización de target_faces
    _IS_UPDATING_TARGET = True
    _IS_UPDATING_GALLERY = True  # Mantenemos ambos por compatibilidad
    print(f"[DEBUG] on_use_face_from_selected - Activando _IS_UPDATING_TARGET = True")

    IS_INPUT = False
    thumbs = []
    temp_frame_path = None

    try:
        # El resto del código...
        current_filename = list_files_process[selected_preview_index].filename if list_files_process and selected_preview_index < len(list_files_process) else None

        # Nota: Antes se limpiaban las caras destino al cambiar de archivo, lo que borraba selecciones
        # de otros videos cuando se usa "Selected faces frame". No hacemos limpieza aquí para
        # preservar selecciones por archivo. Solo validamos que exista un archivo válido.
        print(f"[DEBUG] on_use_face_from_selected - selected_preview_index={selected_preview_index}, current_filename={current_filename}, previous_target={getattr(roop.globals, 'target_path', None)}")
        if not current_filename:
            return get_error_target_state("**Error:** Archivo de destino inválido")

        # Validar entrada - verificar que hay archivos cargados
        if not list_files_process or len(list_files_process) == 0:
            return get_error_target_state("**Error:** No hay archivos seleccionados")

        if not hasattr(roop, "globals") or not hasattr(roop.globals, "target_path"):
            return get_error_target_state("**Error:** Configuración global inválida")

        # Validar índice de vista previa
        if not hasattr(ui, "globals") or not hasattr(ui.globals, "ui_target_thumbs"):
            return get_error_target_state("**Error:** Configuración de interfaz inválida")

        # CORRECCIÓN CRÍTICA: Validar contra list_files_process, no files
        if not list_files_process or not (0 <= selected_preview_index < len(list_files_process)):
            print(f"[ERROR] Índice inválido: {selected_preview_index}, total archivos: {len(list_files_process) if list_files_process else 0}")
            return get_error_target_state(f"**Error:** Índice inválido ({selected_preview_index})")

        try:
            # CORRECCIÓN: Usar list_files_process en lugar de files
            roop.globals.target_path = list_files_process[selected_preview_index].filename
            print(f"[DEBUG] 🎯 Use this frame: Procesando archivo índice {selected_preview_index}: {os.path.basename(roop.globals.target_path)}")
            
            # Guardar la selección de cara para este video específico
            if hasattr(roop.globals, 'selected_faces_frame_selections'):
                # Si ya existe una selección para este video, usarla
                video_key = os.path.basename(roop.globals.target_path)
                if video_key in roop.globals.selected_faces_frame_selections:
                    SELECTED_FACE_INDEX = roop.globals.selected_faces_frame_selections[video_key]
                    print(f"[DEBUG] 🎯 Usando selección guardada para {video_key}: cara {SELECTED_FACE_INDEX}")
            
            if not os.path.exists(roop.globals.target_path):
                raise FileNotFoundError(
                    f"Archivo no encontrado: {roop.globals.target_path}"
                )

            # Manejo de imágenes
            if util.is_image(
                roop.globals.target_path
            ) and not roop.globals.target_path.lower().endswith(("gif")):
                img = cv2.imread(roop.globals.target_path)
                if img is None:
                    raise ValueError("No se pudo cargar la imagen")

                # NO resetear SELECTED_FACE_INDEX aquí - mantener la selección del usuario
                # SELECTED_FACE_INDEX = 0  # Comentado para preservar la selección del usuario
                SELECTION_FACES_DATA = extract_face_images(
                    roop.globals.target_path, (False, 0), target_face_detection=True, is_source_face=False  # DESTINO: menos exigente
                )

                if not SELECTION_FACES_DATA:
                    gr.Warning("⚠️ No se detectaron caras en la imagen (probó rotaciones automáticas)")
                    # NO BORRAR las selecciones previas, mantener el estado actual
                    faces_page, target_page_info, bt_target_prev, bt_target_next = get_default_target_state()
                    # CORRECCIÓN: Preservar el modo actual
                    current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
                    if current_mode == 'selected_faces_frame':
                        no_face_mode = "Selected faces frame"
                    elif current_mode == 'selected_faces':
                        no_face_mode = "Selected faces"
                    elif current_mode == 'selected':
                        no_face_mode = "Selected faces"
                    else:
                        no_face_mode = "All faces"
                    return (
                        gr.update(visible=False),  # Para dynamic_face_selection
                        [],  # Para face_selection (lista vacía, no None)
                        gr.update(),  # Para face_selector_slider (sin cambios)
                        faces_page,  # Mantener caras actuales
                        no_face_mode,
                        "**⚠️ No se detectaron caras en esta imagen**",
                        target_page_info,
                        bt_target_prev,
                        bt_target_next,
                    )

                # Procesar cada cara detectada
                for face_data in SELECTION_FACES_DATA:
                    # CORREGIR: Manejar el formato [face_obj, face_img]
                    if len(face_data) >= 2:
                        face_obj = face_data[0]  # Objeto Face
                        face_img = face_data[1]  # Imagen ya extraída

                        if hasattr(face_obj, "bbox") and len(face_obj.bbox) >= 4:
                            try:
                                # Usar la imagen ya extraída en lugar de reextraer
                                if face_img is not None and face_img.size > 0:
                                    # IMPORTANTE: face_img ya está en RGB desde extract_face_images
                                    image = util.convert_to_gradio(face_img, is_rgb=True)
                                    thumbs.append(image)
                            except Exception as e:
                                print(f"Error procesando cara: {e}")
                                continue

            # Manejo de videos/GIFs
            elif util.is_video(
                roop.globals.target_path
            ) or roop.globals.target_path.lower().endswith(("gif")):

                try:
                    # NO resetear SELECTED_FACE_INDEX aquí - mantener la selección del usuario
                    # SELECTED_FACE_INDEX = 0  # Comentado para preservar la selección del usuario
                    SELECTION_FACES_DATA = extract_face_images(
                        roop.globals.target_path,
                        (True, frame_num),
                        target_face_detection=True,
                        is_source_face=False,  # DESTINO: menos exigente
                    )

                    if not SELECTION_FACES_DATA:
                        gr.Warning("⚠️ No se detectaron caras en el frame del video (probó rotaciones automáticas)")
                        # NO BORRAR las selecciones previas, mantener el estado actual
                        faces_page, target_page_info, bt_target_prev, bt_target_next = get_default_target_state()
                        # CORRECCIÓN: Preservar el modo actual
                        current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
                        if current_mode == 'selected_faces_frame':
                            no_face_video_mode = "Selected faces frame"
                        elif current_mode == 'selected_faces':
                            no_face_video_mode = "Selected faces"
                        elif current_mode == 'selected':
                            no_face_video_mode = "Selected faces"
                        else:
                            no_face_video_mode = "All faces"
                        return (
                            gr.update(visible=False),  # Para dynamic_face_selection
                            [],  # Para face_selection (lista vacía, no None)
                            gr.update(),  # Para face_selector_slider (sin cambios)
                            faces_page,  # Mantener caras actuales
                            no_face_video_mode,
                            "**⚠️ No se detectaron caras en este frame**",
                            target_page_info,
                            bt_target_prev,
                            bt_target_next,
                        )

                    for face_data in SELECTION_FACES_DATA:
                        # CORREGIR: Manejar el formato [face_obj, face_img]
                        if len(face_data) >= 2:
                            face_obj = face_data[0]  # Objeto Face
                            face_img = face_data[1]  # Imagen ya extraída

                            if hasattr(face_obj, "bbox") and len(face_obj.bbox) >= 4:
                                try:
                                    # Usar la imagen ya extraída en lugar de reextraer
                                    if face_img is not None and face_img.size > 0:
                                        # IMPORTANTE: face_img ya está en RGB desde extract_face_images
                                        image = util.convert_to_gradio(face_img, is_rgb=True)
                                        thumbs.append(image)
                                except Exception as e:
                                    print(f"Error procesando cara en video: {e}")
                                    continue

                except Exception as e:
                    print(f"Error al extraer caras del video: {e}")
                    raise e

            # Determinar el modo actual
            current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')

            # Guardar las caras detectadas actualmente para mantener en la galería
            global CURRENT_DETECTED_FACES
            CURRENT_DETECTED_FACES = thumbs.copy()
            
            # GUARDAR SIEMPRE la selección de cara para el archivo actual
            # Esto es necesario tanto para Selected Faces como Selected Faces Frame
            if SELECTION_FACES_DATA:
                video_path = roop.globals.target_path
                video_key = f"selected_face_ref_{os.path.basename(video_path)}"
                
                # Guardar referencia persistente de la cara
                if not hasattr(roop.globals, 'selected_face_references'):
                    roop.globals.selected_face_references = {}
                
                # Guardar la primera cara como referencia por defecto
                if len(SELECTION_FACES_DATA) > 0:
                    face_data = SELECTION_FACES_DATA[0]  # [face_obj, face_img]
                    if len(face_data) >= 2:
                        face_obj = face_data[0]
                        face_img = face_data[1]
                        
                        if hasattr(face_obj, "bbox") and len(face_obj.bbox) >= 4:
                            if face_img is not None and face_img.size > 0:
                                face_ref_data = {
                                    'bbox': face_obj.bbox,
                                    'embedding': face_obj.embedding,
                                    'source_file': video_path,
                                    'face_obj': face_obj,
                                    'face_img': face_img
                                }
                                roop.globals.selected_face_references[video_key] = face_ref_data
                                print(f"[SELECTED_FACES] Cara guardada para {os.path.basename(video_path)}: {len(SELECTION_FACES_DATA)} detectadas, img shape: {face_img.shape}")

    
                
                # Asegurar que SELECTED_FACE_INDEX esté en rango válido
                if SELECTION_FACES_DATA and len(SELECTION_FACES_DATA) > 0:
                    if SELECTED_FACE_INDEX >= len(SELECTION_FACES_DATA):
                        SELECTED_FACE_INDEX = 0
                        print(f"[DEBUG] SELECTED_FACE_INDEX ajustado a 0 (rango: 0-{len(SELECTION_FACES_DATA)-1})")

            # Si solo hay una cara, procesamiento automático
            if len(thumbs) == 1 and SELECTION_FACES_DATA:
                face_data = SELECTION_FACES_DATA[0]  # [face_obj, face_img]
                if len(face_data) >= 2:
                    face_obj = face_data[0]
                    face_img = face_data[1]

                    if hasattr(face_obj, "bbox") and len(face_obj.bbox) >= 4:
                        try:
                            if face_img is not None and face_img.size > 0:
                                image = util.convert_to_gradio(face_img, is_rgb=True)
                                try:
                                    face_obj.source_file = roop.globals.target_path
                                except Exception:
                                    pass
                                
                                # Guardar imagen de referencia en el objeto cara para enhancer
                                face_obj.face_img_ref = face_img.copy()
                                
                                # Añadir a TARGET_FACES para procesamiento
                                roop.globals.TARGET_FACES.append(face_obj)
                                ui.globals.ui_target_thumbs.append(image)
                                SELECTED_TARGET_FACE_INDEX = len(roop.globals.TARGET_FACES) - 1
                                print(f"[DEBUG] Cara única agregada automáticamente, SELECTED_TARGET_FACE_INDEX={SELECTED_TARGET_FACE_INDEX}, face_img shape: {face_img.shape}")

                                faces_page = get_faces_for_page(ui.globals.ui_target_thumbs, "target")
                                target_page_info = update_pagination_info(ui.globals.ui_target_thumbs, "target")
                                bt_target_prev, bt_target_next = update_pagination_buttons(
                                    len(ui.globals.ui_target_thumbs), "target"
                                )

                                # Determinar el modo
                                if current_mode == 'selected_faces_frame':
                                    target_mode = "Selected faces frame"
                                elif current_mode == 'selected_faces':
                                    target_mode = "Selected faces"
                                elif current_mode == 'selected':
                                    target_mode = "Selected faces"
                                else:
                                    target_mode = "All faces"

                                # Una sola cara: ocultar panel de selección
                                return (
                                    gr.update(visible=False),  # Ocultar dynamic_face_selection
                                    [],  # Limpiar face_selection
                                    gr.update(),  # Para face_selector_slider (sin cambios)
                                    faces_page,
                                    target_mode,
                                    f"**Cara de destino:** {len(ui.globals.ui_target_thumbs)} - {os.path.basename(roop.globals.target_path)}",
                                    target_page_info,
                                    bt_target_prev,
                                    bt_target_next,
                                )
                        except Exception as e:
                            print(f"Error al procesar cara única: {e}")

            # Si hay múltiples caras: MOSTRAR GALERÍA para selección manual
            if len(thumbs) >= 1 and SELECTION_FACES_DATA:
                # Resetear flags para permitir selección
                _IS_UPDATING_GALLERY = False
                _IS_UPDATING_TARGET = False
                print(f"[DEBUG] Mostrando galería de selección con {len(thumbs)} caras")
                
                faces_page = get_faces_for_page(ui.globals.ui_target_thumbs, "target")
                target_page_info = update_pagination_info(ui.globals.ui_target_thumbs, "target")
                bt_target_prev, bt_target_next = update_pagination_buttons(
                    len(ui.globals.ui_target_thumbs), "target"
                )

                # Determinar el modo
                if current_mode == 'selected_faces_frame':
                    target_mode = "Selected faces frame"
                elif current_mode == 'selected_faces':
                    target_mode = "Selected faces"
                elif current_mode == 'selected':
                    target_mode = "Selected faces"
                else:
                    target_mode = "All faces"

                return (
                    gr.update(visible=True),  # Mostrar dynamic_face_selection
                    thumbs,  # Mostrar las caras detectadas
                    gr.update(maximum=len(thumbs), value=1),  # Actualizar slider
                    faces_page,
                    target_mode,
                    f"**Selecciona una cara** del archivo: {os.path.basename(roop.globals.target_path)} ({len(thumbs)} caras)",
                    target_page_info,
                    bt_target_prev,
                    bt_target_next,
                )

        except Exception as e:
            error_msg = f"Error al procesar el archivo: {str(e)}"
            print(f"[ERROR] {error_msg}")
            gr.Info(error_msg)
            return get_error_target_state("**Error al procesar la cara seleccionada**")

    except Exception as e:
        error_msg = f"Error inesperado: {str(e)}"
        print(f"[CRITICAL] {error_msg}")
        import traceback
        traceback.print_exc()
        gr.Info("Ocurrió un error inesperado")
        return get_error_target_state("**Error inesperado**")

    # Si no se ha retornado antes, mostrar estado por defecto
    # CORRECCIÓN: Preservar el modo actual
    current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
    if current_mode == 'selected_faces_frame':
        default_mode = "Selected faces frame"
    elif current_mode == 'selected_faces':
        default_mode = "Selected faces"
    elif current_mode == 'selected':
        default_mode = "Selected faces"
    else:
        default_mode = "All faces"
    return get_error_target_state("**Cara de destino seleccionada:** Ninguna")


def on_face_slider_changed(slider_value):
    """
    Maneja el cambio del slider de selección de cara.
    Resalta visualmente la cara seleccionada y guarda el índice temporalmente.
    
    Args:
        slider_value: int - Valor actual del slider (índice de la cara seleccionada, 1-based)
    """
    global TEMP_SELECTED_FACE_INDEX, SELECTION_FACES_DATA
    
    print(f"[DEBUG] on_face_slider_changed - slider_value: {slider_value}")
    
    # PROTEGER contra valores inválidos del slider
    if slider_value is None:
        slider_value = 1
    
    # Convertir a 0-based index
    face_index = int(slider_value) - 1
    
    # Validar que tenemos datos de caras
    if SELECTION_FACES_DATA is None or len(SELECTION_FACES_DATA) == 0:
        print("[ERROR] on_face_slider_changed - No hay SELECTION_FACES_DATA")
        return (
            gr.update(),  # dynamic_face_selection (sin cambios)
            gr.update(),  # face_selection (sin cambios)
            gr.update(),  # target_faces (sin cambios)
            gr.update(),  # selected_face_detection (sin cambios)
            gr.update(value="**[!] No hay caras detectadas**"),  # selected_target_text
            gr.update(),  # target_page_info (sin cambios)
            gr.update(),  # bt_target_prev (sin cambios)
            gr.update(),  # bt_target_next (sin cambios)
        )
    
    # Validar el índice
    if face_index < 0 or face_index >= len(SELECTION_FACES_DATA):
        face_index = 0
    
    # Guardar índice temporalmente
    TEMP_SELECTED_FACE_INDEX = face_index
    print(f"[DEBUG] on_face_slider_changed - TEMP_SELECTED_FACE_INDEX guardado: {TEMP_SELECTED_FACE_INDEX}")
    
    # CREAR GALERÍA CON CARA SELECCIONADA RESALTADA
    highlighted_thumbs = []
    for i, face_data in enumerate(SELECTION_FACES_DATA):
        if len(face_data) >= 2:
            face_img = face_data[1]  # Imagen ya extraída
            if face_img is not None and face_img.size > 0:
                # Añadir borde verde si es la cara seleccionada
                if i == face_index:
                    # Añadir borde verde de 4 píxeles
                    border_size = 4
                    bordered_img = cv2.copyMakeBorder(
                        face_img,
                        border_size, border_size, border_size, border_size,
                        cv2.BORDER_CONSTANT,
                        value=(0, 255, 0)  # Verde en RGB
                    )
                    image = util.convert_to_gradio(bordered_img, is_rgb=True)
                else:
                    image = util.convert_to_gradio(face_img, is_rgb=True)
                highlighted_thumbs.append(image)
    
    # Actualizar la galería con las caras resaltadas
    return (
        gr.update(),  # dynamic_face_selection (sin cambios)
        highlighted_thumbs,  # face_selection - galería con cara resaltada
        gr.update(),  # target_faces (sin cambios)
        gr.update(),  # selected_face_detection (sin cambios)
        gr.update(value=f"**[OK] Cara {face_index + 1} seleccionada - Haz clic en 'Usar cara seleccionada' para confirmar**"),  # selected_target_text
        gr.update(),  # target_page_info (sin cambios)
        gr.update(),  # bt_target_prev (sin cambios)
        gr.update(),  # bt_target_next (sin cambios)
    )


def on_face_click_select_js(face_index):
    """
    Cuando el usuario hace clic en una cara de la galería, la añade directamente a target_faces.
    Recibe el índice desde el componente oculto que JavaScript actualiza.
    
    Args:
        face_index: int - Índice de la cara seleccionada (desde el componente oculto)
    
    Returns:
        tuple: 8 valores para actualizar la UI
    """
    global SELECTED_FACE_INDEX, SELECTION_FACES_DATA, _IS_UPDATING_GALLERY
    
    print(f"[DEBUG] on_face_click_select_js - face_index recibido: {face_index}, tipo: {type(face_index)}")
    
    # Protección contra eventos automáticos durante actualización de galería
    if _IS_UPDATING_GALLERY:
        print(f"[DEBUG] on_face_click_select_js - Ignorando evento automático (_IS_UPDATING_GALLERY=True)")
        return (
            gr.update(),  # dynamic_face_selection
            gr.update(),  # face_selection
            gr.update(),  # target_faces
            gr.update(),  # selected_face_detection
            gr.update(),  # selected_target_text
            gr.update(),  # target_page_info
            gr.update(),  # bt_target_prev
            gr.update(),  # bt_target_next
        )
    
    # Convertir a entero si es necesario
    try:
        face_index = int(face_index)
    except (ValueError, TypeError):
        face_index = -1
    
    # Si el índice es inválido, ignorar
    if face_index is None or face_index < 0:
        print(f"[DEBUG] on_face_click_select_js - Índice inválido, ignorando")
        return (
            gr.update(),  # dynamic_face_selection
            gr.update(),  # face_selection
            gr.update(),  # target_faces
            gr.update(),  # selected_face_detection
            gr.update(),  # selected_target_text
            gr.update(),  # target_page_info
            gr.update(),  # bt_target_prev
            gr.update(),  # bt_target_next
        )
    
    # Guardar el índice seleccionado
    SELECTED_FACE_INDEX = face_index
    print(f"[DEBUG] on_face_click_select_js - SELECTED_FACE_INDEX guardado: {SELECTED_FACE_INDEX}")
    
    # AÑADIR LA CARA DIRECTAMENTE - llamar a on_selected_face
    print(f"[DEBUG] on_face_click_select_js - Añadiendo cara directamente...")
    return on_selected_face(SELECTED_FACE_INDEX)


def on_face_gallery_selected_js(face_index=0):
    """
    Versión de on_face_gallery_selected que recibe el índice desde JavaScript.
    
    IMPORTANTE: Esta función SOLO selecciona la cara, NO la agrega a target_faces.
    La cara se agrega cuando el usuario hace clic en "Confirmar".
    
    Args:
        face_index: int - Índice de la cara seleccionada (pasado desde JavaScript)
    """
    global SELECTED_FACE_INDEX, SELECTION_FACES_DATA, IS_INPUT, _IS_UPDATING_GALLERY
    
    # DEBUG: Verificar el estado del flag cuando se llama al handler
    print(f"[DEBUG] on_face_gallery_selected_js - face_index: {face_index}, _IS_UPDATING_GALLERY: {_IS_UPDATING_GALLERY}")
    
    # Convertir a entero si es necesario
    try:
        face_index = int(face_index)
    except (ValueError, TypeError):
        face_index = 0
    
    print(f"[DEBUG] on_face_gallery_selected_js - Cara seleccionada por usuario: índice {face_index}")
    
    # IMPORTANTE: NO agregar la cara a target_faces aquí
    # Solo actualizar el índice seleccionado para mostrar al usuario qué cara está seleccionada
    # La cara se agregará cuando el usuario haga clic en "Confirmar"
    
    # Validar que tenemos datos de caras
    if SELECTION_FACES_DATA is None or len(SELECTION_FACES_DATA) == 0:
        print("[ERROR] on_face_gallery_selected_js - No hay SELECTION_FACES_DATA")
        return (
            gr.update(visible=False), [],
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],
            gr.Dropdown(visible=True),
            gr.update(value="**Cara de destino seleccionada:** Sin datos de caras"),
            "Página 1 de 1 (0 caras)",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )
    
    # Validar el índice
    if face_index < 0 or face_index >= len(SELECTION_FACES_DATA):
        face_index = 0
    
    print(f"[DEBUG] on_face_gallery_selected_js - índice final: {face_index}")
    
    SELECTED_FACE_INDEX = face_index
    
    # Obtener la cara seleccionada para mostrar información
    if SELECTION_FACES_DATA and face_index < len(SELECTION_FACES_DATA):
        fd = SELECTION_FACES_DATA[face_index]
        if len(fd) >= 2:
            face_obj = fd[0]
            video_path = roop.globals.target_path if hasattr(roop.globals, 'target_path') else None
            if video_path:
                print(f"[DEBUG] on_face_gallery_selected_js - Cara {face_index} seleccionada (pendiente confirmar): {os.path.basename(video_path)}")
    
    # Mostrar mensaje de que debe confirmar la selección
    # Mantener la galería visible con las caras detectadas
    target_page_info = update_pagination_info(ui.globals.ui_target_thumbs, "target") if hasattr(ui.globals, 'ui_target_thumbs') else "Página 1 de 1"
    bt_target_prev, bt_target_next = update_pagination_buttons(len(ui.globals.ui_target_thumbs), "target") if hasattr(ui.globals, 'ui_target_thumbs') else (gr.update(), gr.update())
    
    # Determinar el modo actual
    current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
    if current_mode == 'selected_faces_frame':
        target_mode = "Selected faces frame"
    elif current_mode == 'selected_faces':
        target_mode = "Selected faces"
    elif current_mode == 'selected':
        target_mode = "Selected faces"
    else:
        target_mode = "All faces"
    
    return (
        gr.update(visible=True),  # Mantener visible la galería
        CURRENT_DETECTED_FACES if CURRENT_DETECTED_FACES else [],  # Mantener las caras detectadas
        ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces - mantener sin cambios
        target_mode,
        gr.update(value=f"**Cara seleccionada (haz clic en Confirmar):** Índice {face_index + 1}"),
        target_page_info,
        bt_target_prev,
        bt_target_next,
    )


def on_face_gallery_selected(evt: gr.SelectData = None):
    """
    Maneja la selección de una cara al hacer click en la galería face_selection.
    Recibe SelectData automáticamente de Gradio.
    
    SIMPLIFICADO: Cuando el usuario hace clic en una cara, se añade directamente a target_faces.
    
    Args:
        evt: gr.SelectData - Evento de selección con índice y valor
    """
    global SELECTED_FACE_INDEX, SELECTION_FACES_DATA, IS_INPUT, _IS_UPDATING_GALLERY
    
    # PROTEGER contra eventos de Gradio con datos incompletos
    # Esto ocurre cuando se actualiza la galería programáticamente
    if evt is None:
        print(f"[DEBUG] on_face_gallery_selected - evt es None, ignorando")
        return (
            gr.update(),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.update(),  # selected_face_detection
            gr.update(),  # selected_target_text
            gr.update(),  # target_page_info
            gr.update(),  # bt_target_prev
            gr.update(),  # bt_target_next
        )
    
    # Verificar que evt tiene datos válidos (index y value)
    # Cuando el usuario hace clic, evt.index y evt.value deben estar presentes
    try:
        face_index = evt.index
        evt_value = evt.value
        # Si evt_value es None, probablemente es un evento automático
        if evt_value is None:
            print(f"[DEBUG] on_face_gallery_selected - evt.value es None, ignorando evento automático")
            return (
                gr.update(),  # dynamic_face_selection
                [],  # face_selection
                ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
                gr.update(),  # selected_face_detection
                gr.update(),  # selected_target_text
                gr.update(),  # target_page_info
                gr.update(),  # bt_target_prev
                gr.update(),  # bt_target_next
            )
    except (KeyError, TypeError, AttributeError) as e:
        print(f"[DEBUG] on_face_gallery_selected - Error accediendo a evt: {e}, ignorando")
        return (
            gr.update(),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.update(),  # selected_face_detection
            gr.update(),  # selected_target_text
            gr.update(),  # target_page_info
            gr.update(),  # bt_target_prev
            gr.update(),  # bt_target_next
        )
    
    print(f"[DEBUG] on_face_gallery_selected - evt.index: {face_index}, evt.value: {type(evt_value)}")
    
    # SIMPLIFICADO: Añadir la cara directamente a target_faces
    return on_selected_face(face_index)


def on_face_gallery_selected_js(face_index):
    """
    Maneja la selección de una cara al hacer click en la galería face_selection.
    Recibe el índice directamente desde JavaScript.
    
    SIMPLIFICADO: Cuando el usuario hace clic en una cara, se añade directamente a target_faces.
    
    Args:
        face_index: int - Índice de la cara seleccionada (desde JavaScript)
    """
    global SELECTED_FACE_INDEX, SELECTION_FACES_DATA
    
    print(f"[DEBUG] on_face_gallery_selected_js - face_index recibido: {face_index}")
    
    # Convertir a entero si es necesario
    try:
        face_index = int(face_index)
    except (ValueError, TypeError):
        face_index = 0
        print(f"[DEBUG] on_face_gallery_selected_js - Error convirtiendo índice, usando 0")
    
    # Guardar el índice seleccionado
    SELECTED_FACE_INDEX = face_index
    
    # SIMPLIFICADO: Añadir la cara directamente a target_faces
    return on_selected_face(face_index)


def on_face_gallery_selected_from_js(face_index):
    """
    Maneja la selección de una cara al hacer click en la galería face_selection.
    Recibe el índice directamente desde JavaScript.
    
    IMPORTANTE: Esta función SOLO selecciona la cara, NO la agrega a target_faces.
    La cara se agrega cuando el usuario hace clic en "Confirmar".
    
    Args:
        face_index: int - Índice de la cara seleccionada (desde JavaScript)
    """
    print(f"[DEBUG] on_face_gallery_selected_from_js - face_index recibido: {face_index}")
    return _process_face_gallery_selection(face_index)


def _process_face_gallery_selection(face_index):
    """
    Función común para procesar la selección de una cara en la galería.
    
    Args:
        face_index: int - Índice de la cara seleccionada
    """
    global SELECTED_FACE_INDEX, SELECTION_FACES_DATA, IS_INPUT, _IS_UPDATING_GALLERY
    
    # Convertir a entero si es necesario
    try:
        face_index = int(face_index)
    except (ValueError, TypeError):
        face_index = 0
    
    print(f"[DEBUG] on_face_gallery_selected - Cara seleccionada por usuario: índice {face_index}")
    
    # IMPORTANTE: NO agregar la cara a target_faces aquí
    # Solo actualizar el índice seleccionado para mostrar al usuario qué cara está seleccionada
    # La cara se agregará cuando el usuario haga clic en "Confirmar"
    
    # Validar que tenemos datos de caras
    if SELECTION_FACES_DATA is None or len(SELECTION_FACES_DATA) == 0:
        print("[ERROR] on_face_gallery_selected - No hay SELECTION_FACES_DATA")
        return (
            gr.update(visible=False),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.Dropdown(visible=True),  # selected_face_detection
            gr.update(value="**Cara de destino seleccionada:** Sin datos de caras"),  # selected_target_text
            "Página 1 de 1 (0 caras)",  # target_page_info
            gr.update(interactive=False),  # bt_target_prev
            gr.update(interactive=False),  # bt_target_next
        )
    
    # Validar el índice
    if face_index < 0 or face_index >= len(SELECTION_FACES_DATA):
        face_index = 0
    
    print(f"[DEBUG] on_face_gallery_selected - índice final: {face_index}")
    
    SELECTED_FACE_INDEX = face_index
    
    # Obtener la cara seleccionada para mostrar información
    if SELECTION_FACES_DATA and face_index < len(SELECTION_FACES_DATA):
        fd = SELECTION_FACES_DATA[face_index]
        if len(fd) >= 2:
            face_obj = fd[0]
            video_path = roop.globals.target_path if hasattr(roop.globals, 'target_path') else None
            if video_path:
                print(f"[DEBUG] on_face_gallery_selected - Cara {face_index} seleccionada (pendiente confirmar): {os.path.basename(video_path)}")
    
    # Mostrar mensaje de que debe confirmar la selección
    # Mantener la galería visible con las caras detectadas
    target_page_info = update_pagination_info(ui.globals.ui_target_thumbs, "target") if hasattr(ui.globals, 'ui_target_thumbs') else "Página 1 de 1"
    bt_target_prev, bt_target_next = update_pagination_buttons(len(ui.globals.ui_target_thumbs), "target") if hasattr(ui.globals, 'ui_target_thumbs') else (gr.update(), gr.update())
    
    # Determinar el modo actual
    current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
    if current_mode == 'selected_faces_frame':
        target_mode = "Selected faces frame"
    elif current_mode == 'selected_faces':
        target_mode = "Selected faces"
    elif current_mode == 'selected':
        target_mode = "Selected faces"
    else:
        target_mode = "All faces"
    
    return (
        gr.update(visible=True),  # Mantener visible la galería
        CURRENT_DETECTED_FACES if CURRENT_DETECTED_FACES else [],  # Mantener las caras detectadas
        ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces - mantener sin cambios
        target_mode,
        gr.update(value=f"**Cara seleccionada (haz clic en Confirmar):** Índice {face_index + 1}"),
        target_page_info,
        bt_target_prev,
        bt_target_next,
    )


def on_face_dropdown_selected(selected_face):
    """
    Maneja la selección de una cara desde el Dropdown.
    Este método es MÁS CONFIABLE que la galería porque el Dropdown solo dispara
    eventos cuando el usuario selecciona explícitamente un valor.
    
    Args:
        selected_face: str - Valor seleccionado del Dropdown (ej: "Cara 1", "Cara 2", etc.)
    """
    global SELECTED_FACE_INDEX, SELECTION_FACES_DATA, IS_INPUT, _IS_UPDATING_GALLERY, _IS_UPDATING_TARGET
    
    print(f"[DEBUG] on_face_dropdown_selected - selected_face: {selected_face}")
    
    # Si no hay selección, no hacer nada
    if not selected_face:
        print("[DEBUG] on_face_dropdown_selected - Sin selección, ignorando")
        return (
            gr.update(),  # dynamic_face_selection
            gr.update(),  # face_selection
            gr.update(),  # target_faces
            gr.update(),  # selected_face_detection
            gr.update(),  # selected_target_text
            gr.update(),  # target_page_info
            gr.update(),  # bt_target_prev
            gr.update(),  # bt_target_next
        )
    
    # Extraer el índice del valor seleccionado (ej: "Cara 1" -> 0, "Cara 2" -> 1)
    try:
        # El formato es "Cara X" donde X es el número de cara (1-based)
        face_index = int(selected_face.split()[-1]) - 1
    except (ValueError, IndexError):
        print(f"[ERROR] on_face_dropdown_selected - No se pudo extraer índice de: {selected_face}")
        face_index = 0
    
    print(f"[DEBUG] on_face_dropdown_selected - Índice extraído: {face_index}")
    
    # Validar que tenemos datos de caras
    if SELECTION_FACES_DATA is None or len(SELECTION_FACES_DATA) == 0:
        print("[ERROR] on_face_dropdown_selected - No hay SELECTION_FACES_DATA")
        return get_error_target_state("**Error:** No hay caras detectadas")
    
    # Validar el índice
    if face_index < 0 or face_index >= len(SELECTION_FACES_DATA):
        print(f"[WARNING] on_face_dropdown_selected - Índice fuera de rango: {face_index}, ajustando a 0")
        face_index = 0
    
    # Actualizar el índice seleccionado
    SELECTED_FACE_INDEX = face_index
    
    # AÑADIR LA CARA DIRECTAMENTE A TARGET_FACES
    # Este es el comportamiento deseado: seleccionar del Dropdown añade la cara directamente
    print(f"[DEBUG] on_face_dropdown_selected - Añadiendo cara {face_index} a target_faces")
    
    # Llamar a on_selected_face para añadir la cara
    return on_selected_face(face_index)


def on_confirm_face_selection():
    """
    Confirma la selección de cara actual y la añade a target_faces.
    """
    global SELECTED_FACE_INDEX, SELECTION_FACES_DATA, _IS_UPDATING_GALLERY
    
    print(f"[DEBUG] on_confirm_face_selection - SELECTED_FACE_INDEX: {SELECTED_FACE_INDEX}")
    print(f"[DEBUG] on_confirm_face_selection - SELECTION_FACES_DATA disponible: {SELECTION_FACES_DATA is not None}")
    
    if SELECTION_FACES_DATA:
        print(f"[DEBUG] on_confirm_face_selection - Cantidad de caras: {len(SELECTION_FACES_DATA)}")
    
    # Verificar que tenemos datos de caras
    if SELECTION_FACES_DATA is None or len(SELECTION_FACES_DATA) == 0:
        print("[ERROR] on_confirm_face_selection - No hay SELECTION_FACES_DATA")
        return (
            gr.update(visible=False),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.Dropdown(visible=True),  # selected_face_detection
            gr.update(value="**Cara de destino seleccionada:** Sin datos de caras"),  # selected_target_text
            "Página 1 de 1 (0 caras)",  # target_page_info
            gr.update(interactive=False),  # bt_target_prev
            gr.update(interactive=False),  # bt_target_next
        )
    
    # Validar que el índice está en rango
    face_index = SELECTED_FACE_INDEX
    if face_index < 0 or face_index >= len(SELECTION_FACES_DATA):
        print(f"[WARNING] on_confirm_face_selection - Índice fuera de rango: {face_index}, ajustando a 0")
        face_index = 0
        SELECTED_FACE_INDEX = 0
    
    print(f"[DEBUG] on_confirm_face_selection - Confirmando cara índice: {face_index}")
    
    # Llamar a on_selected_face para añadir la cara
    return on_selected_face(face_index)


def on_select_face_from_gallery_js(face_index=-1):
    """
    Maneja la selección de cara directamente desde la galería face_selection.
    Solo guarda el índice temporalmente - el usuario debe confirmar con el botón.
    Recibe el índice directamente desde JavaScript para evitar errores de SelectData.
    
    Args:
        face_index: int - Índice de la cara seleccionada (desde JavaScript)
    """
    global TEMP_SELECTED_FACE_INDEX, SELECTION_FACES_DATA
    
    print(f"[DEBUG] on_select_face_from_gallery_js - face_index recibido: {face_index}, tipo: {type(face_index)}")
    
    # Validar índice
    if face_index is None or face_index < 0:
        print("[WARNING] on_select_face_from_gallery_js - Índice inválido")
        # No cambiar nada, solo mostrar mensaje
        return (
            gr.update(),  # dynamic_face_selection (sin cambios)
            gr.update(),  # face_selection (sin cambios)
            gr.update(),  # target_faces (sin cambios)
            gr.update(),  # selected_face_detection (sin cambios)
            gr.update(value="**[!] Selecciona una cara válida**"),  # selected_target_text
            gr.update(),  # target_page_info (sin cambios)
            gr.update(),  # bt_target_prev (sin cambios)
            gr.update(),  # bt_target_next (sin cambios)
        )
    
    # Guardar índice temporalmente
    TEMP_SELECTED_FACE_INDEX = int(face_index)
    print(f"[DEBUG] on_select_face_from_gallery_js - TEMP_SELECTED_FACE_INDEX guardado: {TEMP_SELECTED_FACE_INDEX}")
    
    # Mostrar mensaje de confirmación sin añadir la cara todavía
    return (
        gr.update(),  # dynamic_face_selection (sin cambios)
        gr.update(),  # face_selection (sin cambios)
        gr.update(),  # target_faces (sin cambios)
        gr.update(),  # selected_face_detection (sin cambios)
        gr.update(value=f"**[OK] Cara {face_index + 1} seleccionada - Haz clic en 'Usar cara seleccionada' para confirmar**"),  # selected_target_text
        gr.update(),  # target_page_info (sin cambios)
        gr.update(),  # bt_target_prev (sin cambios)
        gr.update(),  # bt_target_next (sin cambios)
    )


def on_select_face_from_gallery_v2(evt: gr.SelectData):
    """
    Maneja la selección de cara directamente desde la galería face_selection.
    Usa SelectData de Gradio 6.x para obtener el índice.
    
    Args:
        evt: gr.SelectData - Evento de selección de Gradio
    """
    global TEMP_SELECTED_FACE_INDEX, SELECTION_FACES_DATA
    
    print(f"[DEBUG] on_select_face_from_gallery_v2 - evt: {evt}")
    print(f"[DEBUG] on_select_face_from_gallery_v2 - evt.index: {evt.index if evt else 'None'}")
    print(f"[DEBUG] on_select_face_from_gallery_v2 - evt.value: {evt.value if hasattr(evt, 'value') else 'N/A'}")
    
    # Obtener índice desde SelectData
    face_index = evt.index if evt else -1
    
    # Validar índice
    if face_index is None or face_index < 0:
        print("[WARNING] on_select_face_from_gallery_v2 - Índice inválido")
        return (
            gr.update(),  # dynamic_face_selection (sin cambios)
            gr.update(),  # face_selection (sin cambios)
            gr.update(),  # target_faces (sin cambios)
            gr.update(),  # selected_face_detection (sin cambios)
            gr.update(value="**[!] Selecciona una cara válida**"),  # selected_target_text
            gr.update(),  # target_page_info (sin cambios)
            gr.update(),  # bt_target_prev (sin cambios)
            gr.update(),  # bt_target_next (sin cambios)
        )
    
    # Guardar índice temporalmente
    TEMP_SELECTED_FACE_INDEX = face_index
    print(f"[DEBUG] on_select_face_from_gallery_v2 - TEMP_SELECTED_FACE_INDEX guardado: {TEMP_SELECTED_FACE_INDEX}")
    
    # Mostrar mensaje de confirmación sin añadir la cara todavía
    return (
        gr.update(),  # dynamic_face_selection (sin cambios)
        gr.update(),  # face_selection (sin cambios)
        gr.update(),  # target_faces (sin cambios)
        gr.update(),  # selected_face_detection (sin cambios)
        gr.update(value=f"**[OK] Cara {face_index + 1} seleccionada - Haz clic en 'Usar cara seleccionada' para confirmar**"),  # selected_target_text
        gr.update(),  # target_page_info (sin cambios)
        gr.update(),  # bt_target_prev (sin cambios)
        gr.update(),  # bt_target_next (sin cambios)
    )
    
    # Validar índice
    if face_index is None or face_index < 0:
        print("[WARNING] on_select_face_from_gallery_js - Índice inválido")
        # No cambiar nada, solo mostrar mensaje
        return (
            gr.update(),  # dynamic_face_selection (sin cambios)
            gr.update(),  # face_selection (sin cambios)
            gr.update(),  # target_faces (sin cambios)
            gr.update(),  # selected_face_detection (sin cambios)
            gr.update(value="**[!] Selecciona una cara válida**"),  # selected_target_text
            gr.update(),  # target_page_info (sin cambios)
            gr.update(),  # bt_target_prev (sin cambios)
            gr.update(),  # bt_target_next (sin cambios)
        )
    
    # Guardar índice temporalmente
    TEMP_SELECTED_FACE_INDEX = face_index
    print(f"[DEBUG] on_select_face_from_gallery_js - TEMP_SELECTED_FACE_INDEX guardado: {TEMP_SELECTED_FACE_INDEX}")
    
    # Mostrar mensaje de confirmación sin añadir la cara todavía
    return (
        gr.update(),  # dynamic_face_selection (sin cambios)
        gr.update(),  # face_selection (sin cambios)
        gr.update(),  # target_faces (sin cambios)
        gr.update(),  # selected_face_detection (sin cambios)
        gr.update(value=f"**[OK] Cara {face_index + 1} seleccionada - Haz clic en 'Usar cara seleccionada' para confirmar**"),  # selected_target_text
        gr.update(),  # target_page_info (sin cambios)
        gr.update(),  # bt_target_prev (sin cambios)
        gr.update(),  # bt_target_next (sin cambios)
    )


def on_confirm_use_selected_face():
    """
    Confirma la selección de cara y la añade a target_faces.
    Usa el índice guardado temporalmente en TEMP_SELECTED_FACE_INDEX.
    """
    global TEMP_SELECTED_FACE_INDEX, SELECTION_FACES_DATA
    
    print(f"[DEBUG] on_confirm_use_selected_face - TEMP_SELECTED_FACE_INDEX: {TEMP_SELECTED_FACE_INDEX}")
    print(f"[DEBUG] on_confirm_use_selected_face - SELECTION_FACES_DATA disponible: {SELECTION_FACES_DATA is not None}")
    
    if SELECTION_FACES_DATA:
        print(f"[DEBUG] on_confirm_use_selected_face - Cantidad de caras: {len(SELECTION_FACES_DATA)}")
    
    # Verificar que tenemos datos de caras
    if SELECTION_FACES_DATA is None or len(SELECTION_FACES_DATA) == 0:
        print("[ERROR] on_confirm_use_selected_face - No hay SELECTION_FACES_DATA")
        return (
            gr.update(visible=False),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.Dropdown(visible=True),  # selected_face_detection
            gr.update(value="**Cara de destino seleccionada:** Sin datos de caras"),  # selected_target_text
            "Página 1 de 1 (0 caras)",  # target_page_info
            gr.update(interactive=False),  # bt_target_prev
            gr.update(interactive=False),  # bt_target_next
        )
    
    # Validar índice temporal
    if TEMP_SELECTED_FACE_INDEX < 0 or TEMP_SELECTED_FACE_INDEX >= len(SELECTION_FACES_DATA):
        print(f"[WARNING] on_confirm_use_selected_face - Índice temporal inválido: {TEMP_SELECTED_FACE_INDEX}")
        return (
            gr.update(),  # dynamic_face_selection (sin cambios)
            gr.update(),  # face_selection (sin cambios)
            gr.update(),  # target_faces (sin cambios)
            gr.update(),  # selected_face_detection (sin cambios)
            gr.update(value="**[!] Índice de cara inválido**"),  # selected_target_text
            gr.update(),  # target_page_info (sin cambios)
            gr.update(),  # bt_target_prev (sin cambios)
            gr.update(),  # bt_target_next (sin cambios)
        )
    
    print(f"[DEBUG] on_confirm_use_selected_face - Confirmando cara índice: {TEMP_SELECTED_FACE_INDEX}")
    
    # Llamar a on_selected_face para añadir la cara
    return on_selected_face(TEMP_SELECTED_FACE_INDEX)


def on_confirm_face_selection_from_dropdown(dropdown_value):
    """
    Confirma la selección de cara desde el Dropdown y la añade a target_faces.
    
    Args:
        dropdown_value: str - Valor del Dropdown (ej: "Cara 1", "Cara 2", etc.)
    """
    global SELECTED_FACE_INDEX, SELECTION_FACES_DATA, _IS_UPDATING_GALLERY
    
    print(f"[DEBUG] on_confirm_face_selection_from_dropdown - dropdown_value: {dropdown_value}")
    print(f"[DEBUG] on_confirm_face_selection_from_dropdown - SELECTION_FACES_DATA disponible: {SELECTION_FACES_DATA is not None}")
    
    if SELECTION_FACES_DATA:
        print(f"[DEBUG] on_confirm_face_selection_from_dropdown - Cantidad de caras: {len(SELECTION_FACES_DATA)}")
    
    # Verificar que tenemos datos de caras
    if SELECTION_FACES_DATA is None or len(SELECTION_FACES_DATA) == 0:
        print("[ERROR] on_confirm_face_selection_from_dropdown - No hay SELECTION_FACES_DATA")
        return (
            gr.update(visible=False),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.Dropdown(visible=True),  # selected_face_detection
            gr.update(value="**Cara de destino seleccionada:** Sin datos de caras"),  # selected_target_text
            "Página 1 de 1 (0 caras)",  # target_page_info
            gr.update(interactive=False),  # bt_target_prev
            gr.update(interactive=False),  # bt_target_next
        )
    
    # Extraer el índice del valor del Dropdown (ej: "Cara 1" -> 0, "Cara 2" -> 1)
    face_index = 0
    if dropdown_value and isinstance(dropdown_value, str):
        try:
            # El formato es "Cara X" donde X es el número (1-based)
            parts = dropdown_value.split()
            if len(parts) >= 2:
                face_index = int(parts[1]) - 1  # Convertir a 0-based
                print(f"[DEBUG] on_confirm_face_selection_from_dropdown - Índice extraído: {face_index}")
        except (ValueError, IndexError) as e:
            print(f"[WARNING] on_confirm_face_selection_from_dropdown - Error extrayendo índice: {e}")
            face_index = 0
    
    # Validar que el índice está en rango
    if face_index < 0 or face_index >= len(SELECTION_FACES_DATA):
        print(f"[WARNING] on_confirm_face_selection_from_dropdown - Índice fuera de rango: {face_index}, ajustando a 0")
        face_index = 0
    
    SELECTED_FACE_INDEX = face_index
    print(f"[DEBUG] on_confirm_face_selection_from_dropdown - Confirmando cara índice: {face_index}")
    
    # Llamar a on_selected_face para añadir la cara
    return on_selected_face(face_index)


def on_select_face_js(face_index):
    """
    Maneja la selección de una cara en la galería face_selection.
    Recibe el índice directamente desde JavaScript para evitar errores de Gradio.
    
    Args:
        face_index: int - Índice de la cara seleccionada
    """
    global SELECTED_FACE_INDEX
    try:
        # Convertir a entero si es necesario
        if face_index is None:
            print(f"[DEBUG] on_select_face_js - Índice es None")
            return
            
        SELECTED_FACE_INDEX = int(face_index)
        print(f"[DEBUG] on_select_face_js - Cara seleccionada: índice {SELECTED_FACE_INDEX}")
        
        # Si estamos en modo selected_faces_frame, guardar la selección para el video actual
        if hasattr(roop.globals, 'face_swap_mode') and roop.globals.face_swap_mode == 'selected_faces_frame':
            print(f"[DEBUG] on_select_face_js - Modo selected_faces_frame detectado")
            if hasattr(roop.globals, 'target_path') and roop.globals.target_path:
                video_key = os.path.basename(roop.globals.target_path)
                print(f"[DEBUG] on_select_face_js - Video actual: {video_key}")
                
                if not hasattr(roop.globals, 'selected_faces_frame_selections'):
                    roop.globals.selected_faces_frame_selections = {}
                    print(f"[DEBUG] on_select_face_js - Creando diccionario de selecciones")
                
                roop.globals.selected_faces_frame_selections[video_key] = SELECTED_FACE_INDEX
                print(f"[DEBUG] on_select_face_js - Guardando selección para {video_key}: cara {SELECTED_FACE_INDEX}")
                print(f"[DEBUG] on_select_face_js - Diccionario actual: {roop.globals.selected_faces_frame_selections}")
            else:
                print(f"[DEBUG] on_select_face_js - No hay target_path disponible")
        else:
            print(f"[DEBUG] on_select_face_js - No está en modo selected_faces_frame o no hay target_path")
        
    except Exception as e:
        print(f"Error al seleccionar la cara: {e}")
        import traceback
        traceback.print_exc()
    return


def on_select_face(evt: gr.SelectData):
    """
    Maneja la selección de una cara en la galería face_selection.
    
    Args:
        evt: gr.SelectData - Evento de selección de Gradio
    
    Returns:
        tuple: Actualizaciones para los componentes de la UI
    """
    global SELECTED_FACE_INDEX
    try:
        # PROTEGER contra eventos de Gradio con datos incompletos
        # Esto ocurre cuando se actualiza la galería programáticamente
        if evt is not None and isinstance(evt, dict):
            # Si es un dict, verificar si tiene los campos necesarios para una selección
            if 'value' not in evt and 'index' not in evt:
                # Es un evento de actualización de UI, no de selección del usuario
                print("[DEBUG] on_select_face - Evento de actualización de UI")
                return (
                    gr.update(visible=False), [],
                    get_faces_for_page(ui.globals.ui_input_thumbs, "input"),
                    get_faces_for_page(ui.globals.ui_target_thumbs, "target"),
                    gr.Dropdown(visible=True),
                    gr.update(value="**Cara de destino seleccionada:** Ninguna"),
                    update_pagination_info(ui.globals.ui_input_thumbs, "input"),
                    *update_pagination_buttons(len(ui.globals.ui_input_thumbs), "input"),
                    update_pagination_info(ui.globals.ui_target_thumbs, "target"),
                    *update_pagination_buttons(len(ui.globals.ui_target_thumbs), "target"),
                    gr.update(visible=False),
                )
        
        # Manejar el caso cuando el evento es None o no tiene información válida
        if evt is None:
            print(f"[DEBUG] on_select_face - Evento es None, verificando slider")
            # Si el evento es None, no hacer nada - dejar que el slider maneje la selección
            return (
                gr.update(visible=False), [],
                get_faces_for_page(ui.globals.ui_input_thumbs, "input"),
                get_faces_for_page(ui.globals.ui_target_thumbs, "target"),
                gr.Dropdown(visible=True),
                gr.update(value="**Cara de destino seleccionada:** Ninguna"),
                update_pagination_info(ui.globals.ui_input_thumbs, "input"),
                *update_pagination_buttons(len(ui.globals.ui_input_thumbs), "input"),
                update_pagination_info(ui.globals.ui_target_thumbs, "target"),
                *update_pagination_buttons(len(ui.globals.ui_target_thumbs), "target"),
                gr.update(visible=False),
            )
            
        # Intentar obtener el índice del evento
        face_index = None
        if hasattr(evt, 'index'):
            face_index = evt.index
        elif isinstance(evt, dict) and 'index' in evt:
            face_index = evt['index']
        elif isinstance(evt, int):
            face_index = evt
        
        if face_index is None:
            print(f"[DEBUG] on_select_face - No se pudo obtener índice del evento")
            return (
                gr.update(visible=False), [],
                get_faces_for_page(ui.globals.ui_input_thumbs, "input"),
                get_faces_for_page(ui.globals.ui_target_thumbs, "target"),
                gr.Dropdown(visible=True),
                gr.update(value="**Cara de destino seleccionada:** Ninguna"),
                update_pagination_info(ui.globals.ui_input_thumbs, "input"),
                *update_pagination_buttons(len(ui.globals.ui_input_thumbs), "input"),
                update_pagination_info(ui.globals.ui_target_thumbs, "target"),
                *update_pagination_buttons(len(ui.globals.ui_target_thumbs), "target"),
                gr.update(visible=False),
            )
        
        SELECTED_FACE_INDEX = int(face_index)
        print(f"[DEBUG] on_select_face - Cara seleccionada: índice {SELECTED_FACE_INDEX}, IS_INPUT={IS_INPUT}")
        
        # Llamar a on_selected_face para añadir la cara seleccionada
        # Esto funciona porque IS_INPUT fue configurado por on_use_face_from_selected
        return on_selected_face(SELECTED_FACE_INDEX)
        
    except Exception as e:
        print(f"Error al seleccionar la cara: {e}")
        import traceback
        traceback.print_exc()
    
    # Return por defecto (sin cambios)
    return (
        gr.update(visible=False), [],
        get_faces_for_page(ui.globals.ui_input_thumbs, "input"),
        get_faces_for_page(ui.globals.ui_target_thumbs, "target"),
        gr.Dropdown(visible=True),
        gr.update(value="**Cara de destino seleccionada:** Ninguna"),
        update_pagination_info(ui.globals.ui_input_thumbs, "input"),
        *update_pagination_buttons(len(ui.globals.ui_input_thumbs), "input"),
        update_pagination_info(ui.globals.ui_target_thumbs, "target"),
        *update_pagination_buttons(len(ui.globals.ui_target_thumbs), "target"),
        gr.update(visible=False),
    )


def on_selected_face(face_index, detected_faces=None):
    """
    Maneja la selección de una cara desde el panel de selección dinámica.
    
    Args:
        face_index: int - Índice de la cara seleccionada desde el slider
        detected_faces: list - Lista opcional de miniaturas de las caras detectadas (para mantener en la galería)
    """
    global IS_INPUT, SELECTED_FACE_INDEX, SELECTION_FACES_DATA, _IS_UPDATING_GALLERY, _IS_UPDATING_TARGET, SELECTED_TARGET_FACE_INDEX, CURRENT_DETECTED_FACES
    
    # Activar flag para prevenir eventos de selección automáticos durante actualización de target_faces
    _IS_UPDATING_TARGET = True
    _IS_UPDATING_GALLERY = True  # Mantenemos ambos por compatibilidad
    print(f"[DEBUG] on_selected_face - _IS_UPDATING_TARGET = True")
    
    print(f"[DEBUG] on_selected_face - INICIO")
    print(f"[DEBUG] on_selected_face - face_index recibido: {face_index}")
    print(f"[DEBUG] on_selected_face - IS_INPUT: {IS_INPUT}")
    print(f"[DEBUG] on_selected_face - SELECTION_FACES_DATA disponible: {SELECTION_FACES_DATA is not None}")
    if SELECTION_FACES_DATA:
        print(f"[DEBUG] on_selected_face - Cantidad de caras: {len(SELECTION_FACES_DATA)}")
    
    # PROTEGER contra valores inválidos de face_index
    if face_index is None:
        face_index = 0
    try:
        face_index = int(face_index)
    except (ValueError, TypeError):
        face_index = 0
    
    # Validar que tenemos datos de caras
    if SELECTION_FACES_DATA is None or len(SELECTION_FACES_DATA) == 0:
        print("[ERROR] on_selected_face - No hay SELECTION_FACES_DATA")
        return (
            gr.update(visible=False),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.Dropdown(visible=True),  # selected_face_detection
            gr.update(value="**Cara de destino seleccionada:** Sin datos de caras"),  # selected_target_text
            "Página 1 de 1 (0 caras)",  # target_page_info
            gr.update(interactive=False),  # bt_target_prev
            gr.update(interactive=False),  # bt_target_next
        )
    
    # Usar el índice pasado desde el slider
    SELECTED_FACE_INDEX = face_index
    
    print(f"[DEBUG] on_selected_face - SELECTED_FACE_INDEX usado: {SELECTED_FACE_INDEX}")
    
    # Validar que el índice está en rango
    if SELECTED_FACE_INDEX < 0 or SELECTED_FACE_INDEX >= len(SELECTION_FACES_DATA):
        print(f"[ERROR] on_selected_face - Índice fuera de rango: {SELECTED_FACE_INDEX} vs {len(SELECTION_FACES_DATA)}")
        SELECTED_FACE_INDEX = 0
    
    try:
        fd = SELECTION_FACES_DATA[SELECTED_FACE_INDEX]
        print(f"[DEBUG] Obteniendo cara de SELECTION_FACES_DATA[{SELECTED_FACE_INDEX}] (total: {len(SELECTION_FACES_DATA)})")
    except IndexError as e:
        print(f"[ERROR] on_selected_face - Error al acceder a SELECTION_FACES_DATA: {e}")
        return (
            gr.update(visible=False),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.Dropdown(visible=True),  # selected_face_detection
            gr.update(value="**Cara de destino seleccionada:** Error de índice"),  # selected_target_text
            "Página 1 de 1 (0 caras)",  # target_page_info
            gr.update(interactive=False),  # bt_target_prev
            gr.update(interactive=False),  # bt_target_next
        )
    
    print(f"[DEBUG] on_selected_face - fd tipo: {type(fd)}, longitud: {len(fd) if hasattr(fd, '__len__') else 'N/A'}")
    
     # Convertir imagen
    try:
        image = util.convert_to_gradio(fd[1], is_rgb=True)
        print(f"[DEBUG] on_selected_face - Imagen convertida: {image is not None}")
    except Exception as e:
        print(f"[ERROR] on_selected_face - Error convirtiendo imagen: {e}")
        import traceback
        traceback.print_exc()
        gr.Error(f"Error al convertir la imagen: {str(e)}")
        return (
            gr.update(visible=False),  # dynamic_face_selection
            [],  # face_selection
            ui.globals.ui_target_thumbs if hasattr(ui.globals, 'ui_target_thumbs') else [],  # target_faces
            gr.Dropdown(visible=True),  # selected_face_detection
            gr.update(value="**Cara de destino seleccionada:** Error de imagen"),  # selected_target_text
            "Página 1 de 1 (0 caras)",  # target_page_info
            gr.update(interactive=False),  # bt_target_prev
            gr.update(interactive=False),  # bt_target_next
        )
    
    # Determinar el modo actual
    current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
    
    if IS_INPUT:
        face_set = FaceSet()
        fd[0].mask_offsets = (0, 0.15, 0, 0, 1, 15)  # Más cobertura inferior para boca
        # Guardar imagen de referencia para enhancer
        if fd[1] is not None and hasattr(fd[1], 'shape'):
            fd[0].face_img_ref = fd[1].copy()
        face_set.faces.append(fd[0])
        roop.globals.INPUT_FACESETS.append(face_set)
        ui.globals.ui_input_thumbs.append(image)
        faces_page = get_faces_for_page(ui.globals.ui_input_thumbs, "input")
        input_page_info = update_pagination_info(ui.globals.ui_input_thumbs, "input")
        bt_input_prev, bt_input_next = update_pagination_buttons(
            len(ui.globals.ui_input_thumbs), "input"
        )
        target_page_info = update_pagination_info(ui.globals.ui_target_thumbs, "target")
        bt_target_prev, bt_target_next = update_pagination_buttons(
            len(ui.globals.ui_target_thumbs), "target"
        )
        return (
            gr.update(visible=False),  # Ocultar dynamic_face_selection
            [],  # Limpiar face_selection (lista vacía)
            ui.globals.ui_target_thumbs,  # target_faces - devolver lista completa
            gr.Dropdown(visible=True),  # selected_face_detection
            gr.update(value="**Cara de destino seleccionada:** Ninguna"),  # selected_target_text
            target_page_info,  # target_page_info
            bt_target_prev,  # bt_target_prev
            bt_target_next,  # bt_target_next
        )
    else:
        # Anotar la cara destino con el archivo desde el que fue tomada
        face_obj = fd[0]
        face_img = fd[1] if len(fd) > 1 else None
        
        try:
            face_obj.source_file = roop.globals.target_path
        except Exception:
            pass
        
        # Guardar imagen de referencia para enhancer
        if face_img is not None and hasattr(face_img, 'shape'):
            face_obj.face_img_ref = face_img.copy()
        
        # CRÍTICO: Guardar la cara seleccionada como referencia permanente para este video
        # Esto permite que ProcessMgr use SOLO esta cara para hacer swap
        video_path = roop.globals.target_path
        if video_path:
            video_key = f"selected_face_ref_{os.path.basename(video_path)}"
            
            # Guardar referencia persistente de la cara seleccionada
            if not hasattr(roop.globals, 'selected_face_references'):
                roop.globals.selected_face_references = {}
            
            # Almacenar información clave para matching futuro
            face_ref_data = {
                'bbox': face_obj.bbox,
                'embedding': face_obj.embedding,
                'source_file': video_path,
                'face_obj': face_obj,
                'face_img': face_img
            }
            
            roop.globals.selected_face_references[video_key] = face_ref_data
            
            print(f"[SELECTED_FACES_FRAME] ✓ Cara seleccionada por usuario guardada para {os.path.basename(video_path)}")
            print(f"[SELECTED_FACES_FRAME]   bbox={face_obj.bbox}")
            print(f"[SELECTED_FACES_FRAME]   embedding disponible={face_obj.embedding is not None}")
            print(f"[SELECTED_FACES_FRAME]   Índice seleccionado: {SELECTED_FACE_INDEX} de {len(SELECTION_FACES_DATA)} caras detectadas")
        
        # En modo selected_faces o selected_faces_frame: REEMPLAZAR la cara anterior para este archivo en lugar de agregar
        if current_mode == 'selected_faces' or current_mode == 'selected_faces_frame':
            # Buscar si ya existe una cara para este archivo
            existing_index = None
            if hasattr(roop.globals, 'TARGET_FACES') and roop.globals.TARGET_FACES:
                for i, existing_face in enumerate(roop.globals.TARGET_FACES):
                    if hasattr(existing_face, 'source_file') and existing_face.source_file == video_path:
                        existing_index = i
                        break
            
            if existing_index is not None:
                # Reemplazar la cara existente
                print(f"[DEBUG] on_selected_face - Reemplazando cara en índice {existing_index} para {os.path.basename(video_path)}")
                roop.globals.TARGET_FACES[existing_index] = face_obj
                ui.globals.ui_target_thumbs[existing_index] = image
            else:
                # No existe, agregar nueva
                roop.globals.TARGET_FACES.append(face_obj)
                ui.globals.ui_target_thumbs.append(image)
                # Actualizar el índice de cara objetivo seleccionada
                SELECTED_TARGET_FACE_INDEX = len(roop.globals.TARGET_FACES) - 1
                print(f"[DEBUG] Cara agregada (reemplazo), SELECTED_TARGET_FACE_INDEX={SELECTED_TARGET_FACE_INDEX}")
        else:
            # En otros modos (All faces): agregar normalmente
            roop.globals.TARGET_FACES.append(face_obj)
            ui.globals.ui_target_thumbs.append(image)
            # Actualizar el índice de cara objetivo seleccionada
            SELECTED_TARGET_FACE_INDEX = len(roop.globals.TARGET_FACES) - 1
            print(f"[DEBUG] Cara agregada (otro modo), SELECTED_TARGET_FACE_INDEX={SELECTED_TARGET_FACE_INDEX}")
        
        # Debug: verificar que se añadió
        print(f"[DEBUG] on_selected_face - target_faces ahora tiene: {len(ui.globals.ui_target_thumbs)} caras")
        
        # IMPORTANTE: Resetear flags para permitir selección en target_faces
        _IS_UPDATING_TARGET = False
        _IS_UPDATING_GALLERY = False
        print(f"[DEBUG] on_selected_face - Flags reseteados")
        
        target_page_info = update_pagination_info(ui.globals.ui_target_thumbs, "target")
        bt_target_prev, bt_target_next = update_pagination_buttons(
            len(ui.globals.ui_target_thumbs), "target"
        )
        # CORRECCIÓN: Mantener el modo original seleccionado por el usuario
        if current_mode == 'selected_faces_frame':
            target_mode = "Selected faces frame"
        elif current_mode == 'selected_faces':
            target_mode = "Selected faces"
        elif current_mode == 'selected':
            target_mode = "Selected faces"
        else:
            target_mode = "All faces"
            
        return (
            gr.update(visible=False),  # Ocultar dynamic_face_selection después de seleccionar
            [],  # Limpiar face_selection
            ui.globals.ui_target_thumbs,  # target_faces - devolver lista completa
            target_mode,  # selected_face_detection
            gr.update(
                value=f"**Cara de destino seleccionada:** {len(ui.globals.ui_target_thumbs)} - {os.path.basename(roop.globals.target_path) if roop.globals.target_path else 'N/A'}"
            ),  # selected_target_text
            target_page_info,  # target_page_info
            bt_target_prev,  # bt_target_prev
            bt_target_next,  # bt_target_next
        )


#        bt_faceselect.click(fn=on_selected_face, outputs=[dynamic_face_selection, face_selection, input_faces, target_faces])




def on_preview_frame_changed(
    frame_num,
    files,
    fake_preview,
    enhancer,
    detection,
    face_distance,
    selected_mask_engine,
    no_face_action,
    vr_mode,
    auto_rotate,
    maskimage,
    num_steps,
):
    global SELECTED_INPUT_FACE_INDEX, manual_masking, current_video_fps

    from roop.core import get_processing_plugins, live_swap

    manual_masking = False
    mask_offsets = (0, 0, 0, 0)
    if len(roop.globals.INPUT_FACESETS) > SELECTED_INPUT_FACE_INDEX:
        if not hasattr(
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0],
            "mask_offsets",
        ):
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[
                0
            ].mask_offsets = mask_offsets
        mask_offsets = (
            roop.globals.INPUT_FACESETS[SELECTED_INPUT_FACE_INDEX].faces[0].mask_offsets
        )

    timeinfo = "0:00:00"
    if files is None or selected_preview_index >= len(files) or frame_num is None:
        return None, None, 1, 1, timeinfo

    try:
        filename = files[selected_preview_index].name
        print(f"[DEBUG] Loading preview for: {filename}, frame: {frame_num}")
        if util.is_video(filename) or filename.lower().endswith("gif"):
            from roop.capturer import get_video_frame

            current_frame = get_video_frame(filename, frame_num)
            if current_video_fps == 0:
                current_video_fps = 1
            secs = (frame_num - 1) / current_video_fps
            minutes = secs / 60
            secs = secs % 60
            hours = minutes / 60
            minutes = minutes % 60
            milliseconds = (secs - int(secs)) * 1000
            timeinfo = f"{int(hours):0>2}:{int(minutes):0>2}:{int(secs):0>2}.{int(milliseconds):0>3}"
        else:
            from roop.capturer import get_image_frame

            current_frame = get_image_frame(filename)
            print(f"[DEBUG] Image loaded: {current_frame is not None}")
        if current_frame is None:
            print(f"[DEBUG] Current frame is None, returning")
            return None, None, gr.Slider(info=timeinfo)

        layers = None
        if maskimage is not None and isinstance(maskimage, dict):
            layers = maskimage.get("layers")
            # Validar las capas de la máscara
            if layers is not None and not isinstance(layers, list):
                layers = None
            elif layers is not None and len(layers) > 0 and (layers[0] is None or not hasattr(layers[0], "shape")):
                layers = None

        if not fake_preview or len(roop.globals.INPUT_FACESETS) < 1:
            return (
                gr.Image(value=util.convert_to_gradio(current_frame), visible=True),
                gr.Image(visible=False),
                gr.Slider(info=timeinfo),
            )

        roop.globals.face_swap_mode = translate_swap_mode(detection)
        roop.globals.selected_enhancer = enhancer
        roop.globals.distance_threshold = face_distance
        # blend_ratio siempre es 1.0 (100% máximo) - ya configurado en línea 558
        roop.globals.blend_ratio = 1.0
        roop.globals.no_face_action = index_of_no_face_action(no_face_action)
        roop.globals.vr_mode = vr_mode
        roop.globals.autorotate_faces = auto_rotate

        mask_engine = map_mask_engine(selected_mask_engine)

        roop.globals.execution_threads = roop.globals.CFG.max_threads
        mask = layers[0] if layers is not None else None
        
        # Determinar el índice de cara a usar
        if roop.globals.face_swap_mode == 'selected_faces_frame':
            # En modo selected_faces_frame, usar la selección guardada para este video
            face_index = SELECTED_FACE_INDEX
            if hasattr(roop.globals, 'target_path') and roop.globals.target_path:
                video_key = os.path.basename(roop.globals.target_path)
                if hasattr(roop.globals, 'selected_faces_frame_selections') and video_key in roop.globals.selected_faces_frame_selections:
                    face_index = roop.globals.selected_faces_frame_selections[video_key]
                    print(f"[DEBUG] 🎯 Usando cara seleccionada para {video_key}: {face_index}")
        else:
            # En otros modos, usar la selección de cara de origen
            face_index = SELECTED_INPUT_FACE_INDEX
            if len(roop.globals.INPUT_FACESETS) <= face_index:
                face_index = 0

        options = ProcessOptions(
            get_processing_plugins(mask_engine),
            roop.globals.distance_threshold,
            roop.globals.blend_ratio,
            roop.globals.face_swap_mode,
            face_index,
            maskimage,
            num_steps,
            False,  # show_face_area (default: False)
        )
        
        # Debug: Mostrar información de las opciones
        print(f"[DEBUG] 🎯 Creando ProcessOptions para {roop.globals.face_swap_mode}")
        print(f"[DEBUG] 🎯 face_index usado: {face_index}")
        print(f"[DEBUG] 🎯 options.selected_index: {options.selected_index}")
        
        # Debug: Mostrar estado de las selecciones guardadas
        if hasattr(roop.globals, 'selected_faces_frame_selections'):
            print(f"[DEBUG] 🎯 Selecciones guardadas: {roop.globals.selected_faces_frame_selections}")
        else:
            print(f"[DEBUG] 🎯 No hay selecciones guardadas")
        
        # Debug: Mostrar información del video actual
        if hasattr(roop.globals, 'target_path') and roop.globals.target_path:
            video_key = os.path.basename(roop.globals.target_path)
            print(f"[DEBUG] 🎯 Video actual: {video_key}")
        else:
            print(f"[DEBUG] 🎯 No hay video actual seleccionado")

        current_frame = live_swap(current_frame, options)
        if current_frame is None:
            return (
                gr.Image(visible=True),
                gr.Image(visible=False),
                gr.Slider(info=timeinfo),
            )
        return (
            gr.Image(value=util.convert_to_gradio(current_frame), visible=True),
            gr.Image(visible=False),
            gr.Slider(info=timeinfo),
        )
    except Exception as e:
        print(f"[ERROR] Failed to change preview frame: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None, gr.Slider(info=timeinfo)


def map_mask_engine(selected_mask_engine):
    if selected_mask_engine == "Clip2Seg":
        mask_engine = "mask_clip2seg"
    elif selected_mask_engine == "DFL XSeg":
        mask_engine = "mask_xseg"
    else:
        mask_engine = None
    return mask_engine


def on_toggle_masking(previewimage, mask):
    global manual_masking

    try:
        manual_masking = not manual_masking
        if manual_masking:
            if previewimage is None:
                gr.Warning("No hay imagen de preview para crear una máscara")
                manual_masking = False
                return None, None, True, False
            
            # Asegurarse de que mask tiene la estructura correcta
            if mask is None or not isinstance(mask, dict) or "layers" not in mask:
                from roop.face_util import create_blank_image
                layers = [create_blank_image(previewimage.shape[1], previewimage.shape[0])]
            else:
                layers = mask["layers"]
                # Validar las capas de la máscara
                if not isinstance(layers, list) or len(layers) == 0:
                    from roop.face_util import create_blank_image
                    layers = [create_blank_image(previewimage.shape[1], previewimage.shape[0])]
                elif len(layers) == 1 and (layers[0] is None or not hasattr(layers[0], "shape")):
                    from roop.face_util import create_blank_image
                    layers = [create_blank_image(previewimage.shape[1], previewimage.shape[0])]
            
            return None, {"background": previewimage, "layers": layers, "composite": None}, False, True
        return None, None, True, False
        
    except Exception as e:
        print(f"[ERROR] Error en on_toggle_masking: {e}")
        import traceback
        traceback.print_exc()
        manual_masking = False
        gr.Error(f"Error al togglear la máscara: {str(e)}")
        return None, None, True, False


def on_set_frame(sender: str, frame_num):
    global selected_preview_index, list_files_process

    idx = selected_preview_index
    if list_files_process[idx].endframe == 0:
        return gen_processing_text(0, 0)

    start = list_files_process[idx].startframe
    end = list_files_process[idx].endframe
    if sender.lower().endswith("start"):
        list_files_process[idx].startframe = min(frame_num, end)
    else:
        list_files_process[idx].endframe = max(frame_num, start)

    return gen_processing_text(
        list_files_process[idx].startframe, list_files_process[idx].endframe
    )


def on_preview_mask(frame_num, files, mask_engine):
    from roop.core import get_processing_plugins, live_swap

    global is_processing

    try:
        if (
            is_processing
            or files is None
            or selected_preview_index >= len(files)
            or frame_num is None
        ):
            return None

        filename = files[selected_preview_index].name
        if util.is_video(filename) or filename.lower().endswith("gif"):
            from roop.capturer import get_video_frame
            current_frame = get_video_frame(filename, frame_num)
        else:
            from roop.capturer import get_image_frame
            current_frame = get_image_frame(filename)
            
        if current_frame is None or mask_engine is None or mask_engine == "None":
            return None
            
        if mask_engine == "DFL XSeg":
            mask_engine = "mask_xseg"
        elif mask_engine == "Clip2Seg":
            mask_engine = "mask_clip2seg"
            
        options = ProcessOptions(
            get_processing_plugins(mask_engine),
            roop.globals.distance_threshold,
            roop.globals.blend_ratio,
            "all",
            0,
            None,
            0,
            False,
            True,
        )

        current_frame = live_swap(current_frame, options)
        return util.convert_to_gradio(current_frame)
        
    except Exception as e:
        print(f"[ERROR] Error en on_preview_mask: {e}")
        import traceback
        traceback.print_exc()
        gr.Error(f"Error al previsualizar la máscara: {str(e)}")
        return None


def on_clear_input_faces():
    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    faces_page = get_faces_for_page(ui.globals.ui_input_thumbs, "input")
    prev_btn, next_btn = update_pagination_buttons(
        len(ui.globals.ui_input_thumbs), "input"
    )
    return (
        faces_page,
        update_pagination_info(ui.globals.ui_input_thumbs, "input"),
        prev_btn,
        next_btn,
    )


def on_clear_destfiles():
    global list_files_process, selected_preview_index
    list_files_process.clear()
    selected_preview_index = 0
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs.clear()
    roop.globals.target_path = None
    prev_btn, next_btn = update_pagination_buttons(
        len(ui.globals.ui_target_thumbs), "target"
    )
    # CORRECCIÓN: Preservar el modo actual al limpiar
    current_mode = getattr(roop.globals, 'face_swap_mode', 'selected')
    if current_mode == 'selected_faces_frame':
        clear_target_mode = "Selected faces frame"
    elif current_mode == 'selected_faces':
        clear_target_mode = "Selected faces"
    elif current_mode == 'selected':
        clear_target_mode = "Selected faces"
    else:
        clear_target_mode = "All faces"
    return (
        ui.globals.ui_target_thumbs,
        clear_target_mode,
        "**Cara de destino seleccionada:** Ninguna",
        update_pagination_info(ui.globals.ui_target_thumbs, "target"),
        prev_btn,
        next_btn,
    )


def index_of_no_face_action(dropdown_text):
    global no_face_choices
    # Allow integers (already an index) and None (default to 0)
    if dropdown_text is None:
        return 0
    if isinstance(dropdown_text, int):
        return dropdown_text
    try:
        return no_face_choices.index(dropdown_text)
    except ValueError:
        # Fallback to 0 to avoid crashing
        return 0


def translate_swap_mode(dropdown_text):
    if dropdown_text == "Selected faces":
        return "selected_faces"
    elif dropdown_text == "Selected faces frame":
        return "selected_faces_frame"
    elif dropdown_text == "All faces":
        return "all"
    
    return "selected_faces"  # Default


def _should_force_single_source(is_video_processing, face_swap_mode):
    """
    Determina si debemos forzar 'use_single_source_for_all' por detección de video.
    Para videos en modos automáticos (all, all_female, all_male) recomendamos activarlo.
    """
    return bool(is_video_processing and face_swap_mode in ['all', 'all_female', 'all_male'])


def start_swap(
    enhancer,
    detection,
    keep_frames,
    wait_after_extraction,
    skip_audio,
    use_enhancer,
    face_distance,
    blend_ratio,
    blend_mode,
    selected_mask_engine,
    processing_method,
    no_face_action,
    vr_mode,
    use_single_source_all,
    autorotate,
    temporal_smoothing,
    num_swap_steps,
    imagemask,
    similarity_threshold=None,
    gender_strictness=None,
    color_correction=None,
    selected_top_k=3,
    selected_assignment_ttl=90,
    # Nuevos parámetros de calidad
    enhancer_blend_factor=None,
    color_match_strength=None,
    brightness_strength=None,
):
    from roop.core import batch_process_regular
    from ui.main import prepare_environment

    global is_processing, list_files_process, is_video_processing, is_image_processing

    # DIAGNÓSTICO DETALLADO ANTES DE INICIAR
    print("[DEBUG] [DIAGNÓSTICO] Verificando configuración antes de iniciar...")

    # Aplicar configuración adicional desde los nuevos parámetros
    if similarity_threshold is not None:
        roop.globals.face_similarity_threshold = similarity_threshold
        print(f"[CONFIG] Similarity Threshold: {similarity_threshold}")
    
    if gender_strictness is not None:
        roop.globals.gender_strictness_mode = gender_strictness
        print(f"[CONFIG] Gender Strictness: {gender_strictness}")
    
    if color_correction is not None:
        roop.globals.use_color_correction = color_correction
        print(f"[CONFIG] Color Correction: {color_correction}")
    
    # NUEVO: Aplicar controles de calidad para preservar identidad
    if enhancer_blend_factor is not None:
        roop.globals.enhancer_blend_factor = enhancer_blend_factor
        print(f"[CONFIG] Enhancer Blend Factor: {enhancer_blend_factor}")
    
    if color_match_strength is not None:
        roop.globals.color_match_strength = color_match_strength
        print(f"[CONFIG] Color Match Strength: {color_match_strength}")
    
    if brightness_strength is not None:
        roop.globals.brightness_strength = brightness_strength
        print(f"[CONFIG] Brightness Strength: {brightness_strength}")
    
    # Aplicar temporal smoothing
    roop.globals.temporal_smoothing = temporal_smoothing if temporal_smoothing is not None else True
    
    # Aplicar autorotate
    roop.globals.autorotate_faces = autorotate if autorotate is not None else True

    # Verificar archivos de destino
    if list_files_process is None or len(list_files_process) <= 0:
        error_msg = "[ERROR] ERROR: No hay archivos de destino configurados. Carga archivos en la sección 'Archivos Destino'."
        print(f"[DIAGNÓSTICO] {error_msg}")
        gr.Error(error_msg)
        return gr.update(variant="primary", interactive=True), gr.update(interactive=False), [], get_metrics_html(0, 0, 0, "--:--", "--:--", "Error: sin archivos")

    # Detectar tipo de procesamiento
    global is_video_processing, is_image_processing
    is_video_processing = any(util.is_video(entry.filename) or entry.filename.lower().endswith("gif") for entry in list_files_process)
    is_image_processing = any(util.has_image_extension(entry.filename) for entry in list_files_process)

    # Verificar caras de entrada
    if (
        not hasattr(roop.globals, "INPUT_FACESETS")
        or len(roop.globals.INPUT_FACESETS) <= 0
    ):
        error_msg = "[ERROR] ERROR: No hay caras de origen configuradas. Carga imágenes con caras en 'Archivos Origen'."
        print(f"[DIAGNÓSTICO] {error_msg}")
        gr.Error(error_msg)
        return gr.update(variant="primary", interactive=True), gr.update(interactive=False), []

    # Verificar modo de detección de caras
    face_swap_mode = translate_swap_mode(detection)
    
    # MODIFICACIÓN CLAVE: Modo "Selected Faces" para imágenes JPG
    if face_swap_mode == "selected_faces":
        # En modo "Selected Faces", no se requieren caras destino seleccionadas
        # Se usarán todas las caras de origen y se asignarán de manera aleatoria
        print("[OK] [DIAGNÓSTICO] Modo Selected Faces activado - usando todas las caras de origen")
    elif face_swap_mode == "selected":
        if (
            not hasattr(roop.globals, "TARGET_FACES")
            or len(roop.globals.TARGET_FACES) < 1
        ):
            error_msg = "[ERROR] ERROR: Modo 'Selected face' requiere caras de destino. Selecciona caras del video/imagen de destino."
            print(f"[DIAGNÓSTICO] {error_msg}")
            gr.Error(error_msg)
            return gr.update(variant="primary", interactive=True), gr.update(interactive=False), []

    print("[OK] [DIAGNÓSTICO] Configuración validada correctamente:")
    print(f"   - Archivos destino: {len(list_files_process)}")
    print(f"   - Caras origen: {len(roop.globals.INPUT_FACESETS)}")
    print(f"   - Modo detección: {face_swap_mode}")
    if hasattr(roop.globals, "TARGET_FACES") and face_swap_mode == "selected":
        print(f"   - Caras destino: {len(roop.globals.TARGET_FACES)}")

    # Prepare environment first (sets roop.globals.output_path etc.)
    prepare_environment()

    # If configured to clear output we must ensure output_path is initialized
    if roop.globals.CFG.clear_output and getattr(roop.globals, 'output_path', None):
        try:
            shutil.rmtree(roop.globals.output_path)
        except Exception as e:
            print(f"[WARNING] Unable to clear output_path: {e}")

    # Verificar FFmpeg usando la misma lógica que util_ffmpeg.py
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ffmpeg_paths = [
        os.path.join(project_dir, 'ffmpeg', 'ffmpeg.exe'),
        os.path.join(project_dir, 'ffmpeg', 'ffmpeg-master-latest-win64-gpl', 'bin', 'ffmpeg.exe'),
    ]
    
    ffmpeg_found = False
    for path in ffmpeg_paths:
        if os.path.exists(path):
            ffmpeg_found = True
            print(f"[OK] [FFmpeg] Encontrado en ruta local: {path}")
            break
    
    if not ffmpeg_found and not util.is_installed("ffmpeg"):
        msg = "⚠️ FFmpeg no está instalado. El procesamiento de video puede fallar.\n\nPara instalar FFmpeg:\n1. Descarga desde: https://ffmpeg.org/download.html\n2. Coloca ffmpeg.exe en la carpeta 'ffmpeg' del proyecto\n3. O instálalo en el sistema y agrégalo al PATH"
        print(f"[WARNING] {msg}")
        gr.Warning(msg)
    elif ffmpeg_found:
        print("[OK] [FFmpeg] Verificación exitosa - FFmpeg disponible")

    if getattr(roop.globals, 'log_level', '').lower() == 'debug':
        print('[DEBUG] start_swap - calling prepare_environment()')
    prepare_environment()

    roop.globals.selected_enhancer = enhancer
    roop.globals.target_path = None
    roop.globals.distance_threshold = face_distance
    roop.globals.blend_ratio = blend_ratio
    if getattr(roop.globals, 'log_level', '').lower() == 'debug':
        print(f"[DEBUG] start_swap - argument use_single_source_all: {use_single_source_all}")
    roop.globals.use_single_source_for_all = use_single_source_all
    if getattr(roop.globals, 'log_level', '').lower() == 'debug':
        print(f"[DEBUG] roop.globals.use_single_source_for_all (immediately after assign): {roop.globals.use_single_source_for_all}")

    # Auto-enable single source for 'all' modes if processing video - priorizar coherencia en videos
    if _should_force_single_source(is_video_processing, face_swap_mode):
        if not roop.globals.use_single_source_for_all:
            print("[INFO] Auto-activando 'Usar una sola fuente para All' por detección de video y modo 'All'.")
        roop.globals.use_single_source_for_all = True
    if getattr(roop.globals, 'log_level', '').lower() == 'debug':
        print(f"[DEBUG] roop.globals.use_single_source_for_all (after auto-check): {roop.globals.use_single_source_for_all}")
    roop.globals.use_enhancer = bool(use_enhancer)
    roop.globals.blend_mode = blend_mode
    roop.globals.keep_frames = keep_frames
    roop.globals.wait_after_extraction = wait_after_extraction
    roop.globals.skip_audio = skip_audio
    roop.globals.face_swap_mode = face_swap_mode
    roop.globals.no_face_action = index_of_no_face_action(no_face_action)
    roop.globals.vr_mode = vr_mode
    roop.globals.autorotate_faces = autorotate
    roop.globals.num_swap_steps = num_swap_steps
    # Asignar parámetros avanzados mínimos para modo 'selected' (no demasiados controles)
    roop.globals.selected_top_k = int(selected_top_k) if selected_top_k is not None else getattr(roop.globals, 'selected_top_k', 3)
    roop.globals.selected_assignment_ttl = int(selected_assignment_ttl) if selected_assignment_ttl is not None else getattr(roop.globals, 'selected_assignment_ttl', 90)
    # Umbral minimo de similitud para evitar matches muy lejos del target cuando se puede evitar
    roop.globals.similarity_threshold_selected = getattr(roop.globals, 'similarity_threshold_selected', 0.15)

    # Detectar automáticamente orientación de caras y aplicar corrección
    from roop.face_util import detectar_orientacion_cara, detectar_rotacion_90_grados

    needs_flip = False
    rotation_angle = 0
    needs_rotation_correction = False

    # Verificar todas las caras de origen para determinar orientación
    if hasattr(roop.globals, "INPUT_FACESETS") and roop.globals.INPUT_FACESETS:
        for face_set in roop.globals.INPUT_FACESETS:
            if hasattr(face_set, "faces") and face_set.faces:
                for face in face_set.faces:
                    # Detectar rotación de 90°, 180°, 270°
                    detected_rotation = detectar_rotacion_90_grados(face)
                    if detected_rotation > 0:
                        rotation_angle = detected_rotation
                        needs_rotation_correction = True
                        break

                    # Detectar caras volteadas completamente (180°)
                    if detectar_orientacion_cara(face):
                        needs_flip = True
                        break

                if needs_rotation_correction or needs_flip:
                    break

    # Configurar variables globales para corrección
    roop.globals.face_rotation_correction = needs_rotation_correction
    roop.globals.face_rotation_angle = rotation_angle
    roop.globals.flip_faces = needs_flip

    # Ajustar parámetros de blending cuando se detecta corrección de orientación
    if needs_flip or rotation_angle != 0:
        # Usar blending más conservador para evitar artefactos en correcciones
        try:
            current_blend = float(roop.globals.blend_ratio)
        except (ValueError, TypeError):
            current_blend = 1.0
        if current_blend > 0.9:
            roop.globals.blend_ratio = 0.9
        # Reducir pasos para mejorar estabilidad cuando hay correcciones
        current_steps = roop.globals.num_swap_steps if hasattr(roop.globals, "num_swap_steps") and roop.globals.num_swap_steps is not None else 10
        if current_steps > 2:
            roop.globals.num_swap_steps = 2

    # Información de debug para el usuario
    if needs_flip:
        print("🔄 [DETECCIÓN] Detectada cara volteada - aplicando corrección de volteo")
    elif rotation_angle != 0:
        print(
            f"🔄 [DETECCIÓN] Detectada rotación de {rotation_angle}° - aplicando corrección de rotación"
        )
    else:
        print("[OK] [DETECCIÓN] Caras en orientación normal - sin corrección necesaria")
    mask_engine = map_mask_engine(selected_mask_engine)

    is_processing = True
    print("[START] [DIAGNÓSTICO] Iniciando procesamiento...")
    start_t = time.time()
    total_files = len(list_files_process)
    yield (
        gr.update(variant="secondary", interactive=False),
        gr.update(variant="primary", interactive=True),
        [],
        get_metrics_html(0, 0, total_files, "0:00", "--:--", "Iniciando..."),
    )

    roop.globals.execution_threads = roop.globals.CFG.max_threads
    roop.globals.video_encoder = roop.globals.CFG.output_video_codec
    roop.globals.video_quality = roop.globals.CFG.video_quality
    roop.globals.max_memory = (
        roop.globals.CFG.memory_limit if roop.globals.CFG.memory_limit > 0 else None
    )

    # Asegurarse de que imagemask tenga una estructura válida
    mask_data = {"layers": []} if imagemask is None else imagemask
    if isinstance(mask_data, dict) and not mask_data.get("layers"):
        mask_data["layers"] = []

    try:
        print(
            f"[DEBUG] Iniciando batch_process_regular con {len(list_files_process)} archivos"
        )
        print(
            f"[DEBUG] INPUT_FACESETS: {len(roop.globals.INPUT_FACESETS) if hasattr(roop.globals, 'INPUT_FACESETS') else 0} facesets"
        )
        print(
            f"[DEBUG] TARGET_FACES: {len(roop.globals.TARGET_FACES) if hasattr(roop.globals, 'TARGET_FACES') else 0} caras"
        )
        print(f"[DEBUG] SELECTED_INPUT_FACE_INDEX: {SELECTED_INPUT_FACE_INDEX}")

        # Llamar a batch_process_regular como generadora y recibir yields
        print("[DEBUG] Iniciando batch_process_regular como generadora...")
        
        # batch_process_regular ahora es una función generadora que hace yield de (porcentaje, mensaje)
        for progress_percent, progress_message in batch_process_regular(
            list_files_process,
            mask_engine,
            "",  # clip_text (removed)
            processing_method == "In-Memory processing",
            mask_data,
            roop.globals.num_swap_steps,
            None,  # progress - no usamos gr.Progress()
            SELECTED_INPUT_FACE_INDEX,
            temporal_smoothing,
        ):
            # Calcular métricas en tiempo real
            try:
                elapsed = time.time() - start_t
                elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                files_done = int(progress_percent / 100.0 * total_files) if total_files > 0 else 0
                time_remaining = "--:--"
                if progress_percent > 0 and files_done > 0:
                    rate = files_done / elapsed
                    remaining = (total_files - files_done) / rate if rate > 0 else 0
                    time_remaining = time.strftime("%H:%M:%S", time.gmtime(remaining))

                yield (
                    gr.update(variant="secondary", interactive=False),
                    gr.update(variant="primary", interactive=True),
                    [],
                    get_metrics_html(progress_percent, files_done, total_files, elapsed_str, time_remaining, "Procesando..."),
                )
                print(f"[UI] Progreso: {progress_percent:.1f}% - {progress_message}")
            except Exception as e:
                print(f"[WARNING] Error actualizando métricas: {e}")
        
        # Resumen final para el usuario
        summary = f"✅ Procesamiento completado.\n📂 Archivos gestionados: {len(list_files_process)}"
        gr.Info(summary)
        print(f"[OK] {summary}")

        cleanup_temp_files()
    except Exception as e:
        error_msg = f"[ERROR] ERROR durante el procesamiento: {str(e)}"
        print(f"[DIAGNÓSTICO] {error_msg}")
        import traceback

        traceback.print_exc()
        gr.Error(error_msg)
        cleanup_temp_files()
        elapsed = time.time() - start_t
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
        yield (
            gr.update(variant="primary", interactive=True),
            gr.update(variant="secondary", interactive=False),
            [],
            get_metrics_html(0, 0, total_files, elapsed_str, "--:--", "Error"),
        )
        return

    is_processing = False
    elapsed = time.time() - start_t
    elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed))
    outdir = pathlib.Path(roop.globals.output_path)
    outfiles = [str(item) for item in outdir.rglob("*") if item.is_file()]

    if len(outfiles) > 0:
        print(f"[OK] [DIAGNÓSTICO] Archivos generados: {len(outfiles)}")
        # Devolver rutas de archivo como strings, que es lo que espera Gradio
        yield (
            gr.update(variant="primary", interactive=True),
            gr.update(variant="secondary", interactive=False),
            outfiles,
            get_metrics_html(100, total_files, total_files, elapsed_str, "--:--", "Completado"),
        )
    else:
        print("⚠️ [DIAGNÓSTICO] No se generaron archivos de salida")
        yield (
            gr.update(variant="primary", interactive=True),
            gr.update(variant="secondary", interactive=False),
            [],
            get_metrics_html(0, 0, total_files, elapsed_str, "--:--", "Sin resultados"),
        )


def stop_swap():
    roop.globals.processing = False
    gr.Info(
        "Abortando el procesamiento: espere a que se detengan los subprocesos restantes"
    )
    return (
        gr.Button(variant="primary", interactive=True),
        gr.Button(variant="secondary", interactive=False),
        None,
    )


def on_fps_changed(fps):
    global selected_preview_index, list_files_process

    if (
        len(list_files_process) < 1
        or list_files_process[selected_preview_index].endframe < 1
    ):
        return
    list_files_process[selected_preview_index].fps = fps


def on_destfiles_changed(destfiles):
    print(f"[DEBUG] [UI] Evento UI: on_destfiles_changed disparado con {len(destfiles) if destfiles else 0} archivos")
    
    # Delegar en la función asíncrona robusta que tiene mejor manejo de errores y logging
    return on_destfiles_changed_async(destfiles)




def on_prev_file(current_frame_num):
    """Navega al archivo de destino anterior"""
    global selected_preview_index, list_files_process, current_video_fps

    if not list_files_process or len(list_files_process) <= 1:
        return current_frame_num, None, "No hay más archivos", gr.Button(interactive=False), gr.Button(interactive=False)

    # Cambiar al archivo anterior
    selected_preview_index = max(0, selected_preview_index - 1)
    
    # Actualizar estado de botones
    prev_enabled = selected_preview_index > 0
    next_enabled = selected_preview_index < len(list_files_process) - 1

    # Cargar el nuevo archivo
    entry = list_files_process[selected_preview_index]
    filename = entry.filename

    if util.is_video(filename) or filename.lower().endswith(".gif"):
        from roop.capturer import get_video_frame, get_video_frame_total
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename) or 30
        if entry.endframe == 0:
            entry.endframe = total_frames
        frame = get_video_frame(filename, 1)
        new_frame_num = 1
        text_info = f"Archivo {selected_preview_index + 1}/{len(list_files_process)}: {os.path.basename(filename)} (Video: {total_frames} frames)"
    else:
        from roop.capturer import get_image_frame
        frame = get_image_frame(filename)
        total_frames = 1
        new_frame_num = 1
        text_info = f"Archivo {selected_preview_index + 1}/{len(list_files_process)}: {os.path.basename(filename)} (Imagen)"

    if frame is not None:
        return (
            gr.Slider(value=new_frame_num, maximum=max(1, total_frames), info="0:00:00"),
            util.convert_to_gradio(frame),
            text_info,
            gr.Button(interactive=prev_enabled),
            gr.Button(interactive=next_enabled)
        )

    return current_frame_num, None, f"Error cargando archivo", gr.Button(interactive=prev_enabled), gr.Button(interactive=next_enabled)


def on_next_file(current_frame_num):
    """Navega al archivo de destino siguiente"""
    global selected_preview_index, list_files_process, current_video_fps

    if not list_files_process or len(list_files_process) <= 1:
        return current_frame_num, None, "No hay más archivos", gr.Button(interactive=False), gr.Button(interactive=False)

    # Cambiar al archivo siguiente
    selected_preview_index = min(len(list_files_process) - 1, selected_preview_index + 1)
    
    # Actualizar estado de botones
    prev_enabled = selected_preview_index > 0
    next_enabled = selected_preview_index < len(list_files_process) - 1

    # Cargar el nuevo archivo
    entry = list_files_process[selected_preview_index]
    filename = entry.filename

    if util.is_video(filename) or filename.lower().endswith(".gif"):
        from roop.capturer import get_video_frame, get_video_frame_total
        total_frames = get_video_frame_total(filename)
        current_video_fps = util.detect_fps(filename) or 30
        if entry.endframe == 0:
            entry.endframe = total_frames
        frame = get_video_frame(filename, 1)
        new_frame_num = 1
        text_info = f"Archivo {selected_preview_index + 1}/{len(list_files_process)}: {os.path.basename(filename)} (Video: {total_frames} frames)"
    else:
        from roop.capturer import get_image_frame
        frame = get_image_frame(filename)
        total_frames = 1
        new_frame_num = 1
        text_info = f"Archivo {selected_preview_index + 1}/{len(list_files_process)}: {os.path.basename(filename)} (Imagen)"

    if frame is not None:
        return (
            gr.Slider(value=new_frame_num, maximum=max(1, total_frames), info="0:00:00"),
            util.convert_to_gradio(frame),
            text_info,
            gr.Button(interactive=prev_enabled),
            gr.Button(interactive=next_enabled)
        )

    return current_frame_num, None, f"Error cargando archivo", gr.Button(interactive=prev_enabled), gr.Button(interactive=next_enabled)


def on_destfiles_selected_js(file_index, files):
    """Maneja la selección de archivos de destino usando índice pasado desde JavaScript"""
    global selected_preview_index, list_files_process, current_video_fps

    print(f"[DEBUG] 📂 on_destfiles_selected_js llamado con file_index: {file_index}")
    print(f"[DEBUG] 📂 list_files_process: {len(list_files_process) if list_files_process else 0} archivos")
    print(f"[DEBUG] 📂 files: {len(files) if files else 0} archivos")
    
    # Imprimir lista de archivos en list_files_process y bt_destfiles para comparar
    if list_files_process:
        print(f"[DEBUG] list_files_process:")
        for i, entry in enumerate(list_files_process):
            print(f"  [{i}] {os.path.basename(entry.filename)}")
    if files:
        print(f"[DEBUG] bt_destfiles:")
        for i, file in enumerate(files):
            print(f"  [{i}] {os.path.basename(file.name)}")
    
    try:
        # Convertir a entero y validar
        try:
            idx = int(file_index) if file_index is not None else 0
        except (ValueError, TypeError):
            idx = 0
        
        # Asegurar que el índice esté en rango
        if not list_files_process or len(list_files_process) == 0:
            print(f"[ERROR] No hay archivos en list_files_process")
            return gr.Slider(value=1, maximum=1, info="0:00:00"), "", None, 0
        
        idx = max(0, min(idx, len(list_files_process) - 1))
        selected_preview_index = idx
        
        print(f"[DEBUG] 📂 Archivo seleccionado: índice {idx} de {len(list_files_process)}")
        
        filename = list_files_process[idx].filename
        print(f"[DEBUG] [IMG] Cargando preview de: {os.path.basename(filename)}")
        fps = list_files_process[idx].fps
        
        if util.is_video(filename) or filename.lower().endswith(".gif"):
            from roop.capturer import get_video_frame, get_video_frame_total

            total_frames = get_video_frame_total(filename)
            current_video_fps = util.detect_fps(filename)

            if current_video_fps <= 0:
                current_video_fps = 30
            if total_frames <= 0:
                total_frames = 1

            current_frame = get_video_frame(filename, 1)
            if list_files_process[idx].endframe == 0:
                list_files_process[idx].endframe = total_frames
        else:
            from roop.capturer import get_image_frame
            total_frames = 1
            current_frame = get_image_frame(filename)

        if current_frame is not None:
            preview_img = util.convert_to_gradio(current_frame)
            print(f"[DEBUG] [IMG] Preview generada correctamente: {preview_img is not None}")
        else:
            preview_img = None
            print(f"[DEBUG] [IMG] Error al generar preview")

        if total_frames > 1:
            start_frame = list_files_process[idx].startframe
            if start_frame < 1:
                start_frame = 1
            return (
                gr.Slider(
                    value=start_frame,
                    minimum=1,
                    maximum=total_frames,
                    info="0:00:00",
                    interactive=True
                ),
                gen_processing_text(start_frame, list_files_process[idx].endframe),
                preview_img,
                idx,  # Devolver el índice para mantener sincronizado el estado oculto
            )
        
        return (
            gr.Slider(value=1, minimum=0, maximum=1, info="Imagen estática", interactive=False),
            gen_processing_text(0, 0),
            preview_img,
            idx,  # Devolver el índice
        )
    except Exception as e:
        print(f"[DEBUG] Error en on_destfiles_selected_js: {e}")
        import traceback
        traceback.print_exc()
        return gr.Slider(value=1, maximum=1, info="0:00:00"), "", None, 0


def on_destfiles_selected(files, evt):
    """Función original mantenida por compatibilidad - usa on_destfiles_selected_js internamente"""
    # Extraer índice del evento si está disponible
    file_index = 0
    if evt is not None:
        if hasattr(evt, 'index'):
            file_index = evt.index
        elif isinstance(evt, dict) and 'index' in evt:
            file_index = evt['index']
        elif isinstance(evt, int):
            file_index = evt
    
    # Llamar a la función con índice
    result = on_destfiles_selected_js(file_index, files)
    # Devolver solo los primeros 3 elementos para compatibilidad con la firma original
    return result[0], result[1], result[2]


def on_resultfiles_selected(evt=None, files=None):
    try:
        if evt is not None and files is not None:
            if hasattr(evt, 'index'):
                selected_index = evt.index
            elif isinstance(evt, dict) and 'index' in evt:
                selected_index = evt['index']
            elif isinstance(evt, int):
                selected_index = evt
            if 0 <= selected_index < len(files):
                # Extraer filename de forma segura (maneja objetos con .name y diccionarios)
                first_file = files[selected_index]
                if hasattr(first_file, 'name'):
                    filename = first_file.name
                elif isinstance(first_file, dict) and 'name' in first_file:
                    filename = first_file['name']
                else:
                    print(f"[ERROR] Formato de archivo no reconocido: {type(first_file)}")
                    return None, None, None
                return display_output(filename)
    except Exception as e:
        print(f"[DEBUG] Error en on_resultfiles_selected: {e}")
    return None, None, None


def on_resultfiles_finished(files):
    selected_index = 0
    if files is None or len(files) < 1:
        # Devolver valores explícitos para evitar errores de Gradio
        return None, None, None
    
    # Extraer el nombre del archivo de forma segura (maneja objetos con .name y diccionarios)
    first_file = files[selected_index]
    if hasattr(first_file, 'name'):
        filename = first_file.name
    elif isinstance(first_file, dict) and 'name' in first_file:
        filename = first_file['name']
    else:
        print(f"[ERROR] Formato de archivo no reconocido: {type(first_file)}")
        return None, None, None
    
    # Determinar si es video o imagen
    if util.is_video(filename) and roop.globals.CFG.output_show_video:
        image_result = None
        video_result = filename
    else:
        if util.is_video(filename) or filename.lower().endswith(".gif"):
            from roop.capturer import get_video_frame
            current_frame = get_video_frame(filename)
        else:
            from roop.capturer import get_image_frame
            current_frame = get_image_frame(filename)
        
        # Convertir a formato Gradio
        image_result = util.convert_to_gradio(current_frame)
        video_result = None
    
    return files, image_result, video_result


def display_output(filename):
    if util.is_video(filename) and roop.globals.CFG.output_show_video:
        return gr.Image(visible=False, value=None), gr.Video(visible=True, value=filename)
    else:
        if util.is_video(filename) or filename.lower().endswith(".gif"):
            from roop.capturer import get_video_frame

            current_frame = get_video_frame(filename)
        else:
            from roop.capturer import get_image_frame

            current_frame = get_image_frame(filename)
        return util.convert_to_gradio(current_frame), None, True, False, None


# Funciones de paginación
def update_pagination_info(thumbs, gallery_type):
    """Actualiza la información de paginación"""
    global current_input_page, current_target_page, FACES_PER_PAGE

    if gallery_type == "input":
        current_page = current_input_page
    else:
        current_page = current_target_page

    total_faces = len(thumbs)
    total_pages = max(1, (total_faces + FACES_PER_PAGE - 1) // FACES_PER_PAGE)

    if current_page >= total_pages:
        if gallery_type == "input":
            current_input_page = max(0, total_pages - 1)
            current_page = current_input_page
        else:
            current_target_page = max(0, total_pages - 1)
            current_page = current_target_page

    start_face = current_page * FACES_PER_PAGE + 1
    end_face = min((current_page + 1) * FACES_PER_PAGE, total_faces)

    if total_faces == 0:
        return "Página 1 de 1 (0 caras)"
    else:
        return f"Página {current_page + 1} de {total_pages} ({start_face}-{end_face} de {total_faces} caras)"


def update_pagination_buttons(total_faces, gallery_type):
    """Actualiza el estado de los botones de paginación"""
    global current_input_page, current_target_page, FACES_PER_PAGE

    if gallery_type == "input":
        current_page = current_input_page
    else:
        current_page = current_target_page

    total_pages = max(1, (total_faces + FACES_PER_PAGE - 1) // FACES_PER_PAGE)

    show_buttons = total_faces > FACES_PER_PAGE
    prev_enabled = current_page > 0
    next_enabled = current_page < total_pages - 1

    # Devolver objetos gr.update para evitar errores al asignarlos directamente a components
    return gr.update(interactive=prev_enabled, visible=show_buttons), gr.update(interactive=next_enabled, visible=show_buttons)


def get_faces_for_page(thumbs, gallery_type):
    """Obtiene las caras para la página actual"""
    global current_input_page, current_target_page, FACES_PER_PAGE

    if gallery_type == "input":
        current_page = current_input_page
    else:
        current_page = current_target_page

    start_idx = current_page * FACES_PER_PAGE
    end_idx = start_idx + FACES_PER_PAGE

    return thumbs[start_idx:end_idx]


def on_input_page_change(direction):
    """Cambia la página de caras de origen"""
    global current_input_page

    try:
        if not hasattr(ui.globals, "ui_input_thumbs") or not ui.globals.ui_input_thumbs:
            prev_btn, next_btn = update_pagination_buttons(0, "input")
            return (
                [],
                "Página 1 de 1 (0 caras)",
                prev_btn,
                next_btn,
            )

        total_faces = len(ui.globals.ui_input_thumbs)
        total_pages = max(1, (total_faces + FACES_PER_PAGE - 1) // FACES_PER_PAGE)

        # Actualizar la página actual
        if direction == "prev":
            current_input_page = max(0, current_input_page - 1)
        elif direction == "next":
            current_input_page = min(total_pages - 1, current_input_page + 1)

        # Obtener solo las caras de la página actual
        start_idx = current_input_page * FACES_PER_PAGE
        end_idx = min(start_idx + FACES_PER_PAGE, total_faces)
        faces_page = ui.globals.ui_input_thumbs[start_idx:end_idx]

        # Actualizar la información de paginación
        page_info = update_pagination_info(ui.globals.ui_input_thumbs, "input")
        prev_btn, next_btn = update_pagination_buttons(total_faces, "input")

        return faces_page, page_info, prev_btn, next_btn

    except Exception as e:
        print(f"[ERROR] Error en paginación: {str(e)}")
        import traceback

        traceback.print_exc()
        return [], "Error en paginación", gr.update(interactive=False), gr.update(interactive=False)


def on_mediapipe_changed(use_mediapipe):
    """Cambia el detector de caras entre InsightFace y MediaPipe"""
    roop.globals.use_mediapipe_detector = use_mediapipe
    if use_mediapipe:
        print("[DEBUG] [DETECTOR] Cambiado a MediaPipe (alternativo)")
        gr.Info("[OK] Detector cambiado a MediaPipe. Más estable para videos difíciles.")
    else:
        print("[DEBUG] [DETECTOR] Cambiado a InsightFace (principal)")
        gr.Info("[OK] Detector cambiado a InsightFace (por defecto).")


def on_target_page_change(direction):
    """Cambia la página de caras de destino"""
    global current_target_page

    total_pages = max(
        1, (len(ui.globals.ui_target_thumbs) + FACES_PER_PAGE - 1) // FACES_PER_PAGE
    )

    if direction == "prev" and current_target_page > 0:
        current_target_page -= 1
    elif direction == "next" and current_target_page < total_pages - 1:
        current_target_page += 1

    faces_page = get_faces_for_page(ui.globals.ui_target_thumbs, "target")
    page_info = update_pagination_info(ui.globals.ui_target_thumbs, "target")
    prev_btn, next_btn = update_pagination_buttons(len(ui.globals.ui_target_thumbs), "target")
    return faces_page, page_info, prev_btn, next_btn


def on_prev_frame(current_frame_num):
    """Navega al frame anterior"""
    global selected_preview_index, list_files_process, current_video_fps

    if not list_files_process or selected_preview_index >= len(list_files_process):
        return current_frame_num, None, "No hay video cargado"

    entry = list_files_process[selected_preview_index]
    filename = entry.filename

    if not (util.is_video(filename) or filename.lower().endswith(".gif")):
        return current_frame_num, None, "Solo disponible para videos"

    new_frame = max(1, current_frame_num - 1)

    # Cargar el frame
    from roop.capturer import get_video_frame
    frame = get_video_frame(filename, new_frame)

    if frame is not None:
        # Calcular tiempo
        if current_video_fps <= 0:
            current_video_fps = util.detect_fps(filename) or 30
        secs = (new_frame - 1) / current_video_fps
        minutes = int(secs / 60)
        secs = secs % 60
        hours = int(minutes / 60)
        minutes = minutes % 60
        timeinfo = f"{hours:0>2}:{minutes:0>2}:{secs:0>2}"

        return new_frame, util.convert_to_gradio(frame), f"Frame {new_frame} - {timeinfo}"

    return current_frame_num, None, f"Error cargando frame {new_frame}"


def on_next_frame(current_frame_num):
    """Navega al frame siguiente"""
    global selected_preview_index, list_files_process, current_video_fps

    if not list_files_process or selected_preview_index >= len(list_files_process):
        return current_frame_num, None, "No hay video cargado"

    entry = list_files_process[selected_preview_index]
    filename = entry.filename
    max_frames = entry.endframe or 1

    if not (util.is_video(filename) or filename.lower().endswith(".gif")):
        return current_frame_num, None, "Solo disponible para videos"

    new_frame = min(max_frames, current_frame_num + 1)

    # Cargar el frame
    from roop.capturer import get_video_frame
    frame = get_video_frame(filename, new_frame)

    if frame is not None:
        # Calcular tiempo
        if current_video_fps <= 0:
            current_video_fps = util.detect_fps(filename) or 30
        secs = (new_frame - 1) / current_video_fps
        minutes = int(secs / 60)
        secs = secs % 60
        hours = int(minutes / 60)
        minutes = minutes % 60
        timeinfo = f"{hours:0>2}:{minutes:0>2}:{secs:0>2}"

        return new_frame, util.convert_to_gradio(frame), f"Frame {new_frame} - {timeinfo}"

    return current_frame_num, None, f"Error cargando frame {new_frame}"
