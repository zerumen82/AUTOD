import sys
import os
import socket
import threading
import time
import requests
import pathlib
import cv2
import numpy as np


def suggest_execution_providers():
    """
    Sugiere los proveedores de ejecución disponibles (CUDA, CPU, etc.)
    """
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        # Priorizar CUDA sobre CPU
        if "CUDAExecutionProvider" in providers:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]
    except ImportError:
        return ["CPUExecutionProvider"]


def get_processing_plugins(mask_engine=None):
    """
    Devuelve un diccionario con los plugins de procesamiento configurados.
    """
    plugins = {}
    if mask_engine:
        plugins["masking"] = {"subtype": mask_engine}
    return plugins


def live_swap(frame, options):
    """
    Procesa un frame con face swap en tiempo real.
    Usado para preview y cámara virtual.
    """
    import roop.globals
    from roop.ProcessMgr import ProcessMgr
    
    if frame is None:
        return None
    
    # Crear ProcessMgr si no existe uno global
    if not hasattr(roop.globals, '_live_process_mgr') or roop.globals._live_process_mgr is None:
        roop.globals._live_process_mgr = ProcessMgr()
        
        # Inicializar con las caras de origen
        if hasattr(roop.globals, 'INPUT_FACESETS') and roop.globals.INPUT_FACESETS:
            target_faces = getattr(roop.globals, 'TARGET_FACES', [])
            roop.globals._live_process_mgr.initialize(
                roop.globals.INPUT_FACESETS,
                target_faces,
                options
            )
    
    # Procesar el frame
    try:
        processed_frame = roop.globals._live_process_mgr.process_frame(frame)
        return processed_frame if processed_frame is not None else frame
    except Exception as e:
        print(f"[ERROR] live_swap: {e}")
        return frame


def cleanup_comfyui():
    """Limpia procesos de ComfyUI al salir"""
    try:
        from ui.tabs.comfy_launcher import stop
        print("[CORE] Limpiando procesos ComfyUI...")
        success, msg = stop()
        print(f"[CORE] {msg}")
    except Exception as e:
        print(f"[CORE] Error en cleanup: {e}")


def find_available_port(start_port=7860, max_attempts=10):
    """Busca un puerto disponible"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError(f"No hay puerto libre")


def is_gradio_running(url, timeout=5):
    """Verifica si Gradio esta ejecutando"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def update_status(message: str):
    """
    Actualiza el estado del sistema.
    Por ahora solo imprime en consola, pero puede extenderse para
    actualizar la UI o enviar eventos.
    """
    print(f"[STATUS] {message}")
    
    # Actualizar variable global si existe
    try:
        import roop.globals
        if hasattr(roop.globals, 'status_callback'):
            roop.globals.status_callback(message)
    except Exception:
        pass


def run():
    """
    Funcion principal con pywebview integrado
    Espera a que Gradio este completamente ejecutando antes de abrir pywebview
    """
    try:
        print("=" * 50)
        print("AUTO-AUTO - ARRANQUE")
        print("=" * 50)
        
        from ui import main as ui_main
        
        # Configurar manejo de excepciones asyncio
        import asyncio
        import logging
        
        print("[INIT] Configurando asyncio...")
        
        # Suppress WinError 10054 in asyncio ProactorEventLoop (Windows)
        if sys.platform == 'win32':
             try:
                 from asyncio.proactor_events import _ProactorBasePipeTransport
                 
                 _original_connection_lost = _ProactorBasePipeTransport._call_connection_lost
                 
                 def _patched_connection_lost(self, exc):
                     try:
                         _original_connection_lost(self, exc)
                     except ConnectionResetError:
                         pass
                     except OSError as e:
                         if e.winerror == 10054:
                             pass
                         else:
                             raise
                 
                 _ProactorBasePipeTransport._call_connection_lost = _patched_connection_lost
                 print("[OK] Patch asyncio aplicado")
             except (ImportError, AttributeError) as e:
                 print(f"[WARNING] No se pudo patch asyncio: {e}")
        
        # Puerto dinamico
        print("[INIT] Buscando puerto...")
        available_port = find_available_port(7860)
        print(f"[OK] Puerto: {available_port}")
        
        server_url = f"http://127.0.0.1:{available_port}"
        
        # Funcion para iniciar servidor
        def start_server():
            os.environ['GRADIO_SERVER_PORT'] = str(available_port)
            ui_main.run()
        
        # Iniciar servidor en hilo separado
        print("[SERVER] Iniciando servidor Gradio...")
        server_thread = threading.Thread(target=start_server, daemon=True, name="GradioServer")
        server_thread.start()
        
        # Esperar y verificar que Gradio esta ejecutando REALMENTE
        print("[SERVER] Verificando que Gradio este ejecutando...")
        max_wait = 180  # Maximo 180 segundos (3 minutos) - carga de IA es lenta
        checked = False
        for i in range(max_wait):
            time.sleep(1)
            if is_gradio_running(server_url):
                print(f"[OK] Gradio ejecutando en {server_url}")
                checked = True
                break
            # Mostrar progreso cada 10 segundos
            if i > 0 and i % 10 == 0:
                print(f"[SERVER] Esperando... {i}s/{max_wait}s")
        
        if not checked:
            print(f"[ERROR] Gradio no respondio despues de {max_wait}s")
            print(f"[INFO] Abre manualmente: {server_url}")
            return
        
        # AHORA si abrir pywebview
        print("[UI] Abriendo pywebview...")
        import webview
        window = webview.create_window(
            "AutoAuto",
            url=server_url,
            width=1400,
            height=900
        )
        webview.start(debug=False, func=None)
        
        print("[EXIT] Ventana cerrada")
        cleanup_comfyui()
        
    except ImportError as e:
        print(f"[ERROR] Pywebview no instalado: {e}")
        print(f"[INFO] Abre manualmente: {server_url}")
    except Exception as e:
        print(f"[FATAL] Error: {e}")
        import traceback
        traceback.print_exc()


def batch_process_regular(
    list_files_process,
    mask_engine,
    clip_text,
    in_memory_processing,
    mask_data,
    num_swap_steps,
    progress_callback,
    selected_input_face_index,
    temporal_smoothing,
):
    """
    Procesa una lista de archivos para face swap.
    Función generadora que hace yield de (progress_percent, progress_message).
    """
    import roop.globals
    import roop.utilities as util
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions
    
    total_files = len(list_files_process)
    if total_files == 0:
        yield (100, "No hay archivos para procesar")
        return
    
    # Asegurar que processing está activo
    roop.globals.processing = True
    
    # Inicializar ProcessMgr
    process_mgr = ProcessMgr()
    
    # Configurar opciones de procesamiento
    processor_options = {}
    if mask_engine and mask_engine != "None":
        processor_options["masking"] = {"subtype": mask_engine}
    
    # ProcessOptions(processordefines, face_distance, blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, show_face_area, ...)
    options = ProcessOptions(
        processor_options,                                    # processordefines
        getattr(roop.globals, 'distance_threshold', 0.6),    # face_distance
        getattr(roop.globals, 'blend_ratio', 0.95),          # blend_ratio
        roop.globals.face_swap_mode if hasattr(roop.globals, 'face_swap_mode') else 'all',  # swap_mode
        selected_input_face_index if selected_input_face_index else 0,  # selected_index
        clip_text if clip_text else "",                       # masking_text
        mask_data,                                            # imagemask
        num_swap_steps if num_swap_steps else 1,              # num_steps
        False,                                                # show_face_area
        False,                                                # show_mask
        getattr(roop.globals, 'use_enhancer', True),          # use_enhancer
        getattr(roop.globals, 'blend_mode', 'seamless')       # blend_mode
    )
    
    # Inicializar ProcessMgr con las caras de origen
    if hasattr(roop.globals, 'INPUT_FACESETS') and roop.globals.INPUT_FACESETS:
        target_faces = getattr(roop.globals, 'TARGET_FACES', [])
        process_mgr.initialize(
            roop.globals.INPUT_FACESETS,
            target_faces,
            options
        )
    else:
        yield (100, "ERROR: No hay caras de origen configuradas")
        return
    
    processed_files = 0
    
    for entry in list_files_process:
        if not roop.globals.processing:
            yield (100, "Procesamiento cancelado")
            return
        
        filename = entry.filename
        start_frame = entry.startframe if hasattr(entry, 'startframe') else 0
        end_frame = entry.endframe if hasattr(entry, 'endframe') else 0
        fps = entry.fps if hasattr(entry, 'fps') else 30.0
        
        # Determinar si es video o imagen
        is_video = util.is_video(filename) or filename.lower().endswith('.gif')
        
        # Construir ruta de salida
        output_filename = os.path.basename(filename)
        output_path = os.path.join(roop.globals.output_path, output_filename)
        
        # Asegurar que el directorio de salida existe
        os.makedirs(roop.globals.output_path, exist_ok=True)
        
        progress_percent = (processed_files / total_files) * 100
        yield (progress_percent, f"Procesando: {output_filename}")
        
        try:
            if is_video:
                # Procesar video usando run_batch_inmem
                for video_progress, video_msg in process_mgr.run_batch_inmem(
                    filename,
                    output_path,
                    start_frame=start_frame,
                    end_frame=end_frame if end_frame > 0 else None,
                    fps=fps,
                    skip_audio=getattr(roop.globals, 'skip_audio', False)
                ):
                    # Calcular progreso total combinando archivo y frame
                    combined_progress = progress_percent + (video_progress / total_files)
                    yield (combined_progress, video_msg)
            else:
                # Procesar imagen
                frame = cv2.imread(filename)
                if frame is not None:
                    processed_frame = process_mgr.process_frame(frame, file_path=filename)
                    if processed_frame is not None:
                        cv2.imwrite(output_path, processed_frame)
                    else:
                        # Si no se procesó, copiar original
                        cv2.imwrite(output_path, frame)
                else:
                    print(f"[WARNING] No se pudo leer: {filename}")
                    
        except Exception as e:
            print(f"[ERROR] Procesando {filename}: {e}")
            import traceback
            traceback.print_exc()
        
        processed_files += 1
    
    # Finalizar
    process_mgr.Release()
    yield (100, f"Procesamiento completado: {processed_files} archivos")


def batch_process_with_options(list_files_process, options, progress_callback):
    """
    Procesa una lista de archivos con opciones específicas.
    Usado principalmente desde extras_tab para filtros y upscalers.
    """
    import roop.globals
    import roop.utilities as util
    from roop.ProcessMgr import ProcessMgr
    
    total_files = len(list_files_process)
    if total_files == 0:
        return
    
    # Inicializar ProcessMgr
    process_mgr = ProcessMgr()
    
    # Inicializar sin caras de origen (solo procesamiento de frame)
    process_mgr.initialize(
        [],  # Sin caras de origen
        [],  # Sin caras destino
        options
    )
    
    for entry in list_files_process:
        if not getattr(roop.globals, 'processing', True):
            break
            
        filename = entry.filename
        
        # Determinar si es video o imagen
        is_video = util.is_video(filename) or filename.lower().endswith('.gif')
        
        # Construir ruta de salida
        output_filename = os.path.basename(filename)
        output_path = os.path.join(roop.globals.output_path, output_filename)
        
        # Asegurar que el directorio de salida existe
        os.makedirs(roop.globals.output_path, exist_ok=True)
        
        try:
            if is_video:
                # Procesar video
                start_frame = entry.startframe if hasattr(entry, 'startframe') else 0
                end_frame = entry.endframe if hasattr(entry, 'endframe') else 0
                fps = entry.fps if hasattr(entry, 'fps') else 30.0
                
                for _ in process_mgr.run_batch_inmem(
                    filename,
                    output_path,
                    start_frame=start_frame,
                    end_frame=end_frame if end_frame > 0 else None,
                    fps=fps,
                    skip_audio=True
                ):
                    pass  # Consumir generador
            else:
                # Procesar imagen
                frame = cv2.imread(filename)
                if frame is not None:
                    processed_frame = process_mgr.process_frame(frame, file_path=filename)
                    if processed_frame is not None:
                        cv2.imwrite(output_path, processed_frame)
                    else:
                        cv2.imwrite(output_path, frame)
                        
        except Exception as e:
            print(f"[ERROR] Procesando {filename}: {e}")
    
    process_mgr.Release()


if __name__ == "__main__":
    run()
