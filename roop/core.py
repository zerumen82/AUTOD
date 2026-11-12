import sys
import os
import socket
import threading
import time
import requests
import cv2
import numpy as np

from roop.stdio_safety import ensure_utf8_stdio

# v5.1339: reconfigura stdout/stderr a UTF-8 antes de que se imprima nada.
# Sin esto, un mensaje con ❌/⚠️/→ en una consola cp1252 lanza
# UnicodeEncodeError y aborta la operación que lo imprimió.
ensure_utf8_stdio()


class _Tee:
    """Duplicate writes to a file alongside the original stream."""
    def __init__(self, original, fileobj):
        self._original = original
        self._file = fileobj
    def write(self, data):
        self._original.write(data)
        try:
            self._file.write(str(data))
            self._file.flush()
        except Exception:
            pass
    def flush(self):
        self._original.flush()
        try:
            self._file.flush()
        except Exception:
            pass


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


APP_VERSION = "v5.1339 (2026-11-11)"


def _active_server_path():
    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "logs", "active_server.json")
    except Exception:
        return os.path.join("logs", "active_server.json")


def _read_active_server():
    import json as _json
    try:
        path = _active_server_path()
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            data = _json.load(handle)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _write_active_server(url, log_path=""):
    import json as _json
    try:
        path = _active_server_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            _json.dump(
                {
                    "version": APP_VERSION,
                    "pid": os.getpid(),
                    "url": url,
                    "log": log_path or "",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                },
                handle,
            )
    except Exception:
        pass


def _tail_primary_log(log_path, own_log_hint=""):
    """La ventana secundaria muestra los logs del servidor primario.

    Sin esto la segunda consola se queda muda en '[UI] Abriendo
    pywebview...' y el usuario cree que no hay logs (reporte 2026-10-30).
    Solo lee: jamas escribe en el log primario (usa sys.__stdout__ para
    no realimentar el _SessionTee propio; seguir el propio log causo el
    bucle de 202MB del 2026-10-31).
    """
    import sys as _sys

    def _emit(text):
        try:
            _sys.__stdout__.write(str(text) + "\n")
            _sys.__stdout__.flush()
        except Exception:
            pass
    try:
        if not log_path or not os.path.exists(str(log_path)):
            _emit("[INFO] Log primario no disponible; esta ventana no sigue logs.")
            return
        try:
            if own_log_hint and os.path.abspath(str(log_path)) == os.path.abspath(str(own_log_hint)):
                _emit("[INFO] El log primario es esta misma ventana; no se sigue a si misma.")
                return
        except Exception:
            return
        _emit(f"[INFO] Siguiendo log primario: {log_path}")
        with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, os.SEEK_END)
            while True:
                line = handle.readline()
                if line:
                    _emit(line.rstrip("\n"))
                else:
                    time.sleep(0.5)
    except Exception as exc:
        _emit(f"[INFO] Fin del seguimiento del log primario ({exc}).")


def find_available_port(start_port=7861, max_attempts=10):
    """Busca un puerto disponible"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError("No hay puerto libre")


def is_gradio_running(url, timeout=5):
    """Verifica si Gradio esta ejecutando"""
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def find_existing_autoauto_server(start_port=7861, max_attempts=10):
    """Reuse this project's Gradio server instead of starting another GPU stack."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        url = f"http://127.0.0.1:{port}"
        # El puerto debe estar en LISTEN de verdad: un TIME_WAIT/FIN_WAIT
        # huerfano no es un servidor vivo (log 2026-10-30: consolas
        # secundarias "reutilizando" un 7861 ya muerto y consolas mudas).
        try:
            probe = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            probe.close()
        except OSError:
            continue
        try:
            response = requests.get(f"{url}/config", timeout=0.5)
            if response.status_code == 200 and response.json().get("title") == "AutoAuto - AI Editor":
                return url
        except (requests.RequestException, ValueError):
            continue
    return None


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
        
        existing_server_url = find_existing_autoauto_server()
        _active = _read_active_server()
        if existing_server_url and _active and str(_active.get("version", "")) != APP_VERSION:
            print(
                f"[SERVER] Servidor existente {_active.get('version', '?')} (PID "
                f"{_active.get('pid', '?')}) distinto de {APP_VERSION}: se ignora "
                "para cargar codigo nuevo. Cierra TODAS las ventanas si ves dos iconos."
            )
            existing_server_url = None
        if existing_server_url:
            print(f"[SERVER] Reutilizando AutoAuto existente en {existing_server_url}")
            _primary_log = (_active or {}).get("log", "") if isinstance(_active, dict) else ""
            _primary_pid = (_active or {}).get("pid", None) if isinstance(_active, dict) else None
            # Sin marcador activo (servidor viejo) NO se sigue nada: seguir el
            # propio log realimento el _SessionTee y genero el bucle de 202MB
            # del 2026-10-31. Solo se sigue si hay primario distinto vivo.
            if _active and _primary_pid and int(_primary_pid) != int(os.getpid()) and _primary_log and os.path.exists(str(_primary_log)):
                print("[INFO] Ventana SECUNDARIA: los logs del Editor salen AQUI tambien (seguimiento activado).")
                print(f"[INFO] Log primario (PID {_primary_pid}): {_primary_log}")
                _tail_thread = threading.Thread(
                    target=_tail_primary_log, args=(_primary_log, ""), daemon=True, name="PrimaryLogTail"
                )
                _tail_thread.start()
            else:
                print("[INFO] Ventana SECUNDARIA: los logs del Editor estan en la PRIMERA consola.")
                print("[INFO] Para verlo todo en una sola, cierra TODAS y abre UNA sola vez.")
        
        # Configurar manejo de excepciones asyncio
        
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

        # Parche Gradio routes para SSE stream disconnection limpia (evita RuntimeError / 404 de ASGI)
        try:
            import gradio.routes  # noqa: F401 - comprueba que el módulo se importa
            # Verificación de que gradio.routes está cargado sin excepciones
            print("[OK] Patch Gradio SSE aplicado")
        except Exception:
            pass

        if existing_server_url:
            server_url = existing_server_url
        else:
            from ui import main as ui_main

            # Un solo servidor ComfyUI y una sola interfaz Gradio por proyecto.
            print("\n[COMFYUI] Iniciando ComfyUI automáticamente...")
            try:
                from ui.tabs.comfy_launcher import start as start_comfyui
                success, msg, port = start_comfyui(port=8188, directly_run=True)
                if success:
                    print(f"[OK] ComfyUI iniciado en puerto {port}")
                else:
                    print(f"[WARNING] ComfyUI no se pudo iniciar: {msg}")
            except Exception as e:
                print(f"[WARNING] Error iniciando ComfyUI: {e}")

            print("[INIT] Buscando puerto...")
            available_port = find_available_port(7861)
            print(f"[OK] Puerto: {available_port}")
            server_url = f"http://127.0.0.1:{available_port}"

            def start_server():
                os.environ['GRADIO_SERVER_PORT'] = str(available_port)
                ui_main.run()

            print("[SERVER] Iniciando servidor Gradio...")
            server_thread = threading.Thread(target=start_server, daemon=True, name="GradioServer")
            server_thread.start()

            print("[SERVER] Verificando que Gradio este ejecutando...")
            max_wait = 180  # Carga inicial de modelos.
            checked = False
            for i in range(max_wait):
                time.sleep(1)
                if is_gradio_running(server_url):
                    print(f"[OK] Gradio ejecutando en {server_url}")
                    checked = True
                    break
                if i > 0 and i % 10 == 0:
                    print(f"[SERVER] Esperando... {i}s/{max_wait}s")

            if not checked:
                print(f"[ERROR] Gradio no respondio despues de {max_wait}s")
                print(f"[INFO] Abre manualmente: {server_url}")
                return
            try:
                import glob as _glob2
                _base2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                _logs2 = sorted(
                    _glob2.glob(os.path.join(_base2, "logs", "ui_start_*.log")),
                    key=lambda p: os.path.getmtime(p),
                )
                _write_active_server(server_url, _logs2[-1] if _logs2 else "")
            except Exception:
                _write_active_server(server_url, "")
        
        # AHORA si abrir pywebview
        print("[UI] Abriendo pywebview...")
        import webview

        _icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icon.ico")

        def _set_window_icon(uid):
            import time
            import ctypes
            from webview.platforms.winforms import BrowserView
            from System.Drawing import Icon
            
            if not os.path.exists(_icon_path):
                return
                
            timeout = 15
            start = time.time()
            while uid not in BrowserView.instances and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            if uid not in BrowserView.instances:
                return
                
            form = BrowserView.instances[uid]
            try:
                from System import Action
                user32 = ctypes.windll.user32
                WM_SETICON = 0x0080
                ICON_SMALL = 0
                ICON_BIG = 1
                GCLP_HICON = -14
                GCLP_HICONSM = -34

                def _set():
                    try:
                        # Set AppUserModelID to prevent taskbar grouping with python.exe
                        try:
                            shell32 = ctypes.windll.shell32
                            setAppId = shell32.SetCurrentProcessExplicitAppUserModelID
                            setAppId.argtypes = [ctypes.c_wchar_p]
                            setAppId.restype = ctypes.c_long
                            hr = setAppId("AutoAuto")
                            if hr == 0:
                                print("[UI] AppUserModelID asignado")
                        except Exception:
                            pass

                        ico = Icon(_icon_path)
                        form.Icon = ico
                        hWnd = form.Handle.ToInt64()
                        hWnd_ptr = ctypes.c_void_p(hWnd)

                        shell32 = ctypes.windll.shell32
                        shell32.ExtractIconExW.argtypes = [ctypes.c_wchar_p, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
                        shell32.ExtractIconExW.restype = ctypes.c_uint
                        large = ctypes.c_void_p()
                        small = ctypes.c_void_p()
                        if shell32.ExtractIconExW(_icon_path, 0, ctypes.byref(large), ctypes.byref(small), 1) > 0:
                            h32 = ctypes.c_void_p(large.value)
                            h16 = ctypes.c_void_p(small.value)
                            user32.SendMessageW(hWnd_ptr, WM_SETICON, ICON_BIG, h32)
                            user32.SetClassLongPtrW(hWnd_ptr, GCLP_HICON, h32)
                            user32.SendMessageW(hWnd_ptr, WM_SETICON, ICON_SMALL, h16)
                            user32.SetClassLongPtrW(hWnd_ptr, GCLP_HICONSM, h16)
                            print("[UI] Icono asignado")
                    except Exception as e:
                        print(f"[DEBUG] Error en _set: {e}")
                form.Invoke(Action(_set))
            except Exception as e:
                print(f"[DEBUG] Error forzando icono: {e}")

        window = webview.create_window(
            "AutoAuto v5.1339",
            url=server_url,
            width=1400,
            height=900
        )
        threading.Thread(target=_set_window_icon, args=(window.uid,), daemon=True).start()
        # Al cerrar la ventana con la X la sesión debe liberar la GPU y cerrar
        # ComfyUI SIEMPRE, aunque esta ventana haya reutilizado una sesión ya
        # abierta (decisión del usuario 2026-10-20: «siempre, sin excepciones»).
        # atexit cubre las salidas que no pasan por la X (Ctrl+C, excepción).
        from roop.session_shutdown import register_shutdown_hook, shutdown_session
        register_shutdown_hook()
        webview.start(debug=False, func=None)
        
        report = shutdown_session("ventana-cerrada")
        _comfyui_report = report.get("comfyui") or {}
        print(
            f"[EXIT] Ventana cerrada; cierre ejecutado en {report.get('seconds', '?')}s — "
            f"ComfyUI detenido={_comfyui_report.get('stopped')} "
            f"({str(_comfyui_report.get('message', 'no configurado'))[:80]}), "
            f"VRAM {report.get('vram_delta_mb')}MB, ok={report.get('ok')}"
        )
        return existing_server_url is not None
        
    except ImportError as e:
        print(f"[ERROR] Pywebview no instalado: {e}")
        _url = server_url if 'server_url' in dir() else 'http://127.0.0.1:7861'
        print(f"[INFO] Abre manualmente: {_url}")
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
    
    input_facesets = getattr(roop.globals, 'INPUT_FACESETS', [])
    total_source_faces = sum(len(fs.faces) for fs in input_facesets if hasattr(fs, 'faces') and fs.faces)
    print(f"[LOAD] Fuentes: {len(input_facesets)} facesets, {total_source_faces} caras de origen")
    
    target_faces_data = getattr(roop.globals, 'TARGET_FACES', [])
    if target_faces_data:
        print(f"[LOAD] Destino: {len(target_faces_data)} cara(s) seleccionada(s)")
    
    total_files = len(list_files_process)
    if total_files == 0:
        yield (100, "No hay archivos para procesar")
        return

    if getattr(roop.globals, 'face_swap_mode', None) in ('selected', 'selected_faces', 'selected_faces_frame'):
        from roop.destination_selection import missing_video_references
        video_paths = [entry.filename for entry in list_files_process
                       if util.is_video(entry.filename) or entry.filename.lower().endswith('.gif')]
        missing_targets = missing_video_references(
            video_paths, getattr(roop.globals, 'selected_face_references', {}),
        )
        if missing_targets:
            names = ', '.join(os.path.basename(path) for path in missing_targets)
            message = f"ERROR: falta seleccionar cara destino en {len(missing_targets)} vídeo(s): {names}"
            print(f"[BATCH] {message}")
            yield (0, message)
            return
    
    # Asegurar que processing está activo
    roop.globals.processing = True
    
    # Session log for photo batches (video batches have their own in run_batch_inmem)
    _batch_log_file = None
    from datetime import datetime
    batch_has_photos = any(not (util.is_video(e.filename) or e.filename.lower().endswith('.gif')) for e in list_files_process)
    if batch_has_photos:
        try:
            os.makedirs(r'D:\PROJECTS\AUTOAUTO\logs', exist_ok=True)
            _log_name = f"process_{datetime.now():%Y%m%d_%H%M%S}.log"
            _log_path = os.path.join(r'D:\PROJECTS\AUTOAUTO\logs', _log_name)
            _batch_log_file = open(_log_path, 'w', encoding='utf-8')
            _batch_log_file.write("=== AUTO-AUTO v5.1190 — BATCH FOTOS ===\n")
            _batch_log_file.write(f"Start: {datetime.now()}\n")
            _batch_log_file.write(f"PID: {os.getpid()}\n")
            _batch_log_file.write(f"Files: {total_files}\n\n")
            _batch_log_file.flush()
        except Exception:
            _batch_log_file = None

    # Redirect stdout to tee into batch log
    _orig_stdout = sys.stdout
    if _batch_log_file:
        sys.stdout = _Tee(sys.stdout, _batch_log_file)
    
    # Inicializar ProcessMgr
    process_mgr = ProcessMgr()

    processed_files = 0
    photos_without_swap = []

    try:
        # Configurar opciones de procesamiento
        processor_options = {'face_swapper': {}}
        if mask_engine and mask_engine != "None":
            processor_options["masking"] = {"subtype": mask_engine}
    
        # ProcessOptions(processordefines, face_distance, blend_ratio, swap_mode, selected_index, masking_text, imagemask, num_steps, show_face_area, ...)
        options = ProcessOptions(
            processor_options,                                    # processordefines
            getattr(roop.globals, 'distance_threshold', 0.6),    # face_distance
            getattr(roop.globals, 'blend_ratio', 0.95),          # blend_ratio
            roop.globals.face_swap_mode if hasattr(roop.globals, 'face_swap_mode') else 'all',  # swap_mode
            0,  # selected_index: source automatica, solo se elige destino
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
            # v5.1042: OpenCV no tiene writer para .jfif/.jpe/.jpeg → imwrite falla silenciosamente
            # (log: "could not find a writer for the specified extension" en 505735164.jfif).
            # Normalizar a .jpg, que cv2.imwrite soporta siempre.
            output_ext = os.path.splitext(output_filename)[1].lower()
            if output_ext in ('.jfif', '.jpe', '.jpeg'):
                output_filename = os.path.splitext(output_filename)[0] + '.jpg'
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
                        skip_audio=getattr(roop.globals, 'skip_audio', False),
                        temporal_smoothing=temporal_smoothing,
                    ):
                        # El porcentaje principal es global del lote; el mensaje conserva
                        # el avance local del video para evitar lecturas ambiguas en la UI.
                        combined_progress = progress_percent + (video_progress / total_files)
                        progress_scope = f"Archivo {processed_files + 1}/{total_files}"
                        if total_files > 1:
                            progress_scope += f" | video {video_progress:.1f}%"
                        yield (combined_progress, f"{progress_scope} | {video_msg}")
                else:
                    # Procesar imagen - usar imdecode para soportar WebP
                    from roop.capturer import get_image_frame
                    frame = get_image_frame(filename)
                    if frame is not None:
                        processed_frame = process_mgr.process_frame(frame, file_path=filename)
                        if processed_frame is not None:
                            if np.array_equal(processed_frame, frame):
                                photos_without_swap.append(output_filename)
                                print(f"[FOTO] SIN_SWAP {os.path.basename(filename)}: salida idéntica al original; revisar los mensajes [FOTO] y [SWAP_SKIP] anteriores", flush=True)
                            if not cv2.imwrite(output_path, processed_frame):
                                print(f"[FOTO] ERROR al guardar {output_path}", flush=True)
                        else:
                            # Si no se procesó, copiar original
                            photos_without_swap.append(output_filename)
                            print(f"[FOTO] SIN_SWAP {os.path.basename(filename)}: procesador devolvió None", flush=True)
                            if not cv2.imwrite(output_path, frame):
                                print(f"[FOTO] ERROR al guardar {output_path}", flush=True)
                    else:
                        print(f"[WARNING] No se pudo leer: {filename}")
                        photos_without_swap.append(output_filename)
                    
            except Exception as e:
                print(f"[ERROR] Procesando {filename}: {e}")
                if not is_video:
                    photos_without_swap.append(output_filename)
                import traceback
                traceback.print_exc()
        
            processed_files += 1
    
    finally:
        # Finalizar
        try:
            process_mgr.Release()
        except Exception:
            pass
        # Restore stdout and close batch log
        try:
            sys.stdout = _orig_stdout
        except Exception:
            pass
        if _batch_log_file:
            try:
                # Sin esta marca el finally no corrió: el proceso murió por
                # fuera. El PID permite correlacionar con ui_start_*.log y
                # con los eventos de Windows del incidente 2026-10-13.
                _batch_log_file.write(f"\n=== END: {processed_files} files (PID {os.getpid()}) ===\n")
                _batch_log_file.close()
            except Exception:
                pass
    if photos_without_swap:
        yield (100, f"Procesados {processed_files} archivos; {len(photos_without_swap)} foto(s) sin swap: {', '.join(photos_without_swap[:5])}")
    else:
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
        # v5.1042: OpenCV no tiene writer para .jfif/.jpe/.jpeg → imwrite falla silenciosamente.
        # Normalizar a .jpg, que cv2.imwrite soporta siempre.
        output_ext = os.path.splitext(output_filename)[1].lower()
        if output_ext in ('.jfif', '.jpe', '.jpeg'):
            output_filename = os.path.splitext(output_filename)[0] + '.jpg'
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
                    skip_audio=True,
                    temporal_smoothing=bool(getattr(roop.globals, 'temporal_smoothing', True)),
                ):
                    pass  # Consumir generador
            else:
                # Procesar imagen - usar imdecode para soportar WebP
                from roop.capturer import get_image_frame
                frame = get_image_frame(filename)
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
