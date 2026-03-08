import queue
import re
import os
import sys
import subprocess
import threading
import time
import urllib.request
import socket
import tempfile
import webbrowser

# Antes se bloqueaba la apertura del navegador; permitimos abrir en caso de fallback.
# webbrowser.open = lambda *args, **kwargs: print("[sd_launcher] BLOQUEADO: Intento de abrir navegador")
# webbrowser.open_new = lambda *args, **kwargs: print("[sd_launcher] BLOQUEADO: Intento de abrir navegador nuevo")
# webbrowser.open_new_tab = lambda *args, **kwargs: print("[sd_launcher] BLOQUEADO: Intento de abrir nueva pestaña")

script_path_default = 'webui-user.bat'

def clean_corrupted_configs(webui_path):
    """Limpia archivos de configuración corruptos que pueden causar errores JSON"""
    import json
    
    config_files = [
        'config.json',
        'ui-config.json', 
        'styles.csv',
        'params.txt'
    ]
    
    for config_file in config_files:
        config_path = os.path.join(webui_path, config_file)
        if os.path.exists(config_path):
            try:
                # Intentar leer archivos JSON
                if config_file.endswith('.json'):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            json.loads(content)
                        else:
                            # Archivo vacío, crear JSON válido
                            with open(config_path, 'w', encoding='utf-8') as fw:
                                fw.write('{}')
                            print(f"[sd_launcher] Archivo vacío reparado: {config_file}")
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                print(f"[sd_launcher] Archivo corrupto detectado: {config_file}, eliminando...")
                try:
                    os.remove(config_path)
                    print(f"[sd_launcher] Archivo corrupto eliminado: {config_file}")
                except Exception as del_e:
                    print(f"[sd_launcher] No se pudo eliminar {config_file}: {del_e}")
            except Exception as e:
                print(f"[sd_launcher] Error verificando {config_file}: {e}")

def kill_process_on_port(port):
    try:
        import psutil
        current_pid = os.getpid()
        for proc in psutil.process_iter(['pid', 'name', 'connections']):
            try:
                connections = proc.connections(kind='inet')
            except Exception:
                continue
            for conn in connections:
                try:
                    if hasattr(conn, 'laddr') and hasattr(conn.laddr, 'port') and conn.laddr.port == port:
                        # Nunca matar nuestro propio proceso
                        if proc.pid == current_pid:
                            continue
                        print(f"[sd_launcher] Matando proceso {proc.pid} ({proc.name()}) en puerto {port}")
                        try:
                            proc.terminate()
                            proc.wait(timeout=3)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass
                        return True
                except Exception:
                    continue
    except Exception as e:
        print(f"[sd_launcher] Error matando proceso en puerto {port}: {e}")
    return False

def find_free_port(start_port=9871, max_port=9900):
    """Encuentra un puerto libre en el rango especificado"""
    def is_port_available(port):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind(('127.0.0.1', port))
                return True
        except Exception:
            return False
    
    # Primero intentar puertos específicos
    for port in [9871, 9872, 9873, 9874, 9875]:
        if is_port_available(port):
            return port
    
    # Luego buscar en el rango completo
    for port in range(start_port, max_port):
        if is_port_available(port):
            return port
    
    return None

def launch_and_get_url(cmd_basename, cwd, timeout=60):
    port = find_free_port()
    if port is None:
        print("[sd_launcher] No se encontró puerto libre")
        return None, None

    env = os.environ.copy()
    env['GRADIO_SERVER_PORT'] = str(port)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['GRADIO_ANALYTICS_ENABLED'] = 'False'
    env['TRANSFORMERS_CACHE'] = os.path.join(cwd, 'cache')
    env['HF_HOME'] = os.path.join(cwd, 'cache')
    # Optimizacion de memoria CUDA para evitar OutOfMemory
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    # Asegurar que CUDA esté disponible - NO deshabilitar la prueba de CUDA
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
    ]
    current_path = env.get('PATH', '')
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path) and cuda_path not in current_path:
            env['PATH'] = cuda_path + os.pathsep + current_path
            print(f"[sd_launcher] Añadido CUDA al PATH: {cuda_path}")
            break
    env['REQS_FILE'] = 'requirements_versions.txt'

    if isinstance(cmd_basename, list):
        # Normalizar argumentos para evitar duplicados
        filtered = []
        skip_next = False
        for i, x in enumerate(cmd_basename):
            if skip_next:
                skip_next = False
                continue
            if x in ['--port', '--inbrowser', '--autolaunch', '--server-name', '--no-browser']:
                skip_next = True
                continue
            if str(x).isdigit() or x == '127.0.0.1':
                continue
            if x in ['--api', '--listen', '--autolaunch', '--inbrowser', '--no-browser']:
                continue
            filtered.append(x)
        
        cmd_list = filtered + [
            '--port', str(port),
            '--api',
            '--server-name', '127.0.0.1',
            '--no-browser'
        ]
        cmd = ['cmd.exe', '/c'] + cmd_list
    else:
        cmd = ['cmd.exe', '/c', cmd_basename] + [
            '--port', str(port),
            '--api',
            '--server-name', '127.0.0.1'
        ]

    try:
        print(f"[sd_launcher] Ejecutando comando: {' '.join(cmd)}")
        print(f"[sd_launcher] Directorio de trabajo: {cwd}")
        print(f"[sd_launcher] Puerto configurado: {port}")
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        print(f"[sd_launcher] Proceso iniciado con PID: {proc.pid}")
    except Exception as e:
        import roop.core
        msg = f"Error al iniciar proceso: {str(e)}"
        print(f"[sd_launcher] {msg}")
        roop.core.update_status(msg)
        raise RuntimeError(msg)

    url_q = queue.Queue()
    error_q = queue.Queue()

    def reader():
        global _last_url
        try:
            import roop.core
            from collections import deque
            last_line = None
            recent = deque(maxlen=50)

            for line in proc.stdout:
                ls = line.rstrip("\r\n")
                # Anti-duplicado simple
                if ls != last_line:
                    print(ls)
                    last_line = ls
                recent.append(ls)

                # Detectar errores comunes
                if ("CUDA out of memory" in ls or
                    "RuntimeError" in ls or
                    "Traceback" in ls or
                    "Exception" in ls or
                    "ERROR" in ls or
                    "Error" in ls or
                    "ImportError" in ls or
                    "Expected value" in ls or
                    "JSON decode error" in ls or
                    "JSONDecodeError" in ls):
                    try:
                        roop.core.update_status(f"SD: {ls}")
                    except Exception:
                        pass
                    error_q.put(ls)

                # Buscar URLs de gradio
                url = None
                if "Running on local URL:" in ls:
                    m = re.search(r'Running on local URL:\s*(http://[0-9.:]+)', ls)
                    if m:
                        url = m.group(1)
                        url = url.replace('0.0.0.0', '127.0.0.1')
                else:
                    m = re.search(r'http://[0-9.]+:\d+', ls)
                    if m:
                        url = m.group(0)
                        url = url.replace('0.0.0.0', '127.0.0.1')

                if url:
                    print(f"[sd_launcher] URL detectada: {url}")
                    _last_url = url
                    url_q.put(url)
                    
                    # Also set the SD URL in MainCase for webview opening
                    try:
                        import MainCase
                        MainCase.set_sd_url(url)
                        print(f"[sd_launcher] SD URL configurada en MainCase: {url}")
                    except Exception as e:
                        print(f"[sd_launcher] No se pudo configurar SD URL en MainCase: {e}")
                    
                    return

        except Exception as e:
            error_q.put(f"Error leyendo salida: {str(e)}")
            print(f"Error en thread reader: {str(e)}")

    t = threading.Thread(target=reader, daemon=True)
    t.start()

    try:
        url = url_q.get(timeout=timeout)
        if url:
            print(f"[sd_launcher] Servidor disponible en: {url}")
            print(f"[sd_launcher] URL obtenida exitosamente: {url}")
        return proc, url
    except queue.Empty:
        print("[sd_launcher] Tiempo de espera agotado esperando URL")
        return proc, None

def run_script(script_path=None):
    import roop.core

    if script_path is None:
        script_path = script_path_default

    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    wib = os.path.join(script_dir, "ui", "tob", "stable-diffusion-webui")

    # Verificar rutas
    if not os.path.exists(wib):
        msg = f"Error: No se encuentra la carpeta de Stable Diffusion WebUI en: {wib}"
        roop.core.update_status(msg)
        raise RuntimeError(msg)

    webui_script = os.path.join(wib, script_path)
    if not os.path.exists(webui_script):
        msg = f"Error: No se encuentra el script de inicio {script_path} en: {webui_script}"
        roop.core.update_status(msg)
        raise RuntimeError(msg)

    roop.core.update_status(f"Usando WebUI en: {wib}")

    # Limpiar archivos de configuración corruptos
    try:
        clean_corrupted_configs(wib)
        issues = diagnose_json_error(wib)
        if not issues:
            roop.core.update_status("Configuración verificada")
    except Exception as e:
        roop.core.update_status(f"Advertencia al verificar configuración: {str(e)}")

    # Liberar puertos primero (intentar matar procesos existentes)
    ports_to_check = [9871, 9872, 9873, 9874, 9875]
    for p in ports_to_check:
        if kill_process_on_port(p):
            roop.core.update_status(f"Puerto {p} liberado")
    
    # Pequeña pausa para que los puertos se liberen completamente
    time.sleep(1)
    
    # Buscar puerto libre
    port = find_free_port(start_port=9871, max_port=9900)
    if port is None:
        msg = "No se encontró puerto libre en el rango 9871-9900"
        roop.core.update_status(msg)
        raise RuntimeError(msg)
    
    roop.core.update_status(f"Usando puerto {port} para Stable Diffusion WebUI")
    print(f"[sd_launcher] Puerto seleccionado: {port}")

    # Configurar entorno
    env = os.environ.copy()
    env['GRADIO_SERVER_PORT'] = str(port)
    env['PYTHONUNBUFFERED'] = '1'
    env['PYTHONIOENCODING'] = 'utf-8'
    env['GRADIO_ANALYTICS_ENABLED'] = 'False'
    # Optimizacion de memoria CUDA para evitar OutOfMemory
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
    # NO usar WEBUI_SKIP_TORCH_CUDA_TEST - queremos que use CUDA
    # Asegurar que CUDA esté disponible en el PATH
    cuda_paths = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin',
    ]
    current_path = env.get('PATH', '')
    for cuda_path in cuda_paths:
        if os.path.exists(cuda_path) and cuda_path not in current_path:
            env['PATH'] = cuda_path + os.pathsep + current_path
            print(f"[sd_launcher] Añadido CUDA al PATH: {cuda_path}")
            break
    # Volver al lanzador oficial para asegurar la inicialización completa del API
    venv_python = os.path.join(wib, "venv", "Scripts", "python.exe")
    launch_py = os.path.join(wib, "launch.py")
    
    args = [
        '--port', str(port),
        '--api',
        '--cors-allow-origins', '*',
        '--xformers',  # USAR XFORMERS - acelera mucho la generacion
        '--opt-channelslast',
        '--lowvram',  # Modo LOW VRAM para 8GB
        '--always-batch-cond-uncond',  # Mejor estabilidad con lowvram
        '--upcast-sampling',  # Ayuda con memoria en ciertas operaciones
        '--no-half-vae',  # VAE en full precision (evita errores de tipo)
        '--enable-insecure-extension-access',  # Permitir extensiones
    ]
    
    # Limpieza total de argumentos en entorno para evitar conflictos
    if 'COMMANDLINE_ARGS' in env:
        del env['COMMANDLINE_ARGS']

    # Verificar CUDA antes de lanzar
    try:
        cuda_check_cmd = [venv_python, "-c", "import torch; print('CUDA_AVAILABLE:', torch.cuda.is_available()); print('CUDA_VERSION:', torch.version.cuda if torch.cuda.is_available() else 'N/A'); print('DEVICE:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"]
        result = subprocess.run(cuda_check_cmd, capture_output=True, text=True, env=env, cwd=wib)
        print(f"[sd_launcher] Verificación CUDA:\n{result.stdout}")
        if result.returncode != 0:
            print(f"[sd_launcher] Error verificando CUDA: {result.stderr}")
    except Exception as e:
        print(f"[sd_launcher] No se pudo verificar CUDA: {e}")
    
    if os.path.exists(venv_python) and os.path.exists(launch_py):
        cmd = [venv_python, launch_py] + args
        print(f"[sd_launcher] Lanzamiento con CUDA (launch.py): {venv_python}")
    else:
        # Fallback si no se encuentra el punto de entrada oficial
        cmd = ['cmd.exe', '/c', os.path.basename(webui_script)] + args
        print("[sd_launcher] Lanzador no encontrado, usando fallback .bat")

    try:
        roop.core.update_status("Iniciando Stable Diffusion WebUI con soporte CUDA...")
        print(f"[sd_launcher] Ejecutando: {' '.join(cmd)}")
        print(f"[sd_launcher] Directorio: {wib}")

        # Iniciar proceso
        proc = subprocess.Popen(cmd, cwd=wib, env=env)
        print(f"[sd_launcher] Proceso iniciado con PID: {proc.pid}")

        # Esperar un poco para que inicie
        time.sleep(5)

        # Verificar si el proceso sigue vivo
        if proc.poll() is None:
            url = f"http://127.0.0.1:{port}"
            roop.core.update_status("Proceso iniciado, esperando que cargue...")

            def launcher_thread(url: str):
                import roop.core
                
                # Esperar 20 segundos para que el servidor inicie
                roop.core.update_status("Esperando 20s para que SD WebUI cargue...")
                time.sleep(20)
                
                # Verificar si el servidor responde
                max_wait = 120  # 2 minutos adicionales
                waited = 0
                server_ready = False
                
                while waited < max_wait:
                    try:
                        # Solo verificar si el puerto responde
                        with socket.create_connection(('127.0.0.1', port), timeout=2):
                            roop.core.update_status("¡Stable Diffusion WebUI listo!")
                            server_ready = True
                            break
                    except Exception:
                        pass
                    
                    time.sleep(2)
                    waited += 2
                    if waited % 30 == 0:
                        roop.core.update_status(f"Esperando... ({waited//60}:{waited%60:02d})")
                
                # Configurar URL
                try:
                    import MainCase
                    MainCase.set_sd_url(url)
                    print(f"[sd_launcher] SD URL configurada: {url}")
                except Exception as e:
                    print(f"[sd_launcher] Error configurando URL: {e}")
                
                # Abrir webview SIEMPRE (incluso si no detectamos el servidor)
                _open_in_webview(url)
                # Solo configurar URL
                try:
                    import MainCase
                    MainCase.set_sd_url(url)
                    print(f"[sd_launcher] SD URL configurada: {url}")
                except Exception as e:
                    print(f"[sd_launcher] Error configurando URL: {e}")

            th = threading.Thread(target=launcher_thread, args=(url,), daemon=True)
            th.start()
            return True
        else:
            roop.core.update_status("Error: El proceso terminó inmediatamente")
            return False

    except Exception as e:
        roop.core.update_status(f"Error iniciando SD WebUI: {str(e)}")
        print(f"[sd_launcher] Error: {e}")
        return False

# Variables globales para control de inicio
_startup_lock = threading.Lock()
_is_running = False
_last_url = None

def _open_in_webview(url):
    """Función auxiliar para abrir URL en WebView separado"""
    import roop.core
    
    try:
        print(f"[sd_launcher] ABRIENDO WEBVIEW: {url}")
        roop.core.update_status("Abriendo WebView separado...")
        
        # Usar MainCase para abrir en proceso separado
        import MainCase
        MainCase.open_sd_window(url)
        
        roop.core.update_status(f"SD WebUI abierto en ventana separada: {url}")
        
    except Exception as e:
        print(f"[sd_launcher] ERROR abriendo WebView: {str(e)}")
        roop.core.update_status(f"Error abriendo WebView: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback a navegador
        import webbrowser
        webbrowser.open(url)
        roop.core.update_status(f"Abierto en navegador: {url}")

def start(script_path=None):
    """Inicia el proceso de SD en background"""
    global _is_running

    # Usar un lock para asegurar exclusión mutua
    if not _startup_lock.acquire(blocking=False):
        import roop.core
        roop.core.update_status("Stable Diffusion ya está iniciándose...")
        return None

    # Si ya está corriendo, liberar el lock y salir
    if _is_running:
        _startup_lock.release()
        import roop.core
        roop.core.update_status("Stable Diffusion ya está ejecutándose")
        return None

    _is_running = True

    def run_with_ui():
        try:
            import roop.core
            # Iniciar proceso y esperar URL
            try:
                success = run_script(script_path)
                if not success:
                    roop.core.update_status("No se pudo iniciar Stable Diffusion")
            except RuntimeError as re:
                # Errores esperados (falta de archivos, etc)
                roop.core.update_status(f"Error: {str(re)}")
                print(f"Error de ejecución: {str(re)}")
            except Exception as e:
                # Errores inesperados
                error_msg = f"Error inesperado: {str(e)}"
                roop.core.update_status(error_msg)
                print(f"Error detallado: {e}")
                print("Traceback completo:")
                import traceback
                traceback.print_exc()
        except Exception as e:
            # Error crítico
            print(f"Error crítico: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # Siempre liberar recursos
            try:
                global _is_running
                _is_running = False
                _startup_lock.release()
            except Exception as e:
                print(f"Error liberando recursos: {str(e)}")
                try:
                    _startup_lock.release()
                except:
                    pass

    # Iniciar en background
    t = threading.Thread(target=run_with_ui, daemon=True)
    t.start()
    return t

def stop():
    """Detiene el proceso de SD si está corriendo"""
    global _is_running, _last_url
    import roop.core

    if _is_running:
        # Intentar matar procesos en el puerto (rango completo)
        ports_to_check = [9871, 9872, 9873, 9874, 9875, 9876, 9877, 9878, 9879, 9880]
        killed = False
        for p in ports_to_check:
            if kill_process_on_port(p):
                roop.core.update_status(f"Proceso SD detenido en puerto {p}")
                killed = True

        if killed:
            _is_running = False
            _last_url = None
            roop.core.update_status("Stable Diffusion detenido")
        else:
            roop.core.update_status("No se encontró proceso SD para detener")
    else:
        roop.core.update_status("Stable Diffusion no está ejecutándose")

def get_last_url():
    return _last_url

def diagnose_json_error(webui_path):
    """Diagnostica problemas comunes que causan errores JSON"""
    import roop.core
    
    issues_found = []
    
    # Verificar archivos de configuración
    config_files = ['config.json', 'ui-config.json']
    for config_file in config_files:
        config_path = os.path.join(webui_path, config_file)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        issues_found.append(f"Archivo vacío: {config_file}")
                    else:
                        import json
                        json.loads(content)
            except json.JSONDecodeError as e:
                issues_found.append(f"JSON inválido en {config_file}: {str(e)}")
            except Exception as e:
                issues_found.append(f"Error leyendo {config_file}: {str(e)}")
    
    # Verificar permisos de escritura
    try:
        test_file = os.path.join(webui_path, 'test_write.tmp')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
    except Exception as e:
        issues_found.append(f"Sin permisos de escritura: {str(e)}")
    
    # Verificar espacio en disco
    try:
        import shutil
        free_space = shutil.disk_usage(webui_path).free
        if free_space < 1024 * 1024 * 100:  # Menos de 100MB
            issues_found.append(f"Poco espacio en disco: {free_space // (1024*1024)}MB")
    except Exception:
        pass
    
    if issues_found:
        roop.core.update_status(f"Problemas detectados: {'; '.join(issues_found)}")
        return issues_found
    else:
        roop.core.update_status("No se encontraron problemas de configuración")
        return []
