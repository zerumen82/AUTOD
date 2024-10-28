import queue
import re
import sys

import gradio as gr

import os
import subprocess
import threading
import webview

script_path = 'webui-user.bat'


def SD_tab() -> None:
    with gr.Tab("as Stable Diffusion"):
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Stable Diffusion")
                gr.Button("Inicia Stable Diffusion", size='sm').click(
                    fn=lambda: stableTabli(script_path))
                gr.Label(
                    "Stable Diffusion es un modelo de difusión de video de alta calidad que puede ser utilizado para mejorar la calidad de video de baja resolución. ")


def is_valid_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None

def open_gradio_url(url):
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")

    def create_window():
        webview.create_window('STABLE-DIFFUSION', url)

    if threading.current_thread() is threading.main_thread():
        create_window()
        webview.start()
    else:
        q = queue.Queue()

        def run_on_main_thread():
            try:
                create_window()
                q.put(None)
            except Exception as e:
                q.put(e)

        main_thread = threading.Thread(target=run_on_main_thread)
        main_thread.start()
        main_thread.join()
        result = q.get()
        if result is not None:
            raise result

def read_output(pipe, queue):
    try:
        with pipe:
            for line in iter(pipe.readline, ''):
                print(line.strip())
                queue.put(line)
    except Exception as e:
        print(f"Error leyendo la salida del proceso: {e}")

def run_script(script_path, url_queue):
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    wib = os.path.join(script_dir, "ui", "tob", "stableDiffusionWebui")
    venv_path = os.path.join(wib, "venv", "Scripts", "activate")
    webui_script = os.path.join(wib, script_path)

    if not os.path.exists(webui_script):
        print(f"Error: La ruta especificada no existe: {webui_script}")
        return

    try:
        # Comprobar si torch está instalado
        check_command = f"{venv_path} && python -c \"import torch\""
        subprocess.run(check_command, shell=True, check=True)
        print("Torch ya está instalado en el entorno virtual.")
    except subprocess.CalledProcessError:
        print("Torch no está instalado. Procediendo a la instalación...")
        try:
            install_command = f"{venv_path} && pip install torch"
            subprocess.run(install_command, shell=True, check=True)
            print("Torch instalado correctamente en el entorno virtual.")
        except subprocess.CalledProcessError as e:
            print(f"Error al instalar torch: {e}")
            return

    try:
        # Ejecutar el script webui
        command = f"{venv_path} && {webui_script}"
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   shell=True, encoding='utf-8', errors='replace')
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar el comando: {e.stderr}")
        return
    except Exception as e:
        print(f"Se produjo un error inesperado: {e}")
        return

    stdout_queue = queue.Queue()
    stderr_queue = queue.Queue()

    stdout_thread = threading.Thread(target=read_output, args=(process.stdout, stdout_queue))
    stderr_thread = threading.Thread(target=read_output, args=(process.stderr, stderr_queue))

    stdout_thread.start()
    stderr_thread.start()

    url_pattern = re.compile(r'Local URL:\s*(http://127.0.0.1:\d+/)')
    try:
        for line in iter(stdout_queue.get, None):
            match = url_pattern.search(line)
            if match:
                url_queue.put(match.group(1))
                break
    except ValueError as e:
        print(f"Error leyendo la salida del proceso: {e}")

def stableTabli(script_path):
    url_queue = queue.Queue()
    script_thread = threading.Thread(target=run_script, args=(script_path, url_queue))
    script_thread.start()

    url = url_queue.get()
    if url:
        if threading.current_thread() is threading.main_thread():
            open_gradio_url(url)
        else:
            main_thread = threading.Thread(target=open_gradio_url, args=(url,))
            main_thread.start()
            main_thread.join()