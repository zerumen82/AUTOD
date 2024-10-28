import os
import sys
import time
import gradio as gr

import roop.globals
import roop.metadata
import roop.utilities as util
import ui.globals as uii
from ui.tabs.DeepNude_Tab import DeepNude_tab
from ui.tabs.SD_tab import SD_tab

from ui.tabs.faceswap_tab import faceswap_tab
from ui.tabs.livecam_tab import livecam_tab
from ui.tabs.facemgr_tab import facemgr_tab
from ui.tabs.extras_tab import extras_tab
from ui.tabs.settings_tab import settings_tab
from ui.tabs.sounds_tab import sounds_tab

roop.globals.keep_fps = None
roop.globals.keep_frames = None
roop.globals.skip_audio = None
roop.globals.use_batch = None


def prepare_environment():
    roop.globals.output_path = os.path.abspath(os.path.join(os.getcwd(), "output"))
    os.makedirs(roop.globals.output_path, exist_ok=True)
    if not roop.globals.CFG.use_os_temp_folder:
        os.environ["TEMP"] = os.environ["TMP"] = os.path.abspath(os.path.join(os.getcwd(), "temp"))
    os.makedirs(os.environ["TEMP"], exist_ok=True)
    os.environ["GRADIO_TEMP_DIR"] = os.environ["TEMP"]
    os.environ['GRADIO_ANALYTICS_ENABLED'] = '0'


def main_function():
    faceswap_tab()
    livecam_tab()
    facemgr_tab()
    extras_tab()
    settings_tab()


def run():
    from roop.core import decode_execution_providers, set_display_ui

    prepare_environment()

    set_display_ui(show_msg)
    roop.globals.execution_providers = decode_execution_providers([roop.globals.CFG.provider])
    print(f'Using provider {roop.globals.execution_providers} - Device: {util.get_device()}')

    run_server = True
    uii.ui_restart_server = False
    mycss = """
    body {font-family: Arial, sans-serif; line-height: 1.6; color: #333; padding: 20px; background-color: #f4f4f4;}
    .container {margin: auto; width: 80%; overflow: auto; padding: 20px;}
    header {background: #50a8a0; color: #ffffff; padding-top: 30px; min-height: 70px; border-bottom: #e8491d 3px solid;}
    header a {color: #ffffff; text-decoration: none; text-transform: uppercase; font-size: 16px;}
    header li {float: left; display: inline; padding: 0 20px 0 20px;}
    header #branding {float: left;}
    header #branding h1 {margin: 0;}
    header nav {float: right;margin-top: 10px;}
    header .highlight, header .current a {color: #e8491d;font-weight: bold;}
    header a:hover {color: #cccccc;font-weight: bold;}
    span {color: var(--block-info-text-color)}
    .selected-image {box-shadow: 0 0 10px #000;}
        #fixedheight {
            max-height: 238.4px;
            overflow-y: auto !important;
        }
    .image-container.svelte-1l6wqyv {height: 100%}
    button:focus img {filter: brightness(0.5)}
    """
    while run_server:
        server_name = roop.globals.CFG.server_name
        if server_name is None or len(server_name) < 1:
            server_name = None
        server_port = roop.globals.CFG.server_port
        if server_port <= 0:
            server_port = None
        ssl_verify = False if server_name == '0.0.0.0' else True

        with gr.Blocks(title=f'AUTO-DEEP IMAGENES Y VIDEOS', theme=roop.globals.CFG.selected_theme, css=mycss) as ui:
            with gr.Row(variant='compact'):
                gr.Markdown(f"### AUTO-DEEP v1- PillSoftware")
                gr.HTML(util.create_version_html(), elem_id="versions")
            faceswap_tab()
            DeepNude_tab()
            SD_tab()
            sounds_tab()
            with gr.Tab("⚙ Configuracion"):
                facemgr_tab()
                extras_tab()
                settings_tab()
            # facemgr_tab()
            # livecam_tab()

            # extras_tab()


        uii.ui_restart_server = False
        try:
            # # Asegúrese de inicializar su objeto Blocks aquí
            # ui.queue().launch(inline=True, server_name=server_name, server_port=server_port, share=True,
            #                   ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True)
            ui.queue().launch(inbrowser=False, server_name=server_name, server_port=server_port, share=True, ssl_verify=ssl_verify, prevent_thread_lock=True, show_error=True)
        except Exception as e:
            print(f'Exception {e} when launching Gradio Server!')
            uii.ui_restart_server = True
            run_server = False
        try:
            import MainCase as mi
            mi.publicurl = ui.share_url
            mi.run_gradio_and_load_url()
            while not uii.ui_restart_server:
                time.sleep(1.0)
        except (KeyboardInterrupt, OSError):
            print("Keyboard interruption in main thread... closing server.")
            run_server = False
        ui.close()

def show_msg(msg: str):
    gr.Info(msg)
