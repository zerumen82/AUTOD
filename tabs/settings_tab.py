import shutil
import os
import gradio as gr
import roop.globals
import ui.globals

available_themes = ["Default", "gradio/glass", "gradio/monochrome", "gradio/seafoam", "gradio/soft", "gstaff/xkcd", "freddyaboulton/dracula_revamped", "ysharma/steampunk"]
image_formats = ['jpg', 'png', 'webp']
video_formats = ['avi', 'mkv', 'mp4', 'webm']
video_codecs = ['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc']
providerlist = None

settings_controls = []


def settings_tab():
    from roop.core import suggest_execution_providers
    global providerlist

    providerlist = suggest_execution_providers()
    with gr.Tab("⚙ Settings"):
        with gr.Row():
            with gr.Column():
                themes = gr.Dropdown(available_themes, label="Temas", info="El cambio necesita un reinicio completo",
                                     value=roop.globals.CFG.selected_theme)
            with gr.Column():
                settings_controls.append(
                    gr.Checkbox(label="Servidor Público", value=roop.globals.CFG.server_share, elem_id='server_share',
                                interactive=True))
                settings_controls.append(gr.Checkbox(label='Borrar la carpeta de salida antes de cada ejecución',
                                                     value=roop.globals.CFG.clear_output, elem_id='clear_output',
                                                     interactive=True))
                output_template = gr.Textbox(label="Plantilla de salida de nombre de archivo",
                                             info="(la extensión del archivo se agrega automáticamente)", lines=1,
                                             placeholder='{file}_{time}', value=roop.globals.CFG.output_template)
            with gr.Column():
                input_server_name = gr.Textbox(label="Nombre del Servidor", lines=1,
                                               info="Déjelo en blanco para ejecutarlo localmente",
                                               value=roop.globals.CFG.server_name)
            with gr.Column():
                input_server_port = gr.Number(label="Puerto del Servidor", precision=0,
                                              info="Déjelo en 0 para usar el valor predeterminado",
                                              value=roop.globals.CFG.server_port)
        with gr.Row():
            with gr.Column():
                settings_controls.append(
                    gr.Dropdown(providerlist, label="Proveedor", value=roop.globals.CFG.provider, elem_id='provider',
                                interactive=True))
                chk_det_size = gr.Checkbox(label="Usar DET-SIZE predeterminado", value=False, elem_id='default_det_size',
                                           interactive=True)
                settings_controls.append(
                    gr.Checkbox(label="Forzar CPU para analizador facial", value=roop.globals.CFG.force_cpu,
                                elem_id='force_cpu', interactive=True))
                max_threads = gr.Slider(1, 32, value=roop.globals.CFG.max_threads, label="Máx. Número de hilos",
                                        info='default: 3', step=1.0, interactive=True)
            with gr.Column():
                memory_limit = gr.Slider(0, 128, value=roop.globals.CFG.memory_limit,
                                         label="Máx. Memoria a utilizar (Gb)", info='0 significa sin límite', step=1.0,
                                         interactive=True)
                settings_controls.append(
                    gr.Dropdown(image_formats, label="Formato de salida de imagen", info='default: png',
                                value=roop.globals.CFG.output_image_format, elem_id='output_image_format',
                                interactive=True))
            with gr.Column():
                settings_controls.append(gr.Dropdown(video_codecs, label="Video Codec", info='default: libx264',
                                                     value=roop.globals.CFG.output_video_codec,
                                                     elem_id='output_video_codec', interactive=True))
                settings_controls.append(
                    gr.Dropdown(video_formats, label="Formato Vide predeterminado", info='default: mp4',
                                value=roop.globals.CFG.output_video_format, elem_id='output_video_format',
                                interactive=True))
                video_quality = gr.Slider(0, 100, value=roop.globals.CFG.video_quality, label="Calidad del Video (crf)",
                                          info='default: 14', step=1.0, interactive=True)
            with gr.Column():
                with gr.Group():
                    settings_controls.append(gr.Checkbox(label='Usa la carpeta de temporales del SO',
                                                         value=roop.globals.CFG.use_os_temp_folder,
                                                         elem_id='use_os_temp_folder', interactive=True))
                    settings_controls.append(
                        gr.Checkbox(label='Mostrar vídeo en el navegador (vuelve a codificar la salida)',
                                    value=roop.globals.CFG.output_show_video, elem_id='output_show_video',
                                    interactive=True))
                button_apply_restart = gr.Button("Restart Server", variant='primary')
                button_clean_temp = gr.Button("Clean temp folder")
                button_apply_settings = gr.Button("Apply Settings")

    chk_det_size.select(fn=on_option_changed)

    # Settings
    for s in settings_controls:
        s.select(fn=on_settings_changed)
    max_threads.input(fn=lambda a, b='max_threads': on_settings_changed_misc(a, b), inputs=[max_threads])
    memory_limit.input(fn=lambda a, b='memory_limit': on_settings_changed_misc(a, b), inputs=[memory_limit])
    video_quality.input(fn=lambda a, b='video_quality': on_settings_changed_misc(a, b), inputs=[video_quality])

    # button_clean_temp.click(fn=clean_temp, outputs=[bt_srcfiles, input_faces, target_faces, bt_destfiles])
    button_clean_temp.click(fn=clean_temp)
    button_apply_settings.click(apply_settings, inputs=[themes, input_server_name, input_server_port, output_template])
    button_apply_restart.click(restart)


def on_option_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals, attribname):
            setattr(roop.globals, attribname, evt.value)
            return
    raise gr.Error(f'Configuración no controlada para {evt.target}')


def on_settings_changed_misc(new_val, attribname):
    if hasattr(roop.globals.CFG, attribname):
        setattr(roop.globals.CFG, attribname, new_val)
    else:
        print("No encuentra el atributo!!")


def on_settings_changed(evt: gr.SelectData):
    attribname = evt.target.elem_id
    if isinstance(evt.target, gr.Checkbox):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.selected)
            return
    elif isinstance(evt.target, gr.Dropdown):
        if hasattr(roop.globals.CFG, attribname):
            setattr(roop.globals.CFG, attribname, evt.value)
            return

    raise gr.Error(f'Configuración no controlada para {evt.target}')


def clean_temp():
    from ui.main import prepare_environment

    if not roop.globals.CFG.use_os_temp_folder:
        shutil.rmtree(os.environ["TEMP"])
    prepare_environment()

    ui.globals.ui_input_thumbs.clear()
    roop.globals.INPUT_FACESETS.clear()
    roop.globals.TARGET_FACES.clear()
    ui.globals.ui_target_thumbs = []
    gr.Info('Archivos temporales eliminados')
    return None, None, None, None


def apply_settings(themes, input_server_name, input_server_port, output_template):
    from ui.main import show_msg

    roop.globals.CFG.selected_theme = themes
    roop.globals.CFG.server_name = input_server_name
    roop.globals.CFG.server_port = input_server_port
    roop.globals.CFG.output_template = output_template
    roop.globals.CFG.save()
    show_msg('Confiruración aplicada. Reinicie el servidor para que los cambios surtan efecto.')


def restart():
    ui.globals.ui_restart_server = True
