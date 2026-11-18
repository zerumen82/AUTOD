import gradio as gr
import ui.globals


camera_frame = None

def livecam_tab():
    with gr.Tab("🎥 Camara en Vivo"):
        with gr.Row(variant='panel'):
            gr.Markdown("""
                        Esta función le permitirá usar su cámara web física y aplicar las caras seleccionadas a la transmisión. 
                        También puedes reenviar la transmisión a una cámara virtual, que se puede utilizar en videollamadas o software de transmisión.<br />
                        Se admiten: v4l2loopback (linux), cámara virtual OBS (macOS/Windows) y unitycapture (Windows).<br />
                        **Tenga en cuenta:** para cambiar la cara o cualquier otra configuración, debe detener y reiniciar una cámara en vivo en funcionamiento.
            """)

        with gr.Row(variant='panel'):
            with gr.Column():
                bt_start = gr.Button("▶ Start", variant='primary')
            with gr.Column():
                bt_stop = gr.Button("⏹ Stop", variant='secondary', interactive=False)
            with gr.Column():
                camera_num = gr.Slider(0, 8, value=0, label="Numero de Camara", step=1.0, interactive=True)
                cb_obs = gr.Checkbox(label="Reenviar transmisión a la cámara virtual", interactive=True)
            with gr.Column():
                dd_reso = gr.Dropdown(choices=["640x480","1280x720", "1920x1080"], value="1280x720", label="Resolución de cámara Fake", interactive=True)

        with gr.Row():
            fake_cam_image = gr.Image(label='Salida de la camara Fake', interactive=False)

    start_event = bt_start.click(fn=start_cam,  inputs=[cb_obs, camera_num, dd_reso], outputs=[bt_start, bt_stop, fake_cam_image])
    bt_stop.click(fn=stop_swap, cancels=[start_event], outputs=[bt_start, bt_stop], queue=False)


def start_cam(stream_to_obs, cam, reso):
    from roop.virtualcam import start_virtual_cam
    from roop.utilities import convert_to_gradio

    start_virtual_cam(stream_to_obs, cam, reso)
    # Los valores de enhancer y blend se leen de roop.globals (configurados en FaceSwap).
    while True:
        yield gr.Button(interactive=False), gr.Button(interactive=True), convert_to_gradio(ui.globals.ui_camera_frame)
        

def stop_swap():
    from roop.virtualcam import stop_virtual_cam
    stop_virtual_cam()
    return gr.Button(interactive=True), gr.Button(interactive=False)
    



