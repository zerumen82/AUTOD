import gradio as gr
import roop.globals
import ui.globals
import ui.tabs.faceswap.logic as logic

def build_faceswap_ui():
    """Define la estructura visual de la pestaña FaceSwap"""
    
    gr.HTML("""
        <style>
            .origin-gallery { border: 2px solid #3b82f6 !important; border-radius: 8px; }
            .target-gallery { border: 2px solid #10b981 !important; border-radius: 8px; }
            .face-counter {
                background: #1e293b;
                color: #3b82f6;
                padding: 8px;
                border-radius: 6px;
                text-align: center;
                font-weight: bold;
                margin-bottom: 8px;
                border: 1px solid #334155;
            }
            .dynamic-panel {
                background: rgba(59, 130, 246, 0.05);
                border: 1px dashed #3b82f6;
                padding: 12px !important;
                border-radius: 10px;
                margin-top: 10px;
            }
        </style>
    """)

    with gr.Row():
        # COLUMNA IZQUIERDA: Selección de Caras
        with gr.Column(scale=2):
            with gr.Row():
                # Origen
                with gr.Column():
                    gr.HTML('<div class="face-counter">👤 CARAS DE ORIGEN</div>')
                    input_faces = gr.Gallery(
                        label="Origen", columns=4, height=350,
                        elem_classes=["origin-gallery"], interactive=True,
                        allow_preview=False, preview=False
                    )
                    input_page_info = gr.Markdown("📄 Página 1 de 1 (0 caras)")
                    with gr.Row():
                        bt_input_prev = gr.Button("⬅", size="sm")
                        bt_input_next = gr.Button("➡", size="sm")
                    bt_remove_selected_input_face = gr.Button("🗑 Quitar seleccionada", size="sm", variant="secondary")

                # Destino
                with gr.Column():
                    gr.HTML('<div class="face-counter">🎯 CARAS DE DESTINO</div>')
                    target_faces = gr.Gallery(
                        label="Destino", columns=4, height=350,
                        elem_classes=["target-gallery"], interactive=True,
                        allow_preview=False, preview=False
                    )
                    target_page_info = gr.Markdown("📄 Página 1 de 1 (0 caras)")
                    with gr.Row():
                        bt_target_prev = gr.Button("⬅", size="sm")
                        bt_target_next = gr.Button("➡", size="sm")
                    bt_remove_selected_target_face = gr.Button("🗑 Quitar seleccionada", size="sm", variant="secondary")

            with gr.Row(variant="panel"):
                with gr.Column():
                    bt_srcfiles = gr.Files(label="📁 1. Caras Origen", file_count="multiple", file_types=["image", ".fsz"], height=120)
                with gr.Column():
                    bt_destfiles = gr.Files(label="📁 2. Archivos Destino", file_count="multiple", file_types=["image", "video"], height=120)

        # COLUMNA DERECHA: Preview y Control
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        bt_prev_frame = gr.Button("⏮ Frame", size="sm")
                        bt_next_frame = gr.Button("Frame ⏭", size="sm")
                with gr.Column(scale=4):
                    preview_frame_num = gr.Slider(1, 1, value=1, label="🎬 Timeline Video", step=1.0)
            
            previewimage = gr.Image(label="Vista Previa", height=500, type="filepath")
            
            with gr.Row():
                bt_prev_file = gr.Button("📂 Archivo Anterior", size="sm")
                bt_next_file = gr.Button("Archivo Siguiente 📂", size="sm")

            # Panel de Selección Dinámica (Aparece al detectar caras en un frame)
            with gr.Row(visible=False, variant="panel", elem_classes=["dynamic-panel"]) as dynamic_face_selection:
                with gr.Column():
                    gr.Markdown("### ✨ Caras detectadas en este frame")
                    gr.Markdown("Haz clic en una cara para añadirla a Destino.")
                    face_selection = gr.Gallery(label="", columns=8, height=200, interactive=True)
                    with gr.Row():
                        face_selector_slider = gr.Slider(1, 1, value=1, step=1, label="O usa el slider", visible=False)
                        bt_use_selected_face = gr.Button("✅ Confirmar selección", variant="primary", visible=False)

            with gr.Row():
                selected_face_detection = gr.Dropdown(
                    choices=["First found", "Selected faces", "Selected faces frame", "All faces"],
                    value="Selected faces", label="🎭 Modo de Intercambio"
                )
                fake_preview = gr.Checkbox(label="Previsualizar Swap", value=False)

            with gr.Row():
                bt_start = gr.Button("▶️ INICIAR PROCESAMIENTO", variant="primary", size="lg")
                bt_stop = gr.Button("⏹️ DETENER", variant="stop", size="lg", interactive=False)
            
            metrics_display = gr.HTML(logic.get_metrics_html(0, 0, 0, "00:00", "--:--", "Listo"))

    return {
        "input_faces": input_faces, "target_faces": target_faces, "bt_srcfiles": bt_srcfiles, "bt_destfiles": bt_destfiles,
        "previewimage": previewimage, "preview_frame_num": preview_frame_num, "bt_prev_frame": bt_prev_frame, "bt_next_frame": bt_next_frame,
        "bt_prev_file": bt_prev_file, "bt_next_file": bt_next_file, "dynamic_face_selection": dynamic_face_selection,
        "face_selection": face_selection, "face_selector_slider": face_selector_slider, "bt_use_selected_face": bt_use_selected_face,
        "selected_face_detection": selected_face_detection, "fake_preview": fake_preview, "bt_start": bt_start, "bt_stop": bt_stop, 
        "metrics_display": metrics_display, "input_page_info": input_page_info, "target_page_info": target_page_info,
        "bt_input_prev": bt_input_prev, "bt_input_next": bt_input_next, "bt_target_prev": bt_target_prev, "bt_target_next": bt_target_next,
        "bt_remove_selected_input_face": bt_remove_selected_input_face, "bt_remove_selected_target_face": bt_remove_selected_target_face
    }
