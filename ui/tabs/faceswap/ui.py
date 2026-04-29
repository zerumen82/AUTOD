import gradio as gr
import roop.globals
import ui.globals
import ui.tabs.faceswap.logic as logic

def build_faceswap_ui():
    """Define la estructura visual de la pestaña FaceSwap"""
    
    gr.HTML("""
        <style>
            .origin-gallery { border: 2px solid #3b82f6 !important; border-radius: 12px; background: rgba(59, 130, 246, 0.03); }
            .target-gallery { border: 2px solid #10b981 !important; border-radius: 12px; background: rgba(16, 185, 129, 0.03); }
            .face-counter {
                background: linear-gradient(90deg, #1e293b, #334155);
                color: #3b82f6;
                padding: 10px;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                margin-bottom: 10px;
                border: 1px solid #475569;
                font-size: 14px;
                letter-spacing: 1px;
            }
            .dynamic-panel {
                background: linear-gradient(180deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
                border: 1px solid #8b5cf6;
                padding: 15px !important;
                border-radius: 15px;
                margin-top: 15px;
                box-shadow: 0 4px 15px rgba(139, 92, 246, 0.1);
            }
            .small-face-gallery { min-height: auto !important; gap: 0 !important; }
            .small-face-gallery > div { gap: 5px !important; padding: 0 !important; }
            .small-face-gallery .grid-container { margin: 0 !important; padding: 0 !important; gap: 5px !important; }
            .small-face-gallery .grid-wrap { gap: 5px !important; padding: 0 !important; margin: 0 !important; }
            .small-face-gallery button.thumbnail-item { width: 120px !important; height: 120px !important; flex: none !important; margin: 0 !important; padding: 0 !important; border-radius: 8px !important; }
            .small-face-gallery img { width: 120px !important; height: 120px !important; object-fit: cover !important; border-radius: 8px !important; }
            .small-face-gallery .gallery-wrapper { padding: 0 !important; margin: 0 !important; }
            .frame-btn { min-width: 45px !important; font-weight: bold !important; }
            .grok-start-btn { 
                background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; 
                color: white !important; 
                font-size: 18px !important;
                height: 60px !important;
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
                        label="Origen", columns=3, height=400,
                        elem_classes=["origin-gallery"], interactive=True,
                        allow_preview=False, preview=False,
                        visible=True, object_fit="cover"
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
                        label="Destino", columns=3, height=400,
                        elem_classes=["target-gallery"], interactive=True,
                        allow_preview=False, preview=False,
                        visible=True, object_fit="cover"
                    )
                    target_page_info = gr.Markdown("📄 Página 1 de 1 (0 caras)")
                    with gr.Row():
                        bt_target_prev = gr.Button("⬅", size="sm")
                        bt_target_next = gr.Button("➡", size="sm")
                    with gr.Row():
                        bt_remove_selected_target_face = gr.Button("🗑 Quitar", size="sm", variant="secondary")
                        bt_clear_all_target = gr.Button("🧹 Todas", size="sm", variant="stop")

            with gr.Row(variant="panel"):
                with gr.Column():
                    bt_srcfiles = gr.Files(label="📁 1. Caras Origen (Fotos)", file_count="multiple", file_types=["image", ".fsz"], height=100)
                with gr.Column():
                    bt_destfiles = gr.Files(label="📁 2. Destino (Fotos o Video)", file_count="multiple", file_types=["image", "video"], height=100)

        # COLUMNA DERECHA: Preview y Control
        with gr.Column(scale=3):
            with gr.Row(variant="compact"):
                with gr.Column(scale=2):
                    with gr.Row():
                        bt_jump_back_100 = gr.Button("-100", size="sm", elem_classes=["frame-btn"])
                        bt_jump_back_10 = gr.Button("-10", size="sm", elem_classes=["frame-btn"])
                        bt_prev_frame = gr.Button("◀", size="sm", elem_classes=["frame-btn"])
                        bt_next_frame = gr.Button("▶", size="sm", elem_classes=["frame-btn"])
                        bt_jump_fwd_10 = gr.Button("+10", size="sm", elem_classes=["frame-btn"])
                        bt_jump_fwd_100 = gr.Button("+100", size="sm", elem_classes=["frame-btn"])
                with gr.Column(scale=3):
                    preview_frame_num = gr.Slider(1, 1, value=1, label="🎬 Timeline del Video", step=1.0)
            
            previewimage = gr.Image(label="Vista Previa", height=480, type="filepath")
            
            with gr.Row():
                bt_prev_file = gr.Button("📂 Anterior", size="sm")
                bt_use_face_from_preview = gr.Button("🔍 BUSCAR CARAS EN ESTE FRAME", variant="primary", scale=2)
                bt_next_file = gr.Button("Siguiente 📂", size="sm")

            # Panel de Selección Dinámica
            with gr.Row(visible=False, variant="panel", elem_classes=["dynamic-panel"]) as dynamic_face_selection:
                with gr.Column():
                    gr.Markdown("### ✨ Caras detectadas en este frame")
                    gr.Markdown("Haz clic en una cara para añadirla a Destino.")
                    face_selection = gr.Gallery(
                        label="", columns=4, 
                        interactive=True, allow_preview=False, 
                        preview=False, object_fit="cover",
                        elem_classes=["small-face-gallery"]
                    )
                    with gr.Row(visible=False): # Ocultamos el slider viejo
                        face_selector_slider = gr.Slider(minimum=1, maximum=1, value=1, step=1, label="Selecciona el número de cara")
                        bt_use_selected_face = gr.Button("✅ AÑADIR")

            with gr.Row():
                selected_face_detection = gr.Dropdown(
                    choices=["First found", "Selected faces", "Selected faces frame", "All faces"],
                    value="Selected faces frame", label="🎭 Modo de Intercambio"
                )
                fake_preview = gr.Checkbox(label="Previsualizar Swap (Realtime)", value=False)

            with gr.Accordion("⚙️ Ajustes Expertos (Fidelidad)", open=False):
                with gr.Row():
                    autorotate = gr.Checkbox(label="🔄 Auto-Rotar (Mejora Perfiles/Inclinadas)", value=True)
                    smoothing = gr.Checkbox(label="🛡️ Suavizado Temporal (Anti-Parpadeo)", value=True)
                with gr.Row():
                    face_distance = gr.Slider(0.01, 1.0, value=0.30, step=0.01, label="📏 Umbral de Similitud (Bajo = Más Estricto)")
                    blend_ratio = gr.Slider(0.0, 1.0, value=1.0, step=0.01, label="🎨 Mezcla de Piel (1.0 = Máxima Calidad)")
                with gr.Row():
                    enhancer = gr.Dropdown(
                        choices=["None", "CodeFormer", "GFPGAN", "Restoreformer++"],
                        value="CodeFormer", label="✨ Enhancer de Calidad (CodeFormer = Máxima Fidelidad)"
                    )

            with gr.Row():
                bt_start = gr.Button("🚀 INICIAR PROCESAMIENTO", variant="primary", elem_classes=["grok-start-btn"], scale=2)
                bt_stop = gr.Button("⏹️", variant="stop", interactive=False)
                bt_open_output = gr.Button("📂", variant="secondary")
            
            metrics_display = gr.HTML(logic.get_metrics_html(0, 0, 0, "00:00", "--:--", "Listo"))

    return {
        "input_faces": input_faces, "target_faces": target_faces, "bt_srcfiles": bt_srcfiles, "bt_destfiles": bt_destfiles,
        "previewimage": previewimage, "preview_frame_num": preview_frame_num, 
        "bt_prev_frame": bt_prev_frame, "bt_next_frame": bt_next_frame,
        "bt_jump_back_10": bt_jump_back_10, "bt_jump_fwd_10": bt_jump_fwd_10,
        "bt_jump_back_100": bt_jump_back_100, "bt_jump_fwd_100": bt_jump_fwd_100,
        "bt_prev_file": bt_prev_file, "bt_next_file": bt_next_file, "bt_use_face_from_preview": bt_use_face_from_preview,
        "dynamic_face_selection": dynamic_face_selection,
        "face_selection": face_selection, "face_selector_slider": face_selector_slider, "bt_use_selected_face": bt_use_selected_face,
        "selected_face_detection": selected_face_detection, "fake_preview": fake_preview, 
        "autorotate": autorotate, "smoothing": smoothing, "face_distance": face_distance, "blend_ratio": blend_ratio, "enhancer": enhancer,
        "bt_start": bt_start, "bt_stop": bt_stop, "bt_open_output": bt_open_output,
        "metrics_display": metrics_display, "input_page_info": input_page_info, "target_page_info": target_page_info,
        "bt_input_prev": bt_input_prev, "bt_input_next": bt_input_next, "bt_target_prev": bt_target_prev, "bt_target_next": bt_target_next,
        "bt_remove_selected_input_face": bt_remove_selected_input_face, "bt_remove_selected_target_face": bt_remove_selected_target_face, "bt_clear_all_target": bt_clear_all_target
    }
