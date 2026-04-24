import gradio as gr
import roop.globals
import ui.tabs.faceswap.ui as faceswap_ui
import ui.tabs.faceswap.events as faceswap_events
import ui.tabs.faceswap.logic as logic
import ui.tabs.faceswap.state as state

def faceswap_tab():
    """
    Entry point modularizado para la pestaña FaceSwap.
    Implementa las mejoras de UX: Selección Directa, Fix OnClick y Smart Tracking.
    """
    
    # 1. Cargar configuración inicial
    logic.load_folder_history()
    logic.cleanup_temp_files()
    
    # 2. Configurar parámetros de calidad por defecto (High Fidelity)
    roop.globals.blend_ratio = 1.0
    roop.globals.distance_threshold = 0.6
    roop.globals.face_swap_mode = 'selected_faces' # Por defecto
    
    # 3. Construir la UI
    ui_components = faceswap_ui.build_faceswap_ui()
    
    # 4. Conectar Eventos
    faceswap_events.wire_events(ui_components)
    
    print("[FaceSwap] Pestaña cargada en modo modular v2.0 (Fix OnClick + Smart Tracking)")
    
    return ui_components
