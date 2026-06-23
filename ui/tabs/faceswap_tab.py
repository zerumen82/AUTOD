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
    
    # 2. Defaults óptimos: calidad GFPGAN + máximo parecido source
    logic.apply_optimal_faceswap_defaults()
    
    # 3. Construir la UI
    ui_components = faceswap_ui.build_faceswap_ui()
    
    # 4. Conectar Eventos
    faceswap_events.wire_events(ui_components)
    
    print("[FaceSwap] Pestaña cargada en modo modular v2.0 (Fix OnClick + Smart Tracking)")
    
    return ui_components
