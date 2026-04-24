import gradio as gr
import ui.tabs.animate.ui as animate_ui
import ui.tabs.animate.events as animate_events

def animate_photo_tab():
    """Pestaña Animate Image - Versión Grok Imagine (Modular)"""
    
    # 1. Construir la UI
    ui_components = animate_ui.build_animate_ui()
    
    # 2. Conectar Eventos
    animate_events.wire_animate_events(ui_components)
    
    print("[AnimatePhoto] Pestaña cargada en modo modular (Grok Imagine v1.0)")
    
    return ui_components
