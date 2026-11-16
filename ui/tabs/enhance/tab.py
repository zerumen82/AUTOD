from .ui import build_enhance_ui
from .events import wire_events


def create_enhance_tab():
    """Crea la pestaña de mejora de fotos local."""
    ui_components = build_enhance_ui()
    wire_events(ui_components)
    print("[Enhance] Pestaña de mejora de fotos local cargada")
    return ui_components