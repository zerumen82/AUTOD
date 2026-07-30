import ui.tabs.img_editor.ui as img_editor_ui
import ui.tabs.img_editor.events as img_editor_events


def create_img_editor_tab():
    ui_components = img_editor_ui.build_img_editor_ui()
    img_editor_events.wire_events(ui_components)
    print("[ImgEditor] Pestaña Image Editor cargada en modo modular (autoregresivo, sin preservación)")
    return ui_components
