import ui.tabs.gen.ui as gen_ui
import ui.tabs.gen.events as gen_events


def create_gen_tab():
    ui_components = gen_ui.build_gen_ui()
    gen_events.wire_events(ui_components)
    print("[GenTab] Pestaña Generate cargada en modo modular")
    return ui_components
