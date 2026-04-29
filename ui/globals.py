import os
import subprocess

ui_restart_server = False

SELECTION_FACES_DATA = None
ui_SELECTED_INPUT_FACE_INDEX = 0

ui_selected_enhancer = None
ui_blend_ratio = None
ui_input_thumbs = []
ui_target_thumbs = []
ui_camera_frame = None
ui_use_enhancer = None
ui_blend_mode = None

def open_output_folder():
    """Abre la carpeta de salida en el explorador de archivos."""
    import roop.globals
    output_path = getattr(roop.globals, 'output_path', 'output')
    
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        
    output_path = os.path.abspath(output_path)
    print(f"[UI] Abriendo carpeta: {output_path}")
    
    try:
        if os.name == 'nt':  # Windows
            os.startfile(output_path)
        elif os.name == 'posix':  # macOS o Linux
            subprocess.Popen(['open' if os.uname().sysname == 'Darwin' else 'xdg-open', output_path])
    except Exception as e:
        print(f"[ERROR] No se pudo abrir la carpeta: {e}")
