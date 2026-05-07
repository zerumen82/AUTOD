import os
import subprocess
import sys
import builtins
from datetime import datetime

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

# Console buffer for export
console_buffer = []
console_buffer_max_size = 10000  # Maximum number of lines to keep

# Store original print
_original_print = builtins.print

def custom_print(*args, **kwargs):
    """Custom print that captures output to buffer"""
    # Call original print
    _original_print(*args, **kwargs)
    
    # Capture to buffer
    try:
        message = " ".join(str(arg) for arg in args)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        console_buffer.append(f"[{timestamp}] {message}")
        
        # Limit buffer size
        if len(console_buffer) > console_buffer_max_size:
            console_buffer.pop(0)
    except:
        pass

def start_capturing_prints():
    """Start capturing print output"""
    builtins.print = custom_print

def stop_capturing_prints():
    """Stop capturing and restore original print"""
    builtins.print = _original_print

def get_console_text():
    """Get all captured console text"""
    return "\n".join(console_buffer)

def clear_console_buffer():
    """Clear the console buffer"""
    global console_buffer
    console_buffer.clear()

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
