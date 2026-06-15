import os
import sys

def get_root_path():
    if getattr(sys, 'frozen', False):
        # Running from EXE
        exe_dir = os.path.dirname(sys.executable)
        # If the EXE is in dist/AutoAuto/, we might want the parent of 'dist'
        # But usually, the user should have the models folder next to the EXE
        # OR we assume the EXE is launched from the project root.
        
        # Check if we are in a 'dist/AutoAuto' subfolder
        if "dist" in exe_dir.lower():
            # Try to go up to the real project root
            # dist/AutoAuto -> dist -> project_root
            return os.path.abspath(os.path.join(exe_dir, "..", ".."))
        return exe_dir
    else:
        # Running from source
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
