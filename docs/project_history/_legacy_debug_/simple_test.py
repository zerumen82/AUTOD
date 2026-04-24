import sys
import os
sys.path.insert(0, os.path.abspath('ui/tob/ComfyUI'))

import sys
import io
import traceback

old_stdout = sys.stdout
old_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

try:
    print("Starting test...")
    import comfy
    print(f"Comfy imported: {comfy}")
    
    print("Trying to import model_base...")
    import comfy.model_base
    print(f"model_base imported: {comfy.model_base}")
    
    print("Checking SV3D_u in model_base...")
    if hasattr(comfy.model_base, 'SV3D_u'):
        print("✓ SV3D_u exists in model_base")
    else:
        print("✗ SV3D_u does NOT exist in model_base")
    
    print("Checking SVD_img2vid in model_base...")
    if hasattr(comfy.model_base, 'SVD_img2vid'):
        print("✓ SVD_img2vid exists in model_base")
    else:
        print("✗ SVD_img2vid does NOT exist in model_base")
        
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
    print("\nStack trace:")
    print(traceback.format_exc())

print("\n=== Captured stdout ===")
print(sys.stdout.getvalue())
print("\n=== Captured stderr ===")
print(sys.stderr.getvalue())

sys.stdout = old_stdout
sys.stderr = old_stderr
