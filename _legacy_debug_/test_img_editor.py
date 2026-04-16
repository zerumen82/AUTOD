#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test simple del Image Editor
"""

import sys
sys.path.insert(0, '.')

print("=" * 60)
print("  TEST IMAGE EDITOR")
print("=" * 60)

# 1. Test import
print("\n[1] Test import...")
try:
    from ui.tabs.img_editor_tab import create_img_editor_tab, img_editor_worker, start_img_editor_thread
    print("✅ Import OK")
except Exception as e:
    print(f"❌ Import FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 2. Test manager
print("\n[2] Test manager...")
try:
    from roop.img_editor.img_editor_manager import get_img_editor_manager
    manager = get_img_editor_manager()
    print("✅ Manager OK")
except Exception as e:
    print(f"❌ Manager FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 3. Test thread start
print("\n[3] Test thread start...")
try:
    start_img_editor_thread()
    print("✅ Thread START OK")
except Exception as e:
    print(f"❌ Thread START FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. Test queue
print("\n[4] Test queue...")
try:
    from ui.tabs.img_editor_tab import img_editor_queue
    from PIL import Image
    import numpy as np
    
    # Crear imagen test
    test_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    # Enviar tarea SIMPLE
    task = (
        test_img,  # image
        "test prompt",  # prompt
        1,  # num_var
        "fast",  # quality
        False,  # face_preserve
        False,  # auto_enhance
        {}  # ref_metadata
    )
    
    img_editor_queue.put(task)
    print(f"✅ Queue PUT OK (len={len(task)})")
    
except Exception as e:
    print(f"❌ Queue FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 5. Esperar resultado
print("\n[5] Esperando resultado (10s)...")
import time
time.sleep(10)

from ui.tabs.img_editor_tab import img_editor_result_queue
if not img_editor_result_queue.empty():
    result = img_editor_result_queue.get_nowait()
    print(f"✅ Resultado recibido: {result.get('message', 'N/A')}")
else:
    print("⚠️ NO hay resultado en la cola (puede estar procesando)")

print("\n" + "=" * 60)
print("  TEST COMPLETADO")
print("=" * 60)
print("\nAhora ejecuta: python run.py")
print("Y prueba el Image Editor desde la UI")
