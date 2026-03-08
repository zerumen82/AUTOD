# -*- coding: utf-8 -*-
"""
Test completo de ClothingSegmenter
"""
import sys
import os

# Aplicar parches de run.py antes de importar cualquier otra cosa
original_remove = os.remove

def resilient_remove(path, *args, **kwargs):
    import time
    import stat
    for i in range(20):
        try:
            try: os.chmod(path, stat.S_IWRITE)
            except: pass
            return original_remove(path)
        except (PermissionError, OSError):
            if i == 19: raise
            time.sleep(0.1)

def resilient_unlink(self, missing_ok=False):
    import time
    import stat
    from pathlib import Path
    for i in range(20):
        try:
            try: os.chmod(str(self), stat.S_IWRITE)
            except: pass
            return Path._original_unlink(self, missing_ok=missing_ok)
        except (PermissionError, OSError):
            if i == 19: raise
            time.sleep(0.1)

os.remove = resilient_remove
from pathlib import Path
Path._original_unlink = Path.unlink
Path.unlink = resilient_unlink

print("=" * 60)
print("TEST COMPLETO DE CLOTHINGSEGMENTER")
print("=" * 60)

try:
    print("\n[1] Importando ClothingSegmenter...")
    from roop.img_editor.clothing_segmenter import ClothingSegmenter, get_clothing_segmenter
    print("    OK - ClothingSegmenter importado")
    
    print("\n[2] Creando instancia...")
    segmenter = get_clothing_segmenter()
    print("    OK - Instancia creada")
    
    print("\n[3] Cargando modelo...")
    success, msg = segmenter.load()
    if not success:
        print(f"    ERROR: {msg}")
        sys.exit(1)
    print(f"    OK - {msg}")
    
    print("\n[4] Creando imagen de prueba...")
    from PIL import Image
    import numpy as np
    
    # Crear imagen con "ropa" (azul) y "piel" (color piel)
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    
    # Zona de "ropa" - mitad superior en azul
    img_array[:128, :, 0] = 50   # R bajo
    img_array[:128, :, 1] = 100  # G medio
    img_array[:128, :, 2] = 200  # B alto (azul)
    
    # Zona de "piel" - mitad inferior en color piel
    img_array[128:, :, 0] = 255  # R alto
    img_array[128:, :, 1] = 200  # G alto
    img_array[128:, :, 2] = 180  # B medio (piel)
    
    test_img = Image.fromarray(img_array, mode="RGB")
    print("    OK - Imagen creada (256x256)")
    
    print("\n[5] Segmentando ropa...")
    mask_img, mask_array = segmenter.segment_clothing(
        test_img,
        threshold=0.5,
        include_skin_exclusion=True
    )
    
    pixels_ropa = mask_array.sum() / 255
    total_pixels = 256 * 256
    porcentaje = (pixels_ropa / total_pixels) * 100
    
    print(f"    OK - Máscara generada")
    print(f"    Píxeles de ropa: {pixels_ropa:.0f} ({porcentaje:.1f}%)")
    
    if pixels_ropa > 0:
        print("\n" + "=" * 60)
        print("SUCCESS - ClothingSegmenter funciona correctamente!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("WARNING - Máscara vacía, revisar detección")
        print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR - Test falló:")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    sys.exit(1)