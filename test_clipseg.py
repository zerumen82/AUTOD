# -*- coding: utf-8 -*-
"""
Test de carga de CLIPSeg
"""
import sys
import os

# Aplicar parches de run.py antes de importar cualquier otra cosa
original_remove = os.remove

def resilient_remove(path, *args, **kwargs):
    """Parche para os.remove - solo acepta un argumento posicional."""
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
    """Parche para Path.unlink - usa self como path."""
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
print("TEST DE CARGA DE CLIPSEG")
print("=" * 60)

try:
    print("\n[1] Importando transformers...")
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
    print("    OK - transformers importado")
    
    print("\n[2] Cargando CLIPSegProcessor...")
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    print("    OK - Processor cargado")
    
    print("\n[3] Cargando CLIPSegForImageSegmentation...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = model.to(device)
    model.eval()
    print(f"    OK - Modelo cargado en {device}")
    
    print("\n[4] Probando segmentación...")
    from PIL import Image
    import numpy as np
    
    # Crear imagen de prueba
    test_img = Image.new("RGB", (256, 256), color=(100, 100, 200))
    
    inputs = processor(
        text=["clothing"],
        images=[test_img],
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    print(f"    OK - Logits shape: {logits.shape}")
    
    print("\n" + "=" * 60)
    print("SUCCESS - CLIPSeg funciona correctamente!")
    print("=" * 60)
    
except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR - CLIPSeg falló:")
    print("=" * 60)
    import traceback
    traceback.print_exc()
    sys.exit(1)