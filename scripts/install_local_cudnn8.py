#!/usr/bin/env python3
"""
Extrae un zip de cuDNN (descargado manualmente desde NVIDIA) y coloca las DLLs necesarias en `dll/cudnn8/`.
Uso: python scripts/install_local_cudnn8.py /ruta/a/cudnn-windows-x86_64-8.x.x.zip
"""
import sys
import os
import zipfile
import shutil

if len(sys.argv) < 2:
    print("Uso: python scripts/install_local_cudnn8.py <ruta_al_zip_de_cudnn>")
    sys.exit(1)

zip_path = sys.argv[1]
if not os.path.exists(zip_path):
    print(f"Archivo no encontrado: {zip_path}")
    sys.exit(1)

base_dir = os.path.dirname(os.path.dirname(__file__))
dll_dest = os.path.join(base_dir, 'dll', 'cudnn8')
if not os.path.exists(dll_dest):
    os.makedirs(dll_dest, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as z:
    members = z.namelist()
    # Extraer únicamente DLLs de la carpeta bin/ y archivos relevantes
    extracted = 0
    for m in members:
        parts = m.split('/')
        if len(parts) >= 2 and parts[0].lower().startswith('cuda') and parts[1].lower() == 'bin':
            for f in z.namelist():
                # handled below
                pass
    # alternativa: recorrer y copiar archivos que terminen en .dll o cudnn.h
    for m in members:
        if m.endswith('/'):
            continue
        name = os.path.basename(m)
        if name.lower().endswith('.dll') or name.lower() == 'cudnn.h' or name.lower().endswith('.lib'):
            dest_path = os.path.join(dll_dest, name)
            with z.open(m) as src, open(dest_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1

print(f"Se han extraído {extracted} archivos a {dll_dest}")
print("Recomendación: reinicia la sesión o el sistema para que Windows reconozca las DLLs si aplicara.")
print("Luego ejecuta `python verificar_cudnn_sistema.py` para verificar.")
