#!/usr/bin/env python3
"""
Descarga modelos de ControlNet automaticamente
Ejecutar: python tools/download_controlnet_models.py
"""

import os
import sys
import urllib.request
import ssl

# Desactivar verificacion SSL para descargas
ssl._create_default_https_context = ssl._create_unverified_context

# Ruta de destino
BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                         "ui", "tob", "stable-diffusion-webui", 
                         "extensions", "sd-webui-controlnet", "models")

# Modelos esenciales de ControlNet v1.1
MODELS = {
    "control_v11p_sd15_canny.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_canny.pth",
    "control_v11p_sd15_openpose.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
    "control_v11f1p_sd15_depth.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth",
    "control_v11p_sd15_lineart.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_lineart.pth",
    "control_v11p_sd15_scribble.pth": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_scribble.pth",
}

def download_file(url, dest_path, desc):
    """Descarga un archivo con progreso"""
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path) / (1024**3)
        print(f"  [YA EXISTE] {desc} ({size:.2f} GB)")
        return True
    
    print(f"  [DESCARGANDO] {desc}...")
    print(f"      Desde: {url}")
    print(f"      Hasta: {dest_path}")
    
    try:
        urllib.request.urlretrieve(url, dest_path)
        size = os.path.getsize(dest_path) / (1024**3)
        print(f"  [OK] {desc} descargado ({size:.2f} GB)")
        return True
    except Exception as e:
        print(f"  [ERROR] {desc}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def main():
    print("="*60)
    print("  DESCARGADOR DE MODELOS CONTROLNET")
    print("="*60)
    
    # Crear directorio si no existe
    os.makedirs(BASE_PATH, exist_ok=True)
    print(f"\nDirectorio destino: {BASE_PATH}\n")
    
    success_count = 0
    total_count = len(MODELS)
    
    for filename, url in MODELS.items():
        dest_path = os.path.join(BASE_PATH, filename)
        if download_file(url, dest_path, filename):
            success_count += 1
        print()
    
    print("="*60)
    print(f"  Descargados: {success_count}/{total_count} modelos")
    print("="*60)
    
    if success_count == total_count:
        print("\n[TODOS LOS MODELOS DESCARGADOS]")
        print("Reinicia SD WebUI para usar ControlNet")
    else:
        print("\n[ALGUNOS MODELOS FALTAN]")
        print("Puedes descargarlos manualmente desde:")
        print("https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
