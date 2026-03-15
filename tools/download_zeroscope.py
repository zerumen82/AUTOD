"""
Download Zeroscope V2 XL models for ComfyUI

Usage:
    python tools/download_zeroscope.py

El modelo se descarga en:
    ui/tob/ComfyUI/models/diffusion_models/zeroscope_v2_XL/UNET/
"""

import os
import sys
import urllib.request
import json

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "tob", "ComfyUI", "models")

def get_file_size(url):
    """Obtiene el tamaño del archivo sin descargarlo"""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            return int(response.headers.get('Content-Length', 0))
    except:
        return 0

def download_file(url: str, dest_path: str, desc: str = "") -> bool:
    """Descarga un archivo con barra de progreso"""
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path) / (1024*1024)
        print(f"   [YA EXISTE] {desc} ({size:.1f} MB)")
        return True
    
    print(f"   [DESCARGANDO] {desc}...")
    print(f"   URL: {url}")
    
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        # Descargar con progreso
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\r      {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n   [OK] {desc}")
        return True
        
    except Exception as e:
        print(f"\n   [ERROR] {e}")
        return False


def download_zeroscope():
    """Descarga Zeroscope desde HuggingFace"""
    
    print("\n" + "="*60)
    print("DESCARGANDO ZEROSCOPE V2 XL")
    print("="*60)
    
    # Ruta donde se guardará
    zeroscope_path = os.path.join(BASE_PATH, "diffusion_models", "zeroscope_v2_XL", "UNET")
    os.makedirs(zeroscope_path, exist_ok=True)
    
    dest_file = os.path.join(zeroscope_path, "diffusion_pytorch_model.safetensors")
    
    # Fuentes de descarga - intentamos varias opciones
    download_urls = [
        # HuggingFace - cerspense/zeroscope_v2_XL (formato binario, no safetensors)
        "https://huggingface.co/cerspense/zeroscope_v2_XL/resolve/main/unet/diffusion_pytorch_model.bin",
    ]
    
    success = False
    
    for url in download_urls:
        print(f"\n[INFO] Intentando descarga desde...")
        print(f"   {url}")
        
        # Verificar si el archivo existe primero
        try:
            size = get_file_size(url)
            if size > 0:
                print(f"   Tamano: {size / (1024*1024):.1f} MB")
            else:
                print(f"   (No se pudo obtener tamano)")
        except:
            pass
        
        if download_file(url, dest_file, "Zeroscope V2 XL UNET"):
            success = True
            break
        else:
            print("   Reintentando con siguiente fuente...")
    
    if success:
        print("\n" + "="*60)
        print("[OK] ZEROSCOPE DESCARGADO")
        print("="*60)
        print(f"\nUbicacion: {dest_file}")
        print("\nAhora necesitas:")
        print("1. Instalar nodos I2V para Zeroscope en ComfyUI")
        print("2. O usar SVD Turbo como alternativa (ya disponible)")
    else:
        print("\n" + "="*60)
        print("[ERROR] No se pudo descargar automaticamente")
        print("="*60)
        print("""
Puedes descargarlo manualmente:
1. Ve a: https://huggingface.co/cerspense/zeroscope_v2_XL
2. Descarga el archivo: zeroscope_v2_XL.safetensors
3. Guardalo en:
   ui/tob/ComfyUI/models/diffusion_models/zeroscope_v2_XL/UNET/
   Y renombralo a: diffusion_pytorch_model.safetensors
        """)


if __name__ == "__main__":
    download_zeroscope()
