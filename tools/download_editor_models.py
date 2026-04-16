#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga los modelos necesarios para el editor de imágenes real:
- ControlNet (para mantener estructura)
- IP-Adapter (para mantener identidad)

NO incluye modelos de inpainting.
"""

import os
import sys
import urllib.request
import shutil
from pathlib import Path

# URLs de los modelos
MODELS = {
    "controlnet": {
        "folder": "ui/tob/ComfyUI/models/controlnet",
        "files": [
            {
                "name": "control_v11f1e_sd15_tile.pth",
                "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1e_sd15_tile.pth",
                "desc": "ControlNet Tile - Mantiene detalles y estructura durante la edición"
            },
            {
                "name": "control_v11p_sd15_softedge.pth",
                "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_softedge.pth",
                "desc": "ControlNet SoftEdge - Mantiene bordes suaves"
            },
            {
                "name": "control_v11p_sd15_openpose.pth",
                "url": "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth",
                "desc": "ControlNet OpenPose - Mantiene pose humana"
            },
        ]
    },
    "ipadapter": {
        "folder": "ui/tob/ComfyUI/models/ipadapter",
        "files": [
            {
                "name": "ip-adapter_sd15.safetensors",
                "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors",
                "desc": "IP-Adapter SD1.5 - Mantiene identidad de persona"
            },
            {
                "name": "ip-adapter-plus_sd15.safetensors",
                "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter-plus_sd15.safetensors",
                "desc": "IP-Adapter Plus SD1.5 - Mejor preservacion de identidad"
            },
        ]
    },
    "clip_vision": {
        "folder": "ui/tob/ComfyUI/models/clip_vision",
        "files": [
            {
                "name": "CLIP-ViT-H-14.safetensors",
                "url": "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors",
                "desc": "CLIP Vision Encoder - Requerido para IP-Adapter"
            },
        ]
    }
}

def get_file_size_mb(url):
    """Intenta obtener el tamaño del archivo de la URL"""
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            size = response.headers.get('Content-Length')
            if size:
                return int(size) / (1024 * 1024)
    except:
        pass
    return None

def download_file(url, dest_path, desc=""):
    """Descarga un archivo con barra de progreso"""
    print(f"\n{'='*60}")
    print(f"Descargando: {os.path.basename(dest_path)}")
    if desc:
        print(f"Descripcion: {desc}")
    print(f"URL: {url}")
    print(f"Destino: {dest_path}")
    print(f"{'='*60}")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Verificar si ya existe
    if os.path.exists(dest_path):
        existing_size = os.path.getsize(dest_path)
        if existing_size > 1000000:  # > 1MB
            print(f"[OK] Ya existe ({existing_size / (1024*1024):.1f} MB) - Saltando")
            return True
    
    try:
        # Descargar con progreso
        def report_progress(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f"\rProgreso: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        print(f"\n[OK] Descarga completada: {os.path.getsize(dest_path) / (1024*1024):.1f} MB")
        return True
    except Exception as e:
        print(f"\n[ERROR] Error descargando: {e}")
        return False

def main():
    print("=" * 60)
    print("DESCARGADOR DE MODELOS PARA EDITOR REAL")
    print("=" * 60)
    print("\nEste script descargará:")
    print("  - ControlNet Tile (mantiene estructura)")
    print("  - ControlNet SoftEdge (mantiene bordes)")
    print("  - ControlNet OpenPose (mantiene pose)")
    print("  - IP-Adapter SD1.5 (mantiene identidad)")
    print("  - IP-Adapter Plus SD1.5 (mejor identidad)")
    print("  - CLIP Vision Encoder (requerido para IP-Adapter)")
    print("\nNOTA: NO se descargan modelos de inpainting")
    print("=" * 60)
    
    # Obtener directorio base
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)
    
    total_files = 0
    successful = 0
    
    for category, data in MODELS.items():
        folder = data["folder"]
        files = data["files"]
        
        print(f"\n\n{'#'*60}")
        print(f"# CATEGORÍA: {category.upper()}")
        print(f"# Carpeta: {folder}")
        print(f"{'#'*60}")
        
        for file_info in files:
            total_files += 1
            dest_path = os.path.join(folder, file_info["name"])
            
            if download_file(file_info["url"], dest_path, file_info["desc"]):
                successful += 1
    
    print("\n\n" + "=" * 60)
    print("RESUMEN DE DESCARGA")
    print("=" * 60)
    print(f"Archivos procesados: {total_files}")
    print(f"Descargas exitosas: {successful}")
    print(f"Errores: {total_files - successful}")
    
    if successful == total_files:
        print("\n[OK] TODOS LOS MODELOS DESCARGADOS CORRECTAMENTE")
        print("\nAhora puedes usar el editor con ControlNet e IP-Adapter")
    else:
        print("\n[WARN] Algunos modelos no se pudieron descargar")
        print("Intenta ejecutar el script de nuevo o descargar manualmente")
    
    return successful == total_files

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
