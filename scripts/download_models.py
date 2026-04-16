#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para descargar modelos varios
"""

import os
import requests

HF_TOKEN = ""  # Set your HuggingFace token here
HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

BASE_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"

MODELS = {
    # FLUX dev de ComfyOrg (formato checkpoint estándar)
    "checkpoints/flux1-dev.safetensors": [
        "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev.safetensors"
    ],
}

def download_file(url: str, dest: str) -> bool:
    print(f"Descargando: {url}")
    print(f"  -> {dest}")
    
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=120)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"  OK")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def main():
    print("=" * 60)
    print("DESCARGAR MODELOS")
    print("=" * 60)
    print(f"Directorio base: {BASE_DIR}")
    print()
    
    os.makedirs(os.path.join(BASE_DIR, "vae"), exist_ok=True)
    
    success = 0
    failed = 0
    
    for relative_path, urls in MODELS.items():
        dest = os.path.join(BASE_DIR, relative_path)
        
        if os.path.exists(dest):
            size = os.path.getsize(dest) / (1024**2)
            print(f"YA EXISTE: {relative_path} ({size:.2f} MB)")
            success += 1
            continue
        
        downloaded = False
        for url in urls:
            if download_file(url, dest):
                downloaded = True
                success += 1
                break
        
        if not downloaded:
            print(f"ERROR: No se pudo descargar {relative_path}")
            failed += 1
    
    print()
    print("=" * 60)
    print(f"RESULTADO: {success} OK, {failed} fallidos")
    print("=" * 60)

if __name__ == "__main__":
    main()