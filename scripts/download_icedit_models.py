#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para descargar modelos de ICEdit a las carpetas correctas
"""

import os
import sys
import requests

# Token de HuggingFace del usuario
HF_TOKEN = ""  # Set your HuggingFace token here

HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}

# URLs de los modelos (HuggingFace)
MODELS = {
    # Qwen Image Edit VAE
    "vae/qwen_vae.safetensors": [
        "https://huggingface.co/Qwen/Qwen-Image-Edit/resolve/main/vae/diffusion_pytorch_model.safetensors"
    ],
}

BASE_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"

def download_file(url: str, dest: str) -> bool:
    """Descarga un archivo usando requests con token HF"""
    print(f"Descargando: {url}")
    print(f"  -> {dest}")
    
    try:
        response = requests.get(url, headers=HEADERS, stream=True, timeout=30)
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
    
    try:
        # Usar huggingface_hub si está disponible
        try:
            from huggingface_hub import hf_hub_download
            # Intentar descargar directamente
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print(f"  OK")
            return True
        except ImportError:
            # Fallback a requests directo
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(dest, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            pct = (downloaded / total_size) * 100
                            print(f"\r  Progreso: {pct:.1f}%", end="", flush=True)
            
            print(f"\n  OK")
            return True
            
    except Exception as e:
            print(f"  ERROR: {e}")
            return False

def main():
    print("=" * 60)
    print("DESCARGAR MODELOS QWEN IMAGE EDIT")
    print("=" * 60)
    print(f"Directorio base: {BASE_DIR}")
    print()
    
    # Crear directorio base si no existe
    os.makedirs(os.path.join(BASE_DIR, "vae"), exist_ok=True)
    
    success = 0
    failed = 0
    
    for relative_path, urls in MODELS.items():
        dest = os.path.join(BASE_DIR, relative_path)
        
        # Verificar si ya existe
        if os.path.exists(dest):
            size = os.path.getsize(dest) / (1024**3)
            print(f"YA EXISTE: {relative_path} ({size:.2f} GB)")
            success += 1
            continue
        
        # Intentar cada URL
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