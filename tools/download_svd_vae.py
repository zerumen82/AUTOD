#!/usr/bin/env python3
"""
Script para descargar el VAE correcto de SVD (Stable Video Diffusion)

El VAE corrupto (svd_xt_image_decoder.safetensors - 9.5GB) NO es válido.
Necesitamos descargar el VAE correcto (~160MB).
"""

import os
import sys
import urllib.request
import json

# Añadir el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuración
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# URL del VAE de SVD en HuggingFace
# El VAE correcto está en el repositorio de stabilityai/stable-video-diffusion
SVD_VAE_URL = "https://huggingface.co/stabilityai/stable-video-diffusion/resolve/main/svd/vae/diffusion_pytorch_model.safetensors"

def download_file(url, dest_path, token=None):
    """Descargar archivo con progress bar"""
    print(f"Descargando: {url}")
    print(f"Destino: {dest_path}")
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    # Headers para request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    if token:
        headers['Authorization'] = f'Bearer {token}'
    
    # Descargar con progress
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        print(f"\rProgreso: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")
    
    urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
    print()  # Nueva línea al terminar
    
    # Verificar tamaño
    size_mb = os.path.getsize(dest_path) / 1024 / 1024
    print(f"Archivo descargado: {size_mb:.1f} MB")
    
    return size_mb

def main():
    # Paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vae_path = os.path.join(base_path, "ui", "tob", "ComfyUI", "models", "vae")
    
    print("=" * 60)
    print("DESCARGANDO VAE DE SVD")
    print("=" * 60)
    
    # El VAE de SVD debería estar en: models/vae/svd_xt/
    # Pero también necesitamos copiarlo a models/vae/ para que ComfyUI lo reconozca
    
    # Destination in models/vae
    vae_dest = os.path.join(vae_path, "svd_xt_image_decoder.safetensors")
    
    # Verificar si ya existe y tiene tamaño correcto
    if os.path.exists(vae_dest):
        size_mb = os.path.getsize(vae_dest) / 1024 / 1024
        print(f"Ya existe: {vae_dest} ({size_mb:.1f} MB)")
        
        # Si es > 1GB, está corrupto - eliminarlo
        if size_mb > 1000:
            print(f"Archivo corrupto ({size_mb:.1f} MB > 1GB), eliminando...")
            os.remove(vae_dest)
        elif 100 < size_mb < 300:
            print(f"Archivo parece válido ({size_mb:.1f} MB)")
            print("No necesitas descargar de nuevo.")
            return
        else:
            print(f"Tamaño inesperado, descargando de nuevo...")
    
    # Descargar el VAE correcto
    print("\nDescargando VAE de SVD...")
    
    # Intentar descargar desde HuggingFace
    try:
        size_mb = download_file(SVD_VAE_URL, vae_dest, HF_TOKEN)
        
        # Verificar que el tamaño sea razonable (debería ser ~160MB)
        if size_mb < 100:
            print(f"ERROR: El archivo descargado es muy pequeño ({size_mb:.1f} MB)")
            print("La descarga puede haber fallado.")
            sys.exit(1)
        elif size_mb > 500:
            print(f"ADVERTENCIA: El archivo es más grande de lo esperado ({size_mb:.1f} MB)")
            print("Puede que no sea el VAE correcto.")
        
        print("\n" + "=" * 60)
        print("VAE DE SVD DESCARGADO CORRECTAMENTE")
        print(f"Ruta: {vae_dest}")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR al descargar: {e}")
        
        # Intentar método alternativo - crear un enlace simbólico o copiar
        print("\nIntentando método alternativo...")
        
        # Buscar si hay algún VAE existente que funcione
        alt_vaes = [
            os.path.join(vae_path, "Wan2.2_VAE_bf16.safetensors"),
            os.path.join(vae_path, "Wan2.2_VAE.safetensors"),
        ]
        
        for alt_vae in alt_vaes:
            if os.path.exists(alt_vae):
                print(f"Usando VAE alternativo: {alt_vae}")
                # Copiar como svd_xt_image_decoder.safetensors
                import shutil
                shutil.copy2(alt_vae, vae_dest)
                print("Copiado correctamente.")
                return
        
        print("No se encontró VAE alternativo.")
        sys.exit(1)

if __name__ == "__main__":
    main()
