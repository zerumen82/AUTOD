"""
Download CogVideoX model for ComfyUI

CogVideoX is one of the best open-source image-to-video models.
Requires:
- CogVideoX model (~10GB for 5B version)
- CLIP model for text encoding
- VAE for video encoding/decoding

Usage:
    python tools/download_cogvideo.py
"""

import os
import urllib.request

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "tob", "ComfyUI", "models")

def get_file_size(url):
    """Get file size without downloading"""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as response:
            return int(response.headers.get('Content-Length', 0))
    except Exception as e:
        print(f"   Error getting file size: {e}")
        return 0

def download_file(url: str, dest_path: str, desc: str = "") -> bool:
    """Download file with progress"""
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path) / (1024*1024*1024)
        print(f"   [YA EXISTE] {desc} ({size:.1f} GB)")
        return True
    
    print(f"   [DESCARGANDO] {desc}...")
    print(f"   URL: {url}")
    
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        def reporthook(block_num, block_size, total_size):
            if total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, downloaded * 100 // total_size)
                gb_downloaded = downloaded / (1024 * 1024 * 1024)
                gb_total = total_size / (1024 * 1024 * 1024)
                print(f"\r      {percent}% ({gb_downloaded:.2f}/{gb_total:.2f} GB)", end="", flush=True)
        
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\n   [OK] {desc}")
        return True
        
    except Exception as e:
        print(f"\n   [ERROR] {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False

def download_cogvideo():
    """Download CogVideoX model"""
    
    print("\n" + "="*60)
    print("DESCARGANDO COGVIDEOX")
    print("="*60)
    print("\nCogVideoX es uno de los mejores modelos I2V open-source")
    print("Version: CogVideoX-5B (recomendado para 8GB VRAM)")
    print("")
    
    # Define paths
    diffusion_path = os.path.join(BASE_PATH, "diffusion_models")
    clip_path = os.path.join(BASE_PATH, "clip")
    vae_path = os.path.join(BASE_PATH, "vae")
    
    files_to_download = [
        # CogVideoX-5B model - try 5B first
        {
            "url": "https://huggingface.co/THUDM/CogVideo/resolve/main/cogvideo-5b.safetensors",
            "dest": os.path.join(diffusion_path, "cogvideo-5b.safetensors"),
            "desc": "CogVideoX-5B Model"
        },
    ]
    
    # Check what exists
    print("\n[INFO] Verificando archivos existentes...")
    total_size_needed = 0
    for f in files_to_download:
        if not os.path.exists(f["dest"]):
            size = get_file_size(f["url"])
            if size > 0:
                total_size_needed += size
                print(f"   [FALTA] {f['desc']} ({size/(1024**3):.1f} GB)")
            else:
                print(f"   [FALTA] {f['desc']} (tamano desconocido)")
        else:
            existing_size = os.path.getsize(f["dest"]) / (1024**3)
            print(f"   [OK] {f['desc']} ({existing_size:.1f} GB ya existe)")
    
    if total_size_needed > 0:
        print(f"\n[INFO] Tamano total a descargar: {total_size_needed/(1024**3):.1f} GB")
        
        confirm = input("\nContinuar descarga? (s/n): ")
        if confirm.lower() != 's':
            print("Descarga cancelada")
            return
        
        print("\n[DESCARGANDO]...")
        success = True
        for f in files_to_download:
            if not download_file(f["url"], f["dest"], f["desc"]):
                success = False
                print(f"   Error descargando {f['desc']}")
                break
        
        if success:
            print("\n" + "="*60)
            print("[OK] CogVideoX descargado correctamente!")
            print("="*60)
            print("\nPara usar en AUTO-AUTO:")
            print("1. Selecciona 'cogvideo' en el modelo de animacion")
            print("2. O ejecuta el workflow manualmente en ComfyUI")
        else:
            print("\n[ERROR] Error en la descarga")
    else:
        print("\n[OK] CogVideoX ya esta instalado")

if __name__ == "__main__":
    download_cogvideo()
