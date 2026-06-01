"""
Download WanVideo VAE model for ComfyUI-WanVideoWrapper

Model: Wan2.1_VAE.pth
This is the specific VAE required by WanVideo models

Usage:
    python tools/download_wan_vae.py
"""

import os
import urllib.request

BASE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ui", "tob", "ComfyUI", "models")

def get_file_size(url):
    """Get file size without downloading"""
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            return int(response.headers.get('Content-Length', 0))
    except:
        return 0

def download_file(url: str, dest_path: str, desc: str = "") -> bool:
    """Download file with progress"""
    if os.path.exists(dest_path):
        size = os.path.getsize(dest_path) / (1024*1024)
        print(f"   [YA EXISTE] {desc} ({size:.1f} MB)")
        return True
    
    print(f"   [DESCARGANDO] {desc}...")
    
    try:
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
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


def download_wan_vae():
    """Download WanVideo VAEs - both 16-channel (Wan2.1, for 14B models) and 48-channel (Wan2.2, for 5B models)"""
    
    print("\n" + "="*60)
    print("DESCARGANDO WANVIDEO VAEs")
    print("="*60)
    
    vae_path = os.path.join(BASE_PATH, "vae")
    os.makedirs(vae_path, exist_ok=True)
    
    files_to_download = [
        (
            "https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP/resolve/main/Wan2.1_VAE.pth",
            os.path.join(vae_path, "Wan2.1_VAE.pth"),
            "WanVideo 2.1 VAE (16 canales, para modelos 14B)"
        ),
        (
            "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_2_VAE_bf16.safetensors",
            os.path.join(vae_path, "Wan2_2_VAE_bf16.safetensors"),
            "WanVideo 2.2 VAE (48 canales, para modelos 5B)"
        ),
    ]
    
    success = True
    
    for url, dest, desc in files_to_download:
        print(f"\n[INFO] {desc}")
        
        size = get_file_size(url)
        if size > 0:
            print(f"   Tamano: {size / (1024*1024):.1f} MB")
        
        if not download_file(url, dest, desc):
            success = False
    
    if success:
        print(f"\n{'-'*60}")
        print("[OK] Descarga completada!")
        print(f"  - Wan2.1_VAE.pth (16 canales, 8x upsampling) - para modelos 14B (I2V/T2V)")
        print(f"  - Wan2_2_VAE_bf16.safetensors (48 canales, 16x upsampling) - para modelos 5B")
        print(f"\nNOTA: Usar Wan2.1 VAE para modelos 14B y Wan2.2 VAE para modelos 5B")
    else:
        print(f"\n{'-'*60}")
        print("[ERROR] Error al descargar algun VAE")
    
    return success

if __name__ == "__main__":
    download_wan_vae()
