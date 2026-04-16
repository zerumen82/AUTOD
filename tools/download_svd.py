"""
Download SVD (Stable Video Diffusion) XT model for ComfyUI

Model: stabilityai/stable-video-diffusion-img2vid-xt
This is the most advanced SVD model for image-to-video

Usage:
    python tools/download_svd.py
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


def download_svd_xt():
    """Download SVD XT model"""
    
    print("\n" + "="*60)
    print("DESCARGANDO SVD XT (Stable Video Diffusion)")
    print("="*60)
    
    # Create directories
    svd_path = os.path.join(BASE_PATH, "diffusion_models", "svd_xt")
    unet_path = os.path.join(svd_path, "unet")
    vae_path = os.path.join(svd_path, "vae")
    image_encoder_path = os.path.join(BASE_PATH, "clip", "svd_image_encoder")
    
    os.makedirs(unet_path, exist_ok=True)
    os.makedirs(vae_path, exist_ok=True)
    os.makedirs(image_encoder_path, exist_ok=True)
    
    # Files to download (SVD XT)
    files_to_download = [
        # UNET
        ("https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/unet/diffusion_pytorch_model.safetensors",
         os.path.join(unet_path, "diffusion_pytorch_model.safetensors"),
         "SVD XT UNET"),
        
        # VAE
        ("https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/vae/diffusion_pytorch_model.safetensors",
         os.path.join(vae_path, "diffusion_pytorch_model.safetensors"),
         "SVD XT VAE"),
        
        # Image Encoder (CLIP Vision)
        ("https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/image_encoder/model.safetensors",
         os.path.join(image_encoder_path, "model.safetensors"),
         "SVD Image Encoder"),
    ]
    
    success = True
    
    for url, dest, desc in files_to_download:
        print(f"\n[INFO] {desc}")
        
        # Check size first
        size = get_file_size(url)
        if size > 0:
            print(f"   Tamano: {size / (1024*1024):.1f} MB")
        
        if not download_file(url, dest, desc):
            success = False
    
    if success:
        print("\n" + "="*60)
        print("[OK] SVD XT DESCARGADO")
        print("="*60)
        print("\nArchivos en:")
        print(f"   {svd_path}")
        print("\nReinicia ComfyUI para usar SVD XT")
    else:
        print("\n" + "="*60)
        print("[ERROR] Algunos archivos no se descargaron")
        print("="*60)


if __name__ == "__main__":
    download_svd_xt()
