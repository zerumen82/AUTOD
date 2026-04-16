"""
Download WanVideo T5 text encoder for ComfyUI-WanVideoWrapper

Model: umt5-xxl-enc-bf16.safetensors
This is the specific T5 text encoder required by WanVideo models

Usage:
    python tools/download_wan_t5.py
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
        print(f"   [EXISTS] {desc} ({size:.1f} MB)")
        return True
    
    print(f"   [DOWNLOADING] {desc}...")
    
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


def download_wan_t5():
    """Download WanVideo T5 text encoder"""
    
    print("\n" + "="*60)
    print("DOWNLOADING WANVIDEO T5 TEXT ENCODER")
    print("="*60)
    
    text_enc_path = os.path.join(BASE_PATH, "text_encoders")
    os.makedirs(text_enc_path, exist_ok=True)
    
    files_to_download = [
        (
            "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/umt5-xxl-enc-bf16.safetensors",
            os.path.join(text_enc_path, "umt5-xxl-enc-bf16.safetensors"),
            "WanVideo T5 Text Encoder (umt5-xxl bf16)"
        )
    ]
    
    success = True
    
    for url, dest, desc in files_to_download:
        print(f"\n[INFO] {desc}")
        
        size = get_file_size(url)
        if size > 0:
            print(f"   Size: {size / (1024*1024):.1f} MB")
        
        if not download_file(url, dest, desc):
            success = False
    
    if success:
        print(f"\n{'-'*60}")
        print("Download completed!")
        print(f"T5 encoder saved to: {os.path.join(text_enc_path, 'umt5-xxl-enc-bf16.safetensors')}")
    else:
        print(f"\n{'-'*60}")
        print("Error downloading T5 text encoder")
    
    return success

if __name__ == "__main__":
    download_wan_t5()
