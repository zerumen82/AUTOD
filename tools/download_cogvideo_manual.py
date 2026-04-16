"""Download CogVideoX using Python requests"""
import os
import requests
from tqdm import tqdm

# Paths
diffusion_path = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\diffusion_models"
os.makedirs(diffusion_path, exist_ok=True)

# Try CogVideoX-2B first (smaller)
url = "https://huggingface.co/THUDM/CogVideo/resolve/main/cogvideo-2b.safetensors"
dest = os.path.join(diffusion_path, "cogvideo-2b.safetensors")

print(f"Downloading from: {url}")
print(f"Destination: {dest}")

# Check if file exists
if os.path.exists(dest) and os.path.getsize(dest) > 1000:
    print(f"[OK] File already exists: {os.path.getsize(dest) / (1024**3):.1f} GB")
else:
    print("Starting download...")
    response = requests.get(url, stream=True, timeout=30)
    if response.status_code == 404:
        print("File not found at this URL")
        print("Trying alternative URL...")
        # Try alternative format
        url2 = "https://huggingface.co/THUDM/CogVideoX-2B/resolve/main/model.safetensors"
        dest2 = os.path.join(diffusion_path, "cogvideo-x-2b.safetensors")
        print(f"Trying: {url2}")
    else:
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        print(f"File size: {total_size / (1024**3):.1f} GB")
        
        with open(dest, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"[OK] Downloaded: {dest}")
        print(f"Size: {os.path.getsize(dest) / (1024**3):.1f} GB")
