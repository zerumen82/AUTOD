import os
import urllib.request

clip_dir = os.path.join('ui', 'tob', 'ComfyUI', 'models', 'clip_vision')
os.makedirs(clip_dir, exist_ok=True)
dest_path = os.path.join(clip_dir, 'open_clip_pytorch_model.bin')

if os.path.exists(dest_path):
    size = os.path.getsize(dest_path)
    print(f"Current file size: {size} bytes")
    if size < 100000000:  # If file is smaller than 100MB, redownload
        print("File is too small, downloading again...")
        os.remove(dest_path)
    else:
        print("File already exists and seems valid")
        exit()

# Download from Hugging Face
url = "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/open_clip_pytorch_model.bin"
print(f"Downloading CLIP Vision model from {url}")
print(f"Destination: {dest_path}")

try:
    urllib.request.urlretrieve(url, dest_path)
    print(f"Download complete! File size: {os.path.getsize(dest_path)} bytes")
except Exception as e:
    print(f"Download failed: {e}")
