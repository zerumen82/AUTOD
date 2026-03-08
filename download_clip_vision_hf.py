import os
from huggingface_hub import hf_hub_download

clip_dir = os.path.join('ui', 'tob', 'ComfyUI', 'models', 'clip_vision')
os.makedirs(clip_dir, exist_ok=True)

try:
    # Stable Video Diffusion uses CLIP-ViT-H-14
    print("Downloading CLIP Vision model (CLIP-ViT-H-14) from Hugging Face...")
    file_path = hf_hub_download(
        repo_id="laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        filename="open_clip_pytorch_model.bin",
        local_dir=clip_dir
    )
    print(f"Download successful: {file_path}")
    print(f"File size: {os.path.getsize(file_path)} bytes")
    
except Exception as e:
    print(f"Download failed: {e}")
    print("\nTrying alternative download...")
    try:
        # Try to download from stabilityai's repository
        file_path = hf_hub_download(
            repo_id="stabilityai/stable-video-diffusion-img2vid-xt",
            filename="open_clip_pytorch_model.bin",
            local_dir=clip_dir
        )
        print(f"Download successful: {file_path}")
        print(f"File size: {os.path.getsize(file_path)} bytes")
    except Exception as e2:
        print(f"Alternative download also failed: {e2}")
        print("\nPlease download the CLIP Vision model manually from Hugging Face")
