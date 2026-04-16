"""Download Qwen Image Edit models for 8GB VRAM - CORRECT REPOS"""
import os
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "0"

# Correct paths
UNET_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\unet"
TE_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\text_encoders"
VAE_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\vae"

os.makedirs(UNET_DIR, exist_ok=True)
os.makedirs(TE_DIR, exist_ok=True)
os.makedirs(VAE_DIR, exist_ok=True)

# 1. Download Q2_K GGUF (7.06GB - fits 8GB VRAM)
print("[1/3] Downloading Qwen Image Edit Q2_K GGUF (7GB)...")
try:
    path = hf_hub_download(
        repo_id="QuantStack/Qwen-Image-Edit-2509-GGUF",
        filename="Qwen-Image-Edit-2509-Q2_K.gguf",
        local_dir=UNET_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[OK] UNet: {path}")
except Exception as e:
    print(f"[ERROR] UNet: {e}")

# 2. Download compatible CLIP from official Qwen repo
print("\n[2/3] Downloading Qwen CLIP (text encoder)...")
try:
    path = hf_hub_download(
        repo_id="Qwen/Qwen-Image-Edit-2509",
        filename="text_encoder/model-00001-of-00004.safetensors",
        local_dir=TE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[OK] CLIP shard 1: {path}")
except Exception as e:
    print(f"[ERROR] CLIP: {e}")

# 3. Download compatible VAE
print("\n[3/3] Downloading Qwen VAE...")
try:
    path = hf_hub_download(
        repo_id="Qwen/Qwen-Image-Edit-2509",
        filename="vae/diffusion_pytorch_model.safetensors",
        local_dir=VAE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[OK] VAE: {path}")
except Exception as e:
    print(f"[ERROR] VAE: {e}")

print("\n[DONE] Models download complete.")
