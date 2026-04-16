"""Download Qwen Image Edit models in compatible format"""
import os
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "0"

# Paths
DM_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\diffusion_models"
TE_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\text_encoders"
VAE_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\vae"

# Download FP8 diffusion model (compatible with ComfyUI standard nodes)
print("[1/3] Downloading Qwen Image Edit FP8...")
try:
    path = hf_hub_download(
        repo_id="Comfy-Org/Qwen-Image-Edit-2509",
        filename="qwen_image_edit_fp8_e4m3fn.safetensors",
        local_dir=DM_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[OK] Diffusion Model: {path}")
except Exception as e:
    print(f"[ERROR] Diffusion Model: {e}")

# Download compatible CLIP
print("\n[2/3] Downloading Qwen CLIP...")
try:
    path = hf_hub_download(
        repo_id="Comfy-Org/Qwen-Image-Edit-2509",
        filename="qwen_2.5_vl_7b_fp8_scaled.safetensors",
        local_dir=TE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[OK] CLIP: {path}")
except Exception as e:
    print(f"[ERROR] CLIP: {e}")

# Download compatible VAE
print("\n[3/3] Downloading Qwen VAE...")
try:
    path = hf_hub_download(
        repo_id="Comfy-Org/Qwen-Image-Edit-2509",
        filename="qwen_image_vae.safetensors",
        local_dir=VAE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"[OK] VAE: {path}")
except Exception as e:
    print(f"[ERROR] VAE: {e}")

print("\n[DONE] Models downloaded to ComfyUI/models/")
