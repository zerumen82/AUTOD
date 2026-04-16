"""Download Qwen VAE and Text Encoder"""
import os
from huggingface_hub import hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "0"

# Qwen VAE
VAE_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\vae"
os.makedirs(VAE_DIR, exist_ok=True)

print("[DOWNLOAD] Qwen VAE...")
try:
    for fname in ["vae/config.json", "vae/diffusion_pytorch_model.safetensors"]:
        path = hf_hub_download(
            repo_id="Qwen/Qwen-Image",
            filename=fname,
            local_dir=VAE_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  OK: {path}")
except Exception as e:
    print(f"[ERROR] VAE: {e}")

# Qwen Text Encoder (4 shards)
TE_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\text_encoders"
os.makedirs(TE_DIR, exist_ok=True)

print("\n[DOWNLOAD] Qwen Text Encoder (4 shards, ~15GB total)...")
te_files = [
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model-00001-of-00004.safetensors",
    "text_encoder/model-00002-of-00004.safetensors",
    "text_encoder/model-00003-of-00004.safetensors",
    "text_encoder/model-00004-of-00004.safetensors",
    "text_encoder/model.safetensors.index.json",
]

for fname in te_files:
    print(f"  Descargando: {fname}...")
    try:
        path = hf_hub_download(
            repo_id="Qwen/Qwen-Image",
            filename=fname,
            local_dir=TE_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"  OK: {os.path.basename(path)}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n[DONE]")
