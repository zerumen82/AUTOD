#!/usr/bin/env python3
import os
from huggingface_hub import hf_hub_download

HF_TOKEN = ""  # Set your HuggingFace token here

output_dir = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\checkpoints"
os.makedirs(output_dir, exist_ok=True)

print("Descargando flux1-dev con hf_hub_download...")
try:
    path = hf_hub_download(
        repo_id="Comfy-Org/flux1-dev",
        filename="flux1-dev.safetensors",
        local_dir=output_dir,
        token=HF_TOKEN,
        local_dir_use_symlinks=False
    )
    print(f"Descargado a: {path}")
except Exception as e:
    print(f"Error: {e}")
