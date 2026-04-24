#!/usr/bin/env python3
# Download FLUX.1-schnell Q4_K_S GGUF
import huggingface_hub, os

repo_id = "city96/FLUX.1-schnell-gguf"
filename = "flux1-schnell-Q4_K_S.gguf"
local_dir = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\diffusion_models"

print(f"Downloading {filename} from {repo_id}...")
filepath = huggingface_hub.hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    resume_download=True
)
size = os.path.getsize(filepath)
print(f"Downloaded: {filepath}")
print(f"Size: {size / (1024**3):.2f} GB")
