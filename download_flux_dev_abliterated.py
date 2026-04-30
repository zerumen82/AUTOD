#!/usr/bin/env python3
import huggingface_hub, os

repo_id = "t8star/flux.1-dev-abliterated-V2-GGUF"
filename = "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf"
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
print(f"\nFLUX.1-dev-abliterated V2 listo (sin censura)")