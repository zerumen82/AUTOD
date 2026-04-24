#!/usr/bin/env python3
# Download T5-XXL Q4_K_S GGUF for FLUX.1-schnell
import huggingface_hub, os, shutil

repo_id = "city96/t5-v1_1-xxl-encoder-gguf"
filename = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
local_dir = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\text_encoders"

print(f"Downloading {filename} from {repo_id}...")
filepath = huggingface_hub.hf_hub_download(
    repo_id=repo_id,
    filename=filename,
    local_dir=local_dir,
    resume_download=True
)
print(f"Downloaded: {filepath}")
print(f"File size: {os.path.getsize(filepath) / (1024**3):.2f} GB")
