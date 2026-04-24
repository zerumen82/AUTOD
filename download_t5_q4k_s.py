#!/usr/bin/env python3
import huggingface_hub, os, sys

repo_id = "city96/t5-v1_1-xxl-encoder-gguf"
filename = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
local_dir = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\text_encoders"

print(f"Downloading {filename} from {repo_id}...")
try:
    filepath = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        resume_download=True
    )
    size = os.path.getsize(filepath)
    print(f"OK: {filepath}")
    print(f"Size: {size / (1024**3):.2f} GB")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
