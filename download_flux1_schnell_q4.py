#!/usr/bin/env python3
import huggingface_hub, os, sys, time

repo_id = "city96/FLUX.1-schnell-gguf"
filename = "flux1-schnell-Q4_K_S.gguf"
local_dir = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\diffusion_models"

dest = os.path.join(local_dir, filename)
if os.path.exists(dest):
    size = os.path.getsize(dest)
    print(f"File already exists: {size/(1024**3):.2f} GB")
    sys.exit(0)

print(f"Downloading {filename} from {repo_id}...")
print("This is 6.78 GB, may take several minutes...")
start = time.time()
try:
    filepath = huggingface_hub.hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        resume_download=True
    )
    elapsed = time.time() - start
    size = os.path.getsize(filepath)
    print(f"Downloaded in {elapsed/60:.1f} min")
    print(f"Size: {size/(1024**3):.2f} GB")
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
