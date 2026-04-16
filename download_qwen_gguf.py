"""Download Qwen-Image-Edit GGUF for 8GB VRAM"""
import os
from huggingface_hub import list_repo_files, hf_hub_download

os.environ["HF_HUB_OFFLINE"] = "0"

REPO_ID = "QuantStack/Qwen-Image-Edit-2509-GGUF"
LOCAL_DIR = r"D:\PROJECTS\models\Qwen-Image-Edit-2509-GGUF"
os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"[DOWNLOAD] Listing files in {REPO_ID}...")
try:
    files = list_repo_files(REPO_ID)
    gguf_files = [f for f in files if f.endswith('.gguf')]
    print(f"Found {len(gguf_files)} GGUF files:")
    for f in gguf_files[:10]:
        print(f"  - {f}")
    
    # Download the main GGUF file (Q8_0 or similar)
    main_file = [f for f in gguf_files if 'Q8' in f or 'q8' in f]
    if not main_file:
        main_file = [f for f in gguf_files if 'Q4' in f or 'q4' in f]
    if not main_file:
        main_file = gguf_files[:1]
    
    if main_file:
        print(f"\n[DOWNLOAD] Downloading: {main_file[0]}")
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=main_file[0],
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"[OK] Downloaded to: {path}")
    else:
        print("[ERROR] No GGUF files found")
        
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
