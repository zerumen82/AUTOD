import safetensors.torch
import os, sys

src = r'D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors'
bak = src.replace('.safetensors', '.safetensors.bak')

if not os.path.exists(bak):
    os.rename(src, bak)
    print(f"Backup: {bak}")

sd = safetensors.torch.load_file(bak, device="cpu")
print(f"Loaded: {len(sd)} keys")

if "scaled_fp8" in sd:
    del sd["scaled_fp8"]
    print("Removed 'scaled_fp8' key")

safetensors.torch.save_file(sd, src)
print(f"Saved patched model to {src}")
print(f"Size: {os.path.getsize(src)//1024//1024}MB")