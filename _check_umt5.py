import torch
import safetensors.torch
path = r'D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors'
sd = safetensors.torch.load_file(path, device="cpu")
keys = list(sd.keys())
has_fp8 = "scaled_fp8" in sd
has_shared = "shared.weight" in sd
has_token = "token_embedding.weight" in sd
print(f"Keys: {len(keys)}")
print(f"scaled_fp8 key present: {has_fp8}")
print(f"shared.weight present: {has_shared}")
print(f"token_embedding.weight present: {has_token}")
for k in keys[:3]:
    v = sd[k]
    print(f"  {k}: {v.shape} {v.dtype}")