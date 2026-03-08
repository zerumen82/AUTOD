"""Test VAE loading for LTX Video 0.9.1"""
import sys
sys.path.insert(0, 'ui/tob/ComfyUI')

import torch
from safetensors import safe_open

# Load the VAE state dict
vae_path = 'ui/tob/ComfyUI/models/vae/ltx-video-0.9.1_vae.safetensors'
f = safe_open(vae_path, framework='pt')
sd = {}
for key in f.keys():
    sd[key] = f.get_tensor(key)

# Convert diffusers format to ComfyUI format
new_sd = {}
for k, v in sd.items():
    new_k = k.replace(".resnets.", ".res_blocks.")
    new_sd[new_k] = v
sd = new_sd

# Check version detection
tensor_conv1 = sd["decoder.up_blocks.0.res_blocks.0.conv1.conv.weight"]
print(f"tensor_conv1.shape: {tensor_conv1.shape}")

version = 0
if tensor_conv1.shape[0] == 512:
    version = 0
    # Check for LTX Video 0.9.1
    if "encoder.down_blocks.1.conv_out.conv1.conv.weight" in sd and "decoder.last_time_embedder.timestep_embedder.linear_1.weight" not in sd:
        version = 3
        print("Detected LTX Video 0.9.1 (version 3)")
elif tensor_conv1.shape[0] == 1024:
    version = 1
    if "encoder.down_blocks.1.conv.conv.bias" in sd:
        version = 2

print(f"Detected version: {version}")

# Try to create the VAE model
from comfy.ldm.lightricks.vae.causal_video_autoencoder import VideoVAE

print(f"\nCreating VideoVAE with version={version}...")
vae = VideoVAE(version=version)

print(f"\nVAE encoder structure:")
for name, module in vae.encoder.named_modules():
    if 'down_blocks' in name and 'conv1' in name:
        print(f"  {name}")

print(f"\nLoading state dict...")
vae.load_state_dict(sd, strict=False)
print("VAE loaded successfully!")

# Clean up
del f
del sd
del new_sd
