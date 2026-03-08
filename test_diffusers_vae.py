"""Test loading LTX Video 0.9.1 VAE with diffusers"""
import sys
sys.path.insert(0, 'ui/tob/ComfyUI/comfy_env/Lib/site-packages')

import torch
from safetensors import safe_open

# Load the VAE state dict
vae_path = 'ui/tob/ComfyUI/models/vae/ltx-video-0.9.1_vae.safetensors'
f = safe_open(vae_path, framework='pt')
sd = {}
for key in f.keys():
    sd[key] = f.get_tensor(key)

print(f"Loaded {len(sd)} keys from VAE")

# Try to load with diffusers
try:
    from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    
    print("\nCreating AutoencoderKLLTXVideo...")
    vae = AutoencoderKLLTXVideo()
    
    print("Loading state dict...")
    vae.load_state_dict(sd, strict=False)
    print("VAE loaded successfully with diffusers!")
    
except Exception as e:
    print(f"Error loading with diffusers: {e}")

# Try with from_pretrained
try:
    from diffusers.models.autoencoders.autoencoder_kl_ltx import AutoencoderKLLTXVideo
    
    print("\nTrying AutoencoderKLLTXVideo.from_pretrained...")
    # Save to temp directory
    import os
    import json
    temp_dir = 'temp_vae'
    os.makedirs(temp_dir, exist_ok=True)
    
    # Save state dict
    from safetensors.torch import save_file
    save_file(sd, f'{temp_dir}/diffusion_pytorch_model.safetensors')
    
    # Create config
    config = {
        "_class_name": "AutoencoderKLLTXVideo",
        "in_channels": 3,
        "out_channels": 3,
        "latent_channels": 128,
        "block_out_channels": [128, 256, 512, 512],
        "decoder_block_out_channels": [128, 256, 512, 512],
        "layers_per_block": [4, 3, 3, 3, 4],
        "decoder_layers_per_block": [4, 3, 3, 3, 4],
        "spatio_temporal_scaling": [True, True, True, False],
        "decoder_spatio_temporal_scaling": [True, True, True, False],
        "patch_size": 4,
        "patch_size_t": 1,
        "scaling_factor": 1.0,
        "timestep_conditioning": False,
    }
    with open(f'{temp_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    vae = AutoencoderKLLTXVideo.from_pretrained(temp_dir)
    print("VAE loaded successfully with from_pretrained!")
    
except Exception as e:
    print(f"Error with from_pretrained: {e}")
    import traceback
    traceback.print_exc()
