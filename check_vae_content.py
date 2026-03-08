import safetensors.torch
import os

vae_path = os.path.join('ui', 'tob', 'ComfyUI', 'models', 'vae', 'svd_xt_image_decoder.safetensors')
print(f"Reading VAE from: {vae_path}")

if os.path.exists(vae_path):
    vae_sd = safetensors.torch.load_file(vae_path)
    print(f"\nVAE state dict keys ({len(vae_sd)} entries):")
    for key in list(vae_sd.keys())[:20]:  # Show first 20 keys
        print(f"  {key}")
        
    # Check for decoder keys
    print("\nChecking for decoder keys:")
    decoder_keys = [k for k in vae_sd.keys() if 'decoder' in k.lower()]
    if decoder_keys:
        print(f"  Found {len(decoder_keys)} decoder keys")
        for key in decoder_keys:
            print(f"  {key}")
    else:
        print("  No decoder keys found")
        
    print(f"\nShape of first key: {vae_sd[list(vae_sd.keys())[0]].shape}")
else:
    print("VAE file not found")
