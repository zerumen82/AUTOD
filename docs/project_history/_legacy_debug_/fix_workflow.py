# Read the file
with open('roop/comfy_workflows.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix LTX path
content = content.replace(
    'ltx_unet = "ltx-video-0.9.1/model.safetensors"',
    'ltx_unet = "ltx-video-0.9.1\\\\model.safetensors"'
)

# Also fix Zeroscope  
content = content.replace(
    'unet_name = "zeroscope_v2_XL/UNET/diffusion_pytorch_model.bin"',
    'unet_name = "zeroscope_v2_XL\\\\UNET\\\\diffusion_pytorch_model.bin"'
)
content = content.replace(
    'clip_name = "zeroscope_v2_XL/TEXT_ENCODER/pytorch_model.bin"',
    'clip_name = "zeroscope_v2_XL\\\\TEXT_ENCODER\\\\pytorch_model.bin"'
)
content = content.replace(
    '"vae_name": "taesd"',
    '"vae_name": "svd_xt_image_decoder.safetensors"'
)

# Write
with open('roop/comfy_workflows.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done!')
