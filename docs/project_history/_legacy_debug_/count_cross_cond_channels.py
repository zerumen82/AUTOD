import sys
import os
sys.path.insert(0, "d:/PROJECTS/AUTOAUTO/ui/tob/ComfyUI")
import torch

# Let's simulate what happens in apply_model

# SVD_img2vid cross conditioning produces 8 channels
# Noise + latent_image

print("=== SVD Cross Conditioning Channels ===")

try:
    # Try to see what SVD_img2vid extra_conds produces
    from ui.tob.ComfyUI.nodes import VAELoader
    from ui.tob.ComfyUI.comfy_extras.nodes_clip_vision import CLIPVisionLoader
    
    vae_loader = VAELoader()
    vae = vae_loader.load_vae("pixel_space")
    print(f"VAE loaded")
    
    clip_loader = CLIPVisionLoader()
    clip_vision = clip_loader.load_clip("open_clip_pytorch_model.bin")
    print(f"CLIP Vision loaded")
    
    from PIL import Image
    import numpy as np
    from io import BytesIO
    dummy_image = Image.fromarray((np.random.rand(480, 720, 3) * 255).astype(np.uint8))
    img_buffer = BytesIO()
    dummy_image.save(img_buffer, format="JPEG")
    img_buffer.seek(0)
    
    from ui.tob.ComfyUI.nodes import LoadImage
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False, dir="d:/PROJECTS/AUTOAUTO/testdata") as tmp:
        tmp.write(img_buffer.read())
        tmp.flush()
        tmp_path = tmp.name
    
    load_image = LoadImage()
    image = load_image.load_image(tmp_path)[0]
    print(f"Image loaded - shape: {image.shape}")
    
    from ui.tob.ComfyUI.comfy_extras.nodes_video_model import SVD_img2vid_Conditioning
    svd_cond = SVD_img2vid_Conditioning()
    result = svd_cond.encode(
        clip_vision, 
        image, 
        vae, 
        width=720, 
        height=480, 
        video_frames=24,
        motion_bucket_id=127, 
        fps=24, 
        augmentation_level=0.0
    )
    
    positive, negative, latent = result
    positive_latent = latent['samples']
    print(f"Latent shape (samples): {positive_latent.shape}")
    
    os.unlink(tmp_path)
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    print(traceback.format_exc())

# Try to get model config for SVD
print("\n=== Model Config ===")

try:
    import requests
    from roop.comfy_workflows import get_svd_turbo_workflow
    
    print("Trying to load model config info")
    # Check if there are any configs in the model directory
    import os
    model_dir = "ui/tob/ComfyUI/models/diffusion_models/StableDiffusionTurbo"
    if os.path.exists(model_dir):
        print(f"Model dir found: {model_dir}")
        for f in os.listdir(model_dir):
            if f.endswith('.json') or f.endswith('.yaml') or f.endswith('.yml'):
                print(f"  Config: {f}")
                try:
                    with open(os.path.join(model_dir, f), encoding='utf-8') as config_file:
                        print(f"  Preview: {config_file.read(500)}...")
                except:
                    continue
    
    # Get UNet info
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        object_info = response.json()
        print(f"UNETLoader found: {'UNETLoader' in object_info}")
        
except Exception as e:
    print(f"ERROR getting config: {e}")
    import traceback
    print(traceback.format_exc())

print("\n=== Expected Channels ===")
print("UNet expects input with 8 channels")
print("We received input with 7 channels")
print("7 = 3 (noise) + 4 (latent_image)?")
print("Maybe latent_image has 4 channels but should be 5?")
