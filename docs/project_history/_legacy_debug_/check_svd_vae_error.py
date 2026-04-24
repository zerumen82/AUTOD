import requests
import json

workflow = {
    "1": {"inputs": {"image": "test_image.png", "upload": "image"}, "class_type": "LoadImage"},
    "2": {"inputs": {"unet_name": "StableDiffusionTurbo\\svd_xt.safetensors", "weight_dtype": "default"}, "class_type": "UNETLoader"},
    "3": {"inputs": {"vae_name": "pixel_space"}, "class_type": "VAELoader"},
    "4": {"inputs": {"clip_name": "open_clip_pytorch_model.bin"}, "class_type": "CLIPVisionLoader"},
    "5": {"inputs": {
        "clip_vision": ["4", 0],
        "init_image": ["1", 0],
        "vae": ["3", 0],
        "width": 720,
        "height": 480,
        "video_frames": 24,
        "motion_bucket_id": 127,
        "fps": 24,
        "augmentation_level": 0.0
    }, "class_type": "SVD_img2vid_Conditioning"},
    "6": {"inputs": {
        "model": ["2", 0],
        "positive": ["5", 0],
        "negative": ["5", 1],
        "latent_image": ["5", 2],
        "seed": 12345,
        "steps": 25,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0
    }, "class_type": "KSampler"},
    "7": {"inputs": {"vae": ["3", 0], "samples": ["6", 0]}, "class_type": "VAEDecode"},
    "8": {"inputs": {"images": ["7", 0], "fps": 24}, "class_type": "CreateVideo"},
    "9": {"inputs": {
        "video": ["8", 0],
        "filename_prefix": "SVD_Turbo_Output",
        "format": "mp4",
        "codec": "auto"
    }, "class_type": "SaveVideo"},
}

try:
    response = requests.post(
        "http://127.0.0.1:8188/prompt",
        json={"prompt": workflow},
        timeout=30
    )
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
