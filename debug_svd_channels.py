
import requests
import json
import base64
from PIL import Image
import io

# Test image file
IMAGE_FILE = "d:/PROJECTS/AUTOAUTO/testdata/test1.jpg"

# ComfyUI API endpoint
COMFYUI_URL = "http://127.0.0.1:8188"

def load_image(image_path):
    with Image.open(image_path) as img:
        img = img.resize((1024, 576))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_base64

def create_svd_workflow():
    image_base64 = load_image(IMAGE_FILE)
    
    prompt = {
        "3": {
            "inputs": {
                "ckpt_name": "svd_xt.safetensors"
            },
            "class_type": "ImageOnlyCheckpointLoader",
            "_meta": {"title": "Load SVD Model"}
        },
        "4": {
            "inputs": {
                "clip_name": "open_clip_pytorch_model.bin"
            },
            "class_type": "CLIPVisionLoader",
            "_meta": {"title": "Load CLIP Vision"}
        },
        "5": {
            "inputs": {
                "vae_name": "sd-vae-ft-mse.safetensors"
            },
            "class_type": "VAELoader",
            "_meta": {"title": "Load VAE"}
        },
        "6": {
            "inputs": {
                "image": {"image": image_base64, "mask": ""}
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Input Image"}
        },
        "7": {
            "inputs": {
                "clip_vision": ["4", 0],
                "init_image": ["6", 0],
                "vae": ["5", 0],
                "width": 1024,
                "height": 576,
                "video_frames": 24,
                "motion_bucket_id": 127,
                "fps": 24,
                "augmentation_level": 0.0
            },
            "class_type": "SVD_img2vid_Conditioning",
            "_meta": {"title": "SVD Conditioning"}
        },
        "8": {
            "inputs": {
                "model": ["3", 0],
                "seed": 12345,
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "normal",
                "positive": ["7", 0],
                "negative": ["7", 1],
                "latent_image": ["7", 2]
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler"}
        },
        "9": {
            "inputs": {
                "samples": ["8", 0],
                "vae": ["5", 0]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "Decode VAE"}
        },
        "10": {
            "inputs": {
                "video": ["9", 0],
                "fps": 24
            },
            "class_type": "CreateVideo",
            "_meta": {"title": "Create Video"}
        },
        "11": {
            "inputs": {
                "filename_prefix": "svd_test",
                "video": ["10", 0],
                "fps": 24
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "Save Video"}
        }
    }
    
    return prompt

def main():
    prompt = create_svd_workflow()
    
    print("Testing SVD Turbo workflow")
    print(f"API URL: {COMFYUI_URL}")
    print()
    
    try:
        # Test connection
        response = requests.get(f"{COMFYUI_URL}/system_stats")
        if response.status_code == 200:
            print("Connected to ComfyUI")
            print()
        else:
            print(f"Failed to connect to ComfyUI: {response.status_code}")
            return
            
        # Queue prompt
        response = requests.post(
            f"{COMFYUI_URL}/prompt",
            json={"prompt": prompt},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            print("Prompt queued successfully")
            data = response.json()
            print(f"Prompt ID: {data['prompt_id']}")
            print()
        else:
            print(f"Failed to queue prompt: {response.status_code}")
            print(response.text)
            return
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
