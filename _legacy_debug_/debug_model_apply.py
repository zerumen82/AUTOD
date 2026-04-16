import sys
import os
import requests
import json

print("=== Debug Apply Model ===")
print(f"Python version: {sys.version}")

# First let's get the object info
try:
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        object_info = response.json()
        print(f"ComfyUI object info retrieved successfully")
        print(f"Number of nodes: {len(object_info)}")
        
        video_nodes = [n for n in object_info if 'SVD' in n or 'Video' in n or 'UNET' in n]
        print(f"\nVideo-related nodes ({len(video_nodes)}):")
        for node in sorted(video_nodes):
            print(f"  - {node}")
            
    else:
        print(f"Failed to get object info: {response.status_code}")
        
except Exception as e:
    print(f"Error connecting to ComfyUI: {e}")
    sys.exit(1)

# Now let's create a debug workflow
test_image = os.path.abspath("testdata/test1.jpg")
print(f"\nTest image: {test_image}")

workflow = {
    "1": {"inputs": {"image": test_image, "upload": "image"}, "class_type": "LoadImage"},
    
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
        "steps": 1,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1.0
    }, "class_type": "KSampler"},
    
    "7": {"inputs": {"vae": ["3", 0], "samples": ["6", 0]}, "class_type": "VAEDecode"},
    
    "8": {"inputs": {"images": ["7", 0], "fps": 24.0}, "class_type": "CreateVideo"},
    
    "9": {"inputs": {
        "video": ["8", 0],
        "filename_prefix": "Debug_SVD",
        "format": "mp4",
        "codec": "auto"
    }, "class_type": "SaveVideo"},
}

payload = {"prompt": workflow}

print(f"\nWorkflow created, sending to ComfyUI...")

try:
    response = requests.post("http://127.0.0.1:8188/prompt", json=payload, timeout=60)
    if response.status_code == 200:
        result = response.json()
        prompt_id = result.get("prompt_id")
        if prompt_id:
            print(f"Prompt sent successfully, ID: {prompt_id}")
            
            # Wait for completion
            import time
            print("Waiting for completion...")
            while True:
                time.sleep(2)
                history_response = requests.get(f"http://127.0.0.1:8188/history/{prompt_id}", timeout=5)
                if history_response.status_code == 200:
                    history = history_response.json()
                    if prompt_id in history:
                        status = history[prompt_id].get("status", {})
                        print(f"Status: {json.dumps(status, ensure_ascii=False, indent=2)}")
                        if status.get("completed", False):
                            print("Process completed!")
                            break
                        if "messages" in status:
                            for msg in status["messages"]:
                                if msg[0] == "execution_error":
                                    print(f"\nERROR: {msg[1]['node_id']}")
                                    print(f"Node type: {msg[1]['node_type']}")
                                    print(f"Message: {msg[1]['exception_message']}")
                                    print("\nTraceback:")
                                    print("\n".join(msg[1]['traceback']))
                                    sys.exit(1)
    else:
        print(f"Error sending prompt: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    print(traceback.format_exc())
