# Fix para obtener modelos correctamente desde ComfyUI
# Este archivo contiene las funciones corregidas

def get_models_from_comfyui():
    """Obtiene los nombres exactos de modelos desde ComfyUI API"""
    import requests
    
    comfy_url = "http://127.0.0.1:8188"
    result = {
        "unets": [],
        "vaes": [],
        "clip_vision": []
    }
    
    try:
        # Obtener UNETs
        response = requests.get(f"{comfy_url}/object_info/UNETLoader", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "UNETLoader" in data:
                result["unets"] = data["UNETLoader"]["input"]["required"].get("unet_name", [[]])[0]
    except Exception as e:
        print(f"Error getting UNETs: {e}")
    
    try:
        # Obtener VAEs
        response = requests.get(f"{comfy_url}/object_info/VAELoader", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "VAELoader" in data:
                result["vaes"] = data["VAELoader"]["input"]["required"].get("vae_name", [[]])[0]
    except Exception as e:
        print(f"Error getting VAEs: {e}")
    
    try:
        # Obtener CLIP Vision
        response = requests.get(f"{comfy_url}/object_info/CLIPVisionLoader", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "CLIPVisionLoader" in data:
                result["clip_vision"] = data["CLIPVisionLoader"]["input"]["required"].get("clip_name", [[]])[0]
    except Exception as e:
        print(f"Error getting CLIP Vision: {e}")
    
    return result


def get_svd_turbo_workflow_fixed(image_filename, prompt, seed=None, width=720, height=480, frames=32, fps=24):
    """SVD Turbo workflow con modelos obtenidos desde ComfyUI"""
    
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    negative_prompt = "low quality, blurry, distorted, bad anatomy"
    
    # Obtener modelos desde ComfyUI
    models = get_models_from_comfyui()
    print(f"[SVD_TURBO] UNETs: {models['unets']}")
    print(f"[SVD_TURBO] VAEs: {models['vaes']}")
    
    # Seleccionar modelo SVD
    svd_model_name = None
    for unet in models["unets"]:
        if "svd" in unet.lower():
            svd_model_name = unet
            break
    
    # Seleccionar VAE (priorizar svd_xt_image_decoder)
    svd_vae = None
    for vae in models["vaes"]:
        if "svd_xt_image_decoder" in vae.lower():
            svd_vae = vae
            break
    
    if not svd_vae:
        for vae in models["vaes"]:
            if vae not in ["pixel_space", "taesd"]:
                svd_vae = vae
                break
    
    print(f"[SVD_TURBO] Seleccionado: {svd_model_name}, VAE: {svd_vae}")
    
    return {
        "1": {"inputs": {"image": image_filename, "upload": "image"}, "class_type": "LoadImage"},
        "2": {"inputs": {"unet_name": svd_model_name, "weight_dtype": "default"}, "class_type": "UNETLoader"},
        "3": {"inputs": {"vae_name": svd_vae}, "class_type": "VAELoader"},
        "4": {"inputs": {"clip_name": "open_clip_pytorch_model.bin"}, "class_type": "CLIPVisionLoader"},
        "5": {"inputs": {
            "clip_vision": ["4", 0],
            "init_image": ["1", 0],
            "vae": ["3", 0],
            "width": width,
            "height": height,
            "video_frames": frames,
            "motion_bucket_id": 127,
            "fps": fps,
            "augmentation_level": 0.0
        }, "class_type": "SVD_img2vid_Conditioning"},
        "6": {"inputs": {
            "model": ["2", 0],
            "positive": ["5", 0],
            "negative": ["5", 1],
            "latent_image": ["5", 2],
            "seed": seed,
            "steps": 10,
            "cfg": 1.0,
            "sampler_name": "euler_ancestral",
            "scheduler": "normal",
            "denoise": 1.0
        }, "class_type": "KSampler"},
        "7": {"inputs": {"vae": ["3", 0], "samples": ["6", 0]}, "class_type": "VAEDecode"},
        "8": {"inputs": {
            "images": ["7", 0],
            "frame_rate": float(fps),
            "loop_count": 0,
            "format": "video/h264-mp4",
            "output_format": "mp4",
            "filename_prefix": "ComfyUI",
            "pix_fmt": "yuv420p",
            "crf": 20,
            "save_metadata": True,
            "pingpong": False,
            "save_output": True
        }, "class_type": "VHS_VideoCombine"},
    }


if __name__ == "__main__":
    # Test
    models = get_models_from_comfyui()
    print("Available models:")
    print("UNETs:", models["unets"])
    print("VAEs:", models["vaes"])
    print("CLIP Vision:", models["clip_vision"])
