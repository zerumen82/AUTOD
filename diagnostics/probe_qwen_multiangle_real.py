"""Test the downloaded Qwen 2511 camera LoRA on the benign office photo."""

import io
import time
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMFY = "http://127.0.0.1:8188"


def main():
    source = Image.open(ROOT / "diagnostics/img_editor_scene_office_final.png").convert("RGB")
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")
    buffer.seek(0)
    uploaded = requests.post(
        f"{COMFY}/upload/image",
        files={"image": ("qwen_multiangle_office.png", buffer, "image/png")},
        timeout=60,
    )
    uploaded.raise_for_status()
    image_name = uploaded.json()["name"]
    prompt = "<sks> front view high-angle shot wide shot"
    nodes = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "qwen-image-edit-2511-Q2_K.gguf"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2", 0],
            "lora_name": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
            "strength_model": 0.9,
        }},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["3", 0],
            "lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
            "strength_model": 1.0,
        }},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["4", 0], "shift": 3.1}},
        "6": {"class_type": "CFGNorm", "inputs": {"model": ["5", 0], "strength": 1.0}},
        "7": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default",
        }},
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "9": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}},
        "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["8", 0]}},
        "11": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["7", 0], "prompt": prompt, "vae": ["8", 0], "image1": ["9", 0],
        }},
        "12": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["7", 0], "prompt": "", "vae": ["8", 0], "image1": ["9", 0],
        }},
        "13": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["11", 0], "reference_latents_method": "index_timestep_zero",
        }},
        "14": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["12", 0], "reference_latents_method": "index_timestep_zero",
        }},
        "15": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0], "positive": ["13", 0], "negative": ["14", 0],
            "latent_image": ["10", 0], "seed": 123456, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0,
        }},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["8", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {
            "images": ["16", 0], "filename_prefix": "QwenMultiAngleOffice",
        }},
    }
    queued = requests.post(f"{COMFY}/prompt", json={"prompt": nodes}, timeout=30)
    queued.raise_for_status()
    payload = queued.json()
    assert "prompt_id" in payload, payload
    prompt_id = payload["prompt_id"]
    print(f"Queued {prompt_id}: {prompt}", flush=True)
    deadline = time.time() + 900
    next_report = 0
    while time.time() < deadline:
        history = requests.get(f"{COMFY}/history/{prompt_id}", timeout=15).json()
        if prompt_id in history:
            entry = history[prompt_id]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise RuntimeError(status)
            images = entry.get("outputs", {}).get("17", {}).get("images", [])
            if images:
                meta = images[0]
                raw = requests.get(f"{COMFY}/view", params={
                    "filename": meta["filename"], "subfolder": meta.get("subfolder", ""),
                    "type": meta.get("type", "output"),
                }, timeout=60)
                raw.raise_for_status()
                target = ROOT / "diagnostics/editor_viewpoint_qwen_multiangle.png"
                target.write_bytes(raw.content)
                with Image.open(target) as result:
                    print(f"SAVED {target} {result.size}", flush=True)
                return
        if time.time() >= next_report:
            queue = requests.get(f"{COMFY}/queue", timeout=15).json()
            print(f"Waiting: running={len(queue.get('queue_running', []))} pending={len(queue.get('queue_pending', []))}", flush=True)
            next_report = time.time() + 15
        time.sleep(2)
    raise TimeoutError("Qwen multi-angle generation exceeded 15 minutes")


if __name__ == "__main__":
    main()
