import os, json, requests, time, io
from typing import Optional
from PIL import Image

COMFY_URL = "http://127.0.0.1:8188"
MODELS_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"


class AnimatePhoto:
    def __init__(self):
        self.base = COMFY_URL.rstrip("/")

    def _post(self, endpoint, **kw):
        r = requests.post(f"{self.base}{endpoint}", **kw, timeout=120)
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    def upload_image(self, image_pil, name=None):
        buf = io.BytesIO()
        image_pil.save(buf, format="PNG")
        buf.seek(0)
        fname = name or f"anim_{int(time.time())}.png"
        r = requests.post(f"{self.base}/upload/image", files={"image": (fname, buf, "image/png")})
        if r.status_code != 200:
            raise RuntimeError(f"Upload failed: {r.status_code}")
        return fname

    def check_comfyui_status(self):
        try:
            r = requests.get(f"{self.base}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False

    def _find_model(self, subdir, patterns, extensions=(".gguf", ".safetensors")):
        base = os.path.join(MODELS_DIR, subdir)
        if not os.path.exists(base):
            return None

        pattern_lw = [p.lower() for p in patterns]
        for root, _, files in os.walk(base):
            for filename in files:
                name_lw = filename.lower()
                if not name_lw.endswith(extensions):
                    continue
                rel = os.path.relpath(os.path.join(root, filename), base)
                rel_lw = rel.lower()
                if all(p in rel_lw for p in pattern_lw):
                    return rel
        return None

    def _normalize_wan_frames(self, frames):
        frames = max(17, int(frames or 81))
        remainder = (frames - 1) % 4
        if remainder:
            frames += 4 - remainder
        return frames

    def build_wan_workflow(self, image_path, prompt, frames=81, fps=16, steps=25):
        seed = int(time.time()) % 1000000
        prompt = (prompt or "").strip() or "natural gentle movement, subtle realistic motion"
        frames = self._normalize_wan_frames(frames)
        negative_prompt = (
            "low quality, blurry, distorted, bad anatomy, static, frozen, "
            "still image, no movement, low resolution, watermark, text"
        )

        wan_model = (
            self._find_model("diffusion_models", ["wan2.2"])
            or self._find_model("diffusion_models", ["wan2_2"])
            or self._find_model("diffusion_models", ["wan2.1"])
            or self._find_model("diffusion_models", ["wan2_1"])
            or self._find_model("diffusion_models", ["wan"])
        )
        vae_name = (
            self._find_model("vae", ["wan2", "vae"])
            or self._find_model("vae", ["wan", "vae"])
            or "Wan2.2_VAE.safetensors"
        )
        t5_name = (
            self._find_model("text_encoders", ["umt5"])
            or self._find_model("text_encoders", ["t5"])
            or "umt5-xxl-enc-bf16.safetensors"
        )

        if not wan_model:
            raise RuntimeError("No se encontró modelo Wan en models/diffusion_models")

        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": image_path}},
            "2": {
                "class_type": "WanVideoModelLoader",
                "inputs": {
                    "model": wan_model,
                    "base_precision": "bf16",
                    "quantization": "disabled",
                    "load_device": "offload_device"
                }
            },
            "3": {
                "class_type": "WanVideoVAELoader",
                "inputs": {"model_name": vae_name, "precision": "bf16"}
            },
            "4": {
                "class_type": "LoadWanVideoT5TextEncoder",
                "inputs": {
                    "model_name": t5_name,
                    "precision": "bf16",
                    "load_device": "offload_device",
                    "quantization": "disabled"
                }
            },
            "5": {
                "class_type": "WanVideoTextEncode",
                "inputs": {
                    "positive_prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "t5": ["4", 0],
                    "force_offload": True,
                    "model_to_offload": ["2", 0]
                }
            },
            "6": {
                "class_type": "WanVideoImageToVideoEncode",
                "inputs": {
                    "width": 832,
                    "height": 480,
                    "num_frames": frames,
                    "noise_aug_strength": 0.0,
                    "start_latent_strength": 1.0,
                    "end_latent_strength": 1.0,
                    "force_offload": True,
                    "vae": ["3", 0],
                    "start_image": ["1", 0],
                    "fun_or_fl2v_model": True
                }
            },
            "7": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": ["2", 0],
                    "image_embeds": ["6", 0],
                    "text_embeds": ["5", 0],
                    "steps": steps,
                    "cfg": 5.0,
                    "shift": 5.0,
                    "seed": seed,
                    "scheduler": "unipc",
                    "riflex_freq_index": 0,
                    "force_offload": True
                }
            },
            "8": {
                "class_type": "WanVideoDecode",
                "inputs": {
                    "vae": ["3", 0],
                    "samples": ["7", 0],
                    "enable_vae_tiling": False,
                    "tile_x": 272,
                    "tile_y": 272,
                    "tile_stride_x": 144,
                    "tile_stride_y": 128
                }
            },
            "9": {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["8", 0],
                    "fps": fps
                }
            },
            "10": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["9", 0],
                    "filename_prefix": "WanAnim",
                    "format": "mp4",
                    "codec": "auto"
                }
            },
        }

    def animate_image(self, image_pil=None, prompt="", output_path="output.mp4",
                      model="wan_video", frames=81, fps=16, timeout=600):
        if not self.check_comfyui_status():
            return False

        if image_pil is None:
            return False

        w, h = image_pil.size
        nw, nh = (w // 16) * 16, (h // 16) * 16
        if nw > 832 or nh > 480:
            scale = min(832 / nw, 480 / nh)
            nw, nh = int(nw * scale * 0.5) * 2, int(nh * scale * 0.5) * 2
        if (nw, nh) != (w, h):
            image_pil = image_pil.resize((nw, nh), Image.LANCZOS)

        fname = self.upload_image(image_pil)
        if model != "wan_video":
            print(f"[AnimatePhoto] Motor {model} no implementado aquí; usando WanVideo para respetar prompt.")

        wf = self.build_wan_workflow(fname, prompt, frames, fps)

        r = requests.post(f"{self.base}/prompt", json={"prompt": wf})
        if r.status_code != 200:
            return False
        pid = r.json().get("prompt_id")
        if not pid:
            return False

        t0 = time.time()
        last_pct = -1
        while time.time() - t0 < timeout:
            r = requests.get(f"{self.base}/history/{pid}")
            if r.status_code == 200 and pid in r.json():
                hist = r.json()[pid]
                if "outputs" in hist:
                    for node_id, node_out in hist["outputs"].items():
                        files = node_out.get("gifs", []) or node_out.get("videos", []) or node_out.get("images", [])
                        for f in files:
                            filename = f.get("filename", "") if isinstance(f, dict) else ""
                            if isinstance(f, dict) and (filename.startswith("WanAnim") or filename.lower().endswith(".mp4")):
                                r2 = requests.get(
                                    f"{self.base}/view",
                                    params={
                                        "filename": filename,
                                        "subfolder": f.get("subfolder", ""),
                                        "type": f.get("type", "output")
                                    }
                                )
                                if r2.status_code == 200:
                                    with open(output_path, "wb") as fw:
                                        fw.write(r2.content)
                                    return True
                if hist.get("status") == "failed":
                    return False
            elapsed = int(time.time() - t0)
            if elapsed // 10 > last_pct:
                last_pct = elapsed // 10
            time.sleep(2)
        return False
