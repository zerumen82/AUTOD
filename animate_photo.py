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

    def build_wan_workflow(self, image_path, prompt, frames=81, fps=16, steps=25):
        seed = int(time.time()) % 1000000
        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": image_path}},
            "2": {"class_type": "VAELoader", "inputs": {"vae_name": "wan2.2_vae.safetensors"}},
            "3": {
                "class_type": "WanVideoImageToVideo",
                "inputs": {
                    "seed": seed, "steps": steps, "cfg": 5.0, "frames": frames,
                    "height": 480, "width": 832,
                    "image": ["1", 0], "vae": ["2", 0],
                    "positive": ["5", 0], "negative": ["5", 0]
                }
            },
            "4": {"class_type": "WanVideoDecode", "inputs": {"vae": ["2", 0], "samples": ["3", 0]}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["6", 0]}},
            "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
            "7": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "model_type": "wan"}},
            "8": {"class_type": "VideoCombine", "inputs": {
                "frame_rate": fps, "images": ["4", 0], "filename_prefix": "WanAnim", "format": "video/mp4"
            }},
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
                        files = node_out.get("gifs", []) or node_out.get("videos", [])
                        for f in files:
                            if isinstance(f, dict) and f.get("filename", "").startswith("WanAnim"):
                                r2 = requests.get(f"{self.base}/view", params={"filename": f["filename"], "subfolder": f.get("subfolder", ""), "type": f.get("type", "output")})
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