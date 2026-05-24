import os, json, requests, time, io, cv2, gc
import numpy as np
from typing import Optional
from PIL import Image
from safetensors import safe_open
from roop.utils import get_vram_gb

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

COMFY_URL = "http://127.0.0.1:8188"
MODELS_DIR = os.path.join(get_project_root(), "ui", "tob", "ComfyUI", "models")


def _get_vae_channels(vae_path):
    """Check VAE output channels from safetensors/pth file without full load."""
    if not os.path.exists(vae_path):
        return None
    try:
        if vae_path.endswith(".safetensors"):
            with safe_open(vae_path, framework="pt") as f:
                key = "model.conv2.weight" if "model.conv2.weight" in f.keys() else "conv2.weight"
                return f.get_tensor(key).shape[0]
        else:
            import torch
            sd = torch.load(vae_path, map_location="cpu", weights_only=True)
            key = "model.conv2.weight" if "model.conv2.weight" in sd else "conv2.weight"
            return sd[key].shape[0]
    except Exception as e:
        print(f"[VAE] Error checking channels: {e}")
        return None

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

    def _get_node_input_options(self, node_name, input_name):
        try:
            r = requests.get(f"{self.base}/object_info/{node_name}", timeout=5)
            if r.status_code != 200:
                return []
            node_info = r.json().get(node_name, {})
            required = node_info.get("input", {}).get("required", {})
            value = required.get(input_name, [])
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], list):
                return value[0]
        except Exception as e:
            print(f"[AnimatePhoto] No se pudieron leer opciones de {node_name}.{input_name}: {e}")
        return []

    def _select_wan_t5_encoder(self):
        available = self._get_node_input_options("LoadWanVideoT5TextEncoder", "model_name")
        if available:
            priorities = (
                ("umt5", "xxl", "fp8"),
                ("umt5", "xxl"),
                ("umt5",),
            )
            for patterns in priorities:
                match = next((name for name in available if all(p in name.lower() for p in patterns)), None)
                if match:
                    return match
            print(f"[AnimatePhoto] ADVERTENCIA: no hay UMT5 en ComfyUI. Opciones T5: {available}")

        return (
            self._find_model("text_encoders", ["umt5", "xxl", "fp8"], extensions=(".safetensors",))
            or self._find_model("text_encoders", ["umt5", "xxl"], extensions=(".safetensors",))
            or self._find_model("text_encoders", ["umt5"], extensions=(".safetensors",))
            or "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        )

    def _normalize_wan_frames(self, frames):
        frames = max(17, int(frames or 33))
        remainder = (frames - 1) % 4
        if remainder:
            frames += 4 - remainder
        return frames

    def build_wan_workflow(self, image_path, prompt, frames=33, fps=16, steps=15, cfg=4.0, width=512, height=512):
        seed = int(time.time()) % 1000000
        prompt = (prompt or "").strip() or "natural gentle movement, subtle realistic motion"
        frames = self._normalize_wan_frames(frames)
        negative_prompt = (
            "low quality, blurry, distorted, bad anatomy, static, frozen, "
            "still image, no movement, low resolution, watermark, text"
        )

        vram_gb = get_vram_gb()
        prefer_5b = vram_gb <= 8
        if prefer_5b:
            # Preferir 5B (TI2V) en GPUs con ≤8GB VRAM
            wan_model = (
                self._find_model("diffusion_models", ["ti2v", "5b", "fp8"], extensions=(".safetensors",))
                or self._find_model("diffusion_models", ["wan2_2", "ti2v", "5b"], extensions=(".safetensors",))
                or self._find_model("diffusion_models", ["wan2.2", "ti2v", "5b"], extensions=(".safetensors",))
                or self._find_model("unet", ["wan2.2", "ti2v", "5b"], extensions=(".gguf",))
                or self._find_model("unet", ["ti2v", "5b"], extensions=(".gguf",))
                or self._find_model("unet", ["wan2.2", "ti2v"], extensions=(".gguf",))
                or self._find_model("unet", ["ti2v"], extensions=(".gguf",))
                or self._find_model("diffusion_models", ["ti2v", "5b"])
                or self._find_model("diffusion_models", ["wan2_2", "ti2v", "5b"])
                or self._find_model("diffusion_models", ["wan2.2", "ti2v", "5b"])
                or self._find_model("unet", ["wan2.2", "i2v"], extensions=(".gguf",))
                or self._find_model("unet", ["i2v"], extensions=(".gguf",))
                or self._find_model("diffusion_models", ["wan2.2"])
                or self._find_model("diffusion_models", ["wan2_2"])
                or self._find_model("diffusion_models", ["wan2.1"])
                or self._find_model("diffusion_models", ["wan"])
            )
        else:
            wan_model = (
                self._find_model("unet", ["wan2.2", "i2v"], extensions=(".gguf",))
                or self._find_model("unet", ["wan2.2", "ti2v"], extensions=(".gguf",))
                or self._find_model("unet", ["ti2v"], extensions=(".gguf",))
                or self._find_model("unet", ["i2v"], extensions=(".gguf",))
                or self._find_model("diffusion_models", ["wan2.2"])
                or self._find_model("diffusion_models", ["wan2_2"])
                or self._find_model("diffusion_models", ["wan2.1"])
                or self._find_model("diffusion_models", ["wan2_1"])
                or self._find_model("diffusion_models", ["wan"])
            )

        vae_extensions = (".gguf", ".safetensors", ".pth")
        is_14b_model = "14b" in (wan_model or "").lower()
        is_5b_ti2v = "5b" in (wan_model or "").lower() and "ti2v" in (wan_model or "").lower()
        if is_14b_model:
            vae_name = (
                self._find_model("vae", ["wan2.1", "vae"], extensions=vae_extensions)
                or self._find_model("vae", ["wan2.1"], extensions=vae_extensions)
                or self._find_model("vae", ["wan", "vae"], extensions=vae_extensions)
                or "Wan2.1_VAE.pth"
            )
        elif is_5b_ti2v:
            # 5B TI2V necesita VAE de 48 canales (WanVideoVAE38)
            # NOTA: ["wan2.2", "vae"] NO matchea "Wan2.1_VAE.pth" (16ch), solo "wan2.2_vae" (48ch)
            vae_name = (
                self._find_model("vae", ["wan2.2", "vae38"], extensions=vae_extensions)
                or self._find_model("vae", ["wan2.2", "48"], extensions=vae_extensions)
                or self._find_model("vae", ["wan2.2", "vae"], extensions=vae_extensions)
                or self._find_model("vae", ["wan2", "vae"], extensions=vae_extensions)
                or self._find_model("vae", ["wan2"], extensions=vae_extensions)
                or self._find_model("vae", ["wan", "vae"], extensions=vae_extensions)
                or "Wan2.2_VAE_bf16.safetensors"
            )
        else:
            vae_name = (
                self._find_model("vae", ["wan2", "vae"], extensions=vae_extensions)
                or self._find_model("vae", ["wan2"], extensions=vae_extensions)
                or self._find_model("vae", ["wan", "vae"], extensions=vae_extensions)
                or "Wan2.2_VAE.safetensors"
            )

        # Validar canales del VAE vs modelo
        vae_path = os.path.join(MODELS_DIR, "vae", vae_name)
        vae_channels = _get_vae_channels(vae_path)
        expected_channels = 48 if (is_5b_ti2v or (not is_14b_model and vae_channels == 48)) else 16
        if vae_channels is not None and is_5b_ti2v and vae_channels != 48:
            print(f"[VAE] ADVERTENCIA: VAE {vae_name} tiene {vae_channels} canales, pero el modelo 5B TI2V necesita 48. Buscando VAE alternativo...")
            alt_vae = (
                self._find_model("vae", ["wan2.2", "vae"], extensions=vae_extensions)
                or self._find_model("vae", ["wan2", "vae"], extensions=vae_extensions)
                or self._find_model("vae", ["wan2"], extensions=vae_extensions)
                or self._find_model("vae", ["wan", "vae"], extensions=vae_extensions)
            )
            if alt_vae:
                alt_path = os.path.join(MODELS_DIR, "vae", alt_vae)
                alt_channels = _get_vae_channels(alt_path)
                if alt_channels == 48:
                    vae_name = alt_vae
                    print(f"[VAE] Usando VAE alternativo: {vae_name} ({alt_channels} canales)")
                else:
                    print(f"[VAE] VAE alternativo también tiene {alt_channels} canales. Forzando canales.")
        elif vae_channels is not None and is_14b_model and vae_channels != 16:
            print(f"[VAE] ADVERTENCIA: VAE {vae_name} tiene {vae_channels} canales, modelo 14B espera 16.")

        # Determinación de modo arquitectónico
        # - Modelos 5B TI2V: NO usan fun_mode (usam flujo Flow Matching puro)
        # - Modelos 14B/1.3B: SI usan fun_mode (usam flujo de concatenación)
        use_fun_mode = is_14b_model

        # WanVideoWrapper espera UMT5 en safetensors. Los T5 GGUF de Flux no validan en este nodo.
        t5_name = self._select_wan_t5_encoder()
        
        print(f"[AnimatePhoto] Config: model={wan_model}, vae={vae_name}, is_5b={is_5b_ti2v}, vae_ch={vae_channels}, fun_mode={use_fun_mode}, t5={t5_name}")
        print(f"[AnimatePhoto] Sampler: flowmatch_pusa, Shift=8.0 (5B)" if is_5b_ti2v else "[AnimatePhoto] Sampler: UniPC, Shift=5.0 (14B)")

        nodes = {
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
        }
        node_model_ref = ["2", 0]

        if is_5b_ti2v:
            image_embeds_node = "11"
            nodes.update({
                "6": {
                    "class_type": "WanVideoEncode",
                    "inputs": {
                        "vae": ["3", 0],
                        "image": ["1", 0],
                        "enable_vae_tiling": False,
                        "tile_x": 272,
                        "tile_y": 272,
                        "tile_stride_x": 144,
                        "tile_stride_y": 128,
                        "noise_aug_strength": 0.0,
                        "latent_strength": 1.0
                    }
                },
                "11": {
                    "class_type": "WanVideoEmptyEmbeds",
                    "inputs": {
                        "width": width,
                        "height": height,
                        "num_frames": frames,
                        "extra_latents": ["6", 0]
                    }
                }
            })
            context_ref = ["12", 0] if frames > 49 else None
            if context_ref:
                nodes["12"] = {
                    "class_type": "WanVideoContextOptions",
                    "inputs": {
                        "context_schedule": "uniform_standard",
                        "context_frames": 49,
                        "context_stride": 4,
                        "context_overlap": 24,
                        "freenoise": True,
                        "verbose": False,
                        "fuse_method": "linear"
                    }
                }
        else:
            image_embeds_node = "6"
            context_ref = None
            nodes.update({
                "6": {
                    "class_type": "WanVideoImageToVideoEncode",
                    "inputs": {
                        "width": width,
                        "height": height,
                        "num_frames": frames,
                        "noise_aug_strength": 0.0,
                        "start_latent_strength": 1.0,
                        "end_latent_strength": 1.0,
                        "force_offload": True,
                        "vae": ["3", 0],
                        "start_image": ["1", 0],
                        "fun_or_fl2v_model": use_fun_mode
                    }
                }
            })

        nodes.update({
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
                    "quantization": "fp8_e4m3fn" if get_vram_gb() <= 12 else "disabled"
                }
            },
            "5": {
                "class_type": "WanVideoTextEncode",
                "inputs": {
                    "positive_prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "t5": ["4", 0],
                    "force_offload": True,
                    "model_to_offload": node_model_ref
                }
            },
            "7": {
                "class_type": "WanVideoSampler",
                "inputs": {
                    "model": node_model_ref,
                    "image_embeds": [image_embeds_node, 0],
                    "text_embeds": ["5", 0],
                    "steps": steps,
                    "cfg": cfg,
                    "shift": 8.0 if is_5b_ti2v else 5.0,
                    "seed": seed,
                    "scheduler": "flowmatch_pusa" if is_5b_ti2v else "unipc",
                    "riflex_freq_index": 0,
                    "force_offload": True
                }
            },
            "8": {
                "class_type": "WanVideoDecode",
                "inputs": {
                    "vae": ["3", 0],
                    "samples": ["7", 0],
                    "enable_vae_tiling": False if is_5b_ti2v else True,
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
        })

        if context_ref:
            nodes["7"]["inputs"]["context_options"] = context_ref

        return nodes

    def check_progress(self, pid):
        try:
            r = requests.get(f"{self.base}/progress", timeout=5)
            if r.status_code == 200:
                data = r.json()
                if data.get("running_prompt") == pid:
                    return {
                        "progress": data.get("progress", 0),
                        "eta": data.get("eta_remaining", 0),
                        "step": data.get("current_step", 0),
                        "total": data.get("total_steps", 0),
                    }
            h = requests.get(f"{self.base}/history/{pid}", timeout=5)
            if h.status_code == 200 and pid in h.json():
                return {"done": True}
            return None
        except:
            return None

    def animate_image(self, image_pil=None, prompt="", output_path="output.mp4",
                      model="wan_video", frames=33, fps=16, steps=15, cfg=4.0, timeout=1800):
        if not self.check_comfyui_status():
            print("[AnimatePhoto] ComfyUI no responde en /system_stats")
            return False

        if image_pil is None:
            print("[AnimatePhoto] image_pil es None")
            return False

        vram_gb = get_vram_gb()
        if vram_gb <= 8:
            steps = min(steps, 20)
            frames = min(frames, 81)

        w, h = image_pil.size
        MAX_DIM = 512
        scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)
        nw = int(w * scale / 32) * 32
        nh = int(h * scale / 32) * 32
        nw = max(nw, 64)
        nh = max(nh, 64)
        if (nw, nh) != (w, h):
            image_pil = image_pil.resize((nw, nh), Image.LANCZOS)

        print(f"[AnimatePhoto] Resized: {w}x{h} -> {nw}x{nh}, frames={frames}, steps={steps}, cfg={cfg}")

        fname = self.upload_image(image_pil)
        if model != "wan_video":
            print(f"[AnimatePhoto] Motor {model} no implementado aquí; usando WanVideo para respetar prompt.")

        wf = self.build_wan_workflow(fname, prompt, frames, fps, steps, cfg, width=nw, height=nh)

        r = requests.post(f"{self.base}/prompt", json={"prompt": wf})
        if r.status_code != 200:
            print(f"[AnimatePhoto] ComfyUI prompt error ({r.status_code}): {r.text[:300]}")
            return False
        pid = r.json().get("prompt_id")
        if not pid:
            print(f"[AnimatePhoto] No prompt_id en respuesta: {r.text[:200]}")
            return False

        t0 = time.time()
        while time.time() - t0 < timeout:
            r = requests.get(f"{self.base}/history/{pid}")
            if r.status_code == 200 and pid in r.json():
                hist = r.json()[pid]
                if hist.get("status") == "failed":
                    error_info = hist.get("status_str") or hist.get("error_message") or "Error desconocido"
                    node_errors = hist.get("errors", {})
                    if node_errors:
                        for nid, err in node_errors.items():
                            error_info += f" | nodo {nid}: {err}"
                    print(f"[AnimatePhoto] ComfyUI error: {error_info}")
                    return False
                if "outputs" in hist and hist["outputs"]:
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
                                    elapsed = time.time() - t0
                                    print(f"[AnimatePhoto] Hecho en {elapsed:.0f}s")
                                    return True
            time.sleep(1)
        print(f"[AnimatePhoto] Timeout tras {timeout}s")
        return False
