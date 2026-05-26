import os, json, requests, time, io, cv2, gc, math
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

    def build_wan_workflow(self, image_path, prompt, frames=33, fps=16, steps=25, cfg=5.5, width=512, height=512, t2v_mode=False):
        seed = int(time.time()) % 1000000
        prompt = (prompt or "").strip() or "cinematic motion, dynamic camera movement, natural flowing movement, 24fps, gentle motion, realistic video"
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
        print(f"[AnimatePhoto] Sampler: flowmatch_pusa, Shift=5.0" if is_5b_ti2v else "[AnimatePhoto] Sampler: UniPC, Shift=5.0")

        nodes = {}
        if not t2v_mode:
            nodes["1"] = {"class_type": "LoadImage", "inputs": {"image": image_path}}
        nodes["2"] = {
            "class_type": "WanVideoModelLoader",
            "inputs": {
                "model": wan_model,
                "base_precision": "bf16",
                "quantization": "disabled",
                "load_device": "offload_device"
            }
        }
        node_model_ref = ["2", 0]

        image_embeds_node = "6"
        context_ref = ["12", 0] if is_5b_ti2v else None
        if t2v_mode:
            # T2V puro: sin imagen de entrada, usa WanVideoEmptyEmbeds
            nodes["6"] = {
                "class_type": "WanVideoEmptyEmbeds",
                "inputs": {
                    "width": width,
                    "height": height,
                    "num_frames": frames,
                }
            }
        else:
            nodes["6"] = {
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
        if context_ref:
            ctx_frames = min(33, max(17, frames - 16))
            print(f"[AnimatePhoto] Context windows: {ctx_frames} frames, stride=4, overlap=24 (total={frames})")
            nodes["12"] = {
                "class_type": "WanVideoContextOptions",
                "inputs": {
                    "context_schedule": "uniform_standard",
                    "context_frames": ctx_frames,
                    "context_stride": 4,
                    "context_overlap": 24,
                    "freenoise": True,
                    "verbose": False,
                    "fuse_method": "linear"
                }
            }

        # 5B TI2V: CFG más alto para compensar FP8 (CFG scheduling vía lista no funciona en API REST)
        if is_5b_ti2v and isinstance(cfg, (int, float)):
            cfg = min(cfg * 1.1, 7.5)

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
                    "shift": 5.0,
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
                    "enable_vae_tiling": True,
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

    def _ensure_framepack_models(self):
        models_dir = os.path.join(get_project_root(), "ui", "tob", "ComfyUI", "models")
        clip_dir = os.path.join(models_dir, "clip")
        vae_dir = os.path.join(models_dir, "vae", "hyvid")
        cv_dir = os.path.join(models_dir, "clip_vision")

        missing = []
        if not os.path.exists(os.path.join(clip_dir, "clip_l.safetensors")):
            missing.append("ComfyUI/models/clip/clip_l.safetensors")
        if not os.path.exists(os.path.join(clip_dir, "llava_llama3_fp16.safetensors")):
            missing.append("ComfyUI/models/clip/llava_llama3_fp16.safetensors")
        if not os.path.exists(os.path.join(vae_dir, "hunyuan_video_vae_bf16_repack.safetensors")):
            missing.append("ComfyUI/models/vae/hyvid/hunyuan_video_vae_bf16_repack.safetensors")
        if not os.path.exists(os.path.join(cv_dir, "sigclip_vision_patch14_384.safetensors")):
            missing.append("ComfyUI/models/clip_vision/sigclip_vision_patch14_384.safetensors")

        if missing:
            print(f"[FramePack] ADVERTENCIA: Faltan {len(missing)} modelos. Descargue manualmente:")
            for m in missing:
                print(f"  - {m}")
            return False
        return True

    def build_framepack_workflow(self, image_path, prompt, frames=121, fps=24, steps=30, cfg=5.5, width=512, height=512):
        seed = int(time.time()) % 1000000
        prompt = (prompt or "").strip() or "cinematic motion, dynamic camera movement, natural flowing movement"
        negative_prompt = "low quality, blurry, distorted, bad anatomy, static, frozen, still image, no movement, low resolution, watermark, text"

        vram_gb = get_vram_gb()
        # Duracion real basada en frames/fps
        total_second_length = max(1.0, frames / float(fps))
        latent_window_size = 9

        use_fp8 = vram_gb <= 12
        gpu_memory_preservation = max(1.0, vram_gb - 4.0) if vram_gb > 0 else 6.0

        nodes = {}

        nodes["1"] = {"class_type": "DualCLIPLoader", "inputs": {
            "clip_name1": "clip_l.safetensors",
            "clip_name2": "llava_llama3_fp16.safetensors",
            "type": "hunyuan_video"
        }}

        nodes["2"] = {"class_type": "CLIPTextEncode", "inputs": {
            "text": prompt,
            "clip": ["1", 0]
        }}

        nodes["3"] = {"class_type": "CLIPTextEncode", "inputs": {
            "text": negative_prompt,
            "clip": ["1", 0]
        }}

        nodes["4"] = {"class_type": "VAELoader", "inputs": {
            "vae_name": "hyvid\\hunyuan_video_vae_bf16_repack.safetensors"
        }}

        nodes["5"] = {"class_type": "CLIPVisionLoader", "inputs": {
            "clip_name": "sigclip_vision_patch14_384.safetensors"
        }}

        nodes["6"] = {"class_type": "LoadImage", "inputs": {
            "image": image_path
        }}

        nodes["7"] = {"class_type": "CLIPVisionEncode", "inputs": {
            "clip_vision": ["5", 0],
            "image": ["6", 0],
            "crop": "center"
        }}

        nodes["8"] = {"class_type": "VAEEncode", "inputs": {
            "pixels": ["6", 0],
            "vae": ["4", 0]
        }}

        nodes["9"] = {"class_type": "LoadFramePackModel", "inputs": {
            "model": "FramePackI2V_HY_bf16.safetensors",
            "base_precision": "bf16",
            "quantization": "fp8_e4m3fn",
            "load_device": "offload_device",
            "attention_mode": "sdpa"
        }}

        guidance_scale = cfg if cfg > 1.5 else 6.0
        nodes["10"] = {"class_type": "FramePackSampler", "inputs": {
            "model": ["9", 0],
            "positive": ["2", 0],
            "negative": ["3", 0],
            "start_latent": ["8", 0],
            "image_embeds": ["7", 0],
            "steps": steps,
            "cfg": 1.0,
            "guidance_scale": guidance_scale,
            "shift": 3.0,
            "seed": seed,
            "use_teacache": True,
            "teacache_rel_l1_thresh": 0.15,
            "latent_window_size": latent_window_size,
            "total_second_length": total_second_length,
            "gpu_memory_preservation": gpu_memory_preservation,
            "sampler": "unipc_bh1",
            "embed_interpolation": "disabled"
        }}

        nodes["11"] = {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["10", 0],
            "vae": ["4", 0],
            "tile_size": 256,
            "overlap": 64,
            "temporal_size": 64,
            "temporal_overlap": 8
        }}

        nodes["12"] = {"class_type": "CreateVideo", "inputs": {
            "images": ["11", 0],
            "fps": fps
        }}

        nodes["13"] = {"class_type": "SaveVideo", "inputs": {
            "video": ["12", 0],
            "filename_prefix": "FramePack",
            "format": "mp4",
            "codec": "auto"
        }}

        print(f"[FramePack] Config: duration={total_second_length:.1f}s, window={latent_window_size}, steps={steps}, cfg={cfg}, guidance={guidance_scale:.1f}, fp8={use_fp8}, gpu_mem={gpu_memory_preservation:.1f}GB")
        return nodes

    def animate_image(self, image_pil=None, prompt="", output_path="output.mp4",
                      model="wan_video", frames=33, fps=16, steps=25, cfg=5.5, timeout=1800, t2v_mode=False):
        if not self.check_comfyui_status():
            print("[AnimatePhoto] ComfyUI no responde en /system_stats")
            return False

        if not t2v_mode and image_pil is None:
            print("[AnimatePhoto] image_pil es None")
            return False

        vram_gb = get_vram_gb()
        if vram_gb <= 8:
            frames = min(frames, 81)

        if t2v_mode:
            # T2V: sin imagen, usa dimensiones por defecto
            nw, nh = 384, 512
            fname = None
            print(f"[AnimatePhoto] T2V mode: {nw}x{nh}, frames={frames}, steps={steps}, cfg={cfg}")
        else:
            w, h = image_pil.size
            MAX_DIM = 704
            MIN_DIM = 384
            scale = min(MAX_DIM / w, MAX_DIM / h, 1.0)
            nw = int(w * scale / 32) * 32
            nh = int(h * scale / 32) * 32
            nw = max(nw, MIN_DIM)
            nh = max(nh, MIN_DIM)
            if (nw, nh) != (w, h):
                image_pil = image_pil.resize((nw, nh), Image.LANCZOS)

            print(f"[AnimatePhoto] Resized: {w}x{h} -> {nw}x{nh}, frames={frames}, steps={steps}, cfg={cfg}")

            fname = self.upload_image(image_pil)

        if model == "framepack":
            self._ensure_framepack_models()
            wf = self.build_framepack_workflow(fname, prompt, frames, fps, steps, cfg, width=nw, height=nh)
        elif model != "wan_video":
            print(f"[AnimatePhoto] Motor {model} no implementado aquí; usando WanVideo para respetar prompt.")
            wf = self.build_wan_workflow(fname, prompt, frames, fps, steps, cfg, width=nw, height=nh, t2v_mode=t2v_mode)
        else:
            wf = self.build_wan_workflow(fname, prompt, frames, fps, steps, cfg, width=nw, height=nh, t2v_mode=t2v_mode)

        r = requests.post(f"{self.base}/prompt", json={"prompt": wf})
        if r.status_code != 200:
            print(f"[AnimatePhoto] ComfyUI prompt error ({r.status_code}): {r.text[:300]}")
            return False
        pid = r.json().get("prompt_id")
        if not pid:
            print(f"[AnimatePhoto] No prompt_id en respuesta: {r.text[:200]}")
            return False

        t0 = time.time()
        last_progress_log = 0
        last_health_check = 0
        while True:
            prog = self.check_progress(pid)
            now = time.time()
            if prog:
                if prog.get("done"):
                    pass
                else:
                    pct = prog.get("progress", 0) * 100
                    step = prog.get("step", 0)
                    total = prog.get("total", 0)
                    eta = prog.get("eta", 0)
                    if now - last_progress_log > 15:
                        print(f"[AnimatePhoto] ComfyUI: {pct:.0f}% | step {step}/{total} | ETA {eta:.0f}s")
                        last_progress_log = now
            else:
                if now - last_health_check > 60:
                    if not self.check_comfyui_status():
                        print("[AnimatePhoto] ComfyUI dejó de responder. Abortando.")
                        return False
                    last_health_check = now
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
                            if isinstance(f, dict) and (filename.startswith("WanAnim") or filename.startswith("FramePack") or filename.lower().endswith(".mp4")):
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
                                    
                                    # --- SISTEMA DE AUDIO INTELIGENTE (DIRECTOR DE SONIDO IA) ---
                                    try:
                                        from scripts.image_analyzer_for_prompt import ImageAnalyzer
                                        from roop.audio_generator import generate_audio, generate_sound
                                        import subprocess
                                        
                                        print("[AnimatePhoto] IA Analizando escena para decidir el audio...")
                                        analyzer = ImageAnalyzer()
                                        info = analyzer.analyze(image_path)
                                        num_people = info.get('num_people', 0)
                                        
                                        audio_paths = []
                                        
                                        # LOGICA DE INTERPRETACION Y PERSONALIZACION DE VOZ
                                        if num_people >= 1:
                                            # Tomar datos de la primera persona para la voz principal
                                            p = info['faces'][0]
                                            gender = p.get('gender', 'unknown')
                                            age_group = p.get('age_group', 'adult')
                                            
                                            print(f"[AnimatePhoto] Perfil detectado: {gender}, {age_group}. Personalizando voz...")
                                            
                                            # Selección de "voz de referencia" (clonación virtual)
                                            speaker_ref = None
                                            # Intentar buscar en assets audios descriptivos
                                            ref_dir = os.path.join(os.getcwd(), "assets", "voices")
                                            if os.path.exists(ref_dir):
                                                # Mapeo simple: buscar archivos como 'female_senior.wav', 'male_adult.wav', etc.
                                                target_ref = f"{gender}_{age_group}.wav"
                                                if os.path.exists(os.path.join(ref_dir, target_ref)):
                                                    speaker_ref = os.path.join(ref_dir, target_ref)

                                            # Selección de exclamación natural según el perfil (SIN NARRADOR)
                                            short_phrases = {
                                                'female': ["¡Qué bueno!", "¡Vaya ritmo!", "¡Esto es genial!", "¡Jajaja, mira!"],
                                                'male': ["¡Eso es!", "¡Dale ahí!", "¡Increíble!", "¡Cómo mola!"],
                                                'senior': ["¡Qué maravilla!", "¡Qué bien!", "¡Así se hace!", "¡Qué alegría!"]
                                            }
                                            
                                            import random
                                            base_list = short_phrases.get(gender, short_phrases['male'])
                                            if age_group == 'senior': base_list = short_phrases['senior']
                                            
                                            # Elegir una frase corta o solo risas/ambiente
                                            dialogo = random.choice(base_list)
                                            
                                            print(f"[AnimatePhoto] Generando exclamación directa ({gender}): '{dialogo}'")
                                            voice = generate_audio(dialogo, lenguaje="Español", speaker_wav=speaker_ref)
                                            if voice: audio_paths.append(voice)
                                            
                                            # Sonido ambiental de la escena (Gente, pasos, risas)
                                            if "dance" in prompt.lower() or "bailar" in prompt.lower():
                                                bg_prompt = "rhythmic music, people laughing and cheering in a party, club atmosphere"
                                            else:
                                                bg_prompt = f"natural ambient sounds of {prompt}, people talking in the background"
                                                
                                            bg_sound = generate_sound(bg_prompt, duration=float(frames/fps))
                                            if bg_sound: audio_paths.append(bg_sound)
                                            
                                        elif num_people > 1 and ("dance" in prompt.lower() or "bailar" in prompt.lower()):
                                            print(f"[AnimatePhoto] Detectadas {num_people} personas. Ambiente de grupo.")
                                            music = generate_sound("rhythmic party music, people laughing and cheering", duration=float(frames/fps))
                                            if music: audio_paths.append(music)
                                        else:
                                            print("[AnimatePhoto] No se detectan personas. Sonido puramente ambiental.")
                                            ambient = generate_sound(f"natural sound of {prompt}, cinematic soundscape", duration=float(frames/fps))
                                            if ambient: audio_paths.append(ambient)

                                        if audio_paths:
                                            print("[AnimatePhoto] Mezclando y uniendo audio al video...")
                                            # Usar el primero por ahora o mezclar con ffmpeg si hay varios
                                            final_audio = audio_paths[0]
                                            
                                            temp_video = output_path.replace(".mp4", "_silent.mp4")
                                            if os.path.exists(temp_video): os.remove(temp_video)
                                            os.rename(output_path, temp_video)
                                            
                                            # Comando FFmpeg con mezcla si tenemos voz + fondo
                                            if len(audio_paths) > 1:
                                                cmd = [
                                                    'ffmpeg', '-y', '-i', temp_video, '-i', audio_paths[0], '-i', audio_paths[1],
                                                    '-filter_complex', '[1:a][2:a]amix=inputs=2:duration=first[a]',
                                                    '-map', '0:v:0', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', output_path
                                                ]
                                            else:
                                                cmd = [
                                                    'ffmpeg', '-y', '-i', temp_video, '-i', final_audio,
                                                    '-c:v', 'copy', '-c:a', 'aac', '-map', '0:v:0', '-map', '1:a:0', output_path
                                                ]
                                            
                                            result = subprocess.run(cmd, capture_output=True, text=True)
                                            if result.returncode == 0:
                                                print(f"[AnimatePhoto] ✅ Video con audio inteligente generado ({len(audio_paths)} fuentes).")
                                                if os.path.exists(temp_video): os.remove(temp_video)
                                            else:
                                                print("[AnimatePhoto] ⚠️ Falló la unión de audio. Devolviendo mudo.")
                                                if not os.path.exists(output_path): os.rename(temp_video, output_path)
                                        else:
                                            print("[AnimatePhoto] ⚠️ La IA no pudo decidir un audio válido.")
                                    except Exception as e:
                                        print(f"[AnimatePhoto] ⚠️ Error en Director de Sonido: {e}")
                                    # -----------------------------------------------------------

                                    elapsed = time.time() - t0
                                    print(f"[AnimatePhoto] Hecho en {elapsed:.0f}s")
                                    return True
            time.sleep(1)
