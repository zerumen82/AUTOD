#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_qwen_edit_real.py - prueba REAL de Qwen-Image-Edit en GPU 8GB.
Responde UNA cosa: completa SIN OOM en la RTX 3060 Ti (8 GB) y cuanto tarda?
Usa lo ya en disco: UNet 2511-Q2_K + text encoder LIVIANO Qwen2.5-VL-7B-Q4_K_M
(~4.4GB, evita el OOM del FP8 de 8.7GB) + LoRA Lightning 4-step.
Entorno: QWEN_PROBE_INPUT=ruta.png QWEN_PROBE_STEPS=4 QWEN_PROBE_TE=q4|fp8 QWEN_PROBE_LIGHTNING=1|0
"""
import io, os, sys, time
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

COMFY = "http://127.0.0.1:8188"
M = os.path.join(ROOT, "ui", "tob", "ComfyUI", "models")
UNET = os.path.join(M, "diffusion_models", "qwen-image-edit-2511-Q2_K.gguf")
TE_Q4 = os.path.join(M, "clip", "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
TE_FP8 = os.path.join(M, "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors")
VAE = os.path.join(M, "vae", "qwen_image_vae.safetensors")
LORA = os.path.join(M, "loras", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
PROMPT = "Change the wall color to light grey. Keep the person, furniture and framing exactly the same."
NEG = "low quality, blurry, bad anatomy, deformed, ugly, extra objects"


def _pick_input():
    env = os.environ.get("QWEN_PROBE_INPUT")
    if env and os.path.exists(env):
        return env
    d = os.path.join(ROOT, "ui", "tob", "ComfyUI", "input")
    fs = [f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    fast = [f for f in fs if f.startswith("fast_")] or fs
    if not fast:
        raise SystemExit("sin fotos en input/")
    fast.sort(key=lambda f: os.path.getmtime(os.path.join(d, f)), reverse=True)
    return os.path.join(d, fast[0])


def _wait_comfy(t=180):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            if requests.get(COMFY + "/system_stats", timeout=2).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    use_q4 = os.environ.get("QWEN_PROBE_TE", "q4").lower() != "fp8"
    lightning = os.environ.get("QWEN_PROBE_LIGHTNING", "1") == "1"
    steps = int(os.environ.get("QWEN_PROBE_STEPS", "4"))
    te = TE_Q4 if use_q4 else TE_FP8
    te_lbl = "Q4_K_M ~4.4GB" if use_q4 else "FP8 ~8.7GB"

    need = [UNET, te, VAE] + ([LORA] if lightning else [])
    miss = [p for p in need if not os.path.exists(p)]
    if miss:
        raise SystemExit("faltan: " + "; ".join(miss))

    in_path = _pick_input()
    im = Image.open(in_path).convert("RGB")
    w, h = im.size
    w, h = (w // 8) * 8, (h // 8) * 8
    im = im.resize((w, h), Image.LANCZOS)

    print("=" * 60)
    print("[PROBE] Qwen-Image-Edit REAL: cabe en 8GB sin OOM?")
    print("[PROBE] foto=%s (%dx%d) TE=%s lightning=%s steps=%d"
          % (os.path.basename(in_path), w, h, te_lbl, lightning, steps))
    print("=" * 60)

    from ui.tabs import comfy_launcher
    started = None
    if not comfy_launcher.is_comfyui_running():
        ok, msg, _ = comfy_launcher.start(directly_run=True)
        if not ok:
            print("[PROBE][FAIL] no arranco ComfyUI: %s" % msg)
            return 2
        started = comfy_launcher.comfy_process
    if not _wait_comfy():
        print("[PROBE][FAIL] ComfyUI no responde en 8188")
        return 2

    try:
        buf = io.BytesIO(); im.save(buf, "PNG"); buf.seek(0)
        iname = "qwenprobe_%d.png" % int(time.time())
        requests.post(COMFY + "/upload/image",
                      files={"image": (iname, buf, "image/png")}, timeout=60).raise_for_status()

        clip_node = "CLIPLoaderGGUF" if use_q4 else "CLIPLoader"
        wf = {
            "1": {"class_type": "LoadImage", "inputs": {"image": iname, "upload": "image"}},
            "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": os.path.basename(UNET)}},
            "3": {"class_type": clip_node, "inputs": {"clip_name": os.path.basename(te), "type": "qwen_image"}},
            "4": {"class_type": "QwenVAELoader", "inputs": {"vae_name": os.path.basename(VAE)}},
            "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["4", 0]}},
            "6": {"class_type": "TextEncodeQwenImageEdit", "inputs": {"clip": ["3", 0], "prompt": PROMPT, "vae": ["4", 0], "image": ["1", 0]}},
            "7": {"class_type": "TextEncodeQwenImageEdit", "inputs": {"clip": ["3", 0], "prompt": NEG, "vae": ["4", 0], "image": None}},
        }
        mlink, clink = ["2", 0], ["3", 0]
        if lightning:
            wf["20"] = {"class_type": "LoraLoader", "inputs": {"model": ["2", 0], "clip": ["3", 0], "lora_name": os.path.basename(LORA), "strength_model": 1.0, "strength_clip": 1.0}}
            mlink, clink = ["20", 0], ["20", 1]
        wf["8"] = {"class_type": "KSampler", "inputs": {"model": mlink, "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0], "seed": 12345, "steps": steps, "cfg": 1.0 if lightning else 2.5, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}}
        wf["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}}
        wf["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "qwen_probe_out"}}

        t0 = time.time()
        r = requests.post(COMFY + "/prompt", json={"prompt": wf}, timeout=60)
        if r.status_code != 200:
            print("[PROBE][FAIL] queue: %s" % r.text[:400]); return 3
        pid = r.json()["prompt_id"]
        print("[PROBE] encolado pid=%s" % pid)

        deadline = t0 + 1800
        last = 0
        while time.time() < deadline:
            try:
                if requests.get(COMFY + "/system_stats", timeout=15).status_code != 200:
                    print("[PROBE][FAIL] server no responde (posible OOM)"); return 4
            except requests.exceptions.ConnectionError:
                print("[PROBE][FAIL] ComfyUI se cerro -> OOM en 8GB"); return 4
            except requests.exceptions.ReadTimeout:
                pass
            try:
                h = requests.get(COMFY + "/history/" + pid, timeout=30).json()
            except requests.exceptions.ReadTimeout:
                time.sleep(1); continue
            if pid in h:
                st = h[pid].get("status", {})
                if st.get("status_str") == "error":
                    print("[PROBE][FAIL] workflow error: %s" % st.get("messages", "")); return 5
                imgs = []
                for n in h[pid].get("outputs", {}).values():
                    if isinstance(n, dict) and "images" in n:
                        imgs += n["images"]
                if imgs:
                    dt = time.time() - t0
                    print("=" * 60)
                    print("[PROBE][PASS] SIN OOM en %.1f s (%.1f min), %d pasos. salidas=%s"
                          % (dt, dt / 60.0, steps, [i.get("filename") for i in imgs]))
                    return 0
            el = int(time.time() - t0)
            if el - last >= 60:
                print("[PROBE] ...%d min" % (el // 60)); last = el
            time.sleep(2)
        print("[PROBE][FAIL] timeout 30 min"); return 6
    finally:
        if started is not None and started.poll() is None:
            print("[PROBE] apagando ComfyUI...")
            started.terminate()
            try:
                started.wait(timeout=10)
            except Exception:
                started.kill()


if __name__ == "__main__":
    sys.exit(main())

