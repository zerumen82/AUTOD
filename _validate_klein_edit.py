# -*- coding: utf-8 -*-
"""Valida el workflow Klein EDIT / Kontext sin enviar nada a ComfyUI."""
import sys
import os
import io
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, "ui")

from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient
from PIL import Image

import roop.img_editor.flux_edit_comfy_client as mod
orig_post = mod.requests.post
orig_get = getattr(mod.requests, "get", None)
orig_wait = mod.wait_for_comfy_image


def _img_png():
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (128, 128, 128)).save(buf, "PNG")
    return buf.getvalue()


def fake_post(url, **kwargs):
    if url.endswith("/upload/image"):
        return type("R", (), {"status_code": 200, "json": lambda *a, **k: {}})()
    if url.endswith("/prompt"):
        captured["prompt"] = kwargs["json"]["prompt"]
        return type("R", (), {"status_code": 200, "json": lambda *a, **k: {"prompt_id": "test"}})()
    if "/history/" in url:
        return type("R", (), {"status_code": 200, "json": lambda *a, **k: {"test": {"outputs": {"9": {"images": [{"filename": "tmp.png", "type": "output", "subfolder": ""}]}}}}} )()
    if url.endswith("/view"):
        return type("R", (), {"status_code": 200, "content": _img_png(), "headers": {"content-type": "image/png"}})()
    raise AssertionError(url)


def fake_get(url, **kwargs):
    if "/view" in url:
        return type("R", (), {"status_code": 200, "content": _img_png()})()
    raise AssertionError(url)


def fake_wait(comfy_url, pid, progress_callback=None, timeout=600, **kwargs):
    return {"filename": "tmp.png", "type": "output", "subfolder": ""}, "mock"


# ---- Klein 4B EDIT ----
c = FluxEditComfyClient()
c._loaded = True
c._flux_version = "flux-2-klein-base-4b-Q4_K_S.gguf"
c._clip_name = "qwen3-4b-abl-q4_0.gguf"
c._clip_name2 = None
c._clip_type = "flux2"
c._dual_clip = False
c._vae_name = "vae.safetensors"
c._is_gguf_clip = True
c._is_kontext_dev = False
c._is_longcat_turbo = False
c._is_qwen_image_edit = False
c._is_qwen_q5 = False
c._is_klein_4b = True
c._is_klein_9b = False

img = Image.new("RGB", (768, 768), (128, 128, 128))
captured = {}

mod.requests.post = fake_post
mod.requests.get = fake_get
mod.wait_for_comfy_image = fake_wait
try:
    res, err = c.generate(
        image=img,
        prompt="test edit",
        negative_prompt="",
        num_inference_steps=4,
        guidance_scale=1.0,
        denoise=1.0,
        seed=42,
        mask_image=None,
    )
finally:
    mod.requests.post = orig_post
    if orig_get is not None:
        mod.requests.get = orig_get
    mod.wait_for_comfy_image = orig_wait

print("generate result:", res, err)
wf = captured.get("prompt", {})
print("=== KLEIN 4B EDIT NODES ===")
for nid in sorted(wf.keys(), key=lambda x: str(x)):
    node = wf[nid]
    print(f"  {nid}: {node['class_type']}")
    if node["class_type"] == "KSampler":
        ins = node["inputs"]
        print(f"       steps={ins['steps']} cfg={ins['cfg']} denoise={ins['denoise']} sampler={ins['sampler_name']} latent={ins['latent_image']}")

required = ["ReferenceLatent", "EmptyFlux2LatentImage", "ModelSamplingAuraFlow", "KSampler"]
print("=== REQUIRED NODES ===")
for r in required:
    present = any(r in node["class_type"] for node in wf.values())
    print(f"  {r}: {'OK' if present else 'MISSING'}")

ksampler = next((n for n in wf.values() if n["class_type"] == "KSampler"), None)
if ksampler:
    latent = ksampler["inputs"]["latent_image"]
    print("KLEIN latent:", latent)
    assert "5e" in str(latent) or "EmptyFlux2LatentImage" in str(latent), f"Expected empty latent, got {latent}"

# ---- Kontext dev ----
c2 = FluxEditComfyClient()
c2._loaded = True
c2._flux_version = "flux1-kontext-dev-Q2_K.gguf"
c2._clip_name = "clip_l.safetensors"
c2._clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
c2._clip_type = "flux"
c2._dual_clip = True
c2._vae_name = "vae.safetensors"
c2._is_gguf_clip = False
c2._is_kontext_dev = True
c2._is_longcat_turbo = False
c2._is_qwen_image_edit = False
c2._is_qwen_q5 = False

captured2 = {}

mod.requests.post = fake_post
mod.requests.get = fake_get
mod.wait_for_comfy_image = fake_wait
try:
    res2, err2 = c2.generate(
        image=img,
        prompt="test edit",
        negative_prompt="",
        num_inference_steps=20,
        guidance_scale=3.5,
        denoise=0.6,
        seed=42,
        mask_image=None,
        engine="kontext",
    )
finally:
    mod.requests.post = orig_post
    if orig_get is not None:
        mod.requests.get = orig_get
    mod.wait_for_comfy_image = orig_wait

print("=== KONTEXT NODES ===")
wf2 = captured2.get("prompt", {})
for nid in sorted(wf2.keys(), key=lambda x: str(x)):
    node = wf2[nid]
    print(f"  {nid}: {node['class_type']}")
    if node["class_type"] == "KSampler":
        ins = node["inputs"]
        print(f"       steps={ins['steps']} cfg={ins['cfg']} denoise={ins['denoise']} sampler={ins['sampler_name']}")

print("VALIDATION DONE")
