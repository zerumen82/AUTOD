#!/usr/bin/env python3
"""Descarga Qwen-Rapid-NSFW-v23 (All-In-One, NSFW explícito) desde HuggingFace"""
import huggingface_hub, os, sys

REPO = "Novice25/Qwen-Image-Edit-Rapid-AIO-GGUF"
VERSION = "v23"
FILENAME = f"{VERSION}/Qwen-Rapid-NSFW-v23_Q2_K.gguf"
LOCAL_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\diffusion_models"

print(f"=== Qwen Rapid NSFW v23 Q2_K ===")
print(f"Repo: {REPO}")
print(f"File: {FILENAME}")
print(f"Size: ~7.44 GB")

# Descarga automática (sin confirmación)

print("\nDescargando...")
filepath = huggingface_hub.hf_hub_download(
    repo_id=REPO,
    filename=FILENAME,
    local_dir=LOCAL_DIR,
    resume_download=True
)
size_gb = os.path.getsize(filepath) / 1e9
print(f"\nDescargado: {filepath}")
print(f"Tamaño: {size_gb:.2f} GB")
print(f"\nModelo listo en: ui/tob/ComfyUI/models/diffusion_models/")
