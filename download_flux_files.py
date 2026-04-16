#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Descarga archivo por archivo el modelo FLUX.1-Fill-dev
"""

import os
from huggingface_hub import hf_hub_download

MODEL_DIR = r"D:\PROJECTS\models\FLUX.1-fill-dev-NF4"
HF_TOKEN = ""  # Set your HuggingFace token here

FILES = [
    "ae.safetensors",
    "flux1-fill-dev.safetensors",
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder_2/config.json",
    "text_encoder_2/model-00001-of-00002.safetensors",
    "text_encoder_2/model-00002-of-00002.safetensors",
    "text_encoder_2/model.safetensors.index.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/spiece.model",
    "tokenizer_2/tokenizer.json",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
]

print("=" * 60)
print("DESCARGA DE FLUX.1-Fill-dev - Archivo por Archivo")
print("=" * 60)
print()

for i, filename in enumerate(FILES, 1):
    print(f"[{i}/{len(FILES)}] Descargando: {filename}")
    try:
        hf_hub_download(
            repo_id="black-forest-labs/FLUX.1-Fill-dev",
            filename=filename,
            local_dir=MODEL_DIR,
            token=HF_TOKEN,
        )
        print(f"  ✅ OK")
    except Exception as e:
        print(f"  ❌ Error: {e}")

print()
print("=" * 60)
print("DESCARGA COMPLETADA!")
print("=" * 60)
