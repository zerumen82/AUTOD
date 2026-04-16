#!/usr/bin/env python3
"""Descarga archivos individuales del modelo NF4 con reintentos"""

from huggingface_hub import hf_hub_download
import time

MODEL_DIR = r"D:\PROJECTS\models\flux-fill-nf4"
HF_TOKEN = ""  # Set your HuggingFace token here

FILES = [
    ("text_encoder_2", "config.json"),
    ("text_encoder_2", "model-00001-of-00002.safetensors"),
    ("text_encoder_2", "model-00002-of-00002.safetensors"),
    ("text_encoder_2", "model.safetensors.index.json"),
    ("transformer", "config.json"),
    ("transformer", "diffusion_pytorch_model.safetensors"),
]

print("=" * 60)
print("DESCARGA NF4 - Optimizado para 8GB VRAM")
print("=" * 60)
print()

for i, (subdir, filename) in enumerate(FILES, 1):
    filepath = f"{subdir}/{filename}"
    print(f"[{i}/{len(FILES)}] Descargando: {filepath}")
    
    for attempt in range(3):
        try:
            hf_hub_download(
                repo_id="lrzjason/flux-fill-nf4",
                filename=filepath,
                local_dir=MODEL_DIR,
                token=HF_TOKEN,
            )
            print(f"  OK")
            break
        except Exception as e:
            print(f"  Intento {attempt+1} fallido: {e}")
            if attempt < 2:
                time.sleep(5)
    else:
        print(f"  FALLO despues de 3 intentos")

print()
print("=" * 60)
print("PROCESO TERMINADO")
print("=" * 60)
