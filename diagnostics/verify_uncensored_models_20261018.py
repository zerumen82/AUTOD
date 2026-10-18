#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verificacion de la descarga de checkpoints sin censura (2026-10-18).

Comprueba, sin descargar ni tocar produccion:
1. Tamano exacto y SHA256 de cada archivo contra el oid LFS que publica el Hub
   (el descargador propio solo validaba tamano, este probe cierra esa laguna).
2. Que cada alias nuevo del registro resuelve al archivo correcto y que el
   desplegable de Generate lo lista como SDXL completo (o GGUF en su caso).
"""
import hashlib
import os
import sys

import requests

BATCH = [
    ("KamCastle/lustify4", "lustifySDXLNSFWSFW_v40.safetensors",
     "checkpoints", "REAL_lustify4_v40.safetensors", 6938042770),
    ("fancyfeast/bigaspv2-5", "bigaspv25-20250716.safetensors",
     "checkpoints", "REAL_bigasp_v25.safetensors", 6938040682),
    ("Aero666/Biglove.Xl4", "bigLove_xl4.safetensors",
     "checkpoints", "REAL_biglove_xl4.safetensors", 6938042722),
    ("LyliaEngine/waiIllustriousSDXL_v170", "waiIllustriousSDXL_v170.safetensors",
     "checkpoints", "ILUS_wai_illustrious_v17.safetensors", 6938040682),
    ("Laxhar/noobai-XL-Vpred-1.0", "NoobAI-XL-Vpred-v1.0.safetensors",
     "checkpoints", "ILUS_noobai_xl_vpred_1.safetensors", 7105350110),
    ("OnomaAIResearch/Illustrious-XL-v2.0", "Illustrious-XL-v2.0.safetensors",
     "checkpoints", "ILUS_illustrious_xl_v2.safetensors", 6938040674),
    ("silveroxides/Chroma1-HD-GGUF", "Chroma1-HD-Q4_K_M.gguf",
     "diffusion_models", "DIT_chroma1-hd_q4_k_m.gguf", 5566533792),
    ("madebyollin/sdxl-vae-fp16-fix", "sdxl_vae.safetensors",
     "vae", "VAE_sdxl_fp16_fix.safetensors", 334641162),
]

ALIASES = {
    "lustify4": "REAL_lustify4_v40.safetensors",
    "biglove_xl4": "REAL_biglove_xl4.safetensors",
    "wai_illustrious_v17": "ILUS_wai_illustrious_v17.safetensors",
    "noobai_vpred": "ILUS_noobai_xl_vpred_1.safetensors",
    "illustrious_xl_v2": "ILUS_illustrious_xl_v2.safetensors",
}


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


ROOT = get_project_root()
MODELS = os.path.join(ROOT, "ui", "tob", "ComfyUI", "models")


def hub_oid(repo, filename):
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    for entry in requests.get(url, timeout=60).json():
        if entry.get("path") == filename:
            return (entry.get("lfs") or {}).get("oid")
    return None


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    failures = 0
    print("== 1. Tamano y SHA256 contra el Hub ==")
    for repo, filename, folder, final_name, expected in BATCH if "--no-hash" not in sys.argv else []:
        path = os.path.join(MODELS, folder, final_name)
        if not os.path.isfile(path):
            print(f"FALTA   {final_name}")
            failures += 1
            continue
        size = os.path.getsize(path)
        oid = hub_oid(repo, filename)
        local = sha256(path)
        size_ok = size == expected
        hash_ok = bool(oid) and local == oid
        state = "OK   " if (size_ok and hash_ok) else "FALLO"
        if state == "FALLO":
            failures += 1
        print(f"{state} {final_name} {size} B | sha256 {'coincide' if hash_ok else 'NO coincide'}")
        if not size_ok:
            print(f"      tamano {size} != {expected}")
        if not hash_ok:
            print(f"      local {local}\n      hub   {oid}")

    print("== 2. Registro de Generate con los archivos reales ==")
    sys.path.insert(0, ROOT)
    from roop.img_editor.flux_gen_comfy_client import (
        get_checkpoint_architecture,
        get_installed_generation_engines,
        resolve_generation_engine,
    )
    for alias, expected_file in ALIASES.items():
        _a, ckpt = resolve_generation_engine(alias)
        arch = get_checkpoint_architecture(ckpt) if ckpt else None
        ok = ckpt == expected_file and arch == "sdxl_full"
        if not ok:
            failures += 1
        print(f"{'OK   ' if ok else 'FALLO'} {alias} -> {ckpt} (arch={arch})")

    engines = get_installed_generation_engines()
    labels_by_alias = {alias: label for label, alias in engines}
    for alias in list(ALIASES) + ["chroma"]:
        label = labels_by_alias.get(alias)
        ok = bool(label)
        if not ok:
            failures += 1
        print(f"{'OK   ' if ok else 'FALLO'} desplegable: {alias} = {label or 'AUSENTE'}")

    print(f"== RESULTADO: {'OK' if failures == 0 else str(failures) + ' fallos'} ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
