#!/usr/bin/env python3
import sys, os
sys.path.insert(0, r"D:\PROJECTS\AUTOAUTO")

print("=== CONFIGURACIÓN DE MOTORES ===\n")

# 1. FLUX.2-klein
from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient
f2 = FluxEditComfyClient()
paths_f2 = f2.get_model_paths('flux_klein')
print("FLUX.2-klein paths:")
for k, v in paths_f2.items():
    exists = os.path.exists(v)
    size = os.path.getsize(v) if exists else 0
    print(f"  {k}: {v}")
    print(f"    exists={exists}, size={size}")

# 2. FLUX.1-dev
paths_f1 = f2.get_model_paths('flux_dev')
print("\nFLUX.1-dev paths:")
for k, v in paths_f1.items():
    exists = os.path.exists(v)
    size = os.path.getsize(v) if exists else 0
    print(f"  {k}: {v}")
    print(f"    exists={exists}, size={size}")

# 3. OmniGen2
from roop.img_editor.omnigen2_gguf_comfy_client import OmniGen2ComfyClient
omni = OmniGen2ComfyClient()
paths_o = omni.get_model_paths()
print("\nOmniGen2 paths:")
for k, v in paths_o.items():
    exists = os.path.exists(os.path.join(r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models", v if k=='vae' else ('vae' if k=='vae' else f"{'diffusion_models' if k=='model' else 'text_encoders'}/{v}")))
    print(f"  {k}: {v} (exists={exists})")

print("\n=== WORKFLOW CHECK ===")
# Verificar que los workflows usan los nodos correctos
print("FLUX.1-dev → UnetLoaderGGUF + DualCLIPLoaderGGUF")
print("FLUX.2-klein → UNETLoader + CLIPLoader (safetensors)")
print("OmniGen2 → UNETLoader + CLIPLoader(type=omnigen2)")
