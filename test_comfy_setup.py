#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test ComfyUI Setup - Verifica que la configuracion de ComfyUI es correcta
"""

import os
import sys

# Puerto dinamico
COMFYUI_PORT = os.environ.get('COMFYUI_PORT', '8189')
COMFY_URL = f"http://127.0.0.1:{COMFYUI_PORT}"

print("=" * 60)
print("TEST DE CONFIGURACION COMFYUI")
print("=" * 60)

# Test 1: Puerto configurado
print(f"\n[1] Puerto configurado: {COMFYUI_PORT}")
if COMFYUI_PORT == "8189":
    print("    [OK] Correcto")
else:
    print(f"    [WARN] Advertencia: Se usa puerto {COMFYUI_PORT}")

# Test 2: ComfyUI disponible
print(f"\n[2] Verificando ComfyUI en {COMFY_URL}...")
try:
    import requests
    response = requests.get(f"{COMFY_URL}/system_stats", timeout=5)
    if response.status_code == 200:
        print("    [OK] ComfyUI esta corriendo")
        
        # Test 3: Checkpoints
        print("\n[3] Verificando checkpoints...")
        response = requests.get(f"{COMFY_URL}/object_info/CheckpointLoaderSimple", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "CheckpointLoaderSimple" in data:
                node = data["CheckpointLoaderSimple"]
                if "input" in node and "required" in node["input"]:
                    ckpt_list = node["input"]["required"].get("ckpt_name")
                    if ckpt_list and len(ckpt_list) > 0 and len(ckpt_list[0]) > 0:
                        print(f"    [OK] {len(ckpt_list[0])} checkpoints disponibles:")
                        for ckpt in ckpt_list[0]:
                            print(f"       - {ckpt}")
                    else:
                        print("    [ERROR] No hay checkpoints cargados")
                else:
                    print("    [ERROR] Formato de object_info inesperado")
            else:
                print("    [ERROR] No se encontro CheckpointLoaderSimple")
        else:
            print(f"    [ERROR] Error obteniendo object_info: {response.status_code}")
    else:
        print(f"    [ERROR] ComfyUI no respondio correctamente: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("    [ERROR] No se pudo conectar a ComfyUI")
    print("    Solucion: Inicia ComfyUI con:")
    print(f"       cd ui/tob/ComfyUI && python main.py --port {COMFYUI_PORT}")
except Exception as e:
    print(f"    [ERROR] Error: {e}")

# Test 4: Modulos Python
print("\n[4] Verificando modulos Python...")
try:
    from roop.comfy_client import ComfyClient
    print("    [OK] roop.comfy_client importado")
except ImportError as e:
    print(f"    [ERROR] Error importando roop.comfy_client: {e}")

try:
    from roop.img_editor.comfy_workflows import build_img2img_workflow
    print("    [OK] roop.img_editor.comfy_workflows importado")
except ImportError as e:
    print(f"    [ERROR] Error importando workflows: {e}")

try:
    from roop.img_editor.img_editor_manager import get_img_editor_manager
    print("    [OK] roop.img_editor.img_editor_manager importado")
except ImportError as e:
    print(f"    [ERROR] Error importando manager: {e}")

# Test 5: Rutas
print("\n[5] Verificando rutas...")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMFYUI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "ui", "tob", "ComfyUI"))

main_py = os.path.join(COMFYUI_DIR, "main.py")
if os.path.exists(main_py):
    print(f"    [OK] main.py encontrado: {main_py}")
else:
    print(f"    [ERROR] main.py no encontrado: {main_py}")

print("\n" + "=" * 60)
print("FIN DEL TEST")
print("=" * 60)
