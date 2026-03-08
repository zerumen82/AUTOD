#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test del Editor Real con ControlNet + IP-Adapter
"""

import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roop.img_editor.comfy_workflows import (
    build_editor_workflow,
    check_controlnet_available,
    check_ipadapter_available,
    get_default_checkpoint
)
import json


def test_workflow():
    """Prueba la generación del workflow"""
    
    print("=" * 60)
    print("TEST: Editor Real con ControlNet + IP-Adapter")
    print("=" * 60)
    
    # Verificar modelos
    print("\n[1] Verificando modelos...")
    has_controlnet = check_controlnet_available()
    has_ipadapter = check_ipadapter_available()
    
    print(f"\n   ControlNet disponible: {has_controlnet}")
    print(f"   IP-Adapter disponible: {has_ipadapter}")
    
    if not has_controlnet and not has_ipadapter:
        print("\n[WARN] No hay modelos disponibles. El workflow usará img2img básico.")
    
    # Verificar checkpoint
    print("\n[2] Verificando checkpoint...")
    try:
        checkpoint = get_default_checkpoint()
        if checkpoint:
            print(f"   Checkpoint: {checkpoint}")
        else:
            print("   [INFO] No se pudo obtener checkpoint (ComfyUI no está corriendo)")
            print("   Usando checkpoint por defecto para test...")
            checkpoint = "v1-5-pruned-emaonly.safetensors"
    except Exception as e:
        print(f"   [INFO] ComfyUI no disponible, usando checkpoint por defecto: {e}")
        checkpoint = "v1-5-pruned-emaonly.safetensors"
    
    # Construir workflow
    print("\n[3] Construyendo workflow...")
    workflow = build_editor_workflow(
        image_filename="test_image.png",
        prompt="a beautiful woman, high quality, detailed",
        negative_prompt="low quality, blurry",
        seed=42,
        steps=30,
        cfg=7.0,
        denoise=0.75,
        checkpoint=checkpoint,
        use_controlnet=has_controlnet,
        use_ipadapter=has_ipadapter
    )
    
    print(f"   Nodos creados: {len(workflow)}")
    
    # Verificar nodos clave
    print("\n[4] Verificando estructura del workflow...")
    
    node_types = {}
    for node_id, node_data in workflow.items():
        class_type = node_data.get("class_type", "Unknown")
        if class_type not in node_types:
            node_types[class_type] = []
        node_types[class_type].append(node_id)
    
    print("   Tipos de nodos:")
    for class_type, node_ids in sorted(node_types.items()):
        print(f"     - {class_type}: {len(node_ids)} nodo(s)")
    
    # Verificar nodos requeridos
    required_nodes = ["LoadImage", "CheckpointLoaderSimple", "CLIPTextEncode", 
                      "VAEEncode", "KSampler", "VAEDecode", "SaveImage"]
    
    missing_nodes = []
    for req in required_nodes:
        if req not in node_types:
            missing_nodes.append(req)
    
    if missing_nodes:
        print(f"\n   [ERROR] Faltan nodos requeridos: {missing_nodes}")
        return False
    
    # Verificar nodos opcionales
    print("\n[5] Verificando nodos de ControlNet/IP-Adapter...")
    
    if "IPAdapterUnifiedLoader" in node_types:
        print("   [OK] IP-Adapter Loader presente")
    else:
        print("   [INFO] IP-Adapter no incluido (modelos no disponibles o deshabilitado)")
    
    if "ControlNetLoader" in node_types:
        print(f"   [OK] ControlNet Loader presente ({len(node_types['ControlNetLoader'])} instancia(s))")
    else:
        print("   [INFO] ControlNet no incluido (modelos no disponibles o deshabilitado)")
    
    if "ControlNetApplyAdvanced" in node_types:
        print(f"   [OK] ControlNet Apply presente ({len(node_types['ControlNetApplyAdvanced'])} instancia(s))")
    
    # Verificar conexiones
    print("\n[6] Verificando conexiones del workflow...")
    
    # Verificar que KSampler tiene las conexiones correctas
    ksampler = None
    for node_id, node_data in workflow.items():
        if node_data.get("class_type") == "KSampler":
            ksampler = node_data
            break
    
    if ksampler:
        inputs = ksampler.get("inputs", {})
        print(f"   KSampler inputs:")
        print(f"     - model: {inputs.get('model')}")
        print(f"     - positive: {inputs.get('positive')}")
        print(f"     - negative: {inputs.get('negative')}")
        print(f"     - latent_image: {inputs.get('latent_image')}")
    
    # Guardar workflow para inspección
    print("\n[7] Guardando workflow para inspección...")
    output_file = "test_workflow_output.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(workflow, f, indent=2, ensure_ascii=False)
    print(f"   Workflow guardado en: {output_file}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETADO EXITOSAMENTE")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_workflow()
    sys.exit(0 if success else 1)
