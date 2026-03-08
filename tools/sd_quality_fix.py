"""
Script para mejorar la calidad de Stable Diffusion
===================================================

Este script configura parámetros óptimos para mejor calidad de imagen.

MEJORAS APLICADAS:
1. Modelo cambiado a illustriousRealismBy_v10 (mejor calidad general)
2. Face Restoration activado con CodeFormer (peso 0.8)
3. Código de hash actualizado

PARA MEJORAR AÚN MÁS LA CALIDAD:
1. Descarga un VAE de alta calidad:
   -vae-ft-mse-840000.safetensors
   -更纱VAE
   -orangemixVAE
   
2. Aumenta los steps de generación (30-50 para mejor calidad)

3. Usa Hires Fix para resoluciones mayores de 512x512

4.activa ControlNet para mejor coherencia
"""

import os
import json

# Rutas
config_path = "ui/tob/stable-diffusion-webui/config.json"
webui_path = "ui/tob/stable-diffusion-webui"

def improve_quality():
    """Mejora la configuración de calidad de SD"""
    
    # Cargar config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Mejoras de calidad
    improvements = {
        # Modelo de mejor calidad
        "sd_model_checkpoint": "illustriousRealismBy_v10.safetensors [85b0a59290]",
        
        # Face Restoration activado
        "face_restoration": True,
        "face_restoration_model": "CodeFormer",
        "code_former_weight": 0.8,  # Mayor peso = más restauración
        
        # Configuración de VAE para mejor decodificación
        "sd_vae": "Automatic",
        "auto_vae_precision": True,
        
        # Mejoras de sampling
        "CLIP_stop_at_last_layers": 2,  # Usar más capas del CLIP
        "upcast_attn": True,  # Mejor precisión para atención
        
        # Hires Fix
        "hires_fix_refiner_pass": "second pass",
    }
    
    # Aplicar mejoras
    for key, value in improvements.items():
        if key in config:
            old_value = config[key]
            config[key] = value
            print(f"  ✓ {key}: {old_value} → {value}")
        else:
            config[key] = value
            print(f"  + {key}: {value}")
    
    # Guardar config
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print("\n[OK] Configuración de calidad mejorada!")
    print("\n📌 RECOMENDACIONES ADICIONALES:")
    print("1. Descarga un VAE de alta calidad y ponlo en: models/VAE/")
    print("   - https://huggingface.co/stabilityai/sd-vae/resolve/main/vae-ft-mse-840000.safetensors")
    print("2. Usa 30-50 steps para mejor calidad")
    print("3. Activa Hires Fix para imágenes grandes")
    print("4. Usa CFG Scale 7-9 para mejor seguimiento del prompt")

if __name__ == "__main__":
    improve_quality()
