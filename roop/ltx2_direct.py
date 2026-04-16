"""
LTX-Video 2 - CONFIGURACION REQUERIDA

⚠️  ERROR ANTERIOR: El workflow usaba nodos que no existian ⚠️

ERROR CORREGIDO:
- LTXVAudioVAELoader -> LowVRAMAudioVAELoader (existe)
- CheckpointLoaderSimple -> LowVRAMCheckpointLoader (existe)

⚠️  PROBLEMA ACTUAL: LTX-2 requiere Gemma 3 como text encoder ⚠️

MODELOS REQUERIDOS:

1. LTX-2 Model (YA DESCARGADO)
   - Ubicacion: ui/tob/ComfyUI/models/diffusion_models/ltx-2-19b-dev-fp4.safetensors
   - Tamanio: ~20GB

2. Gemma 3 Text Encoder (NO DESCARGADO - REQUERIDO)
   - Ubicacion: ui/tob/ComfyUI/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/
   - Tamanio: ~12GB
   - URL: https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized

INSTRUCCIONES DE INSTALACION:
1. cd ui/tob/ComfyUI/models/text_encoders
2. git clone https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized
3. Reiniciar ComfyUI

NOTA: T5-XXL (t5xxl_fp8_e4m3fn.safetensors) NO es compatible con LTX-2.
LTX-2 SOLO funciona con Gemma 3.
"""

def check_ltx2_setup():
    """Verifica si la configuracion de LTX-2 esta completa"""
    import os
    
    results = {
        "ltx2_model": False,
        "gemma3_model": False,
        "errors": []
    }
    
    # Check LTX-2 model
    ltx2_path = "ui/tob/ComfyUI/models/diffusion_models/ltx-2-19b-dev-fp4.safetensors"
    if os.path.exists(ltx2_path):
        results["ltx2_model"] = True
    else:
        results["errors"].append(f"LTX-2 no encontrado en: {ltx2_path}")
    
    # Check Gemma 3
    gemma_path = "ui/tob/ComfyUI/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized"
    if os.path.exists(gemma_path):
        results["gemma3_model"] = True
    else:
        results["errors"].append(f"Gemma 3 no encontrado. Ejecutar:")
        results["errors"].append(f"  cd ui/tob/ComfyUI/models/text_encoders")
        results["errors"].append(f"  git clone https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized")
    
    results["ready"] = results["ltx2_model"] and results["gemma3_model"]
    
    return results


if __name__ == "__main__":
    status = check_ltx2_setup()
    print("Estado de LTX-2:")
    print(f"  LTX-2 Model: {'✅' if status['ltx2_model'] else '❌'}")
    print(f"  Gemma 3: {'✅' if status['gemma3_model'] else '❌'}")
    
    if not status["ready"]:
        print("\nErrores:")
        for error in status["errors"]:
            print(f"  - {error}")
