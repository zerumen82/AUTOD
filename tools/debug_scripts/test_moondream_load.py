
import os
from llama_cpp import Llama
from pathlib import Path

model_path = r"models/moondream/moondream2-text-model-f16.gguf"
mmproj_path = r"models/moondream/moondream2-mmproj-f16.gguf"

print(f"Probando carga de: {model_path}")
try:
    # Intentamos carga mínima en CPU
    model = Llama(
        model_path=model_path,
        n_gpu_layers=0, 
        n_ctx=2048,
        verbose=True
    )
    print("¡ÉXITO! El modelo base cargó en CPU.")
    
    if os.path.exists(mmproj_path):
        print(f"Probando carga de mmproj: {mmproj_path}")
        model.load_mmproj(mmproj_path)
        print("¡ÉXITO! MMProj cargado.")
except Exception as e:
    print(f"FALLO CRÍTICO: {e}")
