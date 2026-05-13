#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moondream 2 Analyzer - Versión Robusta (2026)
"""

import sys
import os
import importlib.util
from pathlib import Path
import torch

# Forzar rutas absolutas
ROOT_DIR = Path(__file__).parent.parent.absolute()
MODELS_DIR = ROOT_DIR / "models" / "moondream"
DLL_DIR = ROOT_DIR / "dll"

# Configurar PATH para dependencias
CUDA_PATH = os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")
CUDA_BIN = Path(CUDA_PATH) / "bin"

paths_to_add = [str(DLL_DIR), str(CUDA_BIN)]
os.environ["PATH"] = ";".join(paths_to_add) + ";" + os.environ.get("PATH", "")

# Fallback basic loader
def _load_roop_module(module_name: str, file_name: str):
    module_path = ROOT_DIR / "roop" / file_name
    if not module_path.exists(): return None
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except:
        return None

_analyser_mod = _load_roop_module("analyser", "analyser.py")

try:
    from llama_cpp import Llama
    HAS_LLAMA = True
except ImportError:
    HAS_LLAMA = False

class MoonDreamImageAnalyzer:
    def __init__(self):
        self.model = None
        # Forzar CPU para evitar conflictos con Flux en GPUs de 8GB
        self.device = "cpu" 
        self._load_model()

    def _load_model(self):
        if not HAS_LLAMA:
            print("[WARN] llama-cpp-python no instalado")
            return

        # Nombres de archivos posibles
        text_names = ["moondream2-text-model-f16.gguf", "moondream2-text-model.gguf", "moondream2.gguf"]
        mmproj_names = ["moondream2-mmproj-f16.gguf", "moondream2-mmproj.gguf"]

        text_model_path = None
        mmproj_path = None

        print(f"[ANALYZER] Buscando modelos en: {MODELS_DIR}")
        
        for name in text_names:
            p = MODELS_DIR / name
            if p.exists():
                text_model_path = p
                break
        
        for name in mmproj_names:
            p = MODELS_DIR / name
            if p.exists():
                mmproj_path = p
                break

        if not text_model_path:
            # Reintentar en carpeta models raíz
            for name in text_names:
                p = ROOT_DIR / "models" / name
                if p.exists():
                    text_model_path = p
                    break

        if not text_model_path:
            print(f"[ERROR] No se encontró el modelo de texto en {MODELS_DIR}")
            return

        try:
            # Detectar VRAM disponible para decidir cuántas capas subir a GPU
            n_gpu_layers = 0
            if torch.cuda.is_available():
                try:
                    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    if total_vram > 6:
                        n_gpu_layers = 30 # Subir casi todo el modelo si hay >6GB
                        print(f"[ANALYZER] GPU detectada ({total_vram:.1f}GB), activando aceleración (layers={n_gpu_layers})")
                    else:
                        n_gpu_layers = 8 # Aceleración parcial para 4-6GB
                        print(f"[ANALYZER] GPU limitada ({total_vram:.1f}GB), aceleración parcial (layers={n_gpu_layers})")
                except:
                    n_gpu_layers = 15 # Fallback conservador
            
            print(f"[ANALYZER] Cargando modelo: {text_model_path.name}")
            
            model_str = str(text_model_path.absolute())
            mm_str = str(mmproj_path.absolute()) if mmproj_path else None
            
            self.model = Llama(
                model_path=model_str,
                clip_model_path=mm_str,
                n_gpu_layers=n_gpu_layers,
                n_ctx=1024, # Reducir contexto para velocidad
                verbose=False
            )
            
            if mm_str:
                print(f"[ANALYZER] Proyector visual cargado: {mmproj_path.name}")

            print(f"[ANALYZER] Moondream (CPU) LISTO")
        except Exception as e:
            print(f"[ERROR] Error al cargar Moondream: {e}")
            self.model = None

    def analyze(self, image_path: str, nsfw_level: str = 'explicit') -> dict:
        if not self.model:
            return self._fallback_basic(image_path, nsfw_level)
        
        print(f"[ANALYZER] Analizando imagen: {os.path.basename(image_path)}...")
        try:
            # Prompt optimizado para Moondream
            prompt = "<IMAGE>\nDescribe the main subject and their clothing in one descriptive paragraph."
            
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=200,
                temperature=0.2,
                stop=["<|endoftext|>", "</s>", "\n\n"]
            )
            
            description = response['choices'][0]['text'].strip()
            if not description or len(description) < 5:
                # Fallback
                response = self.model.create_completion(
                    prompt=f"Question: What is in this image?\nAnswer:",
                    max_tokens=150
                )
                description = response['choices'][0]['text'].strip()
            
            print(f"[ANALYZER] Descripción generada: {description[:100]}...")
            
            return {
                'positive': description,
                'negative': "blurry, low quality",
                'analysis': {'description': description}
            }
        except Exception as e:
            print(f"[ERROR] Durante análisis LLM: {e}")
            return self._fallback_basic(image_path, nsfw_level)

    def _fallback_basic(self, image_path: str, nsfw_level: str):
        try:
            from image_analyzer_for_prompt import analyze_image_for_prompt
            return analyze_image_for_prompt(image_path, nsfw_level)
        except:
            return {'positive': 'a person, high quality', 'negative': 'blurry'}

_analyzer = None
def analyze_image_with_moondream(image_path, nsfw_level='explicit'):
    global _analyzer
    if _analyzer is None: _analyzer = MoonDreamImageAnalyzer()
    return _analyzer.analyze(image_path, nsfw_level)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        res = analyze_image_with_moondream(sys.argv[1])
        print("\nRESULTADO:\n", res['positive'])
    else:
        print("Uso: python moondream_analyzer.py <imagen>")
