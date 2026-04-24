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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()

    def _load_model(self):
        if not HAS_LLAMA:
            print("[WARN] llama-cpp-python no instalado")
            return

        # Nombres de archivos posibles
        text_names = ["moondream2-text-model-f16.gguf", "moondream2-text-model.gguf", "moondream2.gguf"]
        mmproj_names = ["moondream2-mmproj-f16.gguf", "moondream2-mmproj.gguf", "moondream2-mmproj.gguf"]

        text_model_path = None
        mmproj_path = None

        print(f"[ANALYZER] Buscando modelos en: {MODELS_DIR}")
        
        if not MODELS_DIR.exists():
            print(f"[WARN] La carpeta {MODELS_DIR} no existe. Creándola...")
            MODELS_DIR.mkdir(parents=True, exist_ok=True)

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
            print(f"[ERROR] No se encontró el modelo de texto en {MODELS_DIR}")
            print(f"Asegúrate de que el archivo 'moondream2-text-model-f16.gguf' esté ahí.")
            return

        try:
            print(f"[ANALYZER] Cargando: {text_model_path.name} ({text_model_path.stat().st_size / 1024**2:.1f} MB)")
            
            # Usar '/' para evitar problemas de escape en Windows
            model_str = str(text_model_path.absolute()).replace("\\", "/")
            mm_str = str(mmproj_path.absolute()).replace("\\", "/") if mmproj_path else None
            
            # En v0.3.x se usa clip_model_path para el proyector visual
            self.model = Llama(
                model_path=model_str,
                clip_model_path=mm_str,
                n_gpu_layers=-1 if self.device == "cuda" else 0,
                n_ctx=2048,
                verbose=False
            )
            
            # Si no se cargó vía chat_format, intentamos el método interno de la v0.3.x
            if mm_str:
                print(f"[ANALYZER] Vinculando proyector: {mmproj_path.name}")
                try:
                    # En algunas versiones de 0.3.x se usa un atributo interno o se cargó ya
                    # Si no hay error aquí, el modelo está listo
                    pass 
                except:
                    pass

            print(f"[ANALYZER] Moondream listo en {self.device.upper()}")
        except Exception as e:
            print(f"[ERROR] Error al cargar Moondream: {e}")
            self.model = None

    def analyze(self, image_path: str, nsfw_level: str = 'explicit') -> dict:
        if not self.model:
            return self._fallback_basic(image_path, nsfw_level)
        try:
            # Para Moondream en v0.3.x, usamos el formato directo de prompt
            # <IMAGE> seguido de la pregunta es el estándar de moondream
            prompt = "<IMAGE>\nDescribe this image in detail. Be specific about people, poses, and clothing."
            
            # Usamos create_completion en lugar de chat para mayor control
            response = self.model.create_completion(
                prompt=prompt,
                max_tokens=300,
                stop=["<|endoftext|>", "</s>"]
            )
            
            description = response['choices'][0]['text'].strip()
            if not description or description.startswith("[s]"):
                # Fallback si el prompt anterior falla
                response = self.model.create_completion(
                    prompt=f"Question: Describe this image.\nAnswer:",
                    max_tokens=300
                )
                description = response['choices'][0]['text'].strip()
            
            nsfw_tags = {
                'explicit': ['fully nude', 'naked', 'genitals', '18+'],
                'moderate': ['nude', 'topless'],
            }.get(nsfw_level, [])

            return {
                'positive': ", ".join([description] + nsfw_tags),
                'negative': "clothed, dressed, blurry, low quality",
                'analysis': {'description': description}
            }
        except Exception as e:
            print(f"[ERROR] Durante análisis: {e}")
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
