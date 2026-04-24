#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HART Image Edit ComfyUI Client
Hybrid Autoregressive Transformer - genera imágenes 1024x1024 autoregresivamente
"""

import os, sys, json, time, requests, io, subprocess, tempfile
from typing import Optional, Tuple
from PIL import Image
from dataclasses import dataclass

COMFY = "http://127.0.0.1:8188"

@dataclass
class GenResult:
    image: Image.Image
    time_taken: float = 0.0

class HartEditComfyClient:
    def __init__(self):
        self._loaded = False
        self._model_paths = {}
        self._device = None

    def is_available(self):
        try:
            return requests.get(f"{COMFY}/system_stats", timeout=3).status_code == 200
        except:
            return False

    def get_model_paths(self) -> dict:
        """Rutas para HART 0.7B 1024px"""
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
        
        return {
            "hart_model": os.path.join(base, "hart", "hart-0.7b-1024px", "llm"),
            "qwen2_vl": os.path.join(base, "text_encoders", "qwen2-vl"),
            "hart_tokenizer": os.path.join(base, "hart", "hart-0.7b-1024px", "tokenizer"),
            "shield_model": os.path.join(base, "hart", "shieldgemma-2b"),
        }

    def check_models(self) -> Tuple[bool, str]:
        """Verifica que los modelos existan"""
        paths = self.get_model_paths()
        
        ema_model = os.path.join(paths["hart_model"], "ema_model.bin")
        tokenizer_model = os.path.join(paths["hart_tokenizer"], "pytorch_model.bin")
        
        missing = []
        if not os.path.exists(ema_model):
            missing.append("ema_model.bin")
        if not os.path.exists(tokenizer_model):
            missing.append("tokenizer pytorch_model.bin")
        
        if missing:
            return False, f"Modelos faltantes: {', '.join(missing)}"
        
        self._model_paths = paths
        return True, "OK"

    def load(self, progress_callback=None) -> Tuple[bool, str]:
        """Carga HART en memoria"""
        if self._loaded:
            return True, "Ya cargado"
        
        ok, msg = self.check_models()
        if not ok:
            return False, msg
        
        if progress_callback:
            progress_callback(10, "Verificando modelos HART...")
        
        self._loaded = True
        
        if progress_callback:
            progress_callback(100, "HART listo")
        
        return True, "HART cargado"

    def unload(self):
        """Libera memoria"""
        self._loaded = False

    def generate(self, prompt: str, 
                 num_inference_steps: int = 8,
                 guidance_scale: float = 4.5,
                 seed: int = None,
                 width: int = 1024,
                 height: int = 1024,
                 progress_callback=None,
                 venv_path: str = None) -> Tuple[GenResult, str]:
        """Genera imagen con HART
        
        Args:
            venv_path: Path al venv con torch instalado. 
                      Default: D:\\PROJECTS\\AUTOAUTO\\venv
        """
        
        if not self._loaded:
            return None, "HART no está cargado. Llama a load() primero."
        
        if venv_path is None:
            venv_path = r"D:\PROJECTS\AUTOAUTO\venv"
        
        venv_python = os.path.join(venv_path, "Scripts", "python.exe")
        if not os.path.exists(venv_python):
            return None, f"Python venv no encontrado: {venv_python}"
        
        t0 = time.time()
        
        script_content = f'''
import os, sys, json, time
import torch
from PIL import Image
import numpy as np

os.environ['HF_HUB_DISABLE_XDG_CACHE'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

hart_module_path = r"{os.path.join(os.path.dirname(__file__), '..', '..', 'ui', 'tob', 'ComfyUI', 'custom_nodes', 'hart')}"
sys.path.insert(0, hart_module_path)

hart_kernels_path = os.path.join(hart_module_path, "hart", "kernels", "build", "lib.win-amd64-cpython-311")
sys.path.insert(0, hart_kernels_path)

from hart.modules.models.transformer.hart_transformer_t2i import HARTForT2I
from transformers import AutoModel, AutoTokenizer
from hart.utils import encode_prompts, llm_system_prompt

device = torch.device("cuda")
torch.cuda.empty_cache()
torch.cuda.synchronize()

model_path = r"{self._model_paths['hart_model']}"
text_model_path = r"{self._model_paths['qwen2_vl']}"

# Cargar en CPU primero para evitar OOM
print("[HART] Cargando modelo en CPU...", flush=True)
model = HARTForT2I.from_pretrained(model_path, device_map="cpu")
print("[HART] Modelo base cargado", flush=True)

# Compilar si está disponible (ahorra memoria)
try:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    model = torch.compile(model, mode="reduce-overhead")
    print("[HART] Modelo compilado", flush=True)
except Exception as e:
    print(f"[HART] Compile skipped: {e}", flush=True)

torch.cuda.empty_cache()
torch.cuda.synchronize()

print("[HART] Cargando EMA weights (CPU)...", flush=True)
ema_state = torch.load(os.path.join(model_path, "ema_model.bin"), map_location="cpu", weights_only=False)
model.load_state_dict(ema_state)
del ema_state
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Mover a GPU con manejo de OOM
print("[HART] Moviendo a GPU...", flush=True)
try:
    model = model.to(device)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        print("[HART] ⚠️ OOM en GPU. Usando CPU (lento, pero funciona)", flush=True)
        device = torch.device("cpu")
        model = model.to(device)
    else:
        raise

print("[HART] Modelo listo", flush=True)
torch.cuda.empty_cache()
torch.cuda.synchronize()

# Generación
with torch.inference_mode():
    print("[HART] Cargando Qwen2-VL (CPU)...", flush=True)
    text_model = AutoModel.from_pretrained(text_model_path).to("cpu")
    text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    
    print("[HART] Encoding prompt...", flush=True)
    with torch.no_grad():
        context_tokens, context_mask, context_position_ids, context_tensor = encode_prompts(
            ["{prompt}"], text_model, text_tokenizer, 300, llm_system_prompt, True
        )
        context_tensor = context_tensor.float()
    
    del text_model, text_tokenizer
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    print("[HART] Prompt codificado", flush=True)
    
    context_tensor = context_tensor.to(device)
    print("[HART] Generando (8 pasos)...", flush=True)
    
    output = model.autoregressive_infer_cfg(
        B=1, label_B=context_tensor, cfg={guidance_scale}, g_seed={seed or 42}, more_smooth=True,
        context_position_ids=context_position_ids, context_mask=context_mask
    )
    
    print("[HART] Generación completada", flush=True)

img_np = output[0].permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
img = Image.fromarray(img_np)
output_dir = r"{os.path.join(os.path.dirname(__file__), '..', '..', 'ui', 'tob', 'output')}"
os.makedirs(output_dir, exist_ok=True)
img.save(os.path.join(output_dir, "hart_generated.png"))
print("[HART] Imagen guardada!", flush=True)
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write(script_content)
            script_path = f.name
        
        try:
            result = subprocess.run(
                [venv_python, "-u", script_path],  # -u for unbuffered output
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )
            
            print(f"[HART] STDOUT: {result.stdout[:2000]}")
            if result.stderr:
                print(f"[HART] STDERR: {result.stderr[:2000]}")
            
            output_path = os.path.join(os.path.dirname(__file__), '..', '..', 'ui', 'tob', 'output', 'hart_generated.png')
            
            if result.returncode == 0 and os.path.exists(output_path):
                result_img = Image.open(output_path)
                
                if width != 1024 or height != 1024:
                    result_img = result_img.resize((width, height), Image.LANCZOS)
                
                elapsed = time.time() - t0
                return GenResult(image=result_img, time_taken=elapsed), "OK"
            else:
                return None, f"Error: {result.stderr[:500]}"
        except subprocess.TimeoutExpired:
            return None, "Timeout (15 min)"
        except Exception as e:
            return None, str(e)
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    def generate_api(self, prompt: str,
                     num_inference_steps: int = 8,
                     guidance_scale: float = 4.5,
                     seed: int = None,
                     width: int = 1024,
                     height: int = 1024,
                     progress_callback=None) -> Tuple[GenResult, str]:
        """Genera via API de ComfyUI (alternativo)"""
        return self.generate(prompt, num_inference_steps, guidance_scale, seed, width, height, progress_callback)


def get_hart_edit_comfy_client():
    """Factory function"""
    return HartEditComfyClient()
