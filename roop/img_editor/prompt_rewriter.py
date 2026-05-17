#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Rewriter - Inteligencia Semántica Pura (Sin Hardcoding)
"""

import os, sys, json, re
from typing import Optional, Dict, List, Tuple

# Añadir CUDA al PATH si es Windows
if sys.platform == "win32":
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    if cuda_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")

MOONDREAM_TEXT_PATH = r"D:\PROJECTS\AUTOAUTO\models\moondream\moondream2-text-model-f16.gguf"
QWEN_LLM_PATH = r"D:\PROJECTS\AUTOAUTO\models\llm\qwen2.5-0.5b-instruct-q4_k_m.gguf"

def ensure_llm_model():
    """Baixa modelo de lenguaje si no existe"""
    if os.path.exists(QWEN_LLM_PATH):
        return True
    
    llm_dir = os.path.dirname(QWEN_LLM_PATH)
    if not os.path.exists(llm_dir):
        os.makedirs(llm_dir, exist_ok=True)
    
    print("[PromptRewriter] Intentando baixar qwen2.5-0.5b-instruct...")
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            local_dir=llm_dir
        )
        print(f"[PromptRewriter] Modelo baixado: {model_path}")
        return True
    except Exception as e:
        print(f"[PromptRewriter] No se pudo baixar modelo: {e}")
        return False

class PromptRewriter:
    """Analiza la instrucción y extrae todos los parámetros de edición mediante razonamiento LLM"""

    def __init__(self):
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
            
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            
            llm_paths = [
                QWEN_LLM_PATH,
                os.path.join(root, "models", "llm", "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
                os.path.join(root, "models", "llm", "llama-3.2-1b-instruct-q4_k_m.gguf"),
            ]
            
            model_path = None
            for p in llm_paths:
                if os.path.exists(p):
                    model_path = p
                    print(f"[PromptRewriter] Usando modelo de lenguaje dedicado: {os.path.basename(p)}")
                    break
            
            if not model_path:
                moondream_paths = [
                    MOONDREAM_TEXT_PATH,
                    os.path.join(root, "models", "moondream", "moondream2-text-model-f16.gguf"),
                ]
                for p in moondream_paths:
                    if os.path.exists(p):
                        model_path = p
                        print(f"[PromptRewriter] ADVERTENCIA: Usando moondream (menos preciso)")
                        break
                
            if not model_path:
                print("[PromptRewriter] ERROR: No se encontró ningún modelo LLM.")
                return

            # Detectar si podemos usar GPU (aceleración GGUF)
            n_gpu_layers = 0
            try:
                import torch
                if torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    # En una 3060 Ti 8GB, Qwen 0.5B cabe de sobra (ocupa <500MB en VRAM)
                    n_gpu_layers = 32 # Forzar capas a GPU para velocidad
            except: pass

            print(f"[PromptRewriter] Cargando rewriter: {os.path.basename(model_path)} (GPU layers={n_gpu_layers})")
            self._llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=2048,
                verbose=False
            )
            print("[PromptRewriter] Rewriter listo")
        except ImportError:
            print("[PromptRewriter] llama-cpp-python no disponible, usando modo heurístico")
            self._llm = "heuristic"
        except Exception as e:
            print(f"[PromptRewriter] LLM no disponible: {e}, usando modo heurístico")
            self._llm = "heuristic"

    def unload(self):
        """Libera la memoria del LLM inmediatamente"""
        if self._llm is not None and self._llm != "heuristic":
            print("[PromptRewriter] Liberando memoria del rewriter...")
            try:
                # La destrucción del objeto Llama libera la memoria C++ asociada
                del self._llm
                self._llm = None
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def rewrite(self, prompt: str, image_context: str = "") -> Dict:
        """Devuelve un diccionario con el análisis completo del LLM"""
        if self._llm == "heuristic":
            return self._rewrite_heuristic(prompt)
        
        if self._llm is not None:
            try:
                return self._rewrite_with_llm(prompt, self._llm, image_context=image_context)
            except Exception as e:
                print(f"[PromptRewriter] Error con LLM: {e}")
        
        return {
            "prompt": prompt,
            "magnitude": 0.5,
            "mask_target": "subject",
            "reasoning": "Fallback mode"
        }
    
    def _rewrite_heuristic(self, prompt: str) -> Dict:
        """Fallback neutro cuando no hay LLM disponible."""
        print("[PromptRewriter] Heuristic disabled: usando prompt original sin inferencia semántica")
        return {
            "prompt": prompt,
            "magnitude": 0.5,
            "mask_target": "subject",
            "reasoning": "LLM unavailable"
        }

    def _rewrite_with_llm(self, prompt: str, llm, image_context: str = "") -> Dict:
        ctx = image_context[:200] if len(image_context) > 200 else image_context
        full_prompt = (
            f"Analyze this edit request.\n"
            f"Context: {ctx}\n"
            f"Request: {prompt}\n"
            f"Return JSON with magnitude (0-1, >0 if any change) and mask_target (noun to edit)."
        )

        response = llm.create_completion(
            full_prompt, max_tokens=80, temperature=0.1,
            echo=False, stop=["\n\n"]
        )
        
        try:
            text = response['choices'][0]['text'].strip()
            print(f"[PromptRewriter] Raw LLM Response: {text}")
            
            data = _extract_first_json(text)
            if data is None:
                raise ValueError("No valid JSON found in response")

            if not isinstance(data, dict):
                if isinstance(data, list) and len(data) >= 1:
                    mag = data[0] if isinstance(data[0], (int, float)) else 0.5
                    mag = max(0.0, min(1.0, float(mag)))
                    print(f"[PromptRewriter] LLM devolvió lista, interpretando magnitude={mag}")
                    return {"prompt": prompt, "magnitude": mag, "mask_target": "subject", "reasoning": "Parsed from list"}
                print(f"[PromptRewriter] Warning: LLM devolvió tipo {type(data).__name__}")
                return {"prompt": prompt, "magnitude": 0.5, "mask_target": "subject", "reasoning": "Invalid response format"}

            magnitude = max(0.0, min(1.0, float(data.get("magnitude", 0.5))))
            mask_target = str(data.get("mask_target", "subject") or "subject").strip().lower()

            # If LLM says no change but user typed a real request, the LLM likely didn't understand it
            if magnitude < 0.1 and len(prompt.strip()) > 2:
                magnitude = 0.5

            result = {
                "prompt": prompt,
                "magnitude": magnitude,
                "mask_target": mask_target,
                "reasoning": data.get("reasoning", "Semantic analysis completed")
            }
            
            print(f"[PromptRewriter] Mag: {result['magnitude']} | Mask: {result['mask_target']}")
            
            return result
        except Exception as e:
            print(f"[PromptRewriter] Error parsing LLM JSON: {e}")
            return {"prompt": prompt, "magnitude": 0.5, "mask_target": "subject"}

def _extract_first_json(s: str) -> Optional[Dict]:
    """Extract the first complete JSON object from a string using brace-depth matching."""
    s = s.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```$', '', s)
    start = s.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                if '"' not in candidate and "'" in candidate:
                    candidate = candidate.replace("'", '"')
                for attempt in [candidate, re.sub(r'[\x00-\x1f\x7f]', '', candidate)]:
                    try:
                        return json.loads(attempt)
                    except json.JSONDecodeError:
                        pass
                return None
    return None

_rewriter = None
def get_prompt_rewriter() -> PromptRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = PromptRewriter()
    return _rewriter
