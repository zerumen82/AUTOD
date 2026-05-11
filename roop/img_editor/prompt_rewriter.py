#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Rewriter - Inteligencia Semántica Pura (Sin Hardcoding)
"""

import os, json, re
from typing import Optional, Dict, List, Tuple

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
            
            # Primero buscar modelo de lenguaje dedicado (mejor para esta tarea)
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
            
            # Si no hay modelo de lenguaje, usar moondream (menos preciso para esta tarea)
            if not model_path:
                moondream_paths = [
                    MOONDREAM_TEXT_PATH,
                    os.path.join(root, "models", "moondream", "moondream2-text-model-f16.gguf"),
                    os.path.join(root, "models", "moondream2-text-model-f16.gguf"),
                ]
                for p in moondream_paths:
                    if os.path.exists(p):
                        model_path = p
                        print(f"[PromptRewriter] ADVERTENCIA: Usando moondream para rewriter (menos preciso). Considere baixar qwen2.5-0.5b para mejores resultados.")
                        break
                
            if not model_path:
                print("[PromptRewriter] ERROR: No se encontró ningún modelo LLM.")
                return

            print(f"[PromptRewriter] Cargando rewriter: {os.path.basename(model_path)}")
            self._llm = Llama(
                model_path=model_path,
                n_gpu_layers=0,
                n_ctx=2048,
                verbose=False
            )
            print("[PromptRewriter] Rewriter listo en CPU")
        except ImportError:
            print("[PromptRewriter] llama-cpp-python no disponible, usando modo heurístico")
            self._llm = "heuristic"  # Flag para modo heurístico
        except Exception as e:
            print(f"[PromptRewriter] LLM no disponible: {e}, usando modo heurístico")
            self._llm = "heuristic"

    def rewrite(self, prompt: str) -> Dict:
        """Devuelve un diccionario con el análisis completo del LLM"""
        if self._llm == "heuristic":
            return self._rewrite_heuristic(prompt)
        
        if self._llm is not None:
            try:
                return self._rewrite_with_llm(prompt, self._llm)
            except Exception as e:
                print(f"[PromptRewriter] Error con LLM: {e}")
        
        # Fallback de emergencia (mínimo hardcoding solo para no romper el sistema)
        return {
            "prompt": prompt,
            "magnitude": 0.5,
            "mask_target": prompt,
            "reasoning": "Fallback mode"
        }
    
    def _rewrite_heuristic(self, prompt: str) -> Dict:
        """Análisis basado en palabras clave cuando no hay LLM disponible"""
        p_lower = prompt.lower()
        
        # Detectar magnitud basada en palabras clave
        radical_words = ["desnudo", "desnuda", "nude", "naked", "ropa completa", 
                        "full clothing", "cuerpo completo", "cambiar todo", 
                        "transformar", "convertir en", "make completely"]
        medium_words = ["cambiar", "change", "modificar", "pose", "expresión",
                       "outfit", "vestimenta", "ropa", "shirt", "pants", "traje"]
        subtle_words = ["sonreír", "smile", "ojos", "eyes", "mirada", "expresión",
                      "ligero", "slight", "color", "tinte", "peinado", "hair"]
        
        if any(w in p_lower for w in radical_words):
            magnitude = 0.85
            reasoning = "Cambio radical detectado (desnudo/ropa completa)"
        elif any(w in p_lower for w in medium_words):
            magnitude = 0.5
            reasoning = "Cambio medio detectado (ropa/pose)"
        else:
            magnitude = 0.25
            reasoning = "Cambio sutil detectado"
        
        # Detectar mask_target
        if any(w in p_lower for w in ["cara", "face", "rostro", "ojos", "eyes", "boca", "mouth", "sonrisa", "smile"]):
            mask_target = "face"
        elif any(w in p_lower for w in ["ropa", "shirt", "camisa", "pantalón", "pants", "traje", "outfit", "vestimenta"]):
            mask_target = "clothes"
        elif any(w in p_lower for w in ["fondo", "background", "escenario", "paisaje"]):
            mask_target = "background"
        elif any(w in p_lower for w in ["pelo", "hair", "cabello", "peinado"]):
            mask_target = "hair"
        elif any(w in p_lower for w in ["cuerpo", "body"]):
            mask_target = "body"
        else:
            mask_target = "subject"
        
        # Traducir prompt al inglés básico
        translations = {
            "desnudo": "naked", "desnuda": "naked", "sonreír": "smiling",
            "ropa": "clothing", "traje": "suit", "camisa": "shirt",
            "pantalón": "pants", "fondo": "background", "pelo": "hair",
            "ojos": "eyes", "cara": "face", "cuerpo": "body"
        }
        eng_prompt = prompt
        for es, en in translations.items():
            eng_prompt = eng_prompt.replace(es, en)
        
        print(f"[PromptRewriter] Heuristic: mag={magnitude}, mask={mask_target}")
        
        return {
            "prompt": eng_prompt,
            "magnitude": magnitude,
            "mask_target": mask_target,
            "reasoning": reasoning
        }

    def _rewrite_with_llm(self, prompt: str, llm) -> Dict:
        # Prompt más simple y directo que cualquier modelo puede seguir
        system = (
            "You analyze image edit requests. Output JSON with:\n"
            "- magnitude: 0.1 to 0.9 (0.1=small, 0.9=radical like nudity)\n"
            "- mask_target: object to change (face, shirt, clothes, background, etc)\n"
            "- prompt: what to generate in English\n"
            "Output only JSON, no other text. Example: {\"magnitude\":0.5,\"mask_target\":\"shirt\",\"prompt\":\"person wearing red shirt\"}"
        )
        full = f"{system}\n\nEdit request: {prompt}\nOutput:"
        
        response = llm.create_completion(
            full, max_tokens=500, temperature=0.1,
            echo=False
        )
        
        try:
            text = response['choices'][0]['text'].strip()
            print(f"[PromptRewriter] Raw LLM Response: {text}")
            
            # Búsqueda robusta del objeto JSON
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                text = match.group(0)
            else:
                # Si el LLM se olvidó las llaves, intentamos forzarlas si parece un dict
                if ":" in text and ("," in text or "\n" in text):
                    text = "{" + text + "}"
            
            # Limpieza de comillas simples (común en respuestas de LLMs pequeños)
            # Solo si no hay comillas dobles que sugieran que ya es JSON válido
            if '"' not in text and "'" in text:
                text = text.replace("'", '"')
            
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Intento final: limpiar caracteres no imprimibles o basura al inicio/final
                text = re.sub(r'^[^{]*', '', text)
                text = re.sub(r'[^}]*$', '', text)
                data = json.loads(text)

            # Validar que sea un diccionario, no una lista
            if not isinstance(data, dict):
                # Si es una lista, intentar interpretar los valores
                if isinstance(data, list) and len(data) >= 1:
                    # Primera posición podría ser magnitude
                    mag = data[0] if isinstance(data[0], (int, float)) else 0.5
                    mag = max(0.0, min(1.0, float(mag)))  # Clampear entre 0 y 1
                    print(f"[PromptRewriter] LLM devolvió lista, interpretando magnitude={mag}")
                    return {"prompt": prompt, "magnitude": mag, "mask_target": "subject", "reasoning": "Parsed from list"}
                print(f"[PromptRewriter] Warning: LLM devolvió tipo {type(data).__name__}, esperado dict. Respuesta: {text[:100]}")
                return {"prompt": prompt, "magnitude": 0.5, "mask_target": "subject", "reasoning": "Invalid response format"}
            
            # Limpieza y validación básica
            result = {
                "prompt": data.get("prompt", prompt),
                "magnitude": float(data.get("magnitude", 0.5)),
                "mask_target": data.get("mask_target", "subject"),
                "reasoning": data.get("reasoning", "Semantic analysis completed")
            }
            
            print(f"[PromptRewriter] Analysis: {result['reasoning']}")
            print(f"[PromptRewriter] Mag: {result['magnitude']} | Mask: {result['mask_target']}")
            
            return result
        except Exception as e:
            print(f"[PromptRewriter] Error parsing LLM JSON: {e}")
            return {"prompt": prompt, "magnitude": 0.5, "mask_target": "subject"}

_rewriter = None
def get_prompt_rewriter() -> PromptRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = PromptRewriter()
    return _rewriter
