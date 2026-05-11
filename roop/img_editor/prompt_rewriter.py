#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Rewriter - Inteligencia Semántica Pura (Sin Hardcoding)
"""

import os, json, re
from typing import Optional, Dict, List, Tuple

MOONDREAM_TEXT_PATH = r"D:\PROJECTS\AUTOAUTO\models\moondream\moondream2-text-model-f16.gguf"

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
            
            # Rutas de búsqueda unificadas
            root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            possible_paths = [
                MOONDREAM_TEXT_PATH,
                os.path.join(root, "models", "moondream", "moondream2-text-model-f16.gguf"),
                os.path.join(root, "models", "moondream2-text-model-f16.gguf"),
                os.path.join(root, "models", "moondream2.gguf"),
            ]
            
            model_path = None
            for p in possible_paths:
                if os.path.exists(p):
                    model_path = p
                    break
            
            if not model_path:
                print("[PromptRewriter] ERROR: No se encontró el modelo moondream en ninguna ruta conocida.")
                return

            print(f"[PromptRewriter] Cargando rewriter semántico: {os.path.basename(model_path)}")
            self._llm = Llama(
                model_path=model_path,
                n_gpu_layers=0, # Siempre CPU para no molestar a Flux
                n_ctx=1024,     # Contexto suficiente para prompts
                verbose=False
            )
            print("[PromptRewriter] Rewriter listo en CPU")
        except Exception as e:
            print(f"[PromptRewriter] LLM local no disponible: {e}")

    def rewrite(self, prompt: str) -> Dict:
        """Devuelve un diccionario con el análisis completo del LLM"""
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

    def _rewrite_with_llm(self, prompt: str, llm) -> Dict:
        system = (
            "You are an image editor AI. Convert user instruction to technical parameters.\n"
            "Respond ONLY with valid JSON like: {\"reasoning\": \"short explanation\", \"prompt\": \"english description\", \"magnitude\": 0.5, \"mask_target\": \"object\"}\n"
            "magnitude: 0.1=small change, 0.5=medium, 0.9=radical (nudity, full clothing)\n"
            "mask_target: exact object to mask (face, shirt, background, etc)"
        )
        full = f"{system}\n\nUser: \"{prompt}\"\nJSON:"
        
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
