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
            "Eres el núcleo de inteligencia de un editor de imágenes. Tu misión es transformar una instrucción de usuario en parámetros técnicos.\n"
            "Debes responder EXCLUSIVAMENTE en formato JSON con estos campos:\n"
            "1. 'reasoning': Breve explicación de por qué el cambio es sutil, medio o radical.\n"
            "2. 'prompt': Traducción al inglés técnico, muy descriptiva y cinematográfica (ej: 'hyper-realistic photo of a person wearing...').\n"
            "3. 'magnitude': Valor de 0.0 a 1.0 (0.1=sonrisa/ojos, 0.5=objetos/clima, 0.9=ropa completa/desnudo/cuerpo).\n"
            "4. 'mask_target': El objeto físico exacto en inglés que debe ser enmascarado para este cambio (ej: 'shirt', 'eyes', 'background', 'whole body').\n"
            "\nIMPORTANTE: No uses frases genéricas como 'traducción detallada'. Escribe prompts reales."
        )
        full = f"{system}\n\nInstrucción del usuario: \"{prompt}\"\n\nJSON:"
        
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
