#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Rewriter - Reescritor de Prompts con Qwen3 4B local

Convierte prompts simples en prompts detallados automáticamente.
Ejemplo: "bailando" → "dynamic dancing pose, energetic movement, flowing hair, dramatic lighting"
"""

import os
from typing import Optional, Dict, List, Tuple


MOONDREAM_TEXT_PATH = r"D:\PROJECTS\AUTOAUTO\models\moondream\moondream2-text-model-f16.gguf"

class PromptRewriter:
    """Reescribe prompts usando Moondream2 text model local"""

    def __init__(self):
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        if self._llm is not None:
            return
        try:
            import os as _os
            _os.environ['PATH'] = (
                r'D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\llama_cpp\lib'
                + _os.pathsep + r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin'
                + _os.pathsep + _os.environ.get('PATH', '')
            )
            from llama_cpp import Llama
            if not _os.path.exists(MOONDREAM_TEXT_PATH):
                print(f"[PromptRewriter] Modelo no encontrado: {MOONDREAM_TEXT_PATH}")
                return
            print(f"[PromptRewriter] Cargando Moondream2 text model...")
            self._llm = Llama(
                model_path=MOONDREAM_TEXT_PATH,
                n_gpu_layers=0,
                n_ctx=2048,
                verbose=False
            )
            print(f"[PromptRewriter] ✅ Moondream2 text model listo")
        except Exception as e:
            print(f"[PromptRewriter] LLM local no disponible: {e}")

    def rewrite(self, prompt: str, analysis: Dict[str, bool] = None) -> Tuple[str, str]:
        if self._llm is not None:
            try:
                return self._rewrite_with_llm(prompt, self._llm), "medium"
            except Exception as e:
                print(f"[PromptRewriter] Error con LLM: {e}")
        return (self._rewrite_with_templates(prompt, analysis), "medium")

    def _rewrite_with_llm(self, prompt: str, llm) -> str:
        system = "Eres un experto en prompts de IA. Mejora el prompt manteniendo el significado original. Sé descriptivo con iluminación, composición y detalles. Responde SOLO con el prompt mejorado, máximo 100 palabras."
        full = f"{system}\n\nPrompt original: \"{prompt}\"\n\nPrompt mejorado:"
        response = llm.create_completion(
            full, max_tokens=200, temperature=0.4,
            stop=["\"", "\n\n"], echo=False
        )
        result = response['choices'][0]['text'].strip()
        result = result.strip('"').strip()
        if len(result) < 10:
            result = prompt
        print(f"[PromptRewriter] ✅ Reescrito con LLM local: {result[:50]}...")
        return result

    def _rewrite_with_templates(self, prompt: str, analysis: Dict[str, bool] = None) -> Tuple[str, str]:
        """Reescribe usando templates predefinidos y estima intensidad - ESTILO IMAGINE"""

        prompt_lower = prompt.lower()
        enhancements = []

        # Detectar tipo de prompt y aplicar template
        if analysis:
            if analysis.get('use_openpose'):
                # Prompts de pose - estilo imagine
                if any(kw in prompt_lower for kw in ['bailando', 'bailar', 'dance']):
                    enhancements = [
                        "dynamic dancing pose",
                        "energetic movement",
                        "flowing hair",
                        "motion blur background",
                        "dramatic cinematic lighting",
                        "action shot",
                        "wide angle lens"
                    ]
                elif any(kw in prompt_lower for kw in ['sentado', 'sit']):
                    enhancements = [
                        "sitting portraiture",
                        "relaxed elegant pose",
                        "natural posture",
                        "soft cinematic lighting",
                        "detailed face",
                        "portrait photography",
                        "shallow depth of field"
                    ]
                elif any(kw in prompt_lower for kw in ['de pie', 'standing']):
                    enhancements = [
                        "standing full body",
                        "confident posture",
                        "hero shot angle",
                        "professional studio lighting",
                        "cinematic composition",
                        "dynamic pose"
                    ]
                elif any(kw in prompt_lower for kw in ['acostado', 'lying']):
                    enhancements = [
                        "lying down pose",
                        "relaxed reclining",
                        "sensual lighting",
                        "soft shadows",
                        "detailed skin texture"
                    ]

            elif analysis.get('use_tile'):
                # Prompts de mejora de calidad - máxima calidad
                enhancements = [
                    "ultra detailed",
                    "sharp focus",
                    "8K HDR",
                    "professional color grading",
                    "crystal clear",
                    "high resolution scan",
                    "noise free"
                ]

            elif analysis.get('use_inpaint'):
                # Prompts de cambio de ropa/cuerpo - estilo fashion/arte
                if any(kw in prompt_lower for kw in ['desnuda', 'naked', 'nude', 'sin ropa']):
                    enhancements = [
                        "natural skin texture",
                        "anatomical accuracy",
                        "realistic body proportions",
                        "professional lighting",
                        "artistic nude photography",
                        "film noir style"
                    ]
                elif any(kw in prompt_lower for kw in ['ropa', 'clothing', 'outfit', 'vestido']):
                    enhancements = [
                        "fashion photography",
                        "designer outfit",
                        "detailed fabric texture",
                        "professional photoshoot",
                        "runway style",
                        "editorial photography"
                    ]

            elif analysis.get('use_face_edit'):
                # Prompts de expresión facial - retrato de alta calidad
                if any(kw in prompt_lower for kw in ['sonrisa', 'smile', 'feliz']):
                    enhancements = [
                        "beautiful genuine smile",
                        "happy expression",
                        "sparkling eyes",
                        "warm golden hour lighting",
                        "cinematic portrait",
                        "detailed facial features"
                    ]
                elif any(kw in prompt_lower for kw in ['triste', 'sad']):
                    enhancements = [
                        "emotional depth",
                        "melancholic mood",
                        "dramatic chiaroscuro lighting",
                        "cinematic portrait",
                        "tear drops"
                    ]

        # Construir prompt final
        if enhancements:
            enhanced_prompt = f"{prompt}, {', '.join(enhancements)}"
        else:
            enhanced_prompt = f"{prompt}"

        print(f"[PromptRewriter] ✅ Reescrito (limpio): {enhanced_prompt[:50]}...")
        return enhanced_prompt


# Instancia global
_rewriter = None

def get_prompt_rewriter() -> PromptRewriter:
    """Obtiene instancia global del rewriter"""
    global _rewriter
    if _rewriter is None:
        _rewriter = PromptRewriter()
    return _rewriter
