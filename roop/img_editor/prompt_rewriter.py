#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Rewriter - Reescritor de Prompts con LLM

Convierte prompts simples en prompts detallados automáticamente.
Ejemplo: "bailando" → "dynamic dancing pose, energetic movement, flowing hair, dramatic lighting"

Soporta:
- Ollama (local, gratuito)
- Ollama API remota
- Fallback con templates predefinidos
"""

import requests
import json
from typing import Optional, Dict, List, Tuple


class PromptRewriter:
    """Reescribe prompts usando LLM"""
    
    def __init__(self):
        self.ollama_url = "http://127.0.0.1:11434"
        self.model = "llama3.2"  # Modelo ligero y rápido
        self.available = None
        
    def check_availability(self) -> bool:
        """Verifica si Ollama está disponible"""
        if self.available is not None:
            return self.available
            
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            if response.status_code == 200:
                self.available = True
                print(f"[PromptRewriter] ✅ Ollama disponible")
                return True
        except:
            pass
        
        self.available = False
        print("[PromptRewriter] ❌ Ollama no disponible, usando templates")
        return False
    
    def rewrite(self, prompt: str, analysis: Dict[str, bool] = None) -> Tuple[str, str]:
        """
        Reescribe un prompt y también clasifica la intensidad necesaria.
        
        Returns:
            (rewritten_prompt, intensity) where intensity is "low", "medium", or "high"
        """
        # Intentar con Ollama primero
        if self.check_availability():
            try:
                return self._rewrite_with_ollama(prompt, analysis)
            except Exception as e:
                print(f"[PromptRewriter] Error con Ollama: {e}")
        
        # Fallback a templates
        return (self._rewrite_with_templates(prompt, analysis), "medium")
    
    def _rewrite_with_ollama(self, prompt: str, analysis: Dict[str, bool] = None) -> Tuple[str, str]:
        """Reescribe usando Ollama LLM y clasifica intensidad"""
        
        # System prompt mejorado: pide prompt + clasificación de intensidad
        system_prompt = """Eres un experto en Stable Diffusion y generación de imágenes.

Tu tarea es:
1. MEJORAR EL PROMPT: convertir prompts simples en prompts detallados para SD.
2. CLASIFICAR LA INTENSIDAD DEL CAMBIO requerido:

CLASES DE INTENSIDAD:
- HIGH: Transformación COMPLETA del cuerpo o escena. Incluye:
  * Cambios de postura (bailando, sentado, de pie, acostado)
  * Desnudez o quitar toda la ropa
  * Cambiar el entorno/fondo completamente
  * Cambios drásticos de apariencia (cabello, rostro entero)
  * Acciones que afectan toda la imagen (corriendo, saltando, etc.)
  
- MEDIUM: Cambios moderados que afectan parte del cuerpo:
  * Cambio de ropa (pero mantiene postura)
  * Accesorios, colores, expresiones faciales
  * Ajustes de iluminación/estilo
  
- LOW: Mejoras sutiles:
  * Aumentar calidad, nitidez, resolución
  * Cambios mínimos de color/brillo
  * Eliminar pequeños defects

DEVUELVE SOLO ESTE FORMATO JSON (sin texto extra):
{"intensity": "high|medium|low", "prompt": "el prompt mejorado aquí"}

Reglas para el prompt mejorado:
- NO agregues términos de calidad automáticos (masterpiece, etc.) a menos que el usuario lo pida.
- Añade iluminación cinematográfica: "cinematic lighting, dramatic lighting, professional photography"
- Describe la escena con detalle (posición, ambiente, estilo)
- Mantén el idioma original (español o inglés)
- Máximo 150 palabras
- No agregues contenido NSFW"""

        # Construir user prompt (más conciso, el LLM ya sabe clasificar)
        user_prompt = f"""Prompt original: "{prompt}"

Analiza este prompt y devuelve JSON con "intensity" (high/medium/low) y "prompt" (mejorado)."""
        
        # Llamar a Ollama
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,  # Más determinístico para clasificación
                        "max_tokens": 300
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get('response', '').strip()
                
                # Intentar parsear JSON
                import json
                try:
                    # Extraer JSON si está dentro de texto
                    json_start = response_text.find('{')
                    json_end = response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        json_str = response_text[json_start:json_end]
                        data = json.loads(json_str)
                        rewritten = data.get('prompt', '').strip()
                        intensity = data.get('intensity', 'medium').lower()
                        
                        # Validar intensidad
                        if intensity not in ['high', 'medium', 'low']:
                            intensity = 'medium'
                        
                        # Limpiar respuesta
                        rewritten = rewritten.replace('"', '').strip()
                        if len(rewritten) > 10:
                            print(f"[PromptRewriter] ✅ Reescrito con Ollama (intensity={intensity}): {rewritten[:50]}...")
                            return rewritten, intensity
                except Exception as json_e:
                    print(f"[PromptRewriter] Error parseando JSON: {json_e}, usando fallback")
                    # Fallback: usar texto completo como prompt, intensidad basada en palabras
                    intensity = self._estimate_intensity_fallback(prompt)
                    return response_text[:300], intensity
                    
        except Exception as e:
            print(f"[PromptRewriter] Error en Ollama: {e}")
        
        # Fallback
        return (self._rewrite_with_templates(prompt, analysis), self._estimate_intensity_fallback(prompt))
    
    def _estimate_intensity_fallback(self, prompt: str) -> str:
        """Estima intensidad basada en palabras clave (fallback)"""
        prompt_lower = prompt.lower()
        # Palabras que sugieren alta intensidad (transformación completa)
        high_keywords = ['desnuda', 'desnudo', 'naked', 'nude', 'sin ropa', 'cambia de postura', 
                         'cambio de pose', 'bailando', 'sentado', 'de pie', 'acostado', 'cuerpo completo',
                         'cuerpo entero', 'full body', 'transforma', 'convierte']
        if any(kw in prompt_lower for kw in high_keywords):
            return 'high'
        
        # Palabras que sugieren baja intensidad
        low_keywords = ['mejora', 'calidad', 'nitidez', 'resolución', 'afina', 'suave', 'ligero']
        if any(kw in prompt_lower for kw in low_keywords):
            return 'low'
        
        return 'medium'
    
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
