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
from typing import Optional, Dict, List


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
    
    def rewrite(self, prompt: str, analysis: Dict[str, bool] = None) -> str:
        """
        Reescribe un prompt para hacerlo más detallado.
        
        Args:
            prompt: Prompt original del usuario
            analysis: Análisis del prompt (use_openpose, use_tile, etc.)
            
        Returns:
            Prompt mejorado y detallado
        """
        # Intentar con Ollama primero
        if self.check_availability():
            try:
                return self._rewrite_with_ollama(prompt, analysis)
            except Exception as e:
                print(f"[PromptRewriter] Error con Ollama: {e}")
        
        # Fallback a templates
        return self._rewrite_with_templates(prompt, analysis)
    
    def _rewrite_with_ollama(self, prompt: str, analysis: Dict[str, bool] = None) -> str:
        """Reescribe usando Ollama LLM"""
        
        # Construir system prompt
        system_prompt = """Eres un experto en Stable Diffusion y generación de imágenes.
Tu trabajo es convertir prompts simples en prompts detallados y efectivos para SD.

Reglas:
1. Añade términos de calidad: "high quality, detailed, realistic, masterpiece, best quality"
2. Describe la escena con más detalle
3. Añade iluminación: "dramatic lighting, cinematic lighting, soft lighting"
4. Añade ambiente: "atmospheric, moody, vibrant"
5. Mantén el idioma original del prompt (español o inglés)
6. No añadas contenido NSFW o prohibido
7. Sé específico pero conciso (máximo 150 palabras)

Ejemplos:
- "bailando" → "dynamic dancing pose, energetic movement, flowing hair, dramatic lighting, high quality, detailed, masterpiece"
- "sentado" → "person sitting comfortably, relaxed pose, natural posture, soft lighting, detailed face, high quality"
- "mejora calidad" → "ultra detailed, sharp focus, 8K resolution, professional photography, masterpiece, best quality, realistic textures"
- "cambia ropa" → "elegant outfit, fashionable clothing, detailed fabric texture, realistic folds, high quality, professional photography"
"""

        # Construir user prompt
        user_prompt = f"""Mejora este prompt para Stable Diffusion:

Prompt original: "{prompt}"
"""
        
        if analysis:
            enhancements = []
            if analysis.get('use_openpose'):
                enhancements.append("- Es una pose específica, describe el movimiento y posición del cuerpo")
            if analysis.get('use_tile'):
                enhancements.append("- Es mejora de calidad, enfatiza detalles, resolución y nitidez")
            if analysis.get('use_inpaint'):
                enhancements.append("- Es cambio de ropa/cuerpo, describe la nueva apariencia con detalle")
            if analysis.get('use_face_edit'):
                enhancements.append("- Es expresión facial, describe la emoción y expresión")
            
            if enhancements:
                user_prompt += "\nConsideraciones:\n" + "\n".join(enhancements)
        
        user_prompt += "\n\nPrompt mejorado (solo el prompt, sin explicaciones):"

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
                        "temperature": 0.7,
                        "max_tokens": 200
                    }
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                rewritten = result.get('response', '').strip()
                
                # Limpiar respuesta
                rewritten = rewritten.replace('"', '').strip()
                if len(rewritten) > 10:
                    print(f"[PromptRewriter] ✅ Reescrito con Ollama: {rewritten[:50]}...")
                    return rewritten
                    
        except Exception as e:
            print(f"[PromptRewriter] Error en Ollama: {e}")
        
        # Fallback
        return self._rewrite_with_templates(prompt, analysis)
    
    def _rewrite_with_templates(self, prompt: str, analysis: Dict[str, bool] = None) -> str:
        """Reescribe usando templates predefinidos"""
        
        prompt_lower = prompt.lower()
        enhancements = []
        
        # Términos de calidad base
        quality_terms = "high quality, detailed, realistic, masterpiece, best quality, professional photography"
        
        # Detectar tipo de prompt y aplicar template
        if analysis:
            if analysis.get('use_openpose'):
                # Prompts de pose
                if any(kw in prompt_lower for kw in ['bailando', 'bailar', 'dance']):
                    enhancements = [
                        "dynamic dancing pose",
                        "energetic movement",
                        "flowing hair",
                        "dramatic lighting",
                        "action shot"
                    ]
                elif any(kw in prompt_lower for kw in ['sentado', 'sit']):
                    enhancements = [
                        "sitting comfortably",
                        "relaxed pose",
                        "natural posture",
                        "soft lighting",
                        "detailed face"
                    ]
                elif any(kw in prompt_lower for kw in ['de pie', 'standing']):
                    enhancements = [
                        "standing pose",
                        "natural stance",
                        "confident posture",
                        "full body shot",
                        "professional lighting"
                    ]
                elif any(kw in prompt_lower for kw in ['acostado', 'lying']):
                    enhancements = [
                        "lying down pose",
                        "relaxed position",
                        "natural expression",
                        "soft lighting",
                        "detailed"
                    ]
                    
            elif analysis.get('use_tile'):
                # Prompts de mejora de calidad
                enhancements = [
                    "ultra detailed",
                    "sharp focus",
                    "8K resolution",
                    "professional photography",
                    "realistic textures",
                    "crystal clear",
                    "HDR"
                ]
                
            elif analysis.get('use_inpaint'):
                # Prompts de cambio de ropa/cuerpo
                if any(kw in prompt_lower for kw in ['desnuda', 'naked', 'nude', 'sin ropa']):
                    enhancements = [
                        "natural skin",
                        "realistic body",
                        "detailed skin texture",
                        "body details",
                        "natural lighting"
                    ]
                elif any(kw in prompt_lower for kw in ['ropa', 'clothing', 'outfit', 'vestido']):
                    enhancements = [
                        "elegant outfit",
                        "fashionable clothing",
                        "detailed fabric texture",
                        "realistic folds",
                        "professional photography"
                    ]
                    
            elif analysis.get('use_face_edit'):
                # Prompts de expresión facial
                if any(kw in prompt_lower for kw in ['sonrisa', 'smile', 'feliz']):
                    enhancements = [
                        "beautiful smile",
                        "happy expression",
                        "joyful",
                        "warm lighting",
                        "detailed face"
                    ]
                elif any(kw in prompt_lower for kw in ['triste', 'sad']):
                    enhancements = [
                        "sad expression",
                        "melancholic",
                        "emotional",
                        "moody lighting",
                        "detailed face"
                    ]
        
        # Construir prompt final
        if enhancements:
            enhanced_prompt = f"{prompt}, {', '.join(enhancements)}, {quality_terms}"
        else:
            enhanced_prompt = f"{prompt}, {quality_terms}"
        
        print(f"[PromptRewriter] ✅ Reescrito con templates: {enhanced_prompt[:50]}...")
        return enhanced_prompt


# Instancia global
_rewriter = None

def get_prompt_rewriter() -> PromptRewriter:
    """Obtiene instancia global del rewriter"""
    global _rewriter
    if _rewriter is None:
        _rewriter = PromptRewriter()
    return _rewriter
