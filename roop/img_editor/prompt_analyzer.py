# -*- coding: utf-8 -*-
"""
PromptAnalyzer - Analisis de prompts para edicion de imagenes

Este modulo analiza el prompt del usuario y determina automaticamente
si se requiere inpainting, outpainting o img2img.
"""

from enum import Enum
from typing import Tuple
import re


class EditingMode(Enum):
    """Modo de edicion detectado"""
    INPAINT = "inpaint"
    OUTPAINT = "outpaint"
    IMG2IMG = "img2img"


class PromptAnalyzer:
    """
    Analiza prompts para determinar el tipo de edicion necesaria.
    
    Detecta palabras clave que indican:
    - OUTPAINT: anadir contenido, expandir, etc.
    - INPAINT: modificar areas especificas
    - IMG2IMG: modificacion global
    """
    
    # Palabras clave para outpainting (anadir contenido)
    OUTPAINT_KEYWORDS = [
        "anade", "add", "mas", "more", "al lado", "al fondo",
        "a la izquierda", "a la derecha", "detras", "frente",
        "alrededor", "expand", "expande", "amplia", "anadi",
        "pon", "put", "coloca", "place", "insert", "insertar",
        "aumenta", "extend", "extiende"
    ]
    
    # Palabras clave para inpainting (modificar area)
    INPAINT_KEYWORDS = [
        "cambia", "change", "modifica", "modify", "sustituye",
        "replace", "convierte", "convert", "cambiale", "edit",
        "edit", "redisenno", "redisenar", "transformation",
        "transforma", "mejora", "ajusta", "corrige", "fix"
    ]
    
    # Palabras clave para pose/cuerpo
    POSE_KEYWORDS = [
        "de rodillas", "sentado", "de pie", "acostado", "de pie",
        "pose", "postura", "posicion", "gesto", "movimiento",
        "kneeling", "sitting", "standing", "lying", "pose"
    ]
    
    def __init__(self):
        self._build_patterns()
    
    def _build_patterns(self):
        """Construye patrones regex para deteccion"""
        self._outpaint_pattern = self._create_pattern(self.OUTPAINT_KEYWORDS)
        self._inpaint_pattern = self._create_pattern(self.INPAINT_KEYWORDS)
        self._pose_pattern = self._create_pattern(self.POSE_KEYWORDS)
    
    def _create_pattern(self, keywords):
        """Crea un patron regex de palabras clave"""
        escaped = [re.escape(kw) for kw in keywords]
        return re.compile(r'\b(' + '|'.join(escaped) + r')\b', re.IGNORECASE)
    
    def analyze(self, prompt: str) -> Tuple[EditingMode, float]:
        """
        Analiza un prompt y devuelve el modo de edicion.
        
        Args:
            prompt: Texto del prompt del usuario
            
        Returns:
            (EditingMode, confidence)
            confidence: 0.0 a 1.0 indicando certeza de la deteccion
        """
        if not prompt or not prompt.strip():
            return EditingMode.IMG2IMG, 0.0
        
        prompt_lower = prompt.lower()
        
        # Contar coincidencias
        outpaint_matches = len(self._outpaint_pattern.findall(prompt_lower))
        inpaint_matches = len(self._inpaint_pattern.findall(prompt_lower))
        pose_matches = len(self._pose_pattern.findall(prompt_lower))
        
        # Logica de decision
        if outpaint_matches > 0 and outpaint_matches >= inpaint_matches:
            # Probablemente outpainting
            confidence = min(0.5 + (outpaint_matches * 0.2), 0.95)
            return EditingMode.OUTPAINT, confidence
            
        elif inpaint_matches > 0 or pose_matches > 0:
            # Probablemente inpainting
            confidence = min(0.5 + ((inpaint_matches + pose_matches) * 0.2), 0.95)
            return EditingMode.INPAINT, confidence
            
        else:
            # Por defecto, img2img global
            return EditingMode.IMG2IMG, 0.3
    
    def extract_mode_prompt(self, full_prompt: str, mode: EditingMode) -> str:
        """
        Extrae la parte relevante del prompt para el modo.
        
        Args:
            full_prompt: Prompt completo del usuario
            mode: Modo de edicion detectado
            
        Returns:
            Prompt optimizado para el modo
        """
        # Por ahora, retornar el prompt completo
        # En el futuro podriamos extraer solo la parte relevante
        return full_prompt.strip()
    
    def get_suggestions(self, prompt: str, mode: EditingMode) -> list:
        """
        Devuelve sugerencias para mejorar el prompt.
        
        Args:
            prompt: Prompt original
            mode: Modo detectado
            
        Returns:
            Lista de sugerencias
        """
        suggestions = []
        
        if mode == EditingMode.OUTPAINT:
            suggestions.append("Para outpainting, especifica la posicion: 'a la izquierda', 'a la derecha', 'al fondo'")
            suggestions.append("Describe que quieres anadir: 'un hombre', 'arboles', 'edificios'")
            
        elif mode == EditingMode.INPAINT:
            suggestions.append("Para inpainting, describe exactamente que quieres cambiar")
            suggestions.append("Usa coordenadas si es necesario: 'la cara', 'el fondo', 'la ropa'")
            
        elif mode == EditingMode.IMG2IMG:
            suggestions.append("Describe el resultado final que deseas")
            suggestions.append("Incluye estilo, iluminacion, atmosfera")
        
        # Sugerencias generales
        if len(prompt) < 20:
            suggestions.append("Considera agregar mas detalles al prompt para mejores resultados")
        
        return suggestions
    
    def is_simple_edition(self, prompt: str) -> bool:
        """
        Detecta si es una edicion simple (solo color, filtro, etc.)
        """
        simple_keywords = ["color", "filtro", "brillo", "contraste", "saturacion"]
        prompt_lower = prompt.lower()
        
        return any(kw in prompt_lower for kw in simple_keywords)


def analyze_prompt(prompt: str) -> Tuple[EditingMode, float]:
    """
    Funcion helper para analisis rapido de prompts.
    """
    analyzer = PromptAnalyzer()
    return analyzer.analyze(prompt)
