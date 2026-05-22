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
    
    # Palabras clave para cambios estructurales (añadir/quitar objetos/personas)
    STRUCTURAL_KEYWORDS = [
        "anade", "add", "pon", "put", "coloca", "place", "insert", 
        "insertar", "quita", "remove", "borra", "delete", "erase",
        "fondo", "background", "persona", "person", "hombre", "man",
        "mujer", "woman", "chica", "girl", "chico", "boy", "objeto", "object"
    ]

    # Palabras clave para cambios de pose/anatomía
    POSE_KEYWORDS = [
        "de rodillas", "sentado", "de pie", "acostado", "pose", 
        "postura", "posicion", "gesto", "movimiento", "kneeling", 
        "sitting", "standing", "lying", "action", "corriendo", "running",
        "saltando", "jumping", "walking", "caminando"
    ]

    # Palabras clave para cambios de atributo (color, textura, ropa)
    ATTRIBUTE_KEYWORDS = [
        "color", "textura", "ropa", "clothes", "vestido", "dress",
        "camisa", "shirt", "pantalones", "pants", "pelo", "hair",
        "ojos", "eyes", "piel", "skin", "brillo", "bright", "oscuro", "dark",
        "estilo", "style", "filtro", "filter"
    ]

    def __init__(self):
        self._build_patterns()
    
    def _build_patterns(self):
        """Construye patrones regex para deteccion"""
        self._outpaint_pattern = self._create_pattern(self.OUTPAINT_KEYWORDS)
        self._inpaint_pattern = self._create_pattern(self.INPAINT_KEYWORDS)
        self._pose_pattern = self._create_pattern(self.POSE_KEYWORDS)
        self._structural_pattern = self._create_pattern(self.STRUCTURAL_KEYWORDS)
        self._attribute_pattern = self._create_pattern(self.ATTRIBUTE_KEYWORDS)

    def get_suggested_magnitude(self, prompt: str) -> float:
        """
        Calcula una magnitud (0.0 a 1.0) sugerida basada en la complejidad semántica.
        - Cambios de pose/estructura -> Alta magnitud (0.75 - 0.9)
        - Cambios de objeto/persona -> Alta magnitud (0.7 - 0.85)
        - Cambios de atributo/color -> Baja/Media magnitud (0.3 - 0.5)
        """
        if not prompt or not prompt.strip():
            return 0.5
            
        prompt_lower = prompt.lower()
        
        # Conteo de intenciones
        structural_count = len(self._structural_pattern.findall(prompt_lower))
        pose_count = len(self._pose_pattern.findall(prompt_lower))
        attribute_count = len(self._attribute_pattern.findall(prompt_lower))
        outpaint_count = len(self._outpaint_pattern.findall(prompt_lower))
        
        # Base conservadora
        magnitude = 0.4
        
        # Escalamiento por complejidad (No hardcoded if-else, sino acumulativo)
        magnitude += (pose_count * 0.25)       # Las poses requieren mucho denoise
        magnitude += (structural_count * 0.15) # Estructuras requieren denoise
        magnitude += (outpaint_count * 0.10)   # Outpainting requiere espacio para crear
        magnitude += (attribute_count * 0.05)  # Atributos son ligeros
        
        # Ajustes por palabras clave de "fuerza"
        if any(w in prompt_lower for w in ["totalmente", "completamente", "radical", "radicalmente", "total", "complete"]):
            magnitude += 0.2
            
        # Límites de seguridad
        return max(0.25, min(0.95, magnitude))

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
