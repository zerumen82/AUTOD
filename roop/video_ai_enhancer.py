"""
video_ai_enhancer.py - Modulo de mejora de Video AI para LTX-Video 2

Caracteristicas:
- Parseo inteligente de prompts para detectar acciones
- Optimizacion de prompts para LTX-2
- Configuracion de parametros segun tipo de accion
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class ActionSegment:
    """Representa un segmento de accion en el prompt."""
    text: str
    start_pos: int
    end_pos: int
    action_type: str  # 'movement', 'expression', 'camera', 'ambient'
    intensity: float  # 0.0 a 1.0


@dataclass
class VideoConfig:
    """Configuracion del video."""
    frames: int
    fps: int
    width: int
    height: int
    strength: float
    seed: int


# ============================================================================
# DICCIONARIOS DE ANALISIS
# ============================================================================

# Tipos de accion soportados
ACTION_TYPES = {
    'movement': ['walking', 'running', 'flying', 'dancing', 'jumping', 'swimming',
                 'driving', 'floating', 'turning', 'spinning', 'gliding', 'crawling'],
    'expression': ['smiling', 'laughing', 'crying', 'talking', 'singing', 'winking',
                   'frowning', 'surprised', 'angry', 'blinking', 'nodding', 'looking'],
    'camera': ['zoom', 'pan', 'rotate', 'tilt', 'dolly', 'tracking', 'pov',
               'close_up', 'wide_shot', 'orbit', ' handheld'],
    'ambient': ['wind', 'rain', 'fire', 'water', 'leaves', 'clouds', 'particles',
                'lights', 'smoke', 'dust', 'sunlight', 'shadows']
}

# Patrones de acciones multiples
MULTI_ACTION_PATTERNS = [
    (r'(\w+[^,]+),\s*then\s+(\w+[^.]+)', 'then'),
    (r'(\w+[^,]+)\s+and\s+then\s+(\w+[^.]+)', 'and_then'),
    (r'(\w+[^,]+),\s*(\w+[^,]+),\s*(\w+[^.]+)', 'list'),
    (r'(\w+[^,]+)\s+while\s+(\w+[^.]+)', 'while'),
]


# ============================================================================
# PARSEO DE PROMPTS
# ============================================================================

def parse_action_prompt(prompt: str) -> Tuple[List[ActionSegment], str]:
    """
    Parse un prompt para detectar acciones.

    Args:
        prompt: El prompt original del usuario

    Returns:
        Tupla de (lista de segmentos de accion, prompt mejorado)
    """
    segments = []
    prompt_clean = prompt.strip()
    prompt_lower = prompt_clean.lower()

    # Detectar patrones de acciones multiples
    for pattern, pattern_type in MULTI_ACTION_PATTERNS:
        matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
        if matches:
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    for i, action in enumerate(match[:2]):
                        action = action.strip()
                        if action and len(action) > 3:
                            segments.append(ActionSegment(
                                text=action,
                                start_pos=0,
                                end_pos=0,
                                action_type=_detect_action_type(action),
                                intensity=_estimate_action_intensity(action)
                            ))
            break

    # Si no se detecto accion multiple, tratar como una sola accion
    if not segments:
        segments.append(ActionSegment(
            text=prompt_clean,
            start_pos=0,
            end_pos=len(prompt_clean),
            action_type=_detect_action_type(prompt_clean),
            intensity=_estimate_action_intensity(prompt_clean)
        ))

    # Crear prompt mejorado para LTX-2
    enhanced_prompt = _enhance_prompt_for_ltx2(prompt_clean, segments)

    return segments, enhanced_prompt


def _detect_action_type(text: str) -> str:
    """Detecta el tipo de accion basado en palabras clave."""
    text_lower = text.lower()

    for action_type, keywords in ACTION_TYPES.items():
        for keyword in keywords:
            if keyword in text_lower:
                return action_type

    return 'movement'  # Default


def _estimate_action_intensity(text: str) -> float:
    """Estima la intensidad de la accion (0.0 a 1.0)."""
    text_lower = text.lower()

    high_intensity = ['running', 'jumping', 'dancing', 'flying', 'spinning',
                      'fast', 'quickly', 'energetic', 'explosive']
    low_intensity = ['slow', 'gentle', 'subtle', 'soft', 'calm', 'peaceful',
                     'relaxed', 'gradual', 'slowly']

    if any(word in text_lower for word in high_intensity):
        return 0.8
    elif any(word in text_lower for word in low_intensity):
        return 0.3
    else:
        return 0.5


def _enhance_prompt_for_ltx2(original: str, segments: List[ActionSegment]) -> str:
    """Mejora el prompt especificamente para LTX-2."""
    enhanced = original

    # Añadir keywords de movimiento suave (LTX-2 es muy sensible)
    has_movement = any(s.action_type == 'movement' for s in segments)
    has_camera = any(s.action_type == 'camera' for s in segments)

    if has_movement:
        enhanced += ", smooth natural motion, fluid movement"

    if has_camera:
        enhanced += ", smooth camera work, professional cinematography"

    # LTX-2 prefiere descripciones naturales
    enhanced += ", high quality, realistic, cinematic lighting"

    return enhanced


# ============================================================================
# CONFIGURACION DE VIDEO
# ============================================================================

def get_video_config(prompt: str, duration: str = "medium") -> VideoConfig:
    """
    Obtiene la configuracion del video basada en el prompt y duracion.

    Args:
        prompt: El prompt del video
        duration: 'short', 'medium', 'long' (affecta frames)

    Returns:
        Objeto VideoConfig con parametros optimizados
    """
    import random

    segments, _ = parse_action_prompt(prompt)
    has_fast_movement = any(s.intensity > 0.7 for s in segments)

    # Duracion y frames (LTX-2: multiplo de 8+1)
    duration_map = {
        'short': 25,    # ~1 seg
        'medium': 73,   # ~3 seg
        'long': 121     # ~5 seg
    }
    frames = duration_map.get(duration, 73)

    # Resolution (divisible por 64)
    width, height = 704, 480

    # Strength segun el tipo de accion
    if has_fast_movement:
        strength = 0.5  # Menos fuerza para movimiento rapido
    else:
        strength = 0.65  # Default

    return VideoConfig(
        frames=frames,
        fps=24,
        width=width,
        height=height,
        strength=strength,
        seed=random.randint(0, 1000000000)
    )


def format_prompt_for_ltx2(prompt: str) -> str:
    """
    Formatea el prompt especificamente para LTX-2.

    Args:
        prompt: El prompt original

    Returns:
        Prompt formateado para LTX-2
    """
    segments, enhanced = parse_action_prompt(prompt)
    return enhanced


# ============================================================================
# INTERFAZ SIMPLIFICADA
# ============================================================================

def analyze_prompt(prompt: str) -> Dict:
    """
    Analiza un prompt y devuelve la informacion relevante.

    Args:
        prompt: El prompt a analizar

    Returns:
        Diccionario con: segments, enhanced_prompt, video_config
    """
    segments, enhanced_prompt = parse_action_prompt(prompt)
    config = get_video_config(prompt)

    return {
        'original_prompt': prompt,
        'enhanced_prompt': enhanced_prompt,
        'segments': [
            {
                'text': s.text,
                'type': s.action_type,
                'intensity': s.intensity
            }
            for s in segments
        ],
        'video_config': {
            'frames': config.frames,
            'fps': config.fps,
            'width': config.width,
            'height': config.height,
            'strength': config.strength
        }
    }


if __name__ == "__main__":
    # Demo
    test_prompts = [
        "woman walking on beach, then turning around",
        "cat playing with ball, jumping and rolling",
        "person talking to camera with natural expression"
    ]

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print('='*60)
        result = analyze_prompt(prompt)
        print(f"Enhanced: {result['enhanced_prompt']}")
        print(f"Segments: {len(result['segments'])}")
        for seg in result['segments']:
            print(f"  - {seg['type']}: {seg['text'][:50]}")
        print(f"Config: {result['video_config']}")
