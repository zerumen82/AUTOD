# Configuración de SmartPrompt para FLUX.2-klein NSFW

# Niveles de NSFW
NSFW_LEVELS = {
    'mild': {
        'description': 'Contenido sugerido, ropa translúcida, poses modestas',
        'examples': [
            'woman in lingerie, suggestive pose, sheer clothing',
            'male torso, bare chest, athletic body',
            'bikini, swimwear, beach scene'
        ],
        'avoid': ['explicit', 'genitals', 'full nudity', 'sexual acts']
    },
    
    'moderate': {
        'description': 'Nudez parcial, poses sensuales, pechos descubiertos',
        'examples': [
            'nude woman, bare breasts, covering nipples',
            'topless female, natural lighting, artistic',
            'male nude, waist up, muscular torso'
        ],
        'avoid': ['full frontal', 'genitals visible', 'sexual positions']
    },
    
    'explicit': {
        'description': 'Nudez completa explícita, poses abiertas',
        'examples': [
            'fully nude female, spread legs, explicit anatomy',
            'nude male, erect, detailed genitals',
            'explicit nsfw, adult content, 18+'
        ],
        'avoid': ['violence', 'non-consensual', 'minors']
    },
    
    'extreme': {
        'description': 'Contenido pornográfico hardcore',
        'examples': [
            'hardcore sex scene, explicit penetration',
            'pornographic, XXX content, adult film still',
            'sexual intercourse, explicit adult content'
        ],
        'avoid': ['illegal activities', 'non-consensual', 'extreme fetishes']
    }
}

# Palabras clave NSFW por nivel (se añaden automáticamente)
NSFW_KEYWORDS = {
    'mild': ['nsfw', 'suggestive', 'see-through', 'lingerie', 'bikini'],
    'moderate': ['nude', 'naked', 'bare breasts', 'topless', 'explicit'],
    'explicit': ['fully nude', 'genitals', 'spread legs', 'adult content', '18+'],
    'extreme': ['porn', 'hardcore', ' XXX', 'sexual intercourse', 'explicit sex']
}

# Calidad base (siempre incluida)
QUALITY_TAGS = [
    "high quality", "detailed", "realistic", "professional lighting",
    "sharp focus", "8k resolution", "photorealistic", "skin details"
]

# Negative prompts (anti-censura, anti-defectos)
NEGATIVE_PROMPTS = [
    "clothed", "dressed", "safe", "censored", "blurred",
    "low quality", "bad anatomy", "deformed", "ugly", "disfigured",
    "watermark", "signature", "text", "logo", "copyright",
    "children", "minor", "underage", "loli", "shota",
    "worst quality", "jpeg artifacts", "compression artifacts",
    "extra limbs", "extra fingers", "mutated hands"
]

# Comportamiento por defecto
DEFAULT_NSFW_LEVEL = 'moderate'  # Cambiar a 'explicit' si necesitas más
AUTO_DETECT_GENDER = True  # Usar InsightFace para detectar género/edad
AUTO_DETECT_EXPRESSION = True  # Detectar expresión boca/ojos

# Si no se puede detectar, usar estos fallbacks:
FALLBACK_PROMPTS = {
    'male': 'nude man, bare chest, explicit nsfw, adult content, high quality',
    'female': 'nude woman, bare breasts, explicit nsfw, adult content, high quality',
    'unknown': 'nude person, explicit nsfw, adult content, high quality'
}
