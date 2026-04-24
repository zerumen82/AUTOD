#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmartPrompt Generator para FLUX NSFW
Analiza la imagen de referencia (cara) y genera un prompt descriptivo
automático + tags NSFW configurable.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

# Añadir ROOP al path
ROOP_DIR = Path(__file__).parent.parent / "roop"
sys.path.insert(0, str(ROOP_DIR))

try:
    import roop
    from roop.analyser import get_face_analyser
    from roop.face_util import get_face_single
    import cv2
    import numpy as np
    from PIL import Image
except ImportError as e:
    print(f"[ERROR] No se pudo importar roop: {e}")
    print("Asegúrate de estar en el entorno virtual correcto")
    sys.exit(1)


class SmartPromptGenerator:
    """Generador de prompts inteligente basado en análisis de imagen"""
    
    def __init__(self, nsfw_level: str = "moderate"):
        """
        Args:
            nsfw_level: 'mild', 'moderate', 'explicit', 'extreme'
        """
        self.nsfw_level = nsfw_level
        self.face_analyser = None
        
        # Diccionarios de prompts por atributo
        self.gender_prompts = {
            'male': {'mild': ['man', 'male', 'guy'],
                    'moderate': ['nude man', 'male torso', 'bare chest'],
                    'explicit': ['nude male', 'explicit male nudity', 'genitals'],
                    'extreme': ['fully naked man', 'explicit adult content', '18+']},
            'female': {'mild': ['woman', 'female', 'girl'],
                     'moderate': ['nude woman', 'female torso', 'bare breasts'],
                     'explicit': ['nude female', 'explicit female nudity', 'spread legs'],
                     'extreme': ['fully naked woman', 'explicit adult content', '18+', 'porn']},
            'unknown': {'mild': ['person'],
                      'moderate': ['nude person'],
                      'explicit': ['explicit nudity'],
                      'extreme': ['fully naked person']}
        }
        
        self.expression_prompts = {
            'smile': ['smiling', 'happy expression', 'grinning'],
            'open_mouth': ['mouth open', 'tongue out', 'licking lips'],
            'neutral': ['neutral expression', 'calm face'],
            'frown': ['serious expression', 'frowning']
        }
        
        self.age_descriptors = {
            'child': ('child', 'young', 'little'),
            'teen': ('teenager', 'young adult', '18 year old'),
            'young_adult': ('young adult', '20s', 'in their twenties'),
            'adult': ('adult', '30s', 'mature'),
            'older': ('mature', 'older', 'experienced')
        }
    
    def _load_face_analyser(self):
        """Carga el analizador de caras de InsightFace (lazy)"""
        if self.face_analyser is None:
            print("[SmartPrompt] Cargando analizador de caras...")
            self.face_analyser = get_face_analyser()
            print("[SmartPrompt] Analizador listo")
    
    def _detect_gender_age(self, face) -> Tuple[str, str]:
        """
        Detecta género y edad aproximada de la cara.
        Returns: (gender, age_group)
        """
        # InsightFace puede dar género y edad si el modelo está cargado
        # Por defecto, usamos heurísticas si no está disponible
        gender = 'unknown'
        age = 'adult'
        
        try:
            # Intentar obtener datos delFace (depende del modelo buffalo_l)
            if hasattr(face, 'gender') and face.gender is not None:
                gender = 'female' if face.gender == 0 else 'male'  # 0=female, 1=male en algunos modelos
            if hasattr(face, 'age') and face.age is not None:
                age_val = face.age
                if age_val < 13:
                    age = 'child'
                elif age_val < 18:
                    age = 'teen'
                elif age_val < 30:
                    age = 'young_adult'
                elif age_val < 50:
                    age = 'adult'
                else:
                    age = 'older'
        except:
            pass
        
        return gender, age
    
    def _detect_expression(self, image: np.ndarray, face_bbox) -> str:
        """
        Heurística simple para detectar expresión basada en landmarks
        """
        # Por defecto, podemos usar si mouth_open está ya detectado en ProcessMgr
        # Pero aquí haremos una detección simple de aspect ratio de la boca
        return 'neutral'  # Placeholder - se puede expandir
    
    def analyze_image(self, image_path: str) -> Dict:
        """
        Analiza una imagen y extrae atributos.
        
        Returns:
            dict con: gender, age_group, expression, face_count, bbox
        """
        self._load_face_analyser()
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo cargar imagen: {image_path}")
        
        # Convertir a RGB para InsightFace
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Detectar caras
        faces = get_face_single(img_rgb)
        
        if not faces:
            return {
                'error': 'No se detectó ninguna cara',
                'face_count': 0
            }
        
        # Usar la cara más grande (principal)
        main_face = max(faces, key=lambda f: 
            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) if hasattr(f, 'bbox') else 0)
        
        gender, age = self._detect_gender_age(main_face)
        
        return {
            'face_count': len(faces),
            'gender': gender,
            'age_group': age,
            'bbox': main_face.bbox.tolist() if hasattr(main_face, 'bbox') else None,
            'confidence': float(main_face.det_score) if hasattr(main_face, 'det_score') else None
        }
    
    def generate_prompt(self, analysis: Dict, custom_tags: list = None) -> str:
        """
        Genera un prompt completo basado en el análisis.
        
        Args:
            analysis: Resultado de analyze_image()
            custom_tags: Tags adicionales que el usuario quiera añadir
        """
        if 'error' in analysis:
            # Fallback: prompt genérico
            return "nude, explicit nsfw, adult content, high quality"
        
        gender = analysis.get('gender', 'unknown')
        age = analysis.get('age_group', 'adult')
        
        # 1. Construir base
        gender_prompts = self.gender_prompts.get(gender, self.gender_prompts['unknown'])
        age_words = self.age_descriptors.get(age, ('adult',))
        
        # 2. Ensamblar prompt
        parts = []
        
        # Género + edad
        parts.append(f"{gender_prompts[self.nsfw_level][0]}, {age_words[0]}")
        
        # Segunda capa (más descriptiva)
        if self.nsfw_level in ['moderate', 'explicit', 'extreme']:
            parts.append(gender_prompts[self.nsfw_level][1])
        
        # Tercera capa (más explícita)
        if self.nsfw_level in ['explicit', 'extreme']:
            parts.append(gender_prompts[self.nsfw_level][2])
        
        # Cuarta capa (extremo)
        if self.nsfw_level == 'extreme':
            for tag in gender_prompts['extreme'][2:]:
                parts.append(tag)
        
        # Añadir calidad
        parts.extend([
            "high quality", "detailed", "realistic",
            "professional lighting", "sharp focus", "8k resolution"
        ])
        
        # Tags personalizados
        if custom_tags:
            parts.extend(custom_tags)
        
        # Negative prompt implícito (se añade después)
        prompt = ", ".join(parts)
        
        return prompt
    
    def get_negative_prompt(self) -> str:
        """Negative prompt estándar para evitar censura/artefactos"""
        negatives = [
            "clothed", "dressed", "safe", "censored", "blurred",
            "low quality", "bad anatomy", "deformed", "ugly",
            "watermark", "signature", "text", "logo",
            "children", "minor", "underage",
            "worst quality", "jpeg artifacts"
        ]
        return ", ".join(negatives)


def main():
    """CLI para probar el generador"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generador de prompts inteligente para FLUX NSFW")
    parser.add_argument("image", help="Ruta a la imagen de referencia (cara)")
    parser.add_argument("--level", default="moderate", 
                       choices=['mild', 'moderate', 'explicit', 'extreme'],
                       help="Nivel de NSFW")
    parser.add_argument("--tags", nargs='+', default=[],
                       help="Tags adicionales para añadir al prompt")
    parser.add_argument("--output", "-o", default=None,
                       help="Guardar prompt en archivo (json)")
    args = parser.parse_args()
    
    print(f"[SmartPrompt] Analizando: {args.image}")
    print(f"[SmartPrompt] Nivel NSFW: {args.level}")
    
    generator = SmartPromptGenerator(nsfw_level=args.level)
    
    try:
        analysis = generator.analyze_image(args.image)
        print(f"\n[ANÁLISIS]")
        for k, v in analysis.items():
            print(f"  {k}: {v}")
        
        prompt = generator.generate_prompt(analysis, args.tags)
        negative = generator.get_negative_prompt()
        
        print(f"\n[PROMPT GENERADO]")
        print(f"  Positive: {prompt}")
        print(f"\n[NEGATIVE]")
        print(f"  Negative: {negative}")
        
        if args.output:
            out = {
                'prompt': prompt,
                'negative_prompt': negative,
                'analysis': analysis,
                'nsfw_level': args.level
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(out, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Guardado en {args.output}")
            
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
