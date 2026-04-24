#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Analyzer for Prompt Generation
Analiza una imagen y genera un prompt descriptivo automático
para FLUX NSFW
"""

import sys
import os
import importlib.util
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Agregar DLL al path primero para ONNX
DLL_DIR = Path(__file__).parent.parent / "dll"
os.environ["PATH"] = str(DLL_DIR) + ";" + os.environ.get("PATH", "")

# CARGAR MÓDULOS DIRECTAMENTE sin invocar el paquete 'roop' para evitar conflicto con roop/types.py
def _load_roop_module(module_name: str, file_name: str):
    """Carga un módulo de roop sin invocar el paquete completo"""
    module_path = Path(__file__).parent.parent / "roop" / file_name
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        return None

# Intentar cargar los módulos necesarios
_analyser_mod = _load_roop_module("analyser", "analyser.py")
_face_util_mod = _load_roop_module("face_util", "face_util.py")

if _analyser_mod and _face_util_mod:
    # get_face_analyser está en analyser.py
    get_face_analyser = getattr(_analyser_mod, "get_face_analyser", None)
    # get_face_many está en analyser.py
    get_face_many = getattr(_analyser_mod, "get_face_many", None)
    # get_face_single está en analyser.py también
    get_face_single = getattr(_analyser_mod, "get_face_single", None)
    HAS_ROOP = get_face_analyser is not None
    if HAS_ROOP:
        print("[ImageAnalyzer] Módulos de roop cargados correctamente")
    else:
        print("[WARN] No se pudieron cargar las funciones de roop")
else:
    get_face_analyser = None
    get_face_single = None
    get_face_many = None
    HAS_ROOP = False
    print("[WARN] No se pudieron importar módulos de roop. Usando solo heurísticas.")


class ImageAnalyzer:
    """Analiza imágenes para generar prompts descriptivos NATURALES y completos"""

    def __init__(self):
        self.face_analyser = None
        self._load_models()

    def _load_models(self):
        """Carga modelos de análisis (InsightFace + heurísticas)"""
        if not HAS_ROOP:
            return

        try:
            print("[ImageAnalyzer] Cargando modelos de análisis...")
            self.face_analyser = get_face_analyser()
            print("[ImageAnalyzer] Modelos cargados correctamente")
        except Exception as e:
            print(f"[ImageAnalyzer] Warning cargando modelos: {e}")
            self.face_analyser = None

    def analyze(self, image_path: str, nsfw_level: str = 'explicit') -> dict:
        """Analiza una imagen COMPLETAMENTE y devuelve metadatos detallados."""
        result = {
            'faces': [],
            'num_people': 0,
            'scene': 'unknown',
            'lighting': 'unknown',
            'dominant_colors': [],
            'objects': [],
            'image_quality': 'unknown',
            'suggested_prompt': '',
            'nsfw_keywords': []
        }

        # Cargar imagen
        img = cv2.imread(image_path)
        if img is None:
            result['error'] = 'No se pudo cargar la imagen'
            return result

        h, w = img.shape[:2]
        result['image_dimensions'] = (w, h)

        # 1. ANÁLISIS DE CARAS (TODAS, no solo la más grande)
        if self.face_analyser and get_face_many:
            faces = get_face_many(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if faces:
                result['num_people'] = len(faces)
                for i, face in enumerate(faces):
                    face_data = {
                        'index': i,
                        'bbox': face.bbox.tolist() if hasattr(face, 'bbox') else None,
                        'confidence': float(face.det_score) if hasattr(face, 'det_score') else 0.9
                    }

                    # Gender: 0=female, 1=male (InsightFace convention)
                    if hasattr(face, 'gender'):
                        g = face.gender
                        if g == 0:
                            face_data['gender'] = 'female'
                        elif g == 1:
                            face_data['gender'] = 'male'
                        else:
                            face_data['gender'] = 'unknown'
                    else:
                        face_data['gender'] = 'unknown'

                    # Age
                    if hasattr(face, 'age') and face.age > 0:
                        face_data['age'] = int(face.age)
                        age_val = face.age
                        if age_val < 18:
                            face_data['age_group'] = 'teenager'
                        elif age_val < 25:
                            face_data['age_group'] = 'young adult'
                        elif age_val < 35:
                            face_data['age_group'] = 'adult'
                        elif age_val < 50:
                            face_data['age_group'] = 'mature adult'
                        else:
                            face_data['age_group'] = 'senior'
                    else:
                        face_data['age'] = 25
                        face_data['age_group'] = 'adult'

                    # Expresión (básica)
                    face_data['expression'] = self._detect_expression(face)

                    # Posición en la imagen
                    if face_data['bbox']:
                        cx = (face_data['bbox'][0] + face_data['bbox'][2]) / 2
                        cy = (face_data['bbox'][1] + face_data['bbox'][3]) / 2
                        if cy < h/3:
                            face_data['position'] = 'top'
                        elif cy > 2*h/3:
                            face_data['position'] = 'bottom'
                        else:
                            face_data['position'] = 'center'

                    result['faces'].append(face_data)

        # 2. ANÁLISIS DE ESCENA
        result['scene'] = self._detect_scene(img)
        result['lighting'] = self._detect_lighting(img)
        result['dominant_colors'] = self._get_dominant_colors(img)
        result['image_quality'] = self._detect_quality(img)

        # 3. GENERAR DESCRIPCIÓN NATURAL Y COMPLETA
        result['suggested_prompt'] = self._generate_natural_description(result)

        # 4. Keywords NSFW por nivel
        result['nsfw_keywords'] = self._get_nsfw_keywords('explicit')

        return result

    def _detect_expression(self, face) -> str:
        """Detecta expresión facial de forma básica"""
        try:
            if hasattr(face, 'landmark_2d_106'):
                # Calcular ratio de apertura de boca
                landmarks = face.landmark_2d_106
                # Puntos de la boca (60-67)
                mouth_points = landmarks[60:68]
                if len(mouth_points) >= 8:
                    top_lip = np.mean(mouth_points[0:3, 1])
                    bottom_lip = np.mean(mouth_points[4:7, 1])
                    mouth_ratio = bottom_lip - top_lip
                    if mouth_ratio > 5:
                        return 'mouth open'
                    elif mouth_ratio > 2:
                        return 'smiling'
                    else:
                        return 'neutral expression'
            return 'neutral expression'
        except:
            return 'neutral expression'

    def _detect_scene(self, img) -> str:
        """Detecta si es interior o exterior"""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat_mean = np.mean(hsv[:,:,1])
        val_mean = np.mean(hsv[:,:,2])

        # Calcular varianza de luminosidad
        val_std = np.std(hsv[:,:,2])

        if val_std > 40 and sat_mean > 80:
            return 'outdoor'
        elif val_std < 30 and sat_mean < 70:
            return 'indoor'
        else:
            return 'mixed lighting scene'

    def _detect_lighting(self, img) -> str:
        """Detecta condiciones de iluminación"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)

        if brightness > 180:
            return 'bright daylight'
        elif brightness > 120:
            return 'natural daylight'
        elif brightness > 80:
            return 'soft lighting'
        else:
            return 'dark low light'

    def _get_dominant_colors(self, img, k=3) -> list:
        """Obtiene colores dominantes usando k-means"""
        try:
            pixels = np.float32(img.reshape(-1, 3))
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

            colors = []
            for center in centers:
                b, g, r = center.astype(int)
                colors.append(f"rgb({r}, {g}, {b})")
            return colors
        except:
            return ['unknown']

    def _detect_quality(self, img) -> str:
        """Evalúa calidad de imagen"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

        if laplacian > 100:
            return 'high quality sharp'
        elif laplacian > 50:
            return 'medium quality'
        else:
            return 'low quality blurry'

    def _generate_natural_description(self, analysis: dict) -> str:
        """Genera una DESCRIPCIÓN NATURAL para FLUX, concisa y descriptiva"""
        parts = []

        # 1. Número de personas
        n = analysis['num_people']
        if n == 0:
            parts.append('person')
        elif n == 1:
            parts.append('a woman' if any(f.get('gender') == 'female' for f in analysis.get('faces', [])) else 'a man')
        elif n == 2:
            parts.append('two people, a man and a woman')
        elif n <= 5:
            genders = [f.get('gender', 'unknown') for f in analysis.get('faces', [])]
            has_male = 'male' in genders
            has_female = 'female' in genders
            group_desc = []
            if has_male:
                group_desc.append('men')
            if has_female:
                group_desc.append('women')
            parts.append(f'{n} people, {" and ".join(group_desc)}')
        else:
            parts.append(f'a group of {n} people')

        # 2. Solo la primera persona como referencia (para mantener identidad)
        if n > 0 and analysis['faces']:
            first_face = analysis['faces'][0]
            desc = []
            if first_face.get('gender') != 'unknown':
                desc.append(first_face['gender'])
            if first_face.get('age_group') != 'unknown':
                desc.append(first_face['age_group'])
            if first_face.get('expression') and first_face['expression'] != 'neutral expression':
                desc.append(first_face['expression'])
            if desc:
                parts.append(', '.join(desc) + ' person')

        # 3. Escena e iluminación
        if analysis['scene'] != 'unknown':
            parts.append(analysis['scene'])
        parts.append(analysis['lighting'])
        parts.append(analysis['image_quality'])

        # 4. Calidad general
        parts.append('photorealistic')
        parts.append('detailed')
        parts.append('high resolution')

        return ', '.join(parts)

    def _get_nsfw_keywords(self, level: str = 'moderate') -> list:
        """Retorna lista de palabras NSFW según nivel"""
        nsfw_map = {
            'mild': ['nsfw', 'suggestive', 'lingerie', 'bikini'],
            'moderate': ['nude', 'naked', 'bare breasts', 'topless', 'explicit'],
            'explicit': ['fully nude', 'genitals', 'spread legs', 'adult content', '18+'],
            'extreme': ['porn', 'hardcore', 'sexual intercourse', 'XXX']
        }
        return nsfw_map.get(level, nsfw_map['moderate'])

    def generate_full_prompt(self, image_path: str, nsfw_level: str = 'explicit', extra_tags: list = None) -> dict:
        """Genera prompt completo (positivo + negativo)."""
        analysis = self.analyze(image_path)
        base_prompt = analysis['suggested_prompt']

        # Añadir tags NSFW
        nsfw_tags = self._get_nsfw_keywords(nsfw_level)

        # Ensamblar prompt positivo
        positive_parts = [base_prompt] + nsfw_tags
        if extra_tags:
            positive_parts.extend(extra_tags)

        positive_prompt = ", ".join(positive_parts)

        # Negative prompt estándar
        negative_prompt = (
            "clothed, dressed, safe, censored, blurry, low quality, "
            "bad anatomy, deformed, ugly, watermark, text, logo, "
            "children, minor, underage, worst quality, jpeg artifacts, "
            "extra limbs, extra fingers, mutated hands, nipples censored"
        )

        return {
            'positive': positive_prompt,
            'negative': negative_prompt,
            'analysis': analysis
        }


def analyze_image_for_prompt(image_path: str, nsfw_level: str = 'explicit') -> dict:
    """Función de conveniencia para usar desde la UI."""
    analyzer = ImageAnalyzer()
    return analyzer.generate_full_prompt(image_path, nsfw_level)


if __name__ == "__main__":
    # Test desde línea de comandos
    import argparse

    parser = argparse.ArgumentParser(description="Analiza imagen y genera prompt")
    parser.add_argument("image", help="Ruta a la imagen")
    parser.add_argument("--level", default="explicit",
                        choices=['mild', 'moderate', 'explicit', 'extreme'],
                        help="Nivel NSFW")
    parser.add_argument("--json", action="store_true", help="Salida JSON")
    args = parser.parse_args()

    result = analyze_image_for_prompt(args.image, args.level)

    if args.json:
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print("=" * 60)
        print("PROMPT GENERADO:")
        print("  Positivo:", result['positive'])
        print("\n  Negativo:", result['negative'])
        print("\nAnálisis:")
        for face in result['analysis']['faces']:
            print(f"  - Cara: género={face.get('gender')}, edad={face.get('age')}, expresión={face.get('expression')}")
        print("=" * 60)