# -*- coding: utf-8 -*-
import torch
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer, util

class SemanticIntentAnalyzer:
    """
    Analizador de intenciones basado en embeddings semánticos.
    Cero hardcoding de palabras clave, utiliza distancias matemáticas en espacio vectorial.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[NLP] Cargando modelo semántico {model_name} en CPU...")
        # Forzar CPU para ahorrar VRAM (ComfyUI ya usa mucha)
        self.device = "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Definir "Anclas Semánticas" (Conceptos puros en Inglés y Español)
        self.anchors = {
            "pose": [
                "changing character pose or body position",
                "sitting, kneeling, standing, lying down, running",
                "repositioning limbs or changing posture",
                "cambiar la pose o posición del cuerpo",
                "sentado, de rodillas, de pie, acostado, corriendo",
                "cambiar la postura o posición de los brazos y piernas"
            ],
            "structural": [
                "adding new people or large objects",
                "changing the background completely",
                "inserting new elements into the scene",
                "removing people or large objects",
                "añadir gente nueva u objetos grandes",
                "cambiar el fondo completamente",
                "insertar elementos nuevos en la escena",
                "quitar personas u objetos"
            ],
            "attribute": [
                "changing color, texture, or lighting",
                "modifying clothes, hair, or eyes",
                "adjusting brightness, filters, or style",
                "subtle facial tweaks",
                "cambiar color, textura o iluminación",
                "modificar ropa, pelo u ojos",
                "ajustar brillo, filtros o estilo",
                "retoques faciales sutiles"
            ]
        }
        
        # Pre-calcular embeddings de las anclas
        self.anchor_embeddings = {}
        for intent, descriptions in self.anchors.items():
            self.anchor_embeddings[intent] = self.model.encode(descriptions, convert_to_tensor=True, device=self.device)
            
        print("[NLP] Analizador semántico listo.")

    def get_magnitude(self, prompt: str) -> float:
        """
        Calcula la magnitud sugerida (0.0 a 1.0) mediante similitud de coseno.
        No usa 'if word in prompt', sino 'que tan cerca esta esta idea de estas otras'.
        """
        if not prompt or not prompt.strip():
            return 0.5
            
        # 1. Vectorizar el prompt del usuario
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True, device=self.device)
        
        # 2. Calcular similitudes con cada ancla
        similarities = {}
        for intent, embeddings in self.anchor_embeddings.items():
            # Similitud máxima con cualquiera de las descripciones del ancla
            cos_sim = util.cos_sim(prompt_embedding, embeddings)
            similarities[intent] = float(torch.max(cos_sim))
            
        # 3. Mapeo matemático de intención a magnitud
        
        # Base mínima
        magnitude = 0.35
        
        # Contribuciones ponderadas (Usamos el máximo impacto para decidir fuerza)
        pose_weight = similarities.get("pose", 0)
        structural_weight = similarities.get("structural", 0)
        attribute_weight = similarities.get("attribute", 0)
        
        # Escalamiento por impacto
        impact_weights = {
            "pose": pose_weight * 0.6,
            "structural": structural_weight * 0.5,
            "attribute": attribute_weight * 0.25
        }
        
        best_intent = max(impact_weights, key=impact_weights.get)
        max_impact = impact_weights[best_intent]
        
        print(f"[NLP] Similitudes: Pose={pose_weight:.2f}, Struct={structural_weight:.2f}, Attr={attribute_weight:.2f}")
        
        if max_impact > 0.15: # Umbral de activación
            magnitude += max_impact
            
        # Normalización final
        return max(0.30, min(0.95, magnitude))

    def detect_target(self, prompt: str) -> str:
        """Determina el objetivo de la máscara mediante semántica"""
        targets = {
            "background": ["the background, environment, scenery, location", "el fondo, escenario, paisaje, entorno"],
            "face": ["the face, facial features, head, eyes, mouth", "la cara, rostro, rasgos faciales, cabeza, ojos, boca"],
            "clothes": ["clothing, dress, shirt, pants, outfit", "la ropa, vestido, camisa, pantalones, vestimenta"],
            "hair": ["hair, hairstyle, bald", "el pelo, cabello, peinado, calvo"],
            "subject": ["the person, character body, the whole figure", "la persona, el cuerpo del personaje, la figura completa"]
        }
        
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True, device=self.device)
        best_target = "subject"
        max_sim = 0
        
        for target, desc in targets.items():
            desc_embedding = self.model.encode(desc, convert_to_tensor=True, device=self.device)
            sim = float(torch.max(util.cos_sim(prompt_embedding, desc_embedding)))
            if sim > max_sim:
                max_sim = sim
                best_target = target
                
        # Ajuste fino: si hay mucha similitud con pose pero el target es face, 
        # verificar si realmente es una pose (palabras de acción) o solo una zona del cuerpo
        pose_sim = float(torch.max(util.cos_sim(prompt_embedding, self.anchor_embeddings["pose"])))
        if pose_sim > 0.6 and best_target == "face":
            # Solo cambiar a subject si la similitud con pose es muy alta
            best_target = "subject"
        elif pose_sim > 0.4 and best_target == "subject":
            # Mantener subject si ya es subject y hay algo de pose
            pass
            
        return best_target if max_sim > 0.25 else "subject"
