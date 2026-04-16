#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImgEditor Manager - Edición de imágenes con ComfyUI

El usuario escribe un prompt natural y el sistema:
1. Mejora el prompt automáticamente
2. Detecta el área a modificar (desde archivo configurable)
3. Genera con ComfyUI usando inpaint
"""

import os
import sys
import tempfile
import time
import json
import torch
import gc
from typing import Optional, Tuple, Dict
from PIL import Image
import numpy as np
import cv2

from roop.comfy_client import get_comfyui_url
from roop.img_editor.icedit_comfy_client import get_icedit_comfy_client
from roop.img_editor.qwen_edit_comfy_client import get_qwen_edit_comfy_client
from roop.img_editor.clothing_segmenter import get_clothing_segmenter, is_clipseg_available
from roop.img_editor.controlnet_utils import get_controlnet_utils

# ... (resto del código igual)

# Ruta al archivo de áreas configurables
PROMPT_AREAS_FILE = os.path.join(os.path.dirname(__file__), "prompt_areas.json")

def load_prompt_areas() -> Dict:
    """Carga las áreas desde el archivo JSON configurable"""
    try:
        if os.path.exists(PROMPT_AREAS_FILE):
            with open(PROMPT_AREAS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        print(f"[PromptAreas] Error cargando {PROMPT_AREAS_FILE}: {e}")
    
    # Fallback por defecto - IMAGEN COMPLETA
    return {
        "cuerpo": {
            "keywords": ["cuerpo", "body", "persona", "person"],
            "coords": {"y_start": 0.0, "y_end": 1.0, "x_start": 0.0, "x_end": 1.0},
            "add_prompt": ""
        }
    }

class ImgEditorManager:
    """Gestiona la edición de imágenes con ComfyUI y otros motores"""

    def __init__(self):
        self.client = None
        self.face_swapper = None
        self.face_analyzer = None
        self.flux_edit_client = None # FLUX Image Edit fp8
        self.qwen_edit_client = None # Qwen Image Edit fp8
        self.zimage_edit_client = None # Z-Image Turbo GGUF
        self.hart_edit_client = None # HART (Hybrid Autoregressive Transformer)
        self.omnigen2_client = None # OmniGen2 GGUF
        self.icedit_client = None # ICEdit (no funciona)
        self.controlnet_utils = None

    def _init_flux_client(self):
        """Inicializa el cliente FLUX si no está cargado"""
        if self.flux_client is not None:
            return True
        
        try:
            from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
            self.flux_client = get_flux_edit_comfy_client()
            
            # Intentar cargar el cliente
            ok, msg = self.flux_client.load()
            if ok:
                print(f"[ImgEditor] ✅ FLUX client inicializado: {msg}")
                return True
            else:
                print(f"[ImgEditor] ⚠️ FLUX client no pudo cargar: {msg}")
                self.flux_client = None
                return False
        except Exception as e:
            print(f"[ImgEditor] ❌ Error inicializando FLUX client: {e}")
            self.flux_client = None
            return False

    def _cleanup_temp_and_vram(self):
        """Limpia archivos temporales y VRAM después de generar"""
        import gc
        import tempfile
        import os
        
        # 1. Limpiar archivos temporales
        temp_dir = tempfile.gettempdir()
        temp_prefixes = ["sd_editor_", "flux_", "img_editor_"]
        
        try:
            for f in os.listdir(temp_dir):
                if any(f.startswith(prefix) for prefix in temp_prefixes):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                        print(f"[ImgEditor] Temp eliminado: {f}")
                    except:
                        pass
        except Exception as e:
            print(f"[ImgEditor] Error limpiando temps: {e}")
        
        # 2. Limpiar VRAM CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[ImgEditor] VRAM después de limpiar: {allocated:.2f} GB")
        
        # 3. Forzar garbage collector
        gc.collect()
        
        print("[ImgEditor] ✅ Limpieza completada")

    def rewrite_prompt(self, prompt: str) -> str:
        """Mejora el prompt usando el reescritor interno"""
        try:
            from roop.img_editor.prompt_rewriter import get_prompt_rewriter
            rewriter = get_prompt_rewriter()
            return rewriter.rewrite(prompt)
        except Exception as e:
            print(f"[ImgEditor] Error reescribiendo prompt: {e}")
            return prompt + ", high quality, masterpiece, highly detailed"


    def analyze_prompt(self, prompt: str) -> Dict[str, bool]:
        """Analiza el prompt - SIEMPRE usa inpaint + ipadapter"""
        return {
            "use_openpose": False,
            "use_tile": False,
            "use_inpaint": True,
            "use_ipadapter": True,
            "auto_enhance": False,
            "auto_upscale": False,
            "needs_rewriting": True
        }

    def rewrite_prompt(self, prompt: str, analysis: Dict[str, bool] = None) -> str:
        """
        Mejora el prompt automáticamente usando análisis SEMÁNTICO DINÁMICO AVANZADO.
        SISTEMA VIVO 2.0: Inferencia contextual profunda, patrones compuestos, optimización automática.
        """
        prompt_lower = prompt.lower()
        enhanced_prompt = prompt
        all_coords = []
        detected_elements = []
        detected_actions = []
        detected_contexts = []
        detected_intensities = {"low": 0, "medium": 0, "high": 0}
        
        # Cargar áreas configurables (solo como referencia base)
        prompt_areas = load_prompt_areas()

        # ============================================
        # 1. ANÁLISIS SEMÁNTICO MEJORADO
        # ============================================
        stop_words = {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al", "en", "con", "para", "por", "que", "y", "o", "pero", "si", "no", "más", "muy", "tan", "tanto", "haz", "hacer", "que", "sea", "se", "lo", "le", "la", "me", "te", "nos", "os", "quiero", "quieras", "puedes", "podrías", "porfavor", "favor", "hazme", "hacerme"}
        
        meaningful_words = [w.strip('.,!?;:()[]""\'') for w in prompt_lower.split() 
                          if len(w) > 2 and w not in stop_words]
        
        # Detectar intensidad de la edición
        intensity_words = ["poco", "ligero", "suave", "mínimo", "bastante", "mucho", "fuerte", "drástico", "total", "completo", "parcial", "completo"]
        for word in meaningful_words:
            if word in ["poco", "ligero", "suave", "mínimo", "parcial"]:
                detected_intensities["low"] += 1
            elif word in ["bastante", "mucho", "fuerte"]:
                detected_intensities["medium"] += 1
            elif word in ["drástico", "total", "completo"]:
                detected_intensities["high"] += 1
        
        # Determinar intensidad predominante
        intensity = "medium"
        if detected_intensities["low"] > detected_intensities["medium"] and detected_intensities["low"] > detected_intensities["high"]:
            intensity = "low"
        elif detected_intensities["high"] > detected_intensities["medium"]:
            intensity = "high"
        
        print(f"[SemanticAnalyzer] Palabras clave: {meaningful_words}")
        print(f"[SemanticAnalyzer] Intensidad detectada: {intensity}")

        # ============================================
        # 2. INFERENCIA DE ACCIONES MEJORADA
        # ============================================
        # Patrones compuestos y frases
        action_patterns = {
            "eliminar": {
                "direct": ["elimina", "eliminar", "quita", "quitar", "sin", "fuera", "saca", "borra", "remove", "delete"],
                "implicit": ["desnuda", "desnudo", "desvestir", "descalza", "descalzo"],
                "phrases": ["sin ropa", "sin zapatos", "sin pantalones", "quitar la ropa", "eliminar ropa"]
            },
            "cambiar": {
                "direct": ["cambia", "cambiar", "transforma", "transformar", "convierte", "diferente", "otra", "otro", "nuevo", "nueva", "actual", "moderno"],
                "implicit": ["mejora", "actualiza", "renueva", "varía", "altera"],
                "phrases": ["cambia por", "transforma en", "convierte en", "haz que", "ponle"]
            },
            "añadir": {
                "direct": ["añade", "añadir", "agrega", "coloca", "pon", "sube", "monta", "incluye", "aparece", "crea", "genera"],
                "implicit": ["inserta", "incorpora", "agregale", "ponle"],
                "phrases": ["añade un", "agrega una", "pon un", "coloca una"]
            },
            "mejorar": {
                "direct": ["mejora", "mejorar", "enhance", "upgrade", "quality", "calidad", "nitidez", "resolución"],
                "implicit": ["optimiza", "perfecciona", "refina", "pulir"],
                "phrases": ["mejora la", "haz mejor", "más calidad", "más nítido"]
            }
        }
        
        for action_name, patterns in action_patterns.items():
            # Check directo
            if any(p in prompt_lower for p in patterns["direct"]):
                detected_actions.append(action_name)
                print(f"[SemanticAnalyzer] → Acción detectada: {action_name} (directo)")
            # Check implícito
            elif any(p in prompt_lower for p in patterns["implicit"]):
                detected_actions.append(action_name)
                print(f"[SemanticAnalyzer] → Acción detectada: {action_name} (implícito)")
            # Check frases
            elif any(p in prompt_lower for p in patterns["phrases"]):
                detected_actions.append(action_name)
                print(f"[SemanticAnalyzer] → Acción detectada: {action_name} (frase)")

        # ============================================
        # 3. GRUPOS SEMÁNTICOS EXPANDIDOS
        # ============================================
        semantic_groups = {
            "cuerpo_inferior": {
                # CORREGIDO: y_start 0.35→0.50 (desde CINTURA, no desde pecho)
                "roots": ["piern", "pantalon", "zapato", "pie", "falda", "short", "calcetin", "media", "tanga", "braga"],
                "related": ["abajo", "inferior", "bajo", "bottom", "lower", "piernas", "pies"],
                "coords": {"y_start": 0.50, "y_end": 1.0},
                "priority": 1
            },
            "cuerpo_superior": {
                "roots": ["torso", "pecho", "camisa", "camiseta", "blusa", "chaqueta", "abrigo", "ropa", "vestido", "top"],
                "related": ["arriba", "superior", "alto", "top", "upper", "cuerpo"],
                "coords": {"y_start": 0.20, "y_end": 0.55},
                "priority": 1
            },
            "cuerpo_completo": {
                "roots": ["cuerpo", "completo", "entero", "todo", "desnuda", "desnudo", "naked", "nude"],
                "related": ["body", "full body", "sin ropa", "desvestir"],
                "coords": {"y_start": 0.0, "y_end": 1.0},
                "priority": 2
            },
            "cabeza": {
                "roots": ["cara", "rostro", "pelo", "cabello", "cabeza", "ojos", "boca", "expresion", "faz"],
                "related": ["head", "face", "hair", "facial", "capilar"],
                "coords": {"y_start": 0.0, "y_end": 0.30},
                "priority": 1
            },
            "cuello_hombros": {
                "roots": ["cuello", "hombro", "hombros", "nuca", "garganta"],
                "related": ["neck", "shoulder", "shoulders"],
                "coords": {"y_start": 0.15, "y_end": 0.30},
                "priority": 0.5
            },
            "brazos_manos": {
                "roots": ["brazo", "brazos", "mano", "manos", "dedo", "dedos", "antebrazo", "codo"],
                "related": ["arm", "arms", "hand", "hands", "finger"],
                "coords": {"y_start": 0.25, "y_end": 0.75},
                "priority": 0.5
            },
            "cadera_cintura": {
                "roots": ["cadera", "cintura", "cinturon", "talle", "abdomen", "panza", "tripa"],
                "related": ["hip", "hips", "waist", "belly", "abdomen"],
                "coords": {"y_start": 0.45, "y_end": 0.65},
                "priority": 0.5
            },
            "exterior_natural": {
                "roots": ["playa", "mar", "montaña", "bosque", "naturaleza", "campo", "jardin", "parque", "rio", "lago", "cielo", "nube", "sol", "atardecer"],
                "related": ["outside", "outdoor", "nature", "exterior", "natural", "landscape"],
                "coords": {"y_start": 0.0, "y_end": 0.9},
                "priority": 1
            },
            "exterior_urbano": {
                "roots": ["ciudad", "calle", "edificio", "urbano", "town", "rascacielos", "plaza", "avenida"],
                "related": ["urban", "city", "street", "building", "downtown"],
                "coords": {"y_start": 0.0, "y_end": 0.9},
                "priority": 1
            },
            "interior": {
                "roots": ["casa", "habitacion", "salon", "cocina", "interior", "dentro", "room", "home", "baño", "oficina", "estudio"],
                "related": ["inside", "indoor", "interior", "indoor"],
                "coords": {"y_start": 0.0, "y_end": 0.9},
                "priority": 1
            },
            "animal_grande": {
                "roots": ["caballo", "vaca", "toro", "elefante", "jirafa", "leon", "tigre", "oso"],
                "related": ["montar", "ride", "animal grande", "large animal"],
                "coords": {"y_start": 0.0, "y_end": 1.0},
                "priority": 2
            },
            "animal_pequeno": {
                "roots": ["perro", "gato", "pajaro", "raton", "conejo", "hamster"],
                "related": ["pet", "mascota", "animal pequeño", "small animal"],
                "coords": {"y_start": 0.0, "y_end": 1.0},
                "priority": 1
            },
            "accesorio_cabeza": {
                "roots": ["gafas", "sombrero", "gorra", "pendientes", "collar", "diadema", "tocado"],
                "related": ["accesorio", "accessory", "complemento", "headwear"],
                "coords": {"y_start": 0.0, "y_end": 0.3},
                "priority": 0.3
            },
            "accesorio_cuerpo": {
                "roots": ["reloj", "pulsera", "anillo", "cinturon", "bolso", "mochila"],
                "related": ["accesorio", "accessory", "jewelry", "bag"],
                "coords": {"y_start": 0.3, "y_end": 0.8},
                "priority": 0.3
            },
            "estilo_moderno": {
                "roots": ["moderno", "actual", "contemporaneo", "trendy", "fashion", "vanguardia", "innovador"],
                "related": ["nuevo", "recent", "current", "up-to-date", "modern"],
                "coords": {"y_start": 0.0, "y_end": 1.0},
                "priority": 1
            },
            "estilo_vintage": {
                "roots": ["vintage", "retro", "clasico", "antiguo", "old", "nostalgico", "decada"],
                "related": ["viejo", "ancient", "classic", "old school"],
                "coords": {"y_start": 0.0, "y_end": 1.0},
                "priority": 1
            },
            "estilo_artistico": {
                "roots": ["artistico", "arte", "pintura", "cuadro", "oleo", "acuarela", "dibujo"],
                "related": ["art", "artistic", "painting", "drawing"],
                "coords": {"y_start": 0.0, "y_end": 1.0},
                "priority": 1
            },
            "fondo_borroso": {
                "roots": ["borroso", "difuminado", "desenfocado", "blur", "bokeh"],
                "related": ["blurry", "blurred", "out of focus", "depth"],
                "coords": {"y_start": 0.0, "y_end": 0.9},
                "priority": 0.5
            }
        }
        
        # Buscar coincidencias en grupos semánticos con scoring
        for group_name, group_data in semantic_groups.items():
            roots = group_data.get("roots", [])
            related = group_data.get("related", [])
            
            # Coincidencia por raíz con scoring
            root_matches = sum(1 for word in meaningful_words if any(root in word for root in roots))
            related_matches = sum(1 for word in meaningful_words if word in related)
            total_matches = root_matches + related_matches
            
            # Aplicar peso por prioridad
            priority = group_data.get("priority", 1.0)
            weighted_matches = total_matches * priority
            
            if weighted_matches >= 0.5:  # Umbral reducido para mejor detección
                detected_elements.append({
                    "name": group_name,
                    "type": "semantic_group",
                    "coords": group_data["coords"].copy(),
                    "matches": total_matches,
                    "priority": priority
                })
                print(f"[SemanticAnalyzer] → Grupo detectado: {group_name} (matches: {total_matches}, priority: {priority})")

        # ============================================
        # 4. DETECCIÓN POR ÁREAS DEL JSON (fallback)
        # ============================================
        for area_name, area_data in prompt_areas.items():
            keywords = area_data.get("keywords", [])
            match_count = sum(1 for kw in keywords if kw in prompt_lower)
            
            if match_count > 0 and not any(e["name"] == area_name for e in detected_elements):
                coords = area_data.get("coords", {"y_start": 0.0, "y_end": 1.0})
                detected_elements.append({
                    "name": area_name,
                    "type": "json_area",
                    "coords": coords,
                    "matches": match_count,
                    "priority": 1.0
                })
                print(f"[SemanticAnalyzer] → Área JSON: {area_name} (matches: {match_count})")

        # ============================================
        # 5. INFERENCIA CONTEXTUAL PROFUNDA
        # ============================================
        # Reglas de inferencia compuestas
        
        # Desnudez - detectar por múltiples señales
        body_areas = ["cuerpo_inferior", "cuerpo_superior", "cuerpo_completo"]
        has_body = any(g in [e["name"] for e in detected_elements] for g in body_areas)
        has_eliminate = "eliminar" in detected_actions
        has_nude_words = any(w in prompt_lower for w in ["desnuda", "desnudo", "naked", "nude", "sin ropa"])
        
        if (has_body and has_eliminate) or has_nude_words:
            detected_contexts.append("desnudez")
            print("[SemanticAnalyzer] → Contexto inferido: DESNUDEZ")
        
        # Cambio de fondo
        background_areas = ["exterior_natural", "exterior_urbano", "interior", "fondo_borroso"]
        has_background = any(g in [e["name"] for e in detected_elements] for g in background_areas)
        has_change = "cambiar" in detected_actions
        has_background_words = any(w in prompt_lower for w in ["fondo", "background", "escenario", "entorno"])
        
        if (has_background and has_change) or (has_background_words and has_change):
            detected_contexts.append("cambio_fondo")
            print("[SemanticAnalyzer] → Contexto inferido: CAMBIO_DE_FONDO")
        
        # Integración de animal
        animal_areas = ["animal_grande", "animal_pequeno"]
        has_animal = any(g in [e["name"] for e in detected_elements] for g in animal_areas)
        has_animal_words = any(w in prompt_lower for w in ["caballo", "animal", "montar", "subir"])
        
        if has_animal or has_animal_words:
            detected_contexts.append("animal_integration")
            print("[SemanticAnalyzer] → Contexto inferido: ANIMAL_INTEGRATION")
        
        # Cambio de estilo
        style_areas = ["estilo_moderno", "estilo_vintage", "estilo_artistico"]
        has_style = any(g in [e["name"] for e in detected_elements] for g in style_areas)
        has_clothing = "ropa" in prompt_lower or "outfit" in prompt_lower or "vestido" in prompt_lower
        
        if has_style or (has_clothing and has_change):
            detected_contexts.append("cambio_estilo")
            print("[SemanticAnalyzer] → Contexto inferido: CAMBIO_ESTILO")
        
        # Mejora de calidad
        if "mejorar" in detected_actions or any(w in prompt_lower for w in ["calidad", "quality", "nitidez", "resolucion"]):
            detected_contexts.append("mejora_calidad")
            print("[SemanticAnalyzer] → Contexto inferido: MEJORA_CALIDAD")
        
        # Borrado de fondo
        if "fondo_borroso" in [e["name"] for e in detected_elements] or any(w in prompt_lower for w in ["borroso", "blur", "bokeh", "desenfocado"]):
            detected_contexts.append("fondo_borroso")
            print("[SemanticAnalyzer] → Contexto inferido: FONDO_BORROSO")

        # ============================================
        # 6. COMBINACIONES INTELIGENTES AVANZADAS
        # ============================================
        # Expansión según intensidad
        expansion_factor = 0.10 if intensity == "low" else (0.15 if intensity == "medium" else 0.20)

        # Si es desnudez → FORZAR coordenadas correctas (cintura = 0.50)
        if "desnudez" in detected_contexts:
            for elem in detected_elements:
                if elem["name"] in ["cuerpo_inferior", "cuerpo_completo"]:
                    # FORZAR y_start = 0.50 (cintura) ANTES de expansión
                    elem["coords"]["y_start"] = 0.50
                    elem["coords"]["y_end"] = 1.0
                    elem["coords"]["x_start"] = 0.0
                    elem["coords"]["x_end"] = 1.0
                    print(f"[SemanticAnalyzer] → cuerpo_inferior FORZADO a y_start=0.50 (cintura)")
                elif elem["name"] == "pies":
                    elem["coords"]["y_start"] = 0.70  # Solo pies
                    elem["coords"]["y_end"] = 1.0
                    print(f"[SemanticAnalyzer] → pies ajustado a y_start=0.70")
        
        # Si hay animal → usar imagen completa
        if "animal_integration" in detected_contexts:
            if not any(e["name"] == "imagen_completa" for e in detected_elements):
                detected_elements.append({
                    "name": "imagen_completa",
                    "type": "inferred",
                    "coords": {"y_start": 0.0, "y_end": 1.0, "x_start": 0.0, "x_end": 1.0},
                    "matches": 1,
                    "priority": 3
                })
            print("[SemanticAnalyzer] → Imagen completa añadida para animal")
        
        # Si es cambio de fondo → expandir verticalmente
        if "cambio_fondo" in detected_contexts:
            for elem in detected_elements:
                if "exterior" in elem["name"] or "interior" in elem["name"] or "fondo" in elem["name"]:
                    elem["coords"]["y_end"] = 0.95
            print("[SemanticAnalyzer] → Expansión para fondo aplicada")
        
        # Si es estilo → cubrir todo el cuerpo
        if "cambio_estilo" in detected_contexts:
            if not any(e["name"] in ["cuerpo_superior", "cuerpo_completo"] for e in detected_elements):
                detected_elements.append({
                    "name": "cuerpo_completo",
                    "type": "inferred",
                    "coords": {"y_start": 0.0, "y_end": 1.0, "x_start": 0.0, "x_end": 1.0},
                    "matches": 1,
                    "priority": 2
                })
            print("[SemanticAnalyzer] → Cuerpo completo añadido para estilo")

        # ============================================
        # 7. PREPARAR COORDENADAS
        # ============================================
        all_coords = [elem["coords"] for elem in detected_elements]
        
        if not all_coords:
            all_coords = [{"y_start": 0.0, "y_end": 1.0, "x_start": 0.0, "x_end": 1.0}]
            print("[SemanticAnalyzer] → Fallback: imagen completa")

        combined_coords = {
            "y_start": min(c.get("y_start", 0) for c in all_coords),
            "y_end": max(c.get("y_end", 1) for c in all_coords),
            "x_start": min(c.get("x_start", 0) for c in all_coords),
            "x_end": max(c.get("x_end", 1) for c in all_coords)
        }

        print(f"\n[SemanticAnalyzer] === RESUMEN ===")
        print(f"Coords finales: {combined_coords}")
        print(f"Elementos: {[e['name'] for e in detected_elements]}")
        print(f"Acciones: {detected_actions}")
        print(f"Contextos: {detected_contexts}")
        print(f"Intensidad: {intensity}")
        print(f"========================\n")

        # ============================================
        # 8. GENERAR PROMPT MEJORADO OPTIMIZADO
        # ============================================
        quality_terms = ["masterpiece", "best quality", "ultra high quality", "detailed", 
                        "realistic", "professional lighting", "sharp focus", "8k", "UHD"]
        for term in quality_terms:
            if term not in enhanced_prompt.lower():
                enhanced_prompt = f"{term}, {enhanced_prompt}"

        context_additions = []
        
        if "desnudez" in detected_contexts:
            strength = 1.6 if intensity == "low" else (1.8 if intensity == "medium" else 2.0)
            context_additions.extend([
                f"(no clothing:{strength})", f"(no pants:{strength})", f"(no shoes:{strength})", f"(no underwear:{strength-0.1})",
                f"(no shirt:{strength-0.1})", f"(no bra:{strength-0.1})", f"(no fabric:{strength-0.2})",
                "completely naked", "fully nude", "bare skin", "natural body",
                "detailed skin texture", "anatomically correct", "smooth skin transition",
                "seamless skin", "realistic skin tone", "proper anatomy", "correct proportions",
                "natural pose", "soft skin shading", "realistic body curves"
            ])
        
        if "cambio_fondo" in detected_contexts:
            context_additions.extend([
                "seamless background integration", "matching lighting", "proper atmosphere",
                "coherent scene", "environmental consistency", "proper depth of field",
                "matching color palette", "professional composite", "realistic background",
                "background blur", "depth blur", "bokeh effect", "background separation"
            ])
        
        if "animal_integration" in detected_contexts:
            context_additions.extend([
                "realistic animal integration", "natural pose", "proper lighting match",
                "seamless composite", "anatomically correct animal", "proper scale",
                "realistic animal details", "natural interaction", "proper positioning",
                "animal fur details", "realistic animal texture", "animal anatomy"
            ])
        
        if "cambio_estilo" in detected_contexts:
            if "estilo_moderno" in [e["name"] for e in detected_elements]:
                context_additions.extend([
                    "modern style", "contemporary", "up-to-date", "trendy",
                    "professional styling", "current fashion", "latest trends",
                    "sophisticated look", "elegant design", "modern aesthetic",
                    "fashion photography", "editorial style", "high fashion"
                ])
            elif "estilo_vintage" in [e["name"] for e in detected_elements]:
                context_additions.extend([
                    "vintage style", "retro aesthetic", "classic look", "timeless",
                    "vintage fashion", "retro photography", "classic photography",
                    "film photography", "analog look", "nostalgic atmosphere"
                ])
            else:
                context_additions.extend([
                    "stylish", "fashion", "elegant", "sophisticated",
                    "designer clothing", "fashion forward", "trendy outfit"
                ])
        
        if "mejora_calidad" in detected_contexts:
            quality_boost = 1.2 if intensity == "low" else (1.5 if intensity == "medium" else 1.8)
            context_additions.extend([
                f"ultra sharp ({quality_boost})", "crystal clear", "perfectly sharp", "noise free", "clean image",
                "professional retouching", "flawless", "perfect quality",
                "high dynamic range", "perfect exposure", "color corrected",
                "studio quality", "premium quality", "exceptional quality",
                "HDR", "tone mapped", "color graded"
            ])
        
        if "fondo_borroso" in detected_contexts:
            blur_strength = 0.3 if intensity == "low" else (0.5 if intensity == "medium" else 0.7)
            context_additions.extend([
                f"depth of field ({blur_strength})", "bokeh", "blurred background",
                "shallow focus", "selective focus", "background blur",
                "subject focus", "portrait mode"
            ])
        
        if context_additions:
            enhanced_prompt += ", " + ", ".join(context_additions)

        # ============================================
        # 9. GENERAR NEGATIVE PROMPT DINÁMICO
        # ============================================
        negative_terms = [
            "low quality", "worst quality", "bad quality", "jpeg artifacts",
            "blurry", "out of focus", "poorly drawn", "bad anatomy", "deformed",
            "disfigured", "mutated", "extra limbs", "missing limbs", "malformed",
            "cartoon", "3d render", "drawing", "painting", "anime", "cgi", "plastic"
        ]
        
        if "desnudez" in detected_contexts:
            negative_terms.extend([
                "clothing", "pants", "shoes", "underwear", "bra", "shirt",
                "fabric", "textile", "garment", "dress", "skirt", "shorts",
                "socks", "stockings", "belt", "accessories on body", "jewelry"
            ])
        
        if "cambio_fondo" in detected_contexts:
            negative_terms.extend([
                "inconsistent lighting", "wrong perspective", "floating objects",
                "seam visible", "obvious composite", "fake background", "amateur",
                "poor composite", "bad blending", "harsh edges"
            ])
        
        if "animal_integration" in detected_contexts:
            negative_terms.extend([
                "mutated animal", "extra legs", "missing legs", "wrong anatomy",
                "floating animal", "wrong scale", "unnatural pose", "fake animal",
                "cartoon animal", "toy animal"
            ])
        
        if intensity == "high":
            negative_terms.extend([
                "low resolution", "pixelated", "grainy", "noisy", "compressed",
                "artifact", "distorted", "warped"
            ])
        
        import roop.globals as rg
        rg._sd_editor_negative_prompt = ", ".join(negative_terms)
        rg._sd_editor_mask_coords = combined_coords
        rg._sd_editor_intensity = intensity  # Guardar intensidad para generate_intelligent

        print(f"[SemanticAnalyzer] Original: {prompt}")
        print(f"[SemanticAnalyzer] Mejorado: {enhanced_prompt[:250]}...")
        print(f"[SemanticAnalyzer] Negative: {rg._sd_editor_negative_prompt[:150]}...")

        return enhanced_prompt
    
    def is_comfy_available(self):
        """Verifica si ComfyUI esta disponible"""
        from roop.comfy_client import check_comfy_available
        return check_comfy_available()
        
    def _get_client(self):
        """Inicializa el cliente de ComfyUI"""
        if self.client is None:
            from roop.comfy_client import ComfyClient
            self.client = ComfyClient()
        return self.client
    
    def _init_face_swap(self):
        """
        Inicializa el face analyzer y face swapper.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        # Si ya está inicializado, retornar True
        if self.face_analyzer is not None and self.face_swapper is not None:
            return True
        
        try:
            import torch
            use_cuda = torch.cuda.is_available()
            
            # 1. Inicializar Face Analyzer (InsightFace)
            print("[ImgEditor] Inicializando Face Analyzer...")
            import insightface
            from insightface.app import FaceAnalysis
            
            # Configurar para usar solo detección y reconocimiento (sin landmarks)
            # Esto evita el error de ONNX Runtime con landmark_2d_106.onnx
            self.face_analyzer = FaceAnalysis(
                allowed_modules=['detection', 'recognition']  # Sin landmark detection
            )
            # ctx_id=0 usa GPU si está disponible, ctx_id=-1 usa CPU
            ctx_id = 0 if use_cuda else -1
            self.face_analyzer.prepare(ctx_id=ctx_id, det_size=(640, 640))
            print(f"[ImgEditor] ✅ Face Analyzer inicializado (ctx_id={ctx_id}, CUDA={use_cuda})")
            
            # 2. Inicializar Face Swapper (Reactor/Inswapper)
            print("[ImgEditor] Inicializando Face Swapper...")
            from roop.processors.FaceSwap import FaceSwap
            
            self.face_swapper = FaceSwap()
            # Inicializar con configuración para CUDA
            device_name = 'cuda' if use_cuda else 'cpu'
            self.face_swapper.Initialize({
                'devicename': device_name,
                'model': 'inswapper_128.onnx'  # Modelo por defecto
            })
            print(f"[ImgEditor] ✅ Face Swapper inicializado ({device_name})")
            
            return True
            
        except ImportError as e:
            print(f"[ImgEditor] ❌ Error: Módulo no encontrado - {e}")
            print("[ImgEditor] Asegúrate de tener instalado: pip install insightface")
            return False
        except Exception as e:
            print(f"[ImgEditor] ❌ Error inicializando face swap: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_face_embedding(self, image: Image.Image):
        """Extrae el embedding facial de una imagen"""
        try:
            np_image = np.array(image)
            faces = self.face_analyzer.get(np_image)
            
            if len(faces) == 0:
                return None, None
            
            # Retornar la cara mas grande
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return best_face, best_face.embedding
        except Exception as e:
            print(f"[ImgEditor] Error extrayendo cara: {e}")
            return None, None
    
    def _restore_face(self, original: Image.Image, generated: Image.Image) -> Image.Image:
        """
        Restaura la cara original en la imagen generada usando face swap.
        
        Este es el paso 2 del enfoque de dos pasadas.
        """
        try:
            if not self._init_face_swap():
                print("[ImgEditor] Face swap no disponible, devolviendo imagen generada")
                return generated
            
            # Extraer cara del original (RGB para InsightFace)
            original_rgb = np.array(original)
            faces_original = self.face_analyzer.get(original_rgb)
            
            print(f"[ImgEditor] 🔍 Caras detectadas en original: {len(faces_original)}")
            if len(faces_original) > 0:
                # Mostrar bounding boxes
                for i, face in enumerate(faces_original):
                    bbox = face.bbox
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    print(f"   Cara {i}: bbox={bbox.tolist()}, área={area:.1f}")
            
            if len(faces_original) == 0:
                print("[ImgEditor] ❌ No se detectaron caras en el original")
                return generated
            
            # Usar la cara mas grande del original
            source_face = max(faces_original, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            print(f"[ImgEditor] ✅ Cara fuente seleccionada: bbox={source_face.bbox.tolist()}")
            
            # Convertir imagen generada a formato OpenCV (BGR) para face swapper
            generated_cv = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)
            
            # Detectar caras en la imagen generada (RGB para InsightFace)
            generated_rgb = np.array(generated)
            faces_generated = self.face_analyzer.get(generated_rgb)
            
            print(f"[ImgEditor] 🔍 Caras detectadas en generada: {len(faces_generated)}")
            if len(faces_generated) > 0:
                for i, face in enumerate(faces_generated):
                    bbox = face.bbox
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    print(f"   Cara {i}: bbox={bbox.tolist()}, área={area:.1f}")
            
            if len(faces_generated) == 0:
                print("[ImgEditor] No se detectaron caras en la generada, haciendo swap directo")
                # No hay cara en la generada, intentar swap directo
                # Crear una cara target ficticia basada en la posicion del original
                result = self.face_swapper.Run(source_face, source_face, generated_cv, paste_back=True)
                if result is not None:
                    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                return generated
            
            # Hacer face swap: cara del original -> imagen generada
            target_face = max(faces_generated, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            result = self.face_swapper.Run(source_face, target_face, generated_cv, paste_back=True)
            
            if result is not None:
                print("[ImgEditor] Cara restaurada correctamente")
                return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                print("[ImgEditor] Face swap fallo, devolviendo imagen generada")
                return generated
                
        except Exception as e:
            print(f"[ImgEditor] Error restaurando cara: {e}")
            import traceback
            traceback.print_exc()
            return generated

    def generate_intelligent(
        self,
        image,
        prompt: str,
        negative_prompt: str = None,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        strength: float = 0.65,
        seed: int = None,
        face_preserve: bool = True,
        auto_enhance: bool = True,
        use_rewriter: bool = True,
        ref_metadata: dict = None,
        engine: str = "sd",
        qwen_version: str = "q3",  # q3 = Q3_K_M, q2 = Q2_K
        zimage_version: str = "q4",  # q4 = Q4_K_M, q5 = Q5_K_M
        progress_callback=None  # AÑADIR CALLBACK
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Generación de imágenes con ComfyUI inpainting u otros motores:
        1. Mejora prompt
        2. Crea máscara con gradiente
        3. Genera con ComfyUI / VAR / Flux
        4. Face swap (opcional)
        """
        # --- MOTOR FLUX IMAGE EDIT vía ComfyUI ---
        if engine == "flux":
            print("[ImgEditor] === ⚡ USANDO FLUX IMAGE EDIT ===", flush=True)
            print(f"[ImgEditor] Prompt: {prompt[:100]}", flush=True)
            try:
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                print("[ImgEditor] Import OK", flush=True)

                if self.flux_edit_client is None:
                    self.flux_edit_client = get_flux_edit_comfy_client()
                    print("[ImgEditor] FLUX Client creado", flush=True)

                print("[ImgEditor] Cargando modelos FLUX...", flush=True)
                success, msg = self.flux_edit_client.load(progress_callback=progress_callback)
                print(f"[ImgEditor] Load result: {success} - {msg}", flush=True)
                if not success:
                    print(f"[ImgEditor] ❌ Error cargando FLUX: {msg}", flush=True)
                    return None, f"Error Motor FLUX: {msg}"

                print("[ImgEditor] Generando con FLUX...", flush=True)
                result_obj, msg = self.flux_edit_client.generate(
                    image=image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                print(f"[ImgEditor] Generate result: {result_obj is not None} - {msg}", flush=True)

                if result_obj and result_obj.image:
                    return result_obj.image, f"✅ FLUX Edit completada ({result_obj.time_taken:.1f}s)"
                return None, f"❌ Fallo en FLUX Edit: {msg}"
            except Exception as e:
                print(f"[ImgEditor] ❌ Error en motor FLUX: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                return None, f"Error Motor FLUX: {str(e)}"

        # --- MOTOR QWEN IMAGE EDIT vía ComfyUI ---
        if engine in ["qwen", "qwen2509", "qwen2512"]:
            print(f"[ImgEditor] === 🤖 USANDO QWEN IMAGE EDIT ({engine}) ===", flush=True)
            print(f"[ImgEditor] Version Qwen: {qwen_version}", flush=True)
            print(f"[ImgEditor] Prompt: {prompt[:100]}", flush=True)
            try:
                from roop.img_editor.qwen_edit_comfy_client import get_qwen_edit_comfy_client
                print("[ImgEditor] Import OK", flush=True)

                if self.qwen_edit_client is None:
                    self.qwen_edit_client = get_qwen_edit_comfy_client()
                    print("[ImgEditor] Qwen Client creado", flush=True)

                print("[ImgEditor] Cargando modelos Qwen...", flush=True)
                success, msg = self.qwen_edit_client.load(
                    progress_callback=progress_callback,
                    qwen_version=qwen_version
                )
                print(f"[ImgEditor] Load result: {success} - {msg}", flush=True)
                if not success:
                    print(f"[ImgEditor] ❌ Error cargando Qwen: {msg}", flush=True)
                    return None, f"Error Motor Qwen: {msg}"

                print("[ImgEditor] Generando con Qwen...", flush=True)
                result_obj, msg = self.qwen_edit_client.generate(
                    image=image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                print(f"[ImgEditor] Generate result: {result_obj is not None} - {msg}", flush=True)

                if result_obj and result_obj.image:
                    return result_obj.image, f"✅ Qwen Edit ({qwen_version}) completada ({result_obj.time_taken:.1f}s)"
                return None, f"❌ Fallo en Qwen Edit: {msg}"
            except Exception as e:
                print(f"[ImgEditor] ❌ Error en motor Qwen: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                return None, f"Error Motor Qwen: {str(e)}"

        # --- MOTOR Z-IMAGE TURBO GGUF (híbrido, ~6GB VRAM) ---
        if engine == "zimage":
            print("[ImgEditor] === ⚡ USANDO Z-IMAGE TURBO GGUF ===", flush=True)
            try:
                from roop.img_editor.zimage_edit_comfy_client import get_zimage_edit_comfy_client
                print("[ImgEditor] Import OK", flush=True)

                if self.zimage_edit_client is None:
                    self.zimage_edit_client = get_zimage_edit_comfy_client()
                    print("[ImgEditor] Z-Image Client creado", flush=True)

                zimage_version = zimage_version
                print(f"[ImgEditor] Z-Image version: {zimage_version}", flush=True)
                
                print("[ImgEditor] Cargando modelos Z-Image...", flush=True)
                success, msg = self.zimage_edit_client.load(
                    progress_callback=progress_callback,
                    zimage_version=zimage_version
                )
                print(f"[ImgEditor] Load result: {success} - {msg}", flush=True)
                if not success:
                    print(f"[ImgEditor] ❌ Error cargando Z-Image: {msg}", flush=True)
                    return None, f"Error Motor Z-Image: {msg}"

                print("[ImgEditor] Generando con Z-Image...", flush=True)
                result_obj, msg = self.zimage_edit_client.generate(
                    image=image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                print(f"[ImgEditor] Generate result: {result_obj is not None} - {msg}", flush=True)

                if result_obj and result_obj.image:
                    return result_obj.image, f"✅ Z-Image ({zimage_version}) completada ({result_obj.time_taken:.1f}s)"
                return None, f"❌ Fallo en Z-Image: {msg}"
            except Exception as e:
                print(f"[ImgEditor] ❌ Error en motor Z-Image: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                return None, f"Error Motor Z-Image: {str(e)}"

        # --- MOTOR HART (Hybrid Autoregressive Transformer, ~8GB VRAM) ---
        if engine == "hart":
            print("[ImgEditor] === 🔬 USANDO HART (Hybrid Autoregressive Transformer) ===", flush=True)
            print(f"[ImgEditor] Prompt: {prompt[:100]}", flush=True)
            try:
                from roop.img_editor.hart_edit_comfy_client import get_hart_edit_comfy_client
                print("[ImgEditor] Import OK", flush=True)

                if self.hart_edit_client is None:
                    self.hart_edit_client = get_hart_edit_comfy_client()
                    print("[ImgEditor] HART Client creado", flush=True)

                print("[ImgEditor] Cargando modelos HART...", flush=True)
                
                # Cerrar ComfyUI para liberar VRAM antes de generar con HART
                try:
                    from ui.tabs.comfy_launcher import stop
                    print("[ImgEditor] Cerrando ComfyUI para liberar VRAM...", flush=True)
                    stop()
                    import time
                    time.sleep(2)
                    print("[ImgEditor] ComfyUI cerrado", flush=True)
                except Exception as e:
                    print(f"[ImgEditor] Warning: No se pudo cerrar ComfyUI: {e}", flush=True)
                
                success, msg = self.hart_edit_client.load(progress_callback=progress_callback)
                print(f"[ImgEditor] Load result: {success} - {msg}", flush=True)
                if not success:
                    print(f"[ImgEditor] ❌ Error cargando HART: {msg}", flush=True)
                    return None, f"Error Motor HART: {msg}"

                print("[ImgEditor] Generando con HART...", flush=True)
                result_obj, msg = self.hart_edit_client.generate(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    width=1024,
                    height=1024,
                    venv_path=r"D:\PROJECTS\AUTOAUTO\venv_ext"
                )
                print(f"[ImgEditor] Generate result: {result_obj is not None} - {msg}", flush=True)

                # Reiniciar ComfyUI después de HART
                try:
                    from ui.tabs.comfy_launcher import start as start_comfy
                    print("[ImgEditor] Reiniciando ComfyUI...", flush=True)
                    import threading
                    def restart_comfy():
                        import time
                        time.sleep(3)
                        start_comfy()
                    threading.Thread(target=restart_comfy, daemon=True).start()
                    print("[ImgEditor] ComfyUI reiniciando en background", flush=True)
                except Exception as e:
                    print(f"[ImgEditor] Warning: No se pudo reiniciar ComfyUI: {e}", flush=True)

                if result_obj and result_obj.image:
                    return result_obj.image, f"✅ HART completada ({result_obj.time_taken:.1f}s)"
                return None, f"❌ Fallo en HART: {msg}"
            except Exception as e:
                print(f"[ImgEditor] ❌ Error en motor HART: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                return None, f"Error Motor HART: {str(e)}"

        # --- MOTOR ICEDIT vía ComfyUI (Nunchaku, ~4-6GB VRAM) - NO FUNCIONA ---
        if engine == "icedit":
            print("[ImgEditor] === 🧊 USANDO ICEDIT (ComfyUI Nunchaku) ===", flush=True)
            print(f"[ImgEditor] Prompt: {prompt[:100]}", flush=True)
            try:
                from roop.img_editor.icedit_comfy_client import get_icedit_comfy_client
                print("[ImgEditor] Import OK", flush=True)

                if self.icedit_client is None:
                    self.icedit_client = get_icedit_comfy_client()
                    print("[ImgEditor] ICEdit Client creado", flush=True)

                print("[ImgEditor] Cargando modelos ICEdit...", flush=True)
                success, msg = self.icedit_client.load(progress_callback=progress_callback)
                print(f"[ImgEditor] Load result: {success} - {msg}", flush=True)
                if not success:
                    print(f"[ImgEditor] ❌ Error cargando ICEdit: {msg}", flush=True)
                    return None, f"Error Motor ICEdit: {msg}"

                print("[ImgEditor] Generando con ICEdit...", flush=True)
                result_obj, msg = self.icedit_client.generate(
                    image=image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                print(f"[ImgEditor] Generate result: {result_obj is not None} - {msg}", flush=True)

                if result_obj and result_obj.image:
                    return result_obj.image, f"✅ ICEdit completada ({result_obj.time_taken:.1f}s)"
                return None, f"❌ Fallo en ICEdit: {msg}"
            except Exception as e:
                print(f"[ImgEditor] ❌ Error en motor ICEdit: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                return None, f"Error Motor ICEdit: {str(e)}"

        # --- MOTOR OMNIGEN2 GGUF (modelo nuevo, ~6GB VRAM) ---
        if engine == "omnigen2":
            print("[ImgEditor] === 🐷 USANDO OMNIGEN2 GGUF ===", flush=True)
            print(f"[ImgEditor] Prompt: {prompt[:100]}", flush=True)
            try:
                from roop.img_editor.omnigen2_gguf_comfy_client import get_omnigen2_comfy_client
                print("[ImgEditor] Import OK", flush=True)

                if self.omnigen2_client is None:
                    self.omnigen2_client = get_omnigen2_comfy_client()
                    print("[ImgEditor] OmniGen2 Client creado", flush=True)

                print("[ImgEditor] Cargando modelos OmniGen2...", flush=True)
                success, msg = self.omnigen2_client.load(progress_callback=progress_callback)
                print(f"[ImgEditor] Load result: {success} - {msg}", flush=True)
                if not success:
                    print(f"[ImgEditor] ❌ Error cargando OmniGen2: {msg}", flush=True)
                    return None, f"Error Motor OmniGen2: {msg}"

                print("[ImgEditor] Generando con OmniGen2...", flush=True)
                result_obj, msg = self.omnigen2_client.generate(
                    image=image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                print(f"[ImgEditor] Generate result: {result_obj is not None} - {msg}", flush=True)

                if result_obj and result_obj.image:
                    return result_obj.image, f"✅ OmniGen2 GGUF completada ({result_obj.time_taken:.1f}s)"
                return None, f"❌ Fallo en OmniGen2: {msg}"
            except Exception as e:
                print(f"[ImgEditor] ❌ Error en motor OmniGen2: {e}", flush=True)
                import traceback
                print(traceback.format_exc(), flush=True)
                return None, f"Error Motor OmniGen2: {str(e)}"

        # --- FLUJO ESTÁNDAR (SD1.5 / COMFYUI) ---
        original_image = None
        rewritten_prompt = prompt
        final_negative_prompt = negative_prompt
        intensity = "medium"  # Default
        
        # Procesar ref_metadata
        if ref_metadata is None:
            ref_metadata = {}
        character_ref = ref_metadata.get("character_ref", None)
        target_resolution = ref_metadata.get("resolution", None)
        text_overlay = ref_metadata.get("text_overlay", None)

        try:
            # 1. MEJORAR PROMPT (y detectar intensidad)
            print("[ImgEditor] === 🧠 ANALIZANDO PROMPT ===")
            analysis = self.analyze_prompt(prompt)
            
            # Añadir información de character reference al prompt
            if character_ref is not None:
                print("[ImgEditor] 👤 Character Reference detectada - manteniendo consistencia")
                prompt += ", consistent character, same person, character consistency"

            if use_rewriter and analysis.get('needs_rewriting'):
                print("[ImgEditor] === ✍️ MEJORANDO PROMPT ===")
                rewritten_prompt = self.rewrite_prompt(prompt, analysis)
                print(f"[ImgEditor] Original: {prompt}")
                print(f"[ImgEditor] Mejorado: {rewritten_prompt[:150]}...")

                # Obtener intensidad detectada por rewrite_prompt
                import roop.globals as rg
                if hasattr(rg, '_sd_editor_intensity'):
                    intensity = rg._sd_editor_intensity
                    print(f"[ImgEditor] Intensidad detectada: {intensity}")

                # Usar negative prompt generado por rewrite_prompt
                if hasattr(rg, '_sd_editor_negative_prompt') and rg._sd_editor_negative_prompt:
                    final_negative_prompt = rg._sd_editor_negative_prompt
                    print(f"[ImgEditor] Negative prompt: {final_negative_prompt[:100]}...")

            # Negative prompt por defecto si no se generó
            if final_negative_prompt is None:
                final_negative_prompt = "low quality, blurry, bad anatomy, ugly, deformed, child, underage, jpeg artifacts, noisy, out of focus"
            if isinstance(image, Image.Image):
                original_image = image.copy()
            elif hasattr(image, 'name'):
                original_image = Image.open(image.name).copy()
                image = original_image
            elif isinstance(image, str) and os.path.exists(image):
                original_image = Image.open(image).copy()
                image = original_image
            else:
                return None, "Error: Imagen no válida"

            # REDIMENSIONAR según resolución target (si está especificada)
            if target_resolution:
                target_w = target_resolution.get("width", original_image.width)
                target_h = target_resolution.get("height", original_image.height)
                if original_image.width != target_w or original_image.height != target_h:
                    print(f"[ImgEditor] Redimensionando a resolución target: {original_image.size} → {target_w}x{target_h}")
                    original_image = original_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                    image = original_image
            else:
                # REDIMENSIONAR si es muy grande (máximo 1536px para VRAM)
                max_size = 1536
                if original_image.width > max_size or original_image.height > max_size:
                    scale = max_size / max(original_image.width, original_image.height)
                    new_size = (int(original_image.width * scale), int(original_image.height * scale))
                    print(f"[ImgEditor] Redimensionando: {original_image.size} → {new_size}")
                    original_image = original_image.resize(new_size, Image.Resampling.LANCZOS)
                    image = original_image

            print(f"[ImgEditor] Imagen: {original_image.size}")

            # 2. CREAR MÁSCARA CON GRADIENTE - FORZAR coordenadas correctas
            import roop.globals as rg

            # Obtener coordenadas DEL SEMANTIC ANALYZER directamente
            coords = getattr(rg, '_sd_editor_mask_coords', None)

            # Si no hay coords del analyzer, usar default CORRECTO
            if coords is None:
                coords = {"y_start": 0.5, "y_end": 1.0, "x_start": 0.0, "x_end": 1.0}
                print("[ImgEditor] ⚠️ No hay coords del analyzer, usando default: y_start=0.5 (cintura)")

            w, h = original_image.size
            mask = self._create_gradient_mask(w, h, coords)

            # 3. GENERAR - PARÁMETROS OPTIMIZADOS PARA 8GB VRAM
            # ControlNet Tile PRESERVA estructura - NO elimina nada
            # Inpainting puro REGENERA el área - SÍ puede eliminar

            # Calcular denoise según intensidad (OPTIMIZADO para 8GB)
            if intensity == "low":
                denoise = 0.85  # Menos agresivo, mejor preservación
            elif intensity == "high":
                denoise = 0.92  # Eliminación fuerte pero estable
            else:  # medium
                denoise = 0.88  # Balance óptimo calidad/estabilidad

            # Steps y CFG optimizados para 8GB
            opt_steps = max(30, num_inference_steps)  # Mínimo 30 steps
            opt_cfg = min(8.0, guidance_scale)  # Máximo 8.0 para estabilidad

            print(f"[ImgEditor] === 🎨 GENERANDO (OPTIMIZADO 8GB VRAM) ===")
            print(f"[ImgEditor] Steps: {opt_steps}, CFG: {opt_cfg}, Denoise: {denoise}")
            print(f"[ImgEditor] ⚠️ INPAINTING PURO: REGENERA el área enmascarada")
            
            # Añadir character reference al negative prompt si NO se usa
            if character_ref is None and face_preserve:
                final_negative_prompt += ", inconsistent character, different person"

            result, msg = self._generate_inpaint_puro(
                image=original_image,
                mask=mask,
                prompt=rewritten_prompt,
                negative_prompt=final_negative_prompt,
                steps=opt_steps,
                cfg=opt_cfg,
                denoise=denoise,
                seed=seed,
                original_image=original_image
            )

            if not result:
                return None, f"Error: {msg}"

            # 4. CHARACTER REFERENCE - Aplicar consistencia de personaje
            if character_ref is not None:
                print("[ImgEditor] 👤 Aplicando Character Reference para consistencia...")
                result = self._apply_character_reference(character_ref, result, original_image)

            # 5. FACE SWAP (si está habilitado y no hay character reference)
            final_image = result
            if face_preserve and original_image is not None and character_ref is None:
                print("[ImgEditor] 👤 Restaurando cara...")
                final_image = self._restore_face(original_image, result)

            # 6. DEVOLVER como PIL (Gradio lo maneja directamente)
            # NOTA: El text overlay se aplica en el worker thread después
            
            # LIMPIEZA DE TEMPORALES Y VRAM
            self._cleanup_temp_and_vram()
            
            return final_image, "Generación completada"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"


    def _apply_character_reference(self, char_ref: Image.Image, generated: Image.Image, original: Image.Image = None) -> Image.Image:
        """
        Aplica Character Reference para mantener consistencia del personaje.
        
        Estrategia:
        1. Extraer cara de la character reference
        2. Aplicar face swap: char_ref face -> generated image
        3. Retornar imagen con consistencia de personaje
        
        Args:
            char_ref: Imagen de referencia del personaje
            generated: Imagen generada
            original: Imagen original (opcional, fallback)
        
        Returns:
            Imagen con character aplicado
        """
        try:
            if not self._init_face_swap():
                print("[ImgEditor] Face swap no disponible para character reference")
                return generated
            
            # Extraer cara de la character reference
            char_ref_rgb = np.array(char_ref)
            faces_char = self.face_analyzer.get(char_ref_rgb)
            
            if len(faces_char) == 0:
                print("[ImgEditor] ❌ No se detectó cara en character reference")
                return generated
            
            # Usar la cara más grande de la character reference
            source_face = max(faces_char, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            print(f"[ImgEditor] ✅ Cara de character reference: bbox={source_face.bbox.tolist()}")
            
            # Convertir imagen generada a BGR para face swapper
            generated_cv = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)
            
            # Detectar caras en la imagen generada
            generated_rgb = np.array(generated)
            faces_generated = self.face_analyzer.get(generated_rgb)
            
            if len(faces_generated) == 0:
                print("[ImgEditor] ❌ No se detectó cara en imagen generada")
                return generated
            
            # Usar la cara más grande de la imagen generada como target
            target_face = max(faces_generated, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            # Aplicar face swap: character_ref face -> generated image
            result = self.face_swapper.Run(source_face, target_face, generated_cv, paste_back=True)
            
            if result is not None:
                print("[ImgEditor] ✅ Character Reference aplicado correctamente")
                return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                print("[ImgEditor] ⚠️ Face swap falló, devolviendo imagen original")
                return generated
                
        except Exception as e:
            print(f"[ImgEditor] ❌ Error aplicando character reference: {e}")
            import traceback
            traceback.print_exc()
            return generated

    def save_result_to_file(self, image: Image.Image) -> str:
        """Guarda imagen en archivo y retorna la ruta"""
        import hashlib
        import os
        
        output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "ui", "tob", "ComfyUI", "output", "img_editor")
        os.makedirs(output_dir, exist_ok=True)
        
        # Nombre único basado en timestamp + hash
        timestamp = int(time.time())
        img_hash = hashlib.md5(image.tobytes()).hexdigest()[:8]
        filename = f"img_editor_{timestamp}_{img_hash}.png"
        filepath = os.path.join(output_dir, filename)
        
        image.save(filepath, "PNG")
        print(f"[ImgEditor] Imagen guardada: {filepath}")
        return filepath

    def _generate_inpaint(self, image: Image.Image, mask: Image.Image, 
                          prompt: str, negative_prompt: str,
                          steps: int = 30, cfg: float = 7.5, seed: int = None) -> Tuple[Optional[Image.Image], str]:
        """
        Genera con ComfyUI usando inpaint - PRESERVANDO imagen original.
        """
        from roop.comfy_client import check_comfy_available, disable_safety_checker
        
        if not check_comfy_available():
            return None, "ComfyUI no está corriendo"
        
        disable_safety_checker()
        client = self._get_client()
        
        # Verificar checkpoints
        checkpoints = client.get_checkpoints()
        if not checkpoints:
            return None, "No hay checkpoints disponibles"
        
        # Seleccionar checkpoint realista
        checkpoint = checkpoints[0]
        for ckpt in checkpoints:
            if any(x in ckpt.lower() for x in ["pornmaster", "porn", "realistic", "epicrealism", "absolutereality"]):
                checkpoint = ckpt
                break
        
        print(f"[ImgEditor] Checkpoint: {checkpoint}")
        
        # Guardar imágenes temporales
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, "sd_editor_inp.png")
        temp_mask_path = os.path.join(temp_dir, "sd_editor_m.png")
        
        image.save(temp_image_path)
        mask.save(temp_mask_path)
        
        # Subir a ComfyUI
        image_filename = client.upload_image(temp_image_path)
        mask_filename = client.upload_image(temp_mask_path)
        
        if not image_filename or not mask_filename:
            return None, "Error subiendo imágenes a ComfyUI"
        
        # Construir workflow de inpaint
        from roop.img_editor.comfy_workflows import build_inpaint_workflow
        
        # DENOISE: 0.75-0.85 para cambios mayores, mejor cobertura
        denoise = 0.8  # Más alto para cubrir mejor el área combinada
        
        workflow = build_inpaint_workflow(
            image_filename=image_filename,
            mask_filename=mask_filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed if seed else int(time.time()),
            steps=steps,
            cfg=cfg,
            denoise=denoise,  # IMPORTANTE: preservar imagen original
            checkpoint=checkpoint
        )
        
        # Ejecutar workflow
        prompt_id, success, error = client.queue_prompt(workflow)
        if not success:
            return None, f"Error en workflow: {error}"
        
        print(f"[ImgEditor] Generando... ID: {prompt_id[:8]}")
        print(f"[ImgEditor] Denoise: {denoise}, Steps: {steps}, CFG: {cfg}")
        time.sleep(2)
        
        images = client.get_images(prompt_id, "*")
        if not images:
            return None, "No se obtuvo imagen de ComfyUI"
        
        from io import BytesIO
        result_image = Image.open(BytesIO(images[0]))
        
        # Limpiar temporales
        try:
            os.remove(temp_image_path)
            os.remove(temp_mask_path)
        except:
            pass
        
        print(f"[ImgEditor] Imagen generada: {result_image.size}")
        return result_image, "Inpaint completado"

    def _generate_img2img_masked(self, image: Image.Image, mask: Image.Image, 
                          prompt: str, negative_prompt: str,
                          steps: int = 30, cfg: float = 7.5, 
                          denoise: float = 0.65, seed: int = None) -> Tuple[Optional[Image.Image], str]:
        """
        Genera con ComfyUI usando img2img ENMASCARADO - PRESERVA la estructura original.
        A diferencia de inpaint, esto mezcla mejor con el original.
        """
        from roop.comfy_client import check_comfy_available, disable_safety_checker
        
        if not check_comfy_available():
            return None, "ComfyUI no está corriendo"
        
        disable_safety_checker()
        client = self._get_client()
        
        # Verificar checkpoints
        checkpoints = client.get_checkpoints()
        if not checkpoints:
            return None, "No hay checkpoints disponibles"
        
        # Seleccionar checkpoint realista
        checkpoint = checkpoints[0]
        for ckpt in checkpoints:
            if any(x in ckpt.lower() for x in ["pornmaster", "porn", "realistic", "epicrealism", "absolutereality"]):
                checkpoint = ckpt
                break
        
        print(f"[ImgEditor] Checkpoint: {checkpoint}")
        
        # Guardar imágenes temporales
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, "sd_editor_img2img.png")
        temp_mask_path = os.path.join(temp_dir, "sd_editor_m2.png")
        
        image.save(temp_image_path)
        mask.save(temp_mask_path)
        
        # Subir a ComfyUI
        image_filename = client.upload_image(temp_image_path)
        mask_filename = client.upload_image(temp_mask_path)
        
        if not image_filename or not mask_filename:
            return None, "Error subiendo imágenes a ComfyUI"
        
        # Construir workflow de img2img enmascarado
        from roop.img_editor.comfy_workflows import build_inpaint_workflow
        
        workflow = build_inpaint_workflow(
            image_filename=image_filename,
            mask_filename=mask_filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed if seed else int(time.time()),
            steps=steps,
            cfg=cfg,
            denoise=denoise,  # Más bajo = más preservación
            checkpoint=checkpoint
        )
        
        # Ejecutar workflow
        prompt_id, success, error = client.queue_prompt(workflow)
        if not success:
            return None, f"Error en workflow: {error}"
        
        print(f"[ImgEditor] Generando... ID: {prompt_id[:8]}")
        print(f"[ImgEditor] Denoise: {denoise} (preserva estructura), Steps: {steps}, CFG: {cfg}")
        time.sleep(2)
        
        images = client.get_images(prompt_id, "*")
        if not images:
            return None, "No se obtuvo imagen de ComfyUI"
        
        from io import BytesIO
        result_image = Image.open(BytesIO(images[0]))
        
        # Limpiar temporales
        try:
            os.remove(temp_image_path)
            os.remove(temp_mask_path)
        except:
            pass
        
        print(f"[ImgEditor] Imagen generada: {result_image.size}")
        return result_image, "Img2img enmascarado completado"

    def _create_gradient_mask(self, w: int, h: int, coords: Dict) -> Image.Image:
        """
        Crea máscara con gradiente suave para transición natural.
        OPTIMIZADO v3: Sistema CORRECTO - y_start es DESDE ARRIBA (0=arriba, 1=abajo).
        Máscara NEGRO (0=preservar) con área BLANCA (255=modificar) solo en zona especificada.
        """
        # Crear máscara en NEGRO (0 = área a PRESERVAR)
        mask = Image.new('L', (w, h), 0)
        from PIL import ImageDraw, ImageFilter
        draw = ImageDraw.Draw(mask)

        # Calcular coordenadas en píxeles
        # y_start: desde dónde empieza la máscara (0 = arriba, 1 = abajo)
        # y_end: hasta dónde llega la máscara
        y_start_px = int(h * coords.get('y_start', 0.5))
        y_end_px = int(h * coords.get('y_end', 1.0))
        x_start_px = int(w * coords.get('x_start', 0.0))
        x_end_px = int(w * coords.get('x_end', 1.0))

        print(f"[ImgEditor] Creando máscara: y_start={y_start_px}px ({coords.get('y_start', 0.5):.1%}), y_end={y_end_px}px")

        # Dibujar área BLANCA (255 = área a MODIFICAR) con gradiente
        gradient_height = int((y_end_px - y_start_px) * 0.3)  # 30% para gradiente suave
        
        for y in range(y_start_px, y_end_px):
            # Calcular alpha para el gradiente
            if y < y_start_px + gradient_height:
                # Gradiente de entrada: 0 → 255
                progress = (y - y_start_px) / gradient_height
                alpha = int(255 * progress)
            else:
                # Área sólida: 255
                alpha = 255
            
            draw.line([(x_start_px, y), (x_end_px, y)], fill=alpha)

        # Blur gaussiano para suavizar bordes
        blur_radius = max(20, int((y_end_px - y_start_px) * 0.1))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # Calcular porcentaje de área afectada
        mask_array = np.array(mask)
        coverage = (mask_array > 128).sum() / mask_array.size * 100
        
        print(f"[ImgEditor] Máscara: {mask.size}, coverage: {coverage:.1f}%")
        return mask

    def _generate_with_controlnet_tile(self, image: Image.Image, mask: Image.Image, 
                          prompt: str, negative_prompt: str,
                          steps: int = 30, cfg: float = 7.5,
                          denoise: float = 0.7, seed: int = None,
                          original_image: Image.Image = None) -> Tuple[Optional[Image.Image], str]:
        """
        Genera con ComfyUI usando ControlNet Tile - PRESERVA LA ESTRUCTURA ORIGINAL.
        Usa directamente el archivo Tile sin verificar (ya sabemos que existe).
        """
        from roop.comfy_client import check_comfy_available, disable_safety_checker
        
        if not check_comfy_available():
            return None, "ComfyUI no está corriendo"
        
        disable_safety_checker()
        client = self._get_client()
        
        # Verificar checkpoints
        checkpoints = client.get_checkpoints()
        if not checkpoints:
            return None, "No hay checkpoints disponibles"
        
        checkpoint = checkpoints[0]
        for ckpt in checkpoints:
            if any(x in ckpt.lower() for x in ["pornmaster", "porn", "realistic", "epicrealism", "absolutereality"]):
                checkpoint = ckpt
                break
        
        print(f"[ImgEditor] Checkpoint: {checkpoint}")
        print(f"[ImgEditor] ✅ USANDO CONTROLNET TILE DIRECTAMENTE (archivo confirmado)")
        
        # Guardar imágenes temporales
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, "sd_editor_tile.png")
        temp_mask_path = os.path.join(temp_dir, "sd_editor_tile_m.png")
        
        image.save(temp_image_path)
        mask.save(temp_mask_path)
        
        # Subir a ComfyUI
        image_filename = client.upload_image(temp_image_path)
        mask_filename = client.upload_image(temp_mask_path)
        
        if not image_filename or not mask_filename:
            return None, "Error subiendo imágenes a ComfyUI"
        
        # Construir workflow CON ControlNet Tile DIRECTO
        workflow = self._build_tile_inpaint_workflow(
            client=client,
            image_filename=image_filename,
            mask_filename=mask_filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed if seed else int(time.time()),
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            checkpoint=checkpoint
        )
        
        # Ejecutar workflow
        prompt_id, success, error = client.queue_prompt(workflow)
        if not success:
            return None, f"Error en workflow: {error}"
        
        print(f"[ImgEditor] Generando con Tile... ID: {prompt_id[:8]}")
        print(f"[ImgEditor] Denoise: {denoise}, Steps: {steps}, CFG: {cfg}")
        time.sleep(2)
        
        images = client.get_images(prompt_id, "*")
        if not images:
            return None, "No se obtuvo imagen de ComfyUI"

        from io import BytesIO
        result_image = Image.open(BytesIO(images[0]))

        # COMPOSICIÓN EXPLÍCITA: Combinar imagen generada con original usando la máscara
        # Esto asegura que no queden áreas grises
        if original_image is not None:
            print("[ImgEditor] Composición explícita original + generada...")
            result_image = self._composite_images(original_image, result_image, mask)
        else:
            print("[ImgEditor] Sin imagen original para composición, usando resultado directo")

        # Limpiar temporales
        try:
            os.remove(temp_image_path)
            os.remove(temp_mask_path)
        except:
            pass

        print(f"[ImgEditor] Imagen generada: {result_image.size}")
        return result_image, "ControlNet Tile completado"

    def _generate_inpaint_puro(self, image: Image.Image, mask: Image.Image,
                          prompt: str, negative_prompt: str,
                          steps: int = 30, cfg: float = 7.5,
                          denoise: float = 0.95, seed: int = None,
                          original_image: Image.Image = None,
                          use_flux: bool = False) -> Tuple[Optional[Image.Image], str]:
        """
        Genera con ComfyUI usando INPAINTING PURO.
        """
        from roop.comfy_client import check_comfy_available, disable_safety_checker

        if not check_comfy_available():
            return None, "ComfyUI no está corriendo"

        disable_safety_checker()
        client = self._get_client()

        # Verificar checkpoints
        checkpoints = client.get_checkpoints()
        if not checkpoints:
            return None, "No hay checkpoints disponibles"

        checkpoint = checkpoints[0]
        for ckpt in checkpoints:
            if any(x in ckpt.lower() for x in ["pornmaster", "porn", "realistic", "epicrealism", "absolutereality"]):
                checkpoint = ckpt
                break

        print(f"[ImgEditor] Checkpoint: {checkpoint}")
        print(f"[ImgEditor] ⚠️ USANDO INPAINTING PURO (SIN ControlNet Tile) - MODO ELIMINACIÓN")

        # Guardar imágenes temporales
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, "sd_editor_inpaint.png")
        temp_mask_path = os.path.join(temp_dir, "sd_editor_inpaint_m.png")

        image.save(temp_image_path)
        mask.save(temp_mask_path)

        # Subir a ComfyUI
        image_filename = client.upload_image(temp_image_path)
        mask_filename = client.upload_image(temp_mask_path)

        if not image_filename or not mask_filename:
            return None, "Error subiendo imágenes a ComfyUI"

        # Construir workflow de INPAINTING PURO (sin ControlNet)
        workflow = self._build_pure_inpaint_workflow(
            client=client,
            image_filename=image_filename,
            mask_filename=mask_filename,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed if seed else int(time.time()),
            steps=steps,
            cfg=cfg,
            denoise=denoise,
            checkpoint=checkpoint
        )

        # Ejecutar workflow
        prompt_id, success, error = client.queue_prompt(workflow)
        if not success:
            return None, f"Error en workflow: {error}"

        print(f"[ImgEditor] Generando con Inpaint Puro... ID: {prompt_id[:8]}")
        print(f"[ImgEditor] Denoise: {denoise} (ALTO para eliminación), Steps: {steps}, CFG: {cfg}")
        time.sleep(2)

        images = client.get_images(prompt_id, "*")
        if not images:
            return None, "No se obtuvo imagen de ComfyUI"

        from io import BytesIO
        result_image = Image.open(BytesIO(images[0]))

        # COMPOSICIÓN EXPLÍCITA: Combinar imagen generada con original usando la máscara
        if original_image is not None:
            print("[ImgEditor] Composición explícita original + generada...")
            result_image = self._composite_images(original_image, result_image, mask)
        else:
            print("[ImgEditor] Sin imagen original para composición, usando resultado directo")

        # Limpiar temporales
        try:
            os.remove(temp_image_path)
            os.remove(temp_mask_path)
        except:
            pass

        # LIMPIEZA DE VRAM DESPUÉS DE GENERAR CON COMFYUI
        self._cleanup_temp_and_vram()

        print(f"[ImgEditor] Imagen generada: {result_image.size}")
        return result_image, "Inpainting puro completado"

    def _build_pure_inpaint_workflow(self, client, image_filename: str, mask_filename: str,
                                     prompt: str, negative_prompt: str, seed: int,
                                     steps: int, cfg: float, denoise: float, checkpoint: str) -> Dict:
        """
        Construye workflow de INPAINTING PURO sin ControlNet.
        Usa VAEEncodeForInpaint + KSampler + VAEDecode directamente.
        """
        return {
            # Cargar imagen original
            "1": {
                "inputs": {"image": image_filename, "upload": "image"},
                "class_type": "LoadImage"
            },
            # Cargar máscara
            "100": {
                "inputs": {"image": mask_filename, "upload": "image"},
                "class_type": "LoadImage"
            },
            # Cargar Checkpoint
            "2": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple"
            },
            # Prompt positivo
            "3": {
                "inputs": {"clip": ["2", 1], "text": prompt},
                "class_type": "CLIPTextEncode"
            },
            # Prompt negativo
            "4": {
                "inputs": {"clip": ["2", 1], "text": negative_prompt},
                "class_type": "CLIPTextEncode"
            },
            # Crear máscara desde imagen
            "10": {
                "inputs": {"image": ["100", 0], "channel": "red"},
                "class_type": "ImageToMask"
            },
            # VAEEncodeForInpaint - ESTO ES CLAVE para inpainting
            "5": {
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["2", 2],
                    "mask": ["10", 0],
                    "grow_mask_by": 10  # Expandir máscara 10px para bordes suaves
                },
                "class_type": "VAEEncodeForInpaint"
            },
            # KSampler - SIN ControlNet
            "6": {
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],  # Latent enmascarado
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": denoise  # ALTO (0.95) para eliminación
                },
                "class_type": "KSampler"
            },
            # VAEDecode
            "7": {
                "inputs": {"vae": ["2", 2], "samples": ["6", 0]},
                "class_type": "VAEDecode"
            },
            # Guardar
            "8": {
                "inputs": {"filename_prefix": "sd_inpaint", "images": ["7", 0]},
                "class_type": "SaveImage"
            }
        }

    def _composite_images(self, original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Compone explícitamente la imagen generada con la original usando la máscara.
        Esto evita áreas grises o artefactos.
        OPTIMIZADO: Mejor preservación de color en áreas no enmascaradas.
        """
        try:
            # Asegurar mismo tamaño
            if original.size != generated.size:
                generated = generated.resize(original.size, Image.Resampling.LANCZOS)
            if mask.size != original.size:
                mask = mask.resize(original.size, Image.Resampling.LANCZOS)

            # Convertir a numpy
            original_np = np.array(original).astype(np.float32)
            generated_np = np.array(generated).astype(np.float32)
            mask_np = np.array(mask).astype(np.float32) / 255.0

            # Asegurar máscara 3 canales
            if mask_np.ndim == 2:
                mask_np = np.stack([mask_np] * 3, axis=-1)

            # SUAVIZAR MÁSCARA: Aplicar blur gaussiano adicional para transición más suave
            import cv2
            kernel_size = 31  # Kernel grande para transición muy suave
            if kernel_size % 2 == 0:
                kernel_size += 1
            mask_np = cv2.GaussianBlur(mask_np, (kernel_size, kernel_size), 0)
            mask_np = np.clip(mask_np, 0.0, 1.0)

            # Composición: mask * generated + (1 - mask) * original
            # En áreas donde mask=1 → usa generated
            # En áreas donde mask=0 → usa original (PRESERVA COLOR ORIGINAL)
            result_np = mask_np * generated_np + (1 - mask_np) * original_np
            result_np = np.clip(result_np, 0, 255).astype(np.uint8)

            # Calcular porcentaje de área afectada (solo donde mask > 0.5)
            coverage = (mask_np > 0.5).sum() / mask_np.size * 100
            
            print(f"[ImgEditor] Composición completada, máscara range: [{mask_np.min():.2f}, {mask_np.max():.2f}]")
            print(f"[ImgEditor] Composición: {coverage:.1f}% área generada, {100-coverage:.1f}% área original preservada")
            return Image.fromarray(result_np)

        except Exception as e:
            print(f"[ImgEditor] Error en composición: {e}")
            # Fallback: devolver imagen generada
            return generated

    def _build_tile_inpaint_workflow(self, client, image_filename: str, mask_filename: str,
                                     prompt: str, negative_prompt: str, seed: int,
                                     steps: int, cfg: float, denoise: float, checkpoint: str) -> Dict:
        """
        Construye workflow con ControlNet Tile + Inpaint.
        Usa el modelo Tile directamente sin consultar (ya sabemos que existe).
        OPTIMIZADO: Composición explícita para evitar áreas grises.
        """
        # Modelo Tile que sabemos que existe
        tile_model = "control_v11f1e_sd15_tile.pth"

        print(f"[ImgEditor] ControlNet Tile model: {tile_model}")

        # Workflow con ControlNet Tile - CORREGIDO
        return {
            # Cargar imagen original
            "1": {
                "inputs": {"image": image_filename, "upload": "image"},
                "class_type": "LoadImage"
            },
            # Cargar máscara
            "100": {
                "inputs": {"image": mask_filename, "upload": "image"},
                "class_type": "LoadImage"
            },
            # Cargar Checkpoint
            "2": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple"
            },
            # Prompt positivo
            "3": {
                "inputs": {"clip": ["2", 1], "text": prompt},
                "class_type": "CLIPTextEncode"
            },
            # Prompt negativo
            "4": {
                "inputs": {"clip": ["2", 1], "text": negative_prompt},
                "class_type": "CLIPTextEncode"
            },
            # Crear máscara desde imagen
            "10": {
                "inputs": {"image": ["100", 0], "channel": "red"},
                "class_type": "ImageToMask"
            },
            # VAEEncodeForInpaint
            "5": {
                "inputs": {"pixels": ["1", 0], "vae": ["2", 2], "mask": ["10", 0], "grow_mask_by": 6},
                "class_type": "VAEEncodeForInpaint"
            },
            # Cargar ControlNet Tile
            "21": {
                "inputs": {"control_net_name": tile_model},
                "class_type": "ControlNetLoader"
            },
            # ControlNet Apply (usa CONDITIONING, no LATENT)
            "20": {
                "inputs": {
                    "conditioning": ["3", 0],
                    "control_net": ["21", 0],
                    "image": ["1", 0],
                    "strength": 0.8
                },
                "class_type": "ControlNetApply"
            },
            # KSampler - usa el conditioning con ControlNet aplicado
            "6": {
                "inputs": {
                    "model": ["2", 0],
                    "positive": ["20", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": denoise
                },
                "class_type": "KSampler"
            },
            # VAEDecode
            "7": {
                "inputs": {"vae": ["2", 2], "samples": ["6", 0]},
                "class_type": "VAEDecode"
            },
            # Guardar
            "8": {
                "inputs": {"filename_prefix": "image_editor_tile", "images": ["7", 0]},
                "class_type": "SaveImage"
            }
        }

    def _enhance_prompt(self, prompt: str, analysis: Dict[str, bool]) -> str:
        """
        Mejora el prompt automáticamente agregando términos de calidad.
        """
        quality_terms = [
            "high quality", "detailed", "realistic", "professional",
            "sharp focus", "excellent lighting", "masterpiece"
        ]
        
        # Si es cambio de ropa/cuerpo, agregar términos específicos
        if analysis.get("use_inpaint"):
            if any(kw in prompt.lower() for kw in ["desnuda", "naked", "nude", "sin ropa"]):
                prompt = f"{prompt}, natural skin, realistic body, detailed skin texture, body details"
        
        # Agregar términos de calidad
        for term in quality_terms:
            if term not in prompt.lower():
                prompt = f"{prompt}, {term}"
        
        return prompt

    def _generate_with_comfyui_intelligent(
        self,
        image,
        prompt: str,
        negative_prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        strength: float,
        seed: int,
        face_preserve: bool,
        analysis: Dict[str, bool],
        controlnet_available: Dict[str, bool] = None
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Genera con ComfyUI usando técnicas inteligentes según el análisis del prompt.
        
        Soporta:
        - ControlNet OpenPose REAL para poses
        - ControlNet Tile REAL para upscale
        - CLIPSeg para inpaint automático
        - IP-Adapter para mantener identidad
        """
        from roop.comfy_client import check_comfy_available, disable_safety_checker
        if not check_comfy_available():
            return None, "Error: ComfyUI no está corriendo"
        
        disable_safety_checker()
        client = self._get_client()
        
        # Verificar checkpoints
        checkpoints = client.get_checkpoints()
        if not checkpoints:
            return None, "Error: No hay checkpoints disponibles"
        
        # Inicializar ControlNet utils si es necesario
        if controlnet_available is None:
            if self.controlnet_utils is None:
                self.controlnet_utils = get_controlnet_utils()
            controlnet_available = self.controlnet_utils.check_controlnet_available()
        
        # Guardar imagen temporal
        temp_dir = tempfile.gettempdir()
        temp_image_path = os.path.join(temp_dir, "sd_editor_input.png")
        image.save(temp_image_path)
        
        # Subir imagen a ComfyUI
        image_filename = client.upload_image(temp_image_path)
        if not image_filename:
            return None, "Error: No se pudo subir la imagen a ComfyUI"
        
        # ============================================
        # PREPARAR CONTROLNET (si es necesario)
        # ============================================
        pose_image_filename = None
        if analysis.get("use_openpose") and controlnet_available.get("openpose"):
            print("[ImgEditor] === 🦴 Generando pose con OpenPose ===")
            try:
                pose_image = self.controlnet_utils.detect_pose(image)
                if pose_image is not None:
                    # Guardar y subir pose a ComfyUI
                    temp_pose_path = os.path.join(temp_dir, "sd_editor_pose.png")
                    Image.fromarray(pose_image).save(temp_pose_path)
                    pose_image_filename = client.upload_image(temp_pose_path)
                    print(f"[ImgEditor] ✅ Pose detectada: {pose_image.shape}")
            except Exception as e:
                print(f"[ImgEditor] ⚠️ Error detectando pose: {e}")
        
        # ============================================
        # PREPARAR INPAINT (CLIPSeg o fallback)
        # ============================================
        # NOTA: CLIPSeg tiene problemas en Windows, usar fallback si falla
        mask_filename = None
        use_inpaint = False

        if analysis.get("use_inpaint"):
            print("[ImgEditor] === 🎭 Generando máscara para inpaint ===")
            print(f"[ImgEditor] Intentando CLIPSeg primero...")

            # INTENTO 1: CLIPSeg (si funciona)
            clipseg_success = False
            try:
                segmenter = get_clothing_segmenter()
                success, msg = segmenter.load()
                print(f"[ImgEditor] CLIPSeg load result: success={success}, msg={msg}")
                if success:
                    mask_image, mask_array = segmenter.segment_clothing(
                        image=image,
                        threshold=0.4,
                        combine_mode="max",
                        include_skin_exclusion=True,
                        dilation=10
                    )
                    if mask_image is not None and mask_array.sum() > 1000:
                        temp_mask_path = os.path.join(temp_dir, "sd_editor_mask.png")
                        mask_image.save(temp_mask_path)
                        mask_filename = client.upload_image(temp_mask_path)
                        use_inpaint = True
                        clipseg_success = True
                        cobertura = (mask_array.sum() / 255) / (image.size[0] * image.size[1]) * 100
                        print(f"[ImgEditor] ✅ CLIPSeg: Máscara generada, cobertura: {cobertura:.1f}%")
                    else:
                        print(f"[ImgEditor] ⚠️ CLIPSeg: Máscara muy pequeña ({mask_array.sum() if mask_array is not None else 0} pixeles)")
                else:
                    print(f"[ImgEditor] ⚠️ CLIPSeg no cargó: {msg}")
            except Exception as e:
                print(f"[ImgEditor] ⚠️ CLIPSeg excepción: {e}")
                import traceback
                traceback.print_exc()

            # INTENTO 2: Fallback - máscara manual según coordenadas de Ollama
            if not clipseg_success:
                print("[ImgEditor] === 🔄 Ejecutando FALLBACK ===")
                try:
                    w, h = image.size
                    print(f"[ImgEditor] Creando máscara fallback: {w}x{h}")

                    mask_image = Image.new('L', (w, h), 0)
                    from PIL import ImageDraw
                    draw = ImageDraw.Draw(mask_image)

                    # Obtener coordenadas directamente de Ollama (SIN hardcodeo)
                    import roop.globals as rg
                    mask_coords = getattr(rg, '_sd_editor_mask_coords', None)

                    print(f"[ImgEditor] Coordenadas de Ollama: {mask_coords}")

                    if mask_coords and isinstance(mask_coords, dict):
                        # Ollama devolvió coordenadas específicas - USAR DIRECTAMENTE
                        y_start = float(mask_coords.get('y_start', 0.15))
                        y_end = float(mask_coords.get('y_end', 1.0))
                        x_start = float(mask_coords.get('x_start', 0.0))
                        x_end = float(mask_coords.get('x_end', 1.0))

                        x1 = int(w * x_start)
                        y1 = int(h * y_start)
                        x2 = int(w * x_end)
                        y2 = int(h * y_end)
                        draw.rectangle([x1, y1, x2, y2], fill=255)
                        print(f"[ImgEditor] Máscara (coord Ollama): [{x1}, {y1}, {x2}, {y2}]")
                    else:
                        # Fallback genérico: cuerpo completo menos cara (15% hasta abajo)
                        area_start = int(h * 0.15)
                        draw.rectangle([0, area_start, w, h], fill=255)
                        print(f"[ImgEditor] Máscara cuerpo (fallback): [0, {area_start}, {w}, {h}]")

                    # Aplicar blur para bordes suaves
                    import cv2
                    mask_cv = cv2.cvtColor(np.array(mask_image), cv2.COLOR_GRAY2RGB)
                    mask_cv = cv2.GaussianBlur(mask_cv, (51, 51), 0)
                    mask_image = Image.fromarray(mask_cv[:,:,0])

                    temp_mask_path = os.path.join(temp_dir, "sd_editor_mask_fallback.png")
                    mask_image.save(temp_mask_path)
                    print(f"[ImgEditor] Máscara guardada en: {temp_mask_path}")

                    mask_filename = client.upload_image(temp_mask_path)
                    print(f"[ImgEditor] Máscara subida a ComfyUI: {mask_filename}")

                    use_inpaint = True
                    print(f"[ImgEditor] ✅ FALLBACK EXITOSO: Máscara generada ({w}x{h})")
                except Exception as e:
                    print(f"[ImgEditor] ⚠️ FALLBACK FALLÓ: {e}")
                    import traceback
                    traceback.print_exc()
        
        if not use_inpaint and analysis.get("use_inpaint"):
            print("[ImgEditor] ℹ️ Inpaint no disponible (ni CLIPSeg ni fallback), usando solo IP-Adapter")
        elif use_inpaint:
            print(f"[ImgEditor] ✅ Inpaint ACTIVADO, mask_filename={mask_filename}")
        
        # Importar workflows
        from roop.img_editor.comfy_workflows import (
            build_img2img_workflow,
            build_inpaint_workflow,
            build_editor_workflow,
            get_default_checkpoint,
            check_controlnet_available,
            check_ipadapter_available
        )
        
        checkpoint = get_default_checkpoint()
        has_controlnet = check_controlnet_available()
        has_ipadapter = check_ipadapter_available()
        
        # Determinar qué usar
        use_openpose_real = pose_image_filename is not None
        use_tile_real = analysis.get("auto_upscale") and controlnet_available.get("tile")
        use_inpaint_real = mask_filename is not None and use_inpaint
        use_ipadapter = analysis.get("use_ipadapter", False) and has_ipadapter
        
        print(f"[ImgEditor] Workflow:")
        print(f"  - OpenPose REAL: {use_openpose_real}")
        print(f"  - Tile REAL: {use_tile_real}")
        print(f"  - Inpaint CLIPSeg: {use_inpaint_real}")
        print(f"  - IP-Adapter: {use_ipadapter}")
        
        # Construir workflow apropiado
        if use_inpaint_real:
            # Workflow de inpaint con máscara
            print(f"[ImgEditor] Usando workflow de INPAINT")
            print(f"[ImgEditor] Prompt a usar: {prompt}")
            workflow = build_inpaint_workflow(
                image_filename=image_filename,
                mask_filename=mask_filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed if seed is not None else int(time.time()),
                steps=num_inference_steps,
                cfg=guidance_scale,
                denoise=0.85,  # Denoise alto para inpaint
                checkpoint=checkpoint
            )
        elif use_openpose_real or use_tile_real or use_ipadapter:
            # Workflow de editor con ControlNets/IP-Adapter
            print(f"[ImgEditor] Usando workflow de EDITOR")
            print(f"[ImgEditor] Prompt a usar: {prompt}")
            workflow = build_editor_workflow(
                image_filename=image_filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed if seed is not None else int(time.time()),
                steps=num_inference_steps,
                cfg=guidance_scale,
                denoise=strength,
                checkpoint=checkpoint,
                use_controlnet=use_openpose_real or use_tile_real,
                use_ipadapter=use_ipadapter,
                controlnet_strength=0.5,
                ipadapter_strength=0.6 if use_ipadapter else 0.0
            )
        else:
            # Workflow simple img2img
            print(f"[ImgEditor] Usando workflow IMG2IMG simple")
            print(f"[ImgEditor] Prompt a usar: {prompt}")
            workflow = build_img2img_workflow(
                image_filename=image_filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed if seed is not None else int(time.time()),
                steps=num_inference_steps,
                cfg=guidance_scale,
                denoise=strength,
                checkpoint=checkpoint
            )
        
        # Ejecutar workflow
        prompt_id, success, error = client.queue_prompt(workflow)
        if not success:
            return None, f"Error: {error}"
        
        print(f"[ImgEditor] Generando... ID: {prompt_id[:8]}...")
        time.sleep(2)
        images = client.get_images(prompt_id, "*")
        
        if not images:
            return None, "Error: No se pudo obtener la imagen de ComfyUI"
        
        from io import BytesIO
        generated_image = Image.open(BytesIO(images[0]))
        
        # PASADA 2: Restaurar cara original con MÁXIMA preservación
        final_image = generated_image
        if face_preserve and image is not None:
            print("[ImgEditor] 👤 PASADA 2: Restaurando cara original (máxima preservación)...")
            # Ajustar parámetros para máxima preservación
            try:
                import roop.globals as rg
                rg.blend_ratio = 0.98  # 98% cara original
                rg.distance_threshold = 0.3  # Más estricto
            except:
                print("[ImgEditor] ⚠️ No se pudo ajustar blend_ratio, usando valores por defecto")
            final_image = self._restore_face(image, generated_image)
        
        # Limpiar temporal
        try:
            os.remove(temp_image_path)
        except:
            pass
        
        mode_str = []
        if use_openpose_real: mode_str.append("OpenPose")
        if use_tile_real: mode_str.append("Tile")
        if use_inpaint_real: mode_str.append("Inpaint")
        if use_ipadapter: mode_str.append("IPAdapter")
        mode_str = "+".join(mode_str) if mode_str else "Img2Img"
        
        return final_image, f"ComfyUI {mode_str} generado correctamente"

    def generate(
        self,
        image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 8.5,
        strength: float = 0.95,
        seed: int = None,
        face_preserve: bool = True,
        use_ipadapter: bool = False,
        use_controlnet: bool = False,
        controlnet_strength: float = 0.35,
        ipadapter_strength: float = 0.7,
        use_flux: bool = True  # Usar FLUX Fill Pipeline si está disponible
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Genera una imagen editada usando el enfoque de DOS PASADAS:
        
        PASADA 1: Generar imagen con el prompt completo (strength alto)
                  Esto permite que los cambios del prompt se apliquen correctamente
                  
        PASADA 2: Face swap para restaurar la cara original
                  Esto preserva la identidad facial 100%
        
        Args:
            image: Imagen de entrada (PIL Image, path, o objeto con .name)
            prompt: Prompt positivo
            negative_prompt: Prompt negativo
            num_inference_steps: Pasos de inference
            guidance_scale: Escala de guidance
            strength: Fuerza del img2img/denoise (se usara valor alto para aplicar cambios)
            seed: Semilla aleatoria
            face_preserve: Si True, hace face swap despues de generar
            use_ipadapter: Si True, usa IP-Adapter para mantener identidad completa
            use_controlnet: Si True, usa ControlNet para mantener estructura
            controlnet_strength: Fuerza del ControlNet (0.0-1.0)
            ipadapter_strength: Fuerza del IP-Adapter (0.0-1.0)
            use_flux: Si True, intenta usar FLUX Fill Pipeline primero (más rápido, mejor calidad)
        
        Returns:
            Tuple con (imagen resultado o None, mensaje estado)
        """
        original_image = None
        
        try:
            # 1. INTENTAR FLUX PRIMERO (si está disponible y se solicita)
            if use_flux:
                print("[ImgEditor] === INTENTANDO FLUX COMO PRIMARIO ===")
                if self._init_flux_client():
                    # Preparar imagen
                    if isinstance(image, Image.Image):
                        original_image = image.copy()
                    elif hasattr(image, 'name'):
                        original_image = Image.open(image.name).copy()
                        image = original_image
                    elif isinstance(image, str) and os.path.exists(image):
                        original_image = Image.open(image).copy()
                        image = original_image
                    
                    # Generar con FLUX
                    result, msg = self.flux_client.generate(
                        image=image,
                        prompt=prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        seed=seed,
                    )
                    
                    if result and result.image:
                        print(f"[ImgEditor] ✅ FLUX exitoso: {msg}")
                        final_image = result.image
                        
                        # PASADA 2: Restaurar cara si se solicita
                        if face_preserve and original_image is not None:
                            print("[ImgEditor] PASADA 2 (FLUX): Restaurando cara original...")
                            final_image = self._restore_face(original_image, final_image)
                        
                        return final_image, f"FLUX{msg}"
                    else:
                        print(f"[ImgEditor] ⚠️ FLUX falló: {msg}, cayendo a ComfyUI...")
                else:
                    print("[ImgEditor] ℹ️ FLUX no disponible, usando ComfyUI")
            
            # 2. FALLBACK A COMFYUI (SD 1.5 + ControlNet/IP-Adapter)
            print("[ImgEditor] === USANDO COMFYUI WORKFLOWS ===")
            client = self._get_client()
            
            # Verificar ComfyUI
            from roop.comfy_client import check_comfy_available, disable_safety_checker
            if not check_comfy_available():
                return None, "Error: ComfyUI no esta corriendo"
            
            # Intentar desactivar safety checker de ComfyUI
            print("[ImgEditor] Intentando desactivar safety checker de ComfyUI...")
            success = disable_safety_checker()
            if not success:
                print("[ImgEditor] WARN: No se pudo desactivar safety checker via API")
            
            # Verificar checkpoints
            checkpoints = client.get_checkpoints()
            if not checkpoints:
                return None, "Error: No hay checkpoints disponibles en ComfyUI"
            
            # Reducir tamaño de imagen si es muy grande (para evitar problemas de VRAM)
            max_size = 1024  # Máximo 1024px para evitar problemas de VRAM
            if original_image is not None:
                img_width, img_height = original_image.size
                if img_width > max_size or img_height > max_size:
                    scale = min(max_size / img_width, max_size / img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    print(f"[ImgEditor] Reduciendo imagen de {img_width}x{img_height} a {new_width}x{new_height}")
                    original_image = original_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Detectar contenido adulto
            is_adult_content = any(keyword in prompt.lower() for keyword in 
                ["nude", "desnuda", "naked", "adult", "explicit", "topless", "nsfw"])
            
            # PASADA 1: Usar el strength del usuario
            final_strength = strength
            final_steps = num_inference_steps
            final_guidance = guidance_scale
            
            # Si es contenido adulto, optimizar parametros
            if is_adult_content:
                print("[ImgEditor] Contenido adulto detectado - optimizando parametros")
                final_steps = max(final_steps, 30)
                final_guidance = max(final_guidance, 9.0)
            
            # Guardar imagen temporal
            temp_image_path = None
            
            if isinstance(image, Image.Image):
                original_image = image.copy()
                temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_image_path = temp_image.name
                image.save(temp_image_path)
            elif hasattr(image, 'name'):
                temp_image_path = image.name
                original_image = Image.open(temp_image_path).copy()
            elif isinstance(image, str) and os.path.exists(image):
                temp_image_path = image
                original_image = Image.open(temp_image_path).copy()
            else:
                return None, "Error: Imagen no valida"
            
            if not os.path.exists(temp_image_path):
                return None, "Error: No se pudo guardar la imagen temporal"
            
            # Subir imagen a ComfyUI
            image_filename = client.upload_image(temp_image_path)
            if not image_filename:
                return None, "Error: No se pudo subir la imagen a ComfyUI"
            
            # Importar workflows
            from roop.img_editor.comfy_workflows import (
                build_img2img_workflow,
                build_inpaint_workflow,
                build_editor_workflow,
                get_available_checkpoints,
                get_default_checkpoint,
                check_controlnet_available,
                check_ipadapter_available
            )
            
            # Obtener checkpoint
            checkpoint = get_default_checkpoint()
            if checkpoint is None:
                return None, "Error: No hay checkpoints disponibles"
            
            # Verificar si ControlNet/IP-Adapter están disponibles
            has_controlnet = check_controlnet_available()
            has_ipadapter = check_ipadapter_available()
            
            # PASADA 1: Generar con el mejor workflow disponible
            print(f"[ImgEditor] PASADA 1: Generando (strength={final_strength})")
            
            should_use_ipadapter = use_ipadapter and has_ipadapter
            should_use_controlnet = use_controlnet and has_controlnet
            
            # Usar workflow de editor real si el usuario quiere ControlNet o IP-Adapter
            if should_use_controlnet or should_use_ipadapter:
                print(f"[ImgEditor] Modo: Editor Real con adaptadores")
                workflow = build_editor_workflow(
                    image_filename=image_filename,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed if seed is not None else int(time.time()),
                    steps=final_steps,
                    cfg=final_guidance,
                    denoise=final_strength,
                    checkpoint=checkpoint,
                    use_controlnet=should_use_controlnet,
                    use_ipadapter=should_use_ipadapter,
                    controlnet_strength=controlnet_strength,
                    ipadapter_strength=ipadapter_strength
                )
            else:
                # Modo simple: img2img sin adaptadores
                print(f"[ImgEditor] Modo: Img2Img simple (steps={final_steps}, cfg={final_guidance})")
                workflow = build_img2img_workflow(
                    image_filename=image_filename,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed if seed is not None else int(time.time()),
                    steps=final_steps,
                    cfg=final_guidance,
                    denoise=final_strength,
                    checkpoint=checkpoint
                )
            
            prompt_id, success, error = client.queue_prompt(workflow)
            if not success:
                print(f"[ImgEditor] Error en queue_prompt: {error}")
                return None, f"Error: {error}"
            
            print(f"[ImgEditor] Prompt encolado: {prompt_id[:8]}...")
            
            # Esperar resultado
            time.sleep(2)
            print(f"[ImgEditor] Generando... ID: {prompt_id[:8]}...")
            images = client.get_images(prompt_id, "*")
            
            if not images:
                return None, "Error: No se pudo obtener la imagen de ComfyUI"
            
            # Convertir a PIL Image
            from io import BytesIO
            generated_image = Image.open(BytesIO(images[0]))
            
            if generated_image is None:
                return None, "Error: Image.open devolvio None"
            
            # PASADA 2: Restaurar cara original si face_preserve esta activado
            if face_preserve and original_image is not None:
                print("[ImgEditor] PASADA 2: Restaurando cara original...")
                final_image = self._restore_face(original_image, generated_image)
            else:
                final_image = generated_image
            
            # Limpiar temporal
            try:
                os.remove(temp_image_path)
            except:
                pass
            
            # Determinar modo usado
            if should_use_controlnet or should_use_ipadapter:
                mode_str = "Editor Real"
                if should_use_controlnet:
                    mode_str += " + ControlNet"
                if should_use_ipadapter:
                    mode_str += " + IPAdapter"
            else:
                mode_str = "Img2Img"
            
            face_str = " + FaceRestore" if face_preserve else ""
            print(f"[ImgEditor] OK {mode_str}{face_str} lista: {final_image.size}")
            return final_image, f"{mode_str}{face_str} generada correctamente"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"
    
    def generate_selective(
        self,
        image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
        strength: float = 0.9,
        seed: int = None,
        face_preserve: bool = True,
        auto_detect_clothing: bool = True,
        mask_threshold: float = 0.5,
        mask_dilation: int = 6,
        exclude_skin: bool = True,
        use_flux: bool = True
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Genera inpaint SELECTIVO usando detección automática de ropa con CLIPSeg.
        
        Este método:
        1. Detecta automáticamente las áreas de ropa con CLIPSeg
        2. Genera una máscara de esas áreas
        3. Aplica inpaint SOLO en la ropa detectada
        4. Restaura la cara original si face_preserve está activado
        
        Args:
            image: Imagen de entrada
            prompt: Prompt para la edición (ej: "nude woman, natural body")
            negative_prompt: Prompt negativo
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guidance
            strength: Fuerza del inpaint (0.7-1.0 recomendado)
            seed: Semilla aleatoria
            face_preserve: Si True, restaura la cara original
            auto_detect_clothing: Si True, detecta ropa automáticamente
            mask_threshold: Umbral para la máscara (0.3-0.7)
            mask_dilation: Píxeles a expandir la máscara (0-30)
            exclude_skin: Si True, excluye áreas de piel de la máscara
            use_flux: Si True, intenta usar FLUX primero
            
        Returns:
            Tuple con (imagen resultado o None, mensaje estado)
        """
        original_image = None
        mask_image = None
        
        try:
            # Preparar imagen
            if isinstance(image, Image.Image):
                original_image = image.copy()
            elif hasattr(image, 'name'):
                original_image = Image.open(image.name).copy()
                image = original_image
            elif isinstance(image, str) and os.path.exists(image):
                original_image = Image.open(image).copy()
                image = original_image
            else:
                return None, "Error: Imagen no válida"
            
            print(f"[ImgEditor] === INPAINT SELECTIVO ===")
            print(f"[ImgEditor] Imagen: {original_image.size}")
            
            # 1. DETECTAR ROPA CON CLIPSEG
            if auto_detect_clothing:
                print("[ImgEditor] Detectando ropa con CLIPSeg...")
                
                if not is_clipseg_available():
                    return None, "Error: CLIPSeg no disponible. Instala: pip install transformers"
                
                segmenter = get_clothing_segmenter()
                success, msg = segmenter.load()
                if not success:
                    return None, f"Error cargando CLIPSeg: {msg}"
                
                # Generar máscara de ropa
                mask_image, mask_array = segmenter.segment_clothing(
                    image=original_image,
                    threshold=mask_threshold,
                    combine_mode="max",
                    include_skin_exclusion=exclude_skin,
                    dilation=mask_dilation
                )
                
                # Verificar que la máscara no esté vacía
                mask_pixels = mask_array.sum() / 255
                total_pixels = mask_array.shape[0] * mask_array.shape[1]
                mask_coverage = mask_pixels / total_pixels
                
                print(f"[ImgEditor] Máscara: {mask_pixels:.0f} píxeles ({mask_coverage*100:.1f}% de la imagen)")
                
                if mask_coverage < 0.01:
                    print("[ImgEditor] ⚠️ Máscara muy pequeña, usando img2img normal")
                    return self.generate(
                        image=original_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        seed=seed,
                        face_preserve=face_preserve,
                        use_ipadapter=False,
                        use_controlnet=False,
                        use_flux=use_flux
                    )
            else:
                # Sin detección automática, usar img2img normal
                return self.generate(
                    image=original_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed,
                    face_preserve=face_preserve,
                    use_ipadapter=False,
                    use_controlnet=False,
                    use_flux=use_flux
                )
            
            # 2. APLICAR INPAINT SELECTIVO
            # Para inpaint, necesitamos un denoise más alto
            inpaint_denoise = max(0.85, strength)  # Mínimo 0.85 para inpaint efectivo
            
            # Mejorar el prompt para inpaint de cuerpo
            if auto_detect_clothing:
                # Añadir términos de calidad si no están presentes
                quality_terms = ["detailed", "realistic", "natural skin", "high quality"]
                prompt_lower = prompt.lower()
                for term in quality_terms:
                    if term not in prompt_lower:
                        prompt = f"{prompt}, {term}"
            
            # Intentar con FLUX Inpaint primero
            if use_flux and self._init_flux_client():
                print(f"[ImgEditor] Usando FLUX/SD Inpaint (denoise={inpaint_denoise})...")
                
                result, msg = self.flux_client.generate(
                    image=original_image,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed
                )
                
                if result and result.image:
                    print(f"[ImgEditor] ✅ Inpaint exitoso: {msg}")
                    final_image = result.image
                    
                    # Restaurar cara si está activado
                    if face_preserve:
                        print("[ImgEditor] Restaurando cara original...")
                        final_image = self._restore_face(original_image, final_image)
                    
                    return final_image, f"Inpaint Selectivo ({mask_coverage*100:.1f}% modificado)"
                else:
                    print(f"[ImgEditor] ⚠️ Inpaint falló: {msg}")
            
            # 3. FALLBACK A COMFYUI INPAINT
            print("[ImgEditor] Usando ComfyUI Inpaint...")
            
            from roop.comfy_client import check_comfy_available, disable_safety_checker
            if not check_comfy_available():
                return None, "Error: ComfyUI no está corriendo"
            
            disable_safety_checker()
            
            client = self._get_client()
            
            # Reducir tamaño de imagen si es muy grande (para evitar problemas de VRAM)
            max_size = 1024  # Máximo 1024px para evitar problemas de VRAM
            img_width, img_height = original_image.size
            if img_width > max_size or img_height > max_size:
                scale = min(max_size / img_width, max_size / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                print(f"[ImgEditor] Reduciendo imagen de {img_width}x{img_height} a {new_width}x{new_height}")
                original_image = original_image.resize((new_width, new_height), Image.LANCZOS)
                mask_image = mask_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Guardar imagen y máscara temporales
            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, "img_editor_input.png")
            temp_mask_path = os.path.join(temp_dir, "img_editor_mask.png")
            
            original_image.save(temp_image_path)
            mask_image.save(temp_mask_path)
            
            # Subir imagen y máscara a ComfyUI
            image_filename = client.upload_image(temp_image_path)
            mask_filename = client.upload_image(temp_mask_path)
            
            if not image_filename or not mask_filename:
                return None, "Error: No se pudieron subir las imágenes a ComfyUI"
            
            # Construir workflow de inpaint
            from roop.img_editor.comfy_workflows import get_default_checkpoint
            
            checkpoint = get_default_checkpoint()
            if checkpoint is None:
                return None, "Error: No hay checkpoints disponibles"
            
            print(f"[ImgEditor] Usando denoise={inpaint_denoise} para inpaint efectivo")
            
            # Usar workflow de inpaint con máscara
            workflow = self._build_inpaint_mask_workflow(
                image_filename=image_filename,
                mask_filename=mask_filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed if seed is not None else int(time.time()),
                steps=num_inference_steps,
                cfg=guidance_scale,
                denoise=strength,  # Usar el denoise del usuario
                checkpoint=checkpoint,
                mask_dilation=mask_dilation  # Pasar dilatación
            )
            
            prompt_id, success, error = client.queue_prompt(workflow)
            if not success:
                return None, f"Error: {error}"
            
            print(f"[ImgEditor] Generando... ID: {prompt_id[:8]}...")
            time.sleep(2)
            images = client.get_images(prompt_id, "*")
            
            if not images:
                return None, "Error: No se pudo obtener la imagen de ComfyUI"
            
            from io import BytesIO
            generated_image = Image.open(BytesIO(images[0]))
            
            # Restaurar cara
            if face_preserve:
                print("[ImgEditor] Restaurando cara original...")
                final_image = self._restore_face(original_image, generated_image)
            else:
                final_image = generated_image
            
            # Limpiar temporales
            try:
                os.remove(temp_image_path)
                os.remove(temp_mask_path)
            except:
                pass
            
            return final_image, f"Inpaint Selectivo ComfyUI ({mask_coverage*100:.1f}% modificado)"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"
    
    def _build_inpaint_mask_workflow(
        self,
        image_filename: str,
        mask_filename: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        denoise: float,
        checkpoint: str,
        mask_dilation: int = 6
    ) -> dict:
        """Construye un workflow de inpaint con máscara para ComfyUI.
        
        Args:
            mask_dilation: Píxeles a expandir la máscara (se pasa a grow_mask_by)
        """
        
        final_negative = "low quality, blurry, distorted, bad anatomy, ugly, deformed, child, underage, minor"
        if negative_prompt:
            final_negative += f", {negative_prompt}"
        
        return {
            "1": {
                "inputs": {"image": image_filename, "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "LoadImage"}
            },
            "2": {
                "inputs": {"image": mask_filename, "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "LoadMask"}
            },
            "3": {
                # Convertir imagen de máscara a tipo MASK
                "inputs": {
                    "image": ["2", 0],
                    "channel": "red"  # Usar canal rojo (la máscara es blanco/negro)
                },
                "class_type": "ImageToMask",
                "_meta": {"title": "ImageToMask"}
            },
            "4": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": f"Checkpoint ({checkpoint})"}
            },
            "5": {
                "inputs": {"clip": ["4", 1], "text": prompt},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "6": {
                "inputs": {"clip": ["4", 1], "text": final_negative},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            "7": {
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 2],
                    "mask": ["3", 0],  # Usar la máscara convertida
                    "grow_mask_by": mask_dilation  # Usar el parámetro de dilatación
                },
                "class_type": "VAEEncodeForInpaint",
                "_meta": {"title": "VAEEncodeForInpaint"}
            },
            "8": {
                "inputs": {
                    "model": ["4", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["7", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": denoise
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "9": {
                "inputs": {"vae": ["4", 2], "samples": ["8", 0]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAEDecode"}
            },
            "10": {
                "inputs": {
                    "filename_prefix": "inpaint_selective",
                    "images": ["9", 0],
                    "format": "png"
                },
                "class_type": "SaveImage",
                "_meta": {"title": "SaveImage"}
            }
        }
    
    def preview_clothing_mask(
        self,
        image: Image.Image,
        threshold: float = 0.5
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Genera una vista previa de la máscara de ropa detectada.
        
        Útil para verificar qué áreas se modificarán antes de generar.
        
        Returns:
            Tuple de (imagen con máscara superpuesta, mensaje)
        """
        try:
            if not is_clipseg_available():
                return None, "CLIPSeg no disponible. Instala: pip install transformers"
            
            segmenter = get_clothing_segmenter()
            success, msg = segmenter.load()
            if not success:
                return None, f"Error cargando CLIPSeg: {msg}"
            
            # Generar máscara
            mask_image, mask_array = segmenter.segment_clothing(
                image=image,
                threshold=threshold,
                combine_mode="max",
                include_skin_exclusion=True
            )
            
            # Crear visualización
            preview = segmenter.visualize_mask(
                image=image,
                mask=mask_image,
                color=(255, 0, 0),  # Rojo
                alpha=0.5
            )
            
            mask_pixels = mask_array.sum() / 255
            total_pixels = mask_array.shape[0] * mask_array.shape[1]
            coverage = mask_pixels / total_pixels * 100
            
            return preview, f"Ropa detectada: {coverage:.1f}% de la imagen"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def _check_models_available(self):
        """Verifica qué modelos están disponibles"""
        models = {
            "FLUX": False,
            "ComfyUI": False,
            "ControlNet": False,
            "OpenPose": False,
            "Tile": False,
            "Depth": False,
            "Canny": False,
            "IP-Adapter": False,
            "CLIPSeg": False,
            "PromptRewriter": False
        }

        # Verificar FLUX
        try:
            if self.flux_client is None:
                self._init_flux_client()
            models["FLUX"] = is_flux_loaded()
        except:
            pass

        # Verificar ComfyUI
        try:
            from roop.comfy_client import check_comfy_available
            models["ComfyUI"] = check_comfy_available()
        except:
            pass

        # Verificar ControlNet y tipos específicos
        try:
            if self.controlnet_utils is None:
                self.controlnet_utils = get_controlnet_utils()
            controlnet_status = self.controlnet_utils.check_controlnet_available()
            models["ControlNet"] = any(controlnet_status.values())
            models["OpenPose"] = controlnet_status.get("openpose", False)
            models["Tile"] = controlnet_status.get("tile", False)
            models["Depth"] = controlnet_status.get("depth", False)
            models["Canny"] = controlnet_status.get("canny", False)
        except:
            pass

        # Verificar IP-Adapter
        try:
            from roop.img_editor.comfy_workflows import check_ipadapter_available
            models["IP-Adapter"] = check_ipadapter_available()
        except:
            pass

        # Verificar CLIPSeg
        try:
            models["CLIPSeg"] = is_clipseg_available()
        except:
            pass

        # Verificar Prompt Rewriter (Ollama)
        try:
            if self.prompt_rewriter is None:
                self.prompt_rewriter = get_prompt_rewriter()
            models["PromptRewriter"] = self.prompt_rewriter.check_availability()
        except:
            pass

        return models


def is_flux_loaded() -> bool:
    """Verifica si FLUX está cargado y disponible"""
    global _manager
    if _manager is None:
        return False
    
    if _manager.flux_client is None:
        return False
    
    return _manager.flux_client._loaded


# Instancia global
_manager = None

def get_img_editor_manager() -> ImgEditorManager:
    """Obtiene la instancia global del manager"""
    global _manager
    if _manager is None:
        _manager = ImgEditorManager()
    return _manager
