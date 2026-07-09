# -*- coding: utf-8 -*-
import os
from typing import Dict, List, Tuple

# Imports pesados SOLO se hacen cuando se instancia la versión full.
# Así importar este módulo para el Light no satura ni requiere las librerías pesadas.
SentenceTransformer = None
util = None
torch = None

def _resolve_semantic_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Resuelve el modelo de forma lo más local posible.
    - Prefiere copias locales en models/nlp/
    - Fuerza local_files_only=True cuando es posible para evitar pings a huggingface.
    - Respeta HF_HUB_OFFLINE=1 si lo pones.
    - Primera ejecución aún puede necesitar internet para descargar (~90MB).
    """
    # Si el usuario quiere modo completamente offline
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        # Buscar en cache local obligatoriamente
        hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2")
        if os.path.isdir(hf_cache):
            print("[NLP] Modo HF_HUB_OFFLINE activado. Usando solo cache local.")
            return model_name, True
        else:
            print("[NLP] HF_HUB_OFFLINE=1 pero el modelo no está cacheado todavía. Necesitas internet una vez.")

    # 1. Buscar copia dentro del proyecto (recomendado para 100% local después de descargar)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    local_project = os.path.join(project_root, "models", "nlp", "all-MiniLM-L6-v2")
    if os.path.isdir(local_project):
        print(f"[NLP] Usando copia local del proyecto: {local_project}")
        return local_project, True

    # 2. Detectar si ya está en la cache de huggingface
    hf_cache = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2")
    if os.path.isdir(hf_cache):
        return model_name, True   # intentamos con local_files_only=True

    # Primera ejecución: permitimos que descargue
    return model_name, False


class SemanticIntentAnalyzer:
    """
    Analizador de intenciones basado en embeddings semánticos.
    Cero hardcoding de palabras clave, utiliza distancias matemáticas en espacio vectorial.

    NOTA IMPORTANTE SOBRE "LOCAL":
    - La inferencia corre 100% en tu CPU (local).
    - El modelo se descarga de HuggingFace solo la primera vez (~90MB).
    - Después intentamos cargarlo completamente en modo offline (sin pings a internet).
    - Si ves muchas conexiones a huggingface, es porque aún no está cacheado o se está recreando el analizador.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[NLP] Cargando modelo semántico {model_name} en CPU...")
        # Importar pesado solo aquí
        global SentenceTransformer, util, torch
        from sentence_transformers import SentenceTransformer as ST, util as U
        import torch as T
        SentenceTransformer = ST
        util = U
        torch = T

        # Forzar CPU para ahorrar VRAM (ComfyUI ya usa mucha)
        self.device = "cpu"

        model_ref, force_local = _resolve_semantic_model(model_name)
        load_kwargs = {"device": self.device}

        if force_local:
            load_kwargs["local_files_only"] = True

        try:
            self.model = SentenceTransformer(model_ref, **load_kwargs)
        except Exception as e:
            if force_local:
                print(f"[NLP] Error cargando en modo local_files_only: {e}")
                print("[NLP] Probablemente el modelo no está completamente cacheado todavía.")
                print("      Ejecuta una vez con internet para descargarlo, luego pon HF_HUB_OFFLINE=1")
            # Último intento sin forzar local (solo si no estamos en offline forzado)
            if os.environ.get("HF_HUB_OFFLINE") != "1":
                self.model = SentenceTransformer(model_name, device=self.device)
            else:
                raise RuntimeError("Modo offline activado (HF_HUB_OFFLINE=1) pero el modelo no está disponible localmente.") from e
        
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
                "removing clothes, getting naked or nude, barefoot, exposed body, undressed, sin ropa, desnuda",
                "improving quality, sharpness, detail, resolution, clarity, enhance quality",
                "mejorar calidad, nitidez, detalle, sharper, mejorar color, realzar calidad",
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
        if not prompt or not prompt.strip():
            return 0.5
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True, device=self.device)
        similarities = {}
        for intent, embeddings in self.anchor_embeddings.items():
            cos_sim = util.cos_sim(prompt_embedding, embeddings)
            similarities[intent] = float(torch.max(cos_sim))
        magnitude = 0.35
        pose_weight = similarities.get("pose", 0)
        structural_weight = similarities.get("structural", 0)
        attribute_weight = similarities.get("attribute", 0)
        impact_weights = {
            "pose": pose_weight * 0.7,
            "structural": structural_weight * 0.7,
            "attribute": attribute_weight * 0.5
        }
        best_intent = max(impact_weights, key=impact_weights.get)
        max_impact = impact_weights[best_intent]
        print(f"[NLP] Similitudes: Pose={pose_weight:.2f}, Struct={structural_weight:.2f}, Attr={attribute_weight:.2f}")
        if max_impact > 0.06:
            magnitude += max_impact
        return max(0.40, min(0.95, magnitude))

    def get_body_transform_intensity(self, prompt: str) -> float:
        if not prompt or not prompt.strip():
            return 0.0
        prompt_embedding = self.model.encode(prompt, convert_to_tensor=True, device=self.device)
        body_descs = [
            "removing clothes, getting naked, nude, undressed",
            "body exposure, bare skin, nudity, desnuda, sin ropa",
            "breasts, genitals, torso, intimate body parts exposed",
            "taking off clothes, stripping, completely naked",
        ]
        body_embeddings = self.model.encode(body_descs, convert_to_tensor=True, device=self.device)
        body_sim = float(torch.max(util.cos_sim(prompt_embedding, body_embeddings)))
        attr_sim = float(torch.max(util.cos_sim(prompt_embedding, self.anchor_embeddings["attribute"])))
        return min(1.0, body_sim * 0.75 + attr_sim * 0.25)

    def detect_target(self, prompt: str) -> str:
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
        pose_sim = float(torch.max(util.cos_sim(prompt_embedding, self.anchor_embeddings["pose"])))
        if pose_sim > 0.6 and best_target == "face":
            best_target = "subject"
        return best_target if max_sim > 0.25 else "subject"


# Singleton global para evitar recargas y pings repetidos a huggingface
_global_semantic_analyzer = None

def get_semantic_analyzer(full_ai=False):
    global _global_semantic_analyzer
    if full_ai:
        if _global_semantic_analyzer is None or not hasattr(_global_semantic_analyzer, "model"):
            _global_semantic_analyzer = SemanticIntentAnalyzer()
        return _global_semantic_analyzer
    else:
        return LightLocalIntentAnalyzer()


class LightLocalIntentAnalyzer:
    """
    Versión 100% local y ultraligera (sin torch, sin sentence-transformers, sin red).
    Usa solapamiento simple de palabras con las mismas anclas que la versión completa.
    Cero saturación de PC, cero descargas, cero conexiones externas.
    Ideal para "no sature mi PC" + "todo en local".
    """

    def __init__(self):
        print("[NLP] Usando analizador local ligero (sin modelo externo, sin red).")
        self.device = "cpu"  # solo por compatibilidad

        # Mismas anclas que la versión full (para consistencia)
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
                "removing clothes, getting naked or nude, barefoot, exposed body, undressed, sin ropa, desnuda",
                "improving quality, sharpness, detail, resolution, clarity, enhance quality",
                "mejorar calidad, nitidez, detalle, sharper, mejorar color, realzar calidad",
                "cambiar color, textura o iluminación",
                "modificar ropa, pelo u ojos",
                "ajustar brillo, filtros o estilo",
                "retoques faciales sutiles"
            ]
        }

    def _score(self, prompt: str, texts: list) -> float:
        """Similitud por solapamiento de palabras. Efectivo y sin hardcodes de usuario (usa solo anclas)."""
        p_words = set(w for w in prompt.lower().split() if len(w) > 2)
        if not p_words:
            return 0.0
        best = 0.0
        for t in texts:
            t_words = set(w for w in t.lower().split() if len(w) > 2)
            if t_words:
                overlap = len(p_words & t_words) / max(len(t_words), 1)
                # Bonus por coincidencia parcial (ej. "ropa" en "modificar ropa") sin listas extra
                for pw in p_words:
                    for tw in t_words:
                        if pw in tw or tw in pw:
                            overlap += 0.1
                            break
                best = max(best, overlap)
        return min(1.0, best)

    def get_magnitude(self, prompt: str) -> float:
        if not prompt or not prompt.strip():
            return 0.55  # valor razonable por defecto para Imagine style

        prompt_l = prompt.lower()
        pose = self._score(prompt_l, self.anchors["pose"])
        structural = self._score(prompt_l, self.anchors["structural"])
        attribute = self._score(prompt_l, self.anchors["attribute"])

        # Similar a la versión full pero sin embeddings
        magnitude = 0.50  # base decente para Imagine (permite cambios visibles sin saturar)
        impact = max(pose * 0.8, structural * 0.75, attribute * 0.7)
        if impact > 0.04:
            magnitude += impact * 0.9

        # Si hay señal de cambio de atributo/estructura (ropa, cuerpo, fondo), asegurar mag suficiente
        if attribute > 0.06 or structural > 0.06:
            magnitude = max(magnitude, 0.68)

        return max(0.48, min(0.92, magnitude))  # permite edits fuertes manteniendo foto

    def get_body_transform_intensity(self, prompt: str) -> float:
        if not prompt or not prompt.strip():
            return 0.0
        prompt_l = prompt.lower()
        body_anchors = [
            "naked", "nude", "undress", "desnuda", "sin ropa",
            "bare skin", "barefoot", "bare chested", "topless",
            "remove clothes", "take off clothes", "sin ropa",
        ]
        body = self._score(prompt_l, body_anchors)
        body = max(body, self._score(prompt_l, ["naked", "nude", "desnuda", "sin ropa"])) * 0.85
        body += self._score(prompt_l, ["body", "cuerpo", "skin", "piel", "breasts", "tits", "pussy", "vagina", "penis", "ass"]) * 0.15
        return min(1.0, body)

    def detect_target(self, prompt: str) -> str:
        prompt_l = prompt.lower()
        scores = {}
        for target, descs in {
            "background": self.anchors["structural"][:2] + ["background", "fondo"],
            "face": ["face", "cara", "rostro", "eyes", "ojos"],
            "clothes": ["clothes", "ropa", "dress", "outfit", "vestido", "naked", "nude", "desnuda", "sin ropa", "bare skin"],
            "hair": ["hair", "pelo", "cabello"],
            "subject": ["person", "people", "character", "body", "figura", "sujeto", "desnuda", "naked", "undress", "barefoot"]
        }.items():
            scores[target] = self._score(prompt_l, descs)

        best = max(scores, key=scores.get)
        if scores[best] < 0.05:
            best = "subject"

        # For body/clothing/undress (high attribute score), prefer subject/clothes over background
        if self._score(prompt_l, self.anchors.get("attribute", [])) > 0.08 and best == "background":
            best = "subject"

        # Si parece cambio de pose/acción, preferimos subject global
        if self._score(prompt_l, self.anchors["pose"]) > 0.08:
            best = "subject"

        return best

    STRUCTURAL_SIGNAL_THRESHOLD = 0.06

    def get_axis_scores(self, prompt: str) -> Dict[str, float]:
        if not prompt or not prompt.strip():
            return {"pose": 0.0, "structural": 0.0, "attribute": 0.0}
        prompt_l = prompt.lower()
        return {
            "pose": self._score(prompt_l, self.anchors["pose"]),
            "structural": self._score(prompt_l, self.anchors["structural"]),
            "attribute": self._score(prompt_l, self.anchors["attribute"]),
        }

    def is_structural_dominant(self, prompt: str) -> bool:
        scores = self.get_axis_scores(prompt)
        return scores.get("structural", 0.0) > self.STRUCTURAL_SIGNAL_THRESHOLD

    def get_structural_bias(self, prompt: str) -> str:
        prompt_l = (prompt or "").lower()
        remove_score = self._score(
            prompt_l,
            self.anchors["structural"][3:4]
            + ["remove", "quitar", "eliminar", "borrar", "sin personas"],
        )
        add_score = self._score(
            prompt_l,
            self.anchors["structural"][:2]
            + ["add", "añadir", "poner gente", "insertar"],
        )
        if remove_score > add_score and remove_score > 0.06:
            return "remove"
        if add_score > remove_score and add_score > 0.06:
            return "add"
        return "neutral"

    def is_quality_dominant(self, prompt: str) -> bool:
        scores = self.get_axis_scores(prompt)
        quality = scores.get("attribute", 0.0)
        action = max(scores.get("pose", 0.0), scores.get("structural", 0.0))
        return quality > 0.10 and quality > action * 1.15





