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
                "sitting, kneeling, standing, lying down, running, dancing, jumping, waving",
                "repositioning limbs or changing posture, start dancing together",
                "cambiar la pose o posición del cuerpo",
                "sentado, de rodillas, de pie, acostado, corriendo, bailando, bailar, danzando",
                "cambiar la postura o posición de los brazos y piernas, ponerse a bailar",
            ],
            "structural": [
                "adding new people or large objects",
                "adds insert places new person people character",
                "changing the background completely",
                "inserting new elements into the scene",
                "removing people or large objects",
                "remove delete erase persons people from scene",
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
                "improving quality, sharpness, detail, resolution, clarity, enhance quality, deblur, upscale",
                "mejorar calidad, nitidez, detalle, sharper, mejorar color, realzar calidad, desposterizar, ultra realista, hiperrealista",
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
                
        pose_sim = float(torch.max(util.cos_sim(prompt_embedding, self.anchor_embeddings["pose"])))
        if pose_sim > 0.6 and best_target == "face":
            best_target = "subject"
            
        return best_target if max_sim > 0.25 else "subject"


_global_semantic_analyzer = None


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
                "sitting, kneeling, standing, lying down, running, dancing, jumping, waving",
                "repositioning limbs or changing posture, start dancing together",
                "cambiar la pose o posición del cuerpo",
                "sentado, de rodillas, de pie, acostado, corriendo, bailando, bailar, danzando",
                "cambiar la postura o posición de los brazos y piernas, ponerse a bailar",
            ],
            "structural": [
                "adding new people or large objects",
                "adds insert places new person people character",
                "changing the background completely",
                "inserting new elements into the scene",
                "removing people or large objects",
                "remove delete erase persons people from scene",
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
                "improving quality, sharpness, detail, resolution, clarity, enhance quality, deblur, upscale",
                "mejorar calidad, nitidez, detalle, sharper, mejorar color, realzar calidad, desposterizar, ultra realista, hiperrealista",
                "cambiar color, textura o iluminación",
                "modificar ropa, pelo u ojos",
                "ajustar brillo, filtros o estilo",
                "retoques faciales sutiles"
            ]
        }

    _STOPWORDS = frozenset({"the", "and", "for", "are", "but", "not", "you", "all", "can",
                            "had", "her", "was", "one", "our", "out", "has", "how", "its",
                            "may", "now", "old", "see", "way", "who", "did", "got", "let",
                            "say", "she", "too", "use", "any", "per", "que", "del", "las",
                            "los", "con", "por", "una"})

    def _word_set(self, text: str) -> set:
        return {w for w in text.lower().split() if len(w) > 2 and w not in self._STOPWORDS}

    def _score(self, prompt: str, texts: list) -> float:
        """Similitud por solapamiento de palabras sin stopwords. Efectivo y sin hardcodes de usuario (usa solo anclas)."""
        p_words = self._word_set(prompt)
        if not p_words:
            return 0.0
        best = 0.0
        for t in texts:
            t_words = self._word_set(t)
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

    STRUCTURAL_SIGNAL_THRESHOLD = 0.06

    def get_axis_scores(self, prompt: str) -> Dict[str, float]:
        """Scores por eje semántico (anclas fijas). Usado para global en cambios estructurales."""
        if not prompt or not prompt.strip():
            return {"pose": 0.0, "structural": 0.0, "attribute": 0.0}
        prompt_l = prompt.lower()
        return {
            "pose": self._score(prompt_l, self.anchors["pose"]),
            "structural": self._score(prompt_l, self.anchors["structural"]),
            "attribute": self._score(prompt_l, self.anchors["attribute"]),
        }

    def get_speech_intensity(self, prompt: str) -> float:
        """Señal de diálogo/narración a cámara (anclas fijas, no lee palabras sueltas del usuario)."""
        if not prompt or not prompt.strip():
            return 0.0
        prompt_l = prompt.lower()
        speech_anchors = [
            "saying speaking talking to camera dialogue narration voice words phrase",
            "diciendo hablando mira a camara dice algo frase voz narracion habla",
            "tell the audience say something speak to camera lip sync",
            "que diga que digan que hablen que cuenten algo a camara",
        ]
        motion_only = self._score(
            prompt_l,
            [
                "wind blowing hair moving camera cinematic motion",
                "viento sopla cabello movimiento camara",
            ],
        )
        speech = self._score(prompt_l, speech_anchors)
        return max(0.0, min(1.0, speech - motion_only * 0.35))

    def get_body_transform_intensity(self, prompt: str) -> float:
        """Intensidad de cambio corporal/ropa (anclas attribute), excluyendo pedidos solo de calidad."""
        if not prompt or not prompt.strip():
            return 0.0
        prompt_l = prompt.lower()
        axes = self.get_axis_scores(prompt_l)
        body_sub = self._score(
            prompt_l,
            [
                self.anchors["attribute"][1],
                self.anchors["attribute"][4],
                self.anchors["attribute"][8],
            ],
        )
        quality_sub = self._score(
            prompt_l,
            [self.anchors["attribute"][5], self.anchors["attribute"][6]],
        )
        signal = max(0.0, body_sub - quality_sub * 0.7)
        if signal < 0.05:
            return 0.0
        if body_sub <= max(axes["pose"], axes["structural"] * 0.9):
            signal *= 0.55
        return min(1.0, signal)

    def is_structural_dominant(self, prompt: str) -> bool:
        """True si hay señal estructural (gente/fondo/elementos), incluso en prompts compuestos."""
        return self.get_axis_scores(prompt)["structural"] > self.STRUCTURAL_SIGNAL_THRESHOLD

    def get_structural_bias(self, prompt: str) -> str:
        """'add', 'remove' o 'neutral' según sub-anclas estructurales (sin leer palabras del usuario)."""
        if not prompt or not self.is_structural_dominant(prompt):
            return "neutral"
        prompt_l = prompt.lower()
        add_score = self._score(
            prompt_l,
            [
                self.anchors["structural"][0],
                self.anchors["structural"][1],
                self.anchors["structural"][3],
                self.anchors["structural"][6],
                self.anchors["structural"][8],
                "add another person people to the scene photo",
                "añadir anadir agregar incluir otra persona gente en la escena foto",
                "anade agrega introduce una persona alguien en la imagen foto",
                "introduce a new person character into the scene image",
                "place insert a person someone in the image photograph",
                "include add more people persons to this picture",
                "put a person someone next to beside in front of",
                "add a man woman child couple group crowd",
            ],
        )
        remove_score = self._score(
            prompt_l,
            [
                self.anchors["structural"][4],
                self.anchors["structural"][5],
                self.anchors["structural"][9],
                "remove delete erase eliminate person people from photo scene",
                "eliminar quitar borrar persona personas de la foto escena",
                "remove person people from the scene",
                "eliminar persona personas de la escena",
                "delete erase take out all the people in the photo",
                "remove delete the man woman child person people from picture",
                "get rid of remove delete erase all persons everyone",
            ],
        )
        if remove_score >= add_score and remove_score > 0.10:
            return "remove"
        if add_score > remove_score * 1.05:
            return "add"
        if remove_score > add_score * 1.05:
            return "remove"
        return "neutral"

    def get_quality_intensity(self, prompt: str) -> float:
        """Señal de mejora de calidad (nitidez, realismo…) sin cambio de escena/cuerpo."""
        if not prompt or not prompt.strip():
            return 0.0
        prompt_l = prompt.lower()
        quality_sub = self._score(
            prompt_l,
            [self.anchors["attribute"][5], self.anchors["attribute"][6]],
        )
        body_sub = self._score(
            prompt_l,
            [self.anchors["attribute"][1], self.anchors["attribute"][4]],
        )
        return max(0.0, min(1.0, quality_sub - body_sub * 0.65))

    def is_quality_dominant(self, prompt: str) -> bool:
        """True si el prompt pide sobre todo calidad/realismo, no pose/cuerpo/escena."""
        if not prompt or not prompt.strip():
            return False
        axes = self.get_axis_scores(prompt)
        pose = axes["pose"]
        structural = axes["structural"]
        attribute = axes["attribute"]
        quality_int = self.get_quality_intensity(prompt)
        if quality_int < 0.05:
            return False
        if pose >= 0.08 or structural >= 0.06:
            return False
        if self.get_body_transform_intensity(prompt) >= 0.08:
            return False
        return attribute >= max(pose, structural) * 1.1 and quality_int >= 0.06

    def get_magnitude(self, prompt: str) -> float:
        if not prompt or not prompt.strip():
            return 0.55  # valor razonable por defecto para Imagine style

        prompt_l = prompt.lower()
        pose = self._score(prompt_l, self.anchors["pose"])
        structural = self._score(prompt_l, self.anchors["structural"])
        attribute = self._score(prompt_l, self.anchors["attribute"])
        quality_int = self.get_quality_intensity(prompt)

        if self.is_quality_dominant(prompt):
            return max(0.32, min(0.48, 0.34 + quality_int * 0.28))

        # Similar a la versión full pero sin embeddings
        magnitude = 0.50  # base decente para Imagine (permite cambios visibles sin saturar)
        impact = max(pose * 0.8, structural * 0.75, attribute * 0.7)
        if impact > 0.04:
            magnitude += impact * 0.9

        # Cambio corporal/ropa/escena (no solo calidad) → mag alta
        body_sub = self._score(
            prompt_l,
            [self.anchors["attribute"][1], self.anchors["attribute"][4]],
        )
        if (body_sub > 0.05 or structural > 0.06 or pose > 0.06) and attribute > 0.06:
            magnitude = max(magnitude, 0.68)

        return max(0.48, min(0.92, magnitude))  # permite edits fuertes manteniendo foto

    def detect_target(self, prompt: str) -> str:
        prompt_l = prompt.lower()
        scores = {}
        for target, descs in {
            "background": self.anchors["structural"][:2] + [
                "background", "fondo", "entorno", "environment", "escena", "scenery",
            ],
            "face": ["face", "cara", "rostro", "eyes", "ojos"],
            "clothes": ["clothes", "ropa", "dress", "outfit", "vestido", "naked", "nude", "desnuda", "sin ropa", "bare skin"],
            "hair": ["hair", "pelo", "cabello"],
            "subject": ["person", "people", "character", "body", "figura", "sujeto", "desnuda", "naked", "undress", "barefoot"]
        }.items():
            scores[target] = self._score(prompt_l, descs)

        best = max(scores, key=scores.get)
        if scores[best] < 0.05:
            best = "subject"

        # Calidad (hiperrealista, nitidez…) no debe tapar señal clara de fondo/entorno
        if (
            self._score(prompt_l, self.anchors.get("attribute", [])) > 0.08
            and best == "background"
            and scores.get("background", 0) < 0.10
        ):
            best = "subject"

        # Si parece cambio de pose/acción, preferimos subject global
        if self._score(prompt_l, self.anchors["pose"]) > 0.08:
            best = "subject"

        return best


# Fábrica: por defecto usa la versión ligera (local, sin saturar)
# La versión full con embeddings solo si se pide explícitamente
def get_semantic_analyzer(full_ai=False):
    global _global_semantic_analyzer
    if full_ai:
        # Solo carga el modelo pesado si el usuario quiere máxima inteligencia
        if _global_semantic_analyzer is None or not hasattr(_global_semantic_analyzer, "model"):
            _global_semantic_analyzer = SemanticIntentAnalyzer()
        return _global_semantic_analyzer
    else:
        # Ligero por defecto: 100% local, cero red, bajo consumo
        return LightLocalIntentAnalyzer()
