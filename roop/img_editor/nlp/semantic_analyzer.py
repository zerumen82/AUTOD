# -*- coding: utf-8 -*-
"""
Analizador de intención para Image Editor.

Sin hardcoding de prompts de usuario: no hay `if "desnuda"`, ni listas de
verbos/colores del usuario. Solo hay **conceptos** (anclas) y distancias
(embedding o solapamiento ligero). El routing se decide por scores relativos.
"""
import json
import os
import re
import unicodedata
from functools import lru_cache
from typing import Dict

SentenceTransformer = None
util = None
torch = None

# ---------------------------------------------------------------------------
# Espacio de conceptos (descripciones en EN/ES). NO son palabras del usuario.
# El analyzer mide similitud prompt↔concepto y elige el eje dominante.
# ---------------------------------------------------------------------------
# Descripciones de CONCEPTO (idioma natural). No son triggers de usuario:
# el routing compara scores entre conceptos, nunca `if palabra in prompt`.
CONCEPTS: Dict[str, list] = {
    "pose": [
        "change pose posture stance body position limbs",
        "stand sit kneel lie run dance walk turn face camera",
        "move left right up down reposition in frame",
        "cambiar pose postura posición del cuerpo brazos piernas",
        "de pie sentado rodillas acostado bailando corriendo girar mirar cámara",
        "mover izquierda derecha arriba abajo en el encuadre",
        # El matcher de tokens no cruza dance↔dancing («dance» no es
        # subcadena de «dancing»): medido 2026-10-16, "the cat dances"
        # valía 0,41 y "the cat is dancing" 0,000. Texto de concepto, no
        # palabra de usuario (mismo precedente que «paint over/repaint»
        # en attribute_recolor).
        "dance dancing groove sway twirl spin rhythm",
    ],
    "structural": [
        "add insert put new people objects props elements into scene",
        "remove delete erase eliminate people objects from photo",
        "change replace background environment scenery landscape",
        "añadir insertar poner gente objetos accesorios en la escena",
        "quitar eliminar borrar personas objetos de la foto",
        "cambiar reemplazar fondo entorno escenario paisaje playa",
    ],
    "attribute": [
        "change color texture lighting clothes hair eyes style appearance",
        "improve quality sharpness detail resolution clarity enhance",
        "cambiar color textura iluminación ropa pelo ojos estilo apariencia",
        "mejorar calidad nitidez detalle resolución claridad realzar",
    ],
    "structural_add": [
        "add insert put include new element prop accessory hat glasses bag into scene",
        "añadir insertar poner agregar objeto accesorio sombrero gafas bolso en escena",
    ],
    "structural_remove": [
        "remove delete erase eliminate kill kills killing killed elimination quitar eliminar borrar people objects animals from image",
        "quitar eliminar borrar remover matar personas objetos animales de la imagen",
    ],
    "structural_background": [
        "change replace background environment scenery landscape location beach sky",
        "cambiar reemplazar fondo entorno escenario paisaje ubicación playa cielo",
    ],
    "attribute_recolor": [
        "colorize recolor colorea colorear color palette hue tint sepia monochrome grayscale",
        "blanco negro escala grises paleta tinte tono vestido rojo pelo azul cielo",
        "change color of clothing hair sky water dress to another tone",
        "cambiar color de ropa pelo cielo agua vestido a otro tono",
        # v5.1209 — el encuadre de "pintar" también es recolor: sin esta
        # descripción "pinta las paredes de gris" medía 0,149 (< suelo 0,20)
        # e "Paint the walls in grey" 0,000 → ruta global sin ancla (15/10).
        # El ancla lleva la ACCIÓN y el color, nunca el sustantivo de la
        # superficie: "remove the paint from the wall" comparte "paint" y
        # "wall" con el objeto y con 0,432 volteaba la eliminación a recolor.
        "paint over repaint give a coat of paint in one solid color gray grey white black",
        "dar una mano de pintura repintar pintar encima de un color sólido gris blanco negro",
    ],
    "attribute_quality": [
        "improve quality sharpness detail clarity resolution enhance deblur sharper",
        "mejorar calidad nitidez detalle claridad resolución realzar más nítido",
        "more saturated vibrant brighter saturation vibrance brightness lighting exposure",
        "más saturado vibrante brillante saturación brillo luminosidad iluminación",
    ],
    "attribute_body": [
        "nude naked undressed bare skin body exposure remove clothes desnuda sin ropa",
        "quitar ropa exposición corporal piel descubierta desnudo undress strip",
        "thong tanga g-string underwear lingerie swimwear bikini intimate garment exposure",
        "tanga hilo dental ropa interior lencería traje baño biquini prenda íntima escote",
        "barefoot descalzo sin zapatos pies descalzos bare feet no shoes",
        "cambiar ropa outfit vestimenta desvestir cambiarse quitarse poner tanga",
        "genital anatomy penis phallus body proportion size larger bigger smaller",
        "anatomía genital pene falo proporción corporal tamaño",
    ],
    # Dirección del cambio corporal (para negatives sin pelear con el pedido).
    # uncover = quitar/menos ropa; cover = vestir/más ropa (nieve, abrigo...).
    "attribute_uncover": [
        "remove take off undress strip naked bare skin less clothing minimal garment exposure",
        "without clothes no clothes without clothing desnudo sin ropa",
        "quitar quitarse desvestir desnudar piel descubierta menos ropa prenda mínima tanga",
        "thong tanga g-string bikini lingerie underwear swimsuit minimal tiny micro only solo",
        "hilo dental biquini lencería ropa interior bañador mínimo solo únicamente",
    ],
    "body_exposure_full": [
        "fully nude completely naked no clothes at all entire body bare",
        "totalmente desnudo completamente desnuda sin ropa en todo el cuerpo",
    ],
    "body_exposure_local": [
        "barefoot bare feet no shoes remove footwear feet only",
        "descalzo descalza pies desnudos sin zapatos quitar calzado solo pies",
    ],
    "attribute_cover": [
        "dress put on wear coat jacket winter snow warm covered bundled sweater clothes on",
        "vestir poner ponerse abrigo chaqueta nieve invierno ropa abrigada tapado jersey nieve",
    ],
    "attribute_age": [
        "age appear older younger aged elderly juvenile mature wrinkle wrinkle aged skin",
        "appear old aged elderly mature senior wrinkle lines aged skin texture",
        "piel arrugas envejecido mayor edad textura piel arrugas líneas",
        "parecer mayor joven envejecido textura piel arrugas",
    ],
    # v5.1127 — APARIENCIA FÍSICA (calvo/pelo/barba...). Sin esto 'the boy is
    # bald' puntúa ~0 en TODOS los conceptos → mag floor 0.50 → klein EDIT
    # copy-machine (log 12:11: no ejecutó + derivó identidad). Es un concepto
    # de ancla (como las prendas en attribute_body), no un trigger de usuario.
    # Cubre cambios de atributo físico facial/capilar, no edición de piel.
    "attribute_appearance": [
        "bald baldness shaved head no hair hairless scalp visible",
        "calvo calvicie sin pelo cabeza rapada pelada cuero cabelludo",
        "haircut hairstyle long hair short hair curly straight hair beard mustache shaved",
        "corte de pelo peinado pelo largo pelo corto rizado liso barba bigote afeitado",
        "facial features change new look different appearance makeover",
        "cambiar rasgos faciales nuevo look apariencia diferente cambio de imagen",
    ],
    "attribute_skin": [
        "skin texture rough smooth flawless blemish freckle mole scar tattoo",
        "piel textura áspera suave perfecta marca lunar cicatriz tatuaje",
    ],
    "target_background": [
        "background environment scenery landscape surroundings fondo entorno escenario",
    ],
    "target_face": [
        "face facial features head eyes mouth cara rostro cabeza ojos boca",
    ],
    "target_clothes": [
        "clothing clothes outfit dress shirt ropa vestido camisa prendas vestimenta",
    ],
    "target_hair": [
        "hair hairstyle pelo cabello peinado",
    ],
    "target_subject": [
        "person people character body figure subject persona gente personaje cuerpo figura",
    ],
}

# Ejes agregados (unión de sub-conceptos para mag global)
_AXIS_KEYS = {
    "pose": ["pose"],
    "structural": ["structural", "structural_add", "structural_remove", "structural_background"],
    "attribute": [
        "attribute",
        "attribute_recolor",
        "attribute_quality",
        "attribute_body",
        "attribute_age",
        "attribute_skin",
        "attribute_appearance",
    ],
}

# --- Cámaras (v5.1226) ----------------------------------------------------
# La LoRA de ángulos solo produce sus frases entrenadas (8 azimuts ×
# 4 alturas × 3 distancias = 96 poses); la pose se elige midiendo la
# instrucción contra estas descripciones de concepto con el MISMO motor
# _score_texts. Los textos describen conceptos de cámara, no palabras
# trigger de usuario: sin "view", "vista", "camera", "angle" ni "change"
# sueltos (tokens ambiguos que diluirían o secuestrarían la medida, v5.1209).
CAMERA_POSES: Dict[str, Dict[str, list]] = {
    "azimuth": {
        "front view": [
            "pov first person over the shoulder primera persona mirando "
            "por los ojos de alguien sobre el hombro",
            "frente frontal enfrente cara a cara mirando a la persona",
        ],
        "back view": [
            "atras espalda trasera por detras de la persona behind rear",
        ],
        "left side view": [
            "izquierda perfil izquierdo lado izquierdo left side profile lateral",
        ],
        "right side view": [
            "derecha perfil derecho lado derecho right side profile lateral",
        ],
        "front-left quarter view": [
            "diagonal izquierda delante tres cuartos front quarter left",
        ],
        "front-right quarter view": [
            "diagonal derecha delante tres cuartos front quarter right",
        ],
        "back-left quarter view": [
            "atras izquierda diagonal trasera tres cuartos behind left quarter",
        ],
        "back-right quarter view": [
            "atras derecha diagonal trasera tres cuartos behind right quarter",
        ],
    },
    "elevation": {
        "high-angle shot": [
            "overhead bird eye from above top down cenital picado arriba",
        ],
        "elevated shot": [
            "elevated elevado raised mas alto que los ojos higher",
        ],
        "eye-level shot": [
            "eye level altura de los ojos nivel de la mirada",
        ],
        "low-angle shot": [
            "contrapicado abajo bajo por debajo upwards looking under",
        ],
    },
    "distance": {
        "close-up": [
            "primer plano muy cerca tight crop detalle de la cara cerca",
        ],
        "medium shot": [
            "plano medio media distancia medio cuerpo",
        ],
        "wide shot": [
            "plano general cuerpo entero escena completa lejos alejado wide",
            "full body entire scene distant framing",
        ],
    },
}

@lru_cache(maxsize=1)
def camera_pose_config() -> dict:
    """Camera measurement policy, independent of image models and GPU state."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    with open(os.path.join(root, "config", "camera_pose.json"), encoding="utf-8") as stream:
        config = json.load(stream)
    for key in ("signal_threshold", "shared_candidate_fraction"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
            raise ValueError(f"Invalid camera pose configuration: {key}={value!r}")
    return config

_camera_scorer = None


def score_camera_pose(prompt: str) -> Dict[str, Dict[str, float]]:
    """(frase entrenada -> score) por eje de cámara, medido prompt↔ancla."""
    global _camera_scorer
    if _camera_scorer is None:
        _camera_scorer = LightLocalIntentAnalyzer()
    if not prompt or not prompt.strip():
        return {axis: {phrase: 0.0 for phrase in candidates}
                for axis, candidates in CAMERA_POSES.items()}
    # Compare the actual trained phrases as well as their concept descriptions.
    # Shared phrase vocabulary carries no axis choice (e.g. view, shot, angle).
    # Derive it from the candidates instead of maintaining a trigger word list.
    def normalize(text):
        return "".join(c for c in unicodedata.normalize("NFKD", text)
                       if not unicodedata.combining(c))

    normalized_prompt = normalize(prompt)
    fraction = camera_pose_config()["shared_candidate_fraction"]
    scores = {}
    for axis, candidates in CAMERA_POSES.items():
        tokens = {phrase: _camera_scorer._tokenize(normalize(phrase))
                  for phrase in candidates}
        vocabulary = set().union(*tokens.values())
        shared = {token for token in vocabulary
                  if sum(token in words for words in tokens.values()) / len(tokens) >= fraction}
        scores[axis] = {}
        for phrase, anchors in candidates.items():
            distinctive_phrase = " ".join(sorted(tokens[phrase] - shared))
            scores[axis][phrase] = _camera_scorer._score_texts(
                normalized_prompt, [normalize(anchor) for anchor in anchors] + [distinctive_phrase])
    return scores


def _resolve_semantic_model(model_name: str = "all-MiniLM-L6-v2"):
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        hf_cache = os.path.expanduser(
            "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
        )
        if os.path.isdir(hf_cache):
            print("[NLP] Modo HF_HUB_OFFLINE activado. Usando solo cache local.")
            return model_name, True
        print(
            "[NLP] HF_HUB_OFFLINE=1 pero el modelo no está cacheado todavía. "
            "Necesitas internet una vez."
        )

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    local_project = os.path.join(project_root, "models", "nlp", "all-MiniLM-L6-v2")
    if os.path.isdir(local_project):
        print(f"[NLP] Usando copia local del proyecto: {local_project}")
        return local_project, True

    hf_cache = os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2"
    )
    if os.path.isdir(hf_cache):
        return model_name, True
    return model_name, False


class _ConceptRouterMixin:
    """
    API de routing derivada solo de scores de conceptos (relativos).
    Misma interfaz para Light y Full.
    """

    STRUCTURAL_SIGNAL_THRESHOLD = 0.12

    def score_concepts(self, prompt: str) -> Dict[str, float]:
        raise NotImplementedError

    def get_axis_scores(self, prompt: str) -> Dict[str, float]:
        c = self.score_concepts(prompt)
        return {
            axis: max((c.get(k, 0.0) for k in keys), default=0.0)
            for axis, keys in _AXIS_KEYS.items()
        }

    def get_magnitude(self, prompt: str) -> float:
        if not prompt or not prompt.strip():
            return 0.55
        axes = self.get_axis_scores(prompt)
        pose = axes.get("pose", 0.0)
        structural = axes.get("structural", 0.0)
        attribute = axes.get("attribute", 0.0)

        magnitude = 0.50
        impact = max(pose * 0.85, structural * 0.80, attribute * 0.72)
        if impact > 0.04:
            magnitude += impact * 0.95

        # Mag mínima según eje dominante (sin mirar palabras del usuario)
        dominant = max(axes, key=axes.get)
        dom = axes[dominant]
        if dominant == "structural" and dom > 0.05:
            magnitude = max(magnitude, 0.72)
        elif dominant == "pose" and dom > 0.05:
            magnitude = max(magnitude, 0.68)
        elif dominant == "attribute" and dom > 0.06:
            magnitude = max(magnitude, 0.65)
        if self.is_recolor_request(prompt):
            magnitude = max(magnitude, 0.70)
        return max(0.48, min(0.95, magnitude))

    def get_body_transform_intensity(self, prompt: str) -> float:
        if not prompt or not prompt.strip():
            return 0.0
        c = self.score_concepts(prompt)
        body = c.get("attribute_body", 0.0)
        # Si add/remove estructural domina, no es body-transform (evita "remove person" → undress)
        rem = c.get("structural_remove", 0.0)
        add = c.get("structural_add", 0.0)
        if max(rem, add) > body * 1.05:
            return 0.0
        attr = c.get("attribute", 0.0)
        return min(1.0, body * 0.90 + attr * 0.10)

    def get_body_direction(self, prompt: str) -> str:
        """Dirección del cambio de ropa: uncover (quitar) / cover (vestir) / change.

        Compara scores entre conceptos de dirección, sin mirar palabras del usuario.
        Se usa para elegir el negative: solo se niegan prendas que tapen cuando el
        pedido es quitar; si es vestir (nieve, abrigo...), el negative es neutro.
        """
        if not prompt or not prompt.strip():
            return "change"
        c = self.score_concepts(prompt)
        un = c.get("attribute_uncover", 0.0)
        co = c.get("attribute_cover", 0.0)
        if un > co * 1.25 and un > 0.03:
            return "uncover"
        if co > un * 1.25 and co > 0.03:
            return "cover"
        return "change"

    def detect_target(self, prompt: str) -> str:
        if not prompt or not prompt.strip():
            return "subject"
        c = self.score_concepts(prompt)
        targets = {
            "background": c.get("target_background", 0.0),
            "face": c.get("target_face", 0.0),
            "clothes": c.get("target_clothes", 0.0),
            "hair": c.get("target_hair", 0.0),
            "subject": c.get("target_subject", 0.0),
        }
        best = max(targets, key=targets.get)
        if targets[best] < 0.05:
            best = "subject"

        axes = self.get_axis_scores(prompt)
        # Fondo solo si el concepto background gana y structural domina
        if best == "background" and axes.get("structural", 0) < axes.get("attribute", 0) * 1.1:
            if axes.get("attribute", 0) > 0.08:
                best = "subject"
        if axes.get("pose", 0) > 0.08 and best == "background":
            if axes.get("pose", 0) >= axes.get("structural", 0):
                best = "subject"
        return best

    def is_structural_dominant(self, prompt: str) -> bool:
        return self.get_axis_scores(prompt).get("structural", 0.0) > self.STRUCTURAL_SIGNAL_THRESHOLD

    def get_structural_bias(self, prompt: str) -> str:
        c = self.score_concepts(prompt)
        add_s = c.get("structural_add", 0.0)
        rem_s = c.get("structural_remove", 0.0)
        bg_s = c.get("structural_background", 0.0)
        pose_s = c.get("pose", 0.0)
        recolor_s = c.get("attribute_recolor", 0.0)
        # Pose / recolor no deben clasificarse como remove/add
        if pose_s > max(add_s, rem_s) * 1.05:
            return "neutral"
        if recolor_s > max(add_s, rem_s) * 1.05:
            return "neutral"
        if rem_s >= 0.08 and rem_s > add_s and rem_s >= bg_s * 0.85:
            return "remove"
        if add_s >= 0.08 and add_s > rem_s and add_s >= bg_s * 0.85:
            return "add"
        return "neutral"

    def is_quality_dominant(self, prompt: str) -> bool:
        c = self.score_concepts(prompt)
        quality = c.get("attribute_quality", 0.0)
        action = max(
            self.get_axis_scores(prompt).get("pose", 0.0),
            self.get_axis_scores(prompt).get("structural", 0.0),
            c.get("attribute_recolor", 0.0),
            c.get("attribute_body", 0.0),
        )
        return quality > 0.08 and quality > action * 1.05

    def is_recolor_request(self, prompt: str) -> bool:
        """Recolor/colorize generativo: concepto recolor domina quality y no hay acción fuerte."""
        if not prompt or not prompt.strip():
            return False
        c = self.score_concepts(prompt)
        recolor = c.get("attribute_recolor", 0.0)
        quality = c.get("attribute_quality", 0.0)
        axes = self.get_axis_scores(prompt)
        if recolor < 0.08:
            return False
        # Si pose/structural dominan, no es “solo recolor”
        if max(axes.get("pose", 0), axes.get("structural", 0)) > recolor * 1.2:
            return False
        return recolor >= quality * 0.9

    def is_photometric_only(self, prompt: str) -> bool:
        """
        Ajuste fotométrico global (OpenCV): quality alto, sin pose/structural,
        sin recolor dominante, sin body transform.
        """
        if not prompt or not prompt.strip():
            return False
        if self.is_recolor_request(prompt):
            return False
        c = self.score_concepts(prompt)
        axes = self.get_axis_scores(prompt)
        if axes.get("pose", 0) > 0.06 or axes.get("structural", 0) > 0.06:
            return False
        if c.get("attribute_body", 0) > 0.08:
            return False
        quality = c.get("attribute_quality", 0.0)
        recolor = c.get("attribute_recolor", 0.0)
        return quality > 0.06 and quality >= recolor

    def is_pose_dominant(self, prompt: str) -> bool:
        axes = self.get_axis_scores(prompt)
        pose = axes.get("pose", 0.0)
        return pose > 0.08 and pose >= max(
            axes.get("structural", 0.0), axes.get("attribute", 0.0)
        )


class SemanticIntentAnalyzer(_ConceptRouterMixin):
    """Embeddings: similitud coseno prompt↔conceptos. Sin listas de palabras de usuario."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[NLP] Cargando modelo semántico {model_name} en CPU...")
        global SentenceTransformer, util, torch
        from sentence_transformers import SentenceTransformer as ST, util as U
        import torch as T

        SentenceTransformer = ST
        util = U
        torch = T

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
            if os.environ.get("HF_HUB_OFFLINE") != "1":
                self.model = SentenceTransformer(model_name, device=self.device)
            else:
                raise RuntimeError(
                    "Modo offline (HF_HUB_OFFLINE=1) sin modelo local."
                ) from e

        self.concepts = {k: list(v) for k, v in CONCEPTS.items()}
        # compat: anchors = ejes principales
        self.anchors = {
            "pose": self.concepts["pose"],
            "structural": self.concepts["structural"],
            "attribute": self.concepts["attribute"],
        }
        self._concept_embeddings = {}
        for name, descs in self.concepts.items():
            self._concept_embeddings[name] = self.model.encode(
                descs, convert_to_tensor=True, device=self.device
            )
        self.anchor_embeddings = {
            k: self._concept_embeddings[k] for k in ("pose", "structural", "attribute")
        }
        print("[NLP] Analizador semántico listo.")

    def score_concepts(self, prompt: str) -> Dict[str, float]:
        if not prompt or not prompt.strip():
            return {k: 0.0 for k in self.concepts}
        emb = self.model.encode(prompt, convert_to_tensor=True, device=self.device)
        out = {}
        for name, cembs in self._concept_embeddings.items():
            out[name] = float(torch.max(util.cos_sim(emb, cembs)))
        return out


class LightLocalIntentAnalyzer(_ConceptRouterMixin):
    """
    Local ligero: solapamiento prompt↔texto de conceptos.
    Misma API y mismos CONCEPTS que la full. Sin ifs por palabra de usuario.
    """

    _STOPWORDS = frozenset({
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "how", "its",
        "may", "now", "old", "see", "way", "who", "did", "got", "let",
        "say", "she", "too", "use", "any", "per", "que", "del", "las",
        "los", "con", "por", "una", "this", "that", "with", "from",
        "them", "been", "have", "would", "could", "should",
    })

    def __init__(self):
        print("[NLP] Usando analizador local ligero (conceptos, sin hardcode de usuario).")
        self.device = "cpu"
        self.concepts = {k: list(v) for k, v in CONCEPTS.items()}
        self.anchors = {
            "pose": self.concepts["pose"],
            "structural": self.concepts["structural"],
            "attribute": self.concepts["attribute"],
        }

    @staticmethod
    def _token_match(left: str, right: str) -> str:
        """How two tokens overlap. A conjugation or a doubled letter is exact.

        "elimina"/"eliminar" and "removes"/"remove" differ by a short ending.
        "tatoo"/"tattoo" is the same token with one letter typed twice.
        A short stem such as "gray" inside "grayscale" stays a partial:
        the extra letters are a different word, not an ending.
        """
        if left == right:
            return "exact"
        shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
        if len(shorter) >= 4 and len(longer) >= 4 and shorter in longer:
            if len(shorter) >= 5 and len(longer) - len(shorter) <= 3:
                return "exact"
            return "partial"
        if len(left) >= 5 and len(right) >= 5:
            def fold(token):
                return re.sub(r"(.)\1+", r"\1", token)

            if fold(left) == fold(right):
                return "exact"
        return "none"

    @staticmethod
    def _tokenize(text: str) -> set:
        raw = re.findall(r"[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ0-9]+", (text or "").lower())
        stop = LightLocalIntentAnalyzer._STOPWORDS
        return {w for w in raw if len(w) > 2 and w not in stop}

    def _score_texts(self, prompt: str, texts: list) -> float:
        """
        Similitud prompt↔concepto. Normaliza por palabras del PROMPT
        (prompts cortos no se diluyen en descripciones largas de concepto).
        """
        p_words = self._tokenize(prompt)
        if not p_words:
            return 0.0
        best = 0.0
        for t in texts:
            t_words = self._tokenize(t)
            if not t_words:
                continue
            exact = 0
            partial = 0
            for pw in p_words:
                kind = "none"
                if pw in t_words:
                    kind = "exact"
                else:
                    for tw in t_words:
                        kind = self._token_match(pw, tw)
                        if kind != "none":
                            break
                if kind == "exact":
                    exact += 1
                elif kind == "partial":
                    partial += 1
            # Cobertura del prompt en el concepto (no del concepto en el prompt)
            coverage = (exact + 0.55 * partial) / max(len(p_words), 1)
            # Bonus ligero si hay solape denso con el concepto
            density = (exact + 0.4 * partial) / max(len(t_words), 1)
            score = min(1.0, coverage * 0.75 + density * 0.35)
            best = max(best, score)
        return min(1.0, best)

    def score_concepts(self, prompt: str) -> Dict[str, float]:
        if not prompt or not prompt.strip():
            return {k: 0.0 for k in self.concepts}
        return {name: self._score_texts(prompt, descs) for name, descs in self.concepts.items()}


_global_semantic_analyzer = None


def get_semantic_analyzer(full_ai=False):
    global _global_semantic_analyzer
    if full_ai:
        if _global_semantic_analyzer is None or not hasattr(_global_semantic_analyzer, "model"):
            _global_semantic_analyzer = SemanticIntentAnalyzer()
        return _global_semantic_analyzer
    return LightLocalIntentAnalyzer()
