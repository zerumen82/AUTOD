#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import re
from types import SimpleNamespace
from typing import Optional, Tuple, Dict
from PIL import Image, ImageFilter
import numpy as np
from roop.utils import get_vram_gb

from roop.utilities import resolve_relative_path


def _otsu_threshold(values: np.ndarray) -> float:
    """Level that splits a response histogram into its two measured groups.

    OpenCV's Otsu picks, from the data itself, the boundary between the weak
    response (resampling and shading of a sampled region) and the strong one
    (the requested edit). It carries no parameter, so the same measurement
    separates a shaved scalp from a recolored garment.
    """
    data = np.asarray(values, dtype=np.uint8)
    if data.size == 0:
        return 0.0
    scale = 256.0 / float(data.max() + 1) if data.max() > 255 else 1.0
    quantized = np.clip(data.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    threshold, _ = cv2.threshold(
        quantized.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return float(threshold) / scale if scale else float(threshold)


def _skin_pixels(photo: np.ndarray) -> np.ndarray:
    """Pixels whose color is skin, measured on the frame itself.

    Shared by the guards that keep a grown surface out of another person.
    """
    red, green, blue = photo[:, :, 0], photo[:, :, 1], photo[:, :, 2]
    return (
        (red > 90) & (green > 40) & (blue > 20)
        & (red > green) & (red > blue)
        & (np.abs(red - green) > 12)
    )


def _same_color_limit(distances: np.ndarray, surface: np.ndarray, subject: np.ndarray):
    """Distance where the measured color ends and a different one begins.

    The limit sits halfway between the measured color and the median of the
    pixels that are already past it. Lighter and darker changes both land
    past that color, so neither direction is special. When the requested
    surface is most of the subject, the leftover median is still that color;
    only the pixels that actually differ count as the other surface. A subject
    with nothing past the measured color is one color, and the peak stays put.
    """
    surface_d = distances[surface]
    if int(surface.sum()) < 48 or int(subject.sum()) < 48:
        return None
    spread = float(np.percentile(surface_d, 90))
    rest = subject & ~surface
    if not rest.any():
        return None
    # 18 is the separation already used for a flat photo on this path.
    other = distances[rest]
    other = other[other > spread + 18.0]
    if other.size < 48:
        return None
    other_level = float(np.median(other))
    if other_level < spread + 18.0:
        return None
    return spread + 0.5 * (other_level - spread)


class ImgEditorManager:
    def __init__(self):
        self.flux_klein_client = None
        self.flux_schnell_client = None
        self.flux_dev_client = None
        self.flux_dev_abl_client = None
        self.qwen_edit_client = None
        self.omnigen2_client = None
        self.icedit_client = None
        self.boogu_edit_client = None
        self.clip_masker = None
        self.clothing_segmenter = None
        self.prompt_analyzer = None
        self.semantic_analyzer = None
        self._last_context = "normal"
        self._codeformer = None
        self._last_instruction_compliance = None

    def _get_prompt_analyzer(self):
        if self.prompt_analyzer is None:
            from roop.img_editor.prompt_analyzer import PromptAnalyzer, EditingMode
            self.prompt_analyzer = PromptAnalyzer()
            self.EditingMode = EditingMode
        return self.prompt_analyzer

    @staticmethod
    def _edit_is_nearly_unchanged(original: Image.Image, edited: Image.Image) -> bool:
        """Catch visual no-ops before a Quality finish makes them look successful."""
        sample_size = (256, max(1, int(round(256 * original.height / max(1, original.width)))))
        if original.height > original.width:
            sample_size = (max(1, int(round(256 * original.width / original.height))), 256)
        a = np.asarray(original.convert("RGB").resize(sample_size, Image.Resampling.BILINEAR), dtype=np.int16)
        b = np.asarray(edited.convert("RGB").resize(sample_size, Image.Resampling.BILINEAR), dtype=np.int16)
        difference = np.abs(a - b)
        pixel_change = np.max(difference, axis=2)
        if float(difference.mean()) >= 0.8 or float(np.mean(pixel_change > 12)) >= 0.01:
            return False
        h, w = pixel_change.shape
        for y in range(0, h, max(1, h // 8)):
            for x in range(0, w, max(1, w // 8)):
                if float(pixel_change[y:y + max(1, h // 8), x:x + max(1, w // 8)].mean()) >= 2.5:
                    return False
        return True

    @staticmethod
    def _edit_leaves_a_ghost(
        original: Image.Image, edited: Image.Image,
        change_floor: float = 0.004, change_ceiling: float = 0.60,
        corr_min: float = 0.45, amp_max: float = 0.75,
    ) -> bool:
        """Detecta el resto fantasma del material original tras un cambio real.

        Log 2026-10-17 01:49 (calvo sobre foto 718x600): el modelo no
        reemplazó el pelo, lo DESVANECIÓ hacia el fondo a opacidad parcial;
        la cáscara resultante midió cambio (no era no-op) así que se pegó tal
        cual y quedó el halo. La firma medida de ese fantasma, dentro de los
        píxeles que sí cambiaron, es doble: la estructura de alta frecuencia
        del original sigue legible (correlación alta entre HF(original) y
        HF(editado)) mientras su amplitud cae (el material perdió contraste).
        Un reemplazo real reescribe la textura: la correlación se rompe.
        Devuelve False en no-ops (los cubre `_edit_is_nearly_unchanged`) y en
        regeneraciones de escena (casi todo el encuadre cambia).
        """
        sample_size = (
            512, max(1, int(round(512 * original.height / max(1, original.width))))
        )
        if original.height > original.width:
            sample_size = (
                max(1, int(round(512 * original.width / original.height))), 512,
            )
        a = np.asarray(
            original.convert("RGB").resize(sample_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        b = np.asarray(
            edited.convert("RGB").resize(sample_size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        pixel_change = np.abs(a - b).max(axis=2)
        changed = pixel_change > 12
        changed_frac = float(changed.mean())
        if changed_frac < change_floor or changed_frac > change_ceiling:
            return False
        gray_a = Image.fromarray(a.mean(axis=2).astype(np.uint8), "L")
        gray_b = Image.fromarray(b.mean(axis=2).astype(np.uint8), "L")
        hf_a = np.asarray(gray_a, dtype=np.float32) - np.asarray(
            gray_a.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32
        )
        hf_b = np.asarray(gray_b, dtype=np.float32) - np.asarray(
            gray_b.filter(ImageFilter.GaussianBlur(2)), dtype=np.float32
        )
        fa = hf_a[changed]
        fb = hf_b[changed]
        std_a = float(fa.std())
        std_b = float(fb.std())
        if std_a < 1.0:
            return False
        corr = float(np.corrcoef(fa.ravel(), fb.ravel())[0, 1])
        return bool(corr >= corr_min and (std_b / std_a) <= amp_max)

    @staticmethod
    def _detect_edit_faces(image: Image.Image):
        """Caras del resultado editado, en coordenadas de la propia imagen.

        CPU para no pelear la VRAM con ComfyUI (FaceAnalysis CUDA ya se
        liberó antes de generar). Umbral 0.20, el mismo del preflight, y un
        segundo pase ampliado si el primero ve menos de dos caras (patrón de
        _build_person_removal_mask: una cara pequeña de foto pequeña no se
        caza a primera pasada).
        """
        try:
            from roop.face_util import get_face_analyser_cpu, preprocess_image_for_detection
            analyser = get_face_analyser_cpu()
            if analyser is None:
                return []
            bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            faces = list(analyser.get(bgr) or [])
            if len(faces) < 2:
                seen = {tuple(int(v) for v in f.bbox) for f in faces}
                enhanced = preprocess_image_for_detection(bgr)
                for extra in analyser.get(enhanced) or []:
                    key = tuple(int(v) for v in extra.bbox)
                    if key not in seen:
                        faces.append(extra)
                        seen.add(key)
            return [
                f for f in faces
                if float(getattr(f, "det_score", 0.0) or 0.0) >= 0.20
            ]
        except Exception:
            return []

    def _added_face_mask(
        self, original: Image.Image, edited: Image.Image, faces,
    ) -> Optional[Image.Image]:
        """Máscara sobre las caras que el modelo pintó y la foto no tiene.

        Log 2026-10-17 10:23 (calvo sobre foto 2847x4263): la entrada trajo
        2 caras («2 cara(s)» del arranque) y el modelo pintó una TERCERA cara
        sobre la coronilla mientras obedecía el calvo; el pegado por cambio
        medido la conserva porque textura nueva mide como cambio real, y ni
        el no-op ni el fantasma la cogen (contenido nuevo, correlación HF
        rota). Una persona que nadie pidió se mide igual que Fix D mide una
        cara colapsada: cada cara detectada en el resultado debe caer cerca
        de una cara de la entrada; una cuyo centro queda a más de 0.6 de su
        propia altura de TODO centro de la fuente no pertenece a nadie, y su
        bbox (más un margen del 20%) vuelve como máscara para que el caller
        restaure la foto original ahí. Devuelve None cuando no hay nada que
        rechazar (sin referencia no se juzga: ruta de cámara y fotos sin
        caras quedan fuera).
        """
        if original is None or edited is None or not faces:
            return None
        try:
            canvas = edited.convert("RGB")
            if canvas.size != original.size:
                canvas = canvas.resize(original.size, Image.Resampling.LANCZOS)
            generated = self._detect_edit_faces(canvas)
            if not generated:
                return None
            source_centers = []
            for face in faces:
                try:
                    x1, y1, x2, y2 = [float(v) for v in face.bbox]
                except (TypeError, ValueError, AttributeError):
                    continue
                if x2 > x1 and y2 > y1:
                    source_centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            if not source_centers:
                return None
            bad = np.zeros((original.height, original.width), dtype=np.uint8)
            found = False
            soft_sigma = 3.0
            for face in generated:
                try:
                    x1, y1, x2, y2 = [float(v) for v in face.bbox]
                except (TypeError, ValueError, AttributeError):
                    continue
                w, h = x2 - x1, y2 - y1
                if w < 16 or h < 16:
                    continue
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                distance = min(
                    ((cx - sx) ** 2 + (cy - sy) ** 2) ** 0.5
                    for sx, sy in source_centers
                )
                if distance <= 0.6 * h:
                    continue
                radius = max(6, int(round(0.10 * max(w, h))))
                bx1 = max(0, int(x1) - radius)
                by1 = max(0, int(y1) - radius)
                bx2 = min(original.width, int(x2) + radius)
                by2 = min(original.height, int(y2) + radius)
                if bx2 - bx1 < 4 or by2 - by1 < 4:
                    continue
                bad[by1:by2, bx1:bx2] = 255
                found = True
                soft_sigma = max(soft_sigma, radius / 3.0)
                print(
                    f"[ImgEditor] Cara en el resultado a {distance:.0f} px "
                    f"de cualquier cara de la foto ({bx2 - bx1}x{by2 - by1}): "
                    "es nueva, no estaba en la entrada",
                    flush=True,
                )
            if not found:
                return None
            soft = cv2.GaussianBlur(bad, (0, 0), soft_sigma)
            soft = np.clip(soft.astype(np.int32) * 2, 0, 255).astype(np.uint8)
            return Image.fromarray(soft, "L")
        except Exception:
            return None

    def _get_semantic_analyzer(self, full_ai: bool = False):
        if self.semantic_analyzer is None:
            from roop.img_editor.nlp.semantic_analyzer import get_semantic_analyzer
            self.semantic_analyzer = get_semantic_analyzer(full_ai=full_ai)
        return self.semantic_analyzer

    def _get_clothing_segmenter(self):
        if self.clothing_segmenter is None:
            from roop.img_editor.clothing_segmenter import get_clothing_segmenter
            seg = get_clothing_segmenter()
            if seg.is_available():
                self.clothing_segmenter = seg
        return self.clothing_segmenter

    def _get_clipseg_masker(self):
        if self.clip_masker is None:
            from roop.img_editor.clipseg_masker import get_clipseg_masker
            self.clip_masker = get_clipseg_masker()
        return self.clip_masker

    def auto_detect_params(self, analysis: Dict, engine: str) -> Dict:
        """
        Calcula parámetros dinámicos basados puramente en el análisis semántico (embeddings).
        Cero hardcode de palabras clave de usuario.
        Optimizado para 8GB VRAM (LongCat Turbo etc).
        """
        magnitude = analysis.get("magnitude", 0.5)
        quality_only = bool(analysis.get("quality_only", False))
        body_transform = bool(analysis.get("body_transform", False))

        if quality_only:
            tier = str(analysis.get("enhance_tier", "hd")).lower()
            print(
                f"[ImgEditor] Modo MEJORA UI tier={tier}: "
                "pipeline clásico medido (sin LongCat, sin editar contenido)"
            )
            return {
                "denoise": 0.0,
                "num_inference_steps": 0,
                "guidance_scale": 0.0,
                "mode": "quality_classical",
            }

        # Escalamiento lineal dinámico basado en magnitud (0.0 a 1.0)
        # Denoise: 0.20 (sutil) a 0.90 (radical)
        denoise = 0.20 + (magnitude * 0.70)
        denoise = min(denoise, 0.95)

        # Pasos: 12 (rápido) a 28 (calidad) para modelos estándar
        min_steps = 12
        max_steps = 28
        steps = int(min_steps + (magnitude * (max_steps - min_steps)))

        # Guidance: 3.0 a 7.0
        guidance = 3.0 + (magnitude * 4.0)

        if engine in ("longcat", "imagine"):
            if body_transform:
                # Grok Imagine versátil: denoise alto para cambios corporales/ropa completos
                # intensity controla con qué fuerza el cuerpo cambia (0.0-1.0)
                intensity = float(analysis.get("body_transform_intensity", 0.5))
                denoise = 0.84 + (intensity * 0.08) + (magnitude * 0.04)  # 0.84-0.96
                denoise = min(denoise, 0.96)
                guidance = 2.5  # CFG bajo para colores naturales (4.5 sobresatura piel/ropa)
                steps = int(20 + (magnitude * 8))  # 20-28 pasos para calidad en body
            else:
                denoise = 0.60 + (magnitude * 0.20)
                # v5.1111 — POV (sin body): la vista destino NO existe en la fuente,
                # hay que alucinarla → necesita denoise de regeneración casi total.
                # mag 0.85 (siempre en POV por el boost) → 0.77 es insuficiente.
                # La postura nueva tampoco existe → mismo floor.
                if analysis.get("angle_change", False) or analysis.get("pose_change", False):
                    denoise = max(denoise, 0.85)
                guidance = 2.5
        elif engine == "kontext":
            # FLUX.1 Kontext dev: 20-28 steps (EDIT nativo: ReferenceLatent + FluxGuidance).
            steps = 20
            guidance = 3.0
            if body_transform:
                _bdir = str(analysis.get("body_direction", "change")).lower()
                if _bdir == "uncover":
                    # v5.1101: guidance 4.8 NO movía la aguja (log 08:41: ropa
                    # intacta con 4.8). Kontext dev entrena con guidance 2.5 y
                    # la receta oficial USO usa 3.5 — fuera de ese rango el
                    # modelo se sale de distribución y se agarra a la referencia.
                    # La adherencia real la da el método uxo/uno, no más guidance.
                    guidance = 3.5
                else:
                    guidance = 3.5
                # v5.1105: 24→20 steps en Q4 (offload en 8GB: cada step tardío
                # cuesta 10-17s por tráfico de offload; los últimos steps son
                # los más lentos). A 20 steps la adherencia es la misma y se
                # ahorra ~1min por run.
                steps = 20
            else:
                # mag baja → guidance suave (preserva), mag alta → adherencia fuerte
                # (cap 3.5: rango oficial, más alto degrada el following)
                guidance = min(3.5, 2.5 + (magnitude * 2.0))  # 2.5-3.5
        elif engine == "longcat_full":
            steps = int(20 + (magnitude * 10))
            guidance = 2.5  # CFG bajo evita sobresaturación (4.5 quema colores)
        elif engine == "flux_schnell":
            steps = 4 # Limitación técnica del modelo Schnell
        elif "qwen" in engine:
            steps = min(steps, 8) # Qwen es muy lento, limitamos por UX
            guidance = min(guidance, 4.0)
        elif engine == "klein_9b":
            steps = 4  # Klein 9B distilled, 4 pasos nativos
            guidance = 1.0  # Forzado para modelos distilled
            denoise = 0.85  # Denoise alto para edits fuertes
        elif engine == "klein" or "klein" in engine:
            steps = 4  # Klein 4B distilled, 4 pasos nativos (el cliente fuerza 4/CFG1.0/denoise1.0)
            guidance = 1.0
            denoise = 1.0  # Klein EDIT real usa latente vacio + ReferenceLatent: denoise 1.0 obligatorio
        elif engine == "hart":
            # HART es autoregresivo puro. Usa guidance alto y pasos limitados.
            steps = min(8, max(4, steps))
            guidance = 4.5
            denoise = 1.0  # No aplica denoise real (generación)
        elif engine == "qwen_edit":
            steps = min(steps, 20)
            guidance = min(guidance, 4.0)
            denoise = min(denoise, 0.95)

        print(f"[ImgEditor] Dynamic Params: Mag={magnitude:.2f}, Denoise={denoise:.2f}, Steps={steps}, CFG={guidance:.1f}")

        return {
            "denoise": denoise,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "mode": "img2img"
        }

    _STOPWORDS = frozenset({"the", "and", "for", "are", "but", "not", "you", "all", "can",
                            "had", "her", "was", "one", "our", "out", "has", "how", "its",
                            "may", "now", "old", "see", "way", "who", "did", "got", "let",
                            "say", "she", "too", "use", "any", "per", "que", "del", "las",
                            "los", "con", "por", "una"})

    def _light_score(self, prompt_l: str, texts: list) -> float:
        p_words = set(w for w in prompt_l.split() if len(w) > 2 and w not in self._STOPWORDS)
        if not p_words:
            return 0.0
        best = 0.0
        for t in texts:
            t_words = set(w for w in t.lower().split() if len(w) > 2 and w not in self._STOPWORDS)
            if t_words:
                overlap = len(p_words & t_words) / max(len(t_words), 1)
                for pw in p_words:
                    for tw in t_words:
                        if pw in tw or tw in pw:
                            overlap += 0.08
                            break
                best = max(best, overlap)
        return min(1.0, best)

    def _strip_instruction_prefix(self, prompt: str) -> str:
        text = (prompt or "").strip()
        for prefix in ("instruction:", "instrucción:", "instruccion:"):
            if text.lower().startswith(prefix):
                return text[len(prefix):].strip()
        return text

    @staticmethod
    def _discard_rewrite_text(input_text: str, body_direction: str, engine: str) -> bool:
        """¿Descartar el TEXTO del rewriter y usar el traducido intacto?

        v5.1110 — REGLA GENERAL: si la entrada NO es inglés, el texto del
        rewriter 0.5B se descarta SIEMPRE (cualquier engine, cualquier
        dirección). Antes solo protegía uncover+no-kontext: los compuestos
        en change/cover/kontext seguían perdiendo mitades ("sigue sin hacer
        un prompt compuesto"). Evidencia de los 3 modos de fallo del 0.5B:
        inventa tokens (log 11:40), OMITE mitades (log 11:57) y alucina con
        typos (log 12:12). La traducción completa ya existe ANTES del
        rewriter (mappings locales + MyMemory locale completo, test live
        OK) — enriquecer un compuesto perdiendo la mitad es peor que un EN
        plano completo. Los METADATOS del rewriter (mag/target) se conservan.

        Entrada EN → rewriter se conserva (polish en el mismo idioma, seguro).

        NOTA: input_text debe ser el prompt CRUDO del usuario (prompt_raw),
        no el traducido — el texto mezclado puede parecer "inglés" aunque la
        traducción online fallara entera (log 11:57). Los params
        body_direction/engine se mantienen en la firma por compatibilidad
        (call site) pero ya no condicionan la decisión.
        """
        from roop.img_editor.prompt_translator import _is_english
        return not _is_english(input_text or "")

    @staticmethod
    def _rewriter_repairs_short_translation(
        translated: str, rewritten: str, raw_input: str,
    ) -> bool:
        """Acepta un repair del rewriter solo en el caso estrecho y comprobable.

        El rewriter pequeño no es fiable para reformular prompts compuestos en
        español. Pero un traductor puede dejar UN token roto dentro de una
        orden breve; descartar una corrección que conserva el contenido deja al
        modelo con una instrucción sin gramática. No se acepta una expansión
        libre: debe ser un único pedido, corto, eliminar el token sospechoso y
        conservar todos los tokens de contenido estables de la traducción.
        """
        try:
            from roop.img_editor.prompt_translator import (
                _count_conditions, _is_english, translation_suspect_tokens,
            )
            suspect = {token.lower() for token in translation_suspect_tokens(raw_input, translated)}
            if (not suspect or not _is_english(rewritten)
                    or _count_conditions(translated) != 1):
                return False
            translated_words = re.findall(r"[a-zA-Z]+", translated.lower())
            rewritten_words = set(re.findall(r"[a-zA-Z]+", rewritten.lower()))
            # Prompts largos o compuestos siguen por la ruta conservadora: ahí
            # el 0.5B puede borrar una de las condiciones sin que podamos
            # probar semánticamente que no lo hizo.
            if len(translated_words) > 8 or suspect & rewritten_words:
                return False
            stable_content = {
                word for word in translated_words
                if len(word) >= 4 and word not in suspect
            }
            return bool(stable_content) and stable_content.issubset(rewritten_words)
        except Exception:
            return False



    def _resolve_reference_latents_method(
        self,
        analysis: Dict,
        *,
        background_focus: bool,
        structural_inpaint: bool,
        engine: str,
        body_transform: bool = False,
    ) -> Optional[str]:
        if engine not in ("longcat", "imagine", "longcat_full", "kontext"):
            return None
        structural_bias = str(analysis.get("structural_bias", "neutral"))
        if structural_inpaint and structural_bias == "remove":
            return "index"
        # El workflow distribuido de LongCat Image Edit usa `index`: conserva
        # la imagen con menos rigidez que offset, sin depender del contenido del
        # encargo del usuario.
        if engine in ("longcat", "imagine", "longcat_full"):
            return "index"
        # Para body con Kontext (EDIT nativo v5.1099 + método v5.1101/1102/1103):
        # el ReferenceLatent aporta la foto y el nodo FluxKontextMultiRef-
        # erenceLatentMethod fija la fuerza del anclaje de forma explícita.
        # v5.1103: Q4+offset TAMBIÉN copia (log 09:36, std 55.55, sat 0.98) →
        # el ancla domina la instrucción incluso con modelo capaz. uncover
        # vuelve a "uxo/uno": es la combinación nunca probada (modelo capaz +
        # ancla mínima) y el test decisivo. Si Q4+uxo sale gris/incoherente,
        # el siguiente lever es "index" (punto medio entre offset y uxo).
        if body_transform and engine == "kontext":
            _bdir = str(analysis.get("body_direction", "change")).lower()
            if _bdir == "uncover":
                return "uxo/uno"
            return "offset"
        return "offset"

    def _route_body_engine(
        self,
        preferred_engine: str,
        body_direction: str,
        body_intensity: float,
        mag: float,
        vram_gb: float,
        angle_change: bool = False,
        pose_change: bool = False,
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Router centralizado para body_transform (desnudo/vestir/cambiar ropa).

        Usa body_direction del semantic_analyzer (uncover/cover/change) para
        seleccionar el engine que mejor se adapta al tipo de cambio corporal,
        respetando las constraints de VRAM.

        Returns: (engine_override, version_override, log_reason)
                 (None, None, "") si el engine preferido ya es el correcto.
        """
        # v5.1102: Q4_K_M — el Q2_K era una "copy machine" (copia con offset,
        # gris con uxo). Q4 + offload en 8GB, adherencia real.
        _KONTEXT_VERSION = "flux1-kontext-dev-Q4_K_M.gguf"
        _LONGCAT_FULL_VERSION = "LongCat-Image-Edit-Q4_K_S.gguf"
        _LONGCAT_TURBO_VERSION = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
        _LONGCAT_MIN_VRAM = 11.0

        # LongCat Full ya es el régimen fuerte elegido por el usuario. Kontext
        # normalmente se respeta, salvo POV/postura: en esos casos su ancla
        # contradice la instrucción y debe dejar paso al router fuerte.
        if preferred_engine == "longcat_full" or (
            preferred_engine == "kontext" and not (angle_change or pose_change)
        ):
            return None, None, ""

        # Seleccionar engine目标 según dirección del cambio
        if body_direction == "uncover":
            # Desnudo: bypass no_kontext (Qwen-VL ve la foto, sin ancla de
            # latentes): junio lo probó en desnudo ("ha seguido las instrucciones").
            # v5.1120 — FULL también en 8GB (mismo veredicto del change:
            # Turbo 12 pasos/CFG 1.0 no ejecuta cambios complejos; Full 35 NFE
            # sí; el "cuelga" de v5.1090 era sin --disable-async-offload).
            target_e, target_v = "longcat_full", _LONGCAT_FULL_VERSION
        elif body_direction == "cover":
            # Vestir: LongCat Full preserva estructura y añade ropa bien.
            # Kontext en 8GB (único viable sin offload thrashing).
            if vram_gb >= _LONGCAT_MIN_VRAM:
                target_e, target_v = "longcat_full", _LONGCAT_FULL_VERSION
            else:
                target_e, target_v = "kontext", _KONTEXT_VERSION
        else:
            # change: flexible. Kontext rápido + buen following de instrucciones.
            # LongCat Full si VRAM permite y mag alta.
            if vram_gb >= _LONGCAT_MIN_VRAM and mag >= 0.80:
                target_e, target_v = "longcat_full", _LONGCAT_FULL_VERSION
            elif mag >= 0.75:
                # v5.1116 — change FUERTE en 8GB: el contenido destino NO existe
                # en la fuente → régimen bypass (sin ancla, mismo que uncover/POV/pose).
                # Kontext EDIT+offset es la copy machine documentada 3 veces
                # (v5.1101-1103: Q4+offset=copia) — el ancla de latentes preserva
                # la escena original contra mag 0.90. Solo magnitud, cero vocabulario.
                # v5.1120 — TURBO→FULL en 8GB: el bypass ya funciona (ref=none,
                # Qwen-VL ve) pero Turbo es distilled 12 pasos/CFG 1.0 → following
                # limitado (log 08:10: cambia ligeramente, no ejecuta la pose).
                # Full (mismo peso 3.82GB, 35 NFE, CFG 2.5, ~29 steps) es el que
                # ejecutó cambios de verdad en junio ("LongCat sí ejecutó el cambio").
                # El veredicto "Full cuelga en 8GB" (v5.1090) era SIN
                # --disable-async-offload: hoy el offload legacy corre estable
                # (log: TE 5.3GB offloaded + UNet 3.7GB, 3.1s/it sin stall).
                target_e, target_v = "longcat_full", _LONGCAT_FULL_VERSION
            else:
                target_e, target_v = "kontext", _KONTEXT_VERSION

        if preferred_engine == target_e:
            return None, None, ""

        reason = (
            f"body {body_direction} ({vram_gb:.1f}GB): "
            f"{preferred_engine} -> {target_e} "
            f"(intensity={body_intensity:.2f}, mag={mag:.2f})"
        )
        return target_e, target_v, reason

    def _route_edit_engine(
        self,
        preferred_engine: str,
        analysis: Dict,
        vram_gb: float,
    ) -> Tuple[Optional[str], Optional[str], str]:
        """
        Router GENERAL para CUALQUIER tipo de edición de imagen.

        Analiza el análisis semántico completo y selecciona el engine óptimo
        para el tipo de edición detectado. Cubre:
        - Body transforms (desnudo/vestir/cambiar ropa)
        - Structural changes (agregar/quitar objetos)
        - Background changes
        - Quality improvements
        - Pose/POV changes
        - Compound prompts (múltiples ejes)
        - Color/style edits

        Returns: (engine_override, version_override, log_reason)
                 (None, None, "") si el engine preferido ya es correcto.
        """
        # v5.1102: Q4_K_M — el Q2_K era una "copy machine".
        _KONTEXT_VERSION = "flux1-kontext-dev-Q4_K_M.gguf"
        _LONGCAT_FULL_VERSION = "LongCat-Image-Edit-Q4_K_S.gguf"
        _LONGCAT_TURBO_VERSION = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
        _LONGCAT_MIN_VRAM = 11.0

        body_transform = bool(analysis.get("body_transform", False))
        body_direction = str(analysis.get("body_direction", "change")).lower()
        body_intensity = float(analysis.get("body_transform_intensity", 0.0))
        mag = float(analysis.get("magnitude", 0.5))
        high_structural = bool(analysis.get("high_structural", False))
        structural_bias = str(analysis.get("structural_bias", "neutral"))
        quality_only = bool(analysis.get("quality_only", False))
        angle_change = bool(analysis.get("angle_change", False))
        pose_change = bool(analysis.get("pose_change", False))

        # Engines que NO funcionan bien para edits fuertes / body / structural
        _WEAK_FOR_STRONG = ("hart", "boogu", "flux_dev_abliterated", "flux_q2")

        # LongCat Full ya es el régimen fuerte elegido por el usuario. Kontext
        # normalmente se respeta, salvo POV/postura: en esos casos su ancla
        # contradice la instrucción y debe dejar paso al router fuerte.
        if preferred_engine == "longcat_full" or (
            preferred_engine == "kontext" and not (angle_change or pose_change)
        ):
            return None, None, ""

        # --- 1. BODY TRANSFORMS: router dedicado ---
        if body_transform:
            return self._route_body_engine(
                preferred_engine, body_direction, body_intensity, mag, vram_gb,
                angle_change=angle_change, pose_change=pose_change,
            )

        # --- 2. POV/postura/accion fisica: regimen fuerte SIN cambiar de pesos.
        # Turbo es el motor elegido; el bypass sin ancla se activa mas abajo
        # con pose_change (gen_params no_kontext). No se cambia a Full.
        if (angle_change or pose_change) and not body_transform:
            return None, None, ""
        if preferred_engine in _WEAK_FOR_STRONG:
            # Mag alta (>0.65) o structural add: necesita engine fuerte
            needs_strong = (
                mag >= 0.65
                or (high_structural and structural_bias == "add")
            )
            if needs_strong:
                if vram_gb >= _LONGCAT_MIN_VRAM:
                    return (
                        "longcat_full", _LONGCAT_FULL_VERSION,
                        f"edit fuerte (mag={mag:.2f}, struct={high_structural}, "
                        f"pov={angle_change}): {preferred_engine} -> longcat_full"
                    )
                # v5.1118 — en 8GB, mag>=0.75 → LongCat (régimen bypass).
                # El ancla de latentes de kontext EDIT pierde contra CUALQUIER
                # cambio fuerte (evidencia v5.1101-1104; log real 07:22: mag=0.80
                # no-body → kontext+offset → std=52, no obedeció). Kontext queda
                # para edits moderados (0.65-0.75), donde el ancla SÍ ayuda.
                # v5.1120 — FULL en vez de Turbo (mismo bypass, following superior).
                elif mag >= 0.75:
                    return (
                        "longcat_full", _LONGCAT_FULL_VERSION,
                        f"edit fuerte mag={mag:.2f} (8GB): {preferred_engine} -> longcat_full (bypass sin ancla)"
                    )
                else:
                    return (
                        "kontext", _KONTEXT_VERSION,
                        f"edit fuerte (mag={mag:.2f}, struct={high_structural}, "
                        f"pov={angle_change}): {preferred_engine} -> kontext (8GB)"
                    )

        # --- 3. LongCat Turbo es el motor elegido (spec/00-PRINCIPIOS). No hay
        # upgrade automatico a Full: el usuario pidio volver a Turbo.
        # mag>=0.75 se queda en Turbo con regimen fuerte (bypass sin ancla),
        # sin cambiar de pesos.

        # --- 3b. Klein pedido a mano: vuelve a Turbo (motor elegido).
        # Klein ignora la mascara por diseno (ReferenceLatent) y no
        # conserva cuerpos; Turbo con inpaint si la respeta.
        # Solo magnitud, cero vocabulario.
        if preferred_engine in ("klein", "klein_base", "klein_9b"):
            return (
                "longcat", _LONGCAT_TURBO_VERSION,
                "klein -> longcat turbo (mascara + cuerpos)"
            )

        # --- 4. Quality improvements: classical pipeline, sin override ---
        if quality_only:
            return None, None, ""

        # --- 5. Background: Turbo con mascara (motor elegido). No hay
        # override a Full/Kontext: la mascara focal ya manda a LongCat
        # con inpaint mas abajo.

        # --- 6. Default: mantener engine preferido ---
        return None, None, ""

    @staticmethod
    def _select_longcat_version(client, magnitude: float, body_transform: bool) -> str:
        """Turbo es el motor elegido (spec/00-PRINCIPIOS): siempre Turbo.

        Sin alternancia Full<->Turbo: el cambio de pesos descargaba el
        modelo residente y provocaba recargas de varios minutos.
        """
        return "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"

    def _apply_prompt_rewriter(
        self,
        prompt: str,
        *,
        img_context: str = "",
        magnitude: float,
        body_transform: bool,
    ) -> Tuple[str, float, str, bool, bool, bool, list, str]:
        try:
            from roop.img_editor.prompt_rewriter import get_prompt_rewriter
            rewriter = get_prompt_rewriter()
            rw = rewriter.rewrite(prompt, image_context=img_context, mode="img2img")
            if not isinstance(rw, dict):
                return prompt, magnitude, "subject", True, False, False, [], ""
            new_prompt = (rw.get("prompt") or prompt).strip()
            # The small text rewriter is useful for translation and structured
            # planning, but its scalar confidence is not a visual measurement.
            # Routing strength must remain tied to the image-aware semantic
            # analysis, never to a hallucinated number from text alone.
            new_mag = float(magnitude)
            mask_target = str(rw.get("mask_target", "subject")).lower()
            is_global = bool(rw.get("is_global", False))
            requires_scene_regeneration = bool(rw.get("requires_scene_regeneration", False))
            affected_regions = []
            raw_regions = rw.get("affected_regions", [])
            if isinstance(raw_regions, list):
                for raw_region in raw_regions:
                    region = str(raw_region or "").strip().lower()
                    if region and region not in affected_regions:
                        affected_regions.append(region)
            # Para una única edición localizada, la región anatómica (p. ej.
            # una cabeza) no identifica a quién pertenece. El rewriter entrega
            # una consulta visual separada y CLIPSeg la mide en la foto antes de
            # escoger una cara. No contiene reglas de nombres, géneros o lados.
            subject_query = str(rw.get("subject_query", "") or "").strip().lower()
            if subject_query and len(affected_regions) == 1:
                print(
                    f"[ImgEditor] Grounding de sujeto: {affected_regions[0]} → {subject_query}",
                    flush=True,
                )
                affected_regions = [subject_query]
            if body_transform:
                new_mag = max(new_mag, magnitude)
                mask_target = "subject"
                is_global = False
            # SAFETY semántico (sin hardcode): si el LLM dice fondo pero la magnitud
            # semántica es moderada-alta (>=0.60), es contradicción — los edits de fondo
            # reales son de mag baja. Se confía en la magnitud, no en el target LLM.
            if mask_target == "background" and new_mag >= 0.80 and magnitude >= 0.60:
                print(f"[ImgEditor] Rewriter target=background IGNORADO (mag sem={magnitude:.2f}, rw={new_mag:.2f}) → subject + global", flush=True)
                mask_target = "subject"
                is_global = True
            print(
                f"[ImgEditor] Rewriter: mag={new_mag:.2f}, target={mask_target}, "
                f"regions={affected_regions}, global={is_global}, regen={requires_scene_regeneration}"
            )
            return (new_prompt, new_mag, mask_target, is_global, True,
                    requires_scene_regeneration, affected_regions, subject_query)
        except Exception as e:
            print(f"[ImgEditor] Rewriter omitido: {e}")
            return prompt, magnitude, "subject", True, False, False, [], ""

    @staticmethod
    def _apply_pov_hint(prompt_enhanced: str, analysis, quality_only: bool = False) -> str:
        """Descriptores de vista para POV (idempotente: no duplica)."""
        try:
            if analysis.get("angle_change", False) and not quality_only:
                _pl = (prompt_enhanced or "").lower()
                _view_hint = ""
                if any(k in _pl for k in ["detras", "detrás", "espaldas", "behind", "back view"]):
                    _view_hint = ", view from behind, back of head and back visible, no face visible"
                elif any(k in _pl for k in ["lado", "perfil", "side", "profile"]):
                    _view_hint = ", side profile view, face in profile"
                elif any(k in _pl for k in ["arriba", "cenital", "top", "above", "aerial", "overhead"]):
                    _view_hint = ", high angle top-down view"
                elif any(k in _pl for k in ["abajo", "picado", "low angle", "below"]):
                    _view_hint = ", low angle view from below"
                if _view_hint and _view_hint.split(",")[1].strip() not in _pl:
                    prompt_enhanced = (prompt_enhanced or "").rstrip(". ") + _view_hint
                    print(f"[ImgEditor] POV hint añadido:{_view_hint}", flush=True)
        except Exception:
            pass
        return prompt_enhanced

    def _compose_generation_prompt(
        self,
        user_prompt: str,
        img_context: str = "",
        engine: str = "",
        magnitude: float = 0.5,
        quality_only: bool = False,
        has_quality_request: bool = False,
        enhance_tier: str = "hd",
        high_structural: bool = False,
        structural_bias: str = "neutral",
        body_transform: bool = False,
        multi_region_edit: bool = False,
        mask_target: str = "subject",
        quality_hybrid: bool = False,
        detected_age: int = None,
        shot_type: str = "unknown",
        body_direction: str = "change",
        body_intensity: float = 0.0,
        angle_change: bool = False,
        pose_change: bool = False,
    ) -> str:
        user_prompt = (user_prompt or "").strip()
        img_context = (img_context or "").strip()

        is_longcat = engine in ("longcat", "longcat_full", "imagine", "kontext") or "longcat" in engine.lower()
        is_hart = engine == "hart"
        mask_target = (mask_target or "subject").lower()
        if is_longcat:
            raw = (user_prompt or img_context or "").strip()
            if quality_only:
                tier = (enhance_tier or "hd").lower()
                base = (
                    f"Quality enhance tier={tier} (classical pipeline — no generative edit)"
                )
                print(f"[ImgEditor] Modo mejora tier={tier}: sin prompt generativo (preserva foto)")
                return base
            if body_transform:
                # Para body_transform, umbral bajo para que cambios fuertes se apliquen aún con mag < 0.80.
                # mag=0.78 → usa el template "strongest" en lugar de "medium".
                # v5.1093 — preferencia usuario: la cara PUEDE cambiar, lo que importa
                # es que se cumpla el prompt. Sin "same face/person/age" que ate al modelo.
                # v5.1119 — RÉGIMEN BYPASS (mag>=0.75 + longcat-family): sin ancla de
                # latentes, el TEXTO es la única autoridad. El wrapper antiguo
                # ("Keep the same background and composition... clothing change",
                # v5.1096 para kontext EDIT anclado) prohíbe LITERALMENTE el cambio
                # de postura/composición pedido (log 07:53: pose+acción → "keep
                # composition" ganó → sin cambio). Además "clothing change" es
                # falso para pedidos de pose/acción. En bypass: preservar personas/
                # identidad/ubicación, NUNCA la composición (el contenido destino
                # no existe en la fuente). En régimen anclado (mag<0.75) el wrapper
                # antiguo se queda (ahí la composición SÍ se mantiene).
                _bypass_compose = magnitude >= 0.75 and is_longcat
                if _bypass_compose:
                    # v5.1122 — el framing SOLO puede cambiar si el analyzer detectó
                    # señal estructural de vista/postura (flag, no vocabulario).
                    # Log 08:41 ('EL CHICO ES CALVO'): el wrapper antiguo permitía
                    # framing libre en TODO bypass → el modelo reencuadró/zoom una
                    # petición que no lo pedía. Sin ancla de latentes el texto es
                    # la única autoridad: si el pedido no exige pose, se ancla.
                    # 2026-10-08: la cara puede cambiar. El cuarto, los objetos
                    # y la luz se quedan. Sin pose, también el encuadre.
                    if angle_change or pose_change:
                        base = f"Instruction: {raw}. Keep the same room, objects and lighting. The pose, framing and appearance may change as required by the instruction. The face of the edited person may change. Apply the requested change as completely, strongly and obviously as possible."
                    else:
                        base = f"Instruction: {raw}. Keep the same room, objects, lighting and the exact same framing, composition and camera. The face of the edited person may change. Apply the requested change as completely, strongly and obviously as possible."
                else:
                    base = f"Instruction: {raw}. Keep the same background and composition. Apply the requested change as completely, strongly and obviously as possible."
                # v5.1096 — sufijo de ESTADO final por dirección: los modelos img2img
                # obedecen mejor descripciones de estado ("está desnuda") que acciones
                # ("debe desnudarse"). El Klein 4B con ReferenceLatent conserva tanto
                # que sin esto regenera el mismo vestido (log 11:27, 2 runs vestida).
                # Una señal corporal fuerte también puede describir una región
                # local (p. ej. pies). Solo reforzar la desnudez total cuando el
                # concepto de exposición completa domina al local en una cláusula.
                if body_direction == "uncover" and float(body_intensity) >= 0.30:
                    try:
                        scorer = self._get_semantic_analyzer(full_ai=False)
                        full_exposure = any(
                            (scores := scorer.score_concepts(
                                self._requested_change_scope(clause) or clause
                            )).get("body_exposure_full", 0.0) >= 0.50
                            and scores.get("body_exposure_full", 0.0)
                            >= scores.get("body_exposure_local", 0.0) * 1.15
                            for clause in re.split(r"\s*[,;]\s*", raw) if clause.strip()
                        )
                    except Exception:
                        full_exposure = False
                    if full_exposure:
                        # v5.1216: en un compuesto el sujeto del sufijo debe
                        # ser el de la cláusula CORPORAL, no el primer sujeto
                        # del texto (log real 19:49: «el sofá debe ser gris,
                        # las paredes deben ser grises, la mujer debe ir
                        # desnuda» escribió «The sofa is fully nude» — el
                        # sofá nunca se pinto de gris y la zona salió dañada).
                        # Selección medida: la cláusula con mayor señal
                        # corporal del mismo analizador (sin listas nuevas).
                        _suffix_clause = raw
                        _suffix_clauses = self._split_compound_instruction(raw) or [raw]
                        if len(_suffix_clauses) > 1:
                            try:
                                def _body_signal(_clause):
                                    _scores = scorer.score_concepts(
                                        self._requested_change_scope(_clause)
                                        or _clause,
                                    )
                                    return max(
                                        float(_scores.get("attribute_body", 0.0)),
                                        float(_scores.get("attribute_uncover", 0.0)),
                                    )
                                _best_clause = max(_suffix_clauses, key=_body_signal)
                                # Si la cláusula corporal no nombra sujeto
                                # («…calvo e ir desnudo») se conserva el sujeto
                                # del texto entero, como hasta hoy.
                                _best_subject = (
                                    self._visual_subject_description(_best_clause)
                                    or ""
                                ).strip(" ,.;")
                                if _best_subject:
                                    _suffix_clause = _best_clause
                            except Exception:
                                _suffix_clause = raw
                        _subject = (
                            self._visual_subject_description(_suffix_clause) or ""
                        ).strip(" ,.;")
                        if _subject:
                            # Name the person: "The person" with two people in
                            # the frame is read as any of them.
                            base += (
                                f" {_subject[0].upper()}{_subject[1:]} is fully nude, "
                                "bare skin, no clothing at all."
                            )
                        else:
                            base += " The person is fully nude, bare skin, no clothing at all."
                    else:
                        print("[ImgEditor] Sufijo de desnudez total omitido: el alcance medido no es corporal completo", flush=True)
                elif body_direction == "cover" and float(body_intensity) >= 0.30:
                    base += " The person is fully dressed, body covered with warm clothing."
                elif body_transform:
                    print(f"[ImgEditor] Sufijo de estado omitido (dirección={body_direction}, intensity={float(body_intensity):.2f} — sin estado medido que describir; el texto del usuario manda)", flush=True)
            # v5.1131 — CADENA DE WRAPPERS: el body-branch de arriba es la AUTORIDAD
            # para body_transform (incluye el sufijo por dirección uncover/cover).
            # Antes esta cadena corría SIEMPRE después y SOBREESCRIBÍA el wrapper
            # body (el sufijo nude/dressed se perdía desde v5.1122b), y además
            # tenía una asignación duplicada que dejaba el fix v5.1126 de mid-tier
            # en CÓDIGO MUERTO (el "identity unchanged" viejo ganaba contra el
            # "strong likeness" desbloqueado en el rango 0.62-0.80).
            _free_framing = angle_change or pose_change
            if body_transform:
                pass  # wrapper body ya asignado arriba (autoridad, con sufijo por dirección)
            elif magnitude < 0.62:
                base = f"Instruction: Edit this exact photo: {raw}. Keep the face, identity, composition, background and style unchanged. Only apply the requested adjustment."
            elif magnitude < 0.75:
                # v5.1131: eliminado el duplicado muerto — gana el wrapper
                # desbloqueado de v5.1126 (la apariencia puede cambiar, la
                # composición se preserva; el lock de identidad peleaba contra
                # cambios de pelo/gafas/atributos en el rango anclado).
                base = f"Instruction: Edit this exact photo: {raw}. Keep the same people present with a strong likeness to them, and preserve the overall composition. The appearance may change as required by the instruction. Apply the change as completely as possible."
            elif _free_framing:
                # La vista o la postura pedida no existen en la foto. La escena sí.
                base = f"Instruction: Edit this exact photo: {raw}. Keep the same room, objects and lighting. The pose, framing and appearance may change as required by the instruction. The face of the edited person may change. Apply the requested change as completely, strongly and obviously as possible."
            else:
                # 2026-10-08: la cara puede cambiar. El cuarto, los objetos y la
                # cámara se quedan. "strong likeness" hacía que el modelo
                # conservara justo el atributo pedido.
                base = f"Instruction: Edit this exact photo: {raw}. Keep the same room, objects, lighting and the exact same framing, composition and camera. The face of the edited person may change. Apply the requested change as completely, strongly and obviously as possible."
            # Bypass sin ancla: el cambio pedido manda sobre el estado anterior.
            # Genérico, sin vocabulario (vale para pelo, ropa, pose, fondo...).
            # Sin esto, "strong likeness" hace que el modelo conserve justo el
            # atributo a cambiar (log 19:15: calvo no ejecutado, pelo conservado
            # por likeness con prompt corto "THE BOY IS BALD").
            # El uncover corporal siempre va en bypass (manager, sin umbral de
            # mag) → el enforcement aplica aunque mag<0.75 (log 20:46: mag=0.65
            # con wrapper anclado débil). La cláusula de "los demás intactos"
            # desambigua en fotos multi-cara (log 20:46: calva al otro personaje
            # con 2 caras y referente singular); con 1 cara es no-op.
            # 0.70 ya es una instrucción fuerte según el analizador visual. Dejar
            # la barrera en 0.75 dejaba fuera precisamente cambios de apariencia
            # inequívocos (p. ej. pelo) con mag=0.70: el wrapper de parecido
            # terminaba preservando el estado que se pidió cambiar.
            _bypass_enforce = (
                magnitude >= 0.70
                or (body_transform and body_direction == "uncover")
                or multi_region_edit
            )
            if _bypass_enforce and is_longcat and not quality_only:
                base += " The requested change must be clearly visible in the result; do not preserve the previous state of what was asked to change. Leave every other person in the photo exactly as they are."
            # Receta «calvo» medida (log real 21:51, edit_1792180391): ver
            # _bald_recipe_tail — pelo fuera, diadema conservada y cuero
            # cabelludo gris = «gorrito».
            if is_longcat:
                base += self._bald_recipe_tail(raw)
            print(f"[ImgEditor] Imagine/LongCat Prompt: {base}")
            age_hint = ""
            # Solo si claramente madura (InsightFace se equivoca ±10) y NUNCA en
            # body_transform (el hint de arrugas + edición corporal = envejecido).
            if detected_age is not None and detected_age >= 55 and not body_transform:
                age_hint = (
                    " The person appears to be around "
                    + str(detected_age)
                    + " years old. Preserve all natural age characteristics: "
                    "wrinkles, fine lines, skin texture, age spots, and mature appearance. "
                    "Do not smooth or remove any aging features."
                )
                print(f"[ImgEditor] Auto age-preservation (age={detected_age})")
            composition_hint = ""
            # v5.1119 — en bypass (mag>=0.75 + longcat) el framing puede cambiar
            # (poses/ángulos pedidos): el hint de composición prohíbe literalmente
            # el cambio (log 07:22: 'Maintain the same full-body framing' + pose →
            # sin cambio). El hint solo aplica en régimen anclado.
            if not body_transform and not (magnitude >= 0.75 and is_longcat):
                if shot_type == "closeup":
                    composition_hint = " This is a close-up photo. Maintain the same tight framing and composition."
                elif shot_type == "portrait":
                    composition_hint = " This is a portrait photo. Maintain the same framing and head/shoulders composition."
                elif shot_type == "fullbody":
                    composition_hint = " This is a full-body photo. Maintain the same full-body framing."
            return base + age_hint + composition_hint
        elif is_hart:
            # HART autoregressive - prompt as provided (user or context)
            base = user_prompt if user_prompt else img_context
            print(f"[ImgEditor] HART Autoregressive Prompt: {base}")
            return base
        elif img_context:
            base = f"{img_context}. {user_prompt}"
        else:
            base = user_prompt

        if base and not base.lower().startswith(("photo", "a photo", "a picture")):
            # Eliminado prefijo hardcoded 'Photo of' para mayor fidelidad al prompt del LLM
            pass

        return base

    def _apply_skin_tone_match(self, original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
        """Ajusta el tono de la piel generada para que coincida con la original"""
        orig_arr = np.array(original.convert("RGB"))
        gen_arr = np.array(generated.convert("RGB"))
        mask_arr = np.array(mask.convert("L")) / 255.0

        if mask_arr.shape[:2] != gen_arr.shape[:2]:
            import cv2
            mask_arr = cv2.resize(mask_arr, (gen_arr.shape[1], gen_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

        orig_arr_comp = orig_arr
        if orig_arr.shape[:2] != gen_arr.shape[:2]:
            import cv2
            orig_arr_comp = cv2.resize(orig_arr, (gen_arr.shape[1], gen_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask_center = mask_arr > 0.8
        if not np.any(mask_center):
            mask_center = mask_arr > 0.5

        if np.any(mask_center):
            avg_orig = orig_arr_comp[mask_center].mean(axis=0)
            avg_gen = gen_arr[mask_center].mean(axis=0)

            correction = avg_orig / (avg_gen + 1e-6)
            correction = np.clip(correction, 0.8, 1.2)

            for i in range(3):
                gen_arr[:,:,i] = np.clip(gen_arr[:,:,i] * correction[i], 0, 255)

            return Image.fromarray(gen_arr.astype(np.uint8))
        return generated

    def _feather_mask(self, mask: Image.Image, amount: int = 15) -> Image.Image:
        """Aplica un borde suave a la máscara"""
        from PIL import ImageFilter
        return mask.filter(ImageFilter.GaussianBlur(radius=amount))

    def _mask_coverage(self, mask: Image.Image) -> float:
        arr = np.array(mask.convert("L"))
        return float((arr > 128).sum()) / float(arr.size)

    def _mask_from_face_bboxes(
        self,
        image: Image.Image,
        faces,
        width_mult: float = 4.0,
        height_mult: float = 6.0,
        head_pad: float = 0.5,
    ) -> Optional[Image.Image]:
        w, h = image.size
        union = np.zeros((h, w), dtype=np.uint8)
        for face_obj in faces:
            x1, y1, x2, y2 = [int(v) for v in face_obj.bbox]
            fw, fh = max(1, x2 - x1), max(1, y2 - y1)
            cx = (x1 + x2) // 2
            bw = int(fw * width_mult)
            top = max(0, y1 - int(fh * head_pad))
            bottom = min(h, top + int(fh * height_mult))
            left = max(0, cx - bw // 2)
            right = min(w, cx + bw // 2)
            union[top:bottom, left:right] = 255
        if union.max() == 0:
            return None
        union = cv2.GaussianBlur(union, (31, 31), 0)
        return Image.fromarray(union).convert("L")

    def _create_mask_from_face_detection(
        self,
        image: Image.Image,
        target: str = "subject",
    ) -> Optional[Image.Image]:
        """Genera una máscara geométrica a partir de detección facial real.
        No depende del prompt; usa solo la geometría de la imagen detectada.
        """
        try:
            import roop.face_util as face_util
        except Exception:
            return None
        try:
            img_np = np.array(image)
            if img_np.ndim == 3 and img_np.shape[2] == 4:
                img_np = img_np[:, :, :3]
            faces_data = face_util.extract_face_images(
                img_np,
                target_face_detection=True,
            )
            if not faces_data:
                return None
            faces = [fd.get("face") for fd in faces_data if fd.get("face") is not None]
            if not faces:
                return None

            w, h = image.size
            if target == "face":
                return self._mask_from_face_bboxes(
                    image, faces, width_mult=2.2, height_mult=3.0, head_pad=0.35
                )
            if target == "hair":
                return self._mask_from_face_bboxes(
                    image, faces, width_mult=2.8, height_mult=4.0, head_pad=1.35
                )
            if target == "clothes":
                return self._mask_from_face_bboxes(
                    image, faces, width_mult=3.2, height_mult=6.5, head_pad=0.45
                )
            # subject/fallback
            return self._mask_from_face_bboxes(
                image, faces, width_mult=3.0, height_mult=5.5, head_pad=0.5
            )
        except Exception:
            return None

    @staticmethod
    def _focus_on_measured_subject(
        focus: Image.Image, region: np.ndarray,
    ) -> Optional[Image.Image]:
        """Keep focus pixels that sit on the subject already measured."""
        focal = np.asarray(focus.convert("L"))
        subject = region
        if subject.shape[:2] != focal.shape[:2]:
            subject = cv2.resize(
                subject, (focal.shape[1], focal.shape[0]), interpolation=cv2.INTER_LINEAR,
            )
        if getattr(subject, "ndim", 2) == 3:
            subject = subject[:, :, 0]
        kept = np.where(subject >= 96, focal, 0).astype(np.uint8)
        if int((kept >= 160).sum()) < 16:
            return None
        return Image.fromarray(kept, "L")

    def _head_mask_from_region_plan(
        self, image: Image.Image, faces, region_mask: Image.Image,
        continue_region: bool = True,
    ) -> Optional[Image.Image]:
        """Reduce una región visual de persona a la cabeza que le corresponde.

        El rewriter identifica una región textual y CLIPSeg la mide en la foto;
        este paso usa solo la superposición medida con las caras detectadas para
        seleccionar UNA cabeza. No interpreta palabras, sexo ni posiciones.
        """
        try:
            if not faces or region_mask is None:
                return None
            region = np.asarray(region_mask.convert("L"))
            h, w = region.shape[:2]
            best_face, best_score, second_score = None, 0.0, 0.0
            for face_obj in faces:
                x1, y1, x2, y2 = [int(round(float(v))) for v in face_obj.bbox]
                fw, fh = max(1, x2 - x1), max(1, y2 - y1)
                px, py = int(fw * 0.25), int(fh * 0.65)
                ax1, ay1 = max(0, x1 - px), max(0, y1 - py)
                ax2, ay2 = min(w, x2 + px), min(h, y2 + int(fh * 0.15))
                patch = region[ay1:ay2, ax1:ax2]
                score = float((patch > 128).mean()) if patch.size else 0.0
                if score > best_score:
                    second_score = best_score
                    best_face, best_score = face_obj, score
                elif score > second_score:
                    second_score = score
            if best_face is None or best_score < 0.08:
                print(f"[ImgEditor] Región focal sin cabeza asociable (score={best_score:.2f})", flush=True)
                return None
            # Con varias caras, seleccionar la mejor sin una separación medible
            # cambia personas al azar cuando la consulta no quedó bien grounded.
            # La decisión usa solo la diferencia de cobertura visual.
            if len(faces) > 1 and (best_score - second_score) < 0.12:
                print(
                    f"[ImgEditor] Región focal ambigua entre caras (mejor={best_score:.2f}, segunda={second_score:.2f})",
                    flush=True,
                )
                return None
            print(f"[ImgEditor] Región focal por solape medido (score={best_score:.2f})", flush=True)

            focus = self._head_mask_from_faces(image, [best_face])
            if focus is None or not continue_region:
                return focus
            # A close portrait makes half the focus span cover the body and
            # the person beside it. That mask no longer fits the local window,
            # and sampling it replaces the photo. Keep the widest continuation
            # that still fits; otherwise edit only the focus already measured.
            return self._fit_measured_focus(image, focus, region)
        except Exception as exc:
            print(f"[ImgEditor] Máscara de cabeza focal omitida: {exc}", flush=True)
            return None

    @staticmethod
    def _complete_localized_region_mask(
        seed_mask: Image.Image, region_mask: Optional[Image.Image] = None,
    ) -> Image.Image:
        """Cover the complete measured region before pasting a localized edit.

        A tight segmentation or face geometry can cut through the edited object.
        Keep only semantic components touching the grounded seed, then give their
        outline a small image-scaled margin. This depends on the visual masks,
        not the requested attribute or wording.
        """
        seed = np.asarray(seed_mask.convert("L")) >= 128
        if not seed.any():
            return seed_mask.convert("L")
        selected = seed.copy()
        if region_mask is not None:
            region = np.asarray(region_mask.convert("L").resize(
                seed_mask.size, Image.Resampling.BILINEAR,
            )) >= 96
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                region.astype(np.uint8), connectivity=8,
            )
            for index in range(1, count):
                component = labels == index
                area = int(stats[index, cv2.CC_STAT_AREA])
                overlap = int((component & seed).sum())
                if (area >= 12 and overlap >= max(12, int(area * 0.05))
                        and area <= max(int(seed.sum() * 4), int(seed.size * 0.01))):
                    selected |= component
        ys, xs = np.where(selected)
        height = int(ys.max() - ys.min() + 1)
        width = int(xs.max() - xs.min() + 1)
        radius = max(2, min(
            int(round(min(width, height) * 0.15)),
            int(round(min(seed_mask.size) * 0.028)),
        ))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
        )
        completed = cv2.dilate(selected.astype(np.uint8), kernel) * 255
        completed = cv2.GaussianBlur(completed, (0, 0), sigmaX=max(2, radius * 0.4))
        return Image.fromarray(completed, "L")

    @staticmethod
    def _continue_measured_region(
        focal: Optional[Image.Image], region: np.ndarray,
        pad_fraction: float = 0.5, report: bool = True,
    ) -> Optional[Image.Image]:
        """Include the measured region that continues past a tight focus.

        The focus is whatever was already measured. A face box, a partial
        object, or any other seed can cut that region. The extra pixels are
        the same region within a fraction of the focus's own span, in every
        direction, and connected to the focus. Farther pixels of that region
        stay out. No edit-type word list.
        """
        if focal is None:
            return None
        focal_arr = np.asarray(focal.convert("L"))
        core = focal_arr >= 160
        ys, xs = np.where(core)
        if not len(ys):
            return focal
        if region.shape[:2] != core.shape[:2]:
            return focal
        # A full-resolution ellipse whose span is hundreds of pixels makes
        # cv2.dilate sit there with no further log. The same geometry on a
        # long side of 768 is the continuation the editor already uses.
        height, width = core.shape[:2]
        long_side = max(height, width)
        scale = 1.0 if long_side <= 768 else 768.0 / float(long_side)
        if scale < 1.0:
            work_w = max(1, int(round(width * scale)))
            work_h = max(1, int(round(height * scale)))
            core_work = cv2.resize(
                core.astype(np.uint8) * 255, (work_w, work_h),
                interpolation=cv2.INTER_NEAREST,
            ) >= 128
            region_work = cv2.resize(
                (region >= 96).astype(np.uint8) * 255, (work_w, work_h),
                interpolation=cv2.INTER_NEAREST,
            ) >= 128
        else:
            core_work = core
            region_work = region >= 96
        ys, xs = np.where(core_work)
        if not len(ys):
            return focal
        span = max(
            1,
            int(ys.max() - ys.min() + 1),
            int(xs.max() - xs.min() + 1),
        )
        pad = max(2, int(span * pad_fraction))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
        near = cv2.dilate(core_work.astype(np.uint8), kernel) > 0
        extra = near & region_work & ~core_work
        if not extra.any():
            return focal
        count, labels = cv2.connectedComponents((extra | core_work).astype(np.uint8), connectivity=8)
        keep = np.zeros(core_work.shape, dtype=bool)
        for label in set(int(v) for v in labels[core_work] if v):
            keep |= labels == label
        if scale < 1.0:
            keep = cv2.resize(
                keep.astype(np.uint8) * 255, (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) >= 128
        merged = np.where(keep, 255, 0).astype(np.uint8)
        merged = cv2.GaussianBlur(merged, (31, 31), 0)
        if report:
            added = int((keep & ~core).sum())
            print(
                f"[ImgEditor] La zona medida sigue a la región (+{added} px); el resto queda fuera",
                flush=True,
            )
        return Image.fromarray(merged, "L")

    def _fit_measured_focus(
        self, image: Image.Image, focus: Image.Image, region: np.ndarray,
    ) -> Image.Image:
        """Widest continuation of an already measured focus that still fits."""
        for fraction in (0.5, 0.25, 0.12):
            trial = self._continue_measured_region(
                focus, region, pad_fraction=fraction, report=False,
            )
            if trial is None:
                return focus
            if self._crop_focal_context(image, trial, report=False) is None:
                continue
            kept = self._continue_measured_region(
                focus, region, pad_fraction=fraction, report=True,
            )
            if fraction < 0.5:
                print(
                    "[ImgEditor] La continuación medida se recorta a la ventana local; "
                    "el resto de la foto se conserva",
                    flush=True,
                )
            return kept if kept is not None else focus
        print(
            "[ImgEditor] La continuación medida no cabe en la ventana local; "
            "se edita solo el foco medido",
            flush=True,
        )
        return focus

    def _visual_request_queries(self, texts) -> list:
        """Visual remainder of each clause, after the grammatical predicate.

        The segmenter is asked this text. It is not an edit-type list: a
        different instruction produces a different query, and pixels decide
        whether that query exists in the photo.
        """
        queries = []
        seen = set()
        copula = re.compile(
            r"^(?:be|being|been|have|having|has|ser|estar|tener)\b\s*", re.I,
        )
        article = re.compile(r"^(?:a|an|the|el|la|los|las|un|una|unos|unas)\b\s*", re.I)
        for text in texts:
            if not text or not str(text).strip():
                continue
            scope = self._requested_change_scope(str(text))
            pieces = [part.strip() for part in re.split(r"\s*[,;]\s*", scope) if part.strip()]
            for piece in pieces:
                cleaned = copula.sub("", piece).strip(" .,!?;:\"'")
                cleaned = article.sub("", cleaned)
                cleaned = re.sub(r"\s+", " ", cleaned)
                key = cleaned.casefold()
                if len(cleaned) < 2 or key in seen:
                    continue
                seen.add(key)
                queries.append(cleaned)
        return queries

    @staticmethod
    def _singular_surface_query(query: str) -> str:
        """Singular de una frase de superficie con artículo (solo morfología).

        CLIPSeg responde muy distinto al plural y al singular sobre la MISMA
        foto: 'the walls' midió 2,0 % y 'the wall' 13,1 % (pared del 15/10).
        Solo se transforma la última palabra detrás de un artículo; frases
        sin artículo ('de color gris', 'grey') quedan intactas.
        """
        match = re.match(
            r"^((?:the|a|an|el|la|los|las|un|una|unos|unas)\s+)(.+)$",
            query.strip(),
            re.I,
        )
        if not match:
            return query
        article, rest = match.group(1), match.group(2)
        words = rest.split()
        if not words:
            return query
        last = words[-1]
        if article.strip().casefold() in ("the", "a", "an"):
            if (len(last) > 3 and last.endswith("s")
                    and not last.endswith(("ss", "us", "is"))):
                words[-1] = last[:-1]
        else:
            lowered = article.strip().casefold()
            if lowered in ("los", "las"):
                article = "el " if lowered == "los" else "la "
            if len(last) > 4 and last.endswith("es"):
                words[-1] = last[:-2]
            elif len(last) > 3 and last.endswith("s"):
                words[-1] = last[:-1]
        singular = article + " ".join(words)
        return query if singular.casefold() == query.strip().casefold() else singular

    @staticmethod
    def _determiner_phrases(text: str) -> list:
        """Determiner-led noun phrases of a clause (grammar, no vocabulary).

        An imperative carries its subject after the verb ("pinta LAS PAREDES
        de gris") and has no copula for the subject extractor to split, so
        only the full sentence reached CLIPSeg (1,33 % vs 13,1 % de "the
        wall" sobre la misma foto del 15/10).
        """
        return list(dict.fromkeys(
            f"{match.group(1)}{match.group(2)}"
            for match in re.finditer(
                r"\b((?:the|a|an|el|la|los|las|un|una)\s+)(\w+)",
                text or "",
                flags=re.I,
            )
        ))

    @staticmethod
    def _imperative_change(text: str) -> str:
        """Trailing complement after a mid-clause noun phrase (grammar only).

        "pinta las paredes de gris" has no copula for the scope splitter, so
        the full sentence reached the request measurement (1,33 % on the
        15/10 wall photo) and the repaint got a sliver of the wall. The words
        after the determiner-led phrase are the requested change; function
        words are dropped and the user's own content words stay.
        """
        clause = (text or "").strip()
        phrase = None
        for match in re.finditer(
            r"\b((?:the|a|an|el|la|los|las|un|una)\s+)(\w+)", clause, flags=re.I,
        ):
            phrase = match
        if phrase is None or phrase.start() == 0:
            return ""
        tail = clause[phrase.end():].strip(" ,.;:!?")
        tail = re.sub(
            r"^(?:de|del|en|in|into|to|with|a|an|of|on|por|para)\b\s*",
            "", tail, flags=re.I,
        )
        return tail.strip(" ,.;:!?")

    def _request_region_on_subject(
        self, image: Image.Image, subject_mask: Image.Image, texts,
    ) -> Optional[Image.Image]:
        """Region the segmenter assigns to the request, on this subject.

        The same measurement serves any instruction. A query that lights up
        the whole subject is the person, not the part. A shoe or any other
        object sticks out of the body silhouette: the component that touches
        the person is kept whole. Clipping it to the silhouette left the
        skin overlap and dropped the object (2026-10-08, dark shoe luma 38,
        sent mask 62% skin).
        """
        queries = self._visual_request_queries(texts)
        if not queries or subject_mask is None:
            return None
        try:
            masker = self._get_clipseg_masker()
        except Exception:
            return None
        if masker is None or not hasattr(masker, "generate_mask"):
            return None
        subject = np.asarray(subject_mask.convert("L"))
        if subject.shape[:2] != (image.height, image.width):
            subject = cv2.resize(
                subject, image.size, interpolation=cv2.INTER_LINEAR,
            )
        subject_on = subject >= 96
        subject_area = int(subject_on.sum())
        if subject_area < 48:
            return None
        best = None
        best_score = -1.0
        best_query = ""
        image_area = max(1, image.width * image.height)
        photo_rgb = np.asarray(image.convert("RGB"))
        if photo_rgb.shape[:2] != subject_on.shape:
            photo_rgb = cv2.resize(
                photo_rgb, (subject_on.shape[1], subject_on.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        skin = _skin_pixels(photo_rgb)
        for query in queries:
            try:
                candidate = masker.generate_mask(image, query)
            except Exception:
                continue
            if candidate is None:
                continue
            raw = np.asarray(candidate.convert("L"))
            if raw.shape[:2] != subject_on.shape:
                raw = cv2.resize(raw, image.size, interpolation=cv2.INTER_LINEAR)
            found = raw >= 96
            overlap_area = int((found & subject_on).sum())
            if overlap_area < max(48, int(subject_area * 0.005)):
                continue
            count, labels = cv2.connectedComponents(
                found.astype(np.uint8), connectivity=8,
            )
            keep = np.zeros(found.shape, dtype=bool)
            for label in range(1, count):
                component = labels == label
                touched = int((component & subject_on).sum())
                if touched < 48:
                    continue
                coverage = int(component.sum()) / image_area
                part = touched / subject_area
                if coverage > 0.35 or part > 0.55:
                    continue
                keep |= component
            area = int(keep.sum())
            if area < 48:
                continue
            # A skin-colored speck is the person, not a separate object.
            # The 2026-10-08 shoe hit kept that speck (62% skin) and dropped
            # the dark object outside the silhouette.
            skin_share = float((keep & skin).sum()) / float(area)
            part = int((keep & subject_on).sum()) / subject_area
            if skin_share >= 0.50 and part < 0.08:
                print(
                    f"[ImgEditor] '{query}' cae en piel del sujeto "
                    f"({skin_share:.0%}, {part:.1%} de la persona); no es una parte",
                    flush=True,
                )
                continue
            score = (1.0 - skin_share) + min(area / image_area, 0.2)
            if score > best_score:
                best_score = score
                best = keep
                best_query = query
        if best is None:
            return None
        ys, xs = np.where(best)
        span = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
        radius = max(1, int(round(span * 0.02)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
        )
        grown = cv2.dilate(best.astype(np.uint8), kernel) > 0
        if int(grown.sum()) < 48:
            return None
        outside = int((grown & ~subject_on).sum())
        print(
            f"[ImgEditor] Zona pedida '{best_query}' medida "
            f"({grown.mean():.1%}, {outside} px fuera de la silueta)",
            flush=True,
        )
        return Image.fromarray(grown.astype(np.uint8) * 255, "L")

    @staticmethod
    def _complete_hair_surface(image: Image.Image, seed: np.ndarray,
                               subject: np.ndarray, faces) -> np.ndarray:
        """Rest of the hair of the head the measured peak already belongs to.

        The segmenter lights the lit scalp. The hair behind and under it sits
        in shadow at another tone, so the same-color continuation never reaches
        it and the sample leaves dark patches beside the ear and at the nape.
        The rest of that hair is measured, not drawn: the peak must carry two
        tones itself, and the darker half of the head around it is the hair.
        Only pieces touching the peak are kept, the face the peak belongs to
        stays out below the eye line, and a face the peak never touched keeps
        whatever is closer to itself than to the peak.
        """
        empty = np.zeros(seed.shape, dtype=bool)
        if subject is None or int(seed.sum()) < 48 or int(subject.sum()) < 48:
            return empty
        ys, xs = np.where(seed)
        span = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
        photo = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = seed.shape
        # The head around the peak: reach scales with the peak itself, so a
        # small face keeps a small neighbourhood.
        hood_r = max(40, int(round(span * 0.35)))
        hood = cv2.dilate(
            seed.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (hood_r * 2 + 1, hood_r * 2 + 1)
            ),
        ) > 0
        region = hood & subject
        if int(region.sum()) < 48:
            return empty
        luma = np.clip(photo @ np.array([0.299, 0.584, 0.114], dtype=np.float32), 0, 255)
        luma = luma.astype(np.uint8)
        # The peak has to carry two tones itself: the lit surface and the hair
        # lying on it. A peak painted in one flat colour has no second surface
        # to complete, and following the darker half of the frame around it
        # would pull in a dark background that was never hair.
        seed_tone = float(cv2.threshold(
            np.ascontiguousarray(luma[seed]), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0])
        hair_seed = seed & (luma <= seed_tone)
        lit_seed = seed & ~hair_seed
        if int(hair_seed.sum()) < max(48, int(seed.sum()) * 0.15) or int(lit_seed.sum()) < 48:
            return empty
        # The head around the peak is split by tone measured on the hood
        # itself: first the lit surface of the hood, then the dark half of
        # that surface. The peak decides which side of that split its own
        # hair lies on: a dark head keeps the dark half as the limit, and a
        # light head points at the next split above, so hair of any colour
        # stays inside while the room around the head stops at the colour
        # change. That limit also caps the region's own split: a geometric
        # subject carries the room inside its ellipse and would pass that
        # room as hair. The face below the eye line and the nearest other
        # person are cut afterwards.
        hood_tone = float(cv2.threshold(
            np.ascontiguousarray(luma[hood]), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0])
        dark_hood = hood & (luma <= hood_tone)
        hair_limit = None
        if int(dark_hood.sum()) >= 48:
            dark_limit = float(cv2.threshold(
                np.ascontiguousarray(luma[dark_hood]), 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )[0])
            hair_tone = float(np.median(luma[hair_seed]))
            # An Otsu edge is an integer and the peak's tone is a float
            # median: a gap of less than one level is the same tone, not
            # a lighter head, or a hand in the same frame would count.
            # A dark half split at zero is one flat colour: there is no
            # second surface in the hood to follow, and forcing a split
            # there would open the flood to whatever the dark buries.
            if dark_limit <= 0.0:
                hair_limit = dark_limit
            elif hair_tone <= dark_limit + 2.0:
                hair_limit = dark_limit
            else:
                light_hood = hood & (luma > dark_limit)
                if int(light_hood.sum()) >= 48:
                    hair_limit = float(cv2.threshold(
                        np.ascontiguousarray(luma[light_hood]), 0, 255,
                        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                    )[0])
        threshold = float(cv2.threshold(
            np.ascontiguousarray(luma[region]), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )[0])
        if hair_limit is not None:
            threshold = min(threshold, hair_limit)
        dark = region & (luma <= threshold)
        if int(dark.sum()) < 48:
            return empty
        # The strands the split left between them are still hair: they are
        # surrounded by the hair itself, and the edit has to cover them or
        # the original strands show through the new surface. Inside the peak
        # the measured surface already decided what stays.
        gaps = (~dark).astype(np.uint8)
        frame = np.zeros((height + 2, width + 2), dtype=np.uint8)
        gaps[0, 0] = 1
        cv2.floodFill(gaps, frame, (0, 0), 0)
        dark |= gaps.astype(bool) & ~seed
        dark &= region
        # A geometric subject stops at the hairline and would cut the back
        # hair, so the hood carries that hair as long as it keeps that same
        # measured split; what the split rejects stays out of the region too.
        if hair_limit is not None:
            grown = hood & (luma <= hair_limit) & ~seed
            if int(grown.sum()) >= 48:
                dark |= grown
                # Strands brighter than the split but closed in by the hair
                # around them are that hair too: nothing open reaches the
                # frame through them, while the room around the head stays
                # connected to the border and out.
                gaps = (~dark).astype(np.uint8)
                frame = np.zeros((height + 2, width + 2), dtype=np.uint8)
                gaps[0, 0] = 1
                cv2.floodFill(gaps, frame, (0, 0), 0)
                dark |= gaps.astype(bool) & ~seed
                dark &= hood
        # Skin the shadow caught with the hair: a hand resting on the neck
        # and the neck itself are not hair, and painting them repaints
        # somebody else or leaves a mark where nothing was asked for.
        for face in faces or []:
            x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            strip = photo[y1:y1 + max(1, (y2 - y1) // 3), x1:x2].reshape(-1, 3)
            if not strip.size:
                continue
            strip_luma = strip @ np.array([0.299, 0.584, 0.114], dtype=np.float32)
            # A box lying in shadow carries no skin colour to learn from.
            if float(np.median(strip_luma)) < 80.0:
                continue
            lit = strip[strip_luma >= np.percentile(strip_luma, 40)]
            skin_tone = np.median(lit if len(lit) else strip, axis=0)
            dark &= np.linalg.norm(photo - skin_tone, axis=2) > 40.0
            if int(dark.sum()) < 48:
                return empty
        if int(dark.sum()) < 48:
            return empty
        # A face the peak never touched owns its own hair: those pixels stay
        # with whoever is closer to that face than to the measured peak.
        foreign = np.zeros(seed.shape, dtype=bool)
        for face in faces or []:
            x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if not int(seed[y1:y2, x1:x2].sum()):
                foreign[y1:y2, x1:x2] = True
        if int(foreign.sum()):
            to_peak = cv2.distanceTransform((~seed).astype(np.uint8), cv2.DIST_L2, 5)
            to_other = cv2.distanceTransform((~foreign).astype(np.uint8), cv2.DIST_L2, 5)
            dark &= to_peak <= to_other
            if int(dark.sum()) < 48:
                return empty
        near_r = max(3, int(round(span * 0.02)))
        near = cv2.dilate(
            seed.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (near_r * 2 + 1, near_r * 2 + 1)
            ),
        ) > 0
        _count, labels = cv2.connectedComponents(dark.astype(np.uint8), connectivity=8)
        touch = set(int(value) for value in labels[near & dark] if value)
        if not touch:
            return empty
        keep = np.isin(labels, sorted(touch))
        protect = np.zeros(seed.shape, dtype=bool)
        for face in faces or []:
            x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if int(seed[y1:y2, x1:x2].sum()):
                # This is the head being edited: its own face keeps the skin
                # below the eye line, where brows, eyes and beard live.
                eye_y = None
                try:
                    kps = np.asarray(getattr(face, "kps", None), dtype=np.float32)
                    if kps.shape[0] >= 3 and kps.shape[1] >= 2:
                        eye_y = float((kps[0, 1] + kps[1, 1]) * 0.5)
                except Exception:
                    eye_y = None
                face_top = y1 if eye_y is None else max(y1, int(round(eye_y)))
                protect[face_top:y2, x1:x2] = True
            else:
                protect[y1:y2, x1:x2] = True
        keep &= ~protect
        if int(keep.sum()) < 48:
            return empty
        return keep

    def _complete_request_surface(
        self,
        image: Image.Image, seed_mask: Image.Image,
        subject_mask: Image.Image, faces,
        masker=None,
        negative_query=None,
        positive_query=None,
        complete_hair: bool = True,
    ) -> Optional[Image.Image]:
        """Extend a thin measured peak through pixels of the same color.

        The segmenter often lights only the strongest part of a surface.
        Editing that peak alone leaves the rest, and the sample paints a
        patch that still matches what was left around it. The rest of the
        surface is whatever shares that color, lighter or darker, and stays
        connected to the peak inside the subject. The zone stops where the
        color changes. A peak that already covers the surface, or a subject
        that is all one color, stays as measured.

        The hair of that same head is completed apart from the color
        (_complete_hair_surface): it sits in another tone, so following color
        alone leaves dark patches beside the ear and at the nape.
        """
        if seed_mask is None or subject_mask is None:
            return None
        # Color distance and connected components on a 43 MP frame stall the
        # edit. The same 768 px work size used by the other mask math is
        # enough; the mask is scaled back. Probes under that size are unchanged.
        long_side = max(image.size)
        if long_side > 768:
            scale = 768.0 / float(long_side)
            work_size = (
                max(1, int(round(image.width * scale))),
                max(1, int(round(image.height * scale))),
            )
            small_faces = []
            for face in faces or []:
                box = tuple(float(value) * scale for value in face.bbox)
                scaled = SimpleNamespace(bbox=box)
                landmarks = getattr(face, "kps", None)
                if landmarks is not None:
                    scaled.kps = np.asarray(landmarks, dtype=np.float32) * scale
                small_faces.append(scaled)
            found = self._complete_request_surface(
                image.resize(work_size, Image.Resampling.BOX),
                seed_mask.resize(work_size, Image.Resampling.NEAREST),
                subject_mask.resize(work_size, Image.Resampling.NEAREST),
                small_faces,
                masker=masker,
                negative_query=negative_query,
                positive_query=positive_query,
                complete_hair=complete_hair,
            )
            if found is None:
                return None
            return found.resize(image.size, Image.Resampling.NEAREST)
        seed = np.asarray(seed_mask.convert("L").resize(image.size, Image.Resampling.NEAREST)) >= 128
        subject = np.asarray(subject_mask.convert("L").resize(image.size, Image.Resampling.NEAREST)) >= 96
        if int(seed.sum()) < 48 or int(subject.sum()) < 48:
            return None

        def finish(base):
            """The measured continuation plus the hair of the same head."""
            hair = (
                ImgEditorManager._complete_hair_surface(image, seed, subject, faces)
                if complete_hair else np.zeros(seed.shape, dtype=bool)
            )
            zone = base | hair
            if int(zone.sum()) <= int(seed.sum()):
                print(
                    "[ImgEditor] El segmentador no midió color ni pelo que "
                    "añadir a la zona; se queda como midió",
                    flush=True,
                )
                return None
            grown_color = int((base & ~seed).sum())
            if grown_color:
                print(
                    f"[ImgEditor] La zona pedida sigue por la misma superficie (+{grown_color} px); "
                    "donde el color cambia queda fuera",
                    flush=True,
                )
            grown_hair = int((hair & ~seed).sum())
            if grown_hair:
                print(
                    f"[ImgEditor] El pelo de la misma cabeza entra en la zona (+{grown_hair} px); "
                    "la cara bajo la línea de ojos y las caras ajenas quedan fuera",
                    flush=True,
                )
            return Image.fromarray(zone.astype(np.uint8) * 255, "L")

        # A peak that already covers much of the person is the requested
        # region; growing it by color would replace the person. The hair of
        # the same head still has to be measured, or a bald edit keeps every
        # strand behind the peak (2026-10-08: peak 55.8% of the head ellipse
        # and the back hair never reached the zone).
        if int(seed.sum()) > int(subject.sum()) * 0.45:
            print(
                "[ImgEditor] El pico medido cubre casi todo el sujeto; "
                "sin crecimiento por color, solo entra el pelo medido",
                flush=True,
            )
            return finish(seed.copy())
        photo = np.asarray(image.convert("RGB"), dtype=np.float32)
        # The bright rim of a peak is the edge, or something lying on the
        # surface. The surface itself is the core tone. Extending the rim
        # paints over that occluder.
        ys, xs = np.where(seed)
        span = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
        erode_r = max(2, int(round(span * 0.06)))
        erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_r * 2 + 1, erode_r * 2 + 1))
        core = cv2.erode(seed.astype(np.uint8), erode_k) > 0
        if int(core.sum()) < 48:
            core = seed
        median_color = np.median(photo[core], axis=0)
        color_distance = np.linalg.norm(photo - median_color, axis=2)
        color_limit = max(float(np.percentile(color_distance[core], 70)), 28.0)
        surface = seed & (color_distance <= color_limit)
        if int(surface.sum()) < max(48, int(seed.sum()) * 0.40):
            surface = seed
        # Halfway to the next color, in either direction. No radius and no
        # face cut: a long surface is followed to its end, and skin stays out
        # only when its color differs. Another person's skin is removed
        # before the connection, so their hand cannot bridge into the zone.
        limit = _same_color_limit(color_distance, surface, subject)
        if limit is None:
            # One color over the subject: only this head's hair may join.
            return finish(seed.copy())
        # The locator can be smaller than the surface. On edit_1790893724 the
        # head ellipse stopped at the front hair and the same dark hair
        # continued behind it. The zone follows that color, connected to the
        # peak, and stops where the color changes.
        same = color_distance <= limit
        same |= surface
        # Halfway to the next color stays inside the subject, where it
        # completes a long surface. Outside, that halfway reached the wooden
        # headboard from a hair peak (fast_1791497310, limit 131, 22% of the
        # frame) and the model redrew the room. Out there only the peak's
        # own color joins; 18 is the flat separation this path already uses.
        outside = ~(subject | seed)
        if int((same & outside).sum()):
            close = color_distance <= 18.0
            dropped = int((same & outside & ~close).sum())
            same &= ~outside | close
            if dropped >= 48:
                print(
                    "[ImgEditor] Fuera de la persona el color a medias no entra "
                    f"(-{dropped} px); solo el color del pico",
                    flush=True,
                )
        same = ImgEditorManager._drop_other_person_contact(photo, same, subject, faces)
        same = ImgEditorManager._drop_measured_object(
            photo, same, subject, faces, masker, negative_query, positive_query,
        )
        touch = surface & same
        if int(touch.sum()) < 48:
            return finish(seed.copy())
        _count, labels = cv2.connectedComponents(same.astype(np.uint8), connectivity=8)
        keep = np.zeros(seed.shape, dtype=bool)
        for label in set(int(value) for value in labels[touch] if value):
            keep |= labels == label
        if int(keep.sum()) < int(surface.sum()) * 1.08:
            return finish(seed.copy())
        keep = ImgEditorManager._drop_other_person_contact(photo, keep, subject, faces)
        keep = ImgEditorManager._drop_measured_object(
            photo, keep, subject, faces, masker, negative_query, positive_query,
        )
        # The continuation may not paint a face the measured peak never
        # touched. On a 1248x832 frame the hair peak grew into the other
        # person's face (26958 skin px added, 55.6% of her face box) and the
        # zone reached 68% of the frame, so the crop was rejected and the
        # whole photo went to the model: both faces repainted. A face the peak
        # already carries is the one being edited and may come along.
        skin = _skin_pixels(photo)
        added_px = keep & ~seed
        blocked = False
        for face in faces or []:
            x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(photo.shape[1], x2), min(photo.shape[0], y2)
            if x2 <= x1 or y2 <= y1:
                continue
            face_skin = skin[y1:y2, x1:x2]
            face_skin_count = int(face_skin.sum())
            if face_skin_count < 48 or int((seed[y1:y2, x1:x2] & face_skin).sum()):
                continue
            foreign = int((added_px[y1:y2, x1:x2] & face_skin).sum())
            if foreign < 48:
                continue
            # A few edge pixels are the wall meeting a cheek. Dropping the
            # whole surface left a 5.7% wall unpainted (2026-10-08, 124 px).
            # Swallowing most of a face the peak never touched is the other
            # failure: that continuation is discarded.
            share = foreign / float(face_skin_count)
            if share >= 0.15:
                print(
                    f"[ImgEditor] La superficie continuada entra {foreign} px de piel de "
                    "una cara que la zona medida no tocó; se queda en la zona medida",
                    flush=True,
                )
                blocked = True
                break
            keep[y1:y2, x1:x2] &= ~skin[y1:y2, x1:x2]
            print(
                f"[ImgEditor] La superficie continuada roza {foreign} px de otra cara; "
                "se recorta ese borde y la superficie sigue",
                flush=True,
            )
        if blocked:
            # The same-color continuation runs over skin of any tone, so on a
            # frame where two people touch it swallows the other one whole.
            # The measured peak stands; only the hair of its own head may
            # still join it.
            keep = seed.copy()
        return finish(keep)

    @staticmethod
    def _drop_other_person_contact(photo, keep, subject, faces):
        """Keep another person's skin out of a surface that is not theirs.

        A hand or arm lying on the measured peak shares its edge. Editing
        that edge repaints the other person. Their face is the one that
        sits outside the measured subject; the skin connected to that face
        is left as it was.
        """
        if keep is None or not faces or int(keep.sum()) < 48:
            return keep
        height, width = keep.shape
        others = []
        for face in faces:
            x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            box = subject[y1:y2, x1:x2]
            if box.size and float(box.mean()) < 0.25:
                others.append((x1, y1, x2, y2))
        if not others:
            return keep
        long_side = max(height, width)
        scale = 768.0 / long_side if long_side > 768 else 1.0
        small_w = max(1, int(round(width * scale)))
        small_h = max(1, int(round(height * scale)))
        small = photo if scale == 1.0 else cv2.resize(photo, (small_w, small_h), interpolation=cv2.INTER_AREA)
        skin = _skin_pixels(small)
        _count, labels = cv2.connectedComponents(skin.astype(np.uint8), connectivity=8)
        protect = np.zeros((small_h, small_w), np.uint8)
        for x1, y1, x2, y2 in others:
            sx1 = min(small_w - 1, int(x1 * scale))
            sy1 = min(small_h - 1, int(y1 * scale))
            sx2 = min(small_w, max(sx1 + 1, int(round(x2 * scale))))
            sy2 = min(small_h, max(sy1 + 1, int(round(y2 * scale))))
            touched = {int(value) for value in labels[sy1:sy2, sx1:sx2].ravel() if value}
            for label in touched:
                component = labels == label
                if int(component.sum()) < 200:
                    continue
                protect[component] = 255
        if not int(protect.sum()):
            return keep
        # Only the skin pixels themselves. A wide dilate eats the measured
        # surface where the other person touches it, and that surface is
        # then left unchanged.
        full = protect if scale == 1.0 else cv2.resize(protect, (width, height), interpolation=cv2.INTER_NEAREST)
        trimmed = keep & ~(full > 0)
        if int(trimmed.sum()) < max(48, int(keep.sum()) * 0.55):
            return keep
        return trimmed

    @staticmethod
    def _drop_measured_object(photo, keep, subject, faces, masker, negative_query, positive_query=None):
        """Keep a measured object (e.g. chair) out of a surface that is not the requested object.

        Similar to _drop_other_person_contact but uses CLIPSeg to extract masks for
        specific objects (e.g. "chair" as negative, "plant" as positive) instead of
        MediaPipe face detection. This allows excluding specific objects while preserving
        the requested object in the surface.
        """
        if (keep is None or masker is None or int(keep.sum()) < 48
                or not (negative_query or positive_query)):
            return keep
        subject_mask = subject if isinstance(subject, np.ndarray) else np.asarray(subject)
        
        # The color field is a float array. The segmenter expects an image;
        # a raw array raises "Unexpected type numpy.ndarray" and the
        # exclusion never runs (2026-10-08, chair and plant, twice each).
        if isinstance(photo, Image.Image):
            photo_image = photo
        else:
            array = np.asarray(photo)
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)
            photo_image = Image.fromarray(array[..., :3] if array.ndim == 3 else array)

        def _fit(mask_image):
            found = np.asarray(mask_image.convert("L")) >= 96
            if found.shape[:2] != keep.shape[:2]:
                found = cv2.resize(
                    found.astype(np.uint8),
                    (keep.shape[1], keep.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
            return found

        # Extract negative mask (e.g. chair)
        try:
            negative_mask = (
                masker.generate_mask(photo_image, negative_query)
                if negative_query else None
            )
        except Exception:
            negative_mask = None
        
        if negative_mask is not None:
            keep = keep & ~_fit(negative_mask)
        
        # A positive object joins only when the surface already touches it.
        # Unioning it on every recolor painted that object with the wall.
        if positive_query:
            try:
                positive_mask = masker.generate_mask(photo_image, positive_query)
            except Exception:
                positive_mask = None
            
            if positive_mask is not None:
                positive = _fit(positive_mask)
                if int((positive & keep).sum()) >= 48:
                    keep = keep | positive
        
        # Additional protection: ensure we don't leave isolated small patches
        if int(keep.sum()) < max(48, int(subject_mask.sum()) * 0.10):
            return keep
            
        return keep

    def _head_mask_from_faces(self, image: Image.Image, faces) -> Optional[Image.Image]:
        """Create a union of head-only masks from detected faces.

        This is a neutral visual candidate set: it does not decide *which*
        person the request names. The image-edit model receives the prompt and
        can choose among the editable heads, while the rest of the photo stays
        outside the inpaint region.
        """
        try:
            if not faces:
                return None
            w, h = image.size
            head = np.zeros((h, w), dtype=np.uint8)
            for face_obj in faces:
                x1, y1, x2, y2 = [float(v) for v in face_obj.bbox]
                fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                cx = int(round((x1 + x2) * 0.5))
                cy = int(round(y1 + fh * 0.22))
                rx = max(2, int(round(fw * 0.78)))
                ry = max(2, int(round(fh * 1.05)))
                cv2.ellipse(head, (cx, cy), (rx, ry), 0, 0, 360, 255, thickness=-1)

                # A frontal ellipse leaves the posterior hairline/nape out on
                # profiles.  Infer the back side from the face landmarks: the
                # nose shifts toward the visible/front side, so the opposite
                # side receives a small lateral lobe.  This is geometric only;
                # it has no dependency on the wording of the edit request.
                try:
                    kps = np.asarray(getattr(face_obj, "kps", None), dtype=np.float32)
                    if kps.shape[0] >= 3 and kps.shape[1] >= 2:
                        eye_center_x = float((kps[0, 0] + kps[1, 0]) * 0.5)
                        nose_offset = float(kps[2, 0] - eye_center_x)
                        if abs(nose_offset) >= fw * 0.04:
                            back_dir = -1 if nose_offset > 0 else 1
                            nape_cx = int(round(cx + back_dir * fw * 0.48))
                            nape_cy = int(round(y1 + fh * 0.42))
                            nape_rx = max(2, int(round(fw * 0.40)))
                            nape_ry = max(2, int(round(fh * 0.42)))
                            cv2.ellipse(
                                head,
                                (nape_cx, nape_cy),
                                (nape_rx, nape_ry),
                                0, 0, 360, 255,
                                thickness=-1,
                            )
                except Exception:
                    # Bboxes alone remain a valid fallback for detectors that
                    # do not provide five facial landmarks.
                    pass
            if not head.any():
                return None
            head = cv2.GaussianBlur(head, (31, 31), 0)
            return Image.fromarray(head).convert("L")
        except Exception as exc:
            print(f"[ImgEditor] Máscaras de cabezas candidatas omitidas: {exc}", flush=True)
            return None

    @staticmethod
    def _crop_focal_context(
        image: Image.Image, mask: Image.Image, max_area: float = 0.35, report: bool = True,
    ):
        """Return a padded local crop and its mask, or None when not measurable.

        The crop is derived only from non-zero mask geometry.  It lets an EDIT
        model work on a local visual region rather than regenerate the whole
        photo and then trying to put it back together afterwards.
        """
        try:
            arr = np.asarray(mask.convert("L"))
            # La máscara se difumina para componer sin costuras. Sus píxeles
            # débiles no son la región editable: al usarlos como bbox el crop
            # de una cabeza llegó a abarcar 553x533 de una foto 800x533.
            # Solo el núcleo opaco puede decidir los límites del trabajo local.
            ys, xs = np.where(arr >= 160)
            if not len(xs) or not len(ys):
                return None
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            # Start with context, then shrink only the CONTEXT if a close-up
            # would exceed the local area cap. The opaque editable head stays
            # untouched; rejecting a valid close-up or expanding to the full
            # scene both lose the user's intended local edit.
            iw, ih = image.size
            for context in (0.35, 0.25, 0.15, 0.05, 0.0):
                pad_x = max(8, int(round(bw * context))) if context else 0
                pad_y = max(8, int(round(bh * context))) if context else 0
                box = (
                    max(0, x1 - pad_x), max(0, y1 - pad_y),
                    min(iw, x2 + pad_x), min(ih, y2 + pad_y),
                )
                if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                    return None
                crop_area = (box[2] - box[0]) * (box[3] - box[1])
                if crop_area <= iw * ih * max_area:
                    if context < 0.35 and report:
                        print(
                            f"[ImgEditor] Contexto focal ajustado a {context:.0%} "
                            f"(crop={crop_area / (iw * ih):.1%} de la foto)", flush=True
                        )
                    return image.crop(box), mask.crop(box).convert("L"), box
            if report:
                print(
                    f"[ImgEditor] Crop focal rechazado: el núcleo ocupa más del {max_area:.0%} de la foto",
                    flush=True,
                )
            return None
        except Exception:
            return None

    def _appearance_work_region(
        self, image: Image.Image, mask: Optional[Image.Image],
        person: Optional[Image.Image] = None,
    ):
        """Sample around a measured region without letting the mask be the sample.

        LongCat paints an inpaint hole gray before sampling. On a 400x600 photo
        the measured head was a 101x161 crop, the gray hole was most of that
        crop, and the model left the disk (edit_1790788517.png, mean 126).
        The window keeps the photo around the mask. Its area follows the same
        1MP canvas LongCat already normalizes to, and the mask stays within the
        same 0.35 local cap _crop_focal_context uses. The paste still writes
        only the measured mask.

        2026-10-03: cuando se pasa la persona, la ventana no puede recortarla.
        En fast_1790845644.png la persona medida ocupa el 99% del cuadro
        (bbox (0,8,1248,832)), asi que abrir solo un recorte deja la cabeza
        entera fuera de contexto y Turbo la recompone girada (bath_real3/4/5);
        con la foto completa como contexto la geometria se conserva con la
        misma mascara (bath_fw1/bath_fw2, 6.4% de pixeles cambiados). Lo que
        se escribe lo decide la mascara medida, no la ventana.
        """
        if mask is None:
            return image, mask, None
        focal = self._crop_focal_context(image, mask)
        if focal is None:
            return image, mask, None
        arr = np.asarray(mask.convert("L"))
        opaque = arr >= 160
        ys, xs = np.where(opaque)
        if not len(xs):
            return image, mask, None
        iw, ih = image.size
        photo_area = max(1, iw * ih)
        core_area = int(opaque.sum())
        area_cap = 0.35
        # 2026-10-01: el suelo de 1 MP hacia que TODA foto por debajo de
        # 1048576 px abriera la foto entera y el pegado compusiera una escena
        # regenerada sobre el original (edit_1790845755). El suelo sigue
        # valiendo: da pixeles al sampler pero como mucho la mitad de la foto.
        sampler_area = min(1024 * 1024, int(photo_area * 0.5))
        person_box = None
        if person is not None:
            parr = np.asarray(person.convert("L"))
            pys, pxs = np.where(parr >= 160)
            if len(pxs):
                person_box = (
                    int(pxs.min()), int(pys.min()),
                    int(pxs.max()) + 1, int(pys.max()) + 1,
                )
                person_area = max(
                    1, (person_box[2] - person_box[0]) * (person_box[3] - person_box[1])
                )
                sampler_area = min(
                    1024 * 1024, max(sampler_area, person_area)
                )
        target_area = min(
            photo_area,
            max(sampler_area, int(core_area / area_cap)),
        )
        if target_area >= photo_area * 0.98:
            box = (0, 0, iw, ih)
        else:
            aspect = iw / float(ih)
            win_h = max(1.0, (target_area / aspect) ** 0.5)
            win_w = target_area / win_h
            cx = (float(xs.min()) + float(xs.max()) + 1.0) / 2.0
            cy = (float(ys.min()) + float(ys.max()) + 1.0) / 2.0
            x1 = int(round(cx - win_w / 2.0))
            y1 = int(round(cy - win_h / 2.0))
            x2 = int(round(x1 + win_w))
            y2 = int(round(y1 + win_h))
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if x2 > iw:
                x1 -= x2 - iw
                x2 = iw
            if y2 > ih:
                y1 -= y2 - ih
                y2 = ih
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)
            if person_box is not None:
                # La persona medida tiene que caber entera en la ventana.
                x1, y1 = min(x1, person_box[0]), min(y1, person_box[1])
                x2, y2 = max(x2, person_box[2]), max(y2, person_box[3])
            tx1, ty1, tx2, ty2 = [int(v) for v in focal[2]]
            x1, y1 = min(x1, tx1), min(y1, ty1)
            x2, y2 = max(x2, tx2), max(y2, ty2)
            box = (max(0, x1), max(0, y1), min(iw, x2), min(ih, y2))
        work = image.crop(box)
        work_mask = mask.crop(box).convert("L")
        covered = float((np.asarray(work_mask) >= 160).mean()) if work_mask.size else 1.0
        if box == (0, 0, iw, ih):
            print(
                f"[ImgEditor] Muestra la foto completa {iw}x{ih}; "
                f"la zona medida ocupa {covered:.1%} y el resto de la muestra es la foto",
                flush=True,
            )
        else:
            print(
                f"[ImgEditor] Región medida a {work.size[0]}x{work.size[1]} "
                f"con la foto alrededor ({covered:.1%} de máscara); "
                "fuera de esa ventana no entra",
                flush=True,
            )
        return work, work_mask, box

    def _changed_zone_on_subject(
        self,
        generated: Image.Image,
        original_crop: Image.Image,
        subject_mask: Image.Image,
    ) -> Optional[Image.Image]:
        try:
            gen = generated.convert("RGB")
            orig = original_crop.convert("RGB")
            if gen.size != orig.size:
                gen = gen.resize(orig.size, Image.Resampling.LANCZOS)
            gen_arr = np.asarray(gen, dtype=np.int16)
            orig_arr = np.asarray(orig, dtype=np.int16)
            # The strongest channel response is the most sensitive way to see
            # a change that is real in at least one band (dark hair on light
            # skin moves blue and green more than red).
            delta = np.abs(gen_arr - orig_arr).max(axis=2)
            subject = np.asarray(
                subject_mask.convert("L").resize(orig.size, Image.Resampling.BILINEAR)
            )
            # El umbral sale de la DISTRIBUCION REAL del cambio dentro del sujeto, no
            # de una constante. Un 8 fijo descartaba las ediciones sutiles
            # (el pelo que se aclara, la piel que se suaviza, un verde que
            # vira a gris) y las devolvia al pegado viejo con su elipse. Medir
            # el ruido FUERA del sujeto tampoco servia: cuando el modelo
            # reescribe el fondo, ese "ruido" es enorme y el umbral
            # inutilizable.
            # Dentro del sujeto la respuesta es una mezcla de dos grupos: el
            # repintado sin registrar (textura y sombreado, respuesta baja) y
            # la edicion pedida (respuesta alta). Otsu separa los dos sin
            # ningun parametro, asi que el mismo criterio sirve para pelo,
            # ropa, color u objeto anadido: lo unico que decide es la foto.
            inside_delta = delta[subject >= 96].astype(np.uint8)
            if inside_delta.size == 0:
                return None
            strong = float(np.percentile(inside_delta, 98.0))
            if strong < 4.0:
                print(
                    f"[ImgEditor] Cambio medido maximo {strong:.1f} niveles: "
                    "no hay edicion perceptible sobre el sujeto",
                    flush=True,
                )
                return None
            threshold = _otsu_threshold(inside_delta.reshape(-1))
            threshold = float(max(3.0, min(threshold, strong)))
            changed = (delta >= threshold) & (subject >= 96)
            share = float(changed.sum()) / max(1, int((subject >= 96).sum()))
            print(
                f"[ImgEditor] Cambio medido sobre el sujeto: maximo "
                f"{strong:.0f} niveles, umbral Otsu={threshold:.0f}, "
                f"respuesta en {share:.1%} del sujeto",
                flush=True,
            )
            if not changed.any():
                return None
            # A sampled face is never perfectly registered: the model moves
            # skin texture and shading across the whole crop. Keeping every
            # isolated responding pixel would feather the entire subject and
            # bring the translucent rectangle back, so only the connected
            # regions that carry a real edit are kept. The size filter is
            # relative to the measured subject, never an absolute shape.
            count, labels, stats, _ = cv2.connectedComponentsWithStats(
                changed.astype(np.uint8), connectivity=8,
            )
            subject_area = int((subject >= 96).sum())
            if count <= 1 or subject_area <= 0:
                return None
            # Un cambio de baja intensidad reparte su respuesta en manchas
            # pequenas (una piel que se suaviza responde pixel a pixel), y un
            # filtro de tamano las tiraria todas. El piso del filtro baja a
            # un|area del sujeto| cuando la respuesta es debil, para que una
            # edicion sutil se conserve completa en vez de pegar un parche
            # translucido.
            responding = delta[(subject >= 96) & changed]
            weak = float(np.percentile(responding, 50.0)) if responding.size else 0.0
            peak = float(np.percentile(responding, 98.0)) if responding.size else 0.0
            if peak > 0 and weak < 0.5 * peak:
                share = 0.004
            else:
                share = 0.02
            minimum = max(24, int(subject_area * share))
            selected = np.zeros(changed.shape, dtype=bool)
            kept = 0
            for index in range(1, count):
                area = int(stats[index, cv2.CC_STAT_AREA])
                if area >= minimum:
                    selected |= labels == index
                    kept += 1
            if not kept:
                return None
            # Cierra los huecos que abrio el sombreado del propio modelo, y
            # anade un margen pequeño medido sobre el sujeto. Sin este margen
            # el borde del cambio queda sin editar y se ve una orla del color
            # viejo. El margen es una fraccion del sujeto, no una forma.
            margin = max(2, int(round(min(selected.shape) * 0.02)))
            grow = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1),
            )
            grown = cv2.dilate(selected.astype(np.uint8), grow) > 0
            # Close the gaps the model's own shading opened inside the edited
            # region, then feather over that outline. The dilation is a small
            # fraction of the measured zone itself.
            radius = max(2, int(round(min(selected.shape) * 0.012)))
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
            )
            closed = cv2.morphologyEx(
                selected.astype(np.uint8), cv2.MORPH_CLOSE, kernel,
            ) > 0
            # v5.1212 (Fix G): una eliminación se sale de la silueta. El pelo
            # que cuelga contra el fondo es de la persona pero el segmentador
            # lo deja fuera; la generación lo quita (calva) y un pegado
            # recortado al sujeto dejaba los mechones viejos dibujados sobre
            # el fondo (edit_1792096645 y edit_1792096853: arco de pelo sobre
            # la cortina). Esos píxeles entran en la zona cuando coinciden
            # tres hechos medidos: el mismo cambio Otsu, contacto con el
            # cambio ya retenido en el sujeto y un color ORIGINAL igual al
            # del material que la edición retiró (mediana del original bajo
            # el cambio retenido). Una pared reescrita conserva su propio
            # color original y queda fuera.
            subject_on = subject >= 96
            outside_change = (delta >= threshold) & ~subject_on
            base_zone = closed & grown & subject_on
            extension = np.zeros(changed.shape, dtype=bool)
            if outside_change.any() and int(base_zone.sum()) >= 48:
                removed = np.median(orig_arr[base_zone].reshape(-1, 3), axis=0)
                color_d = np.linalg.norm(
                    orig_arr.astype(np.float32) - removed.astype(np.float32),
                    axis=2,
                )
                # El alcance es la propia extensión del cambio medido (su
                # bbox), no una forma inventada: lo que cuelga de la cabeza
                # vive junto a ella.
                ys_b, xs_b = np.where(base_zone)
                span_b = max(
                    int(ys_b.max() - ys_b.min() + 1),
                    int(xs_b.max() - xs_b.min() + 1),
                )
                reach = cv2.distanceTransform(
                    (~base_zone).astype(np.uint8), cv2.DIST_L2, 3,
                )
                near = reach <= span_b
                # Dos hechos medidos, cada uno cobrando su caso (2026-10-15):
                # 1) el anillo exterior tiene que medir un «otro» grupo: si
                #    todo lo que rodea al cambio es del mismo tono que el
                #    material (escenas sin persona dibujada, blanco sobre
                #    blanco), no hay material colgado medible y no se extiende
                #    nada;
                # 2) para entrar se exige el tono EXACTO del material (misma
                #    separación plana de 18 que aplica «solo el color del pico»
                #    en _complete_request_surface): el limite a mitad de camino
                #    admite un fondo del mismo rango que el material cuando la
                #    zona medida ya contiene pixeles de fondo (24094 px de
                #    fondo entraban en el escenario del zapato). El pelo que
                #    cuelga de una cabeza comparte su tono píxel a píxel.
                ring = near & ~base_zone
                same_limit = _same_color_limit(color_d, base_zone, ring)
                if same_limit is not None:
                    close = color_d <= 18.0
                    count_e, labels_e = cv2.connectedComponents(
                        (base_zone | (outside_change & close & near)).astype(np.uint8),
                        connectivity=8,
                    )
                    inside_labels = labels_e[base_zone]
                    inside_labels = inside_labels[inside_labels > 0]
                    if count_e > 1 and inside_labels.size:
                        touching = np.isin(labels_e, np.unique(inside_labels))
                        extension = (
                            outside_change & close & touching & near
                        )
                        if int(extension.sum()) < 48:
                            extension = np.zeros(changed.shape, dtype=bool)
                        else:
                            print(
                                "[ImgEditor] Cambio medido fuera de la silueta: "
                                f"{int(extension.sum())} px del material retirado "
                                "entran en la zona; el fondo de otro color queda fuera",
                                flush=True,
                            )
            zone = np.where(base_zone | extension, 255, 0).astype(np.uint8)
            coverage = float((zone >= 160).mean())
            print(
                f"[ImgEditor] Zona pegada medida sobre el sujeto: "
                f"{coverage:.1%} del recorte ({kept} region(es) con cambio real)",
                flush=True,
            )
            return Image.fromarray(zone, "L")
        except Exception as exc:
            print(f"[ImgEditor] Zona de cambio no medida: {exc}", flush=True)
            return None

    def _integrate_local_generation(
        self, source: Image.Image, generated: Image.Image,
        mask: Image.Image, box, limit_mask: Optional[Image.Image] = None,
        keep_measured_mask: bool = False,
    ) -> Image.Image:
        """Paste the edited sample back onto the original photo.

        The generator still receives the full measured focus. Only the part
        that sits on the already measured subject is written back, so the
        photo behind that focus stays as it was. When the focus is already
        the region the segmenter assigned to the request, that region is
        pasted whole: the strongest pixel spike is not a substitute for it.
        """
        x1, y1, x2, y2 = [int(v) for v in box]
        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 2 or crop_h < 2:
            return source.convert("RGB")
        edited = generated.convert("RGB")
        if edited.size != (crop_w, crop_h):
            scale_up = max(crop_w / max(1, edited.width), crop_h / max(1, edited.height))
            edited = edited.resize((crop_w, crop_h), Image.Resampling.LANCZOS)
            # 2026-10-01: en fotos grandes la generación llega a ~1 MP y el paste
            # la interpola (medido ×3,38 en 6503x4263): dentro de la máscara la
            # nitidez cae a 0,40x la de la foto de alrededor y la cabeza queda
            # borrosa frente al resto nítido. Compensación medida: igualar la
            # nitidez del parche al recorte original con unsharp, solo con
            # interpolación >1,5x y solo si hay déficit real.
            # 2026-10-10: una sola pasada de sigma 1,5 con tope 0,90 subía
            # 7->16 frente a 402 del recorte (4% de la fuente): el parche
            # seguía tan blando como la ventana translucida que se ve al
            # rededor de la cabeza. El escalado ×3,5 reparte el déficit por
            # bandas (fina, media y gruesa), así que se compensa por bandas
            # con tope por pasada, hasta 70% de la nitidez fuente o 2 rondas.
            # Validado en diagnostics/test_sharpness_fix.py sobre foto real:
            # 8->268 de 424 (63%) con desvío en zonas planas p99=21,6.
            if scale_up > 1.5:
                try:
                    import cv2 as _cv2
                    src_arr = np.asarray(
                        source.convert("RGB").crop((x1, y1, x2, y2)).convert("L"),
                        dtype=np.uint8,
                    )
                    ed_arr = np.asarray(edited.convert("L"), dtype=np.uint8)
                    src_sharp = float(_cv2.Laplacian(src_arr, _cv2.CV_64F).var())
                    ed_sharp = float(_cv2.Laplacian(ed_arr, _cv2.CV_64F).var())
                    if src_sharp > 1.0 and ed_sharp < src_sharp * 0.80:
                        arr = np.asarray(edited, dtype=np.float32)
                        ed_initial = ed_sharp
                        target = src_sharp * 0.70
                        passes_used = 0
                        for _round in range(2):
                            if ed_sharp >= target:
                                break
                            for sigma, cap in ((1.0, 0.6), (2.0, 0.6), (4.0, 0.5)):
                                if ed_sharp >= target:
                                    break
                                deficit = src_sharp / max(1.0, ed_sharp) - 1.0
                                strength = float(min(cap, 0.50 * deficit))
                                if strength < 0.05:
                                    continue
                                blur = _cv2.GaussianBlur(arr, (0, 0), sigma)
                                arr = _cv2.addWeighted(
                                    arr, 1.0 + strength, blur, -strength, 0,
                                )
                                passes_used += 1
                                ed_sharp = float(
                                    _cv2.Laplacian(
                                        np.clip(arr, 0, 255).astype(np.uint8),
                                        _cv2.CV_64F,
                                    ).var()
                                )
                        edited = Image.fromarray(
                            np.clip(arr, 0, 255).astype(np.uint8)
                        )
                        print(
                            f"[ImgEditor] Paste x{scale_up:.2f}: nitidez parche "
                            f"{ed_initial:.0f} -> {ed_sharp:.0f} (original "
                            f"{src_sharp:.0f}, {ed_sharp / max(1.0, src_sharp):.0%} "
                            f"de fuente, {passes_used} pasadas multi)",
                            flush=True,
                        )
                except Exception as sharp_err:
                    print(f"[ImgEditor] Compensación de nitidez omitida: {sharp_err}", flush=True)
        local_mask = mask.convert("L")
        if local_mask.size != source.size:
            local_mask = local_mask.resize(source.size, Image.Resampling.BILINEAR)
        local_mask = local_mask.crop((x1, y1, x2, y2))
        # A measured request is already the zone to paste. The person locator
        # can be smaller than that zone once the same color continues past it.
        if limit_mask is not None and not keep_measured_mask:
            limit = limit_mask.convert("L")
            if limit.size != source.size:
                limit = limit.resize(source.size, Image.Resampling.BILINEAR)
            limit = limit.crop((x1, y1, x2, y2))
            limited = self._focus_on_measured_subject(local_mask, np.asarray(limit))
            if limited is None:
                return source.convert("RGB")
            local_mask = limited
        else:
            limit = None
        # 2026-10-01 (edit_1790852880.png): componer la elipse de cabeza hacia
        # un parche que el modelo regenero entero produce el rectangulo
        # translucido y la doble cabeza. Un umbral de altura lo tapaba
        # (0,60) devolviendo la foto sin editar, que es el mismo hardcode de
        # forma que el problema.
        # La zona que se pega sale de MEDIR el cambio: los pixeles que el
        # modelo cambio de verdad, recortados al sujeto ya medido. El borde
        # del feather cae sobre el contorno real del cambio (pelo, ropa,
        # gafas), no sobre una elipse inventada. Sirve para cualquier tipo de
        # edicion porque no mira la forma de nada: mira la diferencia entre
        # la generacion y la foto, y donde cae respecto del sujeto medido.
        # Lo que el modelo reescribio fuera del sujeto no entra, asi que el
        # fondo no puede componer un parche translucido.
        subject_arr = np.asarray(local_mask, dtype=np.uint8)
        measured_zone = None
        if not keep_measured_mask:
            measured_zone = self._changed_zone_on_subject(
                edited, source.crop((x1, y1, x2, y2)), local_mask,
            )
            if measured_zone is not None:
                local_mask = measured_zone
                
        # 2026-10-10: 6px de feather sobre un recorte de 3428px deja el borde
        # del parche durisimo (nitidez 15 pegada contra foto 402) y se ve el
        # rectangulo translucido alrededor de la cabeza. El feather escala con
        # el recorte para que la transicion sea suave en fotos grandes.
        feather_px = max(6, int(min(crop_w, crop_h) * 0.02))
        soft = self._feather_mask(local_mask, amount=feather_px)
        print(
            f"[ImgEditor] Feather paste: {feather_px}px sobre recorte "
            f"{crop_w}x{crop_h}",
            flush=True,
        )
        # The feather is applied around the measured change, so it bleeds a few
        # pixels past it. Those pixels still have to stay inside the subject
        # already located on the photo: an edit that did not ask for a new
        # pose may not write outside the region it was measured on.
        soft_arr = np.asarray(soft)
        keep = subject_arr if limit is None else np.minimum(subject_arr, np.asarray(limit))
        # v5.1212 (Fix G): la zona medida ya incluye el material retirado que
        # se sale de la silueta (delta + contacto + color original medido).
        # Este clip existia para que el feather no escribiera fuera del sujeto;
        # si recorta la zona medida, vuelve a dejar el pelo viejo dibujado
        # sobre el fondo. Lo que la medicion admitio no lo recorta el clip.
        if measured_zone is not None:
            measured_arr = np.asarray(measured_zone, dtype=np.uint8)
            if measured_arr.shape == keep.shape:
                keep = np.maximum(keep, measured_arr)
        if keep.shape[:2] == soft_arr.shape[:2]:
            # This is a paste boundary, not another object segmentation.
            # Otsu separates measured support from its weak rim; zero always
            # remains outside, including when the measured threshold is zero.
            subject_thresh = 0.0
            if keep.size and not np.all(keep == keep[0, 0]):
                subject_thresh, _ = cv2.threshold(
                    keep, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
            soft = Image.fromarray(
                np.where(keep > subject_thresh, soft_arr, 0).astype(np.uint8), "L",
            )
        blended = Image.composite(
            edited, source.convert("RGB").crop((x1, y1, x2, y2)), soft,
        )
        full = source.convert("RGB").copy()
        full.paste(blended, (x1, y1))
        return full

    def _degenerate_faces_mask(
        self, source: Image.Image, generated: Image.Image, faces,
    ) -> Optional[Image.Image]:
        """Faces the generation collapsed into a black (or blown) blob.

        A pass that regenerates a person sometimes delivers a solid black
        mask over the face (edit_1792086220, 15/10: the woman's face pasted
        at median luma 6 against the source's 193 in the same pixels). The
        zone paste cannot tell a face from any other measured pixel, so a
        collapsed face is detected by its own pixels and excluded from the
        paste; the original face stays. The collapse is measured RELATIVE to
        the source face (the generated median falls below a quarter of the
        source's own luma), so any skin tone is protected by the same rule;
        a face that merely shifted, aged or was restyled keeps its own luma
        and is not touched.
        """
        if source is None or generated is None or not faces:
            return None
        try:
            comparison = generated.convert("RGB")
            if comparison.size != source.size:
                comparison = comparison.resize(source.size, Image.Resampling.LANCZOS)
            src = np.asarray(source.convert("RGB"), dtype=np.int16)
            gen = np.asarray(comparison, dtype=np.int16)
        except Exception:
            return None
        bad = np.zeros((source.height, source.width), dtype=np.uint8)
        found = False
        soft_sigma = 3.0
        for face in faces:
            try:
                x1, y1, x2, y2 = [float(v) for v in face.bbox]
            except (TypeError, ValueError, AttributeError):
                continue
            w, h = x2 - x1, y2 - y1
            if w < 8 or h < 8:
                continue
            cx1 = max(0, int(x1 + w * 0.25))
            cy1 = max(0, int(y1 + h * 0.25))
            cx2 = min(source.width, int(x2 - w * 0.25))
            cy2 = min(source.height, int(y2 - h * 0.25))
            if cx2 - cx1 < 4 or cy2 - cy1 < 4:
                continue
            src_luma = float(np.median(src[cy1:cy2, cx1:cx2]))
            gen_luma = float(np.median(gen[cy1:cy2, cx1:cx2]))
            if src_luma < 24 or gen_luma > 0.25 * src_luma:
                continue
            radius = max(6, int(round(0.20 * max(w, h))))
            bx1 = max(0, int(x1) - radius)
            by1 = max(0, int(y1) - radius)
            bx2 = min(source.width, int(x2) + radius)
            by2 = min(source.height, int(y2) + radius)
            bad[by1:by2, bx1:bx2] = 255
            found = True
            soft_sigma = max(soft_sigma, radius / 3.0)
            print(
                f"[ImgEditor] Cara degenerada en el generado "
                f"(luma {gen_luma:.0f} vs fuente {src_luma:.0f}); "
                "esa cara se conserva de la foto original",
                flush=True,
            )
        if not found:
            return None
        soft = cv2.GaussianBlur(bad, (0, 0), soft_sigma)
        soft = np.clip(soft.astype(np.int32) * 2, 0, 255).astype(np.uint8)
        return Image.fromarray(soft, "L")

    def _second_focused_pass(
        self, fast_client, edited: Image.Image, original: Image.Image,
        subject_mask: Image.Image, local_box, gen_size,
        prompt: str, negative_prompt, seed, prompt_authority,
        preserve_frame, vision_reference, lora_name, lora_strength,
        appearance_sample: dict, progress_callback, cancel_check,
    ) -> Optional[Image.Image]:
        """Regenerate the already pasted zone at its own zoom.

        A wide window normalized to the 1MP LongCat canvas delivers the
        measured zone upscaled (measured 2026-10-07: 1674x3733 -> 688x1536,
        head of 94 px, patch texture at 32-45% of the photo), and that soft
        patch is what reads as "se ve el parche". When the paste upscales by
        more than the same 1.5 threshold the sharpness compensation already
        uses, the measured zone is generated once more on a local crop of
        the already edited photo, so its content arrives at native pixels.

        The damage radius stays bounded: the second paste is limited to the
        zone that was already written, so a recomposed crop cannot move
        anything outside it. No fixed shapes: both the zoom window and the
        limit come from the measured change itself.
        """
        crop_w = int(local_box[2]) - int(local_box[0])
        crop_h = int(local_box[3]) - int(local_box[1])
        if crop_w < 2 or crop_h < 2 or not gen_size:
            print(
                f"[ImgEditor] Refino focal omitido: ventana de trabajo "
                f"{crop_w}x{crop_h} sin tamaño de generación usable",
                flush=True,
            )
            return None
        scale_up = max(crop_w / max(1, int(gen_size[0])),
                       crop_h / max(1, int(gen_size[1])))
        if scale_up <= 1.5:
            return None
        try:
            bx, by = int(local_box[0]), int(local_box[1])
            zone = self._changed_zone_on_subject(
                edited.crop((bx, by, bx + crop_w, by + crop_h)),
                original.crop((bx, by, bx + crop_w, by + crop_h)),
                subject_mask.crop((bx, by, bx + crop_w, by + crop_h)),
            )
            if zone is None:
                print(
                    f"[ImgEditor] Refino focal omitido: sin zona medida "
                    f"(escala {scale_up:.2f}x no aprovechada)",
                    flush=True,
                )
                return None
            zone_full = Image.new("L", original.size, 0)
            zone_full.paste(zone, (bx, by))
            focal = self._crop_focal_context(edited, zone_full, report=False)
            if focal is None:
                # Sin esta marca el refino desaparecía sin explicación (log
                # real 01:21: pegado x3.34 y ni «Refino focal» ni motivo).
                print(
                    f"[ImgEditor] Refino focal omitido: el cambio medido no "
                    f"abre una ventana focal (escala {scale_up:.2f}x sin "
                    "aprovechar)",
                    flush=True,
                )
                return None
            zoom_image, zoom_mask, zoom_box = focal
            zoom_w = zoom_box[2] - zoom_box[0]
            zoom_h = zoom_box[3] - zoom_box[1]
            # The cover belongs to the work window. Stretching it onto the
            # zoom warps the surface the sampler is supposed to replace.
            zoom_reference = vision_reference
            if zoom_reference is not None and zoom_reference.size != zoom_image.size:
                zoom_reference = None
            # Por debajo de ~128 px un recorte no trae estructura de la escena:
            # el modelo regeneraría la zona desde texto y esa sí sería una
            # recomposición. El refino solo tiene sentido con estructura real.
            if min(zoom_w, zoom_h) < 128:
                print(
                    f"[ImgEditor] Refino focal omitido: ventana zoom "
                    f"{zoom_w}x{zoom_h} sin estructura aprovechable",
                    flush=True,
                )
                return None
            print(
                f"[ImgEditor] Refino focal: ventana {zoom_box[2]-zoom_box[0]}x"
                f"{zoom_box[3]-zoom_box[1]} sobre zona de "
                f"{int((np.asarray(zone) >= 160).sum())} px (escala previa "
                f"{scale_up:.2f}x)",
                flush=True,
            )
            result, msg = fast_client.generate(
                image=zoom_image, prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=1.0,
                seed=seed,
                prompt_authority=prompt_authority,
                preserve_frame=preserve_frame,
                mask_image=zoom_mask,
                keep_source_pixels=True,
                reference_image=zoom_reference,
                lora_name=lora_name, lora_strength=lora_strength,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                **appearance_sample,
            )
            if result is None:
                print(
                    f"[ImgEditor] Refino focal sin generación: {msg}",
                    flush=True,
                )
                return None
            refined = self._integrate_local_generation(
                edited, result.image.convert("RGB"), subject_mask, zoom_box,
                limit_mask=zone_full,
            )
            print(
                "[ImgEditor] Zona medida refinada en su propio zoom; "
                "el resto de la foto no entra",
                flush=True,
            )
            return refined
        except Exception as refine_err:
            print(f"[ImgEditor] Refino focal omitido: {refine_err}", flush=True)
            return None

    @staticmethod
    def _split_compound_instruction(prompt: str, region_count: int = 0) -> list:
        """Extract independent edit clauses, including comma-separated ones."""
        # Punctuation has priority: a conjunction inside one clause can join
        # attributes of the same subject ("barefoot and without stockings").
        # Splitting it there would turn one requested edit into two subjects.
        text = prompt or ""
        if re.search(r"[,;]", text):
            parts = re.split(r"\s*[,;]\s*", text)
        else:
            parts = re.split(r"\s+(?:and|y)\s+", text, flags=re.IGNORECASE)
        clauses = [re.sub(r"^(?:and|y)\s+", "", part.strip(" .,!?"), flags=re.I)
                   for part in parts]
        if (not 2 <= len(clauses) <= 6
                or (region_count and len(clauses) != region_count)
                or any(len(part.strip()) < 2 for part in clauses)):
            return []
        return clauses

    @staticmethod
    def _compound_change_list(prompt: str) -> str:
        """Repeat each compound clause once, separated by commas."""
        clauses = ImgEditorManager._split_compound_instruction(prompt)
        cleaned = [clause.strip(" .,!?;") for clause in clauses]
        return ", ".join(part for part in cleaned if part)

    @staticmethod
    def _is_scene_location_clause(clause: str) -> bool:
        """Recognize a requested location change, independent of place names."""
        text = re.sub(r"^(?:and|y)\s+", "", (clause or "").strip(), flags=re.I)
        return bool(re.search(
            r"(?:\b(?:am|is|are|was|were|be|being)\b|['’](?:m|s|re)\b)"
            r"\s+(?:in|at|inside)\s+(?:a|an|the)\s+\S+",
            text, flags=re.I,
        ))

    @staticmethod
    def _background_mask_from_people(image: Image.Image, masker, faces: list) -> Optional[Image.Image]:
        """Edit scenery only when the people can be protected reliably."""
        foreground = np.zeros((image.height, image.width), dtype=np.uint8)
        for query in ("person", "human", "man", "woman"):
            part = masker.generate_mask(image, query)
            if part is None:
                continue
            if part.size != image.size:
                part = part.resize(image.size, Image.Resampling.BILINEAR)
            foreground = np.maximum(foreground, np.asarray(part.convert("L"), dtype=np.uint8))
        foreground = (foreground >= 96).astype(np.uint8)
        coverage = float(foreground.mean())
        if not 0.08 <= coverage <= 0.80 or not faces:
            return None
        for face in faces:
            try:
                x1, y1, x2, y2 = [float(v) for v in face.bbox]
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if not (0 <= cx < image.width and 0 <= cy < image.height and foreground[cy, cx]):
                    return None
            except (TypeError, ValueError, AttributeError):
                return None
        foreground = cv2.dilate(foreground, np.ones((7, 7), np.uint8))
        background = (1 - foreground) * 255
        if float((background > 0).mean()) < 0.12:
            return None
        return Image.fromarray(background.astype(np.uint8), "L")

    def _surface_without_people(
        self, image: Image.Image, mask: Optional[Image.Image], faces,
    ) -> Optional[Image.Image]:
        """Take people out of a large surface when they only overlap its edge.

        A wall or floor measured behind someone still includes a strip of
        their body. Painting that strip replaces the person. When the people
        are most of the measured region, the color change is on them and the
        mask stays.
        """
        if mask is None or not faces:
            return mask
        arr = np.asarray(mask.convert("L"))
        if arr.shape[:2] != (image.height, image.width):
            arr = cv2.resize(arr, image.size, interpolation=cv2.INTER_LINEAR)
        region = arr >= 96
        if float(region.mean()) < 0.12:
            return mask
        try:
            masker = self._get_clipseg_masker()
        except Exception:
            return mask
        if masker is None:
            return mask
        background = self._background_mask_from_people(image, masker, faces)
        if background is None:
            return mask
        people = np.asarray(
            background.convert("L").resize(image.size, Image.Resampling.BILINEAR)
        ) < 128
        overlap = float((region & people).sum()) / max(1, int(region.sum()))
        if overlap < 0.02 or overlap >= 0.55:
            return mask
        kept = arr.copy()
        kept[people] = 0
        if int((kept >= 96).sum()) < 16:
            return mask
        print(
            f"[ImgEditor] La superficie medida deja fuera a las personas "
            f"({overlap:.0%} de solape)",
            flush=True,
        )
        return Image.fromarray(kept, "L")

    def _close_recolor_halo(
        self, image: Image.Image, mask: Image.Image, faces,
    ) -> Optional[Image.Image]:
        """The wall gap that hugs a person is wall too.

        CLIPSeg leaves an unmeasured band between the person and the wall it
        lights, and the same-color continuation refuses it because the warm
        bounce light differs from the flat wall. That band survives the paste
        as the ORIGINAL warm wall inside a newly gray room, which reads as a
        glowing halo around the person (edit_1792086138 / edit_1792086220,
        15/10: the untouched pixels measured identical to the pre-edit photo
        while the rest of the wall moved from peach 223,157 to gray 137,138).
        The gap is an INTERIOR hole of the measured region enclosed by that
        same region, so it is filled — but only inside the neighbourhood of a
        measured person, and the people themselves stay out; a hole away from
        any person (a sconce, a picture frame) is an object and is not
        repainted.
        """
        if mask is None or not faces:
            return mask
        arr = np.asarray(mask.convert("L"))
        if arr.shape[:2] != (image.height, image.width):
            arr = cv2.resize(arr, image.size, interpolation=cv2.INTER_LINEAR)
        region = arr >= 96
        if float(region.mean()) < 0.12:
            return mask
        try:
            masker = self._get_clipseg_masker()
        except Exception:
            return mask
        if masker is None:
            return mask
        background = self._background_mask_from_people(image, masker, faces)
        if background is None:
            return mask
        people = np.asarray(
            background.convert("L").resize(image.size, Image.Resampling.BILINEAR)
        ) < 128
        if not people.any() or not region.any():
            return mask
        # Interior holes: non-region pixels whose connected component touches
        # no border. (Seeded floodFill lies when the top-left pixel IS region:
        # nothing propagates and the whole outside counts as a hole, sweeping
        # distant zones into the paste — probe compound, fondo lejano.)
        holes_src = (~region).astype(np.uint8)
        _, labels = cv2.connectedComponents(holes_src, connectivity=4)
        border = np.concatenate([
            labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1],
        ])
        exterior = np.unique(border)
        exterior = exterior[exterior != 0]
        holes = holes_src.astype(bool) & ~np.isin(labels, exterior)
        # Only the band within reach of a person: the bounce-light halo.
        ring_px = max(16, int(round(0.03 * max(image.height, image.width))))
        ring = cv2.dilate(
            people.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (ring_px * 2 + 1, ring_px * 2 + 1),
            ),
        ) > 0
        filled = holes & ring & ~people
        added = int(filled.sum())
        if added < 48:
            return mask
        merged = region | filled
        coverage = float(merged.mean())
        # Same surface floor as the rest of the recolor: a speck is no zone.
        if coverage < 0.08 or coverage > 0.90:
            return mask
        print(
            f"[ImgEditor] El hueco junto a las personas es de la misma "
            f"superficie: +{added} px en la zona ({coverage:.1%}); "
            "las personas quedan fuera",
            flush=True,
        )
        return Image.fromarray(merged.astype(np.uint8) * 255, "L")

    def _closed_structural_zone(
        self, image: Image.Image, mask: Image.Image, faces,
    ) -> Optional[Image.Image]:
        """A removal edits everything inside its zone's own box.

        "Remove the pictures" measures the wall BETWEEN the pictures, so the
        frames arrive as holes and exterior gaps. FastEdit never touches
        anything outside the mask (0 % measured on 2026-10-15), so without
        this nothing could be removed. The box of the measured zone is the
        margin the paste already accepts, so inside it the elimination takes
        the whole box. The people are taken out on top of the same people
        rules as any measured surface; when the box is mostly people the
        removal is not of objects and the measured zone stays as measured.
        """
        if mask is None:
            return None
        arr = np.asarray(mask.convert("L"))
        if arr.shape[:2] != (image.height, image.width):
            arr = cv2.resize(arr, image.size, interpolation=cv2.INTER_LINEAR)
        region = arr > 96
        if not region.any():
            return None
        ys, xs = np.nonzero(region)
        box = np.zeros_like(region, dtype=np.uint8)
        box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
        if faces:
            try:
                masker = self._get_clipseg_masker()
                background = (
                    self._background_mask_from_people(image, masker, faces)
                    if masker is not None else None
                )
            except Exception:
                background = None
            if background is None:
                return mask
            people = np.asarray(
                background.convert("L").resize(image.size, Image.Resampling.BILINEAR)
            ) < 128
            overlap = float((box.astype(bool) & people).sum()) / max(1, int(box.sum()))
            if overlap >= 0.55:
                # The box is more people than surface: this elimination is
                # not of objects; the measured zone stays as measured.
                return mask
            box[people] = 0
        finished = Image.fromarray(box * 255, "L")
        coverage = float((np.asarray(finished) >= 96).mean())
        # Same surface floor as a recolor: a speck is not an object zone.
        if coverage < 0.08:
            return mask
        print(
            f"[ImgEditor] La eliminación toma la caja de la zona medida "
            f"({coverage:.1%}) sin las personas",
            flush=True,
        )
        return finished

    def _compose_scene_preserving_heads(
        self, source: Image.Image, generated: Image.Image,
        source_background_mask: Image.Image, faces: list, masker,
    ) -> Tuple[Optional[Image.Image], str, Optional[Image.Image]]:
        """Accept a scene change only if people stay aligned; retain original heads."""
        try:
            comparison = generated.convert("RGB").resize(source.size, Image.Resampling.LANCZOS)
            generated_background = self._background_mask_from_people(comparison, masker, faces)
            if generated_background is None:
                return None, "personas no localizables en el fondo generado", None
            source_fg = np.asarray(source_background_mask.convert("L")) < 128
            generated_fg = np.asarray(generated_background.convert("L")) < 128
            intersection = int((source_fg & generated_fg).sum())
            union = int((source_fg | generated_fg).sum())
            overlap = intersection / max(1, union)
            displaced = float((generated_fg & ~source_fg).mean())
            source_pixels = np.asarray(source.convert("RGB"), dtype=np.int16)
            generated_pixels = np.asarray(comparison, dtype=np.int16)
            background_change = float((
                np.max(np.abs(generated_pixels - source_pixels), axis=2)[~source_fg] > 30
            ).mean())
            print(
                f"[ImgEditor] Entorno: solape personas={overlap:.2f}, "
                f"desplazamiento={displaced:.1%}, fondo cambiado={background_change:.1%}",
                flush=True,
            )
            if overlap < 0.75 or displaced > 0.05 or background_change < 0.12:
                return None, "la generación desplazó personas o no cambió el entorno", None

            head_geometry = self._head_mask_from_faces(source, faces)
            if head_geometry is None:
                return None, "no se pudieron preservar las cabezas de la primera edición", None
            person_mask = masker.generate_mask(source, "person")
            hair_mask = masker.generate_mask(source, "hair")
            if person_mask is None:
                return None, "no se pudo separar la cabeza del fondo original", None
            person_fg = np.asarray(person_mask.convert("L")) >= 96
            if hair_mask is not None:
                person_fg |= np.asarray(hair_mask.convert("L")) >= 96
            head = (np.asarray(head_geometry.convert("L")) >= 128) & person_fg
            if float(head.mean()) < 0.005:
                return None, "máscara de cabezas insuficiente", None
            from PIL import ImageFilter
            head_mask = Image.fromarray((head * 255).astype(np.uint8), "L")
            head_mask = head_mask.filter(ImageFilter.GaussianBlur(4))
            if generated.size != source.size:
                head_mask = head_mask.resize(generated.size, Image.Resampling.BILINEAR)
                source = source.resize(generated.size, Image.Resampling.LANCZOS)
            return Image.composite(source.convert("RGB"), generated.convert("RGB"), head_mask), "", head_mask
        except Exception as exc:
            return None, f"composición del entorno falló: {exc}", None

    # Grammatical predicate, not an edit-type list. Longer auxiliaries stay
    # before the bare verb so "tiene que" and "has to" are not cut at "tiene"
    # or "has". Bare "tiene"/"has"/"have" are how Spanish and English assign
    # an attribute ("tiene el pelo negro", "has black hair").
    _PREDICATE_RE = re.compile(
        r"\b(?:debe(?:n)?|tiene(?:n)?\s+que|must|should|"
        r"has\s+to|have\s+to|is|are|was|were|goes|go|"
        r"es|son|est[aá](?:n)?|va|van|tiene(?:n)?|has|have)\b",
        re.I,
    )

    @staticmethod
    def _align_compound_regions(prompt: str, regions: list, subject_query: str = "") -> list:
        """Match each clause to one named visual region, without trusting LLM order."""
        if not isinstance(regions, list):
            return []
        named = [str(region or "").strip().lower() for region in regions]
        named = list(dict.fromkeys(region for region in named if region))
        clauses = ImgEditorManager._split_compound_instruction(prompt)
        if not clauses:
            return []
        subject = str(subject_query or "").strip().lower()
        if subject and not any(re.search(rf"\b{re.escape(subject)}\b", region)
                               for region in named):
            named.append(subject)

        function_words = {"the", "a", "an", "of", "in", "on", "with", "and",
                          "his", "her", "he", "she", "their", "him", "them",
                          "el", "la", "los", "las", "de", "del", "en", "y",
                          "su", "sus"}
        def words(value):
            parts = re.findall(r"\w+", value.casefold())
            return {part[:-1] if part.endswith("s") and len(part) > 4 else part
                    for part in parts if part not in function_words}

        def subject_words(region):
            predicate = ImgEditorManager._PREDICATE_RE.search(region)
            return words(region[:predicate.start()] if predicate else region)

        region_words = {region: subject_words(region) for region in named}
        aligned = []
        for clause in clauses:
            tokens = words(clause)
            hits = [region for region in named if region_words[region] & tokens]
            # A clause omitted by the small planner still gets an independent
            # grounding attempt instead of a destructive full-frame pass.
            if len(hits) == 1:
                region = hits[0]
            elif len(hits) > 1:
                # Shared descriptors such as a posture occur in several
                # regions. Use only a word unique to one region if it also
                # occurs in this clause; otherwise leave grounding unresolved.
                unique = {hit: region_words[hit].difference(
                    *(region_words[other] for other in hits if other != hit)
                ) & tokens for hit in hits}
                owners = [hit for hit, terms in unique.items() if terms]
                if len(owners) == 1:
                    owner = owners[0]
                    terms = unique[owner]
                    ordered = [part for part in re.findall(r"\w+", owner.casefold())
                               if part in terms]
                    # A planner sentence is an edit description, not a clean
                    # person mask. Keep it for attribute-level grounding (e.g.
                    # a garment); reducing it to the subject can regenerate
                    # that person's whole body and move their pose.
                    has_predicate = bool(ImgEditorManager._PREDICATE_RE.search(owner))
                    region = (ordered[0] if len(ordered) == 1 and not has_predicate
                              else owner)
                else:
                    region = clause
            else:
                region = clause
            aligned.append((clause, region))
        return aligned

    @staticmethod
    def _head_only_plan(analysis: Dict, mask_target: str, body_transform: bool,
                        cloth_mask_rejected: bool, high_structural: bool) -> bool:
        """A person query is not enough to confine a multi-part edit to a head."""
        regions = analysis.get("affected_regions", [])
        return bool(
            not body_transform
            and not cloth_mask_rejected
            and not high_structural
            and mask_target in ("subject", "face", "hair")
            and float(analysis.get("appearance_score", 0.0)) >= 0.08
            and isinstance(regions, list)
            and len(regions) == 1
        )

    @staticmethod
    def _requested_change_scope(instruction: str) -> str:
        """Separate a subject description from a requested predicate.

        A posture or camera angle in the subject phrase describes who to edit;
        the requested change is after the modal predicate. Imperative prompts
        have no such boundary and retain their full text.
        """
        predicates = []
        for clause in re.split(r"\s*[,;]\s*", (instruction or "").strip()):
            matches = list(ImgEditorManager._PREDICATE_RE.finditer(clause))
            match = matches[0] if matches else None
            # A relative clause describes which subject is in the photo:
            # "the man who is kneeling is bald" requests baldness, not a
            # posture change. Use the next predicate after that description.
            if match and len(matches) > 1 and re.search(
                r"\b(?:who|that|which|que)\b", clause[:match.start()], flags=re.I
            ):
                match = matches[1]
            predicates.append(clause[match.end():].strip() if match else clause.strip())
        return ", ".join(part for part in predicates if part)

    @classmethod
    def _explicit_viewpoint_request(cls, raw_prompt: str, translated_prompt: str) -> bool:
        """Detect a requested camera change before the Turbo early return."""
        strong = re.compile(
            r"\b(?:pov|viewpoint|camera angle|camera view|angle|[aá]ngulo|"
            r"perspective|perspectiva|overhead|cenital|picado|contrapicado|"
            r"top view|bottom view|back view|side view|from behind|"
            r"from the side|from the right|from the left|desde arriba|"
            r"desde abajo|desde atr[aá]s|desde la derecha|desde la izquierda|"
            r"vista de espaldas|vista de lado|vista de perfil|"
            r"vista lateral|opposite view)\b",
            re.I,
        )
        move_view = re.compile(
            r"\b(?:cambia|cambiar|cambie|mueve|mover|muestra|mostrar|"
            r"change|move|show|switch|rotate)\b.{0,35}\b(?:vista|view|plano|shot)\b",
            re.I,
        )
        return any(
            strong.search(scope) or move_view.search(scope)
            for scope in (cls._requested_change_scope(text) for text in
                          (raw_prompt, translated_prompt) if text)
        )

    def _prompt_authority_facts(self, raw_prompt: str, translated_prompt: str) -> dict:
        """Separate a new pose from a body change.

        A whole sentence dilutes one clause: "bald, naked" scored like a weak
        clothing hint, so the nude state was never written and the pose wrapper
        redrew the room. Each clause is scored on its own. A removed object
        shares the token "remove" with clothing concepts, so body must lead.
        """
        facts = {
            "authority": False,
            "pose": False,
            "body": False,
            "body_direction": "change",
            "body_intensity": 0.0,
            "local_appearance": False,
            "local_region": False,
            # v5.1215: el usuario prima que la escena quede BIEN aunque la
            # cara original cambie («mientras me repintes la escenario al 95 %
            # me vale»). Un pedido de escenario NO es una región local: manda
            # la regeneración completa, sin ancla, como la postura.
            "scenario": False,
            # v5.1209: la eliminación estructural que el caller usa para
            # cerrar la zona dentro de su propia caja (sin listas de tipos).
            "structural_remove": False,
        }
        texts = [text for text in (raw_prompt, translated_prompt) if text and text.strip()]
        if not texts:
            return facts
        try:
            from roop.img_editor.prompt_analyzer import PromptAnalyzer
            analyzer = PromptAnalyzer()
        except Exception:
            analyzer = None
        try:
            scorer = self._get_semantic_analyzer(full_ai=False)
        except Exception:
            scorer = None
        # Each wording is judged on its own scores. Taking the maximum of
        # every concept across both languages let the English verb "removes"
        # raise the body score and cancel a removal that the Spanish clause
        # had already measured (2026-10-08: "elimina tatoo de la mujer").
        judged = []
        for text in texts:
            clauses = [part.strip() for part in re.split(r"\s*[,;]\s*", text) if part.strip()]
            clauses.append(text.strip())
            for clause in clauses:
                scope = self._requested_change_scope(clause) or clause
                pose_hit = bool(
                    analyzer is not None and (
                        analyzer._pose_pattern.findall(scope.lower())
                        or analyzer.subset_hits(scope, analyzer.POSE_KEYWORDS) > 0
                    )
                )
                if pose_hit:
                    facts["pose"] = True
                if scorer is None:
                    continue
                try:
                    intensity = float(scorer.get_body_transform_intensity(scope))
                    scores = scorer.score_concepts(scope)
                    direction = str(scorer.get_body_direction(scope))
                except Exception:
                    continue
                body_score = max(
                    float(scores.get("attribute_body", 0.0)),
                    float(scores.get("attribute_uncover", 0.0)),
                )
                structural_score = max(
                    float(scores.get("structural_remove", 0.0)),
                    float(scores.get("structural_add", 0.0)),
                    float(scores.get("structural_background", 0.0)),
                )
                structural_operation = max(
                    float(scores.get("structural_remove", 0.0)),
                    float(scores.get("structural_add", 0.0)),
                )
                remove_score = float(scores.get("structural_remove", 0.0))
                appearance_score = max(
                    float(scores.get("attribute_appearance", 0.0)),
                    float(scores.get("target_hair", 0.0)),
                )
                recolor_score = float(scores.get("attribute_recolor", 0.0))
                body_leads = bool(
                    intensity >= 0.08 and body_score >= 0.20
                    and body_score >= structural_score * 1.25
                )
                appearance_leads = bool(
                    not pose_hit and not body_leads
                    and appearance_score >= 0.20
                    and appearance_score >= max(
                        body_score, structural_score, recolor_score,
                    ) * 1.25
                )
                # v5.1215: un cambio de escenario pide regenerar la escena
                # completa («que esté en una playa»). La competición es el
                # resto de hechos de LA MISMA cláusula, igual que en cuerpo/
                # región: sin lista de palabras nueva, el score medido
                # structural_background ya existe. Gana sobre región local:
                # pintar una zona no es cambiar de escenario.
                background_score = float(
                    scores.get("structural_background", 0.0),
                )
                # v5.1216: el hecho de postura salía SOLO de la lista de
                # palabras del analizador, pero el concepto medido ya
                # existía (CONCEPTS["pose"] describe «dance/bailando»):
                # «el gato está bailando» medía pose 0,289 (0,79 con el
                # alcance «bailando») y seguía en la ruta anclada de cuerpo
                # ("exact same framing") — el gato se vestía pero no
                # bailaba. Misma competición que el resto de hechos
                # (≥0,20 y ×1,25); las keywords siguen mandando sin
                # condición (pose_hit previo intacto, nada regresa).
                pose_score = float(scores.get("pose", 0.0))
                if pose_score >= 0.20 and pose_score >= max(
                    body_score, appearance_score, recolor_score,
                    structural_score, background_score,
                ) * 1.25:
                    pose_hit = True
                    facts["pose"] = True
                scenario_leads = bool(
                    not pose_hit and not body_leads
                    and background_score >= 0.20
                    and background_score >= max(
                        body_score, appearance_score, recolor_score,
                        structural_operation,
                    ) * 1.25
                )
                local_candidate = max(structural_score, recolor_score)
                local_competition = max(body_score, appearance_score)
                region_leads = bool(
                    not pose_hit and not body_leads and not appearance_leads
                    and not scenario_leads
                    and local_candidate >= 0.20
                    and local_candidate >= local_competition * 1.25
                )
                judged.append({
                    "pose": pose_hit,
                    "body": body_leads,
                    # v5.1215: quién manda la cláusula — escenario vs región.
                    "scenario": scenario_leads,
                    "intensity": intensity,
                    "direction": direction,
                    "appearance": appearance_leads,
                    "region": region_leads,
                    "recolor": bool(
                        region_leads and recolor_score >= 0.20
                        # v5.1209: mismo factor 1,25 del resto de
                        # competiciones. El ancla de "pintar" acerca
                        # "quita la pintura" al recolor (0,285 vs 0,289); sin
                        # margen, un empate por ruido voltearía una
                        # eliminación a pintado.
                        and recolor_score >= structural_score * 1.25
                    ),
                    "ambiguous": bool(
                        structural_operation >= 0.20
                        and not pose_hit and not body_leads
                        and not appearance_leads and not region_leads
                    ),
                    # v5.1209: solo ELIMINAR cierra la zona (añadir o
                    # colorear no debe tragarse los objetos medidos).
                    "structural_remove": bool(remove_score >= 0.20),
                })
        body_hits = [item for item in judged if item["body"]]
        if body_hits:
            # v5.1217: la dirección mide tokens, no fonética: un typo
            # («ddesnuda», log real 23:40) deja su cláusula en "change" con
            # intensity=0.70 y, al ganar esa cláusula por intensidad, el
            # sufijo de desnudez se caía aunque la otra grafía de la MISMA
            # petición ya medía uncover ("go naked" también 0,70). La
            # dirección pertenece a la petición, no a una sola grafía: si la
            # cláusula líder no midió dirección, gana la dirección medida de
            # la siguiente cláusula corporal más fuerte.
            ordered = sorted(
                body_hits, key=lambda item: item["intensity"], reverse=True,
            )
            leading = ordered[0]
            facts["body"] = True
            facts["body_intensity"] = leading["intensity"]
            direction = str(leading["direction"])
            if direction not in ("uncover", "cover"):
                direction = next(
                    (str(item["direction"]) for item in ordered[1:]
                     if str(item["direction"]) in ("uncover", "cover")),
                    direction,
                )
            facts["body_direction"] = direction
        facts["authority"] = bool(facts["pose"] or facts["body"])
        # v5.1215: el escenario medido manda sobre la región local. Un
        # cambio de escenario no es «pintar una zona»: pide escena nueva y
        # el usuario acepta que las caras cambien con ella (prioridad:
        # que quede bien). Si alguna cláusula pide escenario, el pass se
        # desancla entero.
        facts["scenario"] = bool(
            not facts["pose"] and not facts["body"]
            and any(item.get("scenario") for item in judged)
        )
        # Same score relation as the body decision above. No edit-type list.
        # A wording that already measured a region is not cancelled by a
        # translation whose shared verb only ties with body.
        facts["local_appearance"] = bool(
            not facts["pose"] and not facts["body"] and not facts["scenario"]
            and any(item["appearance"] for item in judged)
        )
        facts["local_region"] = bool(
            not facts["pose"] and not facts["body"] and not facts["scenario"]
            and not facts["local_appearance"]
            and any(item["region"] for item in judged)
        )
        facts["surface_recolor"] = bool(
            facts["local_region"]
            and any(item["recolor"] for item in judged if item["region"])
        )
        facts["region_ambiguous"] = bool(
            not facts["pose"] and not facts["body"] and not facts["scenario"]
            and not facts["local_appearance"] and not facts["local_region"]
            and any(item["ambiguous"] for item in judged)
        )
        # v5.1209: si alguna cláusula midió una eliminación estructural, la
        # ruta de región cierra la máscara dentro de su caja; el resto (pose
        # / cuerpo / apariencia) ni llega a esa compuerta.
        facts["structural_remove"] = bool(
            any(item["structural_remove"] for item in judged)
        )
        return facts

    def _visual_subject_descriptions(self, text: str) -> list:
        """Visual subject phrases of one instruction, deduped by spelling."""
        if not text or not text.strip():
            return []
        pieces = [part.strip() for part in re.split(r"\s*[,;]\s*", text) if part.strip()]
        pieces.append(text.strip())
        out, seen = [], set()
        for piece in pieces:
            description = self._visual_subject_description(piece)
            key = description.casefold()
            if description and key not in seen:
                seen.add(key)
                out.append(description)
        return out

    def _subject_queries(self, raw_prompt: str, translated_prompt: str) -> list:
        """Visual subject phrases, one entry per spelling regardless of case."""
        queries, seen = [], set()
        for text in (translated_prompt, raw_prompt):
            for description in self._visual_subject_descriptions(text):
                key = description.casefold()
                if key not in seen:
                    seen.add(key)
                    queries.append(description)
        return queries

    def _translated_subject_counterpart(
        self, query: str, raw_prompt: str, translated_prompt: str,
    ) -> str:
        """Same visual subject as `query` but spelled in the translated text.

        The measured mask can win with either spelling (log real 22:25,
        edit_1792182345: 'Man' midió 0,01% y 'el hombre' 6,6% → el grupo
        quedó con el texto en español y la instrucción despachada a LongCat
        viajó sin traducir; el cumplimiento fue casi nulo y el hombre siguió
        con pelo). El emparejamiento es por posición: ambos textos se
        parten en las mismas cláusulas y el traductor es fiel.
        """
        raw_descriptions = self._visual_subject_descriptions(raw_prompt)
        translated_descriptions = self._visual_subject_descriptions(translated_prompt)
        key = (query or "").casefold()
        for index, description in enumerate(raw_descriptions):
            if description.casefold() == key:
                if index < len(translated_descriptions):
                    return translated_descriptions[index]
                return ""
        return ""

    @staticmethod
    def _grow_measured_subject(region: np.ndarray, faces, width: int, height: int) -> Optional[np.ndarray]:
        """Soften the edge of a measured subject that already covers a face.

        A tall full-height band repaints the wall and cuts the person beside
        them. The margin stays a small fraction of the measured region. The
        work copy keeps the long side at or below 768 px.
        """
        long_side = max(int(width), int(height))
        scale = 1.0 if long_side <= 768 else 768.0 / float(long_side)
        work_w = max(1, int(round(width * scale)))
        work_h = max(1, int(round(height * scale)))
        if scale < 1.0:
            small = cv2.resize(region, (work_w, work_h), interpolation=cv2.INTER_AREA)
        else:
            small = region
        matched = False
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            if scale < 1.0:
                x1, y1 = int(x1 * scale), int(y1 * scale)
                x2, y2 = int(round(x2 * scale)), int(round(y2 * scale))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(work_w, x2), min(work_h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if float((small[y1:y2, x1:x2] > 96).mean()) >= 0.15:
                matched = True
                break
        if not matched:
            return None
        ys, xs = np.nonzero(small > 96)
        if ys.size == 0:
            return None
        blob_h = int(ys.max() - ys.min() + 1)
        blob_w = int(xs.max() - xs.min() + 1)
        margin = max(3, int(round(0.06 * max(blob_h, blob_w))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        grown = cv2.dilate((small > 96).astype(np.uint8) * 255, kernel, iterations=1)
        top = max(0, int(ys.min()) - margin)
        bottom = min(work_h, int(ys.max()) + margin + 1)
        left = max(0, int(xs.min()) - margin)
        right = min(work_w, int(xs.max()) + margin + 1)
        limited = np.zeros_like(grown)
        limited[top:bottom, left:right] = grown[top:bottom, left:right]
        soft = cv2.GaussianBlur(limited, (0, 0), 1.2)
        if scale < 1.0:
            soft = cv2.resize(soft, (width, height), interpolation=cv2.INTER_LINEAR)
        return soft

    @staticmethod
    def _region_contains_a_face(region: np.ndarray, faces, width: int, height: int) -> bool:
        """Whether the measured region IS a person: a whole face sits inside it.

        A recolor whose requested part is not measurable must not repaint a
        whole person, so the abstention fires when the measured region is a
        detected face's person. _grow_measured_subject only asks whether the
        region touches a face (15% of its bbox), which a surface mask that
        bleeds over a face edge also satisfies — and then a measured wall
        behind a full-body person was thrown away as «a person without its
        part» and was never painted (2026-10-16: wall measured 13.9% of the
        photo, grazed ~33% of a tiny face bbox, discarded). A person contains
        its face; a surface merely grazes it. Containment is definitional —
        every pixel of the face bbox lies inside the region — so there is no
        tuned share to hardcode.
        """
        binary = np.asarray(region) > 96
        if binary.shape[:2] != (height, width):
            binary = cv2.resize(
                binary.astype(np.uint8), (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        for face in faces:
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            if bool(binary[y1:y2, x1:x2].all()):
                return True
        return False

    @staticmethod
    def _masks_are_same_subject(first: np.ndarray, second: np.ndarray) -> bool:
        left = np.asarray(first) > 96
        right = np.asarray(second) > 96
        if left.shape != right.shape:
            return False
        smaller = int(min(left.sum(), right.sum()))
        if smaller <= 0:
            return False
        shared = int(np.logical_and(left, right).sum())
        return shared / float(smaller) >= 0.45

    def _clauses_with_subject(self, text: str) -> list:
        clauses = []
        current_subject = ""
        for piece in re.split(r"\s*[,;]\s*", text or ""):
            piece = piece.strip()
            if not piece:
                continue
            subject = self._visual_subject_description(piece)
            if subject:
                current_subject = subject.casefold()
                clauses.append((current_subject, piece))
            elif current_subject:
                clauses.append((current_subject, piece))
            else:
                clauses.append(("", piece))
        return clauses

    def _group_measured_subjects(self, measured: list, raw_prompt: str, translated_prompt: str) -> list:
        """One edit per person. The same person worded twice stays one edit."""
        if not measured:
            return []
        parent = list(range(len(measured)))

        def find(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for index, item in enumerate(measured):
            for earlier in range(index):
                if self._masks_are_same_subject(item["region"], measured[earlier]["region"]):
                    parent[find(index)] = find(earlier)
        clusters = {}
        for index, item in enumerate(measured):
            clusters.setdefault(find(index), []).append(item)
        english = self._clauses_with_subject(translated_prompt)
        original = self._clauses_with_subject(raw_prompt)
        groups = []
        for items in clusters.values():
            names = {item["query"].casefold() for item in items}
            # La máscara puede ganar con cualquiera de las dos grafías (log
            # 22:25: 'Man' midió 0,01% y 'el hombre' 6,6% → instrucción
            # despachada en español y cumplimiento casi nulo). El nombre se
            # empareja por posición con su traducción para que el TEXTO que
            # viaja al modelo sea el traducido aunque la consulta cruda
            # haya medido la máscara.
            for item in items:
                counterpart = self._translated_subject_counterpart(
                    item["query"], raw_prompt, translated_prompt,
                )
                if counterpart:
                    names.add(counterpart.casefold())
            chosen = [piece for subject, piece in english if subject in names or (not subject and len(clusters) == 1)]
            if not chosen:
                chosen = [piece for subject, piece in original if subject in names or (not subject and len(clusters) == 1)]
            parts = []
            seen = set()
            for piece in chosen:
                key = piece.casefold()
                if key not in seen:
                    seen.add(key)
                    parts.append(piece)
            if not parts and len(clusters) == 1:
                full_text = (translated_prompt or raw_prompt or "").strip()
                if full_text:
                    parts = [full_text]
            if not parts:
                continue
            union = items[0]["region"]
            for item in items[1:]:
                union = np.maximum(union, item["region"])
            groups.append({
                "text": ", ".join(parts),
                "mask": Image.fromarray(union).convert("L"),
                "coverage": float((union > 96).mean()),
            })
        return groups

    def _instruction_needs_prompt_authority(self, raw_prompt: str, translated_prompt: str) -> bool:
        """Pose or body instructions lose when the photo is also a latent anchor."""
        return self._prompt_authority_facts(raw_prompt, translated_prompt)["authority"]

    def _measured_edit_mask(
        self,
        image: Image.Image,
        raw_prompt: str,
        translated_prompt: str,
        faces,
    ) -> Optional[Image.Image]:
        """Locate the described subject on this photo and grow that region.

        The query is the grammatical subject of the instruction. No gender,
        garment, or edit-type list decides who is edited. Faces only expand
        a region that already overlaps the measured subject.
        """
        passes = self._measured_subject_passes(image, raw_prompt, translated_prompt, faces)
        if len(passes) == 1:
            return passes[0]["mask"]
        return None

    @staticmethod
    def _load_region_model(config: dict):
        """Load the explicitly configured local model without downloading."""
        from roop.img_editor import prompt_rewriter  # prepares Windows DLL path

        configured = config["model_path"]
        root = prompt_rewriter.get_project_root()
        model_path = os.path.join(root, configured)
        if not os.path.isfile(model_path):
            print(f"[ImgEditor] Modelo local de zonas ausente: {model_path}", flush=True)
            return None
        context_size = int(config["context_size"])
        gpu_layers = int(config["gpu_layers"])
        if context_size <= 0 or gpu_layers < 0:
            raise ValueError("Configuración no válida del modelo local de zonas")
        # prompt_rewriter prepares the installed Windows CUDA DLL path.
        from llama_cpp import Llama
        return Llama(model_path=model_path, n_ctx=context_size,
                     n_gpu_layers=gpu_layers, verbose=False)

    @staticmethod
    def _region_proposals(instruction: str, config: dict) -> list:
        """Load only the configured local planner and close it after inference."""
        from roop.img_editor.prompt_rewriter import PromptRewriter

        llm = ImgEditorManager._load_region_model(config)
        if llm is None:
            return []
        try:
            return PromptRewriter._classify_edit_support(instruction, llm)
        finally:
            llm.close()

    def _ambiguous_region_translation(self, instruction: str) -> Optional[str]:
        """Translate only a measured ambiguous edit, using the local planner."""
        from roop.img_editor.prompt_rewriter import PromptRewriter

        try:
            import json

            root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            with open(os.path.join(root, "config", "model_configs.json"), encoding="utf-8") as stream:
                config = json.load(stream)["longcat"]["region_proposal"]
            translation = config["ambiguous_translation"]
            if translation["enabled"] is not True:
                return instruction
            llm = self._load_region_model(config)
            if llm is None:
                return None
            try:
                translated = PromptRewriter._translate_edit_instruction(instruction, llm, translation)
            finally:
                llm.close()
            if translated:
                print(f"[ImgEditor] Traducción local de la instrucción ambigua: {translated}", flush=True)
            return translated
        except Exception as exc:
            print(f"[ImgEditor] Traducción local no disponible: {exc}", flush=True)
            return None

    def _proposed_region_passes(self, image: Image.Image, instruction: str, masker, faces=()) -> list:
        """Ground a local language model's region proposal in this photo.

        This fallback keeps the complete instruction. The model supplies only
        segmenter queries whose words already occur in the request.
        """
        if masker is None:
            return []
        try:
            import json

            root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            with open(os.path.join(root, "config", "model_configs.json"), encoding="utf-8") as stream:
                config = json.load(stream)["longcat"]["region_proposal"]
            if config["enabled"] is not True:
                return []
            max_coverage = float(config["max_coverage"])
            if not 0.0 < max_coverage < 1.0:
                raise ValueError("region_proposal.max_coverage debe estar entre 0 y 1")
            proposals = self._region_proposals(instruction, config)
            tokens = set(re.findall(r"\w+", instruction.casefold()))
            grounded = [
                query for query in proposals
                if (words := set(re.findall(r"\w+", query.casefold())))
                and words.issubset(tokens)
            ]
            if len(grounded) != 1:
                print("[ImgEditor] La propuesta no identifica una sola zona de la instrucción", flush=True)
                return []
            best = None
            best_coverage = 0.0
            for query in grounded:
                try:
                    measured = masker.generate_mask(image, query)
                except Exception:
                    continue
                if measured is None:
                    continue
                measured = measured.convert("L")
                if measured.size != image.size:
                    measured = measured.resize(image.size, Image.Resampling.BILINEAR)
                coverage = self._mask_coverage(measured)
                if faces and self._grow_measured_subject(
                    np.asarray(measured), faces, image.width, image.height,
                ) is not None:
                    continue
                if best_coverage < coverage <= max_coverage:
                    best, best_coverage = measured, coverage
            if best is None:
                return []
            continued = self._continue_measured_region(best, np.asarray(best))
            final_mask = continued if continued is not None else best
            if faces and self._grow_measured_subject(
                np.asarray(final_mask), faces, image.width, image.height,
            ) is not None:
                return []
            print(
                f"[ImgEditor] Zona propuesta por modelo local y medida en la foto: {best_coverage:.1%}",
                flush=True,
            )
            return [{
                "text": instruction,
                "mask": final_mask,
                "coverage": best_coverage,
            }]
        except Exception as exc:
            print(f"[ImgEditor] Propuesta local de zona no disponible: {exc}", flush=True)
            return []

    def _grow_recolor_surface(
        self, image: Image.Image, region, faces=(), masker=None,
    ) -> Optional[Image.Image]:
        """Follow the measured color to the end of the surface.

        A recolor that names no distinct part requests the WHOLE surface
        («las paredes son de color gris» asks for the walls, not a patch of
        them). The segmenter lights only the fragment where its query is
        most visible — 5,8 % of the hotel photo (edit 22:39 of 17/10,
        ui_start_20261017_220140) — and `_continue_measured_region` only
        fills holes INSIDE that fragment, so the sampler hole and the paste
        covered a feathered patch and the rest of the wall kept its old
        color (edit_1792269760: the recurring patchy-walls defect, this
        time from coverage, not from routing — the v5.1209 recolor route
        did activate).

        The bound is the surface's own color envelope: the p90 spread of
        the measured fragment plus the same flat separation of 18 this path
        already uses. The surface's own shading stays in; the next surface
        (headboard, sheet, skin, curtain) lies past it and connectivity
        ends there. The person and measured-object guards run before the
        flood, the same guards the person path uses.
        """
        seed = np.asarray(region)
        if seed.ndim > 2:
            seed = seed[..., 0]
        seed = seed >= 96
        if int(seed.sum()) < 48:
            return None
        height, width = seed.shape
        photo = np.asarray(image.convert("RGB"), dtype=np.float32)
        if photo.shape[:2] != (height, width):
            photo = cv2.resize(
                photo, (width, height), interpolation=cv2.INTER_AREA,
            ).astype(np.float32)
        long_side = max(height, width)
        if long_side > 768:
            # The same work size as the other mask math: the color
            # distances of a 43 MP frame stall the edit before anything is
            # generated.
            scale = 768.0 / float(long_side)
            work_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            small_faces = []
            for face in faces or []:
                box = tuple(float(value) * scale for value in face.bbox)
                small_faces.append(SimpleNamespace(bbox=box))
            grown = self._grow_recolor_surface(
                image.resize(work_size, Image.Resampling.BOX),
                cv2.resize(
                    seed.astype(np.uint8) * 255, work_size,
                    interpolation=cv2.INTER_NEAREST,
                ),
                faces=small_faces, masker=masker,
            )
            if grown is None:
                return None
            return grown.resize((width, height), Image.Resampling.NEAREST)
        ys, xs = np.where(seed)
        span = max(
            int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1),
        )
        erode_r = max(2, int(round(span * 0.06)))
        erode_k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_r * 2 + 1, erode_r * 2 + 1),
        )
        core = cv2.erode(seed.astype(np.uint8), erode_k) > 0
        if int(core.sum()) < 48:
            core = seed
        median_color = np.median(photo[core], axis=0)
        color_distance = np.linalg.norm(photo - median_color, axis=2)
        spread = float(np.percentile(color_distance[core], 90))
        limit = max(spread + 18.0, 18.0)
        same = color_distance <= limit
        same |= seed
        same = ImgEditorManager._drop_other_person_contact(
            photo, same, seed, faces or (),
        )
        same = ImgEditorManager._drop_measured_object(
            photo, same, seed, faces or (), masker, None, None,
        )
        touch = seed & same
        if int(touch.sum()) < 48:
            return None
        _count, labels = cv2.connectedComponents(
            same.astype(np.uint8), connectivity=8,
        )
        keep = np.zeros(seed.shape, dtype=bool)
        for label in np.unique(labels[touch]):
            label = int(label)
            if label:
                keep |= labels == label
        added = int((keep & ~seed).sum())
        if added < 48:
            return None
        print(
            f"[ImgEditor] La zona recolor sigue la superficie completa "
            f"(+{added} px); donde el color cambia queda fuera",
            flush=True,
        )
        merged = np.where(keep, 255, 0).astype(np.uint8)
        merged = cv2.GaussianBlur(merged, (31, 31), 0)
        return Image.fromarray(merged, "L")

    def _measured_region_passes(
        self, image: Image.Image, raw_prompt: str, translated_prompt: str,
        surface_recolor: bool = False, faces=(),
    ) -> list:
        """One mask for a measured add, remove, or recolor.

        Direct measurement runs first. A structural request with no measured
        region may use a local model proposal checked by the same segmenter.
        Recolors never bypass their part selection through this fallback.
        """
        queries = self._subject_queries(raw_prompt, translated_prompt)
        if not queries:
            # A sentence is not a visible region. Ask for the predicate
            # remainder before falling back to the whole clause.
            queries = self._visual_request_queries([raw_prompt, translated_prompt])
        if not queries:
            text = (translated_prompt or raw_prompt or "").strip()
            if text:
                queries = [text]
        if not queries:
            return []
        if surface_recolor:
            # v5.1209: el recolor pregunta también el singular de cada frase
            # con artículo ('the walls' 2,0 % → 'the wall' 13,1 % en la misma
            # foto) y los sintagmas con artículo de la cláusula (el imperativo
            # no tiene predicado que dividir y solo llegaba la frase completa,
            # 1,33 %). Gana la cobertura con los mismos suelos y guards.
            variants = []
            for item in queries:
                singular = self._singular_surface_query(item)
                variants.append(item)
                if singular != item and singular not in variants:
                    variants.append(singular)
            for text in (raw_prompt, translated_prompt):
                for phrase in self._determiner_phrases(text):
                    for candidate in (phrase, self._singular_surface_query(phrase)):
                        if candidate and candidate not in variants:
                            variants.append(candidate)
            queries = variants
        try:
            masker = self._get_clipseg_masker()
        except Exception:
            return []
        if masker is None:
            return []
        best = None
        best_coverage = 0.0
        width, height = image.size
        for query in queries:
            try:
                candidate = masker.generate_mask(image, query)
            except Exception:
                continue
            if candidate is None:
                continue
            coverage = self._mask_coverage(candidate)
            if coverage < 0.04 or coverage > 0.90 or coverage <= best_coverage:
                continue
            region = np.asarray(candidate.convert("L"))
            if region.shape[:2] != (height, width):
                region = cv2.resize(region, (width, height), interpolation=cv2.INTER_LINEAR)
            best_coverage = coverage
            best = region
        if best is None:
            return [] if surface_recolor else self._proposed_region_passes(
                image, (translated_prompt or raw_prompt or "").strip(), masker, faces,
            )
        print(
            f"[ImgEditor] Región medida, cobertura={best_coverage:.1%}",
            flush=True,
        )
        subject = Image.fromarray(best).convert("L")
        requested = None
        if surface_recolor:
            # v5.1209: el imperativo no tiene scope de cambio; sin el
            # complemento "de gris"/"in grey" la medición del request era la
            # frase completa (1,33 %) y su área le ganaba a "grey", dejando el
            # pintado en una esquirla de la pared. Con complemento solo se
            # mide ese (la copula ya medía 'de color gris'/'grey'); sin él,
            # el flujo de partes de persona queda intacto.
            changes = [
                change for change in (
                    self._imperative_change(raw_prompt),
                    self._imperative_change(translated_prompt),
                ) if change
            ]
            requested = self._request_region_on_subject(
                image, subject,
                changes if changes else [raw_prompt, translated_prompt],
            )
        if requested is not None:
            # A recolor can name a part of a subject. The subject locator
            # supplies context; it is not the region to repaint.
            return [{
                "text": (translated_prompt or raw_prompt or "").strip(),
                "mask": requested,
                "person": subject,
                "coverage": self._mask_coverage(requested),
                "measured_request": True,
                "requested_part": True,
            }]
        if (surface_recolor and faces
                and self._region_contains_a_face(best, faces, width, height)):
            # v5.1215: la coincidencia de cara era «cualquier solape» (15 %
            # del bbox en _grow_measured_subject), y la máscara de una
            # superficie que sangra sobre una cara diminta (pared detrás de
            # alguien de cuerpo entero) cumplía ese suelo: la pared medida
            # (13,9 %) se descartaba como «persona sin parte» y no se
            # pintaba. Una persona contiene su cara; una superficie solo la
            # roza — la contención de la cara dentro de la región separa las
            # dos medidas sin preguntar por palabras y sin umbral.
            print(
                "[ImgEditor] Parte pedida no medida en la persona; no se sustituye la foto",
                flush=True,
            )
            return []
        zone = None
        if surface_recolor and requested is None:
            # A recolor that names no distinct part IS the whole surface
            # («las paredes son de color gris»); the segmenter fragment is
            # one patch of it. Follow the measured color to where the
            # surface ends instead of painting a feathered patch on an
            # otherwise untouched wall (edit_1792269760).
            zone = self._grow_recolor_surface(
                image, best, faces=faces, masker=masker,
            )
        if zone is None:
            mask = self._continue_measured_region(
                Image.fromarray(best).convert("L"), best,
            )
            zone = mask if mask is not None else Image.fromarray(best).convert("L")
        if surface_recolor:
            zone = self._close_recolor_halo(image, zone, faces) or zone
        return [{
            "text": (translated_prompt or raw_prompt or "").strip(),
            "mask": zone,
            "coverage": best_coverage,
        }]

    def _measured_subject_passes(
        self,
        image: Image.Image,
        raw_prompt: str,
        translated_prompt: str,
        faces,
        detected_faces=None,
    ) -> list:
        """One tight mask per person named in the instruction.

        The face-overlap gate accepts every DETECTED face, not only the
        credible ones (log real 22:51, edit_1792184313): el hombre a cuatro
        patas se detectó con det 0.31 frente al 0.89 de la mujer, el suelo
        de credibilidad descartó SU cara y la máscara 'Man' (que cubría su
        cabeza al 93,9%) no encontró cara que la confirmara → «Sujeto no
        medido» y la foto salió SIN EDITAR. Aquí la corroboración es la
        propia máscara de la consulta del sujeto cubriendo una caja
        detectada; el suelo de credibilidad sigue vigente para los anclajes
        de calor/head, donde nada más corrobora la detección (el speck del
        pie de 27×22 px en fast_1792015141).
        """
        gate_faces = detected_faces if detected_faces is not None else faces
        if not gate_faces:
            return []
        queries = self._subject_queries(raw_prompt, translated_prompt)
        if not queries:
            return []
        try:
            masker = self._get_clipseg_masker()
        except Exception:
            return []
        if masker is None:
            return []
        measured = []
        width, height = image.size
        for query in queries:
            try:
                candidate = masker.generate_mask(image, query)
            except Exception:
                continue
            if candidate is None:
                continue
            coverage = self._mask_coverage(candidate)
            if coverage <= 0.0 or coverage > 0.90:
                continue
            region = np.asarray(candidate.convert("L"))
            if region.shape[:2] != (height, width):
                region = cv2.resize(region, (width, height), interpolation=cv2.INTER_LINEAR)
            limited = self._grow_measured_subject(region, gate_faces, width, height)
            if limited is None:
                if coverage >= 0.04:
                    print(
                        f"[ImgEditor] '{query}' cubre {coverage:.1%} pero no coincide con una cara; se descarta",
                        flush=True,
                    )
                continue
            grown_coverage = float((limited > 96).mean())
            # The face overlap above is what makes this a person. In a
            # full-body group photo that person can be under 4% of the frame.
            # The old floor dropped them before the requested change was
            # measured on them. A mask with no face is already gone. Under
            # 48 px is still a speck.
            if int((limited > 96).sum()) < 48 or grown_coverage > 0.85:
                continue
            if coverage < 0.04:
                print(
                    f"[ImgEditor] '{query}' cubre {coverage:.1%} de la foto y coincide con una cara; "
                    "se mide esa persona",
                    flush=True,
                )
            measured.append({"query": query, "region": limited})
        groups = self._group_measured_subjects(measured, raw_prompt, translated_prompt)
        for index, group in enumerate(groups, start=1):
            print(
                f"[ImgEditor] Sujeto {index}/{len(groups)} medido, "
                f"cobertura={group['coverage']:.1%}: {group['text']}",
                flush=True,
            )
        return groups

    def _compound_clause_zone(
        self, image: Image.Image, clause: str, source: str, facts: dict, faces,
        detected_faces=None,
    ) -> Optional[Image.Image]:
        """Paste zone of one compound clause, measured without generating.

        The compound route is ONE generation of the full instruction. The
        prompt is still split, but only to measure where each clause lands
        on this photo: a measured surface is its own zone, and a body, pose
        or appearance clause edits a person, so that person is the zone.
        The generation is full-frame, so a 2% request patch would miss a
        limb the model placed a few pixels away. A clause that measures
        nothing returns None and is reported, never guessed.
        """
        raw = (source or clause).strip()
        translated = clause
        if ((facts.get("local_region") or facts.get("region_ambiguous"))
                and not facts.get("local_appearance")):
            passes = self._measured_region_passes(
                image, raw, translated,
                surface_recolor=bool(facts.get("surface_recolor")),
                faces=faces,
            )
        else:
            passes = self._measured_subject_passes(
                image, raw, translated, faces, detected_faces=detected_faces,
            )
            if not passes and facts.get("local_appearance"):
                recovered = self._appearance_head_pass(image, raw, translated, faces)
                if recovered is not None:
                    passes = [recovered]
        zone = None
        for item in passes:
            mask = item.get("mask")
            if mask is not None:
                measured = np.asarray(
                    mask.convert("L").resize(image.size, Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
                zone = measured if zone is None else np.maximum(zone, measured)
        return Image.fromarray(zone, "L") if zone is not None else None

    def _narrow_appearance_passes(
        self, image, passes, score_raw, prompt_raw, prompt, faces,
    ) -> list:
        """Keep the segmenter zone of the request inside each named person.

        A finer part is used only when the segmenter lights it. Otherwise
        the person already measured is the edit. A head shape or the
        strongest pixel spike is not a stand-in for the next instruction.
        """
        narrowed = []
        for item in passes:
            person_mask = item["mask"]
            requested = self._request_region_on_subject(
                image, person_mask,
                [score_raw, prompt_raw, prompt, item.get("text", "")],
            )
            deliver_region = False
            if requested is not None:
                completed = self._complete_request_surface(
                    image, requested, person_mask, faces,
                    masker=self._get_clipseg_masker(),
                )
                local_mask = completed if completed is not None else requested
                measured_request = True
            else:
                local_mask = self._measured_clause_region(person_mask)
                measured_request = False
                deliver_region = local_mask is not None
            if local_mask is None:
                print(
                    "[ImgEditor] Zona no medida en el sujeto; no se sustituye la foto",
                    flush=True,
                )
                continue
            narrowed.append({
                **item,
                "mask": local_mask,
                "person": person_mask,
                "measured_request": measured_request,
                "deliver_region": deliver_region,
            })
        return narrowed

    def _body_paste_zone(
        self, source, generated, subject_mask, request_region, texts, box,
    ) -> Optional[Image.Image]:
        """Measure a body's requested change in source and generated coordinates.

        A requested new state may not exist in the source. Measure it in the
        generated photo too, then accept only changed pixels on that person.
        A collateral change elsewhere does not count as an applied request.
        """
        box = tuple(int(value) for value in box)
        original_crop = source.convert("RGB").crop(box)
        updated_crop = generated.convert("RGB").resize(
            original_crop.size, Image.Resampling.LANCZOS,
        )
        updated_photo = source.convert("RGB").copy()
        updated_photo.paste(updated_crop, box[:2])
        state_region = self._request_region_on_subject(
            updated_photo, subject_mask, texts,
        )
        # v5.1204: sin parte distinta, la persona medida ES la zona (el
        # llamador ya decidió arriba: «Cuerpo sin parte distinta: la
        # persona medida es la zona»). La medida sobre el estado generado
        # solo puede AÑADIR a esa base: en la foto ya pegada, el falso
        # «parte» de piel deja de ser piel (el modelo la cambió) y pasaba
        # el guard de piel, degradando la zona de la persona entera a ese
        # speck (probe_editor_compound_fast_dispatch: 480 px y el píxel
        # (40,60) intacto; con la base correcta la zona vuelve a ser la
        # persona cambiada). Los píxeles sin cambios siguen fuera: la zona
        # final es cambio ∩ permitido.
        allowed = None
        if request_region is None:
            allowed = np.asarray(
                subject_mask.convert("L").resize(source.size, Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        for region in (request_region, state_region):
            if region is not None:
                measured = np.asarray(
                    region.convert("L").resize(source.size, Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
                allowed = measured if allowed is None else np.maximum(allowed, measured)
        if allowed is None:
            allowed = np.asarray(
                subject_mask.convert("L").resize(source.size, Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        local_allowed = Image.fromarray(allowed, "L").crop(box)
        changed = self._changed_zone_on_subject(
            updated_crop, original_crop, local_allowed,
        )
        if changed is None:
            return None
        zone = Image.new("L", source.size, 0)
        zone.paste(changed, box[:2])
        return zone

    def _apply_subject_passes(
        self, client, image, passes, seed, negative_prompt,
        lora_name, lora_strength, progress_callback, cancel_check,
        localize=False, faces=None,
    ):
        """Edit each measured person with only the sentence that names them."""
        current = image
        messages = []
        produced = False
        for index, item in enumerate(passes, start=1):
            if cancel_check and cancel_check():
                return None, "Cancelado", False
            facts = self._prompt_authority_facts(item["text"], item["text"])
            body_pass = bool(facts["body"] and not facts["pose"])
            job_prompt = self._compose_generation_prompt(
                item["text"], engine="longcat", magnitude=0.85,
                pose_change=False,
                body_transform=bool(facts["body"]),
                body_direction=str(facts["body_direction"]),
                body_intensity=float(facts["body_intensity"]),
            )
            if negative_prompt:
                job_prompt += f" Avoid: {negative_prompt}."
            print(
                f"[ImgEditor] Frase {index}/{len(passes)}: solo esta frase, sin cambiar la postura",
                flush=True,
            )
            before = current
            pass_mask = item["mask"]
            pass_limit = pass_mask if facts["local_appearance"] and faces else None
            pass_localize = localize
            measured_request = False
            request_region = None
            deliver_region = False
            if body_pass:
                request_region = self._request_region_on_subject(
                    current, pass_mask, [item.get("text", "")],
                )
                if request_region is not None:
                    request_region = self._continue_body_part(
                        current, request_region, pass_mask, faces,
                    )
            if facts["local_appearance"] and faces:
                requested = self._request_region_on_subject(
                    current, pass_mask, [item.get("text", "")],
                )
                if requested is not None:
                    # The segmenter already returned the requested part; the
                    # same completion as the single pass keeps the measured
                    # surface and the hair of this head instead of dropping
                    # everything the geometric head cut off.
                    completed = self._complete_request_surface(
                        current, requested, pass_mask, faces,
                        masker=self._get_clipseg_masker(),
                    )
                    focused = completed if completed is not None else requested
                    measured_request = True
                else:
                    focused = self._measured_clause_region(pass_mask)
                    measured_request = False
                    deliver_region = focused is not None
                if focused is None:
                    messages.append("zona de apariencia no asociable a una sola cara")
                    continue
                pass_mask = focused
                pass_localize = True
            work_image, work_mask, local_box = current, pass_mask, None
            if body_pass:
                # The requested state can be absent in the original photo.
                # Sample the measured person and verify the requested region
                # in both frames before pasting any change.
                work_image, work_mask, local_box = self._appearance_work_region(
                    current, pass_mask, item.get("person"),
                )
                if local_box is None:
                    local_box = (0, 0, current.size[0], current.size[1])
                    work_mask = pass_mask
            elif pass_localize:
                work_image, work_mask, local_box = self._appearance_work_region(
                    current, pass_mask, item.get("person"),
                )
                if local_box is None and deliver_region:
                    local_box = (0, 0, current.size[0], current.size[1])
                    work_mask = pass_mask
            from roop.img_editor.flux_edit_comfy_client import measured_appearance_sample, measured_region_sample
            sample = (
                measured_appearance_sample(work_mask is not None)
                if facts["local_appearance"] or facts.get("surface_recolor")
                else measured_region_sample(True) if pass_localize and work_mask is not None
                else {
                    "num_inference_steps": 12, "denoise": 1.0,
                    "no_kontext": work_mask is None,
                }
            )
            vision_reference = None
            if facts["local_appearance"] and work_mask is not None and not deliver_region:
                sample, vision_reference = self._with_covered_reference(
                    sample, work_image, work_mask, item.get("person"), local_box,
                )
                print(
                    "[ImgEditor] Apariencia: una pasada, "
                    f"{sample['num_inference_steps']} pasos, fuerza {sample['denoise']:.2f}",
                    flush=True,
                )
            try:
                result, msg = client.generate(
                    image=work_image, prompt=job_prompt,
                    negative_prompt=negative_prompt,
                    guidance_scale=1.0,
                    seed=seed,
                    prompt_authority=True, preserve_frame=True,
                    mask_image=work_mask,
                    keep_source_pixels=bool(pass_localize or body_pass),
                    reference_image=vision_reference,
                    lora_name=lora_name, lora_strength=lora_strength,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    **sample,
                )
            except Exception as error:
                messages.append(str(error))
                continue
            messages.append(msg or "")
            if result is None:
                continue
            produced = True
            updated = result.image.convert("RGB")
            # Appearance already sampled the measured surface. The surrounding
            # fill does not apply the instruction and leaves the previous
            # surface. Region edits can still use it when they insert a face.
            if (pass_localize and work_mask is not None
                    and not facts["local_appearance"] and not body_pass):
                updated, repair_msg = self._repair_inserted_surface(
                    client, work_image, work_mask, updated, faces, local_box,
                    job_prompt, seed, negative_prompt, lora_name, lora_strength,
                    progress_callback, cancel_check,
                )
                if repair_msg:
                    messages.append(repair_msg)
            body_zone = None
            if body_pass and pass_mask is not None:
                zone_box = (
                    local_box
                    if local_box is not None
                    else (0, 0, before.size[0], before.size[1])
                )
                body_zone = self._body_paste_zone(
                    before, updated, pass_mask, request_region,
                    [item.get("text", "")], zone_box,
                )
                updated = (
                    self._integrate_local_generation(
                        before, updated, body_zone, zone_box,
                        keep_measured_mask=True,
                    ) if body_zone is not None else before
                )
            elif local_box is not None:
                updated = self._integrate_local_generation(
                    current, updated, pass_mask, local_box,
                    limit_mask=pass_limit,
                    keep_measured_mask=bool(measured_request or deliver_region),
                )
            elif updated.size != current.size:
                updated = updated.resize(current.size, Image.Resampling.LANCZOS)
            scope_missing = body_pass and body_zone is None
            if ((self._edit_is_nearly_unchanged(before, updated) or scope_missing)
                    and not (cancel_check and cancel_check())):
                print(
                    "[ImgEditor] Cambio casi nulo en la zona; segundo intento "
                    "Turbo con la instrucción de refuerzo",
                    flush=True,
                )
                retry_prompt = (
                    f"The input photo has not yet been changed. Do this visible edit now: {item['text']}. "
                    "Keep the pose and the framing. "
                    "Do not keep the previous state of what was asked to change. "
                    "Leave every other person as they are."
                )
                if negative_prompt:
                    retry_prompt += f" Avoid: {negative_prompt}."
                retry_prompt += self._bald_recipe_tail(item["text"])
                try:
                    retry, retry_msg = client.generate(
                        image=work_image, prompt=retry_prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=1.0,
                        seed=seed,
                        prompt_authority=True, preserve_frame=True,
                        mask_image=work_mask,
                        keep_source_pixels=bool(pass_localize or body_pass),
                        reference_image=vision_reference,
                        lora_name=lora_name, lora_strength=lora_strength,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                        **sample,
                    )
                except Exception as error:
                    retry, retry_msg = None, str(error)
                if retry is not None:
                    updated = retry.image.convert("RGB")
                    if body_pass and pass_mask is not None:
                        zone_box = (
                            local_box
                            if local_box is not None
                            else (0, 0, before.size[0], before.size[1])
                        )
                        body_zone = self._body_paste_zone(
                            before, updated, pass_mask, request_region,
                            [item.get("text", "")], zone_box,
                        )
                        updated = (
                            self._integrate_local_generation(
                                before, updated, body_zone, zone_box,
                                keep_measured_mask=True,
                            ) if body_zone is not None else before
                        )
                    elif local_box is not None:
                        updated = self._integrate_local_generation(
                            before, updated, pass_mask, local_box,
                            limit_mask=pass_limit,
                            keep_measured_mask=bool(measured_request or deliver_region),
                        )
                    elif updated.size != current.size:
                        updated = updated.resize(current.size, Image.Resampling.LANCZOS)
                    print(f"[ImgEditor] Segundo intento aplicado: {retry_msg}", flush=True)
                    messages.append(f"segundo intento: {retry_msg}")
                else:
                    print(f"[ImgEditor] Segundo intento falló: {retry_msg}", flush=True)
                    messages.append(f"segundo intento falló: {retry_msg}")
            if body_pass and pass_mask is not None:
                if body_zone is None:
                    messages.append(
                        f"«{str(item.get('text', '')).strip()}» sin cambio "
                        "medido en la zona pedida"
                    )
                    print(
                        "[ImgEditor] Cuerpo: sin cambio medido en la zona "
                        "pedida; la foto se conserva",
                        flush=True,
                    )
                elif request_region is None:
                    print(
                        "[ImgEditor] El predicado no tiene zona; se pega "
                        "la región ya medida",
                        flush=True,
                    )
                else:
                    print(
                        "[ImgEditor] Cuerpo: solo el cambio medido dentro de "
                        "la petición; el resto de la foto se conserva",
                        flush=True,
                    )
            elif local_box is not None:
                print("[ImgEditor] Región integrada sobre la foto original", flush=True)
            current = updated
        if not produced:
            return None, "; ".join(messages), False
        return current, "; ".join(messages), True

    @staticmethod
    def _visual_subject_description(clause: str) -> str:
        """Keep the visible subject description, including pose and location.

        The predicate describes the edit. The prefix is only a visual query;
        its meaning is checked against pixels by the segmenter before use.
        """
        text = (clause or "").strip()
        matches = list(ImgEditorManager._PREDICATE_RE.finditer(text))
        match = matches[0] if matches else None
        if match and len(matches) > 1 and re.search(
            r"\b(?:who|that|which|que)\b", text[:match.start()], flags=re.I
        ):
            match = matches[1]
        if match:
            return text[:match.start()].strip(" ,.;")
        # v5.1221: una cláusula que ABRE con sintagma determinante nombra a
        # su sujeto aunque el verbo posterior no esté en el patrón («the
        # woman wears no stockings»: wears no casa con is/has/go/tiene; sin
        # sujeto la cláusula no midió persona ni zona — log real 13:47:
        # «zonas medidas=0», foto entera regenerada sin verificación y la
        # pierna que cruza hacia el hombre salió con apariencia de media).
        # Solo gramática, sin lista de prendas: el primer sintagma inicial
        # es la propuesta; el segmentador y la compuerta de cara siguen        # decidiendo si esa persona existe en la foto.
        lead = ImgEditorManager._determiner_phrases(text)
        if lead and text.lower().startswith(lead[0].lower()):
            return lead[0]
        return ""

    @staticmethod
    def _attach_subject(clause: str, subject: str) -> str:
        """Give a stripped predicate back the person the whole instruction names.

        Splitting "Man must be bald and go naked" on the conjunction leaves
        "go naked" with no person, and the global body pass then edits
        whoever the model picks instead of the named one (log 20:49: both
        people nude from a request for the man only). The subject is the
        user's own phrase, so no edit vocabulary is introduced.
        """
        text = (clause or "").strip()
        subject = (subject or "").strip(" ,.;")
        if not text or not subject:
            return clause
        if re.match(r"(?:is|are|was|were)\b", text, flags=re.I):
            return f"{subject} {text}"
        return f"{subject} must {text}"

    @staticmethod
    def _leading_visual_subject(prompt: str, llm_subject: str = "") -> str:
        """Recover an explicitly named subject when the small planner omits regions.

        This only proposes a CLIPSeg query. The face-overlap check must still
        identify one head before any focal crop is allowed.
        """
        tokens = re.findall(r"[a-z]+(?:['-][a-z]+)?", (prompt or "").lower())
        if not tokens:
            return ""
        if llm_subject:
            subject_tokens = re.findall(r"[a-z]+", llm_subject.lower())
            if subject_tokens and any(token in tokens for token in subject_tokens):
                return llm_subject.strip().lower()
        lead_words = {"please", "make", "turn", "change", "edit", "set", "render",
                      "give", "let", "the", "a", "an"}
        while tokens and tokens[0] in lead_words:
            tokens.pop(0)
        # A bare instruction such as "be bald" does not name a person.
        if not tokens or tokens[0] in {"be", "is", "are", "must", "should", "have", "has"}:
            return ""
        return tokens[0] if len(tokens[0]) >= 3 else ""

    @staticmethod
    def _is_hair_edit(prompt: str) -> bool:
        """Keep hair edits on the scalp when a local model changes the face too."""
        return bool(re.search(
            r"\b(?:bald|balding|hair|hairstyle|shaved|shave|wig|calvo|calva|pelo|cabello)\b",
            (prompt or "").lower(),
        ))

    @staticmethod
    def _is_bald_request(prompt: str) -> bool:
        return bool(re.search(r"\b(?:bald|balding|calvo|calva)\b", prompt or "", flags=re.I))

    @staticmethod
    def _bald_recipe_tail(text: str) -> str:
        """Definición del estado calvo en los prompts despachados.

        Log real 21:51 (edit_1792180391): el modelo quitó el pelo pero
        conservó la diadema con pompones sobre la cabeza y pintó el cuero
        cabelludo gris — la cabeza salió como un «gorrito» blanco. Toda
        plantilla que despacha una cláusula calvo define el estado pedido:
        cuero cabelludo pelado, sin cubierta y con el tono natural de la
        piel de esa persona.
        """
        if not ImgEditorManager._is_bald_request(text):
            return ""
        return (
            " A bald head means a completely shaved scalp: no hair and no"
            " headband, hat or any headwear on the head; the bare scalp"
            " keeps the person's natural skin tone and texture."
        )

    @staticmethod
    def _bald_hair_coverage(source: Image.Image, result: Image.Image,
                            head_mask: Image.Image, masker) -> Optional[Tuple[float, float]]:
        """Measure residual hair on the named head, excluding its beard and neck."""
        size = source.size
        head = np.asarray(head_mask.convert("L").resize(size, Image.Resampling.NEAREST)) >= 128
        ys = np.flatnonzero(head.any(axis=1))
        if ys.size == 0:
            return None
        top, bottom = int(ys[0]), int(ys[-1])
        head[int(top + 0.60 * (bottom - top)) + 1:, :] = False
        if float(head.mean()) < 0.005:
            return None
        source_hair = masker.generate_mask(source, "hair")
        if source_hair is None:
            return None
        result_hair = masker.generate_mask(result.resize(size, Image.Resampling.LANCZOS), "hair")
        before = float((np.asarray(source_hair.convert("L").resize(size)) >= 96)[head].mean())
        after = (float((np.asarray(result_hair.convert("L").resize(size)) >= 96)[head].mean())
                 if result_hair is not None else 0.0)
        return before, after

    @staticmethod
    def _unique_subject_hair_mask(image: Image.Image, subject_mask: Image.Image,
                                  hair_mask: Image.Image) -> Optional[Image.Image]:
        """Find one visible hair region inside a grounded subject silhouette.

        A profile can have visible hair while its face is missed by the face
        detector. Multiple hair regions are ambiguous and must not select a
        person arbitrarily.
        """
        if subject_mask is None or hair_mask is None:
            return None
        subject = np.asarray(subject_mask.convert("L")) >= 80
        hair = np.asarray(hair_mask.convert("L")) >= 96
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            hair.astype(np.uint8), connectivity=8,
        )
        candidates = []
        minimum_area = max(60, int(image.width * image.height * 0.001))
        for idx in range(1, count):
            area = int(stats[idx, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            component = labels == idx
            if float((component & subject).sum()) / area >= 0.65:
                candidates.append(component)
        if len(candidates) != 1:
            return None
        radius = max(8, int(round(min(image.size) * 0.03)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
        editable = cv2.dilate(candidates[0].astype(np.uint8), kernel)
        near_subject = cv2.dilate(
            subject.astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
        )
        editable &= near_subject
        return Image.fromarray(editable * 255, "L")

    @staticmethod
    def _grounded_hair_mask(image: Image.Image, head_mask: Image.Image,
                            subject_mask: Image.Image, hair_mask: Image.Image) -> Optional[Image.Image]:
        """Select only the named person's hair, with room for its outer edge."""
        if head_mask is None or subject_mask is None or hair_mask is None:
            return None
        head = np.asarray(head_mask.convert("L")) > 80
        core = np.asarray(head_mask.convert("L")) >= 160
        subject = np.asarray(subject_mask.convert("L")) > 80
        hair = np.asarray(hair_mask.convert("L")) > 80
        ys = np.where(core)[0]
        if ys.size == 0:
            return None
        scalp_end = int(ys.min() + 0.56 * (ys.max() - ys.min()))
        above_face = np.arange(head.shape[0])[:, None] <= scalp_end
        radius = max(8, int(round(min(image.size) * 0.045)))
        near = cv2.dilate(head.astype(np.uint8), np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)) > 0
        visible_hair = hair & subject & near
        # CLIPSeg can see the entire hair mass in some photos and only a thin
        # strip in others. Use its tight mask when it is substantial; otherwise
        # include the upper head geometry to avoid leaving a hairy ring.
        hair_to_head = float(visible_hair.sum()) / max(1, int(head.sum()))
        selected = visible_hair if hair_to_head >= 0.18 else (((head & above_face) | visible_hair) & subject)
        coverage = float(selected.mean())
        if coverage < 0.003 or coverage > 0.20:
            return None
        border_share = 0.012 if hair_to_head >= 0.18 else 0.020
        border = max(4, int(round(min(image.size) * border_share)))
        expanded = cv2.dilate(selected.astype(np.uint8), np.ones((border * 2 + 1, border * 2 + 1), np.uint8))
        return Image.fromarray((expanded * 255).astype(np.uint8), "L")

    @staticmethod
    def _measured_clause_region(person_mask: Optional[Image.Image]) -> Optional[Image.Image]:
        """Region already measured for the clause, when the predicate lights nothing.

        The predicate is the change. It is often not in the photo yet, so
        the segmenter cannot draw it. The clause already measured who or
        what it is about. That region is the edit for any instruction. A
        head, a torso cut, or the strongest pixel spike only fits the
        request that happened to look like it.
        """
        if person_mask is None:
            return None
        subject = np.asarray(person_mask.convert("L"))
        if int((subject >= 96).sum()) < 48:
            return None
        print(
            "[ImgEditor] El predicado no tiene zona; entra la región ya medida "
            "y se pega entera",
            flush=True,
        )
        return person_mask

    def _head_zone_for_subject(
        self, image: Image.Image, subject_mask: Image.Image, faces,
        announce: bool = True,
    ) -> Optional[Image.Image]:
        """Head of the measured person, for appearance edits without a part.

        "bald" or "beard" are often not in the photo yet, so the segmenter
        cannot draw them and the clause falls back to the whole person. A
        full-photo pass has no spatial anchor and the model paints the
        scalp glossy (2026-10-12, edit_1791760178.png). The head of the
        measured person is where those changes live; intersecting the head
        candidates with the measured person keeps the zone on that person.
        """
        if subject_mask is None or not faces:
            return None
        head_union = self._head_mask_from_faces(image, faces)
        if head_union is None:
            return None
        person = np.asarray(subject_mask.convert("L")) >= 128
        head = np.asarray(head_union.convert("L")) >= 128
        zone = Image.fromarray(((head & person).astype(np.uint8)) * 255, "L")
        if int((np.asarray(zone) >= 128).sum()) < 48:
            return None
        if announce:
            print(
                "[ImgEditor] Cabeza del sujeto medida para la edición de apariencia; "
                "pase local con ancla espacial",
                flush=True,
            )
        return zone

    def _covered_surface_view(
        self, image: Image.Image, mask: Image.Image,
        person: Optional[Image.Image] = None,
    ) -> Optional[Image.Image]:
        """Photo with the measured surface painted in the color around it.

        Qwen copies a surface it can still see (a hair mask left the hair).
        The fill is the median of the person ring that is a different color
        from that surface. Pixels of the same color, and pixels outside the
        person, are not the fill. Only the pixels that actually differ are
        painted, so a face that shares the ring color stays. A flat zone has
        nothing to cover and is not sent.
        """
        if image is None or mask is None:
            return None
        photo = np.asarray(image.convert("RGB"), dtype=np.float32)
        zone = np.asarray(
            mask.convert("L").resize(image.size, Image.Resampling.NEAREST)
        ) >= 160
        if int(zone.sum()) < 48:
            return None
        if person is None:
            return None
        subject = np.asarray(
            person.convert("L").resize(image.size, Image.Resampling.NEAREST)
        ) >= 96
        span = 1
        ys, xs = np.where(zone)
        if len(ys):
            span = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1), 1)
        radius = max(3, int(round(span * 0.12)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
        )
        ring = (cv2.dilate(zone.astype(np.uint8), kernel) > 0) & ~zone & subject
        surface = np.median(photo[zone], axis=0)
        distance = np.linalg.norm(photo - surface, axis=2)
        usable = ring & (distance > 18.0)
        if int(usable.sum()) >= 48:
            fill = np.median(photo[usable], axis=0)
        else:
            fill = surface
        covered_px = zone & (np.linalg.norm(photo - fill, axis=2) > 18.0)
        if int(covered_px.sum()) < 48:
            return None
        painted = photo.copy()
        painted[covered_px] = fill
        print(
            f"[ImgEditor] Superficie cubierta para Qwen: {int(covered_px.sum())} px "
            "con el color de alrededor; el resto de la foto sigue",
            flush=True,
        )
        return Image.fromarray(np.clip(painted, 0, 255).astype(np.uint8), "RGB")

    def _reference_for_work(
        self, image: Image.Image, mask: Image.Image,
        person: Optional[Image.Image], local_box,
    ) -> Optional[Image.Image]:
        """Cover aligned to the crop that is about to be sampled.

        The person mask is still the full photo. Cropping it to the work
        window keeps the ring on the same pixels as the zone. A cover that
        does not match the crop is not sent.
        """
        if image is None or mask is None or person is None:
            return None
        person_view = person
        if person_view.size != image.size and local_box is not None:
            person_view = person_view.crop(tuple(int(value) for value in local_box))
        if person_view.size != image.size:
            person_view = person_view.resize(image.size, Image.Resampling.NEAREST)
        return self._covered_surface_view(image, mask, person_view)

    def _with_covered_reference(self, sample: dict, image, mask, person, local_box):
        """Drop the spatial photo anchor only when a cover actually exists.

        no_kontext without that cover removes the photo the sampler was
        anchored to. Body and region edits never reach this.

        When the recipe already anchors the photo spatially
        (no_kontext=False), the cover is NOT sent: the flat fill hides
        the very surface the model must edit, and the bald head came out
        pale and plasticky (2026-10-12, edit_1791758193.png). The
        original photo plus the spatial anchor is the versatile recipe
        for any appearance change.
        """
        if sample.get("no_kontext") is False:
            return sample, None
        reference = self._reference_for_work(image, mask, person, local_box)
        if reference is None:
            return sample, None
        covered = dict(sample)
        covered["no_kontext"] = True
        return covered, reference

    def _continue_body_part(
        self, image: Image.Image, part: Image.Image,
        subject: Image.Image, faces,
    ) -> Image.Image:
        """Same-color rest of a measured body part.

        The segmenter often lights only the start of a long surface. The
        rest is the same color. Hair is not part of this request. A
        continuation that covers the person is the person, and the part
        already measured stays. Pixels of that part outside the silhouette
        stay too: clipping them off drops the object and keeps the skin.
        """
        if part is None or subject is None:
            return part
        try:
            masker = self._get_clipseg_masker()
        except Exception:
            masker = None
        extended = self._complete_request_surface(
            image, part, subject, faces,
            masker=masker,
            complete_hair=False,
        )
        if extended is None:
            return part
        subject_on = np.asarray(
            subject.convert("L").resize(image.size, Image.Resampling.NEAREST)
        ) >= 96
        part_on = np.asarray(
            part.convert("L").resize(image.size, Image.Resampling.NEAREST)
        ) >= 128
        zone = np.asarray(
            extended.convert("L").resize(image.size, Image.Resampling.NEAREST)
        ) >= 128
        zone = (zone & subject_on) | part_on
        area = int(zone.sum())
        original = int(part_on.sum())
        if area <= original:
            return part
        image_area = max(1, image.width * image.height)
        subject_area = max(1, int(subject_on.sum()))
        if area / image_area > 0.35 or area / subject_area > 0.55:
            print(
                "[ImgEditor] La superficie del cuerpo cubre a la persona; "
                "se queda la zona medida",
                flush=True,
            )
            return part
        print(
            f"[ImgEditor] La zona del cuerpo sigue por la misma superficie "
            f"({area} px); la persona no entra entera",
            flush=True,
        )
        return Image.fromarray(zone.astype(np.uint8) * 255, "L")

    def _local_appearance_mask(
        self, image: Image.Image, subject_mask: Image.Image, faces,
        continue_region: bool = True,
    ) -> Optional[Image.Image]:
        """Measured focus that overlaps the subject already found on this photo."""
        return self._head_mask_from_region_plan(
            image, faces, subject_mask, continue_region=continue_region,
        )

    @staticmethod
    def _leading_face(heat: np.ndarray, faces):
        """Face whose box the segmenter scores above the others.

        The margin is the same 1.25 relation used when one measured concept
        leads the rest. A flat map does not choose a face.
        """
        if heat is None or not len(faces):
            return None, 0.0, 0.0
        height, width = heat.shape[:2]
        scores = []
        for face in faces:
            x1, y1, x2, y2 = [int(round(float(v))) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            patch = heat[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None
            scores.append(float(patch.mean()) if patch is not None and patch.size else 0.0)
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        best = scores[order[0]]
        second = scores[order[1]] if len(order) > 1 else 0.0
        baseline = float(np.mean(heat))
        if best < baseline * 1.25:
            return None, best, second
        if len(order) > 1 and best < second * 1.25:
            return None, best, second
        return faces[order[0]], best, second

    @staticmethod
    def _same_face(first, second) -> bool:
        if first is second:
            return True
        left = [float(value) for value in first.bbox]
        right = [float(value) for value in second.bbox]
        return all(abs(a - b) < 2.0 for a, b in zip(left, right))

    @staticmethod
    def _faces_in_work(faces, local_box, work_size):
        """Source faces moved into the crop the sampler just edited."""
        if not faces:
            return []
        origin_x, origin_y = (0, 0) if not local_box else (int(local_box[0]), int(local_box[1]))
        width, height = work_size
        shifted = []
        for face in faces:
            box = [float(value) for value in face.bbox]
            moved = [
                box[0] - origin_x, box[1] - origin_y,
                box[2] - origin_x, box[3] - origin_y,
            ]
            center_x = (moved[0] + moved[2]) / 2.0
            center_y = (moved[1] + moved[3]) / 2.0
            if 0.0 <= center_x < width and 0.0 <= center_y < height:
                shifted.append(type("WorkFace", (), {"bbox": moved}))
        return shifted

    @staticmethod
    def _inserted_face_in_mask(source_faces, result_faces, mask) -> bool:
        """A result face sitting in the measured zone and on none of the source faces.

        The zone can redraw the person already there. It cannot gain another one.
        """
        if mask is None or not result_faces:
            return False
        mask_px = np.asarray(mask.convert("L"))
        height, width = mask_px.shape[:2]

        def center(face):
            x1, y1, x2, y2 = [float(value) for value in face.bbox]
            return (x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1)

        known = [center(face) for face in source_faces or []]
        for face in result_faces:
            center_x, center_y, size = center(face)
            pixel_x, pixel_y = int(round(center_x)), int(round(center_y))
            if not (0 <= pixel_x < width and 0 <= pixel_y < height):
                continue
            if int(mask_px[pixel_y, pixel_x]) < 96:
                continue
            limit = max(12.0, size * 0.45)
            if any(
                (center_x - known_x) ** 2 + (center_y - known_y) ** 2 <= limit ** 2
                for known_x, known_y, _known_size in known
            ):
                continue
            return True
        return False

    @staticmethod
    def _continue_measured_surface(image: Image.Image, mask: Image.Image) -> Image.Image:
        """Fill the measured hole from the pixels around it.

        An empty noise hole is a blank canvas, so the sampler paints a new
        picture there. The surrounding photo already has the surface that
        should continue.
        """
        import cv2
        rgb = np.asarray(image.convert("RGB"))
        mask_px = np.asarray(
            mask.convert("L").resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.NEAREST)
        )
        hole = (mask_px >= 96).astype(np.uint8) * 255
        if int(hole.max()) == 0 or int((hole > 0).sum()) < 16:
            return image
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        hole = cv2.dilate(hole, kernel)
        filled = cv2.inpaint(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), hole, 12, cv2.INPAINT_NS,
        )
        return Image.fromarray(cv2.cvtColor(filled, cv2.COLOR_BGR2RGB))

    def _repair_inserted_surface(
        self, client, work_image, work_mask, result_image, source_faces, local_box,
        prompt, seed, negative_prompt, lora_name, lora_strength,
        progress_callback, cancel_check,
    ):
        """Redo a measured zone when the sample drew a new face into it."""
        if work_image is None or work_mask is None or result_image is None:
            return result_image, None
        if result_image.size != work_image.size:
            result_image = result_image.resize(work_image.size, Image.Resampling.LANCZOS)
        try:
            from roop.face_util import detect_faces_robust, get_face_analyser_cpu
            analyser = get_face_analyser_cpu()
            if analyser is None:
                return result_image, None
            found = detect_faces_robust(
                np.asarray(result_image.convert("RGB")), analyser, 0.30,
            ) or []
        except Exception as error:
            print(f"[ImgEditor] No se pudo medir si la zona insertó una cara: {error}", flush=True)
            return result_image, None
        known = self._faces_in_work(source_faces, local_box, work_image.size)
        if not self._inserted_face_in_mask(known, found, work_mask):
            return result_image, None
        print(
            "[ImgEditor] La zona medida contenía otra cara; se rehace desde la superficie de alrededor",
            flush=True,
        )
        from roop.img_editor.flux_edit_comfy_client import SURFACE_CONTINUATION_DENOISE
        filled = self._continue_measured_surface(work_image, work_mask)
        try:
            repaired, message = client.generate(
                image=filled, prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=12, guidance_scale=1.0,
                denoise=SURFACE_CONTINUATION_DENOISE, seed=seed, no_kontext=True,
                prompt_authority=True, preserve_frame=True,
                mask_image=work_mask, keep_source_pixels=True,
                lora_name=lora_name, lora_strength=lora_strength,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
            )
        except Exception as error:
            print(f"[ImgEditor] La superficie continuada falló: {error}", flush=True)
            return result_image, None
        if repaired is None or getattr(repaired, "image", None) is None:
            return result_image, None
        repaired_image = repaired.image.convert("RGB")
        if repaired_image.size != work_image.size:
            repaired_image = repaired_image.resize(work_image.size, Image.Resampling.LANCZOS)
        return repaired_image, f"superficie continuada: {message}"

    @staticmethod
    def _credible_faces(faces):
        """Keep the detections that can anchor an edit (medido 2026-10-16).

        On fast_1792015141.png («Man must be bald») the detector returned the
        man (det 0.69, 100×137 px) plus a 27×22 px false face on his foot (det
        0.30); the subject heat lit the foot skin (0.225) over the real face
        (0.097) and the whole edit anchored to the toes
        (probe_editor_bald_backhair_20261016). Real second people measured on
        the fixture photos stay at 0.57× or more of the strongest face
        (office 0.84, fast_1791410285 0.82, fast_1791415803 0.83,
        fast_1791420497 0.57) while the speck sits at 0.43×; a detection
        below half of the best score is skin noise, not a person. Faces
        without det_score (probe fixtures) and photos whose only detection is
        itself noise keep every face, since a relative floor has nothing
        stronger to compare against.
        """
        if not faces:
            return list(faces)
        best = max(float(getattr(face, "det_score", 0.0) or 0.0) for face in faces)
        if best <= 0.0:
            return list(faces)
        kept = [
            face for face in faces
            if float(getattr(face, "det_score", 0.0) or 0.0) >= 0.5 * best
        ]
        return kept if kept else list(faces)

    def _appearance_head_pass(self, image: Image.Image, raw_prompt: str, translated_prompt: str, faces):
        """Head of the face the subject query lights, when the binary mask is empty."""
        if not faces:
            return None
        queries = self._subject_queries(raw_prompt, translated_prompt)
        if not queries:
            return None
        try:
            masker = self._get_clipseg_masker()
        except Exception:
            return None
        if masker is None or not hasattr(masker, "probability"):
            return None
        hits = []
        for query in queries:
            try:
                heat = masker.probability(image, query)
            except Exception:
                continue
            face, score, runner = self._leading_face(heat, faces)
            if face is None:
                continue
            hits.append((query, face, score, runner))
        if not hits:
            return None
        # The words before "debe" / "must" are how one person is told from
        # the others. One wording that lights a face more strongly than the
        # rest is enough. A translation that lights a different face does not
        # cancel it. Two wordings that light different faces with similar
        # strength do not say which person was asked for.
        _query, chosen, best_score, second_score = max(hits, key=lambda item: item[2])
        rivals = [item for item in hits if not self._same_face(item[1], chosen)]
        rival_score = max((item[2] for item in rivals), default=0.0)
        if rivals and best_score < rival_score * 1.25:
            print(
                "[ImgEditor] Hay varias personas y la frase no separa cuál; "
                "no se sustituye la foto",
                flush=True,
            )
            return None
        if rivals:
            print(
                f"[ImgEditor] La frase señala una cara "
                f"({best_score:.3f} frente a {rival_score:.3f}); "
                "las demás quedan fuera",
                flush=True,
            )
        head = self._head_mask_from_faces(image, [chosen])
        if head is None:
            return None
        print(
            f"[ImgEditor] Zona por respuesta del segmentador "
            f"({best_score:.3f} frente a {second_score:.3f})",
            flush=True,
        )
        return {
            "text": translated_prompt or raw_prompt,
            "mask": head,
            "coverage": float((np.asarray(head) > 96).mean()),
        }

    @staticmethod
    def _refine_flat_region_color(
        source: Image.Image,
        edited: Image.Image,
        region_mask: Image.Image,
        foreground_mask: Optional[Image.Image] = None,
        generated_reference: Optional[Image.Image] = None,
        measurement_mask: Optional[Image.Image] = None,
        strict_foreground_mask: Optional[Image.Image] = None,
    ) -> Tuple[Image.Image, bool]:
        """Complete a measured color change over a flat surface, without prompt keywords.

        A segmentation mask often omits thin borders. Only a low-texture source
        region with a coherent generated color shift qualifies; textured or
        structural edits are left untouched. The shift is learned from the
        generated pixels, not from a named color or object class.
        """
        try:
            import cv2

            size = edited.size
            src = np.asarray(source.convert("RGB").resize(size, Image.Resampling.LANCZOS))
            out = np.asarray(edited.convert("RGB"))
            measured = np.asarray(
                (generated_reference or edited).convert("RGB").resize(size, Image.Resampling.LANCZOS)
            )
            mask = np.asarray(region_mask.convert("L").resize(size, Image.Resampling.BILINEAR))
            core = mask >= 180
            coverage = float(core.mean())
            if not 0.005 <= coverage <= 0.60:
                return edited, False

            sample = np.asarray(
                (measurement_mask or region_mask).convert("L").resize(size, Image.Resampling.BILINEAR)
            ) >= 180
            if float(sample.mean()) < 0.005:
                return edited, False

            src_lab = cv2.cvtColor(src, cv2.COLOR_RGB2LAB).astype(np.int16)
            out_lab = cv2.cvtColor(measured, cv2.COLOR_RGB2LAB).astype(np.int16)
            source_median = np.median(src_lab[sample], axis=0)
            source_mad = np.median(np.abs(src_lab[sample] - source_median), axis=0)
            shift = np.median(out_lab[sample] - src_lab[sample], axis=0)
            shift_mad = np.median(np.abs((out_lab[sample] - src_lab[sample]) - shift), axis=0)
            target_median = np.median(out_lab[sample], axis=0)
            target_mad = np.median(np.abs(out_lab[sample] - target_median), axis=0)
            gray_source = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
            edge_strength = cv2.magnitude(
                cv2.Sobel(gray_source, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(gray_source, cv2.CV_32F, 0, 1, ksize=3),
            )
            texture_median = float(np.median(edge_strength[sample]))
            texture_p75 = float(np.percentile(edge_strength[sample], 75))
            # A wall can have a broad lighting gradient while its chroma stays
            # consistent. Allow that luminance spread, but retain the narrow
            # chroma and generated-shift checks that reject textured objects.
            if (source_mad[0] > 25 or max(source_mad[1:]) > 6
                    or texture_median > 6.5 or texture_p75 > 10.5
                    or np.linalg.norm(shift) < 12
                    or shift_mad[0] > 35 or max(shift_mad[1:]) > 12):
                return edited, False

            # Expand only near the measured region and onto source pixels with
            # similar intrinsic surface color. Skin and textured objects do not
            # normally share this narrow Lab chroma band.
            # Small CLIPSeg islands can sit well inside a larger surface. Grow
            # far enough to reach its edges; source color and person protection
            # still decide which pixels may actually change.
            color_median = np.median(src_lab[core], axis=0)
            radius = max(8, int(round(min(size) * 0.22)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
            nearby = cv2.dilate((mask >= 100).astype(np.uint8), kernel) > 0
            color_match = (
                (src_lab[:, :, 0] >= color_median[0] - 70)
                & (np.abs(src_lab[:, :, 1] - color_median[1]) < 10)
                & (np.abs(src_lab[:, :, 2] - color_median[2]) < 18)
            )
            region = nearby & color_match
            protected_nonfringe = None
            if foreground_mask is not None:
                foreground = np.asarray(
                    foreground_mask.convert("L").resize(size, Image.Resampling.BILINEAR)
                )
                # A person mask is a safety guard only when it is separate from
                # the requested region; never block a subject edit with it.
                if float(((foreground >= 96) & core).sum()) < 0.02 * float(core.sum()):
                    clear_background = region & (foreground < 96)
                    # CLIPSeg includes a few pixels of bright wall in the person
                    # silhouette at hair edges. Recolor only those near an already
                    # accepted wall pixel and still close to the wall's source
                    # luminance/chroma. Dark hair and skin stay protected.
                    fringe_radius = max(3, int(round(min(size) * 0.02)))
                    fringe_kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (fringe_radius * 2 + 1, fringe_radius * 2 + 1),
                    )
                    near_clear_background = cv2.dilate(
                        clear_background.astype(np.uint8), fringe_kernel
                    ) > 0
                    bright_fringe = (
                        (src_lab[:, :, 0] >= source_median[0] - 22)
                        & (np.abs(src_lab[:, :, 1] - source_median[1]) <= 6)
                        & (np.abs(src_lab[:, :, 2] - source_median[2]) <= 6)
                    )
                    protected_nonfringe = (foreground >= 96) & ~bright_fringe
                    region = clear_background | (
                        region & (foreground >= 96)
                        & near_clear_background & bright_fringe
                    )
            if float(region.mean()) < coverage * 0.65:
                return edited, False

            transformed_lab = np.clip(src_lab + shift, 0, 255).astype(np.uint8)
            # When the model produced a nearly constant target chroma, carry
            # that chroma across the surface while retaining source lighting.
            # A relative chroma shift turns darker warm areas blue.
            if max(target_mad[1:]) <= 8:
                transformed_lab[:, :, 1] = np.uint8(target_median[1])
                transformed_lab[:, :, 2] = np.uint8(target_median[2])
            transformed = cv2.cvtColor(transformed_lab, cv2.COLOR_LAB2RGB)
            alpha = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 1.5)[:, :, None]
            if protected_nonfringe is not None:
                alpha[protected_nonfringe] = 0.0
            if strict_foreground_mask is not None:
                strict = np.asarray(
                    strict_foreground_mask.convert("L").resize(size, Image.Resampling.BILINEAR)
                ) >= 96
                alpha[strict] = 0.0
            result = (transformed * alpha + out * (1.0 - alpha)).clip(0, 255).astype(np.uint8)
            print(
                f"[ImgEditor] Superficie uniforme refinada: núcleo={coverage:.1%}, "
                f"área final={float(region.mean()):.1%}, cambio Lab={tuple(shift.astype(int))}",
                flush=True,
            )
            return Image.fromarray(result, "RGB"), True
        except Exception as exc:
            print(f"[ImgEditor] Refinado de superficie omitido: {exc}", flush=True)
            return edited, False

    def _release_local_gpu_before_comfy(self):
        """Libera VRAM local (InsightFace/CLIPSeg/rewriter) antes de que ComfyUI use la GPU."""
        try:
            from roop.face_util import release_face_analyser
            release_face_analyser()
        except Exception:
            pass
        try:
            seg = self.clothing_segmenter
            if seg is not None and getattr(seg, "is_loaded", lambda: False)():
                seg.unload()
        except Exception:
            pass
        try:
            # CLIPSeg solo localiza la máscara antes de generar. En una RTX de
            # 8 GB no puede quedarse junto a LongCat: incluso cuando la máscara
            # se rechaza como vacía, su ViT permanecía residente y DynamicVRAM
            # acababa intercambiando pesos en cada paso de ComfyUI.
            masker = self.clip_masker
            if masker is not None and getattr(masker, "_initialized", False):
                masker.unload()
        except Exception:
            pass
        try:
            # v5.1096: el rewriter (llama.cpp, 32 capas en GPU) se quedaba residente
            # (~0.5-1GB VRAM) durante TODA la generación ComfyUI → thrash de offload
            # (2º edit seguido: 158s vs 29s). Se descarga acá; se recarga solo en el
            # próximo edit que lo pida (singleton, pocos segundos).
            import roop.img_editor.prompt_rewriter as _pr_mod
            _rw = getattr(_pr_mod, "_rewriter", None)
            if _rw is not None:
                _rw.unload()
                print("[ImgEditor] Rewriter descargado de VRAM (se recarga bajo demanda)")
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        # v5.1133d — sesiones ONNX cacheadas del proceso principal (ESRGAN/LSDIR
        # del finisher, CodeFormer del top_tier, CodeFormer del manager): ocupan
        # VRAM durante la carga ComfyUI de la siguiente petición (2ª lenta).
        try:
            from roop.img_editor.image_quality_pipeline import get_quality_finisher
            get_quality_finisher().release_gpu_caches()
        except Exception:
            pass
        try:
            import roop.img_editor.top_tier_finish as _ttf
            _cf = getattr(_ttf._preserve_faces, "_cf", None)
            if _cf is not None:
                _ttf._preserve_faces._cf = None
                del _cf
                print("[TopTier] Sesión CodeFormer liberada (se recarga bajo demanda)", flush=True)
        except Exception:
            pass
        try:
            if self._codeformer is not None:
                self._codeformer = None
                print("[ImgEditor] CodeFormer del manager liberado (se recarga bajo demanda)", flush=True)
        except Exception:
            pass
        import gc
        gc.collect()
        try:
            import torch
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            local_reserved = torch.cuda.memory_reserved()
            print(
                f"[ImgEditor] VRAM antes de ComfyUI: libre={free_bytes / 1024**2:.0f}MB "
                f"de {total_bytes / 1024**2:.0f}MB, "
                f"reservada por PyTorch local={local_reserved / 1024**2:.0f}MB",
                flush=True,
            )
        except Exception:
            pass
        print("[ImgEditor] VRAM local liberada — ComfyUI puede usar la GPU")

    def _union_grayscale_masks(self, *masks: Image.Image) -> Optional[Image.Image]:
        arrays = []
        w, h = None, None
        for mask in masks:
            if mask is None:
                continue
            arr = np.array(mask.convert("L"))
            if w is None:
                w, h = mask.size
            elif arr.shape[:2] != (h, w):
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
            arrays.append(arr)
        if not arrays:
            return None
        union = arrays[0].copy()
        for arr in arrays[1:]:
            union = np.maximum(union, arr)
        return Image.fromarray(union).convert("L") if union.max() > 0 else None

    def _build_clipseg_person_mask(
        self,
        image: Image.Image,
        dilation: int = 20,
    ) -> Optional[Image.Image]:
        """CLIPSeg segmentación de cuerpo completo. Múltiples thresholds, sin filtro restrictivo."""
        try:
            import torch
            seg = self._get_clothing_segmenter()
            if not seg or not seg.is_available():
                return None
            prev_dev = getattr(seg, "device", None)
            try:
                seg.device = torch.device("cpu")
                if seg.model is not None:
                    seg.model.to(seg.device)
            except Exception:
                pass

            prompts = [
                "person", "human body", "full body person", "people",
                "man", "woman", "body", "torso",
            ]

            best_mask = None
            best_cov = 0.0
            for th in (0.35, 0.40, 0.50, 0.30, 0.60):
                mask_img, _ = seg.segment_with_prompt(
                    image, prompts,
                    threshold=th, dilation=dilation,
                )
                if mask_img is None or np.array(mask_img).max() == 0:
                    continue
                cov = self._mask_coverage(mask_img)
                if cov > 0.05 and cov > best_cov:
                    best_mask = mask_img
                    best_cov = cov

            if prev_dev is not None:
                try:
                    seg.device = prev_dev
                    if seg.model is not None:
                        seg.model.to(prev_dev)
                except Exception:
                    pass

            if best_mask is not None:
                print(f"[ImgEditor] CLIPSeg persona: cov={best_cov*100:.1f}%")
                return best_mask
        except Exception as e:
            print(f"[ImgEditor] CLIPSeg persona falló: {e}")
        return None

    def _build_object_removal_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """Máscara genérica de objeto para inpaint remove (perro, vaso, etc.).
        CLIPSeg con prompts de objeto — sin vocabulario del usuario, solo
        categorías visuales. Devuelve None si no hay objeto localizado."""
        try:
            seg = self._get_clothing_segmenter()
            if not seg or not seg.is_available():
                return None
            prompts = ["dog", "animal", "pet", "object", "item"]
            for th in (0.40, 0.30, 0.50):
                mask_img, _ = seg.segment_with_prompt(
                    image, prompts, threshold=th, dilation=8,
                )
                if mask_img is not None:
                    cov = self._mask_coverage(mask_img)
                    if cov > 0.005:
                        print(f"[ImgEditor] Máscara objeto: cov={cov*100:.1f}% (th={th})")
                        return mask_img
        except Exception as e:
            print(f"[ImgEditor] Máscara objeto falló: {e}")
        return None

    def _build_person_removal_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """Máscara cuerpo/persona para inpaint remove. CLIPSeg + caras + dilatación agresiva + fallback full-frame."""
        faces = []
        face_count = 0
        try:
            from roop.face_util import get_face_analyser_cpu, preprocess_image_for_detection
            img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            analyser = get_face_analyser_cpu()
            faces = analyser.get(img_bgr) or []
            if len(faces) < 2:
                enhanced = preprocess_image_for_detection(img_bgr)
                extra = analyser.get(enhanced) or []
                seen = {tuple(int(v) for v in f.bbox) for f in faces}
                for f in extra:
                    key = tuple(int(v) for v in f.bbox)
                    if key not in seen:
                        faces.append(f)
                        seen.add(key)
            face_count = len(faces)
        except Exception as e:
            print(f"[ImgEditor] InsightFace persona falló: {e}")

        clipseg_mask = self._build_clipseg_person_mask(image)
        face_mask = None
        if faces:
            face_mask = self._mask_from_face_bboxes(
                image, faces, width_mult=4.0, height_mult=6.0, head_pad=0.5,
            )

        merged = None
        parts = []
        if clipseg_mask is not None:
            merged = clipseg_mask
            parts.append("CLIPSeg")
            if face_mask is not None:
                clip_arr = np.array(clipseg_mask.convert("L"))
                face_arr = np.array(face_mask.convert("L"))
                supplement = np.where((face_arr > 128) & (clip_arr <= 128), 255, 0).astype(np.uint8)
                if supplement.max() > 0:
                    merged = Image.fromarray(np.maximum(clip_arr, supplement)).convert("L")
                    parts.append(f"InsightFace({face_count})")
        elif face_mask is not None:
            merged = face_mask
            parts.append(f"InsightFace({face_count})")

        if merged is not None:
            arr = np.array(merged.convert("L"))
            h, w = arr.shape

            close_k = max(7, min(h, w) // 40)
            arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

            dilate_k = max(15, min(h, w) // 20)
            arr = cv2.dilate(arr, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)

            cov = (arr > 128).sum() / float(arr.size)
            if cov < 0.30:
                arr = cv2.dilate(arr, np.ones((31, 31), np.uint8), iterations=1)
                cov = (arr > 128).sum() / float(arr.size)

            merged = Image.fromarray(arr).convert("L")
            print(
                f"[ImgEditor] Máscara persona vía {' + '.join(parts)} "
                f"(cov={cov*100:.1f}%)"
            )
            return merged

        if face_count > 0:
            h, w = np.array(image.convert("L")).shape
            full = np.ones((h, w), dtype=np.uint8) * 255
            margin_x, margin_y = w // 20, h // 20
            full[:margin_y, :] = 0
            full[-margin_y:, :] = 0
            full[:, :margin_x] = 0
            full[:, -margin_x:] = 0
            print(f"[ImgEditor] Fallback full-frame mask ({face_count} caras detectadas)")
            return Image.fromarray(full).convert("L")

        return None

    def _build_person_addition_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Máscara INVERSA para ADD person.
        Protege (negro) las áreas con personas existentes (caras + CLIPSeg cuerpo).
        Deja blanco (editables) las áreas libres donde se puede añadir gente nueva.
        Usa CLIPSeg persona + InsightFace + dilatación generosa + feathering suave.
        """
        faces = []
        face_count = 0
        try:
            from roop.face_util import get_face_analyser_cpu, preprocess_image_for_detection
            img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            analyser = get_face_analyser_cpu()
            faces = analyser.get(img_bgr) or []
            if len(faces) < 2:
                enhanced = preprocess_image_for_detection(img_bgr)
                extra = analyser.get(enhanced) or []
                seen = {tuple(int(v) for v in f.bbox) for f in faces}
                for f in extra:
                    key = tuple(int(v) for v in f.bbox)
                    if key not in seen:
                        faces.append(f)
                        seen.add(key)
            face_count = len(faces)
        except Exception as e:
            print(f"[ImgEditor] InsightFace para add falló: {e}")

        clipseg_mask = self._build_clipseg_person_mask(image, dilation=15)
        face_mask = None
        if faces:
            face_mask = self._mask_from_face_bboxes(
                image, faces, width_mult=4.0, height_mult=5.0, head_pad=0.4,
            )

        # Unir CLIPSeg + caras en una sola máscara de protección (negro = proteger)
        protection = None
        parts = []

        if clipseg_mask is not None and face_mask is not None:
            clip_arr = np.array(clipseg_mask.convert("L"))
            face_arr = np.array(face_mask.convert("L"))
            # Unión de protección
            protection = np.maximum(clip_arr, face_arr)
            parts.append("CLIPSeg+InsightFace")
        elif clipseg_mask is not None:
            protection = np.array(clipseg_mask.convert("L"))
            parts.append("CLIPSeg")
        elif face_mask is not None:
            protection = np.array(face_mask.convert("L"))
            parts.append(f"InsightFace({face_count})")

        if protection is not None:
            h, w = protection.shape[:2]

            # Dilatar protección para dejar espacio alrededor
            dilate_k = max(25, min(h, w) // 15)
            protection = cv2.dilate(protection, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)

            # Feathering en bordes para transición suave
            blur_k = max(15, min(h, w) // 30)
            if blur_k % 2 == 0:
                blur_k += 1
            protection = cv2.GaussianBlur(protection, (blur_k, blur_k), 0)

            # INVERTIR: lo que estaba blanco (persona) ahora es negro (protegido)
            # y lo que estaba negro (fondo) ahora es blanco (área editable)
            invert = 255 - protection

            # Margen mínimo editable alrededor de bordes de imagen
            invert[:max(3, h//40), :] = 0
            invert[-max(3, h//40):, :] = 0
            invert[:, :max(3, w//40)] = 0
            invert[:, -max(3, w//40):] = 0

            cov = (invert > 128).sum() / float(invert.size)
            if cov < 0.15:
                print(f"[ImgEditor] ADD mask cov={cov*100:.1f}% — muy poca área libre, reduciendo protección")
                # Reducir dilatación para dejar más área
                protection = protection.astype(np.float32)
                protection = cv2.GaussianBlur(protection, (blur_k, blur_k), sigmaX=blur_k*0.3)
                invert = (255 - protection).astype(np.uint8)
                cov = (invert > 128).sum() / float(invert.size)

            if face_count > 0:
                print(
                    f"[ImgEditor] Máscara ADD vía {' + '.join(parts)} "
                    f"(area_libre={cov*100:.1f}%, caras={face_count})"
                )
            else:
                print(
                    f"[ImgEditor] Máscara ADD vía {' + '.join(parts)} "
                    f"(area_libre={cov*100:.1f}%)"
                )
            return Image.fromarray(invert).convert("L")

        # Sin detección: máscara global (todo editable)
        print("[ImgEditor] ADD mask: sin detección → todo editable (global)")
        return None

    def _get_codeformer(self):
        if self._codeformer is None:
            from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
            model_path = resolve_relative_path('../models/CodeFormer/CodeFormerv0.1.onnx')
            if os.path.exists(model_path):
                try:
                    cf = Enhance_CodeFormer()
                    opts = {"devicename": "cuda"}
                    cf.Initialize(opts)
                    self._codeformer = cf
                    print("[ImgEditor] CodeFormer listo para enhancement facial")
                except Exception as e:
                    print(f"[ImgEditor] CodeFormer no disponible: {e}")
            else:
                print(f"[ImgEditor] CodeFormer model not found at {model_path}")
        return self._codeformer

    def _enhance_faces(
        self,
        image: Image.Image,
        progress_callback=None,
        quality_mode: bool = False,
        face_enhance_cap: float = None,
    ) -> Image.Image:
        """Mejora rostros en la imagen usando CodeFormer. Sin hardcoding."""
        cf = self._get_codeformer()
        if cf is None:
            return image

        try:
            if progress_callback:
                progress_callback({"phase": "Mejorando caras", "progress": 0.94, "detail": "CodeFormer"})
            import roop.face_util as face_util
            img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            faces_data = face_util.extract_face_images(img_np, target_face_detection=True)
            if not faces_data:
                return image

            h, w = img_np.shape[:2]
            result = img_np.copy()

            for face_obj, _crop in faces_data:
                bbox = face_obj.bbox
                x1, y1, x2, y2 = [int(v) for v in bbox]
                margin = int(max(x2 - x1, y2 - y1) * 0.3)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)

                face_crop = img_np[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                enhanced_bgr, _ = cf.Run(None, None, face_crop)
                if enhanced_bgr.shape[:2] != face_crop.shape[:2]:
                    enhanced_bgr = cv2.resize(
                        enhanced_bgr, (face_crop.shape[1], face_crop.shape[0]),
                        cv2.INTER_LANCZOS4
                    )

                det_score = float(getattr(face_obj, "det_score", 0.9) or 0.9)
                cap = float(face_enhance_cap) if face_enhance_cap is not None else (0.32 if quality_mode else 0.30)
                if quality_mode:
                    blend = float(np.clip(det_score * 0.20, 0.08, cap))
                else:
                    blend = float(np.clip(det_score * 0.15, 0.05, min(cap, 0.30)))

                fh, fw = face_crop.shape[:2]
                cx, cy = fw // 2, fh // 2
                axes = (max(8, int(fw * 0.46)), max(8, int(fh * 0.52)))
                mask = np.zeros((fh, fw), dtype=np.float32)
                cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), max(6.0, min(fh, fw) * 0.08))
                mask *= float(blend)
                mask3 = mask[..., None]
                roi = result[y1:y2, x1:x2].astype(np.float32)
                result[y1:y2, x1:x2] = (
                    roi * (1.0 - mask3) + enhanced_bgr.astype(np.float32) * mask3
                ).astype(np.uint8)

            print(f"[ImgEditor] Enhancement aplicado a {len(faces_data)} rostro(s)")
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        except Exception as e:
            print(f"[ImgEditor] Face enhancement error: {e}")
            return image

    def generate_intelligent(
        self,
        image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        use_rewriter: bool = False,
        use_semantic: bool = True,
        ref_metadata: dict = None,
        engine: str = "imagine",
        mask_image: Optional[Image.Image] = None,
        mask_mode: str = "global",
        mask_prompt: str = "",
        enhance_faces: bool = True,
        lora_name: str = None,
        lora_strength: float = 1.0,
        denoise: float = None,
        progress_callback=None,
        auto_upscale: bool = False,
        quality_mode: bool = False,
        enhance_tier: str = "hd",
        quality_enhance_style: str = "auto",
        quality_use_generative: Optional[bool] = None,
        quality_preserve_faces: bool = True,
        quality_steps: Optional[dict] = None,
        face_preserve: bool = True,
        cancel_check=None,
        _compound_depth: int = 0,
        _compound_region_hint: str = "",
        _source_clause: str = "",
    ) -> Tuple[Optional[Image.Image], str, Optional[Image.Image]]:

        self._last_instruction_compliance = None
        if cancel_check and cancel_check():
            return None, "Cancelado", None

        if isinstance(image, Image.Image):
            img = image.copy().convert("RGB")
        else:
            img = Image.open(image.name).copy().convert("RGB")

        # Calidad sin prompt usa restauración/upscale local y no necesita ComfyUI.
        pure_quality_request = bool(quality_mode and not (prompt or "").strip())
        if pure_quality_request:
            # This request has no semantic edit to plan, translate or verify.
            # Running FaceAnalysis and the full edit router first loaded five
            # models and delayed even an already sharp, full-size photo.
            from roop.img_editor.top_tier_finish import run_top_tier_quality
            result, note, quality_analysis = run_top_tier_quality(
                img,
                enhance_tier=enhance_tier,
                use_generative=False,
                preserve_faces=bool(quality_preserve_faces),
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                quality_steps=quality_steps,
            )
            self._last_quality_analysis = quality_analysis
            if cancel_check and cancel_check():
                return None, "Cancelado", None
            return result, f"Mejora TOP tier={(enhance_tier or 'hd').lower()} | {note}", None

        # Una edición nueva requiere que el backend de generación esté vivo.
        # Si fue detenido entre el arranque de la UI y el click Editar, reiniciar
        # aquí evita analizar/maskar durante segundos y después reintentar contra
        # un puerto muerto sin explicar la causa.
        if not pure_quality_request:
            try:
                from ui.tabs.comfy_launcher import is_comfyui_running, start as start_comfyui
                if not is_comfyui_running():
                    print("[ImgEditor] ComfyUI no estaba activo; arrancando backend para esta edición...", flush=True)
                    _comfy_ok, _comfy_msg, _ = start_comfyui(directly_run=True)
                    if not _comfy_ok:
                        return None, f"ComfyUI no disponible: {_comfy_msg}", None
                    print("[ImgEditor] ComfyUI listo para la edición", flush=True)
            except Exception as _comfy_start_error:
                return None, f"ComfyUI no disponible: {_comfy_start_error}", None

        detected_age = None
        shot_type = "unknown"
        source_faces = []
        # Detected list, no credibility floor: only the subject-mask gate
        # uses it (the credibility floor stays for heat/head anchors).
        detected_faces = []
        # The camera route does not use age, composition or face embeddings.
        # FaceAnalysis reserves GPU memory before Qwen loads on 8 GB cards.
        skip_face_preflight = bool(
            _compound_depth == 0 and mask_image is None
            and engine in ("longcat", "imagine", "qwen_camera")
            and self._explicit_viewpoint_request(prompt, prompt)
        )
        if skip_face_preflight:
            print("[ImgEditor] FaceAnalysis omitido para cambio de cámara", flush=True)
        try:
            if skip_face_preflight:
                raise StopIteration
            import cv2
            from roop.face_util import get_face_analyser, detect_faces_robust
            img_arr = np.array(img)
            img_h, img_w = img_arr.shape[:2]
            if len(img_arr.shape) == 3 and img_arr.shape[2] == 3:
                img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                analyser = get_face_analyser()
                if analyser is not None:
                    faces = detect_faces_robust(img_bgr, analyser, 0.20)
                    if faces:
                        detected_faces = list(faces)
                        source_faces = self._credible_faces(faces)
                        ages = [getattr(f, 'age', None) for f in faces if getattr(f, 'age', None) is not None]
                        if ages:
                            detected_age = int(sum(ages) / len(ages))
                            print(f"[ImgEditor] Edad detectada: {detected_age} años ({len(ages)} cara(s))")
                        best = max(faces, key=lambda f: float(getattr(f, 'det_score', 0.0) or 0.0))
                        bx1, by1, bx2, by2 = best.bbox
                        face_w = bx2 - bx1
                        face_h = by2 - by1
                        face_ratio = (face_w * face_h) / (img_w * img_h)
                        face_top_ratio = by1 / img_h
                        if face_ratio > 0.25:
                            shot_type = "closeup"
                        elif face_ratio > 0.06 and face_top_ratio < 0.35:
                            shot_type = "portrait"
                        elif face_ratio < 0.02 or face_top_ratio > 0.5:
                            shot_type = "fullbody"
                        else:
                            shot_type = "midshot"
                        print(f"[ImgEditor] Composición: {shot_type} (face_ratio={face_ratio:.3f}, top={face_top_ratio:.2f})")
        except StopIteration:
            pass
        except Exception as e:
            print(f"[ImgEditor] Análisis de imagen no disponible: {e}")

        from roop.img_editor.prompt_translator import translate_prompt
        prompt_raw = (prompt or "").strip()
        # NOTA: se probó corrector LLM local (fix_typos en prompt_rewriter) y
        # el 0.5B no da la talla ('DD RODILLAS'→'DROIDS', trunca frases). El
        # método queda para modelos 1B+. La tolerancia a typos se logra abajo
        # con matching por subconjunto de tokens (listas existentes) + aviso.
        quality_enhance = bool(quality_mode)
        quality_only = quality_enhance and not prompt_raw
        quality_hybrid = quality_enhance and bool(prompt_raw)
        quality_style = (quality_enhance_style or "auto").lower()
        if quality_style not in ("auto", "solo_mejora", "hibrido"):
            quality_style = "auto"
        enhance_tier = (enhance_tier or "hd").lower()
        if enhance_tier not in ("hd", "4k", "8k"):
            enhance_tier = "hd"
        if quality_only:
            prompt = ""
            use_rewriter = False
            print(f"[ImgEditor] Modo mejora imagen (UI tier={enhance_tier}) — sin instrucción de prompt")
        elif quality_hybrid:
            prompt = translate_prompt(prompt_raw)
            use_rewriter = False
            short = prompt_raw if len(prompt_raw) <= 64 else prompt_raw[:64] + "…"
            print(
                f"[ImgEditor] Modo híbrido tier={enhance_tier} — "
                f"edit: «{short}» + mejora TOP después"
            )
        elif _compound_depth > 0:
            # The parent already translated the sentence and split it.
            # The original clause stays available for concept scores: the
            # translator can drop the token that made a color change measurable.
            prompt = prompt_raw
        else:
            prompt = translate_prompt(prompt_raw)
        source_clause = (_source_clause or "").strip()
        score_raw = source_clause or prompt_raw
        # Entrada con basura sin traducir (typos/inventos): el rewriter 0.5B
        # alucina con ella (log 13:33: 'naked and barefoot' desde 'SUJERA/
        # CADSERAS/LZA') y su mag inflada NO debe secuestrar a body.
        try:
            from roop.img_editor.prompt_translator import _is_english, translation_suspect_tokens
            translation_suspect = bool(translation_suspect_tokens(prompt_raw, prompt))
            translation_unavailable = bool(
                prompt_raw and not _is_english(prompt_raw) and not _is_english(prompt)
            )
        except Exception:
            translation_suspect = False
            translation_unavailable = False
        if translation_unavailable:
            # El rewriter pequeño es un planificador EN; si el traductor no
            # produjo inglés, dejarlo reformular el texto crudo puede sustituir
            # el pedido por una invención. Se conserva la orden original y el
            # modelo recibe el texto sin una reinterpretación generativa.
            use_rewriter = False
            print(
                "[ImgEditor] Traducción no confirmada: rewriter omitido para preservar la instrucción original",
                flush=True,
            )
        prompt = self._strip_instruction_prefix(prompt)
        negative_prompt = (negative_prompt or "").strip()

        # A requested camera change is not reliably solved by the current
        # single-image models. Detect it early so the result can say so.
        explicit_viewpoint_request = self._explicit_viewpoint_request(prompt_raw, prompt)
        if explicit_viewpoint_request:
            print("[ImgEditor] Cambio de cámara solicitado; cumplimiento visual no verificado", flush=True)
        if engine == "qwen_camera" and (
                not explicit_viewpoint_request or mask_image is not None
                or quality_style == "solo_mejora"
                or (quality_hybrid and quality_use_generative is True)
                or get_vram_gb() <= 0):
            print("[ImgEditor] Qwen Cámara no aplica a esta solicitud; usando LongCat", flush=True)
            engine = "longcat"

        # A camera move needs a model trained for new viewpoints. Qwen's
        # multiple-angle LoRA sees the original photo in its first pass. If
        # there are independent visual edits, LongCat applies them to the new
        # view in a second pass. If Qwen does not finish, LongCat is not
        # started: it keeps the original angle and queues behind a busy GPU.
        if (explicit_viewpoint_request and prompt and _compound_depth == 0
                and mask_image is None and quality_style != "solo_mejora"
                and not (quality_hybrid and quality_use_generative is True)
                and engine in ("longcat", "imagine", "qwen_camera")
                and get_vram_gb() > 0):
            camera_message = "Qwen Camera no terminó"
            try:
                from roop.img_editor.qwen_camera_comfy_client import (
                    QwenCameraComfyClient, camera_pose, remaining_edit,
                )
                self._release_local_gpu_before_comfy()
                pose = camera_pose(prompt_raw, prompt)
                camera_result, camera_message = QwenCameraComfyClient().generate(
                    img, pose, seed=seed, progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
                if camera_result is not None:
                    output = camera_result.image.convert("RGB")
                    extra_edit = remaining_edit(prompt_raw, prompt)
                    route_note = f"Qwen Camera {camera_message} ({pose})"
                    if extra_edit:
                        if cancel_check and cancel_check():
                            return None, "Cancelado", None
                        from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                        edit_client = get_flux_edit_comfy_client()
                        loaded, load_msg = edit_client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
                        if loaded:
                            follow_prompt = (
                                f"Edit this exact photo. Apply these remaining changes: {extra_edit}. "
                                f"The camera is already at {pose.removeprefix('<sks> ')}; "
                                "preserve that camera position, all people and their complete anatomy."
                            )
                            if negative_prompt:
                                follow_prompt += f" Avoid: {negative_prompt}."
                            follow_result, follow_message = edit_client.generate(
                                image=output, prompt=follow_prompt,
                                negative_prompt=negative_prompt,
                                num_inference_steps=12, guidance_scale=1.0,
                                denoise=1.0, seed=seed, no_kontext=True,
                                lora_name=lora_name, lora_strength=lora_strength,
                                progress_callback=progress_callback,
                                cancel_check=cancel_check,
                            )
                            if follow_result is not None:
                                output = follow_result.image.convert("RGB")
                                route_note += " + LongCat Turbo para otras ediciones"
                            else:
                                route_note += f" | segunda edición falló: {follow_message}"
                        else:
                            route_note += f" | segunda edición no disponible: {load_msg}"
                    if quality_hybrid:
                        from roop.img_editor.top_tier_finish import run_quality_enhance_after_edit
                        output, finish_note, quality_analysis = run_quality_enhance_after_edit(
                            output, img, enhance_tier=enhance_tier,
                            use_generative=False,
                            preserve_faces=bool(quality_preserve_faces),
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            quality_steps=quality_steps,
                        )
                        self._last_quality_analysis = quality_analysis
                        route_note += f" + {finish_note}"
                    elif output.size != img.size:
                        output = output.resize(img.size, Image.Resampling.LANCZOS)
                        route_note += " | tamaño original por interpolación"
                    if cancel_check and cancel_check():
                        return None, "Cancelado", None
                    route_note += " | resultado visual no verificado"
                    print(f"[ImgEditor] {route_note}", flush=True)
                    return output, route_note, None
                print(f"[ImgEditor] Qwen Camera no disponible: {camera_message}", flush=True)
            except Exception as camera_error:
                camera_message = f"Qwen Camera falló: {camera_error}"
                print(f"[ImgEditor] {camera_message}", flush=True)
            if cancel_check and cancel_check():
                return None, "Cancelado", None
            print("[ImgEditor] POV no aplicado; LongCat no se encola", flush=True)
            return img, (
                f"POV no aplicado: {camera_message}. "
                "Se conserva el ángulo original."
            ), None

        # For ordinary LongCat edits on this GPU, send the full instruction
        # and image to one strong visual pass. The small text planner
        # can otherwise invent regions or interpret subject descriptions as
        # commands. A Quality edit with a prompt uses the same Turbo edit and
        # then the classical quality finish, avoiding a second LongCat pass.
        if (prompt and mask_image is None and not quality_only
                and quality_style != "solo_mejora"
                and not (quality_hybrid and quality_use_generative is True)
                and engine in ("longcat", "imagine")
                and 0 < get_vram_gb() <= 10):
            # A compound prompt is ONE generation: the full instruction goes
            # to Turbo once, and only the zones each clause measured are
            # pasted back. Per-clause passes were slow (~1 min each), left
            # grey ghosts between overlapping passes, and dropped the clause
            # that ran last. The prompt is still split, to measure each
            # clause's zone on this photo; never to generate once per clause.
            # v5.1211: la ruta compuesta medida TAMBIÉN en modo calidad. Antes
            # `not quality_hybrid` la desactivaba: la cláusula de región que el
            # paso de sujeto descartaba ("'the walls' cubre 39.4% pero no
            # coincide con una cara; se descarta") desaparecía de la
            # instrucción y el color de pared nunca se pedía en modo calidad
            # (edit_1792086034: paredes intactas con el mismo prompt que en
            # edit_1792086138 sí las pintó). El acabado TOP se aplica después
            # del pegado, igual que en la ruta única (run_quality_enhance_after_edit).
            if (_compound_depth == 0
                    and self._split_compound_instruction(prompt)):
                clauses = self._split_compound_instruction(prompt)
                sources = self._split_compound_instruction(prompt_raw)
                if len(sources) != len(clauses):
                    sources = [""] * len(clauses)
                zones = []
                unmeasured = []
                # The full-prompt score dilutes one clause ("bald and naked"
                # reads as weak body). The wrapper flags come from the
                # strongest clause, the same way the single-clause path
                # composes its own instruction.
                compound_pose = False
                compound_body = False
                compound_local = False
                # v5.1215: «que esté en una playa» no admite ancla: la escena
                # se regenera aunque las caras cambien (prioridad del usuario:
                # que quede bien). Misma concesión que la postura.
                compound_scenario = False
                compound_direction = "change"
                compound_intensity = 0.0
                for clause, source in zip(clauses, sources):
                    if cancel_check and cancel_check():
                        return None, "Cancelado", None
                    clause_facts = self._prompt_authority_facts(
                        source or clause, clause,
                    )
                    if clause_facts.get("pose"):
                        compound_pose = True
                    if clause_facts.get("scenario"):
                        compound_scenario = True
                    if clause_facts.get("body") and (
                            float(clause_facts.get("body_intensity", 0.0))
                            > compound_intensity):
                        compound_body = True
                        compound_direction = str(clause_facts.get("body_direction", "change"))
                        compound_intensity = float(clause_facts.get("body_intensity", 0.0))
                    if (clause_facts.get("local_appearance")
                            or clause_facts.get("local_region")
                            or clause_facts.get("region_ambiguous")):
                        compound_local = True
                    zone = self._compound_clause_zone(
                        img, clause, source, clause_facts, source_faces,
                        detected_faces=detected_faces,
                    )
                    if zone is not None:
                        zones.append(zone)
                    else:
                        unmeasured.append(clause)
                print(
                    "[ImgEditor] Compuesto: un solo pase Turbo con los "
                    f"{len(clauses)} cambios; zonas medidas={len(zones)}"
                    + (f", sin zona={len(unmeasured)}" if unmeasured else ""),
                    flush=True,
                )
                self._release_local_gpu_before_comfy()
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                fast_client = get_flux_edit_comfy_client()
                loaded, load_msg = fast_client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
                fast_result = None
                fast_msg = load_msg
                # The union of the measured zones is both the sampler hole
                # and the paste. Measured before generating, it drives the
                # local window the same way the single-instruction path
                # does. A full-photo pass without an anchor regenerated the
                # scene from noise and painted a white blob
                # (edit_1791840439; edit_1791843357).
                union = None
                if zones:
                    union = zones[0]
                    for extra in zones[1:]:
                        left = np.asarray(union.convert("L"), dtype=np.uint8)
                        right = np.asarray(extra.convert("L"), dtype=np.uint8)
                        if left.shape != right.shape:
                            right = cv2.resize(
                                right, (left.shape[1], left.shape[0]),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        union = Image.fromarray(np.maximum(left, right), "L")
                generation_image = img
                generation_mask = None
                local_box = None
                # v5.1215: con cláusula de escenario no hay ventana local ni
                # ancla: la escena completa se regenera (misma concesión que
                # la postura). El usuario acepta que las caras cambien.
                compound_unanchored = bool(compound_pose or compound_scenario)
                if union is not None and not compound_unanchored:
                    generation_image, generation_mask, local_box = (
                        self._appearance_work_region(img, union)
                    )
                    if local_box is None:
                        local_box = (0, 0, img.size[0], img.size[1])
                if loaded:
                    # Same client flags as the single-instruction path: the
                    # measured zone is the sampler hole and the paste, and
                    # the photo stays anchored so the room survives. A pose
                    # or a scenario still redraws the frame.
                    compound_authority = bool(
                        compound_pose or compound_body or compound_local
                        or compound_scenario
                    )
                    compound_preserve = bool(
                        not compound_unanchored
                        and (compound_body or compound_local)
                    )
                    # The full sentence dilutes a clause's exposure score
                    # ("bald and naked" reads as weak body, 0.23 < 0.50,
                    # and the nude suffix was dropped). The change list
                    # keeps every clause, but each one is scored alone,
                    # the same as the single-clause path.
                    fast_prompt = self._compose_generation_prompt(
                        self._compound_change_list(prompt),
                        engine="longcat", magnitude=0.85,
                        pose_change=compound_unanchored,
                        body_transform=compound_body,
                        body_direction=compound_direction,
                        body_intensity=compound_intensity,
                    )
                    if negative_prompt:
                        fast_prompt += f" Avoid: {negative_prompt}."
                    print(
                        "[ImgEditor] Un pase, 12 pasos, fuerza 1.00, "
                        f"{len(clauses)} cambios juntos"
                        + (
                            "; escenario nuevo, la escena completa sale del modelo"
                            if compound_scenario else ""
                        ),
                        flush=True,
                    )
                    fast_result, fast_msg = fast_client.generate(
                        image=generation_image, prompt=fast_prompt,
                        negative_prompt=negative_prompt,
                        guidance_scale=1.0, seed=seed,
                        prompt_authority=compound_authority,
                        preserve_frame=compound_preserve,
                        mask_image=generation_mask,
                        keep_source_pixels=generation_mask is not None,
                        lora_name=lora_name, lora_strength=lora_strength,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                        num_inference_steps=12, denoise=1.0,
                        no_kontext=compound_unanchored,
                    )
                if fast_result is not None:
                    generated = fast_result.image.convert("RGB")
                    if union is not None and not compound_unanchored:
                        dropped_face = self._degenerate_faces_mask(
                            img, generated, source_faces,
                        )
                        if dropped_face is not None:
                            union = Image.fromarray(
                                np.minimum(
                                    np.asarray(union),
                                    255 - np.asarray(dropped_face),
                                ).astype(np.uint8),
                                "L",
                            )
                        output = self._integrate_local_generation(
                            img, generated, union, local_box,
                            keep_measured_mask=True,
                        )
                        # v5.1218: el compuesto también pega a escala ×N
                        # sobre foto grande (log 01:19: 6390x4263 → lienzo
                        # 1248x832, paste x5.12 → la zona medida y su entorno
                        # quedaban suaves, «pintados»). El refino focal
                        # regenera la zona ya pegada en su zoom nativo.
                        if local_box is not None:
                            refined = self._second_focused_pass(
                                fast_client, output, img, union, local_box,
                                generated.size, fast_prompt, negative_prompt,
                                seed, compound_authority, compound_preserve,
                                None, lora_name, lora_strength,
                                {"num_inference_steps": 12, "denoise": 1.0},
                                progress_callback, cancel_check,
                            )
                            if refined is not None:
                                output = refined
                                fast_msg = (
                                    f"{fast_msg}; refino focal de la zona medida"
                                )
                        note = (
                            f"un solo pase Turbo ({len(clauses)} cambios); "
                            f"{len(zones)} zonas medidas pegadas, "
                            "el resto de la foto se conserva"
                        )
                    elif compound_unanchored:
                        output = generated
                        if compound_scenario:
                            note = (
                                f"un solo pase Turbo ({len(clauses)} cambios); "
                                "escenario nuevo, la escena completa sale del modelo"
                            )
                        else:
                            note = (
                                f"un solo pase Turbo ({len(clauses)} cambios); "
                                "postura nueva, la geometría sale del modelo"
                            )
                    else:
                        output = generated
                        note = (
                            f"un solo pase Turbo ({len(clauses)} cambios); "
                            "sin zona medida, la foto entera sale del modelo"
                        )
                    if output.size != img.size:
                        output = output.resize(img.size, Image.Resampling.LANCZOS)
                    if unmeasured:
                        note += " | sin zona medida: " + ", ".join(unmeasured)
                    if quality_hybrid:
                        from roop.img_editor.top_tier_finish import run_quality_enhance_after_edit
                        output, finish_note, quality_analysis = run_quality_enhance_after_edit(
                            output, img, enhance_tier=enhance_tier,
                            use_generative=False,
                            preserve_faces=bool(quality_preserve_faces),
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            quality_steps=quality_steps,
                        )
                        self._last_quality_analysis = quality_analysis
                        if cancel_check and cancel_check():
                            return None, "Cancelado", None
                        note += f" | Turbo + {finish_note}"
                        print(
                            "[ImgEditor] Calidad con prompt: Turbo (compuesto) "
                            "+ acabado clásico",
                            flush=True,
                        )
                    note += " | cumplimiento visual no verificado"
                    print(f"[ImgEditor] {note}", flush=True)
                    return output, f"{fast_msg}; {note}", None
                if cancel_check and cancel_check():
                    return None, "Cancelado", None
                print(
                    f"[ImgEditor] Pase único de {len(clauses)} cambios no generó "
                    f"({fast_msg}); se intenta la instrucción completa",
                    flush=True,
                )
            try:
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                authority_facts = self._prompt_authority_facts(score_raw, prompt)
                appearance_edit = bool(authority_facts.get("local_appearance"))
                ambiguous_region = bool(authority_facts.get("region_ambiguous")) and not quality_hybrid
                region_edit = bool(
                    authority_facts.get("local_region") or ambiguous_region
                ) and not quality_hybrid
                recolor_edit = region_edit and bool(authority_facts.get("surface_recolor"))
                # v5.1215: «que esté en una playa» pide escena nueva; el
                # usuario acepta que la cara original cambie si así queda
                # bien. Como la postura, manda la regeneración completa: ni
                # ancla ni ventana local.
                scenario_edit = bool(
                    authority_facts.get("scenario")
                    and not authority_facts["pose"]
                    and not authority_facts["body"]
                )
                prompt_authority = bool(
                    authority_facts["authority"] or appearance_edit or region_edit
                    or scenario_edit
                )
                # A change that does not ask for a new pose keeps every pixel
                # outside the measured region. A camera move is handled before
                # this path and still sees the whole photo.
                preserve_frame = bool(
                    (authority_facts["body"] and not authority_facts["pose"])
                    or appearance_edit
                    or region_edit
                )
                if scenario_edit:
                    preserve_frame = False
                subject_mask = None
                paste_limit = None
                subject_passes = []
                edit_text = prompt
                if ambiguous_region:
                    # Shared concept terms can leave region/body tied. Ground
                    # the original instruction before allowing any repaint;
                    # the translation must not turn its object into another.
                    edit_text = (score_raw or prompt).strip()
                    # v5.1209: medir PRIMERO, igual que la ruta compuesta
                    # (_compound_clause_zone). El planner local en español
                    # propone 'cuadros' y CLIPSeg no lee español (0,00 %); la
                    # query EN de la misma frase medía 11,4 % y no se
                    # preguntaba, así que "quita los cuadros" abortaba sin
                    # generar (foto intacta, edición del 15/10 a las 16:51).
                    subject_passes = self._measured_region_passes(
                        img, score_raw, prompt,
                        surface_recolor=False,
                        faces=source_faces,
                    )
                    if not subject_passes:
                        subject_passes = self._proposed_region_passes(
                            img, edit_text, self._get_clipseg_masker(), source_faces,
                        )
                        if subject_passes:
                            translated_instruction = self._ambiguous_region_translation(edit_text)
                            if not translated_instruction:
                                return img, "Traducción local no disponible; foto conservada | cumplimiento visual no verificado", None
                            subject_passes = [
                                {**item, "text": translated_instruction} for item in subject_passes
                            ]
                elif region_edit:
                    subject_passes = self._measured_region_passes(
                        img, score_raw, prompt,
                        surface_recolor=recolor_edit,
                        faces=source_faces,
                    )
                elif preserve_frame:
                    subject_passes = self._measured_subject_passes(
                        img, score_raw, prompt, source_faces,
                        detected_faces=detected_faces,
                    )
                    tried_head = False
                    if appearance_edit and not subject_passes:
                        recovered = self._appearance_head_pass(
                            img, score_raw, prompt, source_faces,
                        )
                        tried_head = True
                        if recovered is not None:
                            subject_passes = [recovered]
                    if appearance_edit and subject_passes:
                        subject_passes = self._narrow_appearance_passes(
                            img, subject_passes, score_raw, prompt_raw, prompt, source_faces,
                        )
                        # An appearance predicate the segmenter cannot draw
                        # ("bald", "beard") falls back to the whole person, and
                        # the full-photo pass regenerates everything with no
                        # spatial anchor (glossy scalp, edit_1791760178). The
                        # head of the measured person is where those changes
                        # live; the local pass keeps the photo anchored.
                        if (len(subject_passes) == 1
                                and subject_passes[0].get("deliver_region")
                                and subject_passes[0].get("person") is not None):
                            head_zone = self._head_zone_for_subject(
                                img, subject_passes[0]["person"], source_faces,
                            )
                            if head_zone is not None:
                                subject_passes = [{
                                    **subject_passes[0],
                                    "mask": head_zone,
                                    "deliver_region": False,
                                    "measured_request": True,
                                }]
                        # v5.1211 (medido, sin formas ni palabras): si la parte
                        # que el segmentador midió está CONTENIDA en la cabeza,
                        # el agujero de generación pasa a ser la cabeza — con el
                        # agujero de 0,7% que «calvo» mide sobre un pelo
                        # existente el modelo no puede quitar el pelo que queda
                        # fuera (edit_1792085759 siguió con pelo;
                        # edit_1792085849, por la ruta de cabeza, salió calvo) —
                        # y el pegado mide el cambio real dentro de esa cabeza
                        # (doctrina «cambio medido ∩ permitido»): el pelo
                        # quitado es cambio y entra; pelo, gafas o cara que el
                        # modelo no cambió no entran.
                        if (len(subject_passes) == 1
                                and subject_passes[0].get("measured_request")
                                and not subject_passes[0].get("deliver_region")
                                and subject_passes[0].get("person") is not None):
                            measured_item = subject_passes[0]
                            head_zone = self._head_zone_for_subject(
                                img, measured_item["person"], source_faces,
                                announce=False,
                            )
                            if head_zone is not None and measured_item.get("mask") is not None:
                                part = np.asarray(
                                    measured_item["mask"].convert("L").resize(
                                        img.size, Image.Resampling.NEAREST,
                                    )
                                ) >= 96
                                head = np.asarray(
                                    head_zone.convert("L").resize(
                                        img.size, Image.Resampling.NEAREST,
                                    )
                                ) >= 96
                                part_px = int(part.sum())
                                inside = (
                                    float((part & head).sum()) / part_px
                                    if part_px else 0.0
                                )
                                if part_px >= 48 and inside >= 0.85:
                                    print(
                                        "[ImgEditor] La parte pedida mide dentro "
                                        f"de la cabeza ({inside:.0%}): el agujero "
                                        "es la cabeza y el pegado es el cambio "
                                        "medido en ella",
                                        flush=True,
                                    )
                                    subject_passes = [{
                                        **measured_item,
                                        "mask": head_zone,
                                        "paste_change_only": True,
                                    }]
                                    # v5.1213: esta ruta entrega el mismo
                                    # cambio de apariencia anclado a la cabeza
                                    # que el plan focal; la medición de pelo
                                    # residual tiene que cubrirla también.
                                    _head_hole_plan = True
                                    _head_hole_zone = head_zone
                    if appearance_edit and not subject_passes and not tried_head:
                        recovered = self._appearance_head_pass(
                            img, score_raw, prompt, source_faces,
                        )
                        if recovered is not None:
                            subject_passes = self._narrow_appearance_passes(
                                img, [recovered], score_raw, prompt_raw, prompt, source_faces,
                            )
                if recolor_edit and subject_passes:
                    kept_passes = []
                    whole = Image.new("L", img.size, 255)
                    for item in subject_passes:
                        if item.get("requested_part"):
                            # The requested part may belong to a person.
                            # Completing the whole surface and then removing
                            # people would erase the very part being edited.
                            kept_passes.append(item)
                            continue
                        surface = item.get("mask")
                        extended = self._complete_request_surface(
                            img, surface, whole, source_faces,
                            masker=self._get_clipseg_masker(),
                        )
                        if extended is not None:
                            surface = extended
                        surface = self._surface_without_people(
                            img, surface, source_faces,
                        )
                        if surface is None:
                            continue
                        coverage = float((np.asarray(surface.convert("L")) >= 96).mean())
                        # A speck whose box already fills the photo is not the
                        # surface. Painting it at full strength leaves a stain.
                        if coverage < 0.08:
                            print(
                                f"[ImgEditor] La superficie medida queda en {coverage:.1%}; "
                                "no se sustituye la foto",
                                flush=True,
                            )
                            continue
                        kept_passes.append({
                            **item,
                            "mask": surface,
                            "measured_request": True,
                            "coverage": coverage,
                        })
                    subject_passes = kept_passes
                if (region_edit and not recolor_edit
                        and authority_facts.get("structural_remove")
                        and subject_passes):
                    # v5.1209: lo que se ELIMINA vive entre los parches de la
                    # zona medida (los cuadros sobre la pared medida; las
                    # hojas sueltas del caso planta) y FastEdit no toca nada
                    # fuera de la máscara (0 % medido el 15/10). El cierre
                    # ocurre dentro de la propia caja de la zona y después se
                    # restan las personas; nada se escribe fuera de ella.
                    closed_passes = []
                    for item in subject_passes:
                        zone = item.get("mask")
                        finished = (
                            self._closed_structural_zone(img, zone, source_faces)
                            if zone is not None else None
                        )
                        if finished is None:
                            closed_passes.append(item)
                            continue
                        coverage = float(
                            (np.asarray(finished.convert("L")) >= 96).mean()
                        )
                        closed_passes.append({
                            **item,
                            "mask": finished,
                            "coverage": coverage,
                        })
                    subject_passes = closed_passes
                if preserve_frame:
                    if len(subject_passes) == 1:
                        subject_mask = subject_passes[0]["mask"]
                        paste_limit = subject_passes[0].get("person")
                        edit_text = subject_passes[0]["text"]
                        # The grouped sentence is often English only. Scoring
                        # that text alone drops the concept the user wrote:
                        # "sin medias" is a body change, and "go without
                        # stockings" is not, so the measured subject was sent
                        # as a global pass. The original clause stays in the
                        # score. This block runs for one instruction; a comma
                        # list has already left on its own passes.
                        authority_facts = self._prompt_authority_facts(score_raw, edit_text)
                    elif not subject_passes:
                        if region_edit:
                            print(
                                "[ImgEditor] Región no medida; no se sustituye la foto",
                                flush=True,
                            )
                        else:
                            print(
                                "[ImgEditor] Sujeto no medido; no se sustituye la foto por un fotograma vacío",
                                flush=True,
                            )
                body_edit = bool(authority_facts["body"] and not authority_facts["pose"])
                if scenario_edit:
                    frame_note = "escenario nuevo, la escena completa sale del modelo"
                elif appearance_edit and len(subject_passes) > 1:
                    frame_note = "cada zona medida, el resto de la foto se conserva"
                elif appearance_edit and subject_mask is not None:
                    frame_note = "solo la zona medida, el resto de la foto se conserva"
                elif appearance_edit:
                    frame_note = "zona no medida; no se sustituye la foto"
                elif region_edit and subject_mask is not None:
                    frame_note = "región medida, el resto de la foto se conserva"
                elif region_edit:
                    frame_note = "región no medida; no se sustituye la foto"
                elif body_edit and subject_mask is not None:
                    frame_note = "cuerpo medido, el resto de la foto se conserva"
                elif body_edit:
                    frame_note = "edición corporal global; otros detalles pueden cambiar"
                elif prompt_authority and preserve_frame and len(subject_passes) > 1:
                    frame_note = "cada frase en su sujeto, el resto de la foto se conserva"
                elif prompt_authority and preserve_frame and subject_mask is not None:
                    frame_note = "sujeto medido, el resto de la foto se conserva"
                elif prompt_authority and preserve_frame:
                    frame_note = "foto conservada, sujeto no medido"
                elif prompt_authority:
                    frame_note = "postura/cuerpo sin ancla de latente"
                else:
                    frame_note = ""
                if (appearance_edit or region_edit) and subject_mask is None and not subject_passes:
                    return img, f"{frame_note} | cumplimiento visual no verificado", None
                self._release_local_gpu_before_comfy()
                fast_client = get_flux_edit_comfy_client()
                loaded, load_msg = fast_client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
                if loaded:
                    multi_output = None
                    if len(subject_passes) > 1 and prompt_authority:
                        multi_output, fast_msg, produced = self._apply_subject_passes(
                            fast_client, img, subject_passes, seed, negative_prompt,
                            lora_name, lora_strength, progress_callback, cancel_check,
                            localize=appearance_edit or region_edit, faces=source_faces,
                        )
                        if not produced:
                            multi_output = None
                    fast_result = None
                    generation_image = img
                    # A distinct part is still the sampler hole and the paste.
                    # Painting the whole person at full strength flattened the
                    # skin and left the request undone (2026-10-08 22:09, 19%
                    # skin; 22:12, the man at 35% still had hair). The path
                    # that followed the sentence was one Turbo of the photo:
                    # 12 steps, the instruction, no hole. The person already
                    # measured is only the paste, so the room stays.
                    local_box = None
                    vision_reference = None
                    body_request_region = None
                    body_zone = None
                    deliver_region = bool(
                        len(subject_passes) == 1
                        and subject_passes[0].get("deliver_region")
                    )
                    if body_edit and subject_mask is not None and multi_output is None:
                        body_request_region = self._request_region_on_subject(
                            img, subject_mask, [score_raw, edit_text],
                        )
                        if body_request_region is not None:
                            body_request_region = self._continue_body_part(
                                img, body_request_region, subject_mask, source_faces,
                            )
                    # A body change with no distinct part ("nude") used to
                    # regenerate the whole photo without an anchor, and the
                    # model painted a white blob over the scene
                    # (edit_1791840439). The measured person is the zone; the
                    # local pass keeps the photo anchored, the same recipe
                    # that fixed the appearance fallback.
                    body_local = bool(
                        multi_output is None and body_edit
                        and body_request_region is None and subject_mask is not None
                    )
                    instruction_on_photo = bool(
                        multi_output is None
                        and subject_mask is not None
                        and appearance_edit and deliver_region
                    )
                    generation_mask = (
                        subject_mask
                        if (
                            (appearance_edit or region_edit)
                            and not instruction_on_photo
                        )
                        else None
                    )
                    if (
                        multi_output is None
                        and (appearance_edit or region_edit)
                        and subject_mask is not None
                        and not instruction_on_photo
                    ):
                        generation_image, generation_mask, local_box = self._appearance_work_region(
                            img, subject_mask,
                            subject_passes[0].get("person")
                            if len(subject_passes) == 1 else None,
                        )
                        if local_box is None:
                            local_box = (0, 0, img.size[0], img.size[1])
                            print(
                                "[ImgEditor] La ventana local no cupo; "
                                "se integra solo la zona medida sobre la foto original",
                                flush=True,
                            )
                    elif (
                        multi_output is None and body_edit
                        and body_request_region is not None
                    ):
                        generation_image, generation_mask, local_box = (
                            self._appearance_work_region(img, body_request_region)
                        )
                        if local_box is None:
                            local_box = (0, 0, img.size[0], img.size[1])
                            generation_mask = body_request_region
                    elif body_local:
                        generation_image, generation_mask, local_box = (
                            self._appearance_work_region(
                                img, subject_mask,
                                subject_passes[0].get("person")
                                if len(subject_passes) == 1 else None,
                            )
                        )
                        if local_box is None:
                            local_box = (0, 0, img.size[0], img.size[1])
                        print(
                            "[ImgEditor] Cuerpo sin parte distinta: la persona "
                            "medida es la zona y la foto queda anclada",
                            flush=True,
                        )
                    elif instruction_on_photo:
                        # Anclar el pase en la persona medida. Un pase de la
                        # foto entera con no_kontext y sin máscara a denoise
                        # 1.0 arranca de ruido (el latente de la foto se
                        # reemplaza del todo) y pinta un blob; la persona ya
                        # medida es la zona y la ventana local mantiene la
                        # foto anclada, la misma receta del fallback de
                        # apariencia y del cuerpo sin parte.
                        generation_image, generation_mask, local_box = (
                            self._appearance_work_region(
                                img, subject_mask,
                                subject_passes[0].get("person")
                                if len(subject_passes) == 1 else None,
                            )
                        )
                        if local_box is None:
                            local_box = (0, 0, img.size[0], img.size[1])
                        print(
                            "[ImgEditor] Sin cabeza medida: la persona es la "
                            "zona y la foto queda anclada",
                            flush=True,
                        )
                    if multi_output is None:
                        if appearance_edit:
                            fast_prompt = self._compose_generation_prompt(
                                edit_text, engine="longcat", magnitude=0.85,
                                pose_change=False, body_transform=False,
                            )
                            print(
                                "[ImgEditor] Apariencia: solo la zona medida, el resto queda fuera",
                                flush=True,
                            )
                        elif region_edit:
                            fast_prompt = self._compose_generation_prompt(
                                edit_text, engine="longcat", magnitude=0.85,
                                pose_change=False, body_transform=False,
                            )
                            print(
                                "[ImgEditor] Región: solo la zona medida, el resto de la foto queda fuera",
                                flush=True,
                            )
                        elif prompt_authority:
                            # Pose may change the frame. A body change keeps the
                            # camera and only replaces the requested body state.
                            # v5.1215: un escenario nuevo pide que la escena
                            # cambie — el wrapper recibe el mismo gesto que la
                            # postura para no escribir «keep the same room»
                            # contra «que esté en una playa».
                            fast_prompt = self._compose_generation_prompt(
                                edit_text, engine="longcat", magnitude=0.85,
                                pose_change=bool(authority_facts["pose"]) or scenario_edit,
                                body_transform=bool(authority_facts["body"]),
                                body_direction=str(authority_facts["body_direction"]),
                                body_intensity=float(authority_facts["body_intensity"]),
                            )
                            if scenario_edit:
                                authority_label = "Escenario"
                            elif authority_facts["pose"] and authority_facts["body"]:
                                authority_label = "Postura y cuerpo"
                            elif authority_facts["pose"]:
                                authority_label = "Postura"
                            else:
                                authority_label = "Cuerpo"
                            print(
                                f"[ImgEditor] {authority_label}: Turbo ve la foto y la instrucción manda"
                                + ("; solo la zona medida del cuerpo" if generation_mask is not None else ""),
                                flush=True,
                            )
                        elif quality_hybrid:
                            fast_prompt = (
                                "Edit this exact photo. "
                                f"Apply the requested change fully and make it clearly visible: {prompt}. "
                                "Keep every person, object, room detail, camera view "
                                "and composition that the request does not change "
                                "exactly as in the input. Do not alter anything else. "
                                "Keep people anatomically coherent, with no missing or extra limbs."
                            )
                            fast_prompt += self._bald_recipe_tail(prompt)
                        else:
                            fast_prompt = (
                                "Instruction: Edit this exact photo. "
                                f"Apply every requested change: {prompt}. "
                                "Make all requested changes clearly visible. "
                                "Keep people anatomically coherent, with no missing or extra limbs."
                            )
                            fast_prompt += self._bald_recipe_tail(prompt)
                        if negative_prompt:
                            # Turbo uses CFG=1 and does not encode a separate
                            # negative branch, so keep user exclusions in text.
                            fast_prompt += f" Avoid: {negative_prompt}."
                        from roop.img_editor.flux_edit_comfy_client import measured_appearance_sample, measured_region_sample
                        appearance_sample = (
                            measured_appearance_sample(generation_mask is not None)
                            if appearance_edit or recolor_edit
                            else measured_region_sample(generation_mask is not None) if region_edit
                            else {
                                "num_inference_steps": 12, "denoise": 1.0,
                                # The local body pass keeps the photo anchored;
                                # the full-photo body pass does not.
                                "no_kontext": generation_mask is None,
                            }
                        )
                        if appearance_edit and generation_mask is not None and not deliver_region:
                            appearance_sample, vision_reference = (
                                self._with_covered_reference(
                                    appearance_sample, generation_image,
                                    generation_mask, paste_limit, local_box,
                                )
                            )
                        if appearance_edit and generation_mask is not None:
                            print(
                                "[ImgEditor] Apariencia: una pasada, "
                                f"{appearance_sample['num_inference_steps']} pasos, "
                                f"fuerza {appearance_sample['denoise']:.2f}",
                                flush=True,
                            )
                        elif instruction_on_photo:
                            print(
                                "[ImgEditor] Un pase, "
                                f"{appearance_sample['num_inference_steps']} pasos, "
                                f"fuerza {appearance_sample['denoise']:.2f}, sin agujero",
                                flush=True,
                            )
                        fast_result, fast_msg = fast_client.generate(
                            image=generation_image, prompt=fast_prompt,
                            negative_prompt=negative_prompt,
                            guidance_scale=1.0,
                            seed=seed,
                            prompt_authority=prompt_authority,
                            preserve_frame=preserve_frame,
                            mask_image=generation_mask,
                            keep_source_pixels=generation_mask is not None,
                            reference_image=vision_reference,
                            lora_name=lora_name, lora_strength=lora_strength,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            **appearance_sample,
                        )
                    if multi_output is not None or fast_result is not None:
                        if multi_output is not None:
                            output = multi_output
                        else:
                            output = fast_result.image.convert("RGB")
                            gen_size = output.size
                            if (region_edit and generation_mask is not None
                                    and not (recolor_edit
                                             and any(item.get("requested_part") for item in subject_passes))):
                                output, repair_msg = self._repair_inserted_surface(
                                    fast_client, generation_image, generation_mask, output,
                                    source_faces, local_box, fast_prompt, seed,
                                    negative_prompt, lora_name, lora_strength,
                                    progress_callback, cancel_check,
                                )
                                if repair_msg:
                                    fast_msg = f"{fast_msg}; {repair_msg}"
                            if body_edit and subject_mask is not None:
                                zone_box = (
                                    local_box
                                    if local_box is not None
                                    else (0, 0, img.size[0], img.size[1])
                                )
                                body_zone = self._body_paste_zone(
                                    img, output, subject_mask, body_request_region,
                                    [score_raw, edit_text], zone_box,
                                )
                                output = (
                                    self._integrate_local_generation(
                                        img, output, body_zone, zone_box,
                                        keep_measured_mask=True,
                                    ) if body_zone is not None else img
                                )
                                print(
                                    "[ImgEditor] Cuerpo: la zona pedida se pega "
                                    "entera; el resto de la persona se conserva",
                                    flush=True,
                                )
                                if local_box is not None and body_zone is not None:
                                    refined = self._second_focused_pass(
                                        fast_client, output, img,
                                        body_zone, local_box, gen_size,
                                        fast_prompt, negative_prompt, seed,
                                        prompt_authority, preserve_frame,
                                        vision_reference, lora_name, lora_strength,
                                        appearance_sample, progress_callback,
                                        cancel_check,
                                    )
                                    if refined is not None:
                                        output = refined
                                        fast_msg = (
                                            f"{fast_msg}; refino focal de la zona medida"
                                        )
                                else:
                                    print(
                                        "[ImgEditor] Refino focal omitido: "
                                        f"ventana={local_box is not None} "
                                        f"zona corporal={body_zone is not None}",
                                        flush=True,
                                    )
                            elif local_box is not None:
                                keep_request = bool(
                                    len(subject_passes) == 1
                                    and (
                                        subject_passes[0].get("measured_request")
                                        or subject_passes[0].get("deliver_region")
                                    )
                                    # v5.1211: cabeza como agujero pega el
                                    # cambio medido, no la máscara entera.
                                    and not subject_passes[0].get("paste_change_only")
                                )
                                output = self._integrate_local_generation(
                                    img, output, subject_mask, local_box,
                                    limit_mask=paste_limit,
                                    keep_measured_mask=keep_request,
                                )
                                print(
                                    "[ImgEditor] Región integrada sobre la foto original",
                                    flush=True,
                                )
                                refined = self._second_focused_pass(
                                    fast_client, output, img, subject_mask,
                                    local_box, gen_size, fast_prompt,
                                    negative_prompt, seed, prompt_authority,
                                    preserve_frame, vision_reference,
                                    lora_name, lora_strength, appearance_sample,
                                    progress_callback, cancel_check,
                                )
                                if refined is not None:
                                    output = refined
                                    fast_msg = f"{fast_msg}; refino focal de la zona medida"
                            elif instruction_on_photo:
                                output = self._integrate_local_generation(
                                    img, output, subject_mask,
                                    (0, 0, img.size[0], img.size[1]),
                                    keep_measured_mask=True,
                                )
                                if body_edit:
                                    body_zone = subject_mask
                                print(
                                    "[ImgEditor] La instrucción ve la foto; "
                                    "se pega solo la persona medida",
                                    flush=True,
                                )
                            elif body_edit and generation_mask is not None:
                                output = self._integrate_local_generation(
                                    img, output, generation_mask,
                                    (0, 0, img.size[0], img.size[1]),
                                    keep_measured_mask=True,
                                )
                                body_zone = generation_mask
                                print(
                                    "[ImgEditor] El predicado no tiene zona; "
                                    "se pega la región ya medida",
                                    flush=True,
                                )
                        if quality_hybrid or prompt_authority:
                            no_op = self._edit_is_nearly_unchanged(img, output)
                            # v5.1219: un fantasma NO es no-op: el modelo
                            # desvaneció el material original a opacidad
                            # parcial (log 01:49: cáscara gris sobre el
                            # cuero cabelludo con el tono de la pared) y el
                            # pegado lo conservó porque midió cambio. Firma:
                            # estructura HF retenida + amplitud caída.
                            ghost = (
                                (not no_op)
                                and self._edit_leaves_a_ghost(img, output)
                            )
                            # v5.1220: una cara que el modelo pinta donde la
                            # foto no tiene ninguna (log 10:23: tercera cara
                            # sobre la coronilla del calvo; la foto entró
                            # con 2 caras y salió con 3). El pegado por
                            # cambio medido la conserva porque es textura
                            # nueva; ni el no-op ni el fantasma la cogen.
                            added_face = (
                                (not no_op) and (not ghost)
                                and self._added_face_mask(
                                    img, output,
                                    detected_faces or source_faces,
                                ) is not None
                            )
                            if (multi_output is None
                                    and (no_op or ghost or added_face)
                                    and not (cancel_check and cancel_check())):
                                if added_face:
                                    print(
                                        "[ImgEditor] Cara añadida por el modelo "
                                        "sin contrapartida en la foto; segundo "
                                        "intento Turbo con las mismas personas",
                                        flush=True,
                                    )
                                    retry_prompt = (
                                        "The previous edit painted a new face "
                                        "or person that is not in the input "
                                        "photo. The result must contain exactly "
                                        "the same people as the input: no face "
                                        "may be added, copied or duplicated. "
                                        f"Do this visible edit now: {edit_text}. "
                                        "Keep the pose and the framing, and "
                                        "leave every original person exactly "
                                        "where they are."
                                    )
                                elif ghost:
                                    print(
                                        "[ImgEditor] Resto fantasma del material "
                                        "original dentro de la zona cambiada; "
                                        "segundo intento Turbo con reemplazo "
                                        "completo (sin resto translúcido)",
                                        flush=True,
                                    )
                                    retry_prompt = (
                                        "The previous edit only faded the original "
                                        "content instead of replacing it, which "
                                        "leaves a translucent ghost of the "
                                        "original. Do this visible edit now: "
                                        f"{edit_text}. "
                                        "Fully replace the previous material: no "
                                        "translucent remnant of it may remain, "
                                        "the new content must be fully opaque. "
                                        "Keep the pose and the framing. "
                                        "Leave every other part of the photo "
                                        "exactly as it is."
                                    )
                                else:
                                    print(
                                        "[ImgEditor] Cambio casi nulo sobre la foto; "
                                        "segundo intento Turbo con la instrucción de refuerzo",
                                        flush=True,
                                    )
                                    if prompt_authority and authority_facts["pose"]:
                                        retry_prompt = (
                                            f"The input photo has not yet been changed. Do this visible edit now: {edit_text}. "
                                            "The requested pose or body change must be obvious. "
                                            "Do not keep the previous pose or the previous state of what was asked to change. "
                                            "Leave every other person as they are."
                                        )
                                    elif prompt_authority:
                                        retry_prompt = (
                                            f"The input photo has not yet been changed. Do this visible edit now: {edit_text}. "
                                            "Keep the pose and the framing. "
                                            "Do not keep the previous state of what was asked to change. "
                                            "Leave every other person as they are."
                                        )
                                    else:
                                        retry_prompt = (
                                            f"The input photo has not yet been changed. Do this visible edit now: {prompt}. "
                                            "The requested change must be obvious in the result. "
                                            "Preserve unrelated people, objects, anatomy and the camera view."
                                        )
                                if negative_prompt:
                                    retry_prompt += f" Avoid: {negative_prompt}."
                                retry_prompt += self._bald_recipe_tail(
                                    edit_text if prompt_authority else prompt
                                )
                                try:
                                    retry_result, retry_msg = fast_client.generate(
                                        image=generation_image, prompt=retry_prompt,
                                        negative_prompt=negative_prompt,
                                        guidance_scale=1.0,
                                        seed=seed,
                                        prompt_authority=prompt_authority,
                                        preserve_frame=preserve_frame,
                                        mask_image=generation_mask,
                                        keep_source_pixels=generation_mask is not None,
                                        reference_image=vision_reference,
                                        lora_name=lora_name, lora_strength=lora_strength,
                                        progress_callback=progress_callback,
                                        cancel_check=cancel_check,
                                        **appearance_sample,
                                    )
                                except Exception as retry_error:
                                    retry_result, retry_msg = None, str(retry_error)
                                if retry_result is None:
                                    fast_msg = f"{fast_msg}; segundo intento falló: {retry_msg}"
                                    print(f"[ImgEditor] Segundo intento falló: {retry_msg}", flush=True)
                                if retry_result is not None:
                                    output = retry_result.image.convert("RGB")
                                    if body_edit and subject_mask is not None:
                                        zone_box = (
                                            local_box
                                            if local_box is not None
                                            else (0, 0, img.size[0], img.size[1])
                                        )
                                        body_zone = self._body_paste_zone(
                                            img, output, subject_mask, body_request_region,
                                            [score_raw, edit_text], zone_box,
                                        )
                                        output = (
                                            self._integrate_local_generation(
                                                img, output, body_zone, zone_box,
                                                keep_measured_mask=True,
                                            ) if body_zone is not None else img
                                        )
                                    elif local_box is not None:
                                        keep_request = bool(
                                            len(subject_passes) == 1
                                            and (
                                                subject_passes[0].get("measured_request")
                                                or subject_passes[0].get("deliver_region")
                                            )
                                        )
                                        output = self._integrate_local_generation(
                                            img, output, subject_mask, local_box,
                                            limit_mask=paste_limit,
                                            keep_measured_mask=keep_request,
                                        )
                                    elif instruction_on_photo:
                                        output = self._integrate_local_generation(
                                            img, output, subject_mask,
                                            (0, 0, img.size[0], img.size[1]),
                                            keep_measured_mask=True,
                                        )
                                        if body_edit:
                                            body_zone = subject_mask
                                    elif body_edit and generation_mask is not None:
                                        output = self._integrate_local_generation(
                                            img, output, generation_mask,
                                            (0, 0, img.size[0], img.size[1]),
                                            keep_measured_mask=True,
                                        )
                                        body_zone = generation_mask
                                    # El retry también pega a escala ×N sobre
                                    # foto grande (log 01:22: x3.34 sin
                                    # refinar); el mismo zoom nativo que la
                                    # pasada principal le corresponde aquí.
                                    refine_subject = (
                                        body_zone
                                        if body_edit and body_zone is not None
                                        else subject_mask
                                    )
                                    if local_box is not None and refine_subject is not None:
                                        refined = self._second_focused_pass(
                                            fast_client, output, img,
                                            refine_subject, local_box, gen_size,
                                            fast_prompt, negative_prompt, seed,
                                            prompt_authority, preserve_frame,
                                            vision_reference, lora_name,
                                            lora_strength, appearance_sample,
                                            progress_callback, cancel_check,
                                        )
                                        if refined is not None:
                                            output = refined
                                            fast_msg = (
                                                f"{fast_msg}; refino focal de la zona medida"
                                            )
                                    # v5.1220: si el retry vuelve a pintar la
                                    # misma cara, esa región se restaura de
                                    # la foto original: una foto no gana
                                    # personas por un defecto del modelo.
                                    leftover_face = self._added_face_mask(
                                        img, output,
                                        detected_faces or source_faces,
                                    )
                                    if (leftover_face is not None
                                            and output.size == img.size):
                                        output = Image.composite(
                                            img, output, leftover_face,
                                        )
                                        fast_msg = (
                                            f"{fast_msg}; cara añadida "
                                            "restaurada de la foto original"
                                        )
                                        print(
                                            "[ImgEditor] El segundo intento "
                                            "conserva una cara añadida; esa "
                                            "región se restaura de la foto "
                                            "original",
                                            flush=True,
                                        )
                                    fast_msg = f"{fast_msg}; segundo intento: {retry_msg}"
                                    print(f"[ImgEditor] Segundo intento aplicado: {retry_msg}", flush=True)
                                    no_op = self._edit_is_nearly_unchanged(img, output)
                                if body_edit and body_zone is None:
                                    fast_msg = (
                                        f"{fast_msg}; «{edit_text}» sin cambio "
                                        "medido en la zona pedida"
                                    )
                                    print(
                                        "[ImgEditor] Cuerpo: sin cambio medido "
                                        "en la zona pedida tras el reintento",
                                        flush=True,
                                    )
                            if no_op:
                                self._last_instruction_compliance = {
                                    "compliant": False,
                                    "source": "visual_change",
                                    "detail": "El edit generativo apenas cambió la imagen",
                                }
                            if quality_hybrid:
                                from roop.img_editor.top_tier_finish import run_quality_enhance_after_edit
                                output, finish_note, quality_analysis = run_quality_enhance_after_edit(
                                    output, img, enhance_tier=enhance_tier,
                                    use_generative=False,
                                    preserve_faces=bool(quality_preserve_faces),
                                    progress_callback=progress_callback,
                                    cancel_check=cancel_check,
                                    quality_steps=quality_steps,
                                )
                                self._last_quality_analysis = quality_analysis
                                if cancel_check and cancel_check():
                                    return None, "Cancelado", None
                                pov_warning = (
                                    " | POV no verificado: el modelo puede conservar el ángulo original"
                                    if explicit_viewpoint_request else ""
                                )
                                print(f"[ImgEditor] Calidad con prompt: Turbo + acabado clásico{pov_warning}", flush=True)
                                no_op_warning = (
                                    " | cambio solicitado apenas visible"
                                    if no_op else " | cumplimiento visual no verificado"
                                )
                                authority_note = f" | {frame_note}" if frame_note else ""
                                return output, f"{fast_msg} | Turbo + {finish_note}{authority_note}{no_op_warning}{pov_warning}", None
                        if prompt_authority:
                            no_op = self._edit_is_nearly_unchanged(img, output)
                            if no_op:
                                self._last_instruction_compliance = {
                                    "compliant": False,
                                    "source": "visual_change",
                                    "detail": "El edit generativo apenas cambió la imagen",
                                }
                        if output.size != img.size:
                            output = output.resize(img.size, Image.Resampling.LANCZOS)
                        warning = (
                            " | POV no verificado: el modelo puede conservar el ángulo original"
                            if explicit_viewpoint_request else ""
                        )
                        if prompt_authority:
                            no_op_warning = (
                                " | cambio solicitado apenas visible"
                                if no_op else " | cumplimiento visual no verificado"
                            )
                            print(
                                f"[ImgEditor] {frame_note}{no_op_warning}{warning}",
                                flush=True,
                            )
                            return (output,
                                    f"{fast_msg} | {frame_note}{no_op_warning}{warning}",
                                    None)
                        print(f"[ImgEditor] Prompt traducido enviado a Turbo; cumplimiento visual no verificado{warning}", flush=True)
                        return (output,
                                f"{fast_msg} | prompt traducido enviado a Turbo; cumplimiento visual no verificado{warning}",
                                None)
                    fast_failure = fast_msg
                else:
                    fast_failure = load_msg
                    print(f"[ImgEditor] Ruta rápida no cargada: {load_msg}", flush=True)
            except Exception as fast_error:
                fast_failure = str(fast_error)
                print(f"[ImgEditor] Ruta rápida falló: {fast_error}", flush=True)
            if cancel_check and cancel_check():
                return None, "Cancelado", None
            # The user chose Turbo. A failed upload or inference is not
            # permission to load the slower Full model through the legacy path.
            return None, f"Turbo no pudo aplicar la edición: {fast_failure}", None

        if engine == "flux_dev_abliterated":
            vram = get_vram_gb()
            if 0 < vram <= 8:
                print(f"[ImgEditor] VRAM={vram}GB detectada. Flux Dev puede ser lento o fallar.")
        elif engine == "qwen_edit":
            pass

        img_description = ""  # DELIBERADAMENTE vacío: no usamos descripciones generativas de imagen (evita alucinaciones). El análisis es texto-del-prompt + preservación fuerte.

        structural_score = 0.0
        high_structural = False
        structural_bias = "neutral"
        body_transform_intensity = 0.0
        body_direction = "change"
        background_focus = False
        appearance_score = 0.0
        if quality_only:
            mag_suggested = 0.35
            mask_target = "subject"
            axis_scores = {}
            is_global = True
            # v5.1131b — inicializar flags de detección aquí también: los checks
            # posteriores (`if angle_change and not quality_only:`) evalúan
            # angle_change PRIMERO en el `and` → UnboundLocalError en modo
            # calidad sin prompt (log 15:14, crash x2 en tier hd/4k).
            angle_change = False
            pose_change = False
        elif use_semantic:
            try:
                nlp = self._get_semantic_analyzer(full_ai=False)
                mag_suggested = nlp.get_magnitude(prompt)
                mask_target = nlp.detect_target(prompt)
                axis_scores = nlp.get_axis_scores(prompt) if hasattr(nlp, "get_axis_scores") else {}
                try:
                    appearance_score = float(nlp.score_concepts(prompt).get("attribute_appearance", 0.0))
                except Exception:
                    appearance_score = 0.0
                structural_score = float(axis_scores.get("structural", 0.0))
                if hasattr(nlp, "get_body_transform_intensity"):
                    body_transform_intensity = float(nlp.get_body_transform_intensity(prompt))
                if hasattr(nlp, "get_body_direction"):
                    body_direction = str(nlp.get_body_direction(prompt))
                if hasattr(nlp, "is_structural_dominant"):
                    high_structural = nlp.is_structural_dominant(prompt)
                else:
                    struct_thresh = getattr(nlp, "STRUCTURAL_SIGNAL_THRESHOLD", 0.06)
                    high_structural = structural_score > struct_thresh
                if hasattr(nlp, "get_structural_bias"):
                    structural_bias = nlp.get_structural_bias(prompt)
                background_focus = (
                    mask_target == "background" and structural_bias == "neutral"
                )
                is_global = (
                    not background_focus
                    and (
                        high_structural
                        or (mask_target == "subject" and mag_suggested < 0.45)
                    )
                )
                print(
                    f"[ImgEditor] Semantic: Magnitude={mag_suggested:.2f}, Target={mask_target}, "
                    f"Structural={structural_score:.2f}, Bias={structural_bias}, "
                    f"BodyTransform={body_transform_intensity:.2f}, Appearance={appearance_score:.2f}"
                    + (", BackgroundFocus=True" if background_focus else "")
                )
                # Detección de cambio de ángulo/POV (vista no vista → regenerar + FacePreserver)
                # + cambio de POSTURA (el cuerpo destino no existe en la fuente: el
                # modo EDIT con ancla la conserva y la pose NUNCA cambia; necesita
                # el mismo régimen de regeneración que POV). Patrones existentes,
                # cero vocabulario nuevo.
                # Bilingüe SOLO eje pose: la traducción puede destruir la señal
                # (log 12:56: 'DE RODILLAS'→'ON HIS KNEES', pose 0.19→0.0; el
                # crudo SÍ puntúa en conceptos ES). La magnitud NO se fusiona:
                # el ES puntúa más alto en general y el max secuestraría a body
                # por la puerta mag>=0.80 (p. ej. 'pon un sombrero' crudo 0.81).
                # La fuerza de pose la pone el bloque POSE (mag≥0.85 + global).
                _change_scope = self._requested_change_scope(prompt)
                _raw_change_scope = self._requested_change_scope(prompt_raw)
                try:
                    _raw_axes = nlp.get_axis_scores(_raw_change_scope) if hasattr(nlp, "get_axis_scores") else {}
                    _raw_pose = float((_raw_axes or {}).get("pose", 0.0))
                    try:
                        axis_scores["pose"] = max(float(axis_scores.get("pose", 0.0)), _raw_pose)
                    except Exception:
                        pass
                except Exception:
                    _raw_pose = 0.0
                try:
                    from roop.img_editor.prompt_analyzer import PromptAnalyzer
                    _pa = PromptAnalyzer()
                    _angle_hits = len(_pa._angle_pattern.findall(_change_scope.lower())) if hasattr(_pa, '_angle_pattern') else 0
                    # v5.1131: tolerancia subset (la MISMA que usa pose) — el regex
                    # exige la frase exacta ('de lado') pero frases equivalentes
                    # ('que se vea desde el lado', 'visto desde el lado') dan 0 hits
                    # aunque sus tokens SÍ están en ANGLE_KEYWORDS → POV caía al
                    # régimen genérico y la vista no cambiaba.
                    if _angle_hits == 0 and hasattr(_pa, "subset_hits"):
                        try:
                            _angle_hits = _pa.subset_hits(_raw_change_scope, _pa.ANGLE_KEYWORDS)
                        except Exception:
                            _angle_hits = 0
                    angle_change = _angle_hits > 0
                    if angle_change:
                        print(f"[ImgEditor] Cambio de ángulo/POV detectado ({_angle_hits} señales) → regeneración + FacePreserver", flush=True)
                    _pose_hits = len(_pa._pose_pattern.findall(_change_scope.lower())) if hasattr(_pa, '_pose_pattern') else 0
                    # Umbral 0.17 calibrado: pose real ES 0.19-0.41, ruido de
                    # fondo ('...por una playa') 0.158.
                    # Subconjunto sobre el CRUDO: tolera typos que rompen la
                    # frase ('DD RODILLAS' no casa el regex pero 'rodillas'
                    # está). Solo listas existentes, sin palabras nuevas.
                    _subset = 0
                    try:
                        if hasattr(_pa, "subset_hits"):
                            _subset = _pa.subset_hits(_raw_change_scope, _pa.POSE_KEYWORDS)
                    except Exception:
                        _subset = 0
                    pose_change = (_pose_hits > 0) or (_raw_pose > 0.17) or (_subset > 0)
                    if pose_change:
                        print(f"[ImgEditor] Cambio de postura detectado (patrón={_pose_hits}, eje_crudo={_raw_pose:.2f}, tokens={_subset}) → regeneración sin ancla", flush=True)
                except Exception:
                    angle_change = False
                    pose_change = False
            except Exception as e:
                print(f"[NLP] Error en análisis semántico: {e}. Usando fallback.")
                mag_suggested = 0.58
                mask_target = "subject"
                structural_score = 0.0
                high_structural = False
                structural_bias = "neutral"
                axis_scores = {}
                background_focus = False
                is_global = True
                angle_change = False
                pose_change = False
        else:
            mag_suggested = 0.58
            mask_target = "subject"
            structural_score = 0.0
            high_structural = False
            structural_bias = "neutral"
            axis_scores = {}
            background_focus = False
            is_global = True
            angle_change = False
            pose_change = False
            print("[ImgEditor] Análisis semántico desactivado — params por defecto")

        body_transform = (
            not quality_only
            and body_transform_intensity >= 0.08
            and mag_suggested >= 0.52
        )
        # A weak body signal inside an independently grounded clause describes
        # a local appearance change. Reserve whole-frame body regeneration for
        # strong transformations; otherwise the other people lose detail.
        _weak_compound_body = bool(
            body_transform and _compound_depth > 0
            and _compound_region_hint.strip()
            and body_transform_intensity < 0.30
        )
        if _weak_compound_body:
            body_transform = False
            print(
                f"[ImgEditor] Cambio corporal débil ({body_transform_intensity:.2f}) "
                "en cláusula localizada: edición de la región, no de toda la foto",
                flush=True,
            )
        # v5.1127 — SEÑAL CERO = SEÑAL (estructural, sin vocabulario): el analyzer
        # es un solapador token↔concepto. Cualquier palabra fuera de sus conceptos
        # ('calvo', 'ponle melen', la que sea) puntúa 0 en TODOS los ejes →
        # mag cae al floor 0.50 → klein EDIT (copy machine: no ejecuta y encima
        # deriva identidad, log 12:11). Pero 'cero señal semántica + frase con
        # verbo/atributo' NO es un pedido débil: es un pedido que el vocabulario
        # del analyzer no cubre. El resto del pipeline SÍ sabrá qué hacer si el
        # modelo ejecuta. Régimen fuerte para el sujeto (global, sin ancla).
        # quality_only/structural/pose/POV quedan excluidos (tienen señal propia).
        _zero_signal = (
            not quality_only
            and float(body_transform_intensity) <= 0.0
            and not high_structural
            and not angle_change
            and not pose_change
            and mag_suggested <= 0.55
            and len((prompt_raw or "").split()) >= 2
        )
        if _zero_signal:
            mag_suggested = 0.90
            is_global = True
            background_focus = False
            mask_target = "subject"
            print("[ImgEditor] Señal semántica CERO (fuera de conceptos): régimen fuerte para el sujeto (mag=0.90, global, sin ancla)", flush=True)
        if body_transform:
            mag_suggested = max(mag_suggested, 0.68 + body_transform_intensity * 0.12)
            is_global = True
            background_focus = False
        # POV/ángulo: forzar regeneración global (la vista no existe, debe alucinarse)
        if angle_change and not quality_only:
            mag_suggested = max(mag_suggested, 0.85)
            is_global = True
            background_focus = False
            print("[ImgEditor] POV: mag≥0.85 + global forzado (vista nueva)", flush=True)
        # Postura: el cuerpo destino TAMPOCO existe en la fuente. El modo EDIT
        # con ancla conserva la pose y el cambio NUNCA ocurre (log 12:56:
        # 'DEBE ESTAR DE RODILLAS' → klein+mask = misma pose). Mismo régimen
        # que POV. Body manda si hay señal corporal real (compuesto).
        if pose_change and not quality_only and not body_transform:
            mag_suggested = max(mag_suggested, 0.85)
            is_global = True
            background_focus = False
            print("[ImgEditor] POSE: mag≥0.85 + global forzado (postura nueva)", flush=True)
        # body_transform se decide exclusivamente con la señal corporal mínima
        # de arriba (>=0.08). La antigua promoción por magnitud alta convertía
        # ajustes de fondo/ambiente en body edits y podía cambiar de LongCat a
        # Kontext entre dos peticiones, obligando a una recarga de varios minutos.
        if angle_change and not quality_only and not body_transform:
            print("[ImgEditor] POV puro (sin señal corporal): body_transform NO forzado — regeneración de vista con img2img fuerte", flush=True)
        elif pose_change and not quality_only and not body_transform:
            print("[ImgEditor] POSE puro (sin señal corporal): body_transform NO forzado — regeneración de postura con img2img fuerte", flush=True)
        elif not quality_only:
            background_focus = (
                mask_target == "background"
                and structural_bias == "neutral"
                and not body_transform
            )
            if background_focus:
                is_global = False

        quality_anchors = [
            "improve quality sharpness detail resolution clarity enhance deblur upscale",
            "mejorar calidad nitidez detalle sharper mejorar color realzar ultra realista hiperrealista",
            "better colors vibrant lighting higher detail photographic quality",
            "desposterizar posterización posterization natural skin texture remove artifacts photographic realism",
        ]
        prompt_lower = prompt.lower()
        has_quality_request = self._light_score(prompt_lower, quality_anchors) > 0.03

        env_anchors = [
            "entorno", "environment", "fondo", "background", "escena", "scenery",
            "paisaje", "ambiente", "surroundings", "landscape",
        ]
        bg_in_prompt = self._light_score(prompt_lower, env_anchors) > 0.04
        subject_scene_anchors = [
            "personajes", "personas", "characters", "people", "sujetos", "subjects",
            "personaje", "persona", "cuerpos", "bodies", "caras", "faces", "figuras",
        ]
        subject_in_prompt = self._light_score(prompt_lower, subject_scene_anchors) > 0.03
        global_scene_realism = (
            bg_in_prompt and subject_in_prompt and has_quality_request and quality_hybrid
        )
        if global_scene_realism and not quality_only:
            background_focus = False
            is_global = True
            mask_target = "subject"
            high_structural = False
            structural_bias = "neutral"
            mag_suggested = min(mag_suggested, 0.52)
            print(
                "[ImgEditor] Fondo + personajes realismo → edición global de escena + TOP "
                "(no solo fondo, no structural add)"
            )
        elif bg_in_prompt and quality_hybrid and not quality_only:
            mask_target = "background"
            background_focus = True
            is_global = False

        quality_primary = False
        if quality_style == "solo_mejora" and quality_hybrid:
            if bg_in_prompt:
                mask_target = "background"
                background_focus = True
                is_global = False
                print(
                    "[ImgEditor] Prompt pide fondo/entorno → edit de fondo + TOP "
                    "(«Solo mejora» no aplica a cambios de escena)"
                )
            else:
                quality_primary = True
                mag_suggested = min(mag_suggested, 0.42)
                is_global = True
                print(
                    "[ImgEditor] Modo solo mejora (UI) → pipeline TOP clásico "
                    "(sin edit LongCat fuerte)"
                )
        elif quality_style == "hibrido":
            quality_primary = False
            if quality_hybrid:
                print("[ImgEditor] Modo híbrido forzado (UI) → edit + mejora TOP")
        elif (
            quality_hybrid
            and has_quality_request
            and not body_transform
            and not high_structural
            and not background_focus
            and not bg_in_prompt
        ):
            nlp = self._get_semantic_analyzer(full_ai=False)
            if hasattr(nlp, "is_quality_dominant") and nlp.is_quality_dominant(prompt):
                quality_primary = True
                mag_suggested = min(mag_suggested, 0.42)
                is_global = True
                print(
                    "[ImgEditor] Híbrido calidad-primaria (auto) → pipeline TOP clásico "
                    "(sin edit LongCat fuerte; evita sobre-procesar)"
                )

        if quality_hybrid and not quality_only and bg_in_prompt and not global_scene_realism:
            print(
                "[ImgEditor] Híbrido fondo/entorno → edit focal de fondo + TOP "
                "(personas preservadas)"
            )
        elif (
            quality_hybrid
            and not quality_only
            and not quality_primary
            and has_quality_request
            and not body_transform
            and not background_focus
        ):
            is_global = True
            print(
                "[ImgEditor] Híbrido calidad → edición global "
                "(sin máscara, evita costuras)"
            )

        rewrote = False
        requires_scene_regeneration = False
        multi_region_edit = False
        affected_regions = []
        rw_subject = ""
        # Texto pre-rewriter (intacto) por si el rewrite hay que descartarlo.
        prompt_pre_rewrite = prompt
        # v5.1127b — target DETERMINISTA (pre-rewriter) para el gate de abajo.
        _det_target_pre = mask_target
        _det_structural_pre = bool(high_structural)
        if use_rewriter and not quality_only and not quality_hybrid and prompt:
            prompt, mag_suggested, rw_target, rw_global, rewrote, rw_scene_regeneration, rw_regions, rw_subject = self._apply_prompt_rewriter(
                prompt,
                img_context=img_description,
                magnitude=mag_suggested,
                body_transform=body_transform,
            )
            if rewrote:
                # El plan de regiones es el único referente visual concreto que
                # entrega el rewriter. Debe sobrevivir hasta la construcción de
                # la máscara focal; antes se imprimía correctamente pero se
                # perdía aquí y CLIPSeg recibía el genérico "subject".
                affected_regions = list(rw_regions or [])
                # v5.1127 — el LLM 0.5B es INESTABLE: la misma frase ('the boy is
                # bald') dio mag=0.9 (log 11:37) y mag=0.1 + target=background
                # (log 12:11). El routing no puede depender de su humor. Si el
                # analyzer determinista NO vio señal (_zero_signal), el rewriter
                # no puede ABAJAR el régimen (mag ya está en 0.90 desde el bloque
                # anterior y _apply_prompt_rewriter toma max) ni redirigir a
                # background (el 0.5B alucina targets).
                if _zero_signal:
                    rw_target = "subject"
                    rw_global = True
                if rw_target in ("subject", "background", "clothes", "face", "hair"):
                    # v5.1127b — el 0.5B alucina targets (log 12:11: 'the boy is
                    # bald' → background = copy machine de fondo). El analyzer
                    # determinista ya decidió el target con scores: si dijo
                    # subject SIN señal structural, el LLM no puede redirigir a
                    # background. En la dirección contraria (LLM dice subject,
                    # analyzer fondo) tampoco hay redirección a la inversa:
                    # gana el determinista salvo señal structural.
                    if rw_target != _det_target_pre and not _det_structural_pre and _det_target_pre != "background":
                        rw_target = _det_target_pre
                    mask_target = rw_target
                    background_focus = mask_target == "background" and not body_transform
                if rw_global or high_structural:
                    is_global = True
                    background_focus = False
                if rw_scene_regeneration:
                    # Clasificación semántica del LLM: la geometría de partida no
                    # puede conservarse. Reutilizamos el régimen pose/POV ya
                    # probado (Full + bypass) sin inspeccionar palabras del prompt.
                    requires_scene_regeneration = True
                    pose_change = True
                    is_global = True
                    background_focus = False
                    print("[ImgEditor] Rewriter semántico: recomposición de escena requerida → Full sin ancla", flush=True)
                if _compound_depth == 0 and mask_image is None:
                    # Una única muestra global de LongCat sobre varios destinos
                    # cambió toda la foto sin obedecer ninguno (run 23:18 del 22-sep).
                    # Separar cláusulas independientes cuando se puedan asignar
                    # visualmente. Si el plan falla, conservar la instrucción
                    # completa y dejar que la ruta normal intente generarla.
                    aligned = self._align_compound_regions(
                        prompt_pre_rewrite, rw_regions, rw_subject
                    )
                    if len(aligned) >= 2:
                        print(f"[ImgEditor] Plan multi-región localizado: {aligned}", flush=True)
                        current = img
                        completed = 0
                        skipped = []
                        clause_warnings = []
                        last_mask = None
                        for clause_index, (clause, region_hint) in enumerate(aligned, 1):
                            if cancel_check and cancel_check():
                                return (current if completed else None), "Cancelado", last_mask
                            edited, clause_msg, clause_mask = self.generate_intelligent(
                                image=current, prompt=clause, negative_prompt=negative_prompt,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale, seed=seed,
                                use_rewriter=use_rewriter, use_semantic=use_semantic,
                                ref_metadata=ref_metadata, engine=engine,
                                enhance_faces=False, lora_name=lora_name,
                                lora_strength=lora_strength, denoise=denoise,
                                progress_callback=progress_callback, auto_upscale=False,
                                quality_mode=False, enhance_tier=enhance_tier,
                                face_preserve=face_preserve, cancel_check=cancel_check,
                                _compound_depth=1,
                                _compound_region_hint=str(region_hint or ""),
                            )
                            if edited is None:
                                note = f"Edición parcial ({completed}/{len(aligned)} regiones): {clause_msg}"
                                print(f"[ImgEditor] {note}", flush=True)
                                return (current if completed else None), note, last_mask
                            if clause_msg.startswith("Omitida:"):
                                skipped.append(clause)
                                print(f"[ImgEditor] {clause_msg}", flush=True)
                                continue
                            for part in str(clause_msg).split(" | "):
                                part = part.strip()
                                if part.startswith("Aviso") or part.startswith("Verificación visual: no confirmada"):
                                    clause_warnings.append(f"cláusula {clause_index}: {part}")
                            current, last_mask = edited, clause_mask
                            completed += 1
                        note = f"Edición localizada de {completed}/{len(aligned)} regiones completada"
                        if skipped:
                            note += f" | {len(skipped)} región(es) sin máscara segura, conservadas sin cambios"
                        if clause_warnings:
                            note += " | " + " | ".join(clause_warnings)
                        try:
                            from roop.img_editor.instruction_verifier import get_instruction_verifier
                            verification = get_instruction_verifier().verify(current, prompt_pre_rewrite)
                            self._last_instruction_compliance = verification
                            if verification.get("compliant") is False:
                                note += " | Verificación del conjunto: no confirmada"
                        except Exception as verify_error:
                            print(f"[ImgEditor] Verificación del conjunto omitida: {verify_error}", flush=True)
                        return current, note, last_mask
                    if len(rw_regions) >= 2:
                        # The small rewriter sometimes labels an appearance
                        # attribute as a second visual region ("man", "bald").
                        # A single clause with one grounded subject is still
                        # one local edit, not a compound whole-frame request.
                        _single_subject = (
                            self._leading_visual_subject(prompt_pre_rewrite, rw_subject)
                            if (not self._split_compound_instruction(prompt_pre_rewrite)
                                and appearance_score >= 0.08
                                and mask_target in ("subject", "face", "hair")
                                and not body_transform and not high_structural
                                and not angle_change and not pose_change)
                            else ""
                        )
                        if _single_subject:
                            affected_regions = [_single_subject]
                            is_global = False
                            print(
                                f"[ImgEditor] Atributos del rewriter descartados como regiones; "
                                f"edición focal de {_single_subject}", flush=True,
                            )
                        else:
                            print(
                                "[ImgEditor] Plan multi-región no separable; "
                                "continuando con la instrucción completa por la ruta normal "
                                f"(regiones={rw_regions})",
                                flush=True,
                            )
                            # Las regiones del plan no describen un único destino.
                            # Evitar que el fallback de una cara limite toda la
                            # instrucción compuesta a una máscara facial.
                            is_global = True
                            background_focus = False
                if mag_suggested >= 0.55:
                    quality_primary = False
                # body_transform is exclusively an image+semantic property.  A
                # text-only rewriter must not recategorise an ordinary edit as a
                # body regeneration merely because it emitted a large scalar.

            # v5.1110: si el usuario escribió en otro idioma, el TEXTO
            # reescrito se descarta SIEMPRE (el 0.5B mangla/omite mitades al
            # traducir: logs 11:40, 11:57, 12:12) pero mag/target se
            # conservan. Se usa el traducido intacto (mappings + MyMemory),
            # que ya es inglés completo. Ver método.
            # El idioma se mira en prompt_raw: el texto ya traducido-local
            # (mezclado) puede colar como "inglés" aunque la API fallara.
            if rewrote and self._discard_rewrite_text(prompt_raw, body_direction, engine):
                if self._rewriter_repairs_short_translation(prompt_pre_rewrite, prompt, prompt_raw):
                    print(
                        "[ImgEditor] Rewrite-text aceptado: reparó token dañado en una instrucción breve "
                        "sin perder contenido",
                        flush=True,
                    )
                else:
                    print("[ImgEditor] Rewrite-text descartado (entrada no-EN): traducción completa intacta", flush=True)
                    prompt = prompt_pre_rewrite
            if rewrote and _compound_depth > 0:
                # El plan de regiones sí es útil, pero el 0.5B amplificó una
                # petición de color hasta "no texture left". Cada pase conserva
                # la cláusula exacta que el usuario pidió.
                prompt = prompt_pre_rewrite
                print("[ImgEditor] Subedición: texto original de la cláusula conservado", flush=True)

        if _compound_depth > 0 and _compound_region_hint.strip():
            # El plan global ya identificó una región por cláusula. Un segundo
            # pase del LLM pequeño puede olvidar esa región (incluso devolver []),
            # pero no debe convertir por ello el pase local en una edición global.
            affected_regions = [_compound_region_hint.strip()]
            is_global = False
            print(f"[ImgEditor] Región heredada del plan compuesto: {affected_regions[0]}", flush=True)

        if (
            _compound_depth == 0
            and not affected_regions
            and not quality_only
            and not quality_hybrid
            and not is_global
            and not body_transform
            and not high_structural
            and not angle_change
            and not pose_change
            and mask_target in ("subject", "face", "hair")
            and appearance_score >= 0.20
            and mag_suggested >= 0.65
        ):
            # The planner has repeatedly returned regions=[] for a single
            # person's appearance edit. Query the subject named in the intact
            # translated instruction, then require a unique visual face match.
            # This prevents a whole-frame pass from merely copying the hair.
            grounded_subject = self._leading_visual_subject(prompt_pre_rewrite, rw_subject)
            if grounded_subject:
                affected_regions = [grounded_subject]
                print(
                    f"[ImgEditor] Región omitida por el rewriter; sujeto textual: {grounded_subject}",
                    flush=True,
                )

        # Calidad pura no es una edición: debe conservar cada píxel de contenido
        # salvo las operaciones clásicas de restauración.  El pase LongCat se
        # desactivó para este camino porque puede reinterpretar caras/objetos y,
        # además, cargar el modelo pesado hacía que la siguiente edición pagase
        # un coste de VRAM innecesario.  El híbrido mantiene el pulido generativo
        # salvo que el llamador lo desactive explícitamente.
        quality_generation_enabled = bool(
            quality_hybrid
            and not quality_primary
            and quality_use_generative is not False
            # The edit itself has already run LongCat. A second LongCat polish
            # reloads the model and may change requested content. Keep it only
            # when the caller explicitly asks for that extra generative pass.
            and (quality_use_generative is True
                 or engine not in ("longcat", "imagine", "longcat_full"))
        )
        if quality_only or quality_primary:
            print(
                "[ImgEditor] Calidad pura: pipeline clásico medido, sin LongCat ni edición de contenido",
                flush=True,
            )

        analysis = {
            "prompt": prompt,
            "magnitude": mag_suggested,
            "mask_target": mask_target,
            "is_global": is_global,
            "quality_only": quality_only,
            "quality_enhance": quality_enhance,
            "quality_hybrid": quality_hybrid,
            "quality_primary": quality_primary,
            "quality_enhance_style": quality_style,
            "quality_use_generative": quality_generation_enabled,
            "quality_preserve_faces": bool(quality_preserve_faces),
            "quality_steps": quality_steps,
            "has_quality_request": has_quality_request or quality_enhance,
            "enhance_tier": enhance_tier,
            "structural_score": structural_score,
            "high_structural": high_structural,
            "structural_bias": structural_bias,
            "angle_change": angle_change,
            "pose_change": pose_change,
            "requires_scene_regeneration": requires_scene_regeneration,
            "multi_region_edit": multi_region_edit,
            "affected_regions": affected_regions,
            "appearance_score": appearance_score,
            "axis_scores": axis_scores,
            "body_transform_intensity": body_transform_intensity,
            "body_transform": body_transform,
            "body_direction": body_direction,
            "background_focus": background_focus,
            "global_scene_realism": global_scene_realism,
        }

        # 3. Resolución de Parámetros basada en el Análisis
        params = self.auto_detect_params(analysis, engine)
        mask_target = str(analysis.get("mask_target", "subject")).lower()
        is_global = bool(analysis.get("is_global", False))

        if num_inference_steps:
            params["num_inference_steps"] = num_inference_steps
        if guidance_scale:
            params["guidance_scale"] = guidance_scale
        if denoise:
            params["denoise"] = denoise

        if quality_only and not negative_prompt:
            negative_prompt = (
                "blurry, low quality, noise, jpeg artifacts, soft focus, out of focus, "
                "pixelated, washed out, oversmoothed, plastic skin"
            )
        elif high_structural and not negative_prompt:
            if structural_bias == "remove":
                negative_prompt = (
                    "people visible, person in frame, leftover figure, ghost silhouette, "
                    "incomplete removal, duplicate person, partial body, blurry patch"
                )
            elif structural_bias == "add":
                negative_prompt = (
                    "floating figure, cutout collage, mismatched lighting, pasted person, "
                    "wrong scale, extra limbs, blurry face, bad integration"
                )
        elif body_transform and not negative_prompt:
            # v5.1093: negative consciente de la DIRECCIÓN (semántica, no palabras).
            # uncover (quitar) → negar prendas que tapen. cover/change (vestir de
            # nieve, cambiar outfit...) → negative neutro para no pelear con el pedido.
            if str(analysis.get("body_direction", "change")) == "uncover":
                negative_prompt = (
                    "fully clothed, dress, one-piece swimsuit, covered torso, "
                    "leftover garment, censored, covered skin"
                )
            else:
                negative_prompt = (
                    "blurry, low quality, distorted, leftover garment, "
                    "incomplete clothing change, mismatched outfit"
                )

        has_quality_request = bool(analysis.get("has_quality_request", False))
        high_structural = bool(analysis.get("high_structural", False))
        structural_bias = str(analysis.get("structural_bias", "neutral"))
        if high_structural and structural_bias == "remove":
            has_quality_request = False
        analysis["has_quality_request"] = has_quality_request
        mask_target = str(analysis.get("mask_target", "subject")).lower()
        # El engine puede ser override a longcat_full más adelante (body_transform).
        # Para el prompt, usar longcat_full si body_transform para que se aplique
        # el wrapper de preservación (is_longcat check).
        effective_engine = "longcat_full" if (body_transform and engine in ("klein", "klein_base", "klein_9b")) else engine
        prompt_enhanced = self._compose_generation_prompt(
            prompt, img_context=img_description, engine=effective_engine, magnitude=mag_suggested,
            quality_only=quality_only, has_quality_request=has_quality_request,
            enhance_tier=enhance_tier, high_structural=high_structural,
            structural_bias=structural_bias, body_transform=body_transform,
            multi_region_edit=bool(analysis.get("multi_region_edit", False)),
            mask_target=mask_target, quality_hybrid=quality_hybrid,
            detected_age=detected_age, shot_type=shot_type,
            body_direction=str(analysis.get("body_direction", "change")),
            body_intensity=float(analysis.get("body_transform_intensity", 0.0)),
            angle_change=bool(analysis.get("angle_change", False)),
            pose_change=bool(analysis.get("pose_change", False)),
        )
        is_global = bool(analysis.get("is_global", False))
        mag_suggested = float(analysis.get("magnitude", mag_suggested))
        # POV: añadir descriptores de vista para guiar la regeneración
        prompt_enhanced = self._apply_pov_hint(prompt_enhanced, analysis, quality_only)

        print(f"[ImgEditor] Target: {mask_target} (Global: {is_global})", flush=True)
        print(f"[ImgEditor] Prompt final: {prompt_enhanced}", flush=True)
        if quality_only:
            print(
                f"[ImgEditor] Pipeline TOP: desposterizar + restaurar caras + "
                f"upscale adaptativo tier={enhance_tier} (sin LongCat)"
            )
        elif has_quality_request and mag_suggested > 0.6:
            print("[ImgEditor] Compound prompt detected (main change + quality/color polish)")
        elif body_transform:
            print(
                f"[ImgEditor] Cambio corporal/ropa (semántico, intensity={body_transform_intensity:.2f}, "
                f"dir={body_direction}) → global puro en {engine} (FacePreserver OFF por preferencia)"
            )
        elif background_focus:
            print(
                "[ImgEditor] Edición de fondo focalizada — máscara clipseg, denoise moderado, "
                "sujetos preservados"
            )

        # No automatic anti-clothing negative anymore.
        # It was fighting requests like "ropa raída con mangas".
        # Prompt del usuario + análisis semántico local + preservación mag-dependent.
        # This way it adapts to ANY clothing request (add, modify, remove, specific style) without lists or hardcodes.
        # The semantic + preservation text is enough to guide the model.

        final_mask = mask_image
        _compound_scene_background = False
        _scene_head_preserve_mask = None
        _compound_foreground_mask = None
        _compound_generated_reference = None
        _compound_measurement_mask = None
        _compound_strict_foreground_mask = None
        # Una máscara pintada por el usuario define una edición focal. No es una
        # pista semántica: es una restricción espacial explícita y debe ganar a
        # cualquier router que elija motores EDIT globales sin soporte inpaint.
        manual_focal_mask = final_mask is not None and str(mask_mode).lower() == "manual"

        if high_structural and structural_bias == "remove" and body_transform and final_mask is None:
            # Compuesto cuerpo+quitar: el cuerpo necesita regeneración GLOBAL
            # (bypass no_kontext). Enmascarar a la persona pelearía con el nude
            # y dejaría el objeto intacto fuera de máscara (log 19:54: cov=64.7%
            # persona, perro intacto). Se va a global puro con prompt combinado.
            print("[ImgEditor] Compuesto body+remove → global puro (sin máscara persona; el bypass regenera la escena sin el objeto)")
        elif high_structural and structural_bias == "remove" and final_mask is None:
            final_mask = self._build_object_removal_mask(img)
            if final_mask is None:
                final_mask = self._build_person_removal_mask(img)
            if final_mask is None:
                print("[ImgEditor] Structural remove: sin máscara objeto/persona — global agresivo")
        elif high_structural and structural_bias == "add" and final_mask is None:
            # ADD sin máscara — el prompt + LongCat global da mejor resultado que mask+inpaint
            print("[ImgEditor] Structural add: edit global (sin máscara, el prompt guía todo)")
            pass  # final_mask se queda None → cae en force_global abajo

        structural_inpaint = (
            high_structural and structural_bias in ("remove", "add") and final_mask is not None
        )

        # Intentar máscara de ropa si los ejes semánticos indican cambio de atributo (ropa/color)
        # y NO cambio de pose. Así la decisión es semántica, sin hardcoding de targets.
        # REGLA de cobertura (ver bloque do_cloth_mask abajo): la máscara solo se acepta
        # si cubre una fracción real de la imagen (>=12% body / >5% attr). Si CLIPSeg
        # apenas detecta ropa (planos generales, det 352px, personaje pequeño → 3-6%),
        # se descarta y se usa edición GLOBAL (img2img puro), que es el diseño original:
        # el modelo regenera el cuerpo completo con piel/tono consistentes, sin parches.
        skip_focal_cloth = (
            global_scene_realism
            or quality_primary
            or (quality_hybrid and is_global and has_quality_request)
        )
        # Undress/body fuerte: SOLO si la máscara de ropa resulta fiable (cobertura >=12%)
        # se enmascara la ropa y se inpaintsa esa zona (el resto de la foto queda intacto).
        # Con cobertura baja el inpaint focal pinta un parche diminuto y la edición
        # "no hace nada" o rompe la zona → descartar y caer a global.
        do_cloth_mask = False
        cloth_reason = ""
        cloth_mask_rejected = False
        if final_mask is None and not skip_focal_cloth:
            axes = analysis.get("axis_scores", {})
            attr_score = axes.get("attribute", 0)
            pose_score = axes.get("pose", 0)
            if body_transform and mag_suggested >= 0.62 and pose_score <= attr_score:
                # Body/undress fuerte: NUNCA máscara focal de ropa.
                # Un inpaint focal pinta gris sobre el cuerpo — necesita GLOBAL
                # (mask=None) para que el modelo regeneré cuerpo completo con piel.
                do_cloth_mask = False
                cloth_mask_rejected = True
                cloth_reason = f"body mag={mag_suggested:.2f} → global (skip cloth mask)"
                print(f"[ImgEditor] Body transform: skip cloth mask ({cloth_reason})")
            elif 0.45 <= mag_suggested < 0.62 and not body_transform:
                if attr_score > pose_score or (attr_score > 0 and pose_score == 0):
                    do_cloth_mask = True
                    cloth_reason = f"attr={attr_score:.3f} > pose={pose_score:.3f}"
        if do_cloth_mask:
            try:
                seg = self._get_clothing_segmenter()
                if seg:
                    cloth_mask, _ = seg.segment_clothing(img, threshold=0.45)
                    cov = self._mask_coverage(cloth_mask) if cloth_mask is not None else 0.0
                    # Umbral de aceptación según tipo de cambio:
                    #  - Cambio moderado (attr): basta >5% de la imagen (prenda real
                    #    localizada: camisa, vestido, etc.).
                    #  - Body/undress fuerte (body_transform): CLIPSeg va a 352px y en
                    #    planos generales/fullbody casi no detecta ropa (3-6%). Con un
                    #    umbral bajo la máscara-basura se acepta y el inpaint focal pinta
                    #    un parche diminuto → la edición "no hace nada" o rompe esa zona.
                    #    Exigimos cobertura >=12% (ropa real muy visible); si no llega,
                    #    edición GLOBAL (img2img puro con bypass Kontext), que es el
                    #    diseño original: el modelo regenera el cuerpo sin parches.
                    min_cov = 0.12 if body_transform else 0.05
                    if cloth_mask is not None and cov > min_cov:
                        final_mask = cloth_mask
                        print(f"[ImgEditor] Máscara de ropa para edit focalizado ({cloth_reason}, cobertura={cov:.1%})")
                    else:
                        cloth_mask_rejected = True
                        destino = "GLOBAL (img2img puro)" if body_transform else "global (sin máscara)"
                        print(
                            f"[ImgEditor] Máscara de ropa insuficiente "
                            f"(cobertura={cov:.1%}, mínimo {min_cov:.0%}) → {destino}"
                        )
            except Exception as e:
                print(f"[ImgEditor] ClothingSegmenter no disponible: {e}")

        # force_global solo cuando NO hay máscara y es cambio estructural o quality_hybrid sin máscara exitosa.
        # La máscara de ropa ya se intentó arriba (incluido undress/body); si existe, se usará
        # como inpaint: solo la ropa se repinta, el resto de la foto queda intacto.
        force_global = final_mask is None and not background_focus and (
            high_structural
            or (quality_hybrid and has_quality_request and not quality_primary)
        )

        use_flux_structural_inpaint = bool(structural_inpaint and structural_bias == "remove")
        _head_only_plan = self._head_only_plan(
            analysis, mask_target, body_transform or _weak_compound_body,
            cloth_mask_rejected, high_structural,
        )

        if use_flux_structural_inpaint:
            print("[ImgEditor] Structural remove → Flux Fill inpaint (máscara, sin Kontext)")
        elif structural_inpaint:
            bias_label = structural_bias.upper()
            print(f"[ImgEditor] Structural {bias_label}: inpaint LongCat + denoise alto")
        elif force_global:
            reason = "cambio estructural (ancla)" if high_structural else "alta magnitud"
            bias_note = f", bias={structural_bias}" if high_structural else ""
            ref_label = "offset" if not high_structural else "index"
            print(f"[ImgEditor] Edición global ({reason}{bias_note}) — ref {ref_label}, denoise alto")
            final_mask = None
        elif final_mask is None and (
            # El plan de una sola persona solo autoriza crop de cabeza si el
            # análisis describe una edición de apariencia aislada. Un cambio
            # corporal o de ropa necesita ver la zona fuera de la cabeza.
            _head_only_plan
            or (not is_global and not body_transform and not cloth_mask_rejected)
        ):
            # v5.1114 — EDIT engines (kontext/klein) generan la imagen COMPLETA
            # desde latente vacío con la foto como conditioning: la máscara
            # pixel no existe en su workflow (el cliente ya la ignora al
            # encolar), pero el composite post-generación SÍ la usa y repinta
            # la foto ORIGINAL dentro de ella → la edición se deshace en esa
            # zona. En engines EDIT no tiene sentido máscara focal: es global
            # por construcción. Decisión por capacidad del motor, sin palabras.
            _edit_engine = engine in ("kontext", "klein", "klein_base", "klein_9b")
            if _edit_engine:
                print(f"[ImgEditor] Motor EDIT ({engine}): máscara focal sin sentido (workflow sin máscara) → global")
            else:
                try:
                    mask_query = mask_target
                    _regions_for_mask = analysis.get("affected_regions", [])
                    _focal_head_plan = _head_only_plan
                    # Una región única + un cambio de apariencia física se
                    # resuelve de forma visual: CLIPSeg localiza la región y la
                    # máscara resultante se reduce a la cabeza que más solapa.
                    # No depende de nombres de personas, géneros ni posiciones.
                    if isinstance(_regions_for_mask, list) and len(_regions_for_mask) == 1:
                        _region_query = str(_regions_for_mask[0] or "").strip()
                        if _region_query:
                            mask_query = _region_query
                            print(f"[ImgEditor] Auto-máscara focal desde plan de región: {mask_query}")
                    print(f"[ImgEditor] Intentando auto-máscara para: {mask_query}")
                    masker = self._get_clipseg_masker()
                    _scene_location_request = (
                        _compound_depth > 0 and self._is_scene_location_clause(prompt)
                    )
                    # A destination setting may not exist in the source image;
                    # segmenting its name (e.g. a new room) cannot locate the
                    # pixels to replace. Use the people-protected background.
                    mask_queries = [] if _scene_location_request else [mask_query]
                    _visual_subject_mask = None
                    _visual_subject_query = ""
                    if not _scene_location_request and _compound_depth > 0:
                        _visual_subject_query = self._visual_subject_description(prompt)
                        if _visual_subject_query:
                            _visual_subject_mask = masker.generate_mask(img, _visual_subject_query)
                            if _visual_subject_mask is not None:
                                print(
                                    f"[ImgEditor] Sujeto descrito localizado en la imagen: "
                                    f"'{_visual_subject_query}'",
                                    flush=True,
                                )
                                if _focal_head_plan:
                                    mask_queries.insert(0, _visual_subject_query)
                    if not _scene_location_request and _compound_depth > 0 and _compound_region_hint.strip():
                        # Una cláusula ausente del plan LLM se usa como consulta
                        # visual completa. Si CLIPSeg no la entiende, probar sus
                        # términos por separado sin asumir clases de objetos.
                        terms = re.findall(r"\w+", mask_query)
                        if len(terms) == 2:
                            # En frases de dos términos como "gray walls", el
                            # segundo suele nombrar el objeto. El prompt entero
                            # puede producir una máscara diminuta pero válida.
                            mask_queries = [terms[-1], mask_query, terms[0]]
                        elif len(terms) > 2:
                            mask_queries.extend(reversed(terms))
                    auto_mask = None
                    _preselected_head_mask = None
                    for query in dict.fromkeys(mask_queries):
                        candidate = (_visual_subject_mask if query == _visual_subject_query
                                     and _visual_subject_mask is not None else masker.generate_mask(img, query))
                        if candidate is None:
                            continue
                        if _focal_head_plan:
                            candidate_head = self._head_mask_from_region_plan(
                                img, source_faces, candidate,
                            )
                            if candidate_head is None:
                                continue
                            _preselected_head_mask = candidate_head
                        elif _visual_subject_mask is not None:
                            # Validate the edited object against the person
                            # described in the photo. A whole-person mask is
                            # too broad for an attribute-level subedit.
                            candidate_arr = np.asarray(candidate.convert("L")) >= 128
                            subject_arr = np.asarray(_visual_subject_mask.convert("L")) >= 96
                            if not candidate_arr.any():
                                continue
                            near_subject = cv2.dilate(
                                subject_arr.astype(np.uint8),
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)),
                            ) > 0
                            inside = float((candidate_arr & near_subject).sum()) / float(candidate_arr.sum())
                            relative_size = float(candidate_arr.sum()) / max(1.0, float(subject_arr.sum()))
                            if inside < 0.30 or relative_size > 0.80:
                                print(
                                    f"[ImgEditor] Región '{query}' no corresponde a un atributo "
                                    f"local del sujeto (solape={inside:.2f}, tamaño={relative_size:.2f})",
                                    flush=True,
                                )
                                continue
                        auto_mask = candidate
                        mask_query = query
                        break
                    if auto_mask:
                        if _focal_head_plan:
                            final_mask = _preselected_head_mask
                            if final_mask is not None:
                                # La geometría facial ancla a la persona; la
                                # segmentación visual completa el contorno de
                                # la región que esa geometría puede recortar.
                                head_region = masker.generate_mask(img, "head")
                                hair_region = masker.generate_mask(img, "hair")
                                final_mask = self._complete_localized_region_mask(
                                    final_mask,
                                    self._union_grayscale_masks(head_region, hair_region),
                                )
                                if self._is_bald_request(prompt_pre_rewrite):
                                    hair_mask = hair_region
                                    head = np.asarray(final_mask.convert("L")) >= 128
                                    hair = (np.asarray(hair_mask.convert("L")) >= 96
                                            if hair_mask is not None else None)
                                    ys = np.flatnonzero(head.any(axis=1))
                                    if hair is not None and ys.size:
                                        scalp = head.copy()
                                        scalp[int(ys[0] + 0.60 * (ys[-1] - ys[0])) + 1:, :] = False
                                        selected_hair = float(hair[scalp].mean()) if scalp.any() else 0.0
                                        if selected_hair < 0.04:
                                            alternate = self._unique_subject_hair_mask(img, auto_mask, hair_mask)
                                            if alternate is not None:
                                                final_mask = alternate
                                                _hair_subject_no_face = True
                                                print(
                                                    "[ImgEditor] Cara detectada ya calva; pelo único dentro "
                                                    "del sujeto localizado para edición focal",
                                                    flush=True,
                                                )
                                print(f"[ImgEditor] Máscara de cabeza focal GENERADA para '{mask_query}'")
                            else:
                                # Una máscara regional no identifica por sí sola qué
                                # cabeza debe cambiar. Con varias personas, usarla
                                # como fallback puede editar a la persona equivocada.
                                print(f"[ImgEditor] Región '{mask_query}' sin cabeza inequívoca; máscara focal omitida")
                        else:
                            final_mask = self._complete_localized_region_mask(
                                auto_mask, auto_mask,
                            )
                            _compound_measurement_mask = auto_mask
                            print(f"[ImgEditor] Máscara automática GENERADA para '{mask_query}'")
                            if _compound_depth > 0 and _compound_region_hint.strip():
                                # CLIPSeg's person query misses long hair in this
                                # scene. Protect all measured foreground before
                                # broadening a surface mask or its color finish.
                                from PIL import ImageChops, ImageFilter
                                person_mask = masker.generate_mask(img, "person")
                                human_mask = masker.generate_mask(img, "human")
                                woman_mask = masker.generate_mask(img, "woman")
                                man_mask = masker.generate_mask(img, "man")
                                hair_mask = masker.generate_mask(img, "hair")
                                head_mask = masker.generate_mask(img, "head")
                                if head_mask is not None:
                                    head_mask = head_mask.filter(ImageFilter.MaxFilter(11))
                                face_geometry = np.zeros((img.height, img.width), dtype=np.uint8)
                                for face_obj in source_faces:
                                    x1, y1, x2, y2 = [float(value) for value in face_obj.bbox]
                                    fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                                    cv2.ellipse(
                                        face_geometry,
                                        (int(round((x1 + x2) * 0.5)), int(round(y1 + fh * 0.42))),
                                        (int(round(fw * 0.58)), int(round(fh * 0.65))),
                                        0, 0, 360, 255, -1,
                                    )
                                face_mask = (
                                    Image.fromarray(face_geometry, "L")
                                    if face_geometry.any() else None
                                )
                                strict_parts = [
                                    part for part in (hair_mask, head_mask, face_mask)
                                    if part is not None
                                ]
                                if strict_parts:
                                    _compound_strict_foreground_mask = strict_parts[0]
                                    for part in strict_parts[1:]:
                                        _compound_strict_foreground_mask = ImageChops.lighter(
                                            _compound_strict_foreground_mask, part
                                        )
                                protection_parts = [
                                    part for part in (
                                        person_mask, human_mask, woman_mask, man_mask,
                                        hair_mask, head_mask, face_mask,
                                    )
                                    if part is not None
                                ]
                                if protection_parts:
                                    _compound_foreground_mask = protection_parts[0]
                                    for part in protection_parts[1:]:
                                        _compound_foreground_mask = ImageChops.lighter(
                                            _compound_foreground_mask, part
                                        )
                                    if hair_mask is not None:
                                        _compound_foreground_mask = ImageChops.lighter(
                                            _compound_foreground_mask,
                                            hair_mask.filter(ImageFilter.MaxFilter(15)),
                                        )
                                    core = np.asarray(final_mask.convert("L")) >= 180
                                    foreground = np.asarray(_compound_foreground_mask.convert("L")) >= 96
                                    overlap = float((core & foreground).sum()) / max(1, int(core.sum()))
                                    # A lower CLIPSeg threshold fills thin or
                                    # softly lit surfaces. A candidate may cross
                                    # a small foreground edge, so measure its
                                    # raw overlap and remove those pixels before
                                    # accepting it. Reject masks that mostly
                                    # describe a person instead of the object.
                                    if overlap < 0.20:
                                        for threshold in (0.15, 0.20, 0.25):
                                            expanded = masker.generate_mask(
                                                img, mask_query, threshold=threshold
                                            )
                                            if expanded is None:
                                                continue
                                            expanded_pixels = np.asarray(expanded.convert("L")).copy()
                                            expanded_core = expanded_pixels >= 180
                                            raw_overlap = float((expanded_core & foreground).sum()) / max(
                                                1, int(expanded_core.sum())
                                            )
                                            if raw_overlap >= 0.10:
                                                continue
                                            expanded_pixels[foreground] = 0
                                            clipped_coverage = float((expanded_pixels >= 180).mean())
                                            if (core.mean() * 1.12 <= clipped_coverage
                                                    <= min(core.mean() * 2.5, 0.35)):
                                                final_mask = Image.fromarray(expanded_pixels, "L")
                                                print(
                                                    f"[ImgEditor] Máscara regional ampliada: "
                                                    f"{core.mean():.1%}→{clipped_coverage:.1%} "
                                                    f"(solape recortado={raw_overlap:.1%})",
                                                    flush=True,
                                                )
                                                break
                                        else:
                                            pixels = np.asarray(final_mask.convert("L")).copy()
                                            pixels[foreground] = 0
                                            final_mask = Image.fromarray(pixels, "L")
                except Exception as e:
                    print(f"[ImgEditor] Error en máscara: {e}")

                if (final_mask is None and _compound_depth > 0
                        and self._is_scene_location_clause(prompt)):
                    try:
                        final_mask = self._background_mask_from_people(
                            img, self._get_clipseg_masker(), source_faces,
                        )
                    except Exception as scene_mask_error:
                        print(f"[ImgEditor] Protección de personas no disponible: {scene_mask_error}", flush=True)
                    if final_mask is None:
                        return img, (
                            "Omitida: cambio de entorno sin máscara de personas fiable; "
                            "se conserva la edición anterior"
                        ), None
                    _compound_scene_background = True
                    background_focus = True
                    is_global = False
                    mask_target = "background"
                    scene_clause = re.sub(r"^(?:and|y)\s+", "", prompt.strip(), flags=re.I)
                    prompt_enhanced = (
                        "Instruction: Replace the visible setting and background of this exact photo "
                        f"so that {scene_clause}. Keep the same people, faces, bodies, clothes, "
                        "poses and camera framing. Change the surroundings only."
                    )
                    print(
                        f"[ImgEditor] Cambio de entorno: fondo localizado "
                        f"({self._mask_coverage(final_mask):.1%}); personas protegidas",
                        flush=True,
                    )

                # Si el grounding textual no puede segmentar una región concreta,
                # no se adivina una persona por tamaño, sexo o posición. Se abre
                # como zona editable el conjunto de cabezas detectadas y el modelo
                # de edición resuelve el referente desde la foto + el prompt.
                # Esto mantiene el flujo automático y preserva el resto de imagen.
                if (
                    final_mask is None
                    and locals().get("_focal_head_plan", False)
                    and len(source_faces) == 1
                ):
                    final_mask = self._head_mask_from_faces(img, source_faces)
                    if final_mask is not None:
                        print(
                            "[ImgEditor] Grounding textual no concluyente → "
                            "única cabeza candidata para edición guiada por prompt",
                            flush=True,
                        )

                # Fallback sin hardcoding: si no hay máscara y el target es una zona de la persona,
                # usar detección facial real para generar una máscara elíptica aproximada.
                # No depende de palabras del prompt, solo de geometría de la imagen.
                if (final_mask is None and not locals().get("_focal_head_plan", False)
                        and not (_compound_depth > 0 and _compound_region_hint.strip())
                        and mask_target in ("subject", "clothes", "face", "hair")):
                    try:
                        face_mask = self._create_mask_from_face_detection(img, target=mask_target)
                        if face_mask is not None:
                            final_mask = face_mask
                            print(f"[ImgEditor] Máscara facial/geometrica generada para '{mask_target}' (fallback deteccion)")
                    except Exception as e:
                        print(f"[ImgEditor] Error en máscara facial: {e}")

        if (final_mask is None and _compound_depth > 0
                and _compound_region_hint.strip() and not _compound_scene_background
                and not high_structural):
            note = (
                "Omitida: la región descrita no se pudo confirmar en la imagen; "
                "se conserva la edición anterior"
            )
            print(f"[ImgEditor] {note}", flush=True)
            return img, note, None

        # LongCat Image Edit no interpreta de forma fiable el alpha de una
        # máscara facial automática: en vez de regenerar el área deja el valor
        # de relleno de VAE (elipse gris). La máscara sigue siendo útil para
        # localizar que hay una región focal, pero el workflow estable de
        # LongCat es su referencia visual `index` sin nodo de inpaint. Esto se
        # decide por procedencia/geometry de máscara, no por el contenido de la
        # instrucción; las máscaras manuales y los inpaints estructurales siguen
        # conservando su ruta explícita.
        _longcat_focal_reference = bool(
            final_mask is not None
            and not manual_focal_mask
            and not body_transform
            and not cloth_mask_rejected
            and locals().get("_focal_head_plan", False)
            and not structural_inpaint
            and engine in ("longcat", "longcat_full", "imagine")
        )
        if _longcat_focal_reference:
            _longcat_focal_mask = final_mask
            final_mask = None
            print(
                "[ImgEditor] Región facial localizada; LongCat usará referencia visual index sin inpaint "
                "(evita relleno gris)",
                flush=True,
            )

        # Una petición focal sin región inequívoca no autoriza a regenerar la
        # escena entera: eso cambió personas, texturas y calidad en fotos reales.
        if (final_mask is None and locals().get("_focal_head_plan", False)
                and not _longcat_focal_reference):
            note = (
                "Omitida: no se pudo localizar de forma inequívoca la región "
                "solicitada; se conserva la imagen"
            )
            print(f"[ImgEditor] {note}", flush=True)
            return img, note, None

        if _compound_depth > 0 and _compound_region_hint.strip() and final_mask is None and not _longcat_focal_reference:
            print(
                "[ImgEditor] Subedición sin máscara fiable; continuando con "
                "referencia visual de la imagen anterior",
                flush=True,
            )

        if final_mask is None and is_global:
             print("[ImgEditor] Edición global detectada (sin máscara)")

        if final_mask is None and (body_transform or cloth_mask_rejected):
             print(
                 "[ImgEditor] Edición GLOBAL (img2img puro) — sin máscara focal "
                 "(solo ropa insuficiente / cambio de cuerpo requiere regenerar el cuerpo)"
             )

        try:
            result = None
            msg = ""
            _local_focal_crop = None
            print(f"[ImgEditor] Iniciando generación con motor: {engine}")

            quality_primary = bool(analysis.get("quality_primary", False))
            if quality_only or quality_primary:
                label = "calidad-primaria" if quality_primary else "mejora UI"
                quality_gen = bool(analysis.get("quality_use_generative", False))
                pipeline_kind = "LongCat realismo" if quality_gen else "clásico medido (sin generación)"
                print(
                    f"[ImgEditor] Pipeline TOP tier={enhance_tier} ({label}) — "
                    f"{pipeline_kind} + upscale IA adaptativo"
                )
                result = img.copy()
                msg = f"Mejora TOP tier={enhance_tier}"
                if progress_callback:
                    progress_callback({
                        "phase": "Mejorando imagen",
                        "progress": 0.04,
                        "detail": f"TOP {enhance_tier} {img.size[0]}×{img.size[1]}",
                    })
            else:
                client = None
                version = None
                flux_structural_override = False

                if use_flux_structural_inpaint and engine in ("longcat", "imagine", "longcat_full", "kontext", "klein", "klein_base", "klein_9b"):
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    from roop.img_editor.icedit_comfy_client import ICEditComfyClient
                    if self.flux_dev_abl_client is None:
                        self.flux_dev_abl_client = get_flux_edit_comfy_client()
                    client = self.flux_dev_abl_client
                    icedit = ICEditComfyClient()
                    fill_path = icedit._resolve_flux_fill(
                        os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "ui", "tob", "ComfyUI", "models",
                        )
                    )
                    version = os.path.basename(fill_path) if os.path.isfile(fill_path) else "flux1-dev-Q2_K.gguf"
                    if not os.path.isfile(fill_path):
                        # Sin Fill en disco: no forzar flujo Flux Fill; cae al
                        # inpaint LongCat con la máscara (denoise alto) en vez
                        # de cargar un dev genérico que no hace inpaint.
                        print(f"[ImgEditor] Flux Fill ausente ({fill_path}) → inpaint LongCat (sin Flux Fill)")
                        flux_structural_override = False
                        from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client as _lc_cli
                        if self.flux_klein_client is None:
                            self.flux_klein_client = _lc_cli()
                        client = self.flux_klein_client
                        version = "LongCat-Image-Edit-Q4_K_S.gguf"
                    else:
                        flux_structural_override = True
                elif engine in ("longcat", "imagine"):
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    mag = analysis.get("magnitude", 0.0)
                    # Full para cambios fuertes/corporales y Turbo para sutiles,
                    # pero sin degradar un Full ya residente: esa alternancia
                    # vaciaba el cache correcto entre ediciones consecutivas.
                    is_body = bool(analysis.get("body_transform", False))
                    version = self._select_longcat_version(client, mag, is_body)
                elif engine == "kontext":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    # v5.1102: Q4_K_M (6.46GB) en vez de Q2_K (3.75GB). El Q2_K
                    # demostró ser una "copy machine": con offset copiaba limpio
                    # (std 56, ropa intacta) y con uxo colapsaba a gris (std 9/18)
                    # — la quant de 2-bit destruyó el instruction-following pero
                    # no la vía de copia. Q4 + T5 GGUF ≈ 9.4GB → offload en 8GB
                    # (más lento, ~4-6min) pero adherencia real.
                    version = "flux1-kontext-dev-Q4_K_M.gguf"
                elif engine == "longcat_full":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    version = "LongCat-Image-Edit-Q4_K_S.gguf"
                    # VRAM check: LongCat Full necesita ~12.3GB. En 8GB, el body router
                    # ya redirige a Kontext/Klein. Para non-body, warning si VRAM justa.
                    try:
                        _vram_lc = get_vram_gb()
                    except Exception:
                        _vram_lc = 0
                    if _vram_lc and _vram_lc < 11 and not body_transform:
                        print(f"[ImgEditor] AVISO: LongCat Full en {_vram_lc}GB VRAM (necesita ~12GB) — puede ser lento")
                elif engine == "flux_dev_abliterated":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_dev_abl_client is None:
                        self.flux_dev_abl_client = get_flux_edit_comfy_client()
                    client = self.flux_dev_abl_client
                    version = "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf"
                elif engine == "qwen_edit":
                    from roop.img_editor.qwen_edit_comfy_client import get_qwen_edit_comfy_client
                    if self.qwen_edit_client is None:
                        self.qwen_edit_client = get_qwen_edit_comfy_client()
                    client = self.qwen_edit_client
                    version = "q2"
                elif engine == "omnigen2":
                    from roop.img_editor.omnigen2_gguf_comfy_client import get_omnigen2_comfy_client
                    if self.omnigen2_client is None:
                        self.omnigen2_client = get_omnigen2_comfy_client()
                    client = self.omnigen2_client
                    version = None
                elif engine == "klein_9b":
                    vram = get_vram_gb()
                    if vram < 12:
                        return None, f"FLUX.2 Klein 9B necesita >=12GB VRAM (detectado {vram:.1f}GB). Usá Klein 4B o LongCat en 8GB."
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    version = "flux-2-klein-9b-Q4_K_S.gguf"
                    # body_transform override manejado por _route_body_engine() más abajo
                elif engine == "klein":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    version = "flux-2-klein-base-4b-Q4_K_S.gguf"
                    # body_transform override manejado por _route_body_engine() más abajo
                elif engine == "klein_base":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    version = "flux-2-klein-base-4b-Q4_K_S.gguf"
                    # body_transform override manejado por _route_body_engine() más abajo
                elif engine == "flux_q2":
                     from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                     if self.flux_dev_abl_client is None:
                         self.flux_dev_abl_client = get_flux_edit_comfy_client()
                     client = self.flux_dev_abl_client
                     version = "flux1-dev-Q2_K.gguf"
                elif engine == "hart":
                     from roop.img_editor.hart_edit_comfy_client import get_hart_edit_comfy_client
                     if not hasattr(self, 'hart_client') or self.hart_client is None:
                         self.hart_client = get_hart_edit_comfy_client()
                     client = self.hart_client
                     version = None
                elif engine == "boogu":
                     from roop.img_editor.boogu_edit_comfy_client import get_boogu_edit_comfy_client
                     if self.boogu_edit_client is None:
                         self.boogu_edit_client = get_boogu_edit_comfy_client()
                     client = self.boogu_edit_client
                     version = "boogu-image-edit-Q2_K.gguf"
                else:
                     from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                     if self.flux_klein_client is None:
                         self.flux_klein_client = get_flux_edit_comfy_client()
                     client = self.flux_klein_client
                     version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"

                # --- EDIT ENGINE ROUTER ---
                # Router general: para CUALQUIER tipo de edición, seleccionar
                # el engine óptimo según el análisis semántico + VRAM.
                # Cubre: body, structural, background, quality, POV, compound.
                override_e = None
                if final_mask is not None and not flux_structural_override:
                    # Kontext/Klein descartan mask_image por diseño
                    # (ReferenceLatent global). Toda máscara focal, automática o
                    # manual, debe ir a LongCat, cuyo workflow la conecta al
                    # inpaint.
                    if engine not in ("longcat", "imagine", "longcat_full"):
                        engine = "longcat"
                        version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
                        from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                        if self.flux_klein_client is None:
                            self.flux_klein_client = get_flux_edit_comfy_client()
                        client = self.flux_klein_client
                    _mask_origin = "manual" if manual_focal_mask else "automática"
                    if _compound_scene_background:
                        print("[ImgEditor] Máscara de fondo reservada para composición final; generación visual global", flush=True)
                    else:
                        print(
                            f"[ImgEditor] Máscara focal {_mask_origin} → {engine} con inpaint "
                            "(router global omitido)",
                            flush=True,
                        )
                elif not flux_structural_override:
                    try:
                        _vram_router = get_vram_gb()
                    except Exception:
                        _vram_router = 0
                    override_e, override_v, override_reason = self._route_edit_engine(
                        engine, analysis, _vram_router,
                    )
                    if override_e:
                        print(f"[ImgEditor] Edit router: {override_reason}", flush=True)
                        engine = override_e
                        version = override_v
                        # Actualizar cliente si el engine target usa flux_klein_client
                        if engine in ("longcat", "imagine", "longcat_full", "kontext",
                                       "klein", "klein_base", "klein_9b"):
                            if self.flux_klein_client is None:
                                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                                self.flux_klein_client = get_flux_edit_comfy_client()
                            client = self.flux_klein_client

                # An automatic, localized mask already confines the requested
                # change. On an 8 GB card the Full LongCat inpaint path can
                # degrade from seconds to more than a minute per step; the
                # distilled model uses the same mask workflow without changing
                # the semantic plan. Manual and structural edits keep their
                # separately chosen engines.
                if (engine == "longcat" and final_mask is not None
                        and not manual_focal_mask and not body_transform
                        and not high_structural and not _compound_scene_background):
                    try:
                        _vram_focal = get_vram_gb()
                    except Exception:
                        _vram_focal = 0
                    if 0 < _vram_focal <= 10:
                        version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
                        params = {"denoise": 1.0, "num_inference_steps": 12,
                                  "guidance_scale": 1.0, "mode": "img2img"}
                        print(
                            "[ImgEditor] Máscara automática localizada en GPU de "
                            f"{_vram_focal:.1f} GB: LongCat Turbo, 12 pasos",
                            flush=True,
                        )

                if (
                    engine == "longcat"
                    and _longcat_focal_reference
                    and locals().get("_longcat_focal_mask") is not None
                    and not body_transform
                    and not high_structural
                ):
                    _focal_full_context = False
                    if locals().get("_hair_subject_no_face", False):
                        # With no reliable face, the full scene is needed to
                        # locate the named subject and keep the other people.
                        _local_focal_crop = (
                            img, _longcat_focal_mask, (0, 0, img.width, img.height),
                        )
                        _focal_full_context = True
                    elif locals().get("_focal_head_plan", False):
                        # A cropped face is a different visual reference for
                        # the edit model and may be redrawn at a different
                        # size. Preserve the source frame as the reference for
                        # every localized head edit; the measured head mask
                        # still limits the final change.
                        _local_focal_crop = (
                            img, _longcat_focal_mask, (0, 0, img.width, img.height),
                        )
                        _focal_full_context = True
                        print(
                            "[ImgEditor] Edición de cabeza: referencia de foto completa, "
                            "integración limitada a la cabeza localizada",
                            flush=True,
                        )
                    else:
                        _local_focal_crop = self._crop_focal_context(
                            img, _longcat_focal_mask,
                        )
                    if _local_focal_crop is not None:
                        _crop_image = _local_focal_crop[0]
                        _crop_share = (_crop_image.width * _crop_image.height) / max(1, img.width * img.height)
                        _bald_crop = bool(
                            locals().get("_focal_head_plan", False)
                            and self._is_bald_request(prompt_pre_rewrite)
                        )
                        if locals().get("_hair_subject_no_face", False):
                            version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
                            params = {"denoise": 1.0, "num_inference_steps": 12,
                                      "guidance_scale": 1.0, "mode": "img2img"}
                            prompt_enhanced += (
                                " Edit the person who currently has visible hair; "
                                "leave people who are already bald unchanged."
                            )
                            print(
                                "[ImgEditor] Perfil sin cara detectada: LongCat Turbo "
                                "ve la foto completa; solo se integra el pelo localizado",
                                flush=True,
                            )
                        elif _bald_crop:
                            version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
                            params = {"denoise": 1.0, "num_inference_steps": 12,
                                      "guidance_scale": 1.0, "mode": "img2img"}
                            print(
                                f"[ImgEditor] Edición de calvicie ({_crop_share:.1%} de referencia): "
                                "LongCat Turbo, 12 pasos",
                                flush=True,
                            )
                        elif _crop_share >= 0.20 and not _focal_full_context:
                            _core_y, _core_x = np.where(np.asarray(_longcat_focal_mask.convert("L")) >= 160)
                            if _core_x.size:
                                _box = _local_focal_crop[2]
                                _margins = (
                                    int(_core_x.min()) - _box[0],
                                    _box[2] - int(_core_x.max()) - 1,
                                    int(_core_y.min()) - _box[1],
                                    _box[3] - int(_core_y.max()) - 1,
                                )
                                if max(_margins) <= 2:
                                    _roomier_crop = self._crop_focal_context(
                                        img, _longcat_focal_mask, max_area=0.55
                                    )
                                    if _roomier_crop is not None:
                                        _local_focal_crop = _roomier_crop
                                        print("[ImgEditor] Crop Klein ampliado con contexto para evitar borde cortado", flush=True)
                            # A larger focal appearance edit can use the
                            # lighter local EDIT model while preserving pose.
                            engine = "klein"
                            version = "flux-2-klein-base-4b-Q4_K_S.gguf"
                            params = {"denoise": 1.0, "num_inference_steps": 4,
                                      "guidance_scale": 1.0, "mode": "img2img"}
                            prompt_enhanced = (
                                f"Edit this exact photo: {prompt.strip().rstrip('.')}. "
                                "Preserve the same face, expression, pose, other people and background."
                            )
                            print(
                                f"[ImgEditor] Crop de cabeza ({_crop_share:.1%}): "
                                "Klein EDIT local para conservar cara y pose",
                                flush=True,
                            )

                if client and not (use_flux_structural_inpaint and result is not None):
                    if body_transform or override_e:
                        # Params Y prompt se calcularon con el engine ORIGINAL:
                        # tras el override se recalculan con el real (log 13:33:
                        # klein→turbo corrió con steps=4 de klein + prompt plano
                        # sin instrucción fuerte). Estructural, sin vocabulario.
                        params = self.auto_detect_params(analysis, engine)
                        if num_inference_steps:
                            params["num_inference_steps"] = num_inference_steps
                        if guidance_scale:
                            params["guidance_scale"] = guidance_scale
                        if denoise:
                            params["denoise"] = denoise
                        print(f"[ImgEditor] Params recalculados para {engine}: denoise={params['denoise']:.2f}, steps={params['num_inference_steps']}, cfg={params['guidance_scale']:.1f}")
                        if override_e:
                            _eff_ov = "longcat_full" if (body_transform and engine in ("klein", "klein_base", "klein_9b")) else engine
                            prompt_enhanced = self._compose_generation_prompt(
                                prompt, img_context=img_description, engine=_eff_ov, magnitude=mag_suggested,
                                quality_only=quality_only, has_quality_request=has_quality_request,
                                enhance_tier=enhance_tier, high_structural=high_structural,
                                structural_bias=structural_bias, body_transform=body_transform,
                                mask_target=mask_target, quality_hybrid=quality_hybrid,
                                detected_age=detected_age, shot_type=shot_type,
                                body_direction=str(analysis.get("body_direction", "change")),
                                body_intensity=float(analysis.get("body_transform_intensity", 0.0)),
                                angle_change=bool(analysis.get("angle_change", False)),
                                pose_change=bool(analysis.get("pose_change", False)),
                            )
                            prompt_enhanced = self._apply_pov_hint(prompt_enhanced, analysis, quality_only)
                            print(f"[ImgEditor] Prompt recompuesto para {engine}: {prompt_enhanced}", flush=True)
                    if engine == "hart":
                        print("[ImgEditor] Generando con HART (autoregresivo) - prompt puro...")
                        hart_prompt = prompt_enhanced or prompt
                        try:
                            ok, load_msg = client.load()
                            if not ok:
                                return None, load_msg or "HART no disponible", final_mask
                        except Exception:
                            pass

                        _image_strength = 1.0 if body_transform else (0.9 if analysis.get("mag", 0) >= 0.62 else 0.7)
                        result_obj, msg = client.generate(
                            prompt=hart_prompt,
                            num_inference_steps=params.get("num_inference_steps", 8),
                            guidance_scale=params.get("guidance_scale", 4.5),
                            source_image=img,
                            image_strength=_image_strength,
                        )
                    elif engine == "qwen_edit":
                        self._release_local_gpu_before_comfy()
                        qwen_version = "Qwen_Image_Edit-Q2_K.gguf"
                        success, msg = client.load(qwen_version=qwen_version)
                        if not success:
                            return None, msg or "Qwen-Image-Edit no disponible", final_mask
                        qwen_prompt = prompt_enhanced or prompt
                        result_obj, msg = client.generate(
                            prompt=qwen_prompt,
                            source_image=img,
                            num_inference_steps=params.get("num_inference_steps", 20),
                            guidance_scale=params.get("guidance_scale", 3.5),
                            denoise=params.get("denoise", 0.80),
                        )
                    else:
                        self._release_local_gpu_before_comfy()
                        load_params = {"qwen_version": version} if "qwen" in engine else {"flux_version": version}
                        success, msg = client.load(**load_params)
                        if not success:
                            # Turbo es el motor elegido (spec/00): si no carga
                            # se informa, no se cambia a Klein (ignora mascara).
                            return None, msg, final_mask

# Una región focal detectada no se manda como un inpaint
                        # LongCat (su VAE puede rellenarla de gris), ni se deja
                        # regenerar sobre el fotograma entero. Se genera un crop
                        # con contexto y después se repone exactamente en ese
                        # rectángulo. Así el modelo no puede borrar elementos de
                        # la foto fuera de la región visual localizada.
                        _generation_image = img
                        if (
                            _longcat_focal_reference
                            and locals().get("_longcat_focal_mask") is not None
                            and engine in ("longcat", "longcat_full", "imagine", "klein")
                        ):
                            if _local_focal_crop is None:
                                _local_focal_crop = self._crop_focal_context(
                                    img, locals().get("_longcat_focal_mask")
                                )
                            if _local_focal_crop is None:
                                _longcat_focal_reference = False
                                _longcat_focal_mask = None
                                print(
                                    "[ImgEditor] Crop focal no viable; continuando "
                                    "con la foto completa como referencia visual",
                                    flush=True,
                                )
                            else:
                                _generation_image = _local_focal_crop[0]
                                _cw, _ch = _generation_image.size
                                if engine == "klein" and min(_cw, _ch) < 256:
                                    _scale = 256.0 / min(_cw, _ch)
                                    _generation_image = _generation_image.resize(
                                        (int(round(_cw * _scale)), int(round(_ch * _scale))),
                                        Image.Resampling.LANCZOS,
                                    )
                                    print(
                                        f"[ImgEditor] Crop Klein ampliado: {_cw}x{_ch} → "
                                        f"{_generation_image.width}x{_generation_image.height}",
                                        flush=True,
                                    )
                                if locals().get("_hair_subject_no_face", False):
                                    print(
                                        f"[ImgEditor] Referencia completa {_cw}x{_ch}; "
                                        "solo se integrará el pelo localizado",
                                        flush=True,
                                    )
                                elif locals().get("_focal_full_context", False):
                                    print(
                                        f"[ImgEditor] Referencia completa {_cw}x{_ch}; "
                                        "solo se integrará la cabeza localizada",
                                        flush=True,
                                    )
                                else:
                                    print(
                                        f"[ImgEditor] Focal crop: {_cw}x{_ch}; "
                                        "el modelo no regenerará la foto completa",
                                        flush=True,
                                    )

                        print(f"[ImgEditor] Enviando a ComfyUI ({engine}, version={version})...")
                        gen_params = {
                            "image": _generation_image, "prompt": prompt_enhanced,
                            "negative_prompt": negative_prompt,
                            "num_inference_steps": params["num_inference_steps"],
                            "guidance_scale": params["guidance_scale"],
                            "seed": seed, "denoise": params["denoise"],
                            "lora_name": lora_name,
                            "lora_strength": lora_strength,
                            "progress_callback": progress_callback,
                        }
                        # ControlNet Pose: si hay cambio estructural/postura, generar pose map
                        # para garantizar coherencia anatómica (sin miembros extra/deformes).
                        # Pero NO en pose_change: el esqueleto FUENTE congelaría la pose
                        # vieja y el cambio nunca ocurriría (el bypass ya va sin ancla).
                        if high_structural and not body_transform and not analysis.get("pose_change", False) and engine in ("klein", "longcat", "imagine", "longcat_full", "kontext"):
                            try:
                                from roop.img_editor.controlnet_utils import get_controlnet_utils
                                cn_utils = get_controlnet_utils()
                                pose_np = cn_utils.detect_pose(img)
                                if pose_np is not None:
                                    from PIL import Image as _PILImage
                                    import numpy as _np
                                    pose_pil = _PILImage.fromarray(pose_np.astype(_np.uint8))
                                    gen_params["controlnet_pose_image"] = pose_pil
                                    gen_params["controlnet_strength"] = 0.65
                                    print("[ImgEditor] ControlNet Pose activo (strength=0.65, garantía anatómica)", flush=True)
                            except Exception as e:
                                print(f"[ImgEditor] ControlNet Pose skip: {e}", flush=True)
                        if flux_structural_override:
                            gen_params["denoise"] = min(0.98, max(params["denoise"], 0.97))
                            gen_params["grow_mask_by"] = 48
                            gen_params["mask_image"] = final_mask
                            gen_params["num_inference_steps"] = max(26, params["num_inference_steps"])
                            gen_params["guidance_scale"] = 1.0
                        elif high_structural and engine in ("longcat", "imagine", "longcat_full", "kontext"):
                            if structural_inpaint and structural_bias == "remove":
                                gen_params["grow_mask_by"] = 40
                                gen_params["denoise"] = min(0.98, max(params["denoise"], 0.97))
                            elif structural_inpaint and structural_bias == "add":
                                gen_params["grow_mask_by"] = 30
                                gen_params["denoise"] = min(0.98, max(params["denoise"], 0.94))
                        ref_method = self._resolve_reference_latents_method(
                            analysis,
                            background_focus=background_focus,
                            structural_inpaint=structural_inpaint,
                            engine=engine,
                            body_transform=body_transform,
                        )
                        if ref_method:
                            gen_params["reference_latents_method"] = ref_method
                            print(f"[ImgEditor] Kontext ref={ref_method}")
                        # v5.1104 — uncover en Kontext: img2img PURO (denoise del
                        # auto_detect, ~0.83) SIN ReferenceLatent. Evidencia: con
                        # tokens limpios el modelo siempre resuelve a favor de la
                        # foto (Q2+offset=copia, Q4+offset=copia, Q4+uxo=solo
                        # color). Cover/change/no-body siguen en modo EDIT.
                        _bdir_img = str(analysis.get("body_direction", "change")).lower()
                        if engine == "kontext" and body_transform and _bdir_img == "uncover":
                            gen_params["kontext_img2img_denoise"] = float(params.get("denoise", 0.83))
                            print(f"[ImgEditor] Kontext uncover: img2img PURO denoise={gen_params['kontext_img2img_denoise']:.2f} (sin ReferenceLatent)", flush=True)
                        # v5.1106 — uncover en LongCat: bypass no_kontext CON visión
                        # (Qwen-VL ve género/pose/proporciones, SIN ancla estructural
                        # de latentes → el prompt manda libremente). Régimen probado
                        # en junio para desnudo ("ha seguido las instrucciones").
                        # El patch _patch_strong_edit.py nunca llegó a aplicarse al
                        # manager — este bloque es su implementación real.
                        if body_transform and _bdir_img == "uncover" and engine in ("longcat", "imagine", "longcat_full"):
                            gen_params["no_kontext"] = True
                            gen_params["drop_edit_image_ref"] = False
                            gen_params["prompt_authority"] = True
                            print("[ImgEditor] Bypass LongCat uncover: Qwen-VL ve la foto, sin ancla (sigue el prompt)", flush=True)
                        # v5.1116 — change FUERTE: mismo bypass que uncover. La matriz
                        # de evidencia (v5.1101-1104) demostró que CUALQUIER ancla de
                        # latentes (offset/uxo) gana a la instrucción en cambios
                        # fuertes — el contenido destino no existe en la fuente y el
                        # modelo resuelve a favor de la foto. A partir de 0.70 el
                        # analizador ya ha clasificado una instrucción como fuerte;
                        # dejar 0.70-0.74 anclado produjo un copy-machine aunque el
                        # prompt exigiera que el cambio fuese visible.
                        # v5.1118 — extendido a NO-body: el log real 07:22 (mag=0.80,
                        # pose no flaggeada por el analyzer) cayó en kontext+offset
                        # = copy machine (std=52). El ancla pierde contra el cambio
                        # por MAGNITUD, no por dirección/flag. Cualquier longcat-family
                        # Sin máscara, solo los cambios realmente fuertes usan
                        # bypass global. Un atributo localizado debe conservar
                        # su máscara de inpaint: es la forma de cambiar el
                        # sujeto sin reconstruir la escena completa.
                        _focal_reference_needs_following = bool(
                            locals().get("_longcat_focal_reference", False)
                            and float(analysis.get("appearance_score", 0.0)) >= 0.08
                            and float(analysis.get("magnitude", 0.0)) >= 0.65
                        )
                        if (final_mask is None
                                and (
                                    body_transform
                                    or float(analysis.get("magnitude", 0.0)) >= 0.75
                                    or _focal_reference_needs_following
                                )
                                and (_bdir_img != "uncover" or not body_transform)
                                and engine in ("longcat", "imagine", "longcat_full")):
                            gen_params["no_kontext"] = True
                            gen_params["drop_edit_image_ref"] = False
                            gen_params["prompt_authority"] = True
                            _follow_reason = (
                                "región focal localizada" if _focal_reference_needs_following
                                else f"magnitud={analysis.get('magnitude', 0.0):.2f}"
                            )
                            print(
                                f"[ImgEditor] LongCat fuerte ({_follow_reason}): "
                                "Qwen-VL ve la foto y la instrucción tiene prioridad",
                                flush=True,
                            )
                        if _compound_scene_background:
                            gen_params["no_kontext"] = True
                            gen_params["drop_edit_image_ref"] = False
                            print("[ImgEditor] Entorno: generación fuerte y composición solo del fondo", flush=True)
                        if bool(analysis.get("multi_region_edit", False)) and engine in ("longcat", "imagine", "longcat_full"):
                            # El plan viene del rewriter; para varios destinos se
                            # refuerza el muestreo pero se mantiene la referencia
                            # visual. Es una decisión de complejidad, no una lista
                            # de instrucciones codificada.
                            gen_params["reference_latents_method"] = "index"
                            gen_params["denoise"] = max(float(gen_params.get("denoise", 0.0)), 0.90)
                            gen_params["num_inference_steps"] = max(int(gen_params.get("num_inference_steps", 0)), 30)
                            gen_params["guidance_scale"] = max(float(gen_params.get("guidance_scale", 0.0)), 3.5)
                            print("[ImgEditor] Multi-región: referencia index + muestreo reforzado (30 pasos, denoise 0.90)", flush=True)
                        # v5.1111 — POV (sin body): mismo régimen que uncover —
                        # Qwen-VL ve la foto (género/pose/escena) pero SIN ancla
                        # de latentes → el prompt manda y el modelo alucina la
                        # vista nueva. Con ancla (EDIT+offset) la vista NUNCA
                        # cambia (evidencia v5.1101-1105: offset = copy machine).
                        if (not body_transform) and analysis.get("angle_change", False) and engine in ("longcat", "imagine", "longcat_full"):
                            gen_params["no_kontext"] = True
                            gen_params["drop_edit_image_ref"] = False
                            gen_params["prompt_authority"] = True
                            print("[ImgEditor] POV bypass: Qwen-VL ve la foto, sin ancla → alucina vista nueva", flush=True)
                        # Postura nueva: mismo bypass (el ancla conserva la pose).
                        if (not body_transform) and analysis.get("pose_change", False) and engine in ("longcat", "imagine", "longcat_full"):
                            gen_params["no_kontext"] = True
                            gen_params["drop_edit_image_ref"] = False
                            gen_params["prompt_authority"] = True
                            print("[ImgEditor] POSE bypass: Qwen-VL ve la foto, sin ancla → alucina postura nueva", flush=True)
                        # v5.1118 — pose SIN flag pero mag>=0.75: el analyzer a veces
                        # no detecta la pose (log real: 'leans over' → Structural=0.07,
                        # pose_change=False) pero el cambio igualmente es fuerte. El
                        # bypass de magnitud de arriba ya lo cubre; este elif solo
                        # ajusta el mensaje. (No-op real: la condición mag>=0.75 del
                        # bloque v5.1118 dispara primero.)
                        # Bypass sin ancla = regeneración global: la máscara focal
                        # no tiene sentido aquí (el KSampler arranca de latente
                        # vacío; el cliente la ignora al encolar) y el composite
                        # post-generación repintaría la foto ORIGINAL sobre el
                        # cambio (log 19:15: subject + composite → pelo original
                        # de vuelta, calvo no ejecutado). Global puro, salvo
                        # inpaint estructural que SÍ necesita su máscara.
                        if (bool(gen_params.get("no_kontext", False)) and final_mask is not None
                                and not structural_inpaint and not flux_structural_override
                                and not _compound_scene_background):
                            final_mask = None
                            print("[ImgEditor] Bypass sin ancla → global puro (sin máscara focal ni composite: el cambio manda)", flush=True)
                        # El engine puede ser override a longcat_full más adelante.
                        # Usar effective_engine para decisiones de prompt/denoise.
                        effective_engine_for_check = "longcat_full" if body_transform else engine
                        # v5.1098: Kontext body-uncover entra a strong/bypass como
                        # longcat (ancla VAEEncode 0.93 conserva la ropa).
                        # Sin esto el body-Kontext nunca subía de 0.93.
                        if body_transform and engine == "kontext":
                            effective_engine_for_check = "longcat"
                        strong_edit = body_transform or (mag_suggested >= 0.75 and effective_engine_for_check in ("longcat", "imagine", "longcat_full", "kontext"))
                        if strong_edit and engine == "kontext" and body_transform:
                            _bdir = str(analysis.get("body_direction", "change")).lower()
                            if _bdir == "uncover":
                                # v5.1101: 4.8→3.5 (el 4.8 no movía la aguja, log 08:41;
                                # la adherencia la da el método uxo/uno, no más guidance)
                                gen_params["guidance_scale"] = max(gen_params.get("guidance_scale", params["guidance_scale"]), params["guidance_scale"], 3.5)
                            else:
                                gen_params["guidance_scale"] = max(gen_params.get("guidance_scale", params["guidance_scale"]), params["guidance_scale"], 3.5)
                            print(f"[ImgEditor] Kontext body strong (EDIT): guidance={gen_params['guidance_scale']:.1f}", flush=True)
                        # Los parámetros ya vienen adaptativos desde auto_detect_params().
                        # No se fuerzan overrides agresivos aquí porque rompían la
                        # preservación de identidad/rostro y alejaban el resultado
                        # de la imagen original. Cualquier ajuste fino se hace en
                        # auto_detect_params(), no por condiciones sueltas.
                        if "qwen" not in engine and not flux_structural_override:
                            gen_params["mask_image"] = (
                                final_mask if mask_mode in ["manual", "global"]
                                and not _compound_scene_background else None
                            )
                            if gen_params["mask_image"] is not None:
                                print(
                                    f"[ImgEditor] Máscara focal entregada al cliente "
                                    f"(modo={mask_mode}, cobertura={self._mask_coverage(gen_params['mask_image']):.1%})",
                                    flush=True,
                                )
                            elif final_mask is not None:
                                print(
                                    f"[ImgEditor] Máscara focal no enviada (modo={mask_mode}, motor={engine})",
                                    flush=True,
                                )

                        if cancel_check and cancel_check():
                            return None, "Cancelado", final_mask
                        gen_params["cancel_check"] = cancel_check
                        result_obj, msg = client.generate(**gen_params)
                        if cancel_check and cancel_check():
                            return None, "Cancelado", final_mask
                    if result_obj:
                        result = result_obj.image
                        if _compound_scene_background:
                            scene_masker = self._get_clipseg_masker()
                            try:
                                scene_result, scene_error, _scene_head_preserve_mask = self._compose_scene_preserving_heads(
                                    img, result, final_mask, source_faces, scene_masker,
                                )
                            finally:
                                scene_masker.unload()
                            if scene_result is None:
                                print(f"[ImgEditor] Entorno omitido: {scene_error}", flush=True)
                                return img, f"Omitida: {scene_error}; se conserva la edición anterior", None
                            result = scene_result
                            final_mask = None
                            print("[ImgEditor] Entorno integrado; cabezas de la primera pasada conservadas", flush=True)
                        if _local_focal_crop is not None:
                            try:
                                _crop_source, _crop_mask, _crop_box = _local_focal_crop
                                if result.size != _crop_source.size:
                                    result = result.resize(_crop_source.size, Image.Resampling.LANCZOS)
                                # Use the grounded region from the input. For
                                # head edits this contains the complete head;
                                # a hair-only paste can overlap generated skin
                                # with the original eyes or ears.
                                _composite_mask = locals().get("_longcat_focal_mask")
                                _crop_soft_mask = self._feather_mask(
                                    _composite_mask.crop(_crop_box), amount=6
                                )
                                _crop_result = Image.composite(result, _crop_source, _crop_soft_mask)
                                _full_result = img.copy()
                                _full_result.paste(_crop_result, _crop_box)
                                result = _full_result
                                final_mask = _composite_mask
                                _local_focal_generated = True
                                _focal_output_composited = True
                                print(
                                    "[ImgEditor] Región localizada integrada; "
                                    "exterior de la foto intacto",
                                    flush=True,
                                )
                            except Exception as _crop_integrate_error:
                                note = (
                                    f"Error al integrar el recorte focal: {_crop_integrate_error}. "
                                    "Se entrega el recorte generado sin modificar la foto completa."
                                )
                                print(f"[ImgEditor] {note}", flush=True)
                                return result, note, _local_focal_crop[1]
                        if result.size != img.size and engine != "hart":
                            # v5.1153 — RESOLUCIÓN DE SALIDA: conservar la mayor de las
                            # dos áreas. Antes se forzaba SIEMPRE img.size con LANCZOS:
                            # (a) original pequeño (665×600) → la salida del modelo a
                            # 1MP (1072×976) se REDUCÍA tirando el detalle generado
                            # (log real: edit_1789505211.png salió a 665×600);
                            # (b) original grande (3692×3840) → la salida 1MP se
                            # AMPLIABA con LANCZOS puro = blanda ("poca resolución").
                            _ow, _oh = img.size
                            _rw, _rh = result.size
                            if _rw * _rh >= _ow * _oh:
                                _target_size = (_rw, _rh)
                            else:
                                _target_size = (_ow, _oh)
                            _enlarged_gen = (_rw * _rh) < (_ow * _oh)
                            _ratio_up = (max(_target_size) / max(1.0, float(max(_rw, _rh)))) if _enlarged_gen else 1.0
                            if _target_size != result.size:
                                result = result.resize(_target_size, Image.LANCZOS)
                            if _target_size != img.size:
                                img = img.resize(_target_size, Image.LANCZOS)
                            print(f"[ImgEditor] Resolución de salida: {_target_size[0]}x{_target_size[1]} (modelo {_rw}x{_rh}, original {_ow}x{_oh})", flush=True)
                            # Ampliación grande del resultado del modelo (>=1.6x):
                            # LANCZOS solo interpola. Añadir textura REAL con ESRGAN
                            # x2 full-frame blend (el finisher ya guarda tiles y VRAM;
                            # falla silenciosa si el modelo no está).
                            if _enlarged_gen and _ratio_up >= 1.35:
                                try:
                                    from roop.img_editor.image_quality_pipeline import get_quality_finisher as _gqf
                                    _res_img, _res_note = _gqf().esrgan_x2_fullframe_blend(
                                        result, tier="hd", blend=0.35
                                    )
                                    if _res_note:
                                        result = _res_img
                                        print(f"[ImgEditor] Textura ESRGAN en ampliación: {_res_note}", flush=True)
                                except Exception as _e_esr:
                                    print(f"[ImgEditor] ESRGAN en ampliación omitido: {_e_esr}", flush=True)
                        # Color transfer adaptativo: solo si la generada diverge del original.
                        # Mide divergencia en saturación media (HSV-S) y la corrige vía LAB.
                        # Sin hardcode: el umbral sale de la propia estadística, no de palabras.
                        # v5.1122 — también en RÉGIMEN BYPASS (mag>=0.75 + longcat-family):
                        # la regeneración total (denoise 1.0, sin ancla) quema colores
                        # (log 08:41: std=90.57, mean=144.73) y body_transform era False
                        # → nunca corregía. Mismo régimen, misma corrección.
                        _bypass_color = (
                            mag_suggested >= 0.75
                            and engine in ("longcat", "longcat_full", "imagine")
                        )
                        if (body_transform or _bypass_color) and not quality_only and engine != "hart" and not _compound_scene_background:
                            try:
                                import cv2 as _cv2
                                import numpy as _np
                                _src = _np.array(img.convert("RGB"))
                                _dst = _np.array(result.convert("RGB"))
                                _src_hsv = _cv2.cvtColor(_src, _cv2.COLOR_RGB2HSV).astype(_np.float32)
                                _dst_hsv = _cv2.cvtColor(_dst, _cv2.COLOR_RGB2HSV).astype(_np.float32)
                                _sat_src = float(_src_hsv[:, :, 1].mean())
                                _sat_dst = float(_dst_hsv[:, :, 1].mean())
                                _sat_ratio = _sat_dst / max(1.0, _sat_src)
                                # Solo corregir si la saturación diverge >35% (tolerancia natural del modelo)
                                if abs(_sat_ratio - 1.0) > 0.35:
                                    # Corrección HSV adaptativa: escala la saturación (S) para alinearla
                                    # con la fuente sin destruir la varianza cromática ni los canales de color.
                                    # Reinhard LAB aplastaba los canales A y B (scale ~0.4) provocando
                                    # piel de cadáver grisácea/verdosa y alteración radical de tonos.
                                    # La corrección es bidireccional: el bypass puede tanto
                                    # quemar como deslavar la foto. El límite evita convertir
                                    # una edición intencionalmente neutra en una sobresaturada.
                                    _corr_ratio = min(1.65, max(0.55, 1.0 / _sat_ratio))
                                    _corr_eff = 1.0 + (_corr_ratio - 1.0) * 0.70
                                    _dst_hsv[:, :, 1] = _np.clip(_dst_hsv[:, :, 1] * _corr_eff, 0, 255)
                                    # Alineación suave de luminancia media (V) sin aplastar varianza
                                    _v_src_m = _src_hsv[:, :, 2].mean()
                                    _v_dst_m = _dst_hsv[:, :, 2].mean()
                                    _dst_hsv[:, :, 2] = _np.clip(_dst_hsv[:, :, 2] + (_v_src_m - _v_dst_m) * 0.4, 0, 255)
                                    from PIL import Image as _PILImage
                                    result = _PILImage.fromarray(_cv2.cvtColor(_dst_hsv.astype(_np.uint8), _cv2.COLOR_HSV2RGB))
                                    print(f"[ImgEditor] Color adaptativo HSV aplicado (sat {1.0:.2f}→{_sat_ratio:.2f}, corr={_corr_eff:.2f})", flush=True)
                                else:
                                    print(f"[ImgEditor] Color OK (sat ratio {_sat_ratio:.2f}, sin corrección)", flush=True)
                            except Exception as e:
                                print(f"[ImgEditor] Color transfer omitido: {e}", flush=True)

                        # Restaurar rostros originales (el bypass/body pierde identidad).
                        # FacePreserver DESHABILITADO POR PREFERENCIA DEL USUARIO
                        # (lo ha dicho muchas veces: es lo que menos le importa).
                        # El output del modelo es final, sin paste manual.
                        # Para re-activar: poner _FP_ENABLED = True.
                        _FP_ENABLED = False
                        _do_fp_pov = False
                        _do_fp_body = False
                        if _FP_ENABLED and (_do_fp_pov or _do_fp_body):
                            try:
                                from roop.img_editor.face_preserver import FacePreserver
                                fp = FacePreserver()
                                ok_fp, _ = fp.initialize()
                                if ok_fp:
                                    restored = fp.preserve_faces(
                                        img.convert("RGB"), result.convert("RGB")
                                    )
                                    if restored is not None:
                                        result = restored
                                        _fp_kind = "POV" if _do_fp_pov else "BODY"
                                        print(f"[ImgEditor] FacePreserver {_fp_kind}: cara original restaurada (cuerpo/vista regenerados, identidad facial intacta)", flush=True)
                            except Exception as e:
                                print(f"[ImgEditor] FacePreserver omitido: {e}", flush=True)

                        if final_mask and engine != "hart" and not locals().get("_focal_output_composited", False):
                            if _compound_depth > 0 and _compound_region_hint.strip():
                                _compound_generated_reference = result.copy()
                            # v5.1153: la máscara se crea a tamaño del original; con la
                            # resolución de salida máxima puede no coincidir con result.
                            try:
                                if final_mask.size != result.size:
                                    final_mask = final_mask.resize(result.size)
                            except Exception:
                                pass
                            if flux_structural_override or structural_inpaint:
                                soft_mask = self._feather_mask(final_mask, amount=25)
                                result = Image.composite(result, img.copy(), soft_mask)
                            else:
                                if not _compound_scene_background:
                                    result = self._apply_skin_tone_match(img, result, final_mask)
                                feather_amt = 8 if _compound_scene_background else (32 if (quality_hybrid or background_focus) else 20)
                                soft_mask = self._feather_mask(final_mask, amount=feather_amt)
                                result = Image.composite(result, img.copy(), soft_mask)

            quality_primary = bool(analysis.get("quality_primary", False))
            if result is not None and (quality_only or quality_primary or quality_hybrid or auto_upscale):
                if cancel_check and cancel_check():
                    return None, "Cancelado", final_mask
                q_gen = bool(analysis.get("quality_use_generative", False))
                # Solo el pulido híbrido que realmente abre ComfyUI necesita
                # desalojar caches locales. En Calidad pura sería contraproducente:
                # libera ESRGAN/CodeFormer para recargarlos inmediatamente y deja
                # la siguiente mejora más lenta sin aportar nada.
                if q_gen:
                    try:
                        self._release_local_gpu_before_comfy()
                    except Exception:
                        pass
                else:
                    print(
                        "[ImgEditor] Calidad clásica: caches locales conservados (sin carga ComfyUI)",
                        flush=True,
                    )
                try:
                    q_faces = bool(analysis.get("quality_preserve_faces", True))
                    q_steps = analysis.get("quality_steps")
                    if quality_only or quality_primary:
                        from roop.img_editor.top_tier_finish import run_top_tier_quality
                        result, fin_note, quality_analysis = run_top_tier_quality(
                            result,
                            enhance_tier=enhance_tier,
                            use_generative=q_gen,
                            preserve_faces=q_faces,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            quality_steps=q_steps,
                        )
                    elif quality_hybrid:
                        from roop.img_editor.top_tier_finish import run_quality_enhance_after_edit
                        result, fin_note, quality_analysis = run_quality_enhance_after_edit(
                            result,
                            img,
                            enhance_tier=enhance_tier,
                            use_generative=q_gen,
                            preserve_faces=q_faces,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                            quality_steps=q_steps,
                        )
                    else:
                        from roop.img_editor.image_quality_pipeline import get_quality_finisher
                        finisher = get_quality_finisher()
                        result, fin_note, quality_analysis = finisher.finish(
                            result,
                            upscale=auto_upscale,
                            enhance_tier="hd",
                            sharpen_image=True,
                            denoise=False,
                            ultra=False,
                            depixelize_image=False,
                            tile_restore=False,
                            cancel_check=cancel_check,
                            progress_callback=progress_callback,
                        )
                    self._last_quality_analysis = quality_analysis
                    if cancel_check and cancel_check():
                        return None, "Cancelado", final_mask
                    if quality_analysis and quality_analysis.get("summary"):
                        msg = f"{msg} | {quality_analysis['summary']}"
                    msg = f"{msg} | {fin_note}" if fin_note else msg
                    print(f"[ImgEditor] Post-calidad: {fin_note}")
                except Exception as e:
                    print(f"[ImgEditor] Post-calidad omitido: {e}")

            polish_analysis = None
            if result is not None and not quality_only and not quality_hybrid:
                try:
                    from roop.img_editor.top_tier_finish import (
                        run_top_tier_edit_finish,
                        should_edit_top_finish,
                    )
                    skip_heavy_polish = bool(
                        analysis.get("body_transform")
                        or float(analysis.get("magnitude", 0.0)) >= 0.62
                    )
                    # El pulido LongCat Full en 8GB va con offload (log 12:56:
                    # 547s por 16 steps) — impuesto brutal a cada edit medio.
                    # En VRAM <11GB se salta y las caras las restaura CodeFormer
                    # (ONNX, rápido) en el paso de abajo.
                    try:
                        _vram_polish = get_vram_gb()
                    except Exception:
                        _vram_polish = 0
                    _low_vram_polish = bool(_vram_polish and _vram_polish < 11)
                    if _low_vram_polish and should_edit_top_finish(analysis) and not skip_heavy_polish:
                        print(f"[ImgEditor] Polish Full omitido en {_vram_polish:.0f}GB VRAM (offload ~9min) — caras vía CodeFormer", flush=True)
                        skip_heavy_polish = True
                    if should_edit_top_finish(analysis) and not skip_heavy_polish:
                        print("[ImgEditor] Acabado TOP post-edit (LongCat pulido + caras)")
                        result, polish_note, polish_analysis = run_top_tier_edit_finish(
                            result,
                            img,
                            analysis,
                            enhance_tier=enhance_tier or "hd",
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )
                        if polish_note and "omitido" not in polish_note:
                            msg = f"{msg} | {polish_note}" if msg else polish_note
                            print(f"[ImgEditor] Post-acabado TOP: {polish_note}")
                    elif bool(analysis.get("has_quality_request")):
                        from roop.img_editor.hyperreal_polish import polish_result_image
                        result, polish_note, polish_analysis = polish_result_image(
                            result, tier=enhance_tier or "hd",
                        )
                        if polish_note and "omitido" not in polish_note:
                            msg = f"{msg} | {polish_note}" if msg else polish_note
                            print(f"[ImgEditor] Post-acabado: {polish_note}")
                except Exception as e:
                    print(f"[ImgEditor] Post-acabado omitido: {e}")

            # v5.1145 — COLOR/CALIDAD AUTOMÁTICOS MEDIDOS (una pasada, bucle de feedback).
            # La queja recurrente es que la salida del edit sale con poca nitidez y
            # posterizada. Hasta ahora la mejora solo corría si el USUARIO pedía
            # calidad (quality_enhance) o si el polish top disparaba. Ahora se
            # analiza SIEMPRE la imagen RESULTADO (medida, no asumida): si el
            # finisher detecta posterización alta o nitidez baja, se aplica una
            # pasada adaptativa de color/nitidez/textura — como el bucle v5.1129
            # del modo calidad pero para toda edición. Respeta lo ya hecho por el
            # pipeline TOP (se re-mide el RESULTADO final, solo corrige si hace falta).
            _face_quality_analysis = None
            _candidate_poster = 0.0
            _candidate_restore = 0.0
            if result is not None and not quality_only:
                try:
                    from roop.img_editor.image_quality_pipeline import get_quality_finisher
                    from roop.img_editor.hyperreal_polish import polish_result_image
                    _qfin = get_quality_finisher()
                    _tier_adq = (enhance_tier or "hd").lower()
                    _qa_final = _qfin.analyze_image(result, tier=_tier_adq)
                    _sc = _qa_final.get("scores", {})
                    _poster_f = float(_sc.get("poster", 0.0))
                    _sharp_f = float(_sc.get("sharpness", 1.0))
                    if _poster_f > 0.28 or _sharp_f < 0.30:
                        _polish_img, _polish_note, _ = polish_result_image(
                            result, tier=_tier_adq,
                        )
                        if _polish_note and "omitido" not in _polish_note and "limpia" not in _polish_note:
                            result = _polish_img
                            msg = f"{msg} | {_polish_note}" if msg else _polish_note
                            print(f"[ImgEditor] Auto-acabado medido (poster={_poster_f:.2f}, nit={_sharp_f:.2f}): {_polish_note}", flush=True)
                        else:
                            print(f"[ImgEditor] Auto-acabado: sin cambios necesarios (poster={_poster_f:.2f}, nit={_sharp_f:.2f})", flush=True)
                    else:
                        print(f"[ImgEditor] Auto-acabado: salida limpia (poster={_poster_f:.2f}, nit={_sharp_f:.2f})", flush=True)
                    # El pulido ligero no siempre basta cuando el propio modelo
                    # devuelve planos de color. Re-medimos el RESULTADO y, solo
                    # si sigue alto, aplicamos la limpieza edge-aware del modo
                    # Calidad. Así una edición normal recibe calidad verificable
                    # sin afilar bandas ni inventar detalle.
                    _qa_after = _qfin.analyze_image(result, tier=_tier_adq)
                    _face_quality_analysis = _qa_after
                    _poster_after = float(_qa_after.get("scores", {}).get("poster", 0.0))
                    _candidate_poster = _poster_after
                    _candidate_restore = float(_qa_after.get("scores", {}).get("restore", 0.0))
                    if _poster_after > 0.36:
                        _dep_blend = min(0.46, 0.16 + _poster_after * 0.42)
                        _cleaned = _qfin.depixelize(result, blend_strength=_dep_blend, tier=_tier_adq)
                        _cleaned = _qfin.soften_poster_bands(
                            _cleaned, blend=min(0.26, 0.08 + _poster_after * 0.28)
                        )
                        _cleaned = _qfin.reduce_stripe_artifacts(
                            _cleaned, blend=min(0.18, 0.05 + _poster_after * 0.16)
                        )
                        _poster_clean = float(
                            _qfin.analyze_image(_cleaned, tier=_tier_adq).get("scores", {}).get("poster", 0.0)
                        )
                        if _poster_clean < _poster_after:
                            result = _cleaned
                            print(
                                f"[ImgEditor] Auto-calidad confirmada: poster {_poster_after:.2f}→{_poster_clean:.2f} "
                                f"(limpieza {_dep_blend:.0%})",
                                flush=True,
                            )
                        else:
                            print(
                                f"[ImgEditor] Auto-calidad conservadora: limpieza descartada "
                                f"(poster {_poster_after:.2f}→{_poster_clean:.2f})",
                                flush=True,
                            )
                    # v5.1145b — COLOR ADAPTATIVO vs ORIGINAL para TODA edición:
                    # mismo mecanismo HSV del bloque bypass/body (sin LAB destructivo),
                    # pero disparado por medida (divergencia >35%) y no por flags.
                    # Una edición fiel no se toca (abs(ratio-1)<=0.35); una salida
                    # quemada o desvaída se realinea con la foto fuente.
                    try:
                        import cv2 as _cv2c
                        import numpy as _npc
                        _src = _npc.array(img.convert("RGB"))
                        _dst = _npc.array(result.convert("RGB"))
                        # Nunca reescalar el RESULTADO para medir color: esa
                        # reducción silenciosa anulaba el upscale del modo Calidad
                        # (por ejemplo 1920px volvía a 1152px). Las medias HSV son
                        # comparables aunque las dos imágenes tengan distinta talla.
                        _src_hsv = _cv2c.cvtColor(_src, _cv2c.COLOR_RGB2HSV).astype(_npc.float32)
                        _dst_hsv = _cv2c.cvtColor(_dst, _cv2c.COLOR_RGB2HSV).astype(_npc.float32)
                        _sat_ratio_a = float(_dst_hsv[:, :, 1].mean()) / max(1.0, float(_src_hsv[:, :, 1].mean()))
                        # La corrección global solo es segura si el desvío de
                        # saturación es homogéneo. Una edición puede cambiar un
                        # área concreta y esa variación local es intencional;
                        # no debe ser "corregida" hacia la foto de entrada.
                        _tile_ratios = []
                        _grid = 4
                        for _gy in range(_grid):
                            for _gx in range(_grid):
                                _sy0, _sy1 = _gy * _src_hsv.shape[0] // _grid, (_gy + 1) * _src_hsv.shape[0] // _grid
                                _sx0, _sx1 = _gx * _src_hsv.shape[1] // _grid, (_gx + 1) * _src_hsv.shape[1] // _grid
                                _dy0, _dy1 = _gy * _dst_hsv.shape[0] // _grid, (_gy + 1) * _dst_hsv.shape[0] // _grid
                                _dx0, _dx1 = _gx * _dst_hsv.shape[1] // _grid, (_gx + 1) * _dst_hsv.shape[1] // _grid
                                _src_sat_tile = float(_src_hsv[_sy0:_sy1, _sx0:_sx1, 1].mean())
                                if _src_sat_tile >= 8.0:
                                    _dst_sat_tile = float(_dst_hsv[_dy0:_dy1, _dx0:_dx1, 1].mean())
                                    _tile_ratios.append(_dst_sat_tile / max(1.0, _src_sat_tile))
                        _tile_spread = float(_npc.percentile(_tile_ratios, 75) - _npc.percentile(_tile_ratios, 25)) if len(_tile_ratios) >= 4 else float("inf")
                        _uniform_color_shift = len(_tile_ratios) >= 4 and _tile_spread <= 0.18
                        if abs(_sat_ratio_a - 1.0) > 0.35 and _uniform_color_shift:
                            _corr = min(1.35, max(0.55, 1.0 / _sat_ratio_a))
                            # mezcla parcial (60% hacia el ratio exacto) — corrección suave, no a golpe
                            _corr_eff = 1.0 + (_corr - 1.0) * 0.60
                            _dst_hsv[:, :, 1] = _npc.clip(_dst_hsv[:, :, 1] * _corr_eff, 0, 255)
                            _v_src_m = float(_src_hsv[:, :, 2].mean())
                            _v_dst_m = float(_dst_hsv[:, :, 2].mean())
                            _dst_hsv[:, :, 2] = _npc.clip(_dst_hsv[:, :, 2] + (_v_src_m - _v_dst_m) * 0.30, 0, 255)
                            from PIL import Image as _PILImage
                            result = _PILImage.fromarray(_cv2c.cvtColor(_dst_hsv.astype(_npc.uint8), _cv2c.COLOR_HSV2RGB))
                            print(f"[ImgEditor] Color adaptativo automático (sat ratio {_sat_ratio_a:.2f} -> corr {_corr_eff:.2f})", flush=True)
                        else:
                            _color_reason = "variación regional preservada" if abs(_sat_ratio_a - 1.0) > 0.35 else "color equilibrado"
                            print(f"[ImgEditor] Color OK vs original (sat ratio {_sat_ratio_a:.2f}, {_color_reason})", flush=True)
                    except Exception as _e_color:
                        print(f"[ImgEditor] Color adaptativo automático omitido: {_e_color}", flush=True)
                except Exception as _e_adq:
                    print(f"[ImgEditor] Auto-acabado omitido: {_e_adq}", flush=True)

            # Una muestra de difusión no puede sustituir silenciosamente a las
            # personas de la foto. Para edits no estructurales se valida con los
            # embeddings locales. IMPORTANTE: RetinaFace sobre una salida de
            # difusión reescalada puede devolver muchos falsos positivos (ojos,
            # texturas y manos); exigir el MISMO número bruto de detecciones
            # convertía ese ruido en un veto y descartaba una imagen que sí
            # conservaba a todas las personas originales. La condición segura es
            # una asignación 1:1 de cada rostro de origen a uno del candidato.
            if (result is not None and source_faces and not high_structural
                    and not requires_scene_regeneration):
                try:
                    from roop.face_util import get_face_analyser, detect_faces_robust
                    _cand_bgr = cv2.cvtColor(np.array(result.convert("RGB")), cv2.COLOR_RGB2BGR)
                    _cand_faces = detect_faces_robust(_cand_bgr, get_face_analyser(), 0.20) or []
                    _source_count, _candidate_count = len(source_faces), len(_cand_faces)
                    _source_embeddings = []
                    for _src_idx, _src_face in enumerate(source_faces):
                        _src_emb = np.asarray(getattr(_src_face, "embedding", []), dtype=np.float32).flatten()
                        if _src_emb.size:
                            _src_norm = float(np.linalg.norm(_src_emb))
                            if _src_norm > 1e-8:
                                _source_embeddings.append((_src_idx, _src_emb / _src_norm))

                    # Greedy descending assignment is enough for the very small
                    # number of people in an edit and avoids letting one candidate
                    # falsely "preserve" two different source identities.
                    _identity_pairs = []
                    for _src_idx, _src_emb in _source_embeddings:
                        for _cand_idx, _cand_face in enumerate(_cand_faces):
                            _cand_emb = np.asarray(getattr(_cand_face, "embedding", []), dtype=np.float32).flatten()
                            if _cand_emb.size:
                                _cand_norm = float(np.linalg.norm(_cand_emb))
                                if _cand_norm > 1e-8:
                                    _identity_pairs.append((float(np.dot(_src_emb, _cand_emb / _cand_norm)), _src_idx, _cand_idx))
                    _identity_pairs.sort(reverse=True, key=lambda _item: _item[0])
                    _matched_sources, _matched_candidates, _identity_scores = set(), set(), []
                    for _sim, _src_idx, _cand_idx in _identity_pairs:
                        if (_sim < 0.25 or _src_idx in _matched_sources
                                or _cand_idx in _matched_candidates):
                            continue
                        _matched_sources.add(_src_idx)
                        _matched_candidates.add(_cand_idx)
                        _identity_scores.append(_sim)

                    _identity_floor = min(_identity_scores) if _identity_scores else 0.0
                    _has_identity_evidence = bool(_source_embeddings)
                    _all_sources_preserved = (
                        len(_matched_sources) == len(_source_embeddings)
                        if _has_identity_evidence else _candidate_count >= _source_count
                    )
                    # Un embedding facial es una señal excelente para proteger a
                    # terceras personas, pero puede caer por debajo de su umbral
                    # tras una edición LOCAL de la propia cara (pelo, expresión,
                    # maquillaje, etc.). En una foto de una sola persona no se
                    # descarta el resultado solo por ese cambio si una máscara
                    # focal toca la cara y el detector confirma que la cara sigue
                    # en la misma geometría. No hay vocabulario de prompts: se
                    # mide máscara, posición y escala.
                    _local_face_geometry_ok = False
                    if (
                        not _all_sources_preserved
                        and _source_count == 1
                        and _candidate_count == 1
                        and (final_mask is not None or locals().get("_longcat_focal_mask") is not None)
                    ):
                        try:
                            _integrity_mask = final_mask if final_mask is not None else locals().get("_longcat_focal_mask")
                            _mask_img = _integrity_mask.convert("L") if hasattr(_integrity_mask, "convert") else _integrity_mask
                            _mask_arr = np.asarray(_mask_img)
                            if _mask_arr.ndim == 3:
                                _mask_arr = _mask_arr[:, :, 0]
                            _mh, _mw = _mask_arr.shape[:2]
                            _mx = _mw / max(1.0, float(img.width))
                            _my = _mh / max(1.0, float(img.height))
                            _src_box_mask = np.asarray(source_faces[0].bbox, dtype=np.float32)
                            _x1, _y1, _x2, _y2 = _src_box_mask
                            _x1 = max(0, min(_mw, int(round(_x1 * _mx))))
                            _x2 = max(0, min(_mw, int(round(_x2 * _mx))))
                            _y1 = max(0, min(_mh, int(round(_y1 * _my))))
                            _y2 = max(0, min(_mh, int(round(_y2 * _my))))
                            _mask_global = float(np.mean(_mask_arr > 20))
                            _mask_face = (
                                float(np.mean(_mask_arr[_y1:_y2, _x1:_x2] > 20))
                                if _x2 > _x1 and _y2 > _y1 else 0.0
                            )
                            _src_box_result = np.asarray(source_faces[0].bbox, dtype=np.float32).copy()
                            _src_box_result[[0, 2]] *= result.width / max(1.0, float(img.width))
                            _src_box_result[[1, 3]] *= result.height / max(1.0, float(img.height))
                            _cand_box = np.asarray(_cand_faces[0].bbox, dtype=np.float32)
                            _src_center = (_src_box_result[:2] + _src_box_result[2:]) * 0.5
                            _cand_center = (_cand_box[:2] + _cand_box[2:]) * 0.5
                            _center_delta = float(np.linalg.norm(_src_center - _cand_center)) / max(
                                1.0, float(np.hypot(result.width, result.height))
                            )
                            _src_area = max(1.0, float((_src_box_result[2] - _src_box_result[0]) * (_src_box_result[3] - _src_box_result[1])))
                            _cand_area = max(1.0, float((_cand_box[2] - _cand_box[0]) * (_cand_box[3] - _cand_box[1])))
                            _scale_ratio = float(np.sqrt(_cand_area / _src_area))
                            _local_face_geometry_ok = (
                                _mask_global <= 0.45 and _mask_face >= 0.20
                                and _center_delta <= 0.12 and 0.45 <= _scale_ratio <= 2.20
                            )
                            if _local_face_geometry_ok:
                                print(
                                    f"[ImgEditor] Integridad geométrica OK para edición facial local "
                                    f"(máscara cara={_mask_face:.0%}, centro={_center_delta:.3f}, escala={_scale_ratio:.2f}); "
                                    "embedding no concluyente permitido",
                                    flush=True,
                                )
                        except Exception as _geometry_exc:
                            print(f"[ImgEditor] Integridad geométrica no concluyente: {_geometry_exc}", flush=True)
                    if _local_face_geometry_ok:
                        _all_sources_preserved = True
                    if not _all_sources_preserved:
                        _reason = (
                            f"identidad preservada {len(_matched_sources)}/{len(_source_embeddings)} "
                            f"(mínima {_identity_floor:.2f})"
                            if _has_identity_evidence
                            else f"caras {_source_count}→{_candidate_count} sin embeddings para comprobar identidad"
                        )
                        # La verificación sirve para INFORMAR, no para retirar una
                        # edición que el usuario acaba de pedir. ComfyUI ya produjo
                        # el archivo: ocultarlo convertía minutos de GPU en un
                        # resultado vacío sin autorización del usuario.
                        _integrity_note = f"Aviso de integridad: {_reason}."
                        msg = f"{msg} | {_integrity_note}" if msg else _integrity_note
                        print(
                            f"[ImgEditor] Integridad visual no concluyente; resultado entregado ({_reason})",
                            flush=True,
                        )
                    _detector_note = ""
                    if _candidate_count != _source_count:
                        _detector_note = f", detector {_source_count}→{_candidate_count} (extras no usados)"
                    print(
                        f"[ImgEditor] Integridad visual "
                        f"{'AVISO' if not _all_sources_preserved else 'OK'} "
                        f"(caras={_source_count}, "
                        f"identidad mínima={_identity_floor:.2f}{_detector_note})",
                        flush=True,
                    )
                except Exception as _integrity_exc:
                    # La validación es una red de seguridad: si un detector no
                    # está disponible no debe impedir una edición funcional.
                    print(f"[ImgEditor] Integridad visual omitida: {_integrity_exc}", flush=True)

            if result is not None and enhance_faces and not quality_enhance:
                skip_cf = False
                try:
                    from roop.img_editor.top_tier_finish import should_edit_top_finish
                    skip_cf = bool(analysis and should_edit_top_finish(analysis))
                    # Si el polish Full se saltó por VRAM, CodeFormer SÍ corre.
                    if skip_cf and locals().get("_low_vram_polish", False):
                        skip_cf = False
                except Exception:
                    skip_cf = False
                if not skip_cf:
                    # La decisión facial debe usar la medición DEL resultado
                    # actual, no la de un edit anterior. CodeFormer es
                    # restauración, no un filtro estético: en una imagen ya
                    # nítida añade textura sintética y puede afectar falsos
                    # positivos de detección.
                    qa = _face_quality_analysis or polish_analysis or {}
                    face_scores = qa.get("scores", {}) if isinstance(qa, dict) else {}
                    face_sharpness = float(face_scores.get("sharpness", 0.0))
                    face_restore = float(face_scores.get("restore", 1.0))
                    face_poster = float(face_scores.get("poster", 0.0))
                    if ((face_sharpness >= 0.72 and face_restore <= 0.30)
                            or face_poster >= 0.45 or face_restore >= 0.70):
                        print(
                            f"[ImgEditor] CodeFormer omitido: resultado no apto para restauración facial "
                            f"(nit={face_sharpness:.2f}, rest={face_restore:.2f}, poster={face_poster:.2f})",
                            flush=True,
                        )
                    else:
                        face_cap = None
                        if qa.get("profile"):
                            face_cap = qa["profile"].get("face_enhance_cap")
                        result = self._enhance_faces(
                            result,
                            progress_callback=progress_callback,
                            quality_mode=False,
                            face_enhance_cap=face_cap,
                        )

            # El acabado global puede modificar personas y objetos fuera de una
            # edición localizada incluso después del composite inicial. Restaurar
            # el exterior al FINAL mantiene cada paso de una edición compuesta
            # limitado a su región medible.
            if (result is not None and final_mask is not None
                    and (locals().get("_local_focal_generated", False)
                         or background_focus or manual_focal_mask
                         or (_compound_depth > 0 and bool(_compound_region_hint.strip())))
                    and not body_transform):
                try:
                    _preserve_source = img.convert("RGB")
                    if _preserve_source.size != result.size:
                        _preserve_source = _preserve_source.resize(result.size, Image.Resampling.LANCZOS)
                    _preserve_mask = final_mask.convert("L")
                    if _preserve_mask.size != result.size:
                        _preserve_mask = _preserve_mask.resize(result.size, Image.Resampling.BILINEAR)
                    _preserve_mask = self._feather_mask(
                        _preserve_mask, amount=6 if locals().get("_local_focal_generated", False) else (8 if _compound_scene_background else 16)
                    )
                    result = Image.composite(result.convert("RGB"), _preserve_source, _preserve_mask)
                    print("[ImgEditor] Exterior de la región restaurado tras el acabado", flush=True)
                except Exception as _preserve_error:
                    print(f"[ImgEditor] Restauración exterior omitida: {_preserve_error}", flush=True)

            if (result is not None and final_mask is not None
                    and _compound_depth > 0 and _compound_region_hint.strip()
                    and not locals().get("_local_focal_generated", False)
                    and not _compound_scene_background):
                result, _ = self._refine_flat_region_color(
                    img, result, final_mask, _compound_foreground_mask,
                    _compound_generated_reference, _compound_measurement_mask,
                    _compound_strict_foreground_mask,
                )

            # Una selección manual tiene una expectativa objetiva adicional: el
            # resultado debe modificar de forma material la región que el usuario
            # pintó. Evita guardar una copia casi idéntica por un workflow que
            # ignoró la máscara; no depende del contenido del prompt.
            if result is not None and manual_focal_mask and final_mask is not None:
                try:
                    _manual_src = np.asarray(img.convert("RGB").resize(result.size, Image.Resampling.LANCZOS), dtype=np.int16)
                    _manual_dst = np.asarray(result.convert("RGB"), dtype=np.int16)
                    _manual_arr = np.asarray(final_mask.convert("L").resize(result.size, Image.Resampling.NEAREST)) > 128
                    if _manual_arr.any():
                        _manual_diff = np.abs(_manual_dst - _manual_src)
                        _manual_mae = float(_manual_diff[_manual_arr].mean())
                        _manual_changed = float((_manual_diff.max(axis=2)[_manual_arr] > 20).mean())
                        print(
                            f"[ImgEditor] Cambio medido en selección manual: "
                            f"mae={_manual_mae:.2f}, píxeles>20={_manual_changed:.1%}",
                            flush=True,
                        )
                        if _manual_mae < 2.0 and _manual_changed < 0.01:
                            note = "Aviso: no se detectó un cambio material en la selección manual."
                            print(f"[ImgEditor] {note}", flush=True)
                            self._last_instruction_compliance = {
                                "available": True,
                                "compliant": False,
                                "unmet": "la selección manual no cambió materialmente",
                            }
                            msg = f"{msg} | {note}" if msg else note
                except Exception as _manual_check_exc:
                    print(f"[ImgEditor] Medición de selección manual omitida: {_manual_check_exc}", flush=True)

            # Global finishing can recolor the heads already preserved in the
            # scene composite. Restore them once more on the delivered image.
            if result is not None and _scene_head_preserve_mask is not None:
                _scene_source = img.convert("RGB")
                _scene_mask = _scene_head_preserve_mask
                if _scene_source.size != result.size:
                    _scene_source = _scene_source.resize(result.size, Image.Resampling.LANCZOS)
                if _scene_mask.size != result.size:
                    _scene_mask = _scene_mask.resize(result.size, Image.Resampling.BILINEAR)
                result = Image.composite(_scene_source, result.convert("RGB"), _scene_mask)
                print("[ImgEditor] Cabezas originales restauradas tras el acabado", flush=True)

            # The local visual verifier has called unchanged hair a success in
            # real follow-up edits. Measure the named scalp independently.
            # v5.1213: el hueco de cabeza (Fix F) entrega el mismo cambio de
            # apariencia anclado a la cabeza que el plan focal, y la receta
            # anclada conserva el pelo que el modelo todavía ve — la medición
            # cubre ambas rutas y el reintento mide cuánto queda.
            _hair_plan_zone = None
            if result is not None and locals().get("_focal_head_plan", False):
                _hair_plan_zone = final_mask
            elif result is not None and locals().get("_head_hole_plan", False):
                _hair_plan_zone = locals().get("_head_hole_zone")
            if (_hair_plan_zone is not None
                    and self._is_bald_request(prompt_pre_rewrite)):
                _hair_masker = None
                try:
                    _hair_masker = self._get_clipseg_masker()
                    _hair_coverage = self._bald_hair_coverage(
                        img, result, _hair_plan_zone, _hair_masker,
                    )
                    if _hair_coverage is not None:
                        _hair_before, _hair_after = _hair_coverage
                        print(
                            f"[ImgEditor] Pelo en cabeza solicitada: "
                            f"{_hair_before:.1%}→{_hair_after:.1%}", flush=True,
                        )
                        if (_hair_before >= 0.04
                                and _hair_after > max(0.04, 0.60 * _hair_before)):
                            _unmet = "el pelo de la persona solicitada sigue visible"
                            self._last_instruction_compliance = {
                                "available": True, "compliant": False,
                                "unmet": _unmet, "source": "hair_mask",
                                "hair_before": _hair_before,
                                "hair_after": _hair_after,
                            }
                            msg = f"{msg} | Aviso: {_unmet}" if msg else f"Aviso: {_unmet}"
                except Exception as _hair_check_error:
                    print(f"[ImgEditor] Medición de pelo omitida: {_hair_check_error}", flush=True)
                finally:
                    if _hair_masker is not None:
                        _hair_masker.unload()

            # A diffusion sample is not evidence that it followed the request.
            # Verify the rendered candidate against the full instruction, with
            # an independent local VLM.  Failure is reported to the retry loop;
            # an unavailable/inconclusive verifier never discards a result.
            if result is not None and prompt_raw and not quality_only:
                try:
                    from roop.img_editor.instruction_verifier import get_instruction_verifier
                    verification = get_instruction_verifier().verify(
                        result, prompt
                    )
                    # A VLM verdict never replaces an objective failure from
                    # the manual selection or residual-hair measurement.
                    if (self._last_instruction_compliance is None
                            or self._last_instruction_compliance.get("compliant") is not False):
                        self._last_instruction_compliance = verification
                    if verification.get("compliant") is False:
                        unmet = str(verification.get("unmet") or "").strip()
                        # PASS/FAIL sin explicación es una señal auxiliar, no
                        # evidencia suficiente para afirmar qué parte falló.
                        verify_note = (
                            f"Verificación visual: no confirmada ({unmet})"
                            if unmet else "Verificación visual: no confirmada"
                        )
                        msg = f"{msg} | {verify_note}" if msg else verify_note
                        # Un veredicto negativo aislado es solo una señal. Las
                        # métricas de poster/restauración son heurísticas de
                        # acabado, no una prueba de que la muestra sea inválida
                        # (LongCat las eleva con frecuencia en imágenes naturales).
                        # Nunca deben tirar 10-15 minutos de generación; se
                        # conserva el candidato y el aviso llega al resultado.
                        if _candidate_poster >= 0.55 or _candidate_restore >= 0.75:
                            _quality_reason = (
                                f"poster={_candidate_poster:.2f}, restauración={_candidate_restore:.2f}"
                            )
                            print(
                                f"[ImgEditor] Verificación visual negativa + métrica de calidad alta ({_quality_reason}); "
                                "se conserva el candidato (señales no concluyentes)",
                                flush=True,
                            )
                except Exception as _verify_exc:
                    print(f"[ImgEditor] Verificación visual omitida: {_verify_exc}", flush=True)

            if result is not None:
                return result, msg, final_mask
            return None, msg if msg else "Error: no se generó resultado", final_mask

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}", final_mask

def _calculate_autoregressive_iterations(analysis: Dict, engine: str) -> int:
    """
    Calcula el número óptimo de iteraciones autoregresivas basándose en el análisis semántico.
    - Modo Hart: 3-5 iteraciones dependiendo de la complejidad
    - Otros motores: 1 iteración (solo paso único)
    """
    is_hart = engine == "hart"
    if not is_hart:
        return 1

    # Basado en la magnitud y keywords del análisis
    magnitude = analysis.get("magnitude", 0.5)
    body_transform = bool(analysis.get("body_transform", False))

    # Si hay body_transform, más iteraciones para cambios corporales
    if body_transform:
        if magnitude >= 0.7:
            return 5
        elif magnitude >= 0.5:
            return 4
        else:
            return 3

    # Para cambios generales (POV, ángulo, ropa)
    if magnitude >= 0.7:
        return 5
    elif magnitude >= 0.5:
        return 4
    else:
        return 3

_manager = None
def get_img_editor_manager() -> ImgEditorManager:
    global _manager
    if _manager is None:
        _manager = ImgEditorManager()
    return _manager
