#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, threading, cv2, re
from typing import Optional, Tuple, Dict
from PIL import Image
import numpy as np
import roop.globals
from roop.utils import get_vram_gb

from roop.utilities import resolve_relative_path
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
        has_quality = bool(analysis.get("has_quality_request", False))
        body_transform = bool(analysis.get("body_transform", False))
        body_intensity = float(analysis.get("body_transform_intensity", 0.0))

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

        high_structural = bool(analysis.get("high_structural", False))
        structural_bias = str(analysis.get("structural_bias", "neutral"))
        background_focus = bool(analysis.get("background_focus", False))
        quality_hybrid = bool(analysis.get("quality_hybrid", False))
        quality_primary = bool(analysis.get("quality_primary", False))

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
        background_focus = bool(analysis.get("background_focus", False))
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
                preferred_engine, body_direction, body_intensity, mag, vram_gb
            )

        # --- 2. POV/postura/acción física: regeneración fuerte sin ancla ---
        # Una postura o interacción no existe en la fuente. Turbo (12 pasos) es
        # útil para retoques, pero con reference_latents conserva la anatomía
        # original e ignora el verbo. En 8GB Full Q4 usa offload y es lento, pero
        # deja al prompt reconstruir la relación espacial. El bypass se activa
        # más abajo con pose_change.
        if (angle_change or pose_change) and not body_transform:
            if preferred_engine == "longcat_full":
                return None, None, ""
            return (
                "longcat_full", _LONGCAT_FULL_VERSION,
                f"pov={angle_change} pose={pose_change}: {preferred_engine} -> longcat_full (acción/postura, sin ancla)"
            )
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

        # --- 3. LongCat Turbo para edits fuertes: upgrade a Full ---
        if preferred_engine in ("longcat", "imagine") and mag >= 0.75:
            if vram_gb >= _LONGCAT_MIN_VRAM:
                return (
                    "longcat_full", _LONGCAT_FULL_VERSION,
                    f"mag={mag:.2f} >= 0.75: longcat turbo -> longcat_full (calidad)"
                )

        # --- 3b. Klein 4B para edits fuertes: el distilled + ReferenceLatent
        # conserva demasiado (misma evidencia que motivó el bypass uncover:
        # regenera lo mismo aunque el prompt pida cambio). Klein es débil en
        # instruction-following (v5.1097) → mismo umbral needs_strong que los
        # demás débiles (0.65, log real: rewriter mag=0.70 se colaba entre
        # 0.65 y 0.75 y klein corría nativo sin seguir el prompt).
        # Solo magnitud + tabla de capacidades, cero vocabulario.
        if preferred_engine in ("klein", "klein_base") and mag >= 0.65:
            if vram_gb >= _LONGCAT_MIN_VRAM:
                return (
                    "longcat_full", _LONGCAT_FULL_VERSION,
                    f"mag={mag:.2f} >= 0.65: klein -> longcat_full (edit fuerte)"
                )
            # v5.1118 — 8GB: mag>=0.75 → LongCat (bypass sin ancla).
            # Log real 07:22: no-body mag=0.80 → kontext+offset = copy machine
            # (std=52, no obedeció). Kontext queda para el rango moderado.
            # v5.1120 — FULL en vez de Turbo.
            elif mag >= 0.75:
                return (
                    "longcat_full", _LONGCAT_FULL_VERSION,
                    f"mag={mag:.2f} >= 0.75: klein -> longcat_full (bypass sin ancla, 8GB)"
                )
            else:
                return (
                    "kontext", _KONTEXT_VERSION,
                    f"mag={mag:.2f} >= 0.65: klein -> kontext (edit fuerte, 8GB)"
                )

        # --- 4. Quality improvements: classical pipeline, sin override ---
        if quality_only:
            return None, None, ""

        # --- 5. Background: cualquier engine funciona, LongCat/Kontext mejores ---
        if background_focus and preferred_engine in _WEAK_FOR_STRONG:
            if vram_gb >= _LONGCAT_MIN_VRAM:
                return (
                    "longcat_full", _LONGCAT_FULL_VERSION,
                    f"background edit: {preferred_engine} -> longcat_full"
                )
            else:
                return (
                    "kontext", _KONTEXT_VERSION,
                    f"background edit: {preferred_engine} -> kontext (8GB)"
                )

        # --- 6. Default: mantener engine preferido ---
        return None, None, ""

    @staticmethod
    def _select_longcat_version(client, magnitude: float, body_transform: bool) -> str:
        """Elige LongCat sin alternar Full↔Turbo entre ediciones consecutivas.

        En 8GB ambas variantes comparten el encoder Qwen-VL enorme. Degradar de
        Full a Turbo por una pequeña variación de magnitud descargaba el modelo
        residente y provocaba otra carga/offload de varios minutos. Se conserva
        Full una vez cargado; Turbo solo asciende a Full ante un cambio fuerte.
        """
        full = "LongCat-Image-Edit-Q4_K_S.gguf"
        turbo = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
        desired = full if body_transform or float(magnitude) >= 0.70 else turbo
        active = getattr(client, "_flux_version", None) if getattr(client, "_loaded", False) else None
        if active == full and desired == turbo:
            print(
                "[ImgEditor] LongCat residente Full conservado (evita recarga Full→Turbo)",
                flush=True,
            )
            return full
        return desired

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
        background_focus = mask_target == "background" and not body_transform
        scene_change = background_focus or (
            high_structural and structural_bias not in ("remove", "add")
        )

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
                    # v5.1126 — "identity identical" pelea contra CAMBIOS DE APARIENCIA
                    # (log 11:37 'the boy must be bald': el pelo es parte de la
                    # identidad percibida → el modelo resolvió a favor del lock
                    # y no ejecutó). En bypass el TEXTO es la única autoridad:
                    # se preserva a las PERSONAS y su PARECIDO, jamás el atributo
                    # pedido. La instrucción SIEMPRE puede cambiar apariencia,
                    # pose, POV o posición (editor versátil, cero vocabulario).
                    if angle_change or pose_change:
                        base = f"Instruction: {raw}. Keep the same people present with a strong likeness to them, and the same scene location. The pose, framing and appearance may change as required by the instruction. Apply the requested change as completely, strongly and obviously as possible."
                    else:
                        base = f"Instruction: {raw}. Keep the same people present with a strong likeness to them, and the exact same framing, composition and camera. The appearance may change as required by the instruction. Apply the requested change as completely, strongly and obviously as possible."
                else:
                    base = f"Instruction: {raw}. Keep the same background and composition. Apply the requested change as completely, strongly and obviously as possible."
                # v5.1096 — sufijo de ESTADO final por dirección: los modelos img2img
                # obedecen mejor descripciones de estado ("está desnuda") que acciones
                # ("debe desnudarse"). El Klein 4B con ReferenceLatent conserva tanto
                # que sin esto regenera el mismo vestido (log 11:27, 2 runs vestida).
                # v5.1149 — el sufijo de ESTADO es una afirmación FUERTE ("fully
                # nude"): solo se añade con señal corporal FUERTE (intensity≥0.30,
                # umbral sobre el score del analyzer, cero vocabulario). Con señal
                # residual (log 21:31: 'barefoot' → intensity 0.13 → uncover) el
                # sufijo ALUCINABA un desnudo no pedido que competía con las
                # condiciones reales y el modelo solo cumplía una. Sin sufijo, el
                # texto del usuario manda tal cual.
                if body_direction == "uncover" and float(body_intensity) >= 0.30:
                    base += " The person is fully nude, bare skin, no clothing at all."
                elif body_direction == "cover" and float(body_intensity) >= 0.30:
                    base += " The person is fully dressed, body covered with warm clothing."
                elif body_transform:
                    print(f"[ImgEditor] Sufijo de estado omitido (señal corporal débil intensity={float(body_intensity):.2f} < 0.30 — el texto del usuario manda)", flush=True)
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
                # v5.1126 — rama DEDICADA para bypass con pose/POV: el framing SÍ
                # puede cambiar (la vista/postura destino no existe en la fuente).
                base = f"Instruction: Edit this exact photo: {raw}. Keep the same people present with a strong likeness to them, and the same scene location. The pose, framing and appearance may change as required by the instruction. Apply the requested change as completely, strongly and obviously as possible."
            else:
                # v5.1126 — bypass sin pose: framing anclado, apariencia libre
                # (apariencia ≠ composición: el pelo/gafas/ropa cambian, la cámara no).
                base = f"Instruction: Edit this exact photo: {raw}. Keep the same people present with a strong likeness to them, and the exact same framing, composition and camera. The appearance may change as required by the instruction. Apply the requested change as completely, strongly and obviously as possible."
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
        if not np.any(mask_center): mask_center = mask_arr > 0.5
        
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

    def _head_mask_from_region_plan(self, image: Image.Image, faces, region_mask: Image.Image) -> Optional[Image.Image]:
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
            print(f"[ImgEditor] Región focal → cabeza por solape medido (score={best_score:.2f})", flush=True)

            return self._head_mask_from_faces(image, [best_face])
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
    def _crop_focal_context(image: Image.Image, mask: Image.Image, max_area: float = 0.35):
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
                    if context < 0.35:
                        print(
                            f"[ImgEditor] Contexto focal ajustado a {context:.0%} "
                            f"(crop={crop_area / (iw * ih):.1%} de la foto)", flush=True
                        )
                    return image.crop(box), mask.crop(box).convert("L"), box
            print(
                f"[ImgEditor] Crop focal rechazado: el núcleo ocupa más del {max_area:.0%} de la foto",
                flush=True,
            )
            return None
        except Exception:
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
                or any(len(part.split()) < 2 for part in clauses)):
            return []
        return clauses

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
            predicate = re.search(
                r"\b(?:debe(?:n)?|tiene(?:n)?\s+que|must|should|"
                r"has\s+to|have\s+to|is|are|was|were|goes|go|"
                r"es|son|est[aá](?:n)?|va|van)\b",
                region, flags=re.I,
            )
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
                    has_predicate = bool(re.search(
                        r"\b(?:debe(?:n)?|must|should|is|are|was|were|"
                        r"goes|go|es|son|est[aá](?:n)?|va|van)\b",
                        owner, flags=re.I,
                    ))
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
            pattern = (
                r"\b(?:debe(?:n)?|tiene(?:n)?\s+que|must|should|"
                r"has\s+to|have\s+to|is|are|was|were|goes|go|"
                r"es|son|est[aá](?:n)?|va|van)\b"
            )
            matches = list(re.finditer(pattern, clause, flags=re.I))
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

    @staticmethod
    def _visual_subject_description(clause: str) -> str:
        """Keep the visible subject description, including pose and location.

        The predicate describes the edit. The prefix is only a visual query;
        its meaning is checked against pixels by the segmenter before use.
        """
        text = (clause or "").strip()
        pattern = (
            r"\b(?:debe(?:n)?|tiene(?:n)?\s+que|must|should|"
            r"has\s+to|have\s+to|is|are|was|were|goes|go|"
            r"es|son|est[aá](?:n)?|va|van)\b"
        )
        matches = list(re.finditer(pattern, text, flags=re.I))
        match = matches[0] if matches else None
        if match and len(matches) > 1 and re.search(
            r"\b(?:who|that|which|que)\b", text[:match.start()], flags=re.I
        ):
            match = matches[1]
        return text[:match.start()].strip(" ,.;") if match else ""

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
                        source_faces = list(faces)
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
        else:
            prompt = translate_prompt(prompt_raw)
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
        if (prompt and _compound_depth == 0
                and mask_image is None and not quality_only
                and quality_style != "solo_mejora"
                and not (quality_hybrid and quality_use_generative is True)
                and engine in ("longcat", "imagine")
                and 0 < get_vram_gb() <= 10):
            try:
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                self._release_local_gpu_before_comfy()
                fast_client = get_flux_edit_comfy_client()
                loaded, load_msg = fast_client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
                if loaded:
                    if quality_hybrid:
                        fast_prompt = (
                            "Edit this exact photo with minimal changes. "
                            f"Apply the requested change: {prompt}. "
                            "Keep every person, object, room detail, camera view "
                            "and composition that the request does not change "
                            "exactly as in the input. Do not alter anything else. "
                            "Keep people anatomically coherent, with no missing or extra limbs."
                        )
                    else:
                        fast_prompt = (
                            "Instruction: Edit this exact photo. "
                            f"Apply every requested change: {prompt}. "
                            "Make all requested changes clearly visible. "
                            "Keep people anatomically coherent, with no missing or extra limbs."
                        )
                    if negative_prompt:
                        # Turbo uses CFG=1 and does not encode a separate
                        # negative branch, so keep user exclusions in text.
                        fast_prompt += f" Avoid: {negative_prompt}."
                    fast_result, fast_msg = fast_client.generate(
                        image=img, prompt=fast_prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=12, guidance_scale=1.0,
                        denoise=1.0, seed=seed, no_kontext=True,
                        lora_name=lora_name, lora_strength=lora_strength,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )
                    if fast_result is not None:
                        output = fast_result.image.convert("RGB")
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
                            return output, f"{fast_msg} | Turbo + {finish_note}{pov_warning}", None
                        if output.size != img.size:
                            output = output.resize(img.size, Image.Resampling.LANCZOS)
                        warning = (
                            " | POV no verificado: el modelo puede conservar el ángulo original"
                            if explicit_viewpoint_request else ""
                        )
                        print(f"[ImgEditor] Prompt traducido enviado a Turbo; cumplimiento visual no verificado{warning}", flush=True)
                        return (output,
                                f"{fast_msg} | prompt traducido enviado a Turbo; cumplimiento visual no verificado{warning}",
                                None)
                else:
                    print(f"[ImgEditor] Ruta rápida no cargada: {load_msg}", flush=True)
            except Exception as fast_error:
                print(f"[ImgEditor] Ruta rápida falló: {fast_error}", flush=True)
            if cancel_check and cancel_check():
                return None, "Cancelado", None
            print("[ImgEditor] Ruta rápida no disponible; se usa la ruta normal", flush=True)

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
            print(f"[ImgEditor] POV puro (sin señal corporal): body_transform NO forzado — regeneración de vista con img2img fuerte", flush=True)
        elif pose_change and not quality_only and not body_transform:
            print(f"[ImgEditor] POSE puro (sin señal corporal): body_transform NO forzado — regeneración de postura con img2img fuerte", flush=True)
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
                    print(f"[ImgEditor] Rewrite-text descartado (entrada no-EN): traducción completa intacta", flush=True)
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

        if num_inference_steps: params["num_inference_steps"] = num_inference_steps
        if guidance_scale: params["guidance_scale"] = guidance_scale
        if denoise: params["denoise"] = denoise

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
             print(f"[ImgEditor] Edición global detectada (sin máscara)")

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
                        version = "LongCat-Image-Edit-Q4_K_S.gguf"
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
                        if num_inference_steps: params["num_inference_steps"] = num_inference_steps
                        if guidance_scale: params["guidance_scale"] = guidance_scale
                        if denoise: params["denoise"] = denoise
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
                        print(f"[ImgEditor] Generando con HART (autoregresivo) - prompt puro...")
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
                            # Fallback VRAM 8GB: si el motor pedido no carga (falta/OOM),
                            # reintentar con Klein 4B (~5GB total) antes de fallar.
                            # Nota: LongCat usa qwen_2.5_vl_7b_fp8 (8.74GB solo CLIP) -> en 8GB
                            # va con offload CPU, lento y propenso a gris/OOM/colgado.
                            # v5.1090: fallback a Klein TAMBIEN para body_transform.
                            # Un edit parcial con Klein (que termina) es mejor que un
                            # colgado de 300s+ con LongCat en 8GB.
                            if engine != "klein":
                                try:
                                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client as _klein_cli
                                    fb = _klein_cli()
                                    ok2, msg2 = fb.load(flux_version="flux-2-klein-base-4b-Q4_K_S.gguf")
                                    if ok2:
                                        print(f"[ImgEditor] Fallback VRAM: {engine} no cargo ({msg}) -> Klein 4B" + (" (body: calidad menor pero sin colgado)" if body_transform else ""))
                                        client = fb
                                        engine = "klein"
                                        version = "flux-2-klein-base-4b-Q4_K_S.gguf"
                                        params = {"denoise": 1.0, "num_inference_steps": 4, "guidance_scale": 1.0, "mode": "img2img"}
                                        success, msg = True, msg2
                                        if not success:
                                            return None, msg
                                    else:
                                            return None, msg, final_mask
                                except Exception:
                                    return None, msg, final_mask
                            else:
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
                            print("[ImgEditor] POV bypass: Qwen-VL ve la foto, sin ancla → alucina vista nueva", flush=True)
                        # Postura nueva: mismo bypass (el ancla conserva la pose).
                        if (not body_transform) and analysis.get("pose_change", False) and engine in ("longcat", "imagine", "longcat_full"):
                            gen_params["no_kontext"] = True
                            gen_params["drop_edit_image_ref"] = False
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
            if (result is not None and final_mask is not None
                    and locals().get("_focal_head_plan", False)
                    and self._is_bald_request(prompt_pre_rewrite)):
                _hair_masker = None
                try:
                    _hair_masker = self._get_clipseg_masker()
                    _hair_coverage = self._bald_hair_coverage(
                        img, result, final_mask, _hair_masker,
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
