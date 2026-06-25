#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalImagineService - Servicio estilo Grok Imagine, con soporte Autoregresivo (HART)

Proporciona una API limpia tipo "imagine":

    from roop.img_editor.imagine_local_service import LocalImagineService
    imagine = LocalImagineService(engine="autoregressive")
    result = imagine.edit(image, "undress her, realistic, keep the face")

Características "al estilo Imagine":
- Análisis semántico automático (embeddings) para magnitud y target
- Traducción automática ES->EN
- Parámetros dinámicos
- Soporte explícito para modo autoregresivo (HART)
- Fallbacks seguros
"""

from typing import Optional, Tuple, Dict
from PIL import Image
import os

from roop.img_editor.img_editor_manager import get_img_editor_manager
from roop.img_editor.prompt_translator import translate_prompt


class LocalImagineService:
    """
    Servicio local estilo Grok Imagine.
    - Sin hardcodes de acciones (usa semantic embeddings + prompt general).
    - Sin censura (pasa el prompt tal cual al modelo uncensored).
    - Default: edición fiel sobre la foto original ("misma foto + instrucción").
    - Todas las opciones siguen disponibles (puedes elegir hart etc.).
    - Optimizado para RTX 3060 Ti 8GB (usa LongCat Turbo por defecto).
    """


    SUPPORTED_ENGINES = {
        "imagine": "imagine",       # Grok Imagine style - default (máxima fidelidad a la foto + instrucción)
        "autoregressive": "hart",   # HART = Hybrid Autoregressive Transformer (generación libre)
        "hart": "hart",
        "longcat": "longcat",
        "longcat_full": "longcat_full",
        "edit": "imagine",
    }

    def __init__(self, engine: str = "imagine", use_semantic: bool = True):
        raw = (engine or "imagine").lower()
        self.engine = self.SUPPORTED_ENGINES.get(raw, "imagine")
        self.use_semantic = use_semantic
        self._manager = None
        is_ar = self.engine == "hart"
        print(f"[LocalImagine] Servicio iniciado. Engine={self.engine}")

    def _get_manager(self):
        if self._manager is None:
            self._manager = get_img_editor_manager()
        return self._manager

    def edit(
        self,
        image: Image.Image,
        instruction: str,
        enhance_faces: bool = False,
        num_steps: Optional[int] = None,
        guidance: Optional[float] = None,
        quality_mode: bool = False,
        enhance_tier: str = "hd",
        use_rewriter: bool = False,
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Edición / generación estilo Imagine.

        Args:
            image: Imagen de entrada (referencia visual)
            instruction: Instrucción natural ("undress her", "ponle ropa cyberpunk", etc)
            use_rewriter: Si True, usa LLM rewriter para expandir/mejorar instrucción

        Returns:
            (imagen_resultado, mensaje)
        """
        if image is None:
            return None, "No se proporcionó imagen"

        quality_mode = bool(quality_mode)
        instruction = (instruction or "").strip()
        if not quality_mode and not instruction:
            return None, "Instrucción vacía (o activa quality_mode)"

        if quality_mode:
            final_prompt = ""
        else:
            prompt = translate_prompt(instruction)
            if self.engine in ("longcat", "longcat_full"):
                final_prompt = f"Instruction: {prompt}"
            else:
                final_prompt = prompt

        mgr = self._get_manager()

        try:
            # Usamos el flujo con análisis local ligero automático.
            # El usuario no necesita configurar nada: sube foto + escribe instrucción = funciona.
            result, msg, _mask = mgr.generate_intelligent(
                image=image,
                prompt=final_prompt,
                engine=self.engine,
                use_rewriter=use_rewriter,
                enhance_faces=enhance_faces,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                quality_mode=quality_mode,
                enhance_tier=enhance_tier,
            )
            if result:
                return result, f"[Imagine Local - {self.engine}] {msg}"
            return None, msg or "Fallo en generación"
        except Exception as e:
            return None, f"Error en LocalImagine: {e}"

    def generate_from_prompt(self, instruction: str, width: int = 1024, height: int = 1024) -> Tuple[Optional[Image.Image], str]:
        """
        Modo puro txt2img usando el motor autoregresivo (HART).
        Útil cuando no hay imagen de referencia o se quiere generación completa.
        """
        if self.engine != "hart":
            # Forzar hart para este modo
            self.engine = "hart"

        prompt = translate_prompt(instruction)

        # Llamamos directamente al cliente HART vía manager (cuando esté cableado)
        # Por ahora delegamos al flujo del manager con engine=hart
        mgr = self._get_manager()

        # generate_intelligent espera imagen. Para pure gen usamos un placeholder o ruta especial.
        # Creamos una imagen negra de referencia como "seed visual" (el motor AR la ignora).
        placeholder = Image.new("RGB", (width, height), (128, 128, 128))

        result, msg, _ = mgr.generate_intelligent(
            image=placeholder,
            prompt=f"Instruction: {prompt}" if "longcat" in self.engine else prompt,
            engine="hart",
            use_rewriter=False,
        )
        return result, msg

    def auto_imagine(self, image: Image.Image, instruction: str) -> Tuple[Optional[Image.Image], str]:
        """
        Atajo estilo Grok Imagine: elige automáticamente el mejor motor
        según la instrucción (actualmente prioriza AR si se pidió, o LongCat).
        """
        # Por defecto usa lo que se configuró en __init__
        return self.edit(image, instruction)


# Factory conveniente
def get_local_imagine(engine: str = "imagine") -> LocalImagineService:
    """Devuelve el servicio local estilo Grok Imagine.
    - default "imagine": edición sobre la foto original (usa LongCat Turbo - funciona en 8GB)
    - "hart" / "autoregressive": modo generación autoregresiva (HART)
    Todas las opciones siguen disponibles. Sin hardcodes de intención de usuario.
    """
    return LocalImagineService(engine=engine)


# Ejemplo de uso rápido
if __name__ == "__main__":
    # Por defecto: estilo Grok Imagine (misma foto + instrucción)
    svc = LocalImagineService()                    # o engine="imagine"
    # Para modo autoregresivo puro:
    # svc = LocalImagineService(engine="autoregressive")

    print("LocalImagineService listo.")
    print("Default = Grok Imagine (edición fiel). Todas las opciones disponibles.")
