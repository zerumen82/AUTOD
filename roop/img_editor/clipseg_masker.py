#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple
import cv2


class CLIPSegMasker:
    """
    Maneja la generación de máscaras basadas en texto (Prompt-to-Mask)
    utilizando CLIPSeg.
    """

    # Candidatos de pesos (el repo original de CLIPSeg usa rd64-uni-refined.pth).
    # Rutas relativas al PROYECTO (derivadas de __file__) + cwd: el servidor puede
    # lanzarse desde otro cwd y la ruta relativa simple no resolvería.
    _PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _WEIGHT_CANDIDATES = [
        (os.path.join(_PROJ_ROOT, "models", "CLIP", "rd64-uni-refined.pth"), "uni"),
        (os.path.join(_PROJ_ROOT, "models", "CLIP", "rd64-uni.pth"), "uni"),
        ("models/CLIP/rd64-uni-refined.pth", "uni"),
        ("models/CLIP/clipseg_weights/rd64-uni-refined.pth", "uni"),
    ]

    def __init__(self, backbone_path=None):
        self.model = None
        self.backbone_path = backbone_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False

    def initialize(self) -> Tuple[bool, str]:
        if self._initialized:
            return True, "Ya inicializado"

        try:
            from clip.clipseg import CLIPDensePredT

            # Intentar cargar el modelo (ViT-B/16 es el estándar para CLIPSeg)
            self.model = CLIPDensePredT(
                version='ViT-B/16', reduce_dim=64, complex_trans_conv=True,
                backbone_path=self.backbone_path,
            )

            # v5.1158 — CRÍTICO: cargar los PESOS del modelo.
            # Antes solo se creaba la arquitectura (CLIPDensePredT sin load_state_dict)
            # → 454 keys sin inicializar → la "máscara" era ruido débil (sigmoid de un
            # modelo aleatorio ≈ 0.05 uniforme) → máscara ~vacía tras el umbral 0.4.
            # El inpaint con VAEEncodeForInpaint recibía noise_mask ≈ TODO 0 (el ruido
            # se inyecta solo dentro de la máscara; si no hay máscara no hay ruido)
            # → latente de la foto con ruido cero decodificado → IMAGEN GRIS
            # (std≈0.7, media 127 = gris 50%). Log real 2026-09-17: 4 runs gris
            # seguidas, cada una ~5.5 min de sampling tirado.
            _weights_loaded = False
            _wpath = ""
            for cand, _kind in self._WEIGHT_CANDIDATES:
                try:
                    if not os.path.isfile(cand):
                        continue
                    sd = torch.load(cand, map_location="cpu")
                    missing, unexpected = self.model.load_state_dict(sd, strict=False)
                    # El SD del refined contiene los 54 heads del decoder; el backbone CLIP
                    # (clip_model.*) se carga entrenado via clip.load() dentro de CLIPDenseBase
                    # y quedará en missing SIEMPRE. Criterio: las keys del SD que sí existen
                    # en el modelo deben haberse cargado (unexpected≈0 y missing solo del
                    # backbone clip_model/model).
                    _non_backbone_missing = [k for k in missing if not k.startswith(("clip_model", "model"))]
                    _weights_loaded = len(unexpected) == 0 and len(_non_backbone_missing) <= 5
                    _wpath = cand
                    if _weights_loaded:
                        break
                except Exception as e:
                    print(f"[CLIPSeg] Pesos {cand} no cargables: {e}")
            if _weights_loaded:
                print(f"[CLIPSeg] Pesos cargados: {_wpath}")
            else:
                print("[CLIPSeg] AVISO: pesos no encontrados en rutas conocidas — el modelo "
                      "sin pesos genera máscaras vacías. Revisa models/CLIP/rd64-uni-refined.pth")

            self.model.eval()
            self.model.to(self.device)

            self._initialized = True
            return True, f"CLIPSeg inicializado en {self.device}" + (" (con pesos)" if _weights_loaded else " (SIN pesos — máscaras no fiables)")
        except Exception as e:
            print(f"[CLIPSeg] Error inicializando: {e}")
            return False, str(e)

    @staticmethod
    def _as_image(image) -> Image.Image:
        """The segmenter resizes a PIL image. Callers also pass a float array."""
        if isinstance(image, Image.Image):
            return image
        array = np.asarray(image)
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 2:
            return Image.fromarray(array, mode="L")
        return Image.fromarray(array[..., :3], mode="RGB")

    def probability(self, image: Image.Image, prompt: str) -> Optional[np.ndarray]:
        """Mapa 0–1 del segmentador, del tamaño de la foto. Sin umbral binario."""
        image = self._as_image(image)
        if not self._initialized:
            ok, _msg = self.initialize()
            if not ok:
                return None
        try:
            from torchvision import transforms

            transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                preds = self.model(img_tensor, prompt)[0]
            heat = torch.sigmoid(preds[0, 0]).cpu().numpy()
            return cv2.resize(heat, (image.width, image.height))
        except Exception as exc:
            print(f"[CLIPSeg] Error midiendo '{prompt}': {exc}")
            return None

    def generate_mask(self, image: Image.Image, prompt: str, threshold: float = 0.4) -> Optional[Image.Image]:
        """
        Genera una máscara binaria para el prompt dado.
        """
        image = self._as_image(image)
        try:
            mask_img = self.probability(image, prompt)
            if mask_img is None:
                return None

            # Aplicar umbral
            binary_mask = (mask_img > threshold).astype(np.uint8) * 255

            # v5.1158 — Verificación de la máscara: con pesos sin cargar (o prompt
            # que CLIPSeg no ve) la máscara sale vacía o casi llena. Ambas matan
            # el inpaint (vacía = sin ruido = gris; llena = inpaint inútil).
            cov = float((binary_mask > 0).mean())
            if cov < 0.005:
                print(f"[CLIPSeg] Máscara VACÍA para '{prompt}' (cobertura={cov:.2%}) — devuelvo None (evita inpaint gris)")
                return None
            if cov > 0.97:
                print(f"[CLIPSeg] Máscara SATURADA para '{prompt}' (cobertura={cov:.2%}) — devuelvo None (inpaint inútil)")
                return None
            print(f"[CLIPSeg] Máscara OK para '{prompt}': cobertura={cov:.1%}", flush=True)

            # Suavizar máscara (opcional pero recomendado para inpaint)
            binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)

            return Image.fromarray(binary_mask).convert("L")

        except Exception as e:
            print(f"[CLIPSeg] Error generando máscara: {e}")
            return None

    def unload(self) -> None:
        """Release CLIPSeg before a ComfyUI generation on a shared 8 GB GPU.

        A text-mask prediction is a short preflight operation.  Keeping its
        ViT-B/16 model resident while LongCat loads leaves too little headroom
        and can make ComfyUI page weights between CPU and GPU on every step.
        The masker is deliberately lazy, so the next edit that needs a mask
        will initialize it again.
        """
        model = self.model
        self.model = None
        self._initialized = False
        try:
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:
                    pass
                del model
        finally:
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
        print("[CLIPSeg] Modelo descargado de VRAM (se recarga solo si otra edición necesita máscara)", flush=True)

_masker = None
def get_clipseg_masker() -> CLIPSegMasker:
    global _masker
    if _masker is None:
        _masker = CLIPSegMasker()
    return _masker
