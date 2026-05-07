#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
from typing import Optional, List, Tuple
import cv2

class CLIPSegMasker:
    """
    Maneja la generación de máscaras basadas en texto (Prompt-to-Mask)
    utilizando CLIPSeg.
    """
    
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._initialized = False

    def initialize(self) -> Tuple[bool, str]:
        if self._initialized:
            return True, "Ya inicializado"
            
        try:
            from clip.clipseg import CLIPDensePredT
            
            # Intentar cargar el modelo (ViT-B/16 es el estándar para CLIPSeg)
            self.model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            self.model.eval()
            self.model.to(self.device)
            
            # Nota: Los pesos deben estar descargados. 
            # Si no están, CLIPSeg suele fallar al cargar el pickle de pesos.
            # Aquí asumimos que están o que el usuario los tiene.
            
            self._initialized = True
            return True, f"CLIPSeg inicializado en {self.device}"
        except Exception as e:
            print(f"[CLIPSeg] Error inicializando: {e}")
            return False, str(e)

    def generate_mask(self, image: Image.Image, prompt: str, threshold: float = 0.4) -> Optional[Image.Image]:
        """
        Genera una máscara binaria para el prompt dado.
        """
        if not self._initialized:
            ok, msg = self.initialize()
            if not ok: return None

        try:
            from torchvision import transforms
            
            # Preprocesar imagen
            transform = transforms.Compose([
                transforms.Resize((352, 352)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            img_tensor = transform(image).unsqueeze(0).to(self.device)
            
            # Generar predicción
            with torch.no_grad():
                preds = self.model(img_tensor, prompt)[0]
            
            # Post-procesar
            mask = torch.sigmoid(preds[0, 0])
            mask = mask.cpu().numpy()
            
            # Reescalar a tamaño original
            mask_img = cv2.resize(mask, (image.width, image.height))
            
            # Aplicar umbral
            binary_mask = (mask_img > threshold).astype(np.uint8) * 255
            
            # Suavizar máscara (opcional pero recomendado para inpaint)
            binary_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)
            
            return Image.fromarray(binary_mask).convert("L")
            
        except Exception as e:
            print(f"[CLIPSeg] Error generando máscara: {e}")
            return None

_masker = None
def get_clipseg_masker() -> CLIPSegMasker:
    global _masker
    if _masker is None:
        _masker = CLIPSegMasker()
    return _masker
