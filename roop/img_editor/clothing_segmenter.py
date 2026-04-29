# -*- coding: utf-8 -*-
"""
ClothingSegmenter - Detección automática de ropa con CLIPSeg (Versión LOCAL corregida)

Usa CLIPSeg cargado manualmente para detectar áreas de ropa en imágenes
evitando errores de Processor en Windows.
"""

import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image
from typing import Optional, Tuple, List
import logging
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prompts para detectar diferentes tipos de ropa
CLOTHING_PROMPTS = [
    "clothing",
    "shirt",
    "dress",
    "pants",
    "skirt",
    "top",
    "jacket",
    "sweater",
    "blouse",
    "jeans",
    "shorts",
    "swimsuit",
    "bikini",
    "underwear",
    "bra",
    "lingerie",
]

# Prompts para detectar piel (para excluir de la máscara)
SKIN_PROMPTS = [
    "skin",
    "face",
    "arm",
    "leg",
    "hand",
    "neck",
]


class ClothingSegmenter:
    """Segmentador de ropa usando CLIPSeg cargado localmente."""
    
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_loaded = False
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((352, 352)), # CLIPSeg prefiere múltiplos de 32 o 352
        ])
        
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def is_available(self) -> bool:
        """Verifica si los archivos necesarios están disponibles."""
        model_path = os.path.join("models", "CLIP", "rd64-uni-refined.pth")
        return os.path.exists(model_path)
    
    def load(self) -> Tuple[bool, str]:
        """Carga el modelo CLIPSeg manualmente desde la carpeta models."""
        if self._is_loaded:
            return True, "CLIPSeg ya cargado"

        try:
            logger.info("[ClothingSegmenter] Cargando CLIPSeg LOCAL (Fix Windows)...")
            
            from clip.clipseg import CLIPDensePredT
            
            model_path = os.path.join("models", "CLIP", "rd64-uni-refined.pth")
            if not os.path.exists(model_path):
                return False, f"No se encontró el modelo en {model_path}. Por favor, descárgalo o verifica la ruta."

            # Inicializar arquitectura
            self.model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
            
            # Cargar pesos
            state_dict = torch.load(model_path, map_location='cpu')
            self.model.load_state_dict(state_dict, strict=False)
            
            # Mover a dispositivo
            self.model.to(self.device)
            self.model.eval()
            
            self._is_loaded = True
            logger.info(f"[ClothingSegmenter] ✅ CLIPSeg cargado localmente en {self.device}")
            return True, "CLIPSeg cargado correctamente"

        except Exception as e:
            logger.error(f"[ClothingSegmenter] ❌ Error cargando CLIPSeg: {e}")
            import traceback
            traceback.print_exc()
            return False, f"CLIPSeg error: {str(e)}"
    
    def unload(self):
        """Descarga el modelo."""
        if self.model is not None:
            del self.model
            self.model = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[ClothingSegmenter] Modelo descargado")
    
    def segment_with_prompt(
        self,
        image: Image.Image,
        prompts: List[str],
        threshold: float = 0.4,
        dilation: int = 4
    ) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """
        Segmenta áreas específicas basadas en prompts personalizados.
        """
        if not self._is_loaded:
            success, msg = self.load()
            if not success: return None, None
            
        try:
            # Asegurar RGB para evitar error de tensores (4 canales -> 3 canales)
            if image.mode != "RGB":
                image = image.convert("RGB")
                
            original_size = image.size # (W, H)
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Replicar imagen para el número de prompts
                preds = self.model(img_tensor.repeat(len(prompts), 1, 1, 1), prompts)[0]
            
            # Sigmoid para obtener probabilidades
            probs = torch.sigmoid(preds[:, 0, :, :])
            
            # Combinar detecciones (max)
            combined = probs.max(dim=0)[0].cpu().numpy()
            
            # Redimensionar al original
            combined_resized = cv2.resize(combined, original_size, interpolation=cv2.INTER_LINEAR)
            
            # Aplicar umbral
            mask_array = (combined_resized > threshold).astype(np.uint8) * 255
            
            # Limpiar
            mask_array = self._clean_mask(mask_array, dilation=dilation)
            
            return Image.fromarray(mask_array, mode="L"), mask_array
            
        except Exception as e:
            logger.error(f"[ClothingSegmenter] Error en segment_with_prompt: {e}")
            return None, None

    def segment_clothing(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        combine_mode: str = "max",
        include_skin_exclusion: bool = True,
        dilation: int = 6
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Segmenta las áreas de ropa en una imagen usando carga manual.
        """
        if not self._is_loaded:
            success, msg = self.load()
            if not success:
                logger.info("[ClothingSegmenter] Fallback de detección por color...")
                return self._fallback_clothing_detection(image, threshold)
        
        try:
            # Convertir PIL a array para transformaciones si es necesario
            original_size = image.size # (W, H)
            
            # Preparar prompts
            all_prompts = CLOTHING_PROMPTS.copy()
            logger.info(f"[ClothingSegmenter] Procesando {len(all_prompts)} prompts localmente...")
            
            # Transformar imagen
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Generar segmentaciones (una por prompt)
            with torch.no_grad():
                # El modelo espera (img, prompts)
                # preds shape: [num_prompts, 1, 352, 352]
                preds = self.model(img_tensor.repeat(len(all_prompts), 1, 1, 1), all_prompts)[0]
            
            # Convertir a probabilidades con sigmoid [num_prompts, 352, 352]
            probs = torch.sigmoid(preds[:, 0, :, :])
            
            # Combinar detecciones en el tensor
            if combine_mode == "max":
                combined = probs.max(dim=0)[0]
            elif combine_mode == "mean":
                combined = probs.mean(dim=0)
            else:
                combined = probs.max(dim=0)[0]
            
            # Redimensionar a tamaño original
            combined_np = combined.cpu().numpy()
            combined_resized = cv2.resize(combined_np, original_size, interpolation=cv2.INTER_LINEAR)
            
            # Excluir piel si está activado
            if include_skin_exclusion:
                skin_mask = self._detect_skin(image)
                if skin_mask is not None:
                    # skin_mask es 0-1 float
                    combined_resized = combined_resized * (1 - skin_mask)
            
            # Aplicar threshold
            mask_array = (combined_resized > threshold).astype(np.uint8) * 255
            
            # Aplicar operaciones morfológicas para limpiar la máscara
            mask_array = self._clean_mask(mask_array, dilation=dilation)
            
            # Convertir a PIL Image
            mask_image = Image.fromarray(mask_array, mode="L")
            
            logger.info(f"[ClothingSegmenter] ✅ Máscara local generada")
            
            return mask_image, mask_array
            
        except Exception as e:
            logger.error(f"[ClothingSegmenter] ❌ Error en segmentación local: {e}")
            import traceback
            traceback.print_exc()
            return self._fallback_clothing_detection(image, threshold)

    def _detect_skin(self, image: Image.Image) -> np.ndarray:
        """Detecta piel usando el mismo modelo pero con prompts de piel."""
        try:
            original_size = image.size
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                preds = self.model(img_tensor.repeat(len(SKIN_PROMPTS), 1, 1, 1), SKIN_PROMPTS)[0]
            
            probs = torch.sigmoid(preds[:, 0, :, :])
            combined = probs.max(dim=0)[0].cpu().numpy()
            
            return cv2.resize(combined, original_size, interpolation=cv2.INTER_LINEAR)
        except:
            return None

    def _clean_mask(self, mask: np.ndarray, dilation: int = 6) -> np.ndarray:
        """Limpia y expande la máscara."""
        if dilation > 0:
            kernel = np.ones((dilation, dilation), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)
        
        # Blur suave para suavizar bordes
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        return mask

    def _fallback_clothing_detection(self, image: Image.Image, threshold: float) -> Tuple[Image.Image, np.ndarray]:
        """Detección ultra-básica por color como último recurso."""
        logger.warning("[ClothingSegmenter] Usando fallback por color (extremadamente impreciso)")
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape
        # Asumir que el centro es ropa (heurística básica)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (w//4, h//4), (3*w//4, 3*h//4), 255, -1)
        return Image.fromarray(mask, mode="L"), mask

def get_clothing_segmenter() -> ClothingSegmenter:
    """Singleton para el segmentador."""
    if not hasattr(get_clothing_segmenter, "_instance"):
        get_clothing_segmenter._instance = ClothingSegmenter()
    return get_clothing_segmenter._instance

def is_clipseg_available() -> bool:
    """Verifica disponibilidad."""
    return get_clothing_segmenter().is_available()
