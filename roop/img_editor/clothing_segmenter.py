# -*- coding: utf-8 -*-
"""
ClothingSegmenter - Detección automática de ropa con CLIPSeg

Usa CLIPSeg para detectar áreas de ropa en imágenes y generar máscaras
para inpainting selectivo.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from typing import Optional, Tuple, List
import logging

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
    """Segmentador de ropa usando CLIPSeg."""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_loaded = False
        
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    def is_available(self) -> bool:
        """Verifica si CLIPSeg está disponible."""
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            return True
        except ImportError:
            return False
    
    def load(self) -> Tuple[bool, str]:
        """Carga el modelo CLIPSeg."""
        if self._is_loaded:
            return True, "CLIPSeg ya cargado"
        
        try:
            logger.info("[ClothingSegmenter] Cargando CLIPSeg...")
            
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
            
            model_name = "CIDAS/clipseg-rd64-refined"
            
            self.processor = CLIPSegProcessor.from_pretrained(model_name)
            self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self._is_loaded = True
            logger.info(f"[ClothingSegmenter] ✅ CLIPSeg cargado en {self.device}")
            return True, "CLIPSeg cargado correctamente"
            
        except Exception as e:
            logger.error(f"[ClothingSegmenter] ❌ Error cargando CLIPSeg: {e}")
            import traceback
            traceback.print_exc()
            return False, str(e)
    
    def unload(self):
        """Descarga el modelo."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self._is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("[ClothingSegmenter] Modelo descargado")
    
    def segment_clothing(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        combine_mode: str = "max",
        include_skin_exclusion: bool = True,
        dilation: int = 6
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Segmenta las áreas de ropa en una imagen.
        
        Args:
            image: Imagen de entrada (PIL Image)
            threshold: Umbral para binarizar la máscara (0.0-1.0)
            combine_mode: Cómo combinar múltiples detecciones ("max", "mean", "any")
            include_skin_exclusion: Si True, excluye áreas de piel detectadas
            dilation: Píxeles a expandir la máscara (0-30)
            
        Returns:
            Tuple de (máscara PIL Image, máscara numpy array)
            Blanco = área de ropa a modificar
            Negro = área a preservar
        """
        if not self._is_loaded:
            success, msg = self.load()
            if not success:
                # Usar fallback de detección por color si CLIPSeg falla
                logger.info("[ClothingSegmenter] Usando fallback de detección por color...")
                return self._fallback_clothing_detection(image, threshold)
        
        try:
            original_size = image.size
            
            # Preparar prompts
            all_prompts = CLOTHING_PROMPTS.copy()
            
            logger.info(f"[ClothingSegmenter] Detectando ropa con {len(all_prompts)} prompts...")
            
            # Procesar imagen con todos los prompts
            inputs = self.processor(
                text=all_prompts,
                images=[image] * len(all_prompts),
                return_tensors="pt"
            ).to(self.device)
            
            # Generar segmentaciones
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Obtener logits y normalizar
            logits = outputs.logits  # [num_prompts, H, W]
            
            # Redimensionar a tamaño original
            logits_resized = torch.nn.functional.interpolate(
                logits.unsqueeze(1),
                size=original_size[::-1],
                mode="bilinear",
                align_corners=False
            ).squeeze(1)
            
            # Convertir a probabilidades con sigmoid
            probs = torch.sigmoid(logits_resized)
            
            # Combinar detecciones
            if combine_mode == "max":
                combined = probs.max(dim=0)[0]
            elif combine_mode == "mean":
                combined = probs.mean(dim=0)
            elif combine_mode == "any":
                combined = (probs > threshold).any(dim=0).float()
            else:
                combined = probs.max(dim=0)[0]
            
            # Excluir piel si está activado
            if include_skin_exclusion:
                skin_mask = self._detect_skin(image)
                if skin_mask is not None:
                    # Convertir skin_mask a tensor CUDA para poder multiplicar
                    skin_tensor = torch.from_numpy(skin_mask).to(self.device)
                    # Restar áreas de piel de la máscara de ropa
                    combined = combined * (1 - skin_tensor)
            
            # Mover a CPU antes de convertir a numpy
            combined_cpu = combined.cpu().numpy()
            
            # Aplicar threshold
            mask_array = (combined_cpu > threshold).astype(np.uint8) * 255
            
            # Aplicar operaciones morfológicas para limpiar la máscara
            mask_array = self._clean_mask(mask_array, dilation=dilation)
            
            # Convertir a PIL Image
            mask_image = Image.fromarray(mask_array, mode="L")
            
            logger.info(f"[ClothingSegmenter] ✅ Máscara generada: {mask_array.sum() / 255} píxeles")
            
            return mask_image, mask_array
            
        except Exception as e:
            logger.error(f"[ClothingSegmenter] ❌ Error en segmentación: {e}")
            import traceback
            traceback.print_exc()
            return Image.new("L", image.size, 0), np.zeros(image.size[::-1], dtype=np.uint8)
    
    def _detect_skin(self, image: Image.Image) -> Optional[np.ndarray]:
        """Detecta áreas de piel para excluirlas de la máscara de ropa."""
        try:
            # Usar detección de color de piel simple en HSV
            img_array = np.array(image.convert("RGB"))
            
            # Convertir a HSV
            import cv2
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Rango de tonos de piel (amplio para diferentes tonos)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Segundo rango para tonos más oscuros
            lower_skin2 = np.array([0, 20, 40], dtype=np.uint8)
            upper_skin2 = np.array([15, 255, 150], dtype=np.uint8)
            
            mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
            
            # Combinar máscaras
            skin_mask = (mask1 + mask2) > 0
            
            # Dilatar un poco para cubrir bordes
            kernel = np.ones((5, 5), np.uint8)
            skin_mask = cv2.dilate(skin_mask.astype(np.uint8), kernel, iterations=2)
            
            return skin_mask.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"[ClothingSegmenter] No se pudo detectar piel: {e}")
            return None
    
    def _fallback_clothing_detection(self, image: Image.Image, threshold: float = 0.5) -> Tuple[Image.Image, np.ndarray]:
        """
        Fallback para detectar ropa cuando CLIPSeg no está disponible.
        
        Usa detección de color: asume que las áreas que NO son piel son ropa.
        También detecta colores comunes de ropa.
        """
        try:
            import cv2
            img_array = np.array(image.convert("RGB"))
            h, w = img_array.shape[:2]
            
            # 1. Detectar piel
            skin_mask = self._detect_skin(image)
            if skin_mask is None:
                skin_mask = np.zeros((h, w), dtype=np.float32)
            
            # 2. Detectar colores comunes de ropa (negro, blanco, azul, rojo, etc.)
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Rango para colores de ropa comunes
            clothing_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Negro (baja saturación y valor)
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 50])
            clothing_mask = np.maximum(clothing_mask, cv2.inRange(hsv, lower_black, upper_black))
            
            # Blanco (baja saturación, alto valor)
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            clothing_mask = np.maximum(clothing_mask, cv2.inRange(hsv, lower_white, upper_white))
            
            # Azul (jeans, etc.)
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            clothing_mask = np.maximum(clothing_mask, cv2.inRange(hsv, lower_blue, upper_blue))
            
            # Rojo/Rosa
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            clothing_mask = np.maximum(clothing_mask, cv2.inRange(hsv, lower_red1, upper_red1))
            clothing_mask = np.maximum(clothing_mask, cv2.inRange(hsv, lower_red2, upper_red2))
            
            # Verde
            lower_green = np.array([35, 50, 50])
            upper_green = np.array([85, 255, 255])
            clothing_mask = np.maximum(clothing_mask, cv2.inRange(hsv, lower_green, upper_green))
            
            # Amarillo/Marrón
            lower_yellow = np.array([15, 50, 50])
            upper_yellow = np.array([35, 255, 255])
            clothing_mask = np.maximum(clothing_mask, cv2.inRange(hsv, lower_yellow, upper_yellow))
            
            # 3. Combinar: ropa = colores de ropa detectados O (no piel en zona central)
            # Crear máscara de zona central (excluir bordes)
            center_mask = np.zeros((h, w), dtype=np.uint8)
            margin = int(min(h, w) * 0.1)
            center_mask[margin:h-margin, margin:w-margin] = 255
            
            # Área que no es piel y está en zona central
            non_skin_center = ((1 - skin_mask) > 0.5).astype(np.uint8) * 255
            non_skin_center = cv2.bitwise_and(non_skin_center, center_mask)
            
            # Combinar detección de colores con no-piel
            combined_mask = cv2.bitwise_or(clothing_mask, non_skin_center)
            
            # 4. Limpiar máscara
            combined_mask = self._clean_mask(combined_mask, dilation=6)  # Usar default en fallback
            
            logger.info(f"[ClothingSegmenter] Fallback: máscara generada ({combined_mask.sum() / 255} píxeles)")
            
            return Image.fromarray(combined_mask, mode="L"), combined_mask
            
        except Exception as e:
            logger.error(f"[ClothingSegmenter] Error en fallback: {e}")
            import traceback
            traceback.print_exc()
            # Retornar máscara que cubra zona central como último recurso
            h, w = image.size[::-1]
            mask = np.zeros((h, w), dtype=np.uint8)
            margin = int(min(h, w) * 0.15)
            mask[margin:h-margin, margin:w-margin] = 255
            return Image.fromarray(mask, mode="L"), mask
    
    def _clean_mask(self, mask: np.ndarray, dilation: int = 6) -> np.ndarray:
        """Limpia la máscara con operaciones morfológicas.
        
        Args:
            mask: Máscara binaria
            dilation: Píxeles a expandir la máscara (0-30)
        """
        try:
            import cv2
            
            # Kernel para operaciones morfológicas
            kernel = np.ones((5, 5), np.uint8)
            
            # Cerrar huecos pequeños
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Abrir para eliminar ruido
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Dilatar para cubrir bordes - usar el parámetro dilation
            if dilation > 0:
                # Calcular iteraciones basadas en dilation
                iterations = max(1, dilation // 5)
                dilate_kernel = np.ones((dilation, dilation), np.uint8) if dilation > 5 else kernel
                mask = cv2.dilate(mask, dilate_kernel, iterations=iterations)
            
            # Desenfocar ligeramente los bordes para transiciones suaves
            blur_size = max(5, min(15, dilation + 3))
            if blur_size % 2 == 0:
                blur_size += 1  # Debe ser impar
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
            
            # Re-binarizar
            mask = (mask > 127).astype(np.uint8) * 255
            
            return mask
            
        except Exception as e:
            logger.warning(f"[ClothingSegmenter] Error limpiando máscara: {e}")
            return mask
    
    def segment_with_prompt(
        self,
        image: Image.Image,
        custom_prompts: List[str],
        threshold: float = 0.5
    ) -> Tuple[Image.Image, np.ndarray]:
        """
        Segmenta usando prompts personalizados.
        
        Útil para detectar objetos específicos.
        """
        if not self._is_loaded:
            success, msg = self.load()
            if not success:
                return Image.new("L", image.size, 0), np.zeros(image.size[::-1], dtype=np.uint8)
        
        try:
            original_size = image.size
            
            inputs = self.processor(
                text=custom_prompts,
                images=[image] * len(custom_prompts),
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            logits_resized = torch.nn.functional.interpolate(
                logits.unsqueeze(1),
                size=original_size[::-1],
                mode="bilinear",
                align_corners=False
            ).squeeze(1)
            
            probs = torch.sigmoid(logits_resized)
            combined = probs.max(dim=0)[0]
            
            mask_array = (combined.cpu().numpy() > threshold).astype(np.uint8) * 255
            mask_array = self._clean_mask(mask_array, dilation=6)  # Usar default en segment_with_prompt
            
            return Image.fromarray(mask_array, mode="L"), mask_array
            
        except Exception as e:
            logger.error(f"[ClothingSegmenter] Error: {e}")
            return Image.new("L", image.size, 0), np.zeros(image.size[::-1], dtype=np.uint8)
    
    def visualize_mask(
        self,
        image: Image.Image,
        mask: Image.Image,
        color: Tuple[int, int, int] = (255, 0, 0),
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Crea una visualización de la máscara superpuesta sobre la imagen.
        
        Args:
            image: Imagen original
            mask: Máscara a visualizar
            color: Color de la superposición (RGB)
            alpha: Transparencia de la superposición
            
        Returns:
            Imagen con máscara superpuesta
        """
        img_array = np.array(image.convert("RGB"))
        mask_array = np.array(mask)
        
        # Crear overlay
        overlay = np.zeros_like(img_array)
        overlay[mask_array > 127] = color
        
        # Mezclar
        result = img_array.copy()
        mask_bool = mask_array > 127
        for i in range(3):
            result[:, :, i] = np.where(
                mask_bool,
                img_array[:, :, i] * (1 - alpha) + overlay[:, :, i] * alpha,
                img_array[:, :, i]
            )
        
        return Image.fromarray(result)


# Instancia global
_segmenter = None


def get_clothing_segmenter() -> ClothingSegmenter:
    """Obtiene la instancia global del segmentador."""
    global _segmenter
    if _segmenter is None:
        _segmenter = ClothingSegmenter()
    return _segmenter


def is_clipseg_available() -> bool:
    """Verifica si CLIPSeg está disponible.
    
    Siempre retorna True porque tenemos fallback a detección por color.
    """
    return True
