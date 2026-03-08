"""
Mejoras de calidad para Face Swapping en AutoDeep

Este módulo contiene funciones auxiliares para mejorar la calidad
del proceso de intercambio de caras.
"""

import cv2
import numpy as np
from typing import Tuple


def match_color_histogram(source: np.ndarray, target: np.ndarray, blend_factor: float = 0.25) -> np.ndarray:
    """
    Ajusta el histograma de color de la cara origen para que coincida con el destino.
    Esto mejora la integración visual de la cara intercambiada.
    
    MEJORADO: Matching más suave y natural con control de intensidad.
    NOTA: blend_factor reducido a 0.25 por defecto para preservar mejor la identidad de origen.
    
    Args:
        source: Imagen de la cara origen (BGR)
        target: Imagen de la cara destino (BGR)
        blend_factor: Factor de mezcla (0.0 = sin cambio, 1.0 = matching completo)
    
    Returns:
        Cara origen con colores ajustados
    """
    try:
        # Validar inputs
        if source is None or target is None or source.size == 0 or target.size == 0:
            return source
        
        # Asegurar mismo tamaño
        if source.shape != target.shape:
            target = cv2.resize(target, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convertir a LAB para mejor ajuste de color (LAB separa luminancia de color)
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        result_lab = source_lab.copy()
        
        # Canal L (luminancia) - ajuste más fuerte
        # Canales A y B (color) - ajuste más suave para no perder naturalidad
        channel_factors = [0.8, 0.6, 0.6]  # L, A, B
        
        for i in range(3):
            source_mean = np.mean(source_lab[:, :, i])
            source_std = np.std(source_lab[:, :, i])
            target_mean = np.mean(target_lab[:, :, i])
            target_std = np.std(target_lab[:, :, i])
            
            if source_std > 0.1:  # Evitar división por valores muy pequeños
                # Calcular ajuste
                std_ratio = min(target_std / source_std, 2.0)  # Limitar ratio para evitar excesos
                mean_diff = target_mean - source_mean
                
                # Aplicar con factor de mezcla por canal
                adjusted = source_lab[:, :, i] * std_ratio + mean_diff
                
                # Mezclar con original según blend_factor
                result_lab[:, :, i] = source_lab[:, :, i] * (1 - blend_factor * channel_factors[i]) + \
                                      adjusted * (blend_factor * channel_factors[i])
        
        # Clip seguro
        result_lab = np.clip(result_lab, 0, 255)
        
        # Convertir de vuelta a BGR
        result = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        # Blending final suave para evitar cambios bruscos
        result = cv2.addWeighted(source, 1 - blend_factor * 0.3, result, blend_factor * 0.3, 0)
        
        return result
        
    except Exception as e:
        print(f"[COLOR_MATCH] Error: {e}")
        return source


def create_soft_mask(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int], feather: int = 20) -> np.ndarray:
    """
    Crea una máscara suave ELÍPTICA con cobertura central y bordes difuminados.
    
    Args:
        bbox: Bounding box de la cara (x1, y1, x2, y2)
        frame_shape: Forma del frame (height, width)
        feather: Píxeles de difuminado en los bordes
    
    Returns:
        Máscara suave (valores 0-1)
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame_shape[:2]
    
    # Crear máscara base
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Calcular centro y radios de la elipse
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    width = x2 - x1
    height = y2 - y1
    
    # SOLUCIÓN: Reducir el radio para evitar que toque los bordes del bbox
    # 0.40 = 80% del ancho/alto, dejando un margen del 10% a cada lado
    radius_x = int(width * 0.40) 
    radius_y = int(height * 0.40)
    
    # Dibujar elipse COMPLETA en la máscara
    cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 1.0, -1)
    
    # Aplicar difuminado SOLO en los bordes para transición suave
    if feather > 0:
        # Kernel moderado para suavizar solo los bordes
        kernel_size = max(feather * 2 + 1, 21)  # Kernel de 21px
        if kernel_size % 2 == 0:
            kernel_size += 1
        sigma = max(feather / 2, 8)  # Sigma moderado
        mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigma)
    
    return mask


def enhance_face_details(face_img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """
    Mejora los detalles de la cara usando unsharp masking.
    
    Args:
        face_img: Imagen de la cara
        strength: Intensidad del realce (0.0 - 1.0)
    
    Returns:
        Cara con detalles mejorados
    """
    # Aplicar desenfoque gaussiano
    blurred = cv2.GaussianBlur(face_img, (0, 0), 3)
    
    # Unsharp mask
    sharpened = cv2.addWeighted(face_img, 1.0 + strength, blurred, -strength, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def blend_with_poisson(source: np.ndarray, target: np.ndarray, mask: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
    """
    Realiza blending usando Poisson blending para resultados más naturales.
    
    Args:
        source: Imagen fuente
        target: Imagen destino
        mask: Máscara binaria
        center: Centro para el blending
    
    Returns:
        Imagen con blending aplicado
    """
    try:
        # Asegurar que la máscara sea uint8
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # CRÍTICO: Poisson blending requiere que la máscara no toque los bordes de la imagen fuente
        # Forzar bordes negros en la máscara
        mask[0, :] = 0
        mask[-1, :] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0
        
        # Poisson blending (seamless clone)
        result = cv2.seamlessClone(
            source.astype(np.uint8),
            target.astype(np.uint8),
            mask,
            center,
            cv2.MIXED_CLONE  # MIXED_CLONE preserva mejor los detalles
        )
        return result
    except Exception as e:
        print(f"[WARN] Poisson blending failed: {e}, using weighted blend")
        # Fallback a weighted blend
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = (source * mask_3ch + target * (1 - mask_3ch)).astype(np.uint8)
        return result


def adjust_face_brightness(face: np.ndarray, target_region: np.ndarray, strength: float = 0.2) -> np.ndarray:
    """
    Ajusta el brillo de la cara para que coincida con la región objetivo.
    
    NOTA: strength reducido a 0.2 por defecto para preservar mejor la identidad de origen.
    
    Args:
        face: Imagen de la cara a ajustar
        target_region: Región objetivo para matching
        strength: Fuerza del ajuste (0.0 - 1.0)
    
    Returns:
        Cara con brillo ajustado
    """
    try:
        if face is None or target_region is None or face.size == 0 or target_region.size == 0:
            return face
        
        # Asegurar mismo tamaño
        if face.shape != target_region.shape:
            target_region = cv2.resize(target_region, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Calcular brillo promedio
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)
        
        face_brightness = np.mean(face_gray)
        target_brightness = np.mean(target_gray)
        
        # Calcular factor de ajuste con límites para evitar excesos
        if face_brightness > 10:  # Evitar división por valores muy pequeños
            brightness_ratio = target_brightness / face_brightness
            # Limitar ratio para evitar cambios extremos
            brightness_ratio = np.clip(brightness_ratio, 0.7, 1.3)
            
            # Aplicar con strength
            adjustment = 1.0 + (brightness_ratio - 1.0) * strength
            
            # Ajustar en espacio LAB (solo canal L)
            face_lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
            face_lab[:, :, 0] = np.clip(face_lab[:, :, 0] * adjustment, 0, 255)
            
            result = cv2.cvtColor(face_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
            return result
        
        return face
        
    except Exception as e:
        print(f"[BRIGHTNESS] Error: {e}")
        return face


def apply_quality_enhancements(
    swapped_face: np.ndarray,
    target_region: np.ndarray,
    bbox: Tuple[int, int, int, int],
    frame: np.ndarray,
    options: dict = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica todas las mejoras de calidad a una cara intercambiada.
    
    Args:
        swapped_face: Cara ya intercambiada
        target_region: Región original del frame donde va la cara
        bbox: Bounding box de la cara
        frame: Frame completo
        options: Opciones de configuración
    
    Returns:
        (cara_mejorada, máscara_suave)
    """
    if options is None:
        options = {}
    
    # 1. Ajustar colores para mejor integración
    color_match_strength = options.get('color_match_strength', 0.3)  # Reducido de 0.8 a 0.3
    if color_match_strength > 0:
        swapped_face = match_color_histogram(swapped_face, target_region)
    
    # 2. Ajustar brillo
    brightness_match_strength = options.get('brightness_match_strength', 0.2)  # Reducido de 0.7 a 0.2
    if brightness_match_strength > 0:
        swapped_face = adjust_face_brightness(swapped_face, target_region, brightness_match_strength)
    
    # 3. Mejorar detalles (opcional, puede introducir artefactos)
    enhance_details = options.get('enhance_details', False)
    if enhance_details:
        detail_strength = options.get('detail_strength', 0.2)
        swapped_face = enhance_face_details(swapped_face, detail_strength)
    
    # 4. Crear máscara suave para blending
    feather_amount = options.get('feather_amount', 15)
    soft_mask = create_soft_mask(bbox, frame.shape, feather=feather_amount)
    
    return swapped_face, soft_mask


def advanced_blend(
    swapped_face: np.ndarray,
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    soft_mask: np.ndarray = None,
    blend_mode: str = "poisson"
) -> np.ndarray:
    """
    Realiza blending avanzado de la cara en el frame.
    
    Args:
        swapped_face: Cara intercambiada
        frame: Frame destino
        bbox: Bounding box
        soft_mask: Máscara suave (opcional)
        blend_mode: Modo de blending ("poisson", "weighted", "direct")
    
    Returns:
        Frame con la cara integrada
    """
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1
    
    # Asegurar que la cara tenga el tamaño correcto
    if swapped_face.shape[:2] != (h, w):
        swapped_face = cv2.resize(swapped_face, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    result = frame.copy()
    
    if blend_mode == "poisson":
        # Crear máscara para Poisson
        if soft_mask is not None:
            mask = (soft_mask[y1:y2, x1:x2] * 255).astype(np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8) * 255
        
        center = (x1 + w // 2, y1 + h // 2)
        result = blend_with_poisson(swapped_face, result, mask, center)
        
    elif blend_mode == "weighted":
        # Blending ponderado con máscara suave
        if soft_mask is not None:
            mask_3ch = soft_mask[y1:y2, x1:x2]
            mask_3ch = np.stack([mask_3ch] * 3, axis=-1)
        else:
            mask_3ch = np.ones((h, w, 3), dtype=np.float32)
        
        target_region = result[y1:y2, x1:x2].astype(np.float32)
        blended = (swapped_face.astype(np.float32) * mask_3ch + 
                  target_region * (1 - mask_3ch))
        result[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        
    else:  # direct
        result[y1:y2, x1:x2] = swapped_face
    
    return result
