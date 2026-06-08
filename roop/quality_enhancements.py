"""
Mejoras de calidad para Face Swapping en AutoDeep

Este módulo contiene funciones auxiliares para mejorar la calidad
del proceso de intercambio de caras.
"""

import cv2
import numpy as np
from typing import Tuple


def detect_foreground_occlusion(face_img: np.ndarray, target_region: np.ndarray) -> np.ndarray:
    """
    Detecta pelo, flequillo o manos sobre la cara para protegerlos durante el swap. (v5.3)
    Incluye detección de oclusiones con tono de piel (manos) mediante distancia de color LAB.
    """
    try:
        if face_img.shape != target_region.shape:
            target_region = cv2.resize(target_region, (face_img.shape[1], face_img.shape[0]))
            
        # 1. Análisis de texturas y bordes (Sobel)
        src_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        tgt_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)
        
        tgt_sob = cv2.Sobel(tgt_gray, cv2.CV_32F, 1, 1, ksize=3)
        src_sob = cv2.Sobel(src_gray, cv2.CV_32F, 1, 1, ksize=3)
        diff_sob = cv2.absdiff(tgt_sob, src_sob)
        
        # v5.4.1: Umbral conservador (22) — umbrales muy bajos detectan el swap como oclusión
        _, texture_mask = cv2.threshold(diff_sob.astype(np.uint8), 22, 255, cv2.THRESH_BINARY)
        
        # 2. Análisis de color (No-Piel) - Rango más restrictivo
        tgt_hsv = cv2.cvtColor(target_region, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 15, 50])  
        upper_skin = np.array([18, 180, 200]) 
        skin_mask = cv2.inRange(tgt_hsv, lower_skin, upper_skin)
        not_skin_mask = cv2.bitwise_not(skin_mask)
        
        # 3. Oclusión base: textura extra DONDE NO ES PIEL (pelo, objetos no-piel)
        occlusion = cv2.bitwise_and(texture_mask, not_skin_mask)
        
        # 4. v5.4: Detección de oclusiones CON TONO DE PIEL más agresiva
        # Comparar color LAB local vs promedio de piel de la cara
        tgt_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_pixels = tgt_lab[skin_mask > 0]
        if len(skin_pixels) > 30:
            avg_skin_lab = np.mean(skin_pixels, axis=0)
            lab_diff = np.sqrt(np.sum((tgt_lab - avg_skin_lab.reshape(1, 1, 3)) ** 2, axis=2))
            _, skin_occ_mask = cv2.threshold(lab_diff.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
            skin_occ_mask = cv2.bitwise_and(skin_occ_mask, cv2.bitwise_and(texture_mask, skin_mask))
            occlusion = cv2.bitwise_or(occlusion, skin_occ_mask)
        
        # 5. v5.4: Análisis de Detalle Fino (Laplaciano) — umbral mínimo para capturar oclusiones sutiles
        tgt_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)
        swp_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        tgt_lap = cv2.Laplacian(tgt_gray, cv2.CV_32F)
        swp_lap = cv2.Laplacian(swp_gray, cv2.CV_32F)
        
        diff_lap = np.clip(np.abs(tgt_lap) - np.abs(swp_lap) * 1.1, 0, 255)
        _, detail_mask = cv2.threshold(diff_lap.astype(np.uint8), 22, 255, cv2.THRESH_BINARY)
        
        # v5.4: Center-Sparing reducido (5%) para preservar oclusiones centrales
        h_d, w_d = detail_mask.shape[:2]
        center_mask = np.zeros((h_d, w_d), dtype=np.uint8)
        cv2.circle(center_mask, (w_d//2, int(h_d*0.6)), int(min(w_d, h_d)*0.05), 255, -1)
        detail_mask = cv2.bitwise_and(detail_mask, cv2.bitwise_not(center_mask))
        
        detail_mask = cv2.dilate(detail_mask, np.ones((5, 5), np.uint8), iterations=2)
        
        occlusion = cv2.bitwise_or(occlusion, detail_mask)
        
        # 6. Post-procesamiento: limpiar ruido pero mantener detecciones reales
        kernel = np.ones((5,5), np.uint8)
        occlusion = cv2.erode(occlusion, kernel, iterations=1)
        occlusion = cv2.dilate(occlusion, kernel, iterations=1)
        
        occlusion = cv2.GaussianBlur(occlusion, (7, 7), 0)
        
        return occlusion.astype(np.float32) / 255.0
    except:
        return np.zeros(face_img.shape[:2], dtype=np.float32)


def match_color_histogram(source: np.ndarray, target: np.ndarray, blend_factor: float = 0.25) -> np.ndarray:
    """
    Ajusta el histograma de color de forma ultra-estable para evitar parpadeos. (v2.7)
    """
    try:
        if source is None or target is None or source.size == 0 or target.size == 0:
            return source
        
        if source.shape != target.shape:
            target = cv2.resize(target, (source.shape[1], source.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Convertir a LAB para preservar luminancia y tono natural
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Estadísticas globales de la cara
        for i in range(3):
            s_mean, s_std = np.mean(source_lab[:,:,i]), np.std(source_lab[:,:,i])
            t_mean, t_std = np.mean(target_lab[:,:,i]), np.std(target_lab[:,:,i])
            
            if s_std > 0.1:
                # Ratio de desviación más restrictivo para evitar colores lavados
                ratio = np.clip(t_std / (s_std + 1e-6), 0.90, 1.10)
                adjusted = (source_lab[:,:,i] - s_mean) * ratio + t_mean
                
                # Mezcla conservadora por canal
                source_lab[:,:,i] = source_lab[:,:,i] * (1 - blend_factor) + adjusted * blend_factor
        
        matched = cv2.cvtColor(np.clip(source_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return matched
    except Exception as e:
        return source


def create_soft_mask(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int], feather: int = 25, occlusion_mask: np.ndarray = None) -> np.ndarray:
    """
    Crea una máscara de integración optimizada con degradado dinámico. (v5.2.3)
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    
    # Centro y ejes para elipse
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (int(bw * 0.46), int(bh * 0.49))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    
    # v5.2.3: Atenuación superior (Top-Down Fade) para proteger el flequillo (fringe)
    gradient = np.ones((bh, bw), dtype=np.float32)
    fade_height = int(bh * 0.25)
    for i in range(fade_height):
        # 0.0 en el borde superior, 1.0 al llegar a fade_height
        gradient[i, :] = (i / fade_height) ** 0.5 
    
    full_gradient = np.ones_like(mask)
    full_gradient[y1:y2, x1:x2] = gradient
    mask = mask * full_gradient

    # Protección de oclusiones
    if occlusion_mask is not None:
        full_occ_mask = np.zeros_like(mask)
        occ_resized = cv2.resize(occlusion_mask, (bw, bh), interpolation=cv2.INTER_LINEAR)
        full_occ_mask[y1:y2, x1:x2] = occ_resized
        mask = np.clip(mask - (full_occ_mask * 0.95), 0, 1)
    
    # Blur dinámico
    blur_size = int(max(feather, min(bw, bh) // 5)) | 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    return np.clip(mask, 0, 1.0)



def blend_with_poisson(source: np.ndarray, target: np.ndarray, mask: np.ndarray, center: Tuple[int, int]) -> np.ndarray:
    """
    Realiza blending usando Poisson blending mejorado para evitar halos negros.
    """
    try:
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        original_mask = mask.copy()
        
        # IMPORTANTE: Encoger ligeramente la máscara para Poisson para evitar que toque bordes
        kernel = np.ones((5, 5), np.uint8)
        poisson_mask = cv2.erode(mask, kernel, iterations=1)
        if cv2.countNonZero(poisson_mask) < 10:
            poisson_mask = original_mask.copy()
        
        # Recortar la zona activa. seamlessClone falla si se le pasa un frame completo
        # centrado en una cara, porque el source no cabe dentro del destino.
        x, y, bw, bh = cv2.boundingRect(poisson_mask)
        if bw <= 1 or bh <= 1:
            raise ValueError("Poisson mask is empty")

        pad = 8
        src_h, src_w = source.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(src_w, x + bw + pad)
        y2 = min(src_h, y + bh + pad)

        source_crop = source[y1:y2, x1:x2]
        mask_crop = poisson_mask[y1:y2, x1:x2]

        if source_crop.size == 0 or mask_crop.size == 0:
            raise ValueError("Poisson crop is empty")

        # Bordes negros obligatorios para evitar crash de seamlessClone
        mask_crop[0, :] = 0
        mask_crop[-1, :] = 0
        mask_crop[:, 0] = 0
        mask_crop[:, -1] = 0

        crop_center = (x1 + source_crop.shape[1] // 2, y1 + source_crop.shape[0] // 2)
        
        # NORMAL_CLONE es a veces más estable contra reflejos negros que MIXED_CLONE
        result = cv2.seamlessClone(
            source_crop.astype(np.uint8),
            target.astype(np.uint8),
            mask_crop,
            crop_center,
            cv2.NORMAL_CLONE 
        )
        return result
    except Exception as e:
        print(f"[WARN] Poisson failed ({e}), using alpha fallback")
        mask_3ch = cv2.cvtColor(original_mask if 'original_mask' in locals() else mask, cv2.COLOR_GRAY2BGR) / 255.0
        return (source * mask_3ch + target * (1.0 - mask_3ch)).astype(np.uint8)


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
    """
    if options is None:
        options = {}
    
    # 1. Ajustar colores para mejor integración
    color_match_strength = options.get('color_match_strength', 0.3)
    if color_match_strength > 0:
        swapped_face = match_color_histogram(swapped_face, target_region)
    
    # 2. Ajustar brillo
    brightness_match_strength = options.get('brightness_match_strength', 0.2)
    if brightness_match_strength > 0:
        swapped_face = adjust_face_brightness(swapped_face, target_region, brightness_match_strength)
    
    # 3. NUEVO: Detectar oclusiones (pelo, flequillo, objetos)
    # Esto evita que el swap cubra el pelo real o cree reflejos negros
    occlusion_mask = detect_foreground_occlusion(swapped_face, target_region)
    
    # 4. Crear máscara suave para blending (ahora con protección de oclusiones)
    feather_amount = options.get('feather_amount', 30)
    soft_mask = create_soft_mask(bbox, frame.shape, feather=feather_amount, occlusion_mask=occlusion_mask)
    
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
