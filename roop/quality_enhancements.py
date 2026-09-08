"""
Mejoras de calidad para Face Swapping en AutoDeep

Este módulo contiene funciones auxiliares para mejorar la calidad
del proceso de intercambio de caras.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def match_color_histogram(
    source: np.ndarray,
    reference: np.ndarray,
    blend_factor: float = 1.0,
) -> np.ndarray:
    """
    Ajusta color/tonalidad de `source` hacia `reference` (espacio LAB, media+std).

    Usado por ProcessMgr (swap + temporal) y apply_quality_enhancements.
    blend_factor=0 → source intacto; 1 → match completo.
    """
    try:
        if source is None or reference is None:
            return source
        if source.size == 0 or reference.size == 0:
            return source

        blend = float(np.clip(blend_factor, 0.0, 1.0))
        if blend <= 1e-6:
            return source

        src = source
        ref = reference
        if src.dtype != np.uint8:
            src_u8 = np.clip(src, 0, 255).astype(np.uint8)
        else:
            src_u8 = src
        if ref.dtype != np.uint8:
            ref_u8 = np.clip(ref, 0, 255).astype(np.uint8)
        else:
            ref_u8 = ref

        # Mismo tamaño de referencia para estadísticas (no reescalar source)
        if ref_u8.shape[:2] != src_u8.shape[:2]:
            ref_u8 = cv2.resize(
                ref_u8, (src_u8.shape[1], src_u8.shape[0]), interpolation=cv2.INTER_LINEAR
            )

        if src_u8.ndim == 2:
            src_u8 = cv2.cvtColor(src_u8, cv2.COLOR_GRAY2BGR)
        if ref_u8.ndim == 2:
            ref_u8 = cv2.cvtColor(ref_u8, cv2.COLOR_GRAY2BGR)
        if src_u8.shape[2] == 4:
            src_u8 = src_u8[:, :, :3]
        if ref_u8.shape[2] == 4:
            ref_u8 = ref_u8[:, :, :3]

        src_lab = cv2.cvtColor(src_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        ref_lab = cv2.cvtColor(ref_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

        matched = src_lab.copy()
        for c in range(3):
            s = src_lab[:, :, c]
            r = ref_lab[:, :, c]
            s_mean, s_std = float(s.mean()), float(s.std())
            r_mean, r_std = float(r.mean()), float(r.std())
            s_std = max(s_std, 1e-3)
            r_std = max(r_std, 1e-3)
            channel = (s - s_mean) * (r_std / s_std) + r_mean
            matched[:, :, c] = (1.0 - blend) * s + blend * channel

        matched = np.clip(matched, 0, 255).astype(np.uint8)
        out = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
        return out
    except Exception as e:
        print(f"[COLOR_MATCH] Error: {e}")
        return source


def detect_foreground_occlusion(face_img: np.ndarray, target_region: np.ndarray) -> np.ndarray:
    """
    Detecta pelo, flequillo, manos u objetos sobre la cara para protegerlos durante el swap.

    v5.1045: TODAS las señales son SOLO del target (swap-independientes). Antes comparaban
    el swapped con el original: |swap − original| ES el efecto del swap, así que cualquier
    swap fuerte marcaba 66-86% del crop como "oclusión" y restaba 0.88 de la máscara
    (probe_ab: A/B/D finales 1.3-1.6 vs raw 10.7). Ahora los bordes/textura se miden sobre
    target_region — el original YA contiene pelo/manos/objetos si existen.
    """
    try:
        if face_img is None or target_region is None:
            return np.zeros((1, 1), dtype=np.float32)
        if face_img.shape != target_region.shape:
            target_region = cv2.resize(target_region, (face_img.shape[1], face_img.shape[0]))

        h, w = face_img.shape[:2]

        # 1. Análisis de texturas y bordes (Sobel) — SOLO del target (v5.1045)
        # El target ya contiene pelo/manos/objetos: sus bordes se detectan directo
        # sobre target_region, sin comparar con el swapped (que disparaba falsos
        # positivos: cualquier swap fuerte altera los bordes → "oclusión" falsa).
        tgt_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)

        tgt_sob = cv2.Sobel(tgt_gray, cv2.CV_32F, 1, 1, ksize=3)
        tgt_sob_mag = np.abs(tgt_sob)

        sob_mean = np.mean(tgt_sob_mag)
        sob_thresh = max(25, min(40, sob_mean * 1.5))
        _, texture_mask = cv2.threshold(tgt_sob_mag.astype(np.uint8), sob_thresh, 255, cv2.THRESH_BINARY)

        # 2. Análisis de color (piel)
        tgt_hsv = cv2.cvtColor(target_region, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 12, 40])
        upper_skin = np.array([20, 170, 220])
        skin_mask = cv2.inRange(tgt_hsv, lower_skin, upper_skin)

        # Zona facial expandida (mejillas, mandíbula, frente, sienes)
        center_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(center_mask, (w // 2, h // 2), (int(w * 0.82), int(h * 0.85)), 0, 0, 360, 255, -1)

        not_skin_mask = cv2.bitwise_and(cv2.bitwise_not(skin_mask), center_mask)

        # 3. Oclusión base: textura extra donde no es piel
        occlusion = cv2.bitwise_and(texture_mask, not_skin_mask)

        # v5.1045: ELIMINADA la detección por diferencia de color v5.1040 (swap vs original).
        # Era autodestructiva: |swapped − original| ES el efecto del swap. Cualquier swap fuerte
        # (>35 de diff) marcaba el 68-86% de la cara como "oclusión" y restaba 0.88 de la
        # máscara (ProcessMgr), dejando el swap casi invisible (FINAL 1.3-1.6 vs raw 10.7).
        # Verificado en probe_ab: A/B/D débiles (occ 68-86%) vs C fuerte (occ 52%, fuerza 0.70).
        # Las señales Sobel/LAB/Laplaciano restantes comparan solo el target (swap-independiente).

        # 4. Oclusiones con tono de piel (manos) + objetos
        tgt_lab = cv2.cvtColor(target_region, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_pixels = tgt_lab[skin_mask > 0]
        skin_occ_mask = np.zeros((h, w), dtype=np.uint8)
        object_mask = np.zeros((h, w), dtype=np.uint8)
        if len(skin_pixels) > 30:
            avg_skin_lab = np.mean(skin_pixels, axis=0)
            lab_diff = np.sqrt(np.sum((tgt_lab - avg_skin_lab.reshape(1, 1, 3)) ** 2, axis=2))

            _, skin_occ_mask = cv2.threshold(lab_diff.astype(np.uint8), 28, 255, cv2.THRESH_BINARY)
            skin_occ_mask = cv2.bitwise_and(skin_occ_mask, cv2.bitwise_and(texture_mask, skin_mask))

            _, object_mask = cv2.threshold(lab_diff.astype(np.uint8), 65, 255, cv2.THRESH_BINARY)
            object_mask = cv2.bitwise_and(object_mask, center_mask)

            occlusion = cv2.bitwise_or(occlusion, cv2.bitwise_or(skin_occ_mask, object_mask))

        # 5. Detalle fino (Laplaciano) — SOLO del target (v5.1045)
        # Antes: diff_lap = |tgt_lap| − |swp_lap|×1.5 → el swapped (upscale 128→256 del
        # inswapper) es MÁS SUAVE que el original, así que |tgt_lap| > |swp_lap|×1.5 en
        # toda la cara → detail_mask cubría el 60-85% → máscara restada 0.88 → swap casi
        # invisible (probe_ab: A=1.43, B=1.30, D=1.63). Ahora solo la textura propia del
        # target (micrófonos, gafas, manos, objetos) genera detalle a proteger.
        tgt_lap = cv2.Laplacian(tgt_gray, cv2.CV_32F)
        tgt_lap_abs = np.abs(tgt_lap)
        lap_thresh = max(35, np.mean(tgt_lap_abs[tgt_lap_abs > 0]) * 0.6 if np.any(tgt_lap_abs > 0) else 40)
        _, detail_mask = cv2.threshold(tgt_lap_abs.astype(np.uint8), lap_thresh, 255, cv2.THRESH_BINARY)

        protect_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(protect_mask, (int(w * 0.35), int(h * 0.38)), int(min(w, h) * 0.06), 255, -1)
        cv2.circle(protect_mask, (int(w * 0.65), int(h * 0.38)), int(min(w, h) * 0.06), 255, -1)

        mouth_area = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mouth_area, (w // 2, int(h * 0.70)), (int(w * 0.24), int(h * 0.15)), 0, 0, 360, 255, -1)

        has_mouth_object = False
        mouth_obj_score = float(np.mean(cv2.bitwise_and(object_mask, mouth_area)))
        if mouth_obj_score > 1.5:
            has_mouth_object = True
        if not has_mouth_object:
            mouth_skin_score = float(np.mean(cv2.bitwise_and(skin_occ_mask, mouth_area)))
            if mouth_skin_score > 1.5:
                has_mouth_object = True
        if not has_mouth_object:
            mouth_tex_score = float(np.mean(cv2.bitwise_and(texture_mask, mouth_area)))
            if mouth_tex_score > 1.0:
                has_mouth_object = True

        if not has_mouth_object:
            detail_mask = cv2.bitwise_and(detail_mask, cv2.bitwise_not(mouth_area))

        detail_mask = cv2.bitwise_and(detail_mask, cv2.bitwise_not(protect_mask))
        occlusion = cv2.bitwise_or(occlusion, detail_mask)

        kernel_open = np.ones((3, 3), np.uint8)
        occlusion = cv2.morphologyEx(occlusion, cv2.MORPH_OPEN, kernel_open, iterations=2)
        kernel_dilate = np.ones((7, 7), np.uint8)
        occlusion = cv2.dilate(occlusion, kernel_dilate, iterations=1)
        occlusion = cv2.GaussianBlur(occlusion, (11, 11), 0)

        return occlusion.astype(np.float32) / 255.0
    except Exception as e:
        print(f"[OCCLUSION] Error: {e}")
        try:
            return np.zeros(face_img.shape[:2], dtype=np.float32)
        except Exception:
            return np.zeros((1, 1), dtype=np.float32)


def create_soft_mask(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int], feather: int = 35, occlusion_mask: np.ndarray = None) -> np.ndarray:
    """
    Crea una máscara de integración con feathering suave y degradado dinámico. (v5.10)
    """
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0:
        return np.zeros((h, w), dtype=np.float32)

    mask = np.zeros((h, w), dtype=np.float32)
    
    # Centro y ejes para elipse (v5.4.10: reducido para evitar borde sharp/círculo gris)
    center = ((x1 + x2) // 2, (y1 + y2) // 2)
    axes = (int(bw * 0.40), int(bh * 0.44))  # v5.42: Reducido para máscara más ajustada
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    
    # v5.8: Atenuación superior progresiva (Top-Down Fade) más suave
    gradient = np.ones((bh, bw), dtype=np.float32)
    fade_height = int(bh * 0.30)
    for i in range(fade_height):
        gradient[i, :] = (i / fade_height) ** 0.6
    
    full_gradient = np.ones_like(mask)
    full_gradient[y1:y2, x1:x2] = gradient
    mask = mask * full_gradient

    # Protección de oclusiones (más suave para evitar bordes sharp)
    if occlusion_mask is not None:
        full_occ_mask = np.zeros_like(mask)
        occ_resized = cv2.resize(occlusion_mask, (bw, bh), interpolation=cv2.INTER_LINEAR)
        occ_resized = cv2.GaussianBlur(occ_resized, (5, 5), 0)
        full_occ_mask[y1:y2, x1:x2] = occ_resized
        mask = np.clip(mask - (full_occ_mask * 0.85), 0, 1)
    
    # Blur dinámico más amplio para feathering suave (v5.10: doble blur para eliminar halo)
    blur_size = int(max(feather, min(bw, bh) // 8)) | 1  # v5.42: Blur reducido para no exceder cara
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    # Segundo blur con kernel más pequeño para suavizado fino
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    
    return np.clip(mask, 0, 1.0)



def blend_with_poisson(source: np.ndarray, target: np.ndarray, mask: np.ndarray, center: Tuple[int, int], clone_type: int = cv2.MIXED_CLONE) -> np.ndarray:
    """
    Realiza blending usando Poisson blending mejorado para evitar halos negros. (v5.10)
    clone_type: cv2.MIXED_CLONE (default) o cv2.NORMAL_CLONE
    """
    try:
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        original_mask = mask.copy()
        
        # v5.15: Sin re-blur — la máscara ya llega suave desde ProcessMgr (blur_sz + tail truncation).
        # Re-difuminar aquí re-extiende la zona de transición y anula el tail truncation,
        # creando halo visible. Para NORMAL_CLONE, máscara ajustada = mejor integración.
        
        # Recortar la zona activa
        x, y, bw, bh = cv2.boundingRect(mask)
        if bw <= 1 or bh <= 1:
            raise ValueError("Poisson mask is empty")

        pad = 18
        src_h, src_w = source.shape[:2]
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(src_w, x + bw + pad)
        y2 = min(src_h, y + bh + pad)

        source_crop = source[y1:y2, x1:x2]
        mask_crop = mask[y1:y2, x1:x2]

        if source_crop.size == 0 or mask_crop.size == 0:
            raise ValueError("Poisson crop is empty")

        # Bordes degradados para evitar crash de seamlessClone
        mask_crop[0, :] = 0
        mask_crop[-1, :] = 0
        mask_crop[:, 0] = 0
        mask_crop[:, -1] = 0

        crop_center = (x1 + source_crop.shape[1] // 2, y1 + source_crop.shape[0] // 2)
        
        result = cv2.seamlessClone(
            source_crop.astype(np.uint8),
            target.astype(np.uint8),
            mask_crop,
            crop_center,
            clone_type
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
    
    # 3. Detectar oclusiones (pelo, flequillo, objetos)
    occlusion_mask = detect_foreground_occlusion(swapped_face, target_region)
    
    # 4. Crear máscara suave para blending con feathering generoso (v5.10: 50px)
    feather_amount = options.get('feather_amount', 50)
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
