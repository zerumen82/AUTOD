def adjust_face_brightness(
    face: np.ndarray,
    target_region: np.ndarray,
    strength: float = 0.2,
    brightness_ratio_clip: Tuple[float, float] = (0.7, 1.3),
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: Tuple[int, int] = (8, 8),
    balance_shadows_highlights: bool = False,
    preserve_luminosity: bool = True,
    detail_method: str = 'clahe',
    color_temperature: float = 6500.0,
    denoising_strength: float = 0.0,
    sharpness: float = 0.0,
) -> np.ndarray:
    """
    Ajusta el brillo de la cara para que coincida con la regi�n objetivo.

    NOTA: strength reducido a 0.2 por defecto para preservar mejor la identidad de origen.

    Args:
        face: Imagen de la cara a ajustar
        target_region: Regi�n objetivo para matching
        strength: Fuerza del ajuste (0.0 - 1.0)

    Returns:
        Cara con brillo ajustado
    """
    try:
        if face is None or target_region is None or face.size == 0 or target_region.size == 0:
            return face

        # Asegurar mismo tama�o
        if face.shape != target_region.shape:
            target_region = cv2.resize(target_region, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Calcular brillo promedio
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target_region, cv2.COLOR_BGR2GRAY)

        face_brightness = np.mean(face_gray)
        target_brightness = np.mean(target_gray)

        # Calcular factor de ajuste con l�mites para evitar excesos
        if face_brightness > 10:  # Evitar divisi�n por valores muy peque�os
            brightness_ratio = target_brightness / face_brightness
            # Limitar ratio para evitar cambios extremos
            brightness_ratio = np.clip(brightness_ratio, 0.7, 1.3)

            # Aplicar con strength
            adjustment = 1.0 + (brightness_ratio - 1.0) * strength

            # Ajustar en espacio LAB (solo canal L)
            face_lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB).astype(np.float32)
            face_lab[:, :, 0] = np.clip(face_lab[:, :, 0] * adjustment, 0, 255)

            result = cv2.cvtColor(face_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Aplicar improvement de detalles (detail_method)
        if detail_method == 'clahe':
            clape = _cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
            result_lab = _cv2.cvtColor(result, _cv2.COLOR_BGR2LAB).astype(np.float32)
            result_lab[:, :, 0] = _cv2.normalize(clape.apply(result_lab[:, :, 0].astype(np.uint8)), None, 0, 255, _cv2.NORM_MINMAX)
            result = _cv2.cvtColor(result_lab.astype(np.uint8), _cv2.COLOR_LAB2BGR)
        elif detail_method == 'local':
            # Local enhancement using unsharp mask
            blurred = _cv2.GaussianBlur(result, (0, 0), 3.0)
            result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
        elif detail_method == 'both':
            # CLAHE first
            clahe = _cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid_size)
            result_lab = _cv2.cvtColor(result, _cv2.COLOR_BGR2LAB).astype(np.float32)
            result_lab[:, :, 0] = _cv2.normalize(clahe.apply(result_lab[:, :, 0].astype(np.uint8)), None, 0, 255, _cv2.NORM_MINMAX)
            result = _cv2.cvtColor(result_lab.astype(np.uint8), _cv2.COLOR_LAB2BGR)
            # Then local enhancement
            blurred = _cv2.GaussianBlur(result, (0, 0), 3.0)
            result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)

        # Placeholder para color_temperature (balanceo por Kelvin)
        # TODO: Implementar ajuste de temperatura de color cuando sea necesario
        # if color_temperature != 6500.0:
        #     # Ajustar balances de canal basados en temperatura
        #     pass

        # Placeholder para denoising_strength (filtrado bilateral)
        # TODO: Implementar denoising bilateral cuando sea necesario
        # if denoising_strength > 0:
        #     result = _cv2.bilateralFilter(result, 9, 75, 75)

        # Placeholder para sharpness (unsharp mask)
        # TODO: Implementar unsharp mask cuando sea necesario
        # if sharpness > 0:
        #     result = _cv2.addWeighted(result, 1.5 + sharpness, _cv2.GaussianBlur(result, (0, 0), 3), -0.5 + sharpness/2, 0)

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

    # 1. Ajustar colores para mejor integraci�n
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
    # Los parámetros de detección de bordes son configurables:
    # - edge_sobel_threshold: Umbral para detección de bordes Sobel
    # - edge_strength: Peso de reducción en bandas de borde
    # - edge_kernel_size: Tamaño del kernel para dilatar bordes
    # - edge_blur_kernel: Tamaño del kernel para suavizar máscara de bordes
    feather_amount = options.get('feather_amount', 50)
    # Los parámetros de detección de bordes son configurables desde options
    edge_sobel_threshold = options.get('edge_sobel_threshold', 28.0)
    edge_strength = options.get('edge_strength', 0.85)
    edge_kernel_size = options.get('edge_kernel_size', (5, 5))
    edge_blur_kernel = options.get('edge_blur_kernel', (0, 0))

    

    soft_mask = create_soft_mask(bbox, frame.shape, feather=feather_amount, occlusion_mask=occlusion_mask)

    # Aplicar máscara consciente de bordes para reducir halo
    # Esto reduce el peso del blend solo en bandas de borde, protegiendo contornos
    # y mejorando fidelidad, mientras mantiene el blend pleno en zonas planas
    soft_mask = _apply_edge_aware_mask(swapped_face, frame, bbox, soft_mask,
                                       edge_strength=edge_strength,
                                       edge_sobel_threshold=edge_sobel_threshold,
                                       edge_kernel_size=edge_kernel_size,
                                       edge_blur_kernel=edge_blur_kernel)

    # 5. Mejorar detalle facial (CLAHE + nitidez local suave)
    # Solo se aplica si se solicita o si hay detalle detectado
    if options is not None:
        detail_strength = options.get('detail_strength', 0.2)
        if detail_strength > 0:
            swapped_face = _enhance_face_detail(swapped_face, options)

    # 6. Re-blur suave para suavizar bordes de transici�n
    # (V5.10: blur generoso para eliminar halo)
    feather_amount = options.get('feather_amount', 50)
    soft_mask = create_soft_mask(bbox, frame.shape, feather=feather_amount, occlusion_mask=None)
    if details_mask is not None:
        # Aplica blur suave solo en la zona de transici�n
        blurred_face = cv2.GaussianBlur(swapped_face, (max(1, int(feather_amount/8)), max(1, int(feather_amount/8))), 0)
        alpha = np.clip(details_mask, 0, 1)
        swapped_face = (source * (1 - alpha) + blurred_face * alpha).astype(np.uint8)

    return swapped_face, soft_mask


def _apply_edge_aware_mask(
    swapped_face: np.ndarray,
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    soft_mask: np.ndarray,
    edge_sobel_threshold: float = 28.0,
    edge_kernel_size: Tuple[int, int] = (5, 5),
    edge_blur_kernel: Tuple[int, int] = (0, 0)
) -> np.ndarray:
    """
    Aplica una m�scara consciente de bordes para reducir halo.
    
    En los contornos del sujeto (pelo/hombros vs fondo) la imagen regenerada
    discrepa ligeramente de la original ? 'borde'/halo visible + p�rdida de
    fidelidad. Esta mezcla reduce el peso generativo SOLO en las bandas de
    borde (protege contornos, mejora fidelidad) y mantiene el blend pleno en
    zonas planas (piel, fondos).
    """
    try:
        import cv2 as _cv2
        x1, y1, x2, y2 = map(int, bbox)
        h, w = y2 - y1, x2 - x1
        
        # Extraer regi�n del frame original para detecci�n de bordes
        original_region = frame[y1:y2, x1:x2]
        if original_region.size == 0:
            return soft_mask
        
        # Convertir a gris y detectar bordes con Sobel
        gray = _cv2.cvtColor(original_region, _cv2.COLOR_BGR2GRAY)
        gx = _cv2.Sobel(gray, _cv2.CV_32F, 1, 0, ksize=3)
        gy = _cv2.Sobel(gray, _cv2.CV_32F, 0, 1, ksize=3)
        mag = _cv2.sqrt(gx * gx + gy * gy)
        
        # Normalizar y umbralizar bordes
        edge_mask = (mag > edge_sobel_threshold).astype(np.uint8) * 255
        
        # Dilatar los bordes para crear una banda de protecci�n
        k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, edge_kernel_size)
        edge_mask = _cv2.dilate(edge_mask, k)
        
        # Suavizar la m�scara de bordes
        edge_mask = _cv2.GaussianBlur(edge_mask, edge_blur_kernel, 2.0)
        edge_mask = edge_mask.astype(np.float32) / 255.0
        
        # Aplicar reducci�n del peso solo en bandas de borde
        # (1.0 - 0.85 * em) reduce el peso en bordes pero lo mantiene pleno en zonas planas
        if edge_mask.shape[:2] == soft_mask.shape[:2]:
            soft_mask = soft_mask * (1.0 - edge_strength * edge_mask)
            soft_mask = np.clip(soft_mask, 0.0, 1.0)
        
        return soft_mask
    except Exception:
        # Si falla la detecci�n de bordes, devolver la m�scara original
        return soft_mask


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
        soft_mask: M�scara suave (opcional)
        blend_mode: Modo de blending ("poisson", "weighted", "direct")

    Returns:
        Frame con la cara integrada
    """
    x1, y1, x2, y2 = map(int, bbox)
    w, h = x2 - x1, y2 - y1

    # Asegurar que la cara tenga el tama�o correcto
    if swapped_face.shape[:2] != (h, w):
        swapped_face = cv2.resize(swapped_face, (w, h), interpolation=cv2.INTER_LANCZOS4)

    result = frame.copy()

    if blend_mode == "poisson":
        # Crear m�scara para Poisson
        if soft_mask is not None:
            mask = (soft_mask[y1:y2, x1:x2] * 255).astype(np.uint8)
        else:
            mask = np.ones((h, w), dtype=np.uint8) * 255

        center = (x1 + w // 2, y1 + h // 2)
        result = blend_with_poisson(swapped_face, result, mask, center)

    elif blend_mode == "weighted":
        # Blending ponderado con m�scara suave
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
