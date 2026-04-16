import random
import cv2
import numpy as np
from typing import List, Tuple, Optional
from roop.face_util import get_face_analyser, Face
import roop.globals

"""
Funciones auxiliares para detección de caras con rotación automática
Ahora soporta MediaPipe como alternativa más moderna a InsightFace
"""

# Intentar importar detector MediaPipe
MEDIAPIPE_DETECTOR_AVAILABLE = False
try:
    from roop.face_detector_mediapipe import (
        detect_faces_mediapipe, 
        MEDIAPIPE_AVAILABLE,
        DetectedFace
    )
    MEDIAPIPE_DETECTOR_AVAILABLE = MEDIAPIPE_AVAILABLE
except ImportError:
    pass


def calculate_iou(bbox1, bbox2):
    """
    Calcula Intersection over Union entre dos bboxes.

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        float: IoU value (0.0 - 1.0)
    """
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Área de intersección
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)

    # Áreas de cada bbox
    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    # IoU
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def validate_face_detection(face_data, min_face_size: int = 4, max_aspect_ratio: float = 10.0) -> bool:
    """
    Valida que una detección de cara sea razonable (no un falso positivo).
    ULTRA PERMISIVA para maximizar detección en videos difíciles.

    Args:
        face_data: Datos de la cara detectada por insightface
        min_face_size: Tamaño mínimo de cara en píxeles (MUY REDUCIDO)
        max_aspect_ratio: Relación de aspecto máxima permitida (MUY AUMENTADO)

    Returns:
        bool: True si la cara es válida, False si es un falso positivo
    """
    try:
        # Validar bbox
        if not hasattr(face_data, 'bbox') or face_data.bbox is None:
            return False

        x1, y1, x2, y2 = face_data.bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Validar tamaño mínimo - ULTRA PERMISIVO para caras pequeñas/difíciles
        if width < 1 or height < 1:  # Mínimo 1x1 píxel (casi nada)
            return False

        # Validar relación de aspecto - MUY PERMISIVA para caras rotadas/ángulos extraños
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > 20.0:  # Aumentado de 10.0 a 20.0 para más tolerancia
                return False

        # Validar keypoints si están disponibles - MUY PERMISIVA
        if hasattr(face_data, 'kps') and face_data.kps is not None:
            kps = face_data.kps
            if len(kps) >= 5:
                # Verificar que AL MENOS UN keypoint esté cerca del bbox (muy permisivo)
                valid_kps = 0
                bbox_margin = max(width, height) * 0.5  # Margen amplio
                for kp in kps:
                    kx, ky = kp
                    # Keypoint válido si está dentro del bbox expandido
                    if (x1 - bbox_margin) <= kx <= (x2 + bbox_margin) and (y1 - bbox_margin) <= ky <= (y2 + bbox_margin):
                        valid_kps += 1

                # Requerir que al menos 1 keypoint esté cerca (MUY permisivo para caras difíciles)
                if valid_kps < 1:
                    # Solo rechazar si TODOS los keypoints están muy lejos
                    return False

                # Verificar posiciones relativas - ULTRA PERMISIVA para caras en movimiento/ángulos
                if len(kps) >= 5:
                    left_eye_y = kps[0][1]
                    right_eye_y = kps[1][1]
                    nose_y = kps[2][1]
                    left_mouth_y = kps[3][1]
                    right_mouth_y = kps[4][1]

                    # Los ojos deberían estar por encima de la nariz - ULTRA PERMISIVO
                    avg_eye_y = (left_eye_y + right_eye_y) / 2
                    if nose_y < avg_eye_y - 200:  # Tolerancia extrema para ángulos raros
                        # print(f"[DEBUG] Cara rechazada: nariz demasiado arriba (nose_y={nose_y:.1f}, avg_eye_y={avg_eye_y:.1f})")
                        return False  # Nariz demasiado arriba = inválido

                    # La boca debería estar debajo de la nariz - ULTRA PERMISIVO
                    avg_mouth_y = (left_mouth_y + right_mouth_y) / 2
                    if avg_mouth_y < nose_y - 50:  # Tolerancia extrema
                        # print(f"[DEBUG] Cara rechazada: boca demasiado arriba (avg_mouth_y={avg_mouth_y:.1f}, nose_y={nose_y:.1f})")
                        return False  # Boca demasiado arriba = inválido

        return True

    except Exception as e:
        # print(f"[WARN] Error validando cara: {e}")
        return False


def filter_duplicate_faces(faces: List[Face], iou_threshold: float = 0.5) -> List[Face]:
    """
    Filtra caras duplicadas basándose en IoU (Intersection over Union).
    Mantiene solo la cara con mayor score de detección.

    Args:
        faces: Lista de caras detectadas
        iou_threshold: Umbral de IoU para considerar caras como duplicadas (default: 0.5 - más estricto)

    Returns:
        Lista de caras filtradas sin duplicados
    """
    if len(faces) <= 1:
        return faces

    # Ordenar por score descendente (manejar None values)
    sorted_faces = sorted(faces, key=lambda f: f.det_score if f.det_score is not None else 0.0, reverse=True)

    filtered = []
    for face in sorted_faces:
        is_duplicate = False
        for kept_face in filtered:
            iou = calculate_iou(face.bbox, kept_face.bbox)
            if iou > iou_threshold:
                is_duplicate = True
                # print(f"[DEBUG] Cara duplicada detectada (IoU={iou:.3f}), omitiendo")
                break

        if not is_duplicate:
            filtered.append(face)

    # if len(faces) != len(filtered):
        # print(f"[INFO] Filtrado de duplicados: {len(faces)} -> {len(filtered)} caras")

    return filtered


def get_all_faces_with_rotation(frame: np.ndarray, min_score: float = None) -> List[Face]:
    """
    Detecta caras en un frame probando solo la orientación original para máxima velocidad.
    Retorna las caras encontradas con sus coordenadas corregidas a la orientación original.

    Args:
        frame: Imagen en formato numpy array (BGR)
        min_score: Puntuación mínima para filtrar caras detectadas (opcional)
                   Si es None, usa roop.globals.distance_threshold con fallback permisivo

    Returns:
        Lista de objetos Face detectados
    """
    try:
        analizador = get_face_analyser()
        if analizador is None:
            print("[ERROR] No se pudo obtener el analizador de caras")
            return []

        # Convertir a RGB para el analizador
        if len(frame.shape) == 2:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        all_faces = []

        # Usar el umbral configurado por el usuario o un valor por defecto
        if min_score is None:
            threshold = getattr(roop.globals, 'distance_threshold', 0.25)
        else:
            threshold = min_score
        
        # ULTRA PERMISIVO: InsightFace puede dar scores bajos para caras difíciles
        # Usar un umbral mínimo de 0.1 para maximizar detección
        effective_threshold = max(0.1, threshold * 0.5)  # Mitad del umbral configurado, mínimo 0.1
        
        try:
            faces = analizador.get(frame_rgb)
            if faces and len(faces) > 0:
                valid_faces = []
                for f in faces:
                    # Usar f.score que es el score de detección de insightface
                    face_score = getattr(f, 'score', None)
                    if face_score is None:
                        face_score = getattr(f, 'det_score', 0.0)
                    
                    # Aceptar caras con score sobre el umbral efectivo
                    if face_score >= effective_threshold:
                        valid_faces.append(f)
                
                if valid_faces:
                    converted = convert_faces_to_face_objects(valid_faces)
                    all_faces.extend(converted)
                    # Filtrar duplicados
                    all_faces = filter_duplicate_faces(all_faces)
        except Exception as e:
            pass

        return all_faces

    except Exception as e:
        return []


def convert_faces_to_face_objects(faces) -> List[Face]:
    """
    Convierte una lista de caras de insightface a objetos Face.
    """
    result = []
    for face_data in faces:
        # Extraer landmark_106 si está disponible
        landmark_106 = None
        if hasattr(face_data, 'landmark_2d_106') and face_data.landmark_2d_106 is not None:
            landmark_106 = face_data.landmark_2d_106.tolist()
        elif hasattr(face_data, 'landmark') and face_data.landmark is not None:
            # Algunas versiones de insightface usan 'landmark' en lugar de 'landmark_2d_106'
            lm = face_data.landmark
            if len(lm) >= 106:
                landmark_106 = lm.tolist() if hasattr(lm, 'tolist') else lm
        
        # Extraer gender y age si están disponibles
        gender = getattr(face_data, 'gender', None)
        age = getattr(face_data, 'age', None)
        
        face_obj = Face(
            bbox=face_data.bbox.astype(int).tolist(),
            score=face_data.score,
            det_score=face_data.score,
            kps=face_data.kps.tolist() if face_data.kps is not None else None,
            embedding=face_data.embedding.tolist() if face_data.embedding is not None else None,
            landmark_106=landmark_106,
            gender=gender,
            age=age
        )
        result.append(face_obj)
    return result


def transform_faces_from_rotation(faces, angle: int, width: int, height: int) -> List[Face]:
    """
    Transforma las coordenadas de caras detectadas en una imagen rotada
    a las coordenadas correspondientes en la imagen original.
    """
    result = []
    for face_data in faces:
        # Crear objeto Face
        # Extraer landmark_106 si está disponible
        landmark_106 = None
        if hasattr(face_data, 'landmark_2d_106') and face_data.landmark_2d_106 is not None:
            landmark_106 = face_data.landmark_2d_106.tolist()
        
        face_obj = Face(
            bbox=face_data.bbox.astype(int).tolist(),
            score=face_data.score,
            det_score=face_data.score,
            kps=face_data.kps.tolist() if face_data.kps is not None else None,
            embedding=face_data.embedding.tolist() if face_data.embedding is not None else None,
            landmark_106=landmark_106,
        )

        # Transformar bbox según el ángulo
        if angle == 90:
            # Rotación 90° horario: (x, y) -> (height - y, x)
            x1, y1, x2, y2 = face_obj.bbox
            new_x1 = height - y2
            new_y1 = x1
            new_x2 = height - y1
            new_y2 = x2
            face_obj.bbox = [new_x1, new_y1, new_x2, new_y2]

        elif angle == 180:
            # Rotación 180°: (x, y) -> (width - x, height - y)
            x1, y1, x2, y2 = face_obj.bbox
            new_x1 = width - x2
            new_y1 = height - y2
            new_x2 = width - x1
            new_y2 = height - y1
            face_obj.bbox = [new_x1, new_y1, new_x2, new_y2]

        elif angle == 270:
            # Rotación 270° horario: (x, y) -> (y, width - x)
            x1, y1, x2, y2 = face_obj.bbox
            new_x1 = y1
            new_y1 = width - x2
            new_x2 = y2
            new_y2 = width - x1
            face_obj.bbox = [new_x1, new_y1, new_x2, new_y2]

        # Transformar keypoints si existen
        if face_obj.kps is not None:
            kps = np.array(face_obj.kps)
            if angle == 90:
                kps[:, [0, 1]] = kps[:, [1, 0]]  # Intercambiar x, y
                kps[:, 1] = height - kps[:, 1]   # Invertir nuevo y
            elif angle == 180:
                kps[:, 0] = width - kps[:, 0]    # Invertir x
                kps[:, 1] = height - kps[:, 1]   # Invertir y
            elif angle == 270:
                kps[:, [0, 1]] = kps[:, [1, 0]]  # Intercambiar x, y
                kps[:, 0] = width - kps[:, 0]    # Invertir nuevo x
            face_obj.kps = kps.tolist()

        result.append(face_obj)

    return result


def get_first_face_with_rotation(frame: np.ndarray) -> Optional[Face]:
    """
    Obtiene el primer rostro detectado probando múltiples rotaciones.
    """
    faces = get_all_faces_with_rotation(frame)
    return faces[0] if faces else None


def get_all_faces_mediapipe(frame: np.ndarray) -> List[Face]:
    """
    Detecta caras usando MediaPipe (si está disponible).
    """
    if not MEDIAPIPE_DETECTOR_AVAILABLE:
        return []

    try:
        detected_faces = detect_faces_mediapipe(frame)
        result = []
        for df in detected_faces:
            face_obj = Face(
                bbox=df.bbox,
                score=df.score,
                det_score=df.score,
                kps=df.keypoints,
                embedding=None,  # MediaPipe no proporciona embeddings
                landmark_106=None,  # MediaPipe no proporciona landmarks 106
            )
            result.append(face_obj)
        return result
    except Exception as e:
        # print(f"[ERROR] Error en detección MediaPipe: {e}")
        return []


def get_first_face_mediapipe(frame: np.ndarray) -> Optional[Face]:
    """
    Obtiene el primer rostro detectado usando MediaPipe.
    """
    faces = get_all_faces_mediapipe(frame)
    return faces[0] if faces else None


# Alias para compatibilidad
get_all_faces = get_all_faces_with_rotation
get_all_faces_smart = get_all_faces_with_rotation
get_first_face = get_first_face_with_rotation

# Exportar funciones principales
__all__ = [
    "get_all_faces_with_rotation",
    "get_first_face_with_rotation",
    "get_all_faces_mediapipe",
    "get_first_face_mediapipe",
    "get_all_faces",
    "get_first_face",
    "filter_duplicate_faces",
    "validate_face_detection",
    "calculate_iou",
    "convert_faces_to_face_objects",
    "transform_faces_from_rotation",
    "MEDIAPIPE_DETECTOR_AVAILABLE",
    "Face",
]

