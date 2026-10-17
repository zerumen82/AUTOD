import cv2
import numpy as np
from typing import List, Optional
from roop.face_util import get_face_analyser, Face, _has_face_geometry
import roop.globals
from roop.faceswap_config import landmarks_fit_bbox

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


def _filter_false_positive_faces(faces: List[Face]) -> List[Face]:
    """v5.1054b: Elimina detecciones sin geometría anatómica (falsos positivos de
    RetinaFace en baja calidad: texto, manos, objetos, ruido de compresión).
    La contención de landmarks en su bbox se exige a cualquier confianza.
    El gate anatómico restante SOLO se aplica a det<0.50. Con det>=0.50 la
    confianza del detector es alta = cara real aunque los kps parezcan "invertidos"
    (en perfiles detectados vía pasada rotada 90/270 el orden horizontal le/re queda
    invertido legítimamente al transformar los kps de vuelta al frame original).
    Los falsos positivos reales de baja calidad se medían con det 0.10-0.40.
    Reutiliza _has_face_geometry de face_util (mismo gate que las caras de source).
    Solo filtra caras con kps (5,2) de InsightFace — MediaPipe u otro formato pasan.
    """
    try:
        if not faces:
            return faces
        kept = []
        for f in faces:
            kps = getattr(f, 'kps', None)
            if kps is None:
                kept.append(f)
                continue
            kps_arr = np.asarray(kps, dtype=np.float32) if not isinstance(kps, np.ndarray) else kps
            if kps_arr.ndim != 2 or kps_arr.shape != (5, 2):
                kept.append(f)
                continue
            det = float(getattr(f, 'det_score', 0.0) or 0.0)
            if not landmarks_fit_bbox(kps_arr, getattr(f, 'bbox', None)):
                print(f"[FACE] Landmarks fuera del bbox (det={det:.2f}); detección descartada")
                continue
            if det >= 0.50:
                kept.append(f)
                continue
            try:
                orig_kps = f.kps
                f.kps = kps_arr
                try:
                    _ok = _has_face_geometry(f)
                finally:
                    f.kps = orig_kps
                if _ok:
                    kept.append(f)
                else:
                    print(f"[FACE] Detección sin geometría kps rechazada (det={det:.2f}) - falso positivo")
            except Exception:
                kept.append(f)
        return kept
    except Exception as error:
        print(f"[FACE] No se pudo validar la geometría: {error}; detecciones descartadas")
        return []


def _same_face_boxes(bbox1, bbox2) -> bool:
    """Detecta si dos bboxes corresponden a la MISMA cara física aunque sean
    detecciones a distinta escala (1024 base vs 640 closeup → cajas anidadas o
    desplazadas con IoU < 0.5 que el dedup clásico no capturaba).
    Duplicados = misma cara → el tracking los puntuaba casi idénticos contra el
    embedding lockeado (margin 0.01-0.05) y mis gates de identidad los rechazaban
    → bucle RE-BUSQUEDA/ghost/hold = parpadeo (v5.1053 fix de artefactos).
    """
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox1]
        u1, v1, u2, v2 = [float(v) for v in bbox2]
        inter_w = max(0.0, min(x2, u2) - max(x1, u1))
        inter_h = max(0.0, min(y2, v2) - max(y1, v1))
        inter = inter_w * inter_h
        area1 = max(1.0, (x2 - x1) * (y2 - y1))
        area2 = max(1.0, (u2 - u1) * (v2 - v1))
        if inter / max(1.0, area1 + area2 - inter) > 0.50:
            return True
        # Caja anidada/contenida: la intersección cubre gran parte de la menor.
        # v5.1203: solo cuenta como duplicada si ambos tamaños son comparables.
        # Misma cara detectada a dos escalas (1024 vs closeup 640) difiere
        # 1,04x-1,59x de área (medido en 02457); el tope 6x deja margen. Sin él,
        # una cara grande válida se borraba al declararla duplicada de una
        # detección minúscula (29x34 px) contenido dentro de su propio bbox:
        # en 02457 la sujeto principal de perfil (358x653, det=0.47) perdía su
        # sitio frente a un falso positivo de det=0.10 y el swap acababa en una
        # persona de fondo.
        if inter / min(area1, area2) > 0.35:
            area_ratio = max(area1, area2) / min(area1, area2)
            if area_ratio <= 6.0:
                return True
        # Centros cercanos + tamaños comparables (duplicado desplazado)
        cx1, cy1 = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        cx2, cy2 = (u1 + u2) / 2.0, (v1 + v2) / 2.0
        w_max = max(x2 - x1, u2 - u1)
        size_ratio = max(area1, area2) / min(area1, area2)
        if ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5 < 0.35 * w_max and size_ratio < 3.0:
            return True
        return False
    except Exception:
        return False


def validate_face_detection(face_data, min_face_size: int = 15, max_aspect_ratio: float = 4.0) -> bool:
    """
    Valida que una detección de cara sea razonable.
    Versión RELAJADA para evitar falsos negativos.
    """
    try:
        # Validar bbox
        if not hasattr(face_data, 'bbox') or face_data.bbox is None:
            return False

        x1, y1, x2, y2 = face_data.bbox
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        # Validar tamaño mínimo - Evitar ruidos minúsculos
        if width < min_face_size or height < min_face_size:
            return False

        # Validar relación de aspecto - Una cara no es extremadamente alargada
        if width > 0 and height > 0:
            aspect_ratio = max(width, height) / min(width, height)
            if aspect_ratio > max_aspect_ratio:
                return False

        # Validar score de detección (si existe)
        det_score = face_data.get('det_score', face_data.get('score', 0.0))
        if det_score is None:
            det_score = 0.0
        if det_score < 0.08:
            return False

        # OMITIR validaciones estructurales complejas que están fallando en caras reales
        # Solo verificar que existan keypoints si los tiene
        if hasattr(face_data, 'kps') and face_data.kps is not None:
            if len(face_data.kps) < 3:
                return False

        return True

    except Exception:
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
            # v5.1053: _same_face_boxes captura además duplicados a distinta escala
            # (IoU clásico 0.5 se perdía las cajas anidadas 1024-vs-640)
            if _same_face_boxes(face.bbox, kept_face.bbox):
                is_duplicate = True
                # print(f"[DEBUG] Cara duplicada detectada (IoU={iou:.3f}), omitiendo")
                break

        if not is_duplicate:
            filtered.append(face)

    # if len(faces) != len(filtered):
        # print(f"[INFO] Filtrado de duplicados: {len(faces)} -> {len(filtered)} caras")

    return filtered


def get_all_faces_with_rotation(frame: np.ndarray, min_score: float = None, for_target: bool = False, prefer_fast: bool = False, include_180: bool = False) -> List[Face]:
    """
    Detecta caras en la orientación original y en 90°/270°; puede incluir 180° para fotos.
    Retorna las caras encontradas con sus coordenadas corregidas a la orientación original.

    Args:
        frame: Imagen en formato numpy array (BGR)
        min_score: Puntuación mínima para filtrar caras detectadas (opcional)
                   Si es None, usa roop.globals.distance_threshold con fallback permisivo
        for_target: Si True, usar umbral más permisivo para detección de destino

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
        
        # Para detección de destino (target): usar umbral permisivo para captar perfiles extremos
        if for_target:
            effective_threshold = 0.10  # v5.11: Reducido de 0.20 para captar caras en perfiles extremos
        else:
            effective_threshold = max(0.10, threshold * 0.5)
        
        try:
            def _run_passes():
                found = []
                # 1. Intentar orientación original (fast path)
                faces = analizador.get(frame_rgb)
                if faces and len(faces) > 0:
                    valid_faces = []
                    for f in faces:
                        face_score = f.get('det_score', f.get('score', None))
                        if face_score is None:
                            face_score = 0.0
                        if face_score >= effective_threshold:
                            valid_faces.append(f)

                    if valid_faces:
                        converted = convert_faces_to_face_objects(valid_faces)
                        found.extend(converted)

                # 2. Probar rotaciones adicionales para capturar perfiles extremos
                #    v5.11: siempre intentar rotaciones (no solo si original falló)
                h, w = frame_rgb.shape[:2]
                angles = ([90, 270, 180] if include_180 else [90, 270])
                if getattr(roop.globals, 'autorotate_faces', True) is False:
                    angles = []
                for angle in angles:
                    try:
                        if angle == 90:
                            rotated = cv2.rotate(frame_rgb, cv2.ROTATE_90_CLOCKWISE)
                        elif angle == 270:
                            rotated = cv2.rotate(frame_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        else:
                            rotated = cv2.rotate(frame_rgb, cv2.ROTATE_180)

                        rot_faces = analizador.get(rotated)
                        if rot_faces and len(rot_faces) > 0:
                            valid_rot = []
                            for f in rot_faces:
                                fs = f.get('det_score', f.get('score', 0.0))
                                if fs >= effective_threshold:
                                    valid_rot.append(f)
                            if valid_rot:
                                transformed = transform_faces_from_rotation(valid_rot, angle, w, h)
                                found.extend(transformed)
                    except Exception as exc:
                        print(f"[FACE] Rotación {angle}° falló: {exc}", flush=True)

                # Filtrar duplicados si hay múltiples detecciones
                return filter_duplicate_faces(found) if found else []

            orig_det_size = analizador.det_size
            orig_input_size = analizador.det_model.input_size
            try:
                if prefer_fast:
                    # v5.1053: closeup activo -> det_size 640 primero (fast path, ~2.5x más rápido).
                    # Si 640 encuentra caras, se omite la pasada 1024 completa (original + rotaciones).
                    analizador.det_size = (640, 640)
                    analizador.det_model.input_size = (640, 640)
                    all_faces = _run_passes()
                    if not all_faces:
                        # Fallback a 1024 solo si 640 no encontró nada (caras diminutas)
                        analizador.det_size = orig_det_size
                        analizador.det_model.input_size = orig_input_size
                        all_faces = _run_passes()
                        if all_faces:
                            print(f"[FACE] Closeup 640 sin caras - fallback det_size 1024: {len(all_faces)} cara(s)")
                else:
                    all_faces = _run_passes()
            finally:
                analizador.det_size = orig_det_size
                analizador.det_model.input_size = orig_input_size

            # 3. v5.1022: Closeup detection con det_size reducido (640)
            #    RetinaFace con det_size=1024 NO detecta caras que ocupan >50% del frame
            #    al usar det_size=640, la cara gigante en modelo input es menor y
            #    entra dentro del rango de anchors (~512px máximo)
            #    (Solo para modo no-closeup: con prefer_fast la pasada 640 ya se hizo)
            if not prefer_fast:
                try:
                    h_cam, w_cam = frame_rgb.shape[:2]
                    frame_aspect = w_cam / float(max(1, h_cam))
                    is_portrait = frame_aspect < 0.85
                    # Activar si: no hay caras, o todas las caras detectadas son pequeñas
                    # (sugiere que la cara principal en closeup no fue detectada por det_size=1024)
                    max_face_ratio = 0.08 if is_portrait else 0.12
                    all_small = all(
                        (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]) < max_face_ratio * w_cam * h_cam
                        for f in all_faces
                    ) if all_faces else True
                    needs_closeup = not all_faces or all_small
                    if needs_closeup:
                        orig_det_size = analizador.det_size
                        orig_input_size = analizador.det_model.input_size
                        try:
                            analizador.det_size = (640, 640)
                            analizador.det_model.input_size = (640, 640)
                            cu_faces = analizador.get(frame_rgb)
                            if cu_faces:
                                valid_cu = [f for f in cu_faces if f.get('det_score', 0.0) >= effective_threshold]
                                if valid_cu:
                                    converted_cu = convert_faces_to_face_objects(valid_cu)
                                    new_faces = 0
                                    for cf in converted_cu:
                                        # v5.1053: dedup robusta a escala (antes IoU 0.5
                                        # → la pasada 640 añadía la misma cara duplicada)
                                        is_dup = any(_same_face_boxes(cf.bbox, ef.bbox) for ef in all_faces)
                                        if not is_dup:
                                            all_faces.append(cf)
                                            new_faces += 1
                                    if new_faces > 0:
                                        print(f"[FACE] Closeup det_size=640: {new_faces} cara(s) adicional(es) - total {len(all_faces)}")
                                    # v5.1053: dedup final — los añadidos pueden duplicarse entre sí
                                    all_faces = filter_duplicate_faces(all_faces)
                        finally:
                            analizador.det_size = orig_det_size
                            analizador.det_model.input_size = orig_input_size
                except Exception:
                    pass

            # 4. Fallback a MediaPipe si RetinaFace no detectó nada
            if not all_faces:
                if MEDIAPIPE_DETECTOR_AVAILABLE:
                    try:
                        mp_faces = get_all_faces_mediapipe(frame)
                        if mp_faces:
                            print(f"[FACE] MediaPipe fallback detectó {len(mp_faces)} cara(s) — RetinaFace no encontró")
                            all_faces = mp_faces
                    except Exception:
                        pass

            # 5. v5.1139 — CLOSEUP EXTREMO (descalado): si el lado largo escalado al
            # det_size supera ~1024px, la cara es MÁS GRANDE que el máximo de anchors
            # del detector (~512px) y RetinaFace NO la ve a ninguna det_size (ni 640
            # ni 1024 ni rotaciones). "Foto destino = primer plano que casi ocupa todo
            # y no detecta la cara". Fix: reducir la imagen a la mitad (la cara pasa
            # bajo el techo de anchors) y re-mapear las cajas/kps al frame original.
            if not all_faces:
                try:
                    _h5, _w5 = frame_rgb.shape[:2]
                    _ds5 = float(analizador.det_size[0]) if getattr(analizador, 'det_size', None) else 640.0
                    _long_scaled5 = max(_w5, _h5) * (_ds5 / float(min(_w5, _h5)))
                    if _long_scaled5 > 1024.0:
                        _scale5 = 0.5
                        _small5 = cv2.resize(
                            frame_rgb,
                            (max(1, int(_w5 * _scale5)), max(1, int(_h5 * _scale5))),
                            interpolation=cv2.INTER_AREA,
                        )
                        _ds_faces5 = analizador.get(_small5)
                        if _ds_faces5:
                            _valid5 = [
                                _f5 for _f5 in _ds_faces5
                                if float(_f5.get('det_score', _f5.get('score', 0.0)) or 0.0) >= effective_threshold
                            ]
                            if _valid5:
                                _conv5 = convert_faces_to_face_objects(_valid5)
                                for _cf5 in _conv5:
                                    _cf5.bbox = [
                                        _cf5.bbox[0] / _scale5, _cf5.bbox[1] / _scale5,
                                        _cf5.bbox[2] / _scale5, _cf5.bbox[3] / _scale5,
                                    ]
                                    if getattr(_cf5, 'kps', None) is not None:
                                        _cf5.kps = [[_p[0] / _scale5, _p[1] / _scale5] for _p in _cf5.kps]
                                    _cf5.det_score = float(getattr(_cf5, 'det_score', 0.0) or 0.0) * 0.98
                                    _cf5.score = _cf5.det_score
                                    all_faces.append(_cf5)
                                if all_faces:
                                    print(
                                        f"[FACE] Primer plano extremo (lado escalado {_long_scaled5:.0f}px > 1024): "
                                        f"descalado 0.5 detectó {len(_conv5)} cara(s)"
                                    )
                except Exception:
                    pass
        except Exception:
            pass

        # v5.1054b: Filtro anti falsos positivos — RetinaFace marca texto/manos/objetos
        # como "caras" en baja calidad con det_score 0.10-0.40 (ruido, compresión).
        # Reutiliza el mismo gate geométrico que las caras de source (v5.1050/52):
        # solo las caras con kps anatómicos (ojos arriba de boca, nariz entre medias)
        # sobreviven. Tolerante a perfiles/poses. No rompe MediaPipe (kps distinto).
        all_faces = _filter_false_positive_faces(all_faces)

        return all_faces

    except Exception:
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
        gender = face_data.get('gender', None)
        age = face_data.get('age', None)
        
        score_value = face_data.get('det_score', face_data.get('score', 0.0))
        if score_value is None:
            score_value = 0.0
        
        face_obj = Face(
            bbox=face_data.bbox.astype(int).tolist(),
            score=score_value,
            det_score=score_value,
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
        
        score_value = face_data.get('det_score', face_data.get('score', 0.0))
        if score_value is None:
            score_value = 0.0
        
        face_obj = Face(
            bbox=face_data.bbox.astype(int).tolist(),
            score=score_value,
            det_score=score_value,
            kps=face_data.kps.tolist() if face_data.kps is not None else None,
            embedding=face_data.embedding.tolist() if face_data.embedding is not None else None,
            landmark_106=landmark_106,
        )

        # Transformar bbox según el ángulo
        if angle == 90:
            # Inversa de 90° horario: (x, y) original = (y', height - 1 - x').
            x1, y1, x2, y2 = face_obj.bbox
            new_x1 = y1
            new_y1 = height - x2
            new_x2 = y2
            new_y2 = height - x1
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
            # Inversa de 270° horario: (x, y) original = (width - 1 - y', x').
            x1, y1, x2, y2 = face_obj.bbox
            new_x1 = width - y2
            new_y1 = x1
            new_x2 = width - y1
            new_y2 = x2
            face_obj.bbox = [new_x1, new_y1, new_x2, new_y2]

        # Transformar keypoints si existen
        if face_obj.kps is not None:
            kps = np.array(face_obj.kps, dtype=np.float32)
            if angle == 90:
                new_x = kps[:, 1].copy()
                new_y = height - 1 - kps[:, 0]
                kps[:, 0] = new_x
                kps[:, 1] = new_y
            elif angle == 180:
                kps[:, 0] = width - 1 - kps[:, 0]
                kps[:, 1] = height - 1 - kps[:, 1]
            elif angle == 270:
                new_x = width - 1 - kps[:, 1]
                new_y = kps[:, 0].copy()
                kps[:, 0] = new_x
                kps[:, 1] = new_y
            face_obj.kps = kps.tolist()

        if face_obj.landmark_106 is not None:
            points = np.asarray(face_obj.landmark_106, dtype=np.float32)
            if angle == 90:
                points = np.column_stack((points[:, 1], height - 1 - points[:, 0]))
            elif angle == 180:
                points = np.column_stack((width - 1 - points[:, 0], height - 1 - points[:, 1]))
            elif angle == 270:
                points = np.column_stack((width - 1 - points[:, 1], points[:, 0]))
            face_obj.landmark_106 = points.tolist()

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
    except Exception:
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

