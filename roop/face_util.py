"""Version CORREGIDA de face_util.py para preservar colores en la extraccion de caras

"""

# AGREGAR CARPETA DLL AL PATH AL INICIO - CRITICO PARA CUDA
import os
dll_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dll"))
os.environ["PATH"] = dll_path + ";" + os.environ.get("PATH", "")

import logging
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, List, Optional, Tuple
import warnings

import cv2
import numpy as np

# Suprimir warnings de asyncio
warnings.filterwarnings("ignore", category=DeprecationWarning, module="asyncio")
warnings.filterwarnings("ignore", category=FutureWarning, message="`rcond` parameter")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Simple logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Importaciones basicas
from roop.capturer import get_image_frame, get_video_frame
from roop.utilities import convert_to_gradio

# Variable global simple (sin inicializacion automatica)
FACE_ANALYSER = None
FACE_ANALYSER_CPU = None  # Fallback a CPU
BLOQUEO = threading.Lock()
FACE_ANALYSER_STATUS = "not_initialized"
USE_CPU_FALLBACK = False  # Activar fallback a CPU si CUDA falla

# Configuracion simple pero optimizada
DETECTION_SIZE = (640, 640)
MAX_DETECTION_ATTEMPTS = 3
INIT_TIMEOUT = 60
face_detection_padding = 0.3

# Caches optimizados para velocidad
CACHE_RESULTADOS = {}
CACHE_DETECCIONES = {}
MAX_FACES_TO_DETECT = 100
MAX_CACHE_SIZE = 100


def _initialize_face_analyser_cpu():
    """Inicializa el analizador con CPU (fallback)"""
    try:
        print("[INFO] Inicializando FaceAnalysis con CPU (fallback)...")

        from insightface.app import FaceAnalysis

        analyser_instance = FaceAnalysis(
            name="buffalo_l",
            root=os.path.dirname(os.path.dirname(__file__)),
            download=True,
            providers=['CPUExecutionProvider'],
        )
        analyser_instance.prepare(ctx_id=-1, det_size=DETECTION_SIZE)

        print("[SUCCESS] FaceAnalysis CPU inicializado")
        return analyser_instance

    except Exception as e:
        print(f"[ERROR] Error inicializando FaceAnalysis CPU: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_face_analyser_cpu():
    """Obtiene el analizador de CPU (fallback)"""
    global FACE_ANALYSER_CPU

    if FACE_ANALYSER_CPU is not None:
        return FACE_ANALYSER_CPU

    with BLOQUEO:
        if FACE_ANALYSER_CPU is not None:
            return FACE_ANALYSER_CPU

        FACE_ANALYSER_CPU = _initialize_face_analyser_cpu()
        return FACE_ANALYSER_CPU


def preprocess_image_for_detection(img: np.ndarray) -> np.ndarray:
    """Preprocesa una imagen para mejorar la deteccion de caras"""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
        sharpened = cv2.addWeighted(img_enhanced, 1.5, blurred, -0.5, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        return sharpened
    except Exception as e:
        logger.warning(f"Error en preprocesamiento: {e}")
        return img


# Thread pool para operaciones asincronas
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="FaceAnalysis")


class Face(dict):
    def __init__(self, bbox=None, score=0.0, det_score=0.0, kps=None, embedding=None, normed_embedding=None, landmark_106=None, gender=None, age=None):
        super().__init__()
        self['bbox'] = bbox
        self['score'] = score
        self['det_score'] = det_score
        self['kps'] = kps
        self['embedding'] = embedding
        self['normed_embedding'] = normed_embedding
        self['landmark_106'] = landmark_106
        self['gender'] = gender
        self['age'] = age
        self.bbox = bbox
        self.score = score
        self.det_score = det_score
        self.kps = kps
        self.embedding = embedding
        self.normed_embedding = normed_embedding
        self.landmark_106 = landmark_106
        self.gender = gender
        self.age = age
        self.mask_offsets = (0, 0.1, 0, 0, 1, 20)


def _initialize_face_analyser_sync():
    """Funcion interna sincrona para inicializar el analizador"""
    try:
        print("[INFO] Inicializando FaceAnalysis optimizado...")

        import roop.globals
        from insightface.app import FaceAnalysis
        import onnxruntime as ort

        model_root = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "models", "buffalo_l"
        )

        if not os.path.exists(model_root):
            print(f"[ERROR] No se encontro el modelo en: {model_root}")
            return None

        available_providers = ort.get_available_providers()
        print(f"[INFO] Proveedores ONNX Runtime disponibles: {available_providers}")

        # USAR CUDA POR DEFECTO
        current_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print("[INFO] USANDO CUDA PARA FACE DETECTION")

        analyser_instance = None
        try:
            analyser_instance = FaceAnalysis(
                name="buffalo_l",
                root=os.path.dirname(os.path.dirname(__file__)),
                download=True,
                providers=current_providers,
            )

            # Preparar el analizador antes de verificar proveedores
            analyser_instance.prepare(
                ctx_id=0,
                det_size=DETECTION_SIZE,
                det_thresh=0.1
            )

            # Verificar y forzar CUDA en cada modelo
            if hasattr(analyser_instance, 'models'):
                print("\n[INFO] Verificando proveedores de cada modelo:")
                for model_name, model in analyser_instance.models.items():
                    if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                        providers_used = model.session.get_providers()
                        print(f"  - {model_name}: {providers_used}")

                        if 'CPUExecutionProvider' in providers_used and 'CUDAExecutionProvider' in available_providers:
                            print(f"    Recargando {model_name} con CUDA...")
                            try:
                                model_path = model.session._model_path if hasattr(model.session, '_model_path') else None
                                if not model_path and hasattr(model, 'model_path'):
                                    model_path = model.model_path

                                if model_path and os.path.exists(model_path):
                                    sess_options = ort.SessionOptions()
                                    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                                    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

                                    new_session = ort.InferenceSession(
                                        model_path,
                                        providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                                        provider_options=[{
                                            'device_id': 0,
                                            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                                            'arena_extend_strategy': 'kSameAsRequested',
                                            'cudnn_conv_algo_search': 'EXHAUSTIVE',
                                            'do_copy_in_default_stream': True,
                                        }, {}],
                                        sess_options=sess_options
                                    )
                                    model.session = new_session
                                    print(f"    OK {model_name} ahora usa: {new_session.get_providers()}")
                                else:
                                    print(f"    ERROR No se pudo encontrar ruta del modelo para {model_name}")
                            except Exception as e:
                                print(f"    ERROR No se pudo recargar {model_name} con CUDA: {e}")

            if hasattr(analyser_instance, 'models') and 'genderage' in analyser_instance.models:
                print("[INFO] Modelo genderage disponible en FaceAnalysis")
            else:
                print("[WARN] Modelo genderage NO disponible en FaceAnalysis")

            print("\n[SUCCESS] FaceAnalysis inicializado con CUDA")
            return analyser_instance

        except Exception as e_cuda:
            print(f"[ERROR] Error inicializando FaceAnalysis con CUDA: {e_cuda}")
            import traceback
            traceback.print_exc()
            return None

    except Exception as e_general:
        print(f"[ERROR] Error general inicializando FaceAnalysis: {e_general}")
        import traceback
        traceback.print_exc()
        return None


def get_face_analyser():
    """Funcion optimizada para obtener el analizador de caras"""
    global FACE_ANALYSER, FACE_ANALYSER_STATUS

    if FACE_ANALYSER is not None:
        return FACE_ANALYSER

    with BLOQUEO:
        if FACE_ANALYSER is not None:
            return FACE_ANALYSER

        if FACE_ANALYSER_STATUS == "initializing":
            print("FaceAnalysis ya esta inicializando, esperando...")
            start_time = time.time()
            while FACE_ANALYSER_STATUS == "initializing" and (time.time() - start_time) < INIT_TIMEOUT:
                time.sleep(0.5)

            if FACE_ANALYSER_STATUS == "ready":
                return FACE_ANALYSER
            elif FACE_ANALYSER_STATUS == "failed":
                return None
            else:
                print("Timeout esperando inicializacion de FaceAnalysis")
                FACE_ANALYSER_STATUS = "failed"
                return None

        try:
            FACE_ANALYSER_STATUS = "initializing"
            print("[INFO] Iniciando FaceAnalysis con timeout de 15 segundos...")

            future = EXECUTOR.submit(_initialize_face_analyser_sync)
            FACE_ANALYSER = future.result(timeout=INIT_TIMEOUT)

            if FACE_ANALYSER is not None:
                FACE_ANALYSER_STATUS = "ready"
                print("[INFO] FaceAnalysis listo")
            else:
                FACE_ANALYSER_STATUS = "failed"
                print("[ERROR] Fallo la inicializacion de FaceAnalysis")

            return FACE_ANALYSER

        except TimeoutError:
            print("[ERROR] TIMEOUT: FaceAnalysis tardo mas de 15 segundos")
            FACE_ANALYSER_STATUS = "failed"
            return None
        except Exception as e:
            print(f"[ERROR] Error en inicializacion de FaceAnalysis: {e}")
            FACE_ANALYSER_STATUS = "failed"
            return None


def get_cache_key(image_path: str, options: tuple, target_face_detection: bool) -> str:
    """Genera una clave unica para el cache"""
    return f"{image_path}_{options}_{target_face_detection}"


def extract_face_images(
    image_path,
    options: tuple = (False, 0),
    target_face_detection=False,
    is_source_face=False,
) -> List[Tuple[Face, np.ndarray]]:
    """
    Extrae caras de una imagen con deteccion optimizada.
    INCLUYE FALLBACK AUTOMATICO A CPU SI CUDA NO DETECTA CARAS.
    """
    global FACE_ANALYSER, FACE_ANALYSER_CPU, CACHE_RESULTADOS, CACHE_DETECCIONES, USE_CPU_FALLBACK

    is_video, frame_index = options
    image_id = f"{image_path}_{frame_index}" if is_video else str(image_path)

    # 1. Verificar cache de RESULTADOS COMPLETOS
    cache_key = get_cache_key(image_path if isinstance(image_path, str) else str(id(image_path)), options, target_face_detection)
    if cache_key in CACHE_RESULTADOS:
        return CACHE_RESULTADOS[cache_key]

    # Inicializar analizador CUDA si es necesario
    if FACE_ANALYSER is None:
        print("[INFO] Inicializando FACE_ANALYSER...")
        FACE_ANALYSER = get_face_analyser()
        if FACE_ANALYSER is None:
            logger.error("No se pudo inicializar el analizador de caras")
            return []

    # Inicializar analizador CPU si no existe (para fallback)
    if FACE_ANALYSER_CPU is None and USE_CPU_FALLBACK:
        FACE_ANALYSER_CPU = get_face_analyser_cpu()

    try:
        # Cargar la imagen
        is_video, frame_index = options
        if is_video:
            img = get_video_frame(image_path, frame_index)
        elif isinstance(image_path, str):
            if not os.path.exists(image_path):
                logger.error(f"Archivo no encontrado: {image_path}")
                return []
            img = cv2.imread(image_path)
        else:
            img = image_path.copy()

        if img is None:
            logger.error(f"No se pudo cargar la imagen: {image_path}")
            return []

        # Convertir a RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # ESTRATEGIA OPTIMIZADA PARA CARAS DE ORIGEN
        if is_source_face:
            img_to_detect = img_rgb

            faces = None
            if image_id in CACHE_DETECCIONES:
                faces = CACHE_DETECCIONES[image_id]

            if faces is None:
                # Detección directa con CUDA
                try:
                    faces = FACE_ANALYSER.get(img_to_detect)
                    if len(CACHE_DETECCIONES) < MAX_CACHE_SIZE:
                        CACHE_DETECCIONES[image_id] = faces

                    # FALLBACK A CPU SI CUDA NO DETECTA NADA
                    if not faces and FACE_ANALYSER_CPU is not None:
                        print("[WARN] CUDA no detecto caras, probando con CPU...")
                        faces = FACE_ANALYSER_CPU.get(img_to_detect)
                        if faces:
                            USE_CPU_FALLBACK = True
                            print(f"[OK] CPU detecto {len(faces)} caras")

                except Exception as e:
                    logger.warning(f"Error en deteccion de origen: {e}")
                    faces = None
        else:
            # DESTINO
            img_preprocessed = preprocess_image_for_detection(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_to_detect = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB)

            faces = None
            if image_id in CACHE_DETECCIONES:
                faces = CACHE_DETECCIONES[image_id]

            if faces is None:
                # Intento 1: Deteccion con preprocesamiento
                try:
                    faces = FACE_ANALYSER.get(img_to_detect)
                    if len(CACHE_DETECCIONES) < MAX_CACHE_SIZE:
                        CACHE_DETECCIONES[image_id] = faces

                    # FALLBACK A CPU SI NO DETECTA NADA
                    if not faces and FACE_ANALYSER_CPU is not None:
                        print("[WARN] CUDA no detecto caras, probando con CPU...")
                        faces = FACE_ANALYSER_CPU.get(img_to_detect)
                        if faces:
                            USE_CPU_FALLBACK = True
                            print(f"[OK] CPU detecto {len(faces)} caras")

                except Exception as e:
                    logger.warning(f"Error en deteccion de destino: {e}")
                    faces = None

            # Intento 2: Re-escalado si no hay caras
            if not faces:
                try:
                    h, w = img_to_detect.shape[:2]
                    small_img = cv2.resize(img_to_detect, (w//2, h//2))
                    faces_small = FACE_ANALYSER.get(small_img)

                    if faces_small:
                        scale = 2.0
                        for face in faces_small:
                            face.bbox = face.bbox * scale
                            if face.kps is not None:
                                face.kps = face.kps * scale
                        faces = faces_small
                        logger.info("Cara destino detectada con re-escalado (0.5x)")
                        if len(CACHE_DETECCIONES) < MAX_CACHE_SIZE:
                            CACHE_DETECCIONES[image_id] = faces

                    # FALLBACK A CPU SI RE-ESCALADO TAMPOCO FUNCIONA
                    if not faces and FACE_ANALYSER_CPU is not None:
                        print("[WARN] Re-escalado no funciono, probando CPU...")
                        faces_small_cpu = FACE_ANALYSER_CPU.get(small_img)
                        if faces_small_cpu:
                            scale = 2.0
                            for face in faces_small_cpu:
                                face.bbox = face.bbox * scale
                                if face.kps is not None:
                                    face.kps = face.kps * scale
                            faces = faces_small_cpu
                            USE_CPU_FALLBACK = True
                            print(f"[OK] CPU detecto {len(faces)} caras con re-escalado")

                except Exception as e:
                    logger.warning(f"Error en deteccion con re-escalado: {e}")
                    faces = None

        if not faces:
            logger.warning(f"No se detectaron caras en {image_path}")
            CACHE_RESULTADOS[cache_key] = []
            return []

        logger.info(f"Se detectaron {len(faces)} caras")

        # Procesar caras con filtros de calidad
        results = []
        for face_idx, face_data in enumerate(faces):
            try:
                # FILTRO DE CALIDAD: Verificar score mínimo
                if hasattr(face_data, 'det_score') and face_data.det_score < 0.25:
                    continue
                
                # FILTRO DE CALIDAD: Verificar que el bbox sea razonable
                x1, y1, x2, y2 = face_data.bbox.astype(int)
                face_w = x2 - x1
                face_h = y2 - y1
                
                # Cara muy pequeña
                if face_w < 20 or face_h < 20:
                    continue
                    
                # Aspect ratio muy extremo (no es una cara humana)
                aspect_ratio = face_w / face_h if face_h > 0 else 0
                if aspect_ratio < 0.3 or aspect_ratio > 2.0:
                    continue
                
                # FILTRO DE CALIDAD: Verificar que tenga embedding válido
                if not hasattr(face_data, 'embedding') or face_data.embedding is None:
                    continue
                
                # FILTRO DE CALIDAD: Verificar que tenga keypoints válidos
                if not hasattr(face_data, 'kps') or face_data.kps is None or len(face_data.kps) < 5:
                    continue
                
                # Calcular normed_embedding
                normed_embedding = None
                if face_data.embedding is not None:
                    embedding_np = np.array(face_data.embedding)
                    norm = np.linalg.norm(embedding_np)
                    if norm > 0:
                        normed_embedding = embedding_np / norm
                
                # Extraer landmark_106 si está disponible
                landmark_106 = None
                if hasattr(face_data, 'landmark_2d_106') and face_data.landmark_2d_106 is not None:
                    landmark_106 = face_data.landmark_2d_106.tolist()
                
                face_obj = Face(
                    bbox=face_data.bbox.astype(int).tolist(),
                    score=face_data.score,
                    det_score=face_data.det_score,
                    kps=face_data.kps.tolist() if face_data.kps is not None else None,
                    embedding=face_data.embedding.tolist() if face_data.embedding is not None else None,
                    normed_embedding=normed_embedding,
                    landmark_106=landmark_106
                )

                padding_x = face_w * face_detection_padding
                padding_y = face_h * face_detection_padding

                x1_pad = max(0, int(x1 - padding_x))
                y1_pad = max(0, int(y1 - padding_y))
                x2_pad = min(img_rgb.shape[1], int(x2 + padding_x))
                y2_pad = min(img_rgb.shape[0], int(y2 + padding_y))

                face_img = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad].copy()

                if face_img is None or face_img.size == 0:
                    continue

                if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                    continue

                if len(face_img.shape) == 2:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                elif len(face_img.shape) == 3 and face_img.shape[2] == 4:
                    face_img = cv2.cvtColor(face_img, cv2.COLOR_RGBA2RGB)

                if face_img.shape[-1] != 3:
                    continue

                results.append([face_obj, face_img])

            except Exception as e:
                logger.error(f"Error procesando cara {face_idx + 1}: {e}")
                continue

        # Ordenar por tamano si no es target_face_detection
        if not target_face_detection and len(results) > 0:
            results.sort(
                key=lambda x: (x[0].bbox[2] - x[0].bbox[0]) * (x[0].bbox[3] - x[0].bbox[1]),
                reverse=True,
            )
            results = [results[0]]

        CACHE_RESULTADOS[cache_key] = results
        return results

    except Exception as e:
        logger.error(f"Error general en extract_face_images: {e}")
        CACHE_RESULTADOS[cache_key] = []
        return []


def get_first_face(frame: np.ndarray) -> Optional[Face]:
    """Obtiene el primer rostro detectado"""
    try:
        analizador = get_face_analyser()
        if analizador is None:
            return None

        faces = analizador.get(frame)
        if not faces:
            return None

        face_data = faces[0]
        
         # Calcular normed_embedding
        normed_embedding = None
        if face_data.embedding is not None:
            embedding_np = np.array(face_data.embedding)
            norm = np.linalg.norm(embedding_np)
            if norm > 0:
                normed_embedding = embedding_np / norm
        
        # Extraer landmark_106 si está disponible
        landmark_106 = None
        if hasattr(face_data, 'landmark_2d_106') and face_data.landmark_2d_106 is not None:
            landmark_106 = face_data.landmark_2d_106.tolist()
        
        return Face(
            bbox=face_data.bbox.astype(int).tolist(),
            score=face_data.score,
            det_score=face_data.score,
            kps=face_data.kps.tolist() if face_data.kps is not None else None,
            embedding=face_data.embedding.tolist() if face_data.embedding is not None else None,
            normed_embedding=normed_embedding,
            landmark_106=landmark_106
        )
    except Exception as e:
        print(f"[ERROR] Error obteniendo primer rostro: {e}")
        return None


def get_all_faces(frame: np.ndarray) -> List[Face]:
    """Obtiene todos los rostros detectados"""
    try:
        analizador = get_face_analyser()
        if analizador is None:
            print("[ERROR] Analizador no inicializado")
            return []

        print(f"[DEBUG] Frame shape: {frame.shape}")
        faces = analizador.get(frame)
        print(f"[DEBUG] Faces count: {len(faces)}")
        
        if not faces:
            print("[DEBUG] No faces detected")
            return []

        results = []
        for face_data in faces:
            print(f"[DEBUG] Face data: {face_data}")
            print(f"[DEBUG] BBox: {face_data.bbox}")
            print(f"[DEBUG] Det score: {face_data.det_score}")
            
             # Calcular normed_embedding
            normed_embedding = None
            if face_data.embedding is not None:
                embedding_np = np.array(face_data.embedding)
                norm = np.linalg.norm(embedding_np)
                if norm > 0:
                    normed_embedding = embedding_np / norm
            
            # Extraer landmark_106 si está disponible
            landmark_106 = None
            if hasattr(face_data, 'landmark_2d_106') and face_data.landmark_2d_106 is not None:
                landmark_106 = face_data.landmark_2d_106.tolist()
            
            results.append(
                Face(
                    bbox=face_data.bbox.astype(int).tolist(),
                    score=face_data.det_score,
                    det_score=face_data.det_score,
                    kps=face_data.kps.tolist() if face_data.kps is not None else None,
                    embedding=face_data.embedding.tolist() if face_data.embedding is not None else None,
                    normed_embedding=normed_embedding,
                    landmark_106=landmark_106
                )
            )

        return results

    except Exception as e:
        print(f"[ERROR] Error obteniendo todos los rostros: {type(e).__name__}: {e}")
        import traceback
        print(traceback.format_exc())
        return []


def detectar_orientacion_cara(face) -> str:
    """Detecta la orientacion de una cara"""
    try:
        if not hasattr(face, "kps") or face.kps is None:
            return "normal"

        kps = face.kps
        if len(kps) < 5:
            return "normal"

        left_eye = kps[0] if len(kps) > 0 else None
        right_eye = kps[1] if len(kps) > 1 else None
        nose = kps[2] if len(kps) > 2 else None
        left_mouth = kps[3] if len(kps) > 3 else None
        right_mouth = kps[4] if len(kps) > 4 else None

        if not all([left_eye, right_eye, nose, left_mouth, right_mouth]):
            return "normal"

        for point in [left_eye, right_eye, nose, left_mouth, right_mouth]:
            if len(point) < 2 or not all(isinstance(coord, (int, float)) for coord in point):
                return "normal"

        left_eye_x, left_eye_y = left_eye[0], left_eye[1]
        right_eye_x, right_eye_y = right_eye[0], right_eye[1]
        nose_x, nose_y = nose[0], nose[1]
        left_mouth_x, left_mouth_y = left_mouth[0], left_mouth[1]
        right_mouth_x, right_mouth_y = right_mouth[0], right_mouth[1]

        eye_center_x = (left_eye_x + right_eye_x) / 2
        eye_center_y = (left_eye_y + right_eye_y) / 2
        mouth_center_x = (left_mouth_x + right_mouth_x) / 2
        mouth_center_y = (left_mouth_y + right_mouth_y) / 2

        if mouth_center_y < eye_center_y - 15:
            return "volteada"

        vertical_distance = abs(eye_center_y - mouth_center_y)
        horizontal_distance = abs(eye_center_x - mouth_center_x)

        if horizontal_distance > vertical_distance * 1.8:
            eye_angle = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / np.pi

            if abs(eye_angle) > 5:
                return "rotada_90_derecha" if eye_angle > 0 else "rotada_90_izquierda"
            else:
                if right_eye_y > left_eye_y + 5:
                    return "rotada_90_derecha"
                elif left_eye_y > right_eye_y + 5:
                    return "rotada_90_izquierda"

        return "normal"

    except Exception as e:
        return "normal"


def detectar_rotacion_90_grados(face) -> int:
    """Detecta si una cara esta rotada"""
    try:
        if not hasattr(face, "kps") or face.kps is None:
            return 0

        kps = face.kps
        if len(kps) < 5:
            return 0

        left_eye = kps[0] if len(kps) > 0 else None
        right_eye = kps[1] if len(kps) > 1 else None
        nose = kps[2] if len(kps) > 2 else None
        left_mouth = kps[3] if len(kps) > 3 else None
        right_mouth = kps[4] if len(kps) > 4 else None

        if not all([left_eye, right_eye, nose, left_mouth, right_mouth]):
            return 0

        left_eye_x, left_eye_y = left_eye[0], left_eye[1]
        right_eye_x, right_eye_y = right_eye[0], right_eye[1]
        mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
        eye_center_y = (left_eye_y + right_eye_y) / 2

        if mouth_center_y < eye_center_y - 15:
            return 180

        vertical_distance = abs(eye_center_y - mouth_center_y)
        horizontal_distance = abs(left_eye_x - right_eye_x)

        if horizontal_distance > vertical_distance * 1.8:
            eye_angle = np.arctan2(right_eye_y - left_eye_y, right_eye_x - left_eye_x) * 180 / np.pi

            if abs(eye_angle) > 5:
                return 90 if eye_angle > 0 else 270
            else:
                if right_eye_y > left_eye_y + 5:
                    return 90
                elif left_eye_y > right_eye_y + 5:
                    return 270

        return 0

    except Exception as e:
        return 0


def clamp_cut_values(bbox: List[int], width: int, height: int, padding: int = 0) -> List[int]:
    """Asegura que los valores del bbox esten dentro de los limites"""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(width, x2 + padding)
    y2 = min(height, y2 + padding)
    x2 = max(x1 + 1, x2)
    y2 = max(y1 + 1, y2)
    return [int(x1), int(y1), int(x2), int(y2)]


def rotate_image_180(image: np.ndarray) -> np.ndarray:
    """Rota una imagen 180 grados"""
    return cv2.rotate(image, cv2.ROTATE_180)


def rotate_anticlockwise(image: np.ndarray) -> np.ndarray:
    """Rota una imagen 90 grados en sentido antihorario"""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def rotate_clockwise(image: np.ndarray) -> np.ndarray:
    """Rota una imagen 90 grados en sentido horario"""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def align_crop(image: np.ndarray, face: Face, size: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Alinea y recorta una imagen de cara"""
    if image is None or not isinstance(image, np.ndarray):
        if size > 0:
            return np.zeros((size, size, 3), dtype=np.uint8), np.eye(3)[:2].astype(np.float32)
        return image, np.eye(3)[:2].astype(np.float32)

    if face is None:
        if size > 0:
            return np.zeros((size, size, 3), dtype=np.uint8), np.eye(3)[:2].astype(np.float32)
        return image, np.eye(3)[:2].astype(np.float32)

    if not hasattr(image, "shape") or len(image.shape) < 2 or image.size == 0:
        if size > 0:
            return np.zeros((size, size, 3), dtype=np.uint8), np.eye(3)[:2].astype(np.float32)
        return image, np.eye(3)[:2].astype(np.float32)

    try:
        if not hasattr(face, "bbox") or face.bbox is None:
            if size > 0:
                resized = cv2.resize(image, (size, size))
                M = np.array([[size / image.shape[1], 0, 0], [0, size / image.shape[0], 0]], dtype=np.float32)
                return resized, M
            return image, np.eye(3)[:2].astype(np.float32)

        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        x1, y1, x2, y2 = map(int, face.bbox[:4])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            if size > 0:
                return np.zeros((size, size, 3), dtype=np.uint8), np.eye(3)[:2].astype(np.float32)
            return image, np.eye(3)[:2].astype(np.float32)

        cropped = image[y1:y2, x1:x2]

        if size > 0 and cropped.size > 0:
            cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_CUBIC)

        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)[:2].astype(np.float32)
        return cropped, M

    except Exception as e:
        print(f"[ERROR] align_crop: {e}")
        if size > 0:
            return np.zeros((size, size, 3), dtype=np.uint8), np.eye(3)[:2].astype(np.float32)
        return image, np.eye(3)[:2].astype(np.float32)


# Variables adicionales para compatibilidad
FACE_ANALYSER = None
MAX_DETECTION_ATTEMPTS = 3

# Alias para compatibilidad con UI
extract_face_images_fast = extract_face_images
extract_faces_fast = extract_face_images

# Exportar funciones principales
__all__ = [
    "get_face_analyser",
    "extract_face_images",
    "Face",
    "get_first_face",
    "get_all_faces",
    "detectar_orientacion_cara",
    "detectar_rotacion_90_grados",
    "clamp_cut_values",
    "rotate_image_180",
    "rotate_anticlockwise",
    "rotate_clockwise",
    "align_crop",
    "extract_face_images_fast",
    "extract_faces_fast",
]
