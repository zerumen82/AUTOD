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

# Configuracion optimizada para MAXIMA DETECCION de caras
DETECTION_SIZE = (1024, 1024)  # Aumentado de 640 a 1024 para mejor calidad
MAX_DETECTION_ATTEMPTS = 3
INIT_TIMEOUT = 60
# Dynamic padding: less for close-ups (they're already zoomed), more for wide shots
face_detection_padding = 0.15  # Tighter crop for better UI recognition

# Caches optimizados para velocidad
CACHE_RESULTS = {}
CACHE_DETECTIONS = {}
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
    """Preprocesa una imagen de forma suave para mejorar la deteccion sin generar ruido"""
    try:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
        l_enhanced = clahe.apply(l)
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        img_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        # Sharpening sutil
        blurred = cv2.GaussianBlur(img_enhanced, (3, 3), 0)
        sharpened = cv2.addWeighted(img_enhanced, 1.3, blurred, -0.3, 0)
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
        print("[INFO] Inicializando FaceAnalysis...")

        import roop.globals
        from insightface.app import FaceAnalysis

        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        try:
            import onnxruntime as ort
            use_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
        except:
            use_cuda = False
        ctx_id = 0 if use_cuda else -1
        
        print(f"[INFO] Root de modelos: {models_dir}")

        analyser_instance = None
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
            
            analyser_instance = FaceAnalysis(
                name="buffalo_l",
                root=models_dir,
                download=True,
                providers=providers
            )
            
            # Prepare con tamaño 1024x1024 para máxima calidad y detección de perfiles
            analyser_instance.prepare(
                ctx_id=ctx_id,
                det_size=(1024, 1024),
                det_thresh=0.1 # Muy permisivo para no perder ninguna cara
            )
            
            print(f"[SUCCESS] FaceAnalysis listo (1024x1024, thresh=0.1)")
            return analyser_instance

        except Exception as e:
            print(f"[ERROR] Error inicializando FaceAnalysis: {e}")
            return None

    except Exception as e_general:
        print(f"[ERROR] Error general inicializando FaceAnalysis: {e_general}")
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
            start_time = time.time()
            while FACE_ANALYSER_STATUS == "initializing" and (time.time() - start_time) < INIT_TIMEOUT:
                time.sleep(0.5)

            if FACE_ANALYSER_STATUS == "ready":
                return FACE_ANALYSER
            else:
                return None

        try:
            FACE_ANALYSER_STATUS = "initializing"
            print("[INFO] Iniciando FaceAnalysis...")

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
            print("[ERROR] TIMEOUT: FaceAnalysis tardo mas de 60 segundos")
            FACE_ANALYSER_STATUS = "failed"
            return None
        except Exception as e:
            print(f"[ERROR] Error en inicializacion de FaceAnalysis: {e}")
            FACE_ANALYSER_STATUS = "failed"
            return None


def get_cache_key(image_path: str, options: tuple, target_face_detection: bool) -> str:
    """Genera una clave unica para el cache"""
    return f"{image_path}_{options}_{target_face_detection}"


def detect_faces_robust(img_rgb, analyser, thresh_override=None):
    """Detecta caras usando el analizador ya inicializado (NO llamar prepare())"""
    try:
        # El analizador ya está inicializado con det_thresh=0.1 desde get_face_analyser()
        faces = analyser.get(img_rgb)
        if faces:
            # Filtrar por umbral si se especificó uno
            if thresh_override is not None:
                faces = [f for f in faces if hasattr(f, 'det_score') and f.det_score >= thresh_override]
            return faces if faces else []
        
        return []
    except Exception as e:
        print(f"[ERROR] detect_faces_robust: {e}")
        return []


def extract_face_images(
    image_path,
    options: tuple = (False, 0),
    target_face_detection=False,
    is_source_face=False,
    ui_padding=None,
) -> List[Tuple[Face, np.ndarray]]:
    """
    Extrae caras de una imagen con deteccion optimizada.
    INCLUYE FALLBACK AUTOMATICO A CPU SI CUDA NO DETECTA CARAS.
    PARA SOURCE: Usa toda la imagen directamente (el usuario proporciono la foto!)
    """
    global FACE_ANALYSER, FACE_ANALYSER_CPU, CACHE_RESULTS, CACHE_DETECTIONS, USE_CPU_FALLBACK
    
    is_video, frame_index = options
    image_id = f"{image_path}_{frame_index}" if is_video else str(image_path)
    
    # 1. Verificar cache de RESULTADOS COMPLETOS
    cache_key = get_cache_key(image_path if isinstance(image_path, str) else str(id(image_path)), options, target_face_detection)
    if cache_key in CACHE_RESULTS:
        return CACHE_RESULTS[cache_key]
    
    # Inicializar analizador CUDA si es necesario
    if FACE_ANALYSER is None:
        FACE_ANALYSER = get_face_analyser()
        if FACE_ANALYSER is None:
            return []
    
    # Inicializar analizador CPU si no existe (para fallback)
    if FACE_ANALYSER_CPU is None and USE_CPU_FALLBACK:
        FACE_ANALYSER_CPU = get_face_analyser_cpu()
    
    try:
        # Cargar la imagen
        if is_video:
            img = get_video_frame(image_path, frame_index)
        elif isinstance(image_path, str):
            if not os.path.exists(image_path):
                return []
            img = get_image_frame(image_path)
        else:
            img = image_path.copy()
        
        if img is None:
            return []
        
        # Convertir a RGB
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ============ SOURCE FACES: USAR TODA LA IMAGEN ============
        if is_source_face:
            h, w = img_rgb.shape[:2]
            print(f"[SOURCE_DETECT] Cargando cara de origen: {os.path.basename(str(image_path))} ({w}x{h})")
            
            # 1. Intentar detección normal (Permisiva)
            faces = detect_faces_robust(img_rgb, FACE_ANALYSER, 0.15)
            
            # 2. Si falla, intentar con PADDING (Ayuda con crops muy cerrados)
            if not faces:
                print(f"[SOURCE_DETECT] No se detectó cara directamente. Intentando con padding...")
                pad_h, pad_w = h // 2, w // 2
                img_padded = cv2.copyMakeBorder(img_rgb, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                faces_padded = detect_faces_robust(img_padded, FACE_ANALYSER, 0.10)
                if faces_padded:
                    faces = faces_padded
                    for f in faces:
                        # Ajustar coordenadas al original
                        f.bbox[0] -= pad_w
                        f.bbox[2] -= pad_w
                        f.bbox[1] -= pad_h
                        f.bbox[3] -= pad_h
                        if hasattr(f, 'kps') and f.kps is not None:
                            f.kps[:, 0] -= pad_w
                            f.kps[:, 1] -= pad_h
                    print(f"[SOURCE_DETECT] ¡Cara detectada exitosamente usando padding!")

            # 3. Si sigue fallando, intentar con RE-SCALADO
            if not faces and (w < 200 or h < 200):
                print(f"[SOURCE_DETECT] Imagen pequeña, intentando upscale para detección...")
                img_up = cv2.resize(img_rgb, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
                faces_up = detect_faces_robust(img_up, FACE_ANALYSER, 0.10)
                if faces_up:
                    faces = faces_up
                    for f in faces:
                        f.bbox = (f.bbox / 2.0).astype(int)
                        if hasattr(f, 'kps') and f.kps is not None:
                            f.kps = f.kps / 2.0
                    print(f"[SOURCE_DETECT] ¡Cara detectada exitosamente usando upscale!")

            if faces:
                # Ordenar por tamaño y coger la más grande
                faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
                face_data_full = faces[0]
                
                # Asegurar que el embedding NO sea None
                face_embedding = getattr(face_data_full, 'embedding', None)
                if face_embedding is None:
                    print(f"[SOURCE_DETECT] ⚠️ Cara detectada pero SIN EMBEDDING. Intentando recuperar...")
                    # Extraer el crop de la cara
                    x1, y1, x2, y2 = face_data_full.bbox.astype(int)
                    face_crop = img_rgb[y1:y2, x1:x2].copy()
                    
                    if face_crop.size > 0:
                        # Si el bbox es toda la imagen (crop ya recortado), añadir padding para dar contexto
                        h_img, w_img = img_rgb.shape[:2]
                        if x1 == 0 and y1 == 0 and (x2 - x1) >= w_img * 0.8 and (y2 - y1) >= h_img * 0.8:
                            print(f"[SOURCE_DETECT] Crop ocupa casi toda la imagen, añadiendo padding...")
                            pad = min(h_img, w_img) // 3
                            face_padded = cv2.copyMakeBorder(face_crop, pad, pad, pad, pad, 
                                                              cv2.BORDER_CONSTANT, value=[128, 128, 128])
                            face_crop = face_padded
                        else:
                            pad = 0
                        
                        # Re-analizar el crop
                        try:
                            faces_crop = analyser.get(face_crop)
                            if faces_crop and getattr(faces_crop[0], 'embedding', None) is not None:
                                face_data_full = faces_crop[0]
                                # Ajustar bbox si se añadió padding
                                if pad > 0:
                                    face_data_full.bbox[0] -= pad
                                    face_data_full.bbox[2] -= pad
                                    face_data_full.bbox[1] -= pad
                                    face_data_full.bbox[3] -= pad
                                print(f"[SOURCE_DETECT] ✅ Embedding recuperado (crop shape: {face_crop.shape})")
                            else:
                                # Intentar con CPU si está disponible
                                if FACE_ANALYSER_CPU is not None:
                                    print(f"[SOURCE_DETECT] Intentando con CPU fallback...")
                                    faces_crop_cpu = FACE_ANALYSER_CPU.get(face_crop)
                                    if faces_crop_cpu and getattr(faces_crop_cpu[0], 'embedding', None) is not None:
                                        face_data_full = faces_crop_cpu[0]
                                        if pad > 0:
                                            face_data_full.bbox[0] -= pad
                                            face_data_full.bbox[2] -= pad
                                            face_data_full.bbox[1] -= pad
                                            face_data_full.bbox[3] -= pad
                                        print(f"[SOURCE_DETECT] ✅ Embedding recuperado con CPU")
                                    else:
                                        print(f"[SOURCE_DETECT] ❌ CPU tampoco pudo generar embedding")
                                else:
                                    print(f"[SOURCE_DETECT] ❌ No se pudo obtener embedding del crop")
                        except Exception as e:
                            print(f"[SOURCE_DETECT] ❌ Error re-analizando crop: {e}")
                    else:
                        print(f"[SOURCE_DETECT] ❌ Crop vacío, no se puede re-analizar")
                
                normed_emb = None
                if hasattr(face_data_full, 'embedding') and face_data_full.embedding is not None:
                    emb = np.array(face_data_full.embedding)
                    n = np.linalg.norm(emb)
                    if n > 0: normed_emb = emb / n

                face_obj = Face(
                    bbox=face_data_full.bbox.astype(int).tolist(),
                    score=face_data_full.score,
                    det_score=face_data_full.det_score,
                    kps=face_data_full.kps.tolist() if face_data_full.kps is not None else None,
                    embedding=face_data_full.embedding,
                    normed_embedding=normed_emb
                )
                res = [[face_obj, img_rgb.copy()]]
                CACHE_RESULTS[cache_key] = res
                return res
            else:
                print(f"[SOURCE_DETECT] ❌ FALLO CRÍTICO: No se detectó ninguna cara en la imagen de origen.")
                # No podemos hacer swap sin embedding, así que mejor no devolver nada o devolver un objeto nulo
                # que al menos sea procesado como "sin cara" en lugar de causar errores silenciosos.
                return []

        # ============ TARGET FACES: DETECCIÓN NORMAL ============
        target_thresh = 0.25 # Permisivo para encontrar caras giradas en video
        
        # Preprocesamiento opcional
        img_preprocessed = preprocess_image_for_detection(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        img_to_detect = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB)
        
        faces = detect_faces_robust(img_to_detect, FACE_ANALYSER, target_thresh)
        if not faces and FACE_ANALYSER_CPU is not None:
            faces = detect_faces_robust(img_to_detect, FACE_ANALYSER_CPU, target_thresh)
            
        if not faces:
            CACHE_RESULTS[cache_key] = []
            return []
        
        # Procesar caras con filtros de calidad
        results = []
        for face_idx, face_data in enumerate(faces):
            try:
                # 1. Filtro de confianza base
                if hasattr(face_data, 'det_score') and face_data.det_score < target_thresh:
                    continue
                
                # 2. Dimensiones mínimas
                x1, y1, x2, y2 = face_data.bbox.astype(int)
                face_w = x2 - x1
                face_h = y2 - y1
                min_size = 20
                if face_w < min_size or face_h < min_size:
                    continue
                
                img_h, img_w = img_rgb.shape[:2]
                if x2 <= 0 or y2 <= 0 or x1 >= img_w or y1 >= img_h:
                    continue
                
                # Calcular normed_embedding
                normed_embedding = None
                if hasattr(face_data, 'embedding') and face_data.embedding is not None:
                    embedding_np = np.array(face_data.embedding)
                    norm = np.linalg.norm(embedding_np)
                    if norm > 0:
                        normed_embedding = embedding_np / norm
                
                # landmark_106 si esta disponible
                landmark_106 = None                     if hasattr(face_data, 'landmark_2d_106') and face_data.landmark_2d_106 is not None                     else None
                
                face_obj = Face(
                    bbox=face_data.bbox.astype(int).tolist(),
                    score=face_data.score,
                    det_score=face_data.det_score,
                    kps=face_data.kps.tolist() if face_data.kps is not None else None,
                    embedding=face_data.embedding if face_data.embedding is not None else None,
                    normed_embedding=normed_embedding,
                    landmark_106=landmark_106.tolist() if landmark_106 is not None else None
                )
                
                # Padding dinamico
                current_padding = ui_padding if ui_padding is not None else face_detection_padding
                padding_x = face_w * current_padding
                padding_y = face_h * current_padding
                
                x1_pad = max(0, int(x1 - padding_x))
                y1_pad = max(0, int(y1 - padding_y))
                x2_pad = min(img_rgb.shape[1], int(x2 + padding_x))
                y2_pad = min(img_rgb.shape[0], int(y2 + padding_y))
                
                face_img = img_rgb[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                
                if face_img is None or face_img.size == 0:
                    continue
                
                results.append([face_obj, face_img])
                
            except Exception as e:
                continue
        
        if len(results) > 0:
            results.sort(key=lambda x: (x[0].bbox[2] - x[0].bbox[0]) * (x[0].bbox[3] - x[0].bbox[1]), reverse=True)
        
        CACHE_RESULTS[cache_key] = results
        return results
    
    except Exception as e:
        CACHE_RESULTS[cache_key] = []
        return []
