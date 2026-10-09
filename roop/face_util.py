"""Version CORREGIDA de face_util.py para preservar colores en la extraccion de caras

"""

# AGREGAR CARPETA DLL AL PATH AL INICIO - CRITICO PARA CUDA
import os
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Tuple
import warnings

dll_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "dll"))
os.environ["PATH"] = dll_path + ";" + os.environ.get("PATH", "")

import cv2  # noqa: E402 -- native DLL setup above
import numpy as np  # noqa: E402 -- native DLL setup above

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
from roop.capturer import get_image_frame, get_video_frame  # noqa: E402 -- native DLL setup above
from roop.faceswap_config import get_faceswap_config  # noqa: E402 -- native DLL setup above

# Variable global simple (sin inicializacion automatica)
FACE_ANALYSER = None
FACE_ANALYSER_CPU = None  # Fallback a CPU
BLOQUEO = threading.Lock()
FACE_ANALYSER_STATUS = "not_initialized"
USE_CPU_FALLBACK = False  # Solo CUDA

# Configuracion optimizada para MAXIMA DETECCION de caras
DETECTION_SIZE = (1024, 1024)  # Aumentado de 640 a 1024 para mejor calidad
MAX_DETECTION_ATTEMPTS = 3
INIT_TIMEOUT = 60
# Dynamic padding: less for close-ups (they're already zoomed), more for wide shots
face_detection_padding = 0.3  # 30% de margen para caras completas

# Caches optimizados para velocidad
CACHE_RESULTS = {}
CACHE_DETECTIONS = {}
MAX_FACES_TO_DETECT = 100
MAX_CACHE_SIZE = 100


def _cache_put(cache_dict: dict, key: str, value):
    """Inserta en cache con eviction LRU si supera MAX_CACHE_SIZE."""
    if len(cache_dict) >= MAX_CACHE_SIZE:
        # Eliminar la entrada más antigua (primera key insertada)
        oldest_key = next(iter(cache_dict))
        del cache_dict[oldest_key]
    cache_dict[key] = value


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
        analyser_instance.prepare(
            ctx_id=-1,
            det_size=DETECTION_SIZE,
            det_thresh=0.1,
        )

        print("[SUCCESS] FaceAnalysis CPU inicializado (thresh=0.1)")
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
        lightness, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) 
        l_enhanced = clahe.apply(lightness)
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
    def __init__(self, bbox=None, score=0.0, det_score=0.0, kps=None, embedding=None, normed_embedding=None, landmark_106=None, gender=None, age=None, face_img=None):
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
        self['face_img'] = face_img
        self.bbox = bbox
        self.score = score
        self.det_score = det_score
        self.kps = kps
        self.embedding = embedding
        self.normed_embedding = normed_embedding
        self.landmark_106 = landmark_106
        self.gender = gender
        self.age = age
        self.face_img = face_img
        self.mask_offsets = (0, 0.1, 0, 0, 1, 20)


def _initialize_face_analyser_sync():
    """Funcion interna sincrona para inicializar el analizador"""
    try:
        print("[INFO] Inicializando FaceAnalysis...")

        from insightface.app import FaceAnalysis

        models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        
        try:
            import onnxruntime as ort
            use_cuda = 'CUDAExecutionProvider' in ort.get_available_providers()
        except BaseException:
            use_cuda = False
        ctx_id = 0 if use_cuda else -1
        
        print(f"[INFO] Root de modelos: {models_dir}")

        analyser_instance = None
        try:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_cuda else ['CPUExecutionProvider']
            # EXHAUSTIVE benchmarks every cuDNN kernel again each time the
            # editor releases this session to free the 8 GB card. HEURISTIC
            # picks a kernel without that search. The weights stay the same.
            provider_options = (
                [{"cudnn_conv_algo_search": "HEURISTIC"}, {}]
                if use_cuda else None
            )

            analyser_instance = FaceAnalysis(
                name="buffalo_l",
                root=models_dir,
                download=True,
                providers=providers,
                provider_options=provider_options,
            )
            
            # Prepare con tamaño 1024x1024 para máxima calidad y detección de perfiles
            analyser_instance.prepare(
                ctx_id=ctx_id,
                det_size=(1024, 1024),
                det_thresh=0.1 # Muy permisivo para no perder ninguna cara
            )
            
            print("[SUCCESS] FaceAnalysis listo (1024x1024, thresh=0.1)")
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


def release_face_analyser():
    """Drop the resident CUDA FaceAnalysis session.

    The editor warmup keeps buffalo_l on the GPU. On an 8 GB card that
    session is still allocated when Qwen or LongCat loads, so the edit
    swaps weights before the first sampling step.
    """
    global FACE_ANALYSER, FACE_ANALYSER_STATUS
    with BLOQUEO:
        analyser = FACE_ANALYSER
        FACE_ANALYSER = None
        if FACE_ANALYSER_STATUS == "ready":
            FACE_ANALYSER_STATUS = "not_initialized"
    if analyser is None:
        return
    try:
        models = getattr(analyser, "models", None)
        if isinstance(models, dict):
            for model in list(models.values()):
                session = getattr(model, "session", None)
                if session is not None:
                    try:
                        del model.session
                    except Exception:
                        pass
            models.clear()
    except Exception:
        pass
    del analyser
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    print("[FACE] FaceAnalysis CUDA liberado", flush=True)


def get_cache_key(
    image_path: str,
    options: tuple,
    target_face_detection: bool,
    is_source_face: bool = False,
) -> str:
    """Genera una clave unica para el cache (source vs target no se mezclan)."""
    return f"{image_path}_{options}_{target_face_detection}_src={bool(is_source_face)}"


def detect_faces_robust(img_rgb, analyser, thresh_override=None):
    """Detecta caras usando el analizador ya inicializado (NO llamar prepare())"""
    try:
        if img_rgb is None or getattr(img_rgb, "size", 0) == 0:
            return []
        # InsightFace det suele esperar BGR; probar BGR y luego RGB
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        faces = None
        for img_try in (img_bgr, img_rgb):
            try:
                faces = analyser.get(img_try)
            except Exception:
                faces = None
            if faces:
                break
        if faces:
            if thresh_override is not None:
                thr = float(thresh_override)
                faces = [
                    f for f in faces
                    if float(getattr(f, "det_score", getattr(f, "score", 0.0)) or 0.0) >= thr
                ]
            return list(faces) if faces else []
        return []
    except Exception as e:
        print(f"[ERROR] detect_faces_robust: {e}")
        return []


def _has_face_geometry(face, max_tilt=None):
    """v5.1050+1052: Valida geometría real de kps (5 landmarks: ojos, nariz, boca).

    Rechaza falsos positivos de RetinaFace en caras de origen (texto, manos, objetos,
    cuerpos) que pasan con det_score bajo. Los 5 kps de una cara real siguen un orden
    anatómico. v5.1052: TOLERANTE A PERFILES/POSES — la v5.1050 exigía orden estricto
    (ojos < nariz < boca en Y + le[0] < re[0] + eye_dist 0.10-1.5) y rechazaba caras
    COMPLETAS válidas: en perfil los ojos se superponen en X (le[0]≈re[0]) y el bbox
    incluye nuca/oreja → eye_dist/face_w < 0.10 → falso negativo. Ahora la guardia
    real es "ojos arriba de la boca + nariz entre ojos y boca" (universal en caras),
    con tolerancias para cabeza inclinada. El orden horizontal solo se exige si hay
    separación real (>2% del ancho); en perfiles casi-verticales no penaliza.
    """
    try:
        kps = getattr(face, 'kps', None)
        if kps is None:
            return False
        # v5.1074: Aceptar tanto numpy (5,2) como lista Python — el padding fallback
        # guarda kps como .tolist(), y _has_face_geometry rechazaba TODAS esas caras.
        if not hasattr(kps, 'shape'):
            kps = np.asarray(kps, dtype=np.float32)
        if kps.shape != (5, 2):
            return False
        kpsf = kps.astype(np.float32)
        if not np.all(np.isfinite(kpsf)):
            return False
        le, re, nose, lm, rm = kpsf
        fw = face.bbox[2] - face.bbox[0]
        fh = face.bbox[3] - face.bbox[1]
        if fw <= 0 or fh <= 0:
            return False
        tol_v = 0.20 * fh  # tolerancia vertical: cabeza inclinada / mirando abajo
        # 1. Ojos arriba de la boca (universal, incluso en perfiles)
        if not (le[1] < lm[1] + tol_v and re[1] < rm[1] + tol_v):
            return False
        # 2. Nariz entre ojos y boca (con margen generoso)
        if not (nose[1] > min(le[1], re[1]) - tol_v and nose[1] < max(lm[1], rm[1]) + tol_v):
            return False
        # 3. Orden horizontal: solo si hay separación real (perfil ≈ vertical no penaliza)
        #    v5.1073: en PERFILES detectados vía pasada rotada (90/270), los kps se
        #    transforman de vuelta y el orden le/re queda invertido legítimamente
        #    (el ojo del sujeto que en el frame rotado estaba a la izquierda pasa a la
        #    derecha al des-rotar). NO se puede usar el orden horizontal como veto
        #    universal: penaliza perfiles reales. La inversión SOLO es señal de falso
        #    positivo (texto/manos) cuando además la confianza es baja — el filtro de
        #    confianza ya la maneja en _filter_false_positive_faces (det<0.50).
        eye_sep_x = abs(re[0] - le[0])
        # 4. Distancia interocular razonable — perfil casi de canto: ojos a <1% del ancho.
        #    Piso 0.005 (0.5px en cara de 100px): rechaza "ojos en el mismo píxel",
        #    acepta perfiles extremos. Techo 1.5: rechaza bboxes mal alineados.
        if not (0.005 < eye_sep_x / max(1.0, fw) < 1.5):
            return False
        # v5.1138: 5. (OPCIONAL, solo source) Línea de ojos con inclinación
        # <= max_tilt grados. Invariante al yaw (los perfiles no penalizan) pero
        # rechaza texto/patrones ROTADOS que RetinaFace marca como cara con det
        # medio ("detecta como caras cosas que no son"). NO se aplica al filtro
        # de target: las pasadas rotadas 90/270 devuelven caras de lado legítimas.
        # v5.1139: 5b. CONTENCIÓN kps-in-bbox — en una cara REAL los 5 landmarks
        # SIEMPRE están dentro del bbox del detector (el bbox se ajusta a los kps).
        # Un falso positivo (texto/patrón/mano marcado como "cara") suele tener los
        # kps FUERA del bbox: el head predictor regresó landmarks coherentes para
        # otra región. Margen 2% del lado mayor (1-2px) por imprecisión de regresión.
        _bx1, _by1, _bx2, _by2 = [float(v) for v in face.bbox]
        _bwk = max(_bx2 - _bx1, _by2 - _by1)
        _marg = max(2.0, 0.02 * _bwk)
        for _p in kpsf:
            if _p[0] < _bx1 - _marg or _p[0] > _bx2 + _marg or _p[1] < _by1 - _marg or _p[1] > _by2 + _marg:
                return False
        if max_tilt is not None:
            _tilt = abs(np.degrees(np.arctan2(float(re[1] - le[1]), float(re[0] - le[0]))))
            if _tilt > float(max_tilt):
                return False
            # En una cara real la nariz cae entre los ojos y la boca. En un
            # falso positivo (mano, texto, patrón) queda fuera de ese grupo.
            _span_lo = min(float(le[0]), float(re[0]), float(lm[0]), float(rm[0])) - 0.18 * float(fw)
            _span_hi = max(float(le[0]), float(re[0]), float(lm[0]), float(rm[0])) + 0.18 * float(fw)
            if not (_span_lo <= float(nose[0]) <= _span_hi):
                return False
        # 6. Ojos dentro de la banda vertical razonable del bbox (15%-80% de la
        # altura): falsos positivos de cuerpo/texto sitúan los "ojos" fuera de
        # esa banda. Caras reales (frontal o perfil) la cumplen siempre.
        _eye_y = (float(le[1]) + float(re[1])) / 2.0
        _eye_rel = (_eye_y - float(face.bbox[1])) / max(1.0, float(fh))
        if not (0.15 < _eye_rel < 0.80):
            return False
        return True
    except Exception:
        return False


def _is_face_complete(face, img_w, img_h):
    """v5.1055: Rechaza caras de source INCOMPLETAS.

    El usuario exige que las caras de origen sean COMPLETAS (de frente o de
    perfil, pero enteras). RetinaFace a veces detecta MEDIA cara cortada por el
    borde de la foto (bbox pegado al borde) o un recorte donde la cara sobresale
    (kps tocando los bordes del bbox). Ambas producen swaps de baja calidad.
    """
    try:
        bbox = getattr(face, 'bbox', None)
        if bbox is None or len(bbox) != 4:
            return False
        x1, y1, x2, y2 = [float(v) for v in bbox]
        fw = x2 - x1
        fh = y2 - y1
        if fw <= 0 or fh <= 0:
            return False
        # v5.1074: Solo rechazar caras CORTADAS por el borde (tocan borde exacto),
        # no caras que estén CERCA del borde. Antes 0.5% del lado menor era
        # demasiado agresivo — selfies, close-ups y fotos de perfil tenían
        # x1=0 o x2≈img_w → TODAS rechazadas como "incompleta".
        # Solo rechazamos si la cara está cortada: toca el borde exacto (±1px).
        if x1 <= 1.0 or y1 <= 1.0:
            return False
        if x2 >= img_w - 1.0 or y2 >= img_h - 1.0:
            return False
        # 2. kps dentro del bbox con margen amplio = cara sobresaliendo del recorte
        # v5.1074: Márgenes reducidos (1%→0.5% de fh/fw) — solo rechaza kps
        # que REALMENTE están fuera del bbox, no los que están ligeramente dentro.
        kps = getattr(face, 'kps', None)
        if kps is None:
            return True  # sin kps el check de borde ya validó la foto
        if not hasattr(kps, 'shape'):
            kps = np.asarray(kps, dtype=np.float32)
        if kps.shape != (5, 2):
            return True
        min_top = max(1.0, 0.01 * fh)
        min_bottom = max(1.0, 0.02 * fh)
        min_side = max(1.0, 0.01 * fw)
        eyes_y = min(float(kps[0][1]), float(kps[1][1]))
        if eyes_y < y1 + min_top:
            return False
        mouth_y = max(float(kps[3][1]), float(kps[4][1]))
        if mouth_y > y2 - min_bottom:
            return False
        for p in kps:
            if p[0] < x1 + min_side or p[0] > x2 - min_side:
                return False
        return True
    except Exception:
        return True  # ante duda no bloquear (la geometría ya filtra)


def _get_faces_at_detector_size(image_bgr, analyser, size):
    """Run a configured detector scale without changing shared analyser state."""
    from insightface.app.common import Face as InsightFace

    size = int(size)
    if size <= 0:
        raise ValueError("Source verification detector size must be positive")
    boxes, keypoints = analyser.det_model.detect(image_bgr, input_size=(size, size))
    faces = []
    for index, box in enumerate(boxes):
        face = InsightFace(
            bbox=box[:4], det_score=box[4],
            kps=None if keypoints is None else keypoints[index],
        )
        for task, model in analyser.models.items():
            if task != "detection":
                model.get(image_bgr, face)
        faces.append(face)
    return faces


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
    _image_id = f"{image_path}_{frame_index}" if is_video else str(image_path)
    
    # 1. Verificar cache de RESULTADOS COMPLETOS
    cache_key = get_cache_key(
        image_path if isinstance(image_path, str) else str(id(image_path)),
        options,
        target_face_detection,
        is_source_face=is_source_face,
    )
    if cache_key in CACHE_RESULTS:
        return CACHE_RESULTS[cache_key]
    
    # Inicializar analizador CUDA
    if FACE_ANALYSER is None:
        FACE_ANALYSER = get_face_analyser()
        if FACE_ANALYSER is None:
            print("[FACES] ❌ FaceAnalysis (CUDA) no disponible.")
            print("[FACES]    Verifica que models/buffalo_l/ exista o que la descarga automatica funcione.")
            return []
    
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
            source_settings = get_faceswap_config()
            source_identity_threshold = float(source_settings["source_identity_similarity_threshold"])
            if not np.isfinite(source_identity_threshold) or not 0 < source_identity_threshold <= 1:
                raise ValueError("Source identity similarity threshold must be in (0, 1]")
            print(f"[SOURCE_DETECT] Cargando cara de origen: {os.path.basename(str(image_path))} ({w}x{h})")

            def _run_detect(img_try, analyser, thresh):
                return detect_faces_robust(img_try, analyser, thresh) or []

            faces = []
            # Cascada permisiva: 0.50 era demasiado alto (perfiles / fotos difíciles = 0 caras)
            for thr in (0.35, 0.22, 0.12, 0.08):
                faces = _run_detect(img_rgb, FACE_ANALYSER, thr)
                if faces:
                    print(f"[SOURCE_DETECT] Detectadas {len(faces)} cara(s) (thresh={thr})")
                    break

            # Preproceso (contraste) si sigue vacío
            if not faces:
                try:
                    pre = preprocess_image_for_detection(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
                    pre_rgb = cv2.cvtColor(pre, cv2.COLOR_BGR2RGB)
                    faces = _run_detect(pre_rgb, FACE_ANALYSER, 0.10)
                    if faces:
                        print(f"[SOURCE_DETECT] Detectadas con preproceso: {len(faces)}")
                except Exception as e:
                    print(f"[SOURCE_DETECT] Preproceso omitido: {e}")

            # A tiny face-like detection must not stop the closeup/alternate
            # scale fallbacks.  In 00689.jpg (768px), a 30x32px background
            # detection was accepted while the main portrait was missed, then
            # became the fixed source for two full videos.
            if faces and not any(
                min(float(f.bbox[2] - f.bbox[0]), float(f.bbox[3] - f.bbox[1])) >= 40
                for f in faces if getattr(f, 'bbox', None) is not None
            ):
                print("[SOURCE_DETECT] Solo detecciones <40px; buscando cara principal a otra escala")
                faces = []

            # Padding controlado
            if not faces and w > 100 and h > 100:
                print("[SOURCE_DETECT] No se detectó cara directamente. Intentando con padding...")
                pad_h, pad_w = h // 3, w // 3
                img_padded = cv2.copyMakeBorder(
                    img_rgb, pad_h, pad_h, pad_w, pad_w, cv2.BORDER_CONSTANT, value=[127, 127, 127]
                )
                faces_padded = _run_detect(img_padded, FACE_ANALYSER, 0.10)
                if faces_padded:
                    for f in faces_padded:
                        fw = f.bbox[2] - f.bbox[0]
                        fh = f.bbox[3] - f.bbox[1]
                        if fw > min(w * 0.15, 35) and fh > min(h * 0.15, 35):
                            # v5.1072b: copiar face antes de mutar para no corromper cache de InsightFace
                            new_bbox = [
                                int(f.bbox[0] - pad_w), int(f.bbox[1] - pad_h),
                                int(f.bbox[2] - pad_w), int(f.bbox[3] - pad_h),
                            ]
                            new_kps = None
                            if hasattr(f, "kps") and f.kps is not None:
                                new_kps = f.kps.copy()
                                new_kps[:, 0] -= pad_w
                                new_kps[:, 1] -= pad_h
                            f_copy = Face(
                                bbox=new_bbox,
                                score=f.score if hasattr(f, 'score') else f.det_score,
                                det_score=f.det_score if hasattr(f, 'det_score') else f.score,
                                kps=new_kps.tolist() if new_kps is not None else None,
                                embedding=f.embedding.tolist() if hasattr(f, 'embedding') and f.embedding is not None else None,
                            )
                            faces.append(f_copy)
                    if faces:
                        print(f"[SOURCE_DETECT] Cara detectada con padding ({len(faces)})")

            # Fallback CPU
            if not faces:
                try:
                    cpu_an = get_face_analyser_cpu()
                    if cpu_an is not None:
                        faces = _run_detect(img_rgb, cpu_an, 0.08)
                        if faces:
                            print(f"[SOURCE_DETECT] Detectadas en CPU: {len(faces)}")
                except Exception as e:
                    print(f"[SOURCE_DETECT] CPU fallback omitido: {e}")

            # Un retrato muy cercano puede superar el tamaño de anchors del
            # detector a 1024. La ruta usada en vídeo prueba 640 y devuelve
            # bbox, landmarks y embedding en coordenadas de la foto original.
            if faces and not any(
                min(float(f.bbox[2] - f.bbox[0]), float(f.bbox[3] - f.bbox[1])) >= 40
                for f in faces if getattr(f, 'bbox', None) is not None
            ):
                faces = []
            _faces_from_smart_fallback = False
            # Detecciones reales pero diminutas también pueden ocultar el
            # retrato principal: mar.JPG daba solo caras del fondo (0.3 % de
            # la foto), mientras otra escala recupera la cara de primer plano.
            _largest_direct_area = max((
                max(0.0, float(f.bbox[2] - f.bbox[0]))
                * max(0.0, float(f.bbox[3] - f.bbox[1]))
                for f in faces if getattr(f, 'bbox', None) is not None
            ), default=0.0)
            if not faces or _largest_direct_area < 0.01 * w * h:
                try:
                    from roop.face_util_rotation import get_all_faces_smart
                    _smart_faces = get_all_faces_smart(
                        img, min_score=0.10, for_target=False, prefer_fast=True,
                    ) or []
                    _largest_smart_area = max((
                        max(0.0, float(f.bbox[2] - f.bbox[0]))
                        * max(0.0, float(f.bbox[3] - f.bbox[1]))
                        for f in _smart_faces if getattr(f, 'bbox', None) is not None
                    ), default=0.0)
                    if _smart_faces and (_largest_smart_area > max(_largest_direct_area * 2, 0.01 * w * h)):
                        faces = _smart_faces
                        _faces_from_smart_fallback = True
                        print(f"[SOURCE_DETECT] Recuperadas {len(faces)} cara(s) con detector de primer plano")
                except Exception as e:
                    print(f"[SOURCE_DETECT] Detector de primer plano omitido: {e}")

            # Rotation deduplication can retain an incomplete or distorted
            # candidate. Keep a direct, complete candidate at another scale
            # before filtering; it still must pass the crop identity check.
            if faces and not any(
                _has_face_geometry(candidate, max_tilt=45.0)
                and _is_face_complete(candidate, w, h)
                for candidate in faces
            ):
                try:
                    for detector_size in source_settings["source_verification_detector_sizes"]:
                        direct_faces = _get_faces_at_detector_size(img, FACE_ANALYSER, detector_size)
                        faces.extend(
                            candidate for candidate in direct_faces
                            if _has_face_geometry(candidate, max_tilt=45.0)
                            and _is_face_complete(candidate, w, h)
                        )
                except Exception as scale_error:
                    print(f"[SOURCE_DETECT] Escala directa omitida: {scale_error}")

            # Cuando una cara ocupa claramente el primer plano, las caras
            # minúsculas de fondo no pertenecen al retrato de origen.
            if faces:
                _areas = [
                    max(0.0, float(f.bbox[2] - f.bbox[0]))
                    * max(0.0, float(f.bbox[3] - f.bbox[1]))
                    for f in faces
                ]
                _largest = max(_areas)
                if _largest >= 0.04 * w * h and all(
                    a == _largest or a <= _largest * 0.25 for a in _areas
                ):
                    faces = [f for f, a in zip(faces, _areas) if a > _largest * 0.25]

            if faces:
                # Filtrar ruido: tamaño mínimo, aspecto razonable, confianza mínima, nitidez
                # v5.1050: + geometría de kps (ojos/nariz/boca) para rechazar falsos positivos
                # (texto, manos, objetos, cuerpos) que RetinaFace marca como "cara" con
                # det_score 0.30-0.50 en la cascada permisiva.
                # v5.1052: det 0.30→0.22 — la cascada detecta caras COMPLETAS de perfil/
                # difíciles con det 0.22-0.30 y el filtro final las descartaba (falso
                # negativo). La geometría de kps es ahora la guardia real contra falsos
                # positivos; el det bajo + geometría válida = cara real difícil.
                # v5.1154: det 0.22→0.14 — el gate 0.22 mataba caras REALES del usuario
                # (medido: 185 caras reales rechazadas solo por det en su carpeta MARETA,
                # 740→69 en la carga). La segunda pasada v5.1153b (re-detección con
                # padding+downscale, umbral sim 0.50 calibrado) es ahora el VERDADERO
                # cazador de falsos positivos: un FP con det 0.14-0.22 no supera sim 0.50.
                _SRC_DET_MIN = 0.14
                valid_source_faces = []
                for f in faces:
                    fw = f.bbox[2] - f.bbox[0]
                    fh = f.bbox[3] - f.bbox[1]
                    aspect = fw / fh if fh > 0 else 0
                    det = getattr(f, 'det_score', 0.0) or 0.0
                    if not (fw >= 40 and fh >= 40 and 0.25 < aspect < 3.5 and det >= _SRC_DET_MIN):
                        if det < _SRC_DET_MIN:
                            print(f"[SOURCE_DETECT] ⚠️ det={det:.2f} < {_SRC_DET_MIN} — cara descartada ({fw}x{fh}px)")
                        continue
                    if not _has_face_geometry(f, max_tilt=45.0):
                        print(
                            f"[SOURCE_DETECT] ❌ Sin geometría de cara (kps no anatómicos) "
                            f"det={det:.2f}. Omitiendo."
                        )
                        continue
                    # v5.1055: cara debe estar COMPLETA (no cortada por borde de foto,
                    # no sobresaliendo del bbox). De frente o perfil, pero entera.
                    if not _is_face_complete(f, w, h):
                        print(
                            f"[SOURCE_DETECT] ❌ Cara INCOMPLETA (cortada/saliendo del recorte) "
                            f"bbox=({f.bbox[0]:.0f},{f.bbox[1]:.0f},{f.bbox[2]:.0f},{f.bbox[3]:.0f}) "
                            f"img={w}x{h} det={det:.2f}. Omitiendo."
                        )
                        continue
                    if det < 0.50:
                        from roop.source_face_validator import confirm_source_region
                        confirmed, detail = confirm_source_region(img, f)
                        if not confirmed:
                            print(f"[SOURCE_DETECT] Región dudosa descartada (det={det:.2f}; {detail})")
                            continue
                    valid_source_faces.append(f)
                # v5.1153 — VERIFICACIÓN DE SEGUNDA PASADA: los gates previos (det+geometría+
                # completitud+dedup) dejan pasar falsos positivos PERSISTENTES: manos/textos/
                # patrones que RetinaFace marca con det 0.3-0.5 y kps anatómicamente plausibles.
                # Prueba estructural: un crop que ES una cara real vuelve a detectarse como cara
                # (misma identidad, sim alta); un falso positivo NO (o detecta otra cosa). Se
                # re-detecta SOBRE el crop de la galería y se compara embedding con la detección
                # original: sim < 0.55 o cero caras → no es la cara que decía ser → descartada.
                # Solo en caras >= 40px (un re-detect en crops diminutos es poco fiable; ahí
                # siguen mandando los gates previos).
                try:
                    _verified = []
                    _smart_verify_faces = None
                    for _f in valid_source_faces:
                        _fw_v = _f.bbox[2] - _f.bbox[0]
                        _fh_v = _f.bbox[3] - _f.bbox[1]
                        if min(_fw_v, _fh_v) < 40:
                            _verified.append(_f)
                            continue
                        _x1v, _y1v, _x2v, _y2v = [int(max(0, v)) for v in _f.bbox]
                        # v5.1153b — 2 bugs de la v5.1153 corregidos (causa del 740->0):
                        # (1) el bbox apretado hace que la cara LLENE el crop — RetinaFace tiene
                        #     techo de anchor ~512px (v5.1139 FIX 2) y una cara de 900-2000px en
                        #     su propio crop es INVISIBLE → n_faces_crop=0 → rechazo en cadena.
                        #     FIX: padding 50% alrededor del bbox + downscale para que la cara
                        #     quede en ~350px (zona de detección segura del detector).
                        # (2) crop en RGB mientras el recognizer corre en BGR → embeddings con
                        #     canales cambiados. FIX: crop BGR nativo + fallback RGB.
                        _pad_x = int((_x2v - _x1v) * 0.5)
                        _pad_y = int((_y2v - _y1v) * 0.5)
                        _cx1 = max(0, _x1v - _pad_x)
                        _cy1 = max(0, _y1v - _pad_y)
                        _cx2 = min(w, _x2v + _pad_x)
                        _cy2 = min(h, _y2v + _pad_y)
                        _crop_v = img[_cy1:_cy2, _cx1:_cx2]  # img es BGR aquí
                        if _crop_v.size == 0:
                            continue
                        _face_px_v = max(_x2v - _x1v, _y2v - _y1v)
                        if _face_px_v > 350:
                            _scale_v = 350.0 / _face_px_v
                            _crop_v = cv2.resize(
                                _crop_v,
                                (max(1, int(_crop_v.shape[1] * _scale_v)), max(1, int(_crop_v.shape[0] * _scale_v))),
                                interpolation=cv2.INTER_AREA,
                            )
                        _faces_v = None
                        _emb_orig = getattr(_f, 'embedding', None)
                        _best_sim_v = 0.0
                        _ok_v = False
                        if _emb_orig is not None:
                            _emb_o = np.asarray(_emb_orig, dtype=np.float32).flatten()
                            _no = np.linalg.norm(_emb_o)
                            if _no > 0:
                                _emb_o = _emb_o / _no
                                # Una detección cualquiera en BGR no demuestra
                                # que el color corresponda al embedding de la
                                # detección original (obtenida en RGB). Probar
                                # ambos y aceptar solo la misma identidad.
                                for _cv in (_crop_v, cv2.cvtColor(_crop_v, cv2.COLOR_BGR2RGB)):
                                    try:
                                        _faces_v = FACE_ANALYSER.get(_cv)
                                    except Exception:
                                        _faces_v = None
                                    for _fv in (_faces_v or []):
                                        _ev = getattr(_fv, 'embedding', None)
                                        if _ev is None:
                                            continue
                                        _ev = np.asarray(_ev, dtype=np.float32).flatten()
                                        _nv = np.linalg.norm(_ev)
                                        if _nv > 0:
                                            _sim_v = float(np.dot(_emb_o, _ev / _nv))
                                            _best_sim_v = max(_best_sim_v, _sim_v)
                                            if _sim_v >= source_identity_threshold:
                                                _ok_v = True
                                                break
                                    if _ok_v:
                                        break
                        if not _ok_v and _emb_orig is not None:
                            # The 1024 anchors can miss even a padded crop.
                            # Confirm the same identity on that crop at the
                            # configured scales, without rotated deduplication.
                            try:
                                for detector_size in source_settings["source_verification_detector_sizes"]:
                                    for crop_image in (_crop_v, cv2.cvtColor(_crop_v, cv2.COLOR_BGR2RGB)):
                                        _faces_v = _get_faces_at_detector_size(crop_image, FACE_ANALYSER, detector_size)
                                        for verified_face in _faces_v:
                                            if not _has_face_geometry(verified_face, max_tilt=45.0):
                                                continue
                                            verified_embedding = getattr(verified_face, 'embedding', None)
                                            if verified_embedding is None:
                                                continue
                                            verified_embedding = np.asarray(verified_embedding, dtype=np.float32).reshape(-1)
                                            verified_norm = np.linalg.norm(verified_embedding)
                                            if verified_norm > 0 and verified_embedding.size == _emb_o.size:
                                                verified_similarity = float(np.dot(_emb_o, verified_embedding / verified_norm))
                                                _best_sim_v = max(_best_sim_v, verified_similarity)
                                                if verified_similarity >= source_identity_threshold:
                                                    _ok_v = True
                                                    print(f"[SOURCE_DETECT] Crop confirma identidad a {detector_size}px (sim={verified_similarity:.3f})")
                                                    break
                                        if _ok_v:
                                            break
                                    if _ok_v:
                                        break
                            except Exception as scale_error:
                                print(f"[SOURCE_DETECT] Verificación de escala omitida: {scale_error}")
                        if not _ok_v and _emb_orig is not None and not _faces_from_smart_fallback:
                            # A face can disappear when cropped even with
                            # padding (closeups and unusual expressions).
                            # Confirm it at another detector scale in the
                            # original image before rejecting the embedding.
                            if _smart_verify_faces is None:
                                try:
                                    from roop.face_util_rotation import get_all_faces_smart
                                    _smart_verify_faces = get_all_faces_smart(
                                        img, min_score=0.10, for_target=False,
                                        prefer_fast=True,
                                    ) or []
                                except Exception as _smart_exc:
                                    _smart_verify_faces = []
                                    print(f"[SOURCE_DETECT] Segunda escala omitida: {_smart_exc}")
                            from roop.face_util_rotation import _same_face_boxes
                            _eo = np.asarray(_emb_orig, dtype=np.float32).reshape(-1)
                            _eo_norm = np.linalg.norm(_eo)
                            if _eo_norm > 0:
                                for _alt in _smart_verify_faces:
                                    _alt_emb = getattr(_alt, 'embedding', None)
                                    if _alt_emb is None or not _same_face_boxes(_f.bbox, _alt.bbox):
                                        continue
                                    _ea = np.asarray(_alt_emb, dtype=np.float32).reshape(-1)
                                    _ea_norm = np.linalg.norm(_ea)
                                    if _ea.size == _eo.size and _ea_norm > 0:
                                        _alt_sim = float(np.dot(_eo / _eo_norm, _ea / _ea_norm))
                                        _best_sim_v = max(_best_sim_v, _alt_sim)
                                        if _alt_sim >= source_identity_threshold:
                                            _ok_v = True
                                            print(
                                                f"[SOURCE_DETECT] Segunda escala confirma cara "
                                                f"(sim={_alt_sim:.3f})",
                                                flush=True,
                                            )
                                            break
                        if _ok_v:
                            _verified.append(_f)
                        else:
                            print(
                                f"[SOURCE_DETECT] ❌ Segunda pasada: el crop NO re-detecta la misma "
                                f"cara (falso positivo persistente) bbox=({_x1v},{_y1v},{_x2v},{_y2v}) "
                                f"det={float(getattr(_f, 'det_score', 0.0) or 0.0):.2f} "
                                f"sim={_best_sim_v:.3f} n_faces_crop={len(_faces_v) if _faces_v else 0}. Omitiendo."
                            )
                    valid_source_faces = _verified
                except Exception as _e_verify:
                    valid_source_faces = _verified
                    print(f"[SOURCE_DETECT] Verificación interrumpida; se conservan solo las caras verificadas: {_e_verify}")
                # v5.1139: dedup — RetinaFace a veces devuelve DOS cajas para la MISMA
                # cara (a distinta escala/offset). El usuario veía caras repetidas en la
                # galería de fuentes. Se queda la de mayor det_score.
                try:
                    from roop.face_util_rotation import _same_face_boxes
                    valid_source_faces.sort(key=lambda _f: float(getattr(_f, 'det_score', 0.0) or 0.0), reverse=True)
                    _kept_sf = []
                    for _f in valid_source_faces:
                        if any(_same_face_boxes(_f.bbox, _k.bbox) for _k in _kept_sf):
                            continue
                        _kept_sf.append(_f)
                    valid_source_faces = _kept_sf
                except Exception:
                    pass
                faces = valid_source_faces
                if not faces:
                    print("[SOURCE_DETECT] Caras filtradas por tamaño/aspecto/confianza/geometría (ninguna válida)")

            if faces:
                # Procesar todas las caras detectadas
                results = []
                for face_data_full in faces:
                    # Asegurar que el embedding NO sea None
                    face_embedding = getattr(face_data_full, 'embedding', None)
                    
                    if face_embedding is None:
                        print("[SOURCE_DETECT] ⚠️ Cara detectada pero SIN EMBEDDING. Intentando recuperación agresiva...")
                        # Extraer el crop de la cara
                        x1, y1, x2, y2 = np.asarray(face_data_full.bbox, dtype=np.int32)
                        # Clamping
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        face_crop = img_rgb[y1:y2, x1:x2].copy()
                        
                        # Intento 1: Padding
                        if face_crop.size > 0:
                            h_c, w_c = face_crop.shape[:2]
                            pad = max(h_c, w_c) // 2
                            face_padded = cv2.copyMakeBorder(face_crop, pad, pad, pad, pad, 
                                                              cv2.BORDER_CONSTANT, value=[128, 128, 128])
                            try:
                                faces_p = FACE_ANALYSER.get(face_padded)
                                # Hueco: la cara recuperada entraba SIN filtros (geometría/
                                # completitud). Un falso positivo (texto/mano) se "recuperaba"
                                # como cara válida y acababa en la galería de fuentes.
                                # La geometría es invariante a traslación: vale en el crop.
                                if faces_p and getattr(faces_p[0], 'embedding', None) is not None:
                                    if _has_face_geometry(faces_p[0], max_tilt=45.0):
                                        # El bbox/kps vienen en coords del crop ACOLCHADO
                                        # (face_crop + pad). Sin remapear, el recorte de la
                                        # galería cae en otra región ("recortes mal" que no
                                        # son caras). Remap a coords de la imagen original.
                                        _rp = faces_p[0]
                                        try:
                                            _rb = np.asarray(_rp.bbox, dtype=np.float32).copy()
                                            _rb[0::2] += (x1 - pad)
                                            _rb[1::2] += (y1 - pad)
                                            _rb[0::2] = np.clip(_rb[0::2], 0, w)
                                            _rb[1::2] = np.clip(_rb[1::2], 0, h)
                                            _rk = None
                                            if getattr(_rp, 'kps', None) is not None:
                                                _rk = np.asarray(_rp.kps, dtype=np.float32).copy()
                                                _rk[:, 0] += (x1 - pad)
                                                _rk[:, 1] += (y1 - pad)
                                            face_data_full = Face(
                                                bbox=_rb,
                                                score=getattr(_rp, 'score', 0.5),
                                                det_score=getattr(_rp, 'det_score', 0.5),
                                                kps=_rk,
                                                embedding=_rp.embedding,
                                            )
                                        except Exception:
                                            face_data_full = _rp
                                        print("[SOURCE_DETECT] ✅ Embedding recuperado con padding")
                                        face_embedding = face_data_full.embedding
                                    else:
                                        print("[SOURCE_DETECT] ❌ Recuperado con padding sin geometría — se descarta")
                            except BaseException:
                                pass

                        # Intento 2: Sharpening + Brightness
                        if face_embedding is None and face_crop.size > 0:
                            enhanced = preprocess_image_for_detection(face_crop)
                            try:
                                faces_e = FACE_ANALYSER.get(enhanced)
                                if faces_e and getattr(faces_e[0], 'embedding', None) is not None:
                                    if _has_face_geometry(faces_e[0], max_tilt=45.0):
                                        # `enhanced` conserva el tamaño de face_crop: el
                                        # bbox/kps están en coords del crop. Remap a la
                                        # imagen original (ver bloque padding arriba).
                                        _re = faces_e[0]
                                        try:
                                            _eb = np.asarray(_re.bbox, dtype=np.float32).copy()
                                            _eb[0::2] += x1
                                            _eb[1::2] += y1
                                            _eb[0::2] = np.clip(_eb[0::2], 0, w)
                                            _eb[1::2] = np.clip(_eb[1::2], 0, h)
                                            _ek = None
                                            if getattr(_re, 'kps', None) is not None:
                                                _ek = np.asarray(_re.kps, dtype=np.float32).copy()
                                                _ek[:, 0] += x1
                                                _ek[:, 1] += y1
                                            face_data_full = Face(
                                                bbox=_eb,
                                                score=getattr(_re, 'score', 0.5),
                                                det_score=getattr(_re, 'det_score', 0.5),
                                                kps=_ek,
                                                embedding=_re.embedding,
                                            )
                                        except Exception:
                                            face_data_full = _re
                                        print("[SOURCE_DETECT] ✅ Embedding recuperado con preprocesamiento")
                                        face_embedding = face_data_full.embedding
                                    else:
                                        print("[SOURCE_DETECT] ❌ Recuperado con preproceso sin geometría — se descarta")
                            except BaseException:
                                pass

                    if face_embedding is None:
                        # Si no hay embedding tras todo esto, omitimos esta cara
                        print("[SOURCE_DETECT] ❌ Falló extracción de embedding para una cara. Omitiendo.")
                        continue

                    normed_emb = None
                    emb = np.array(face_embedding)
                    n = np.linalg.norm(emb)
                    if n > 0:
                        normed_emb = emb / n

                    face_obj = Face(
                        bbox=np.asarray(face_data_full.bbox, dtype=np.int32).tolist(),
                        score=getattr(face_data_full, 'score', 0.5),
                        det_score=getattr(face_data_full, 'det_score', 0.5),
                        kps=np.asarray(face_data_full.kps).tolist() if getattr(face_data_full, 'kps', None) is not None else None,
                        embedding=face_embedding,
                        normed_embedding=normed_emb
                    )
                    face_obj.source_image = str(image_path)
                    face_obj.source_image_size = (w, h)
                    
                    # Extraer el recorte real para la UI
                    current_padding = ui_padding if ui_padding is not None else face_detection_padding
                    x1, y1, x2, y2 = face_obj.bbox
                    fw, fh = x2 - x1, y2 - y1
                    px, py = int(fw * current_padding), int(fh * current_padding)
                    x1p, y1p = max(0, x1 - px), max(0, y1 - py)
                    x2p, y2p = min(w, x2 + px), min(h, y2 + py)
                    face_img = img_rgb[y1p:y2p, x1p:x2p].copy()
                    face_obj.face_img_origin = (x1p, y1p)
                    face_obj.face_img_is_rgb = True

                    # Filtro de nitidez: rechazar caras muy borrosas
                    # v5.1074: Laplacian variance normalizada (var/mean²) — invariante
                    # al tamaño del crop. Fotos de 6000px daban lap_var~3.0 (falso borroso);
                    # fotos de 1000px daban lap_var~12 (correcto). Con la normalización,
                    # ambas dan ~12-14 si la cara es nítida.
                    gray_crop = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                    lap_raw = cv2.Laplacian(gray_crop, cv2.CV_64F)
                    lap_var = lap_raw.var()
                    lap_mean = float(np.mean(np.abs(lap_raw)))
                    lap_norm = lap_var / max(1.0, lap_mean ** 2)  # normalizado
                    if lap_norm < 1.5:
                        print(f"[SOURCE_DETECT] ❌ Cara borrosa (lap_norm={lap_norm:.1f} < 1.5, raw={lap_var:.1f}). Omitiendo.")
                        continue

                    # Guardar el recorte en el objeto
                    face_obj.face_img = face_img
                    
                    results.append([face_obj, face_img])
                
                _cache_put(CACHE_RESULTS, cache_key, results)
                return results
            else:
                print(f"[SOURCE_DETECT] ❌ No se detectó ninguna cara real en {os.path.basename(str(image_path))}")
                _cache_put(CACHE_RESULTS, cache_key, [])
                return []


        # ============ TARGET FACES: DETECCIÓN NORMAL ============
        target_thresh = 0.15 # Reducido de 0.35 para no perder caras en fotos difíciles o grupos
        
        # Preprocesamiento opcional
        img_preprocessed = preprocess_image_for_detection(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        img_to_detect = cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB)
        
        faces = detect_faces_robust(img_to_detect, FACE_ANALYSER, target_thresh)

        # v5.1138 — CLOSEUP: foto de primer plano (la cara ocupa casi todo el
        # frame). RetinaFace @1024 a menudo NO detecta caras >50% del frame.
        # Mismos fallbacks que el path de source: (a) pasada det_size 640,
        # (b) padding gris (la cara "se aleja" del detector), (c) CPU.
        if not faces:
            try:
                _orig_ds = getattr(FACE_ANALYSER, "det_size", None)
                _orig_is = None
                if hasattr(FACE_ANALYSER, "det_model") and FACE_ANALYSER.det_model is not None:
                    _orig_is = getattr(FACE_ANALYSER.det_model, "input_size", None)
                try:
                    FACE_ANALYSER.det_size = (640, 640)
                    if hasattr(FACE_ANALYSER, "det_model") and FACE_ANALYSER.det_model is not None:
                        FACE_ANALYSER.det_model.input_size = (640, 640)
                    faces = detect_faces_robust(img_to_detect, FACE_ANALYSER, target_thresh)
                    if faces:
                        print(f"[FACES] Primer plano: det_size 640 detectó {len(faces)} cara(s)")
                finally:
                    if _orig_ds is not None:
                        FACE_ANALYSER.det_size = _orig_ds
                    if _orig_is is not None and hasattr(FACE_ANALYSER, "det_model") and FACE_ANALYSER.det_model is not None:
                        FACE_ANALYSER.det_model.input_size = _orig_is
            except Exception:
                pass

        if not faces and img_rgb.shape[0] > 100 and img_rgb.shape[1] > 100:
            try:
                _h_t, _w_t = img_rgb.shape[:2]
                _pad_h, _pad_w = _h_t // 3, _w_t // 3
                _img_pad = cv2.copyMakeBorder(
                    img_to_detect, _pad_h, _pad_h, _pad_w, _pad_w,
                    cv2.BORDER_CONSTANT, value=[127, 127, 127],
                )
                _faces_p = detect_faces_robust(_img_pad, FACE_ANALYSER, 0.12)
                if _faces_p:
                    _shims = []
                    for _f in _faces_p:
                        _fw = float(_f.bbox[2] - _f.bbox[0])
                        _fh = float(_f.bbox[3] - _f.bbox[1])
                        if _fw < min(_w_t * 0.15, 35) or _fh < min(_h_t * 0.15, 35):
                            continue

                        class _PadFace:
                            pass

                        _sh = _PadFace()
                        _sh.bbox = np.array(
                            [_f.bbox[0] - _pad_w, _f.bbox[1] - _pad_h,
                             _f.bbox[2] - _pad_w, _f.bbox[3] - _pad_h],
                            dtype=_f.bbox.dtype,
                        )
                        _sh.kps = None
                        if getattr(_f, "kps", None) is not None:
                            _sh.kps = _f.kps.copy()
                            _sh.kps[:, 0] -= _pad_w
                            _sh.kps[:, 1] -= _pad_h
                        _sh.embedding = getattr(_f, "embedding", None)
                        _sh.det_score = float(getattr(_f, "det_score", 0.5) or 0.5)
                        _sh.score = getattr(_f, "score", _sh.det_score)
                        _sh.landmark_2d_106 = getattr(_f, "landmark_2d_106", None)
                        _shims.append(_sh)
                    if _shims:
                        faces = _shims
                        print(f"[FACES] Primer plano: padding detectó {len(faces)} cara(s)")
            except Exception:
                pass

        if not faces:
            try:
                _cpu_an = get_face_analyser_cpu()
                if _cpu_an is not None:
                    faces = detect_faces_robust(img_to_detect, _cpu_an, 0.12)
                    if faces:
                        print(f"[FACES] Primer plano: CPU detectó {len(faces)} cara(s)")
            except Exception:
                pass

        if not faces:
            _cache_put(CACHE_RESULTS, cache_key, [])
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
                landmark_106 = getattr(face_data, 'landmark_2d_106', None)
                
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
                
            except Exception:
                continue
        
        if len(results) > 0:
            results.sort(key=lambda x: (x[0].bbox[2] - x[0].bbox[0]) * (x[0].bbox[3] - x[0].bbox[1]), reverse=True)
        _cache_put(CACHE_RESULTS, cache_key, results)
        return results
    

    except Exception:
        _cache_put(CACHE_RESULTS, cache_key, [])
        return []
