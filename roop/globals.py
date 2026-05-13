from typing import List

from settings import Settings

# Configuración global
CFG = Settings('config.yaml')

# Variables para almacenamiento de facesets y caras de destino
INPUT_FACESETS = []  # Lista de facesets de entrada
TARGET_FACES = []    # Lista de caras de destino

source_path = None
target_path = None
output_path = None
target_folder_path = None

frame_processors: List[str] = []
keep_fps = None
keep_frames = None
autorotate_faces = None
vr_mode = None
skip_audio = None
wait_after_extraction = None
many_faces = None
use_batch = None
source_face_index = 0
target_face_index = 0
face_position = None
video_encoder = None
video_quality = None
max_memory = None
execution_providers: List[str] = ["CUDAExecutionProvider", "CPUExecutionProvider"]  # Default to CUDA with CPU fallback
execution_threads = None
headless = None
log_level = "error"
selected_enhancer = None
face_swap_mode = None
flip_faces = None
face_rotation_correction = None
face_rotation_angle = None
blend_ratio = 1.0  # 100% source - MÁXIMA SIMILITUD AL ORIGEN
distance_threshold = 0.30  # MÁS ESTRICTO: solo matches muy cercanos al origen

# Thresholds optimizados para MÁXIMA FIDELIDAD
similarity_threshold_selected = 0.2  # ESTRICTO para máxima semejanza al origen
similarity_threshold_auto = 0.15     # ESTRICTO para matching automático
similarity_threshold_fallback = 0.1 # FALLBACK estricto
min_similarity_threshold = 0.25  # MÍNIMO para hacer swap (skip frame si es menor)
default_det_size = True  # Usar tamaño de detector por defecto - más rápido

# Threshold más estricto para MÁXIMA FIDELIDAD al origen
face_match_embedding_threshold = 0.25  # ESTRICTO - solo usar caras muy similares
face_match_bbox_iou_threshold = 0.50   # Muy estricto para mejor coincidencia
show_face_area = False  # Variable para mostrar área de cara en preview
use_enhancer = False  # Desactivado por defecto para velocidad real-time (CodeFormer ~1.2 FPS, muy lento)
blend_mode = 'seamless'  # Mejor integración visual que Poisson
use_color_correction = True  # REACTIVADO para preservar cara origen (source face)
use_color_matching = True  # REACTIVADO para tono de piel del origen

# CONFIGURACIÓN ÓPTIMA PARA MÁXIMO PARECIDO SIN PERDER CALIDAD
# Balance entre parecido al origen y calidad de imagen

# Factor de blending del enhancer (0-1)
# 0 = sin enhancer, 1 = enhancer completo
# 0.3 para que el swap sea visible pero con calidad
enhancer_blend_factor = 0.3

# Ajuste de brillo (0-1)
# 0.15 para no lavar demasiado la cara original
brightness_strength = 0.15

# Matching de color (0-1)
# 0.10 para preservar mejor el origen (source face)
color_match_strength = 0.10
# GFPGAN preserva mejor la identidad en face swap
# CodeFormer suaviza demasiado perdiendo rasgos faciales
default_enhancer = 'GFPGAN'  # GFPGAN para mejor preservación de identidad

# BATCH PROCESSING - Procesamiento en paralelo para videos
# Número de frames a procesar simultáneamente
# 1 = Procesamiento secuencial (lento)
# 2-4 = Procesamiento en batch (rápido, recomendado para GPU con 4GB+ VRAM)
# 8-16 = Procesamiento masivo (muy rápido, requiere GPU con 8GB+ VRAM)
batch_processing_size = 4  # Default: 4 frames simultáneos (balance velocidad/memoria)
max_batch_threads = 4  # Máximo número de hilos para procesamiento paralelo

# PRESERVACIÓN DE EXPRESIÓN DE BOCA
# Cuando la cara destino tiene la boca abierta (hablando, comiendo, etc.),
# preserva la zona de la boca original en lugar de imponer la boca cerrada del origen
preserve_mouth_expression = True

# MODO CALIDAD MÁXIMA
CONSERVATIVE_MODE = False  # Desactiva protecciones para contenido adulto
face_rotation_correction = True  # Corrige automáticamente la orientación de las caras

# Comportamiento para modos automáticos: si True, incluirá caras desconocidas en 'all_female'/'all_male'
# (útil si el modelo genderage devuelve None con frecuencia y quieres más inclusividad)
include_unknown_in_gender_auto = True  # Cambiado a True para mayor inclusividad

# Threshold de confianza para detección de género en modos automáticos
# Valores entre 0.0 y 1.0. Un valor más bajo permite más caras ambiguas.
# 0.5 = moderado, 0.3 = muy permisivo, 0.7 = muy estricto
gender_confidence_threshold = 0.35  # Más permisivo para mejor detección de género

# NUEVO: Modo de strictness para detección de género en all_female/all_male
# 'permissive' = Acepta más caras (posibles falsos positivos pero no se pierde nadie)
# 'balanced' = Equilibrado (default)
# 'strict' = Solo caras con alta confianza de género
gender_strictness_mode = 'permissive'  # Acepta más caras para evitar falsos negativos

# NUEVO: Umbrales específicos por modo de strictness
# Estos definen la "zona gris" donde el género es incierto
gender_threshold_permissive = (0.30, 0.70)  # Zona gris pequeña: más caras aceptadas
gender_threshold_balanced = (0.40, 0.60)     # Zona gris media
gender_threshold_strict = (0.45, 0.55)       # Zona gris amplia: solo caras muy claras

# Usar una sola fuente para mayor coherencia, especialmente en videos
use_single_source_for_all = True

# Habilitar suavizado temporal para videos (reduce el parpadeo)
temporal_smoothing = True

# Usar MediaPipe como detector principal (más preciso y rápido)
use_mediapipe_detector = True

no_face_action = 0

processing = False
use_enhancer = True # Activado por defecto para máxima calidad en swaps


def apply_conservative_defaults():
    """
    Aplica configuración optimizada por defecto para MÁXIMA FIDELIDAD AL ORIGEN + VELOCIDAD
    """
    global blend_ratio, distance_threshold, face_swap_mode, execution_providers
    global CONSERVATIVE_MODE

    # Verificar si CFG está disponible y usar valores del config.yaml
    if CFG is not None:
        try:
            blend_ratio = getattr(CFG, 'face_swap_blend_ratio', 1.0)
            distance_threshold = getattr(CFG, 'face_swap_distance_threshold', 0.35)
            face_swap_mode = getattr(CFG, 'face_swap_mode', 'selected_faces')
            CONSERVATIVE_MODE = getattr(CFG, 'conservative_mode', False)

            if CFG.force_cpu or CFG.provider.lower() == 'cpu':
                execution_providers = ["CPUExecutionProvider"]
            else:
                execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

            print("[CONFIG] CONFIGURACION DESDE ARCHIVO CARGADA:")
            print(f"   [OK] Blend ratio: {blend_ratio} ({blend_ratio*100:.0f}% cara origen, {(1-blend_ratio)*100:.0f}% cara target)")
            print(f"   [OK] Distance threshold: {distance_threshold} (similitud mínima para match facial)")
            print(f"   [OK] Face swap mode: {face_swap_mode}")
            print(f"   [OK] Execution providers: {execution_providers}")
        except:
            blend_ratio = 1.0
            distance_threshold = 0.35
            face_swap_mode = "selected_faces"
            execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            print("[CONFIG] CONFIGURACION OPTIMIZADA POR DEFECTO:")
            print(f"   [OK] Blend ratio: {blend_ratio} ({blend_ratio*100:.0f}% cara origen, {(1-blend_ratio)*100:.0f}% cara target)")
            print(f"   [OK] Distance threshold: {distance_threshold} (similitud mínima para match facial)")
            print(f"   [OK] Face swap mode: {face_swap_mode}")
            print(f"   [OK] Execution providers: {execution_providers}")
    else:
        blend_ratio = 1.0
        distance_threshold = 0.35
        face_swap_mode = "selected_faces"
        execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        print("[CONFIG] CONFIGURACION OPTIMIZADA ACTIVADA:")
        print(f"   [OK] Blend ratio: {blend_ratio} ({blend_ratio*100:.0f}% cara origen, {(1-blend_ratio)*100:.0f}% cara target)")
        print(f"   [OK] Distance threshold: {distance_threshold} (similitud mínima para match facial)")
        print(f"   [OK] Face swap mode: {face_swap_mode}")
        print(f"   [OK] Execution providers: {execution_providers}")


def apply_config_from_file():
    """
    Aplica configuración desde config.yaml después de que se haya cargado
    """
    global blend_ratio, distance_threshold, face_swap_mode

    if CFG is not None:
        try:
            blend_ratio = getattr(CFG, 'face_swap_blend_ratio', 1.0)
            distance_threshold = getattr(CFG, 'face_swap_distance_threshold', 0.35)
            face_swap_mode = getattr(CFG, 'face_swap_mode', 'selected_faces')
            
            # Optimización de Hardware (VRAM)
            low_vram = getattr(CFG, 'low_vram', True)
            enable_fp16 = getattr(CFG, 'enable_fp16', True)
            batch_processing_size = getattr(CFG, 'batch_processing_size', 4)
            max_batch_threads = getattr(CFG, 'max_batch_threads', 4)
            cuda_malloc_async = getattr(CFG, 'cuda_malloc_async', True)

            print("CONFIGURACION DESDE config.yaml APLICADA:")
            print(f"   - Blend ratio: {blend_ratio} ({(1-blend_ratio)*100:.0f}% cara original, {blend_ratio*100:.0f}% cara target)")
            print(f"   - Distance threshold: {distance_threshold} (similitud mínima para match facial)")
            print(f"   - Face swap mode: {face_swap_mode} (desde config.yaml)")
        except Exception as e:
            print(f"Error aplicando config.yaml: {e}")

# ============================================================================
# DETECTOR DE CARAS (MediaPipe vs InsightFace)
# ============================================================================
# Si True, usa MediaPipe como detector principal (más moderno/estable)
# Si False, usa InsightFace (original) con MediaPipe como fallback
use_mediapipe_detector = False  # Cambiar a True para usar MediaPipe por defecto

# Face similarity threshold (para matching de caras) - MÁXIMA FIDELIDAD
face_similarity_threshold = 0.2  # Más estricto para mejor matching
use_gender_filter = False

# Aplicar configuración conservadora automáticamente
apply_conservative_defaults()

