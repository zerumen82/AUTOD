import queue

# Variables globales para control de hilos
_source_thread_pool = None
_target_thread_pool = None
_processing_queue = queue.Queue()
MAX_CONCURRENT_THREADS = 8
FACE_DETECTION_CACHE = {}
MAX_CACHE_SIZE = 100

# Variables para recordar últimas carpetas
last_source_folder = None
last_target_folder = None

# Variables para almacenar archivos seleccionados
source_filenames = []

# Variables para trackear tipo de procesamiento
is_video_processing = False
is_image_processing = False

# Variables para paginación
current_input_page = 0
current_target_page = 0
FACES_PER_PAGE = 32

# Variables para selección de caras
SELECTED_FACE_INDEX = 0
SELECTED_TARGET_FACE_INDEX = 0
IS_INPUT = True
SELECTION_FACES_DATA = []
CURRENT_DETECTED_FACES = []
TEMP_SELECTED_FACE_INDEX = 0

# Flags para prevenir eventos de selección automáticos de Gradio
_IS_UPDATING_GALLERY = False
_IS_UPDATING_TARGET = False

# Variables para procesamiento de videos
selected_preview_index = 0
list_files_process = []
current_video_fps = 30

# Variables de control de estado
is_processing = False
manual_masking = False

# Referencias para Propagación Inteligente (Smart Tracking)
# Formato: { 'filename_ext': { 'bbox': [], 'embedding': [] } }
selected_face_references = {}
