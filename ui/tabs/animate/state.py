import os
import torch

# Variables de estado para animación
is_animating = False  # reset on app start; worker sets True/False per job
current_video_path = None
_animation_thread = None

# Caché de modelos y workflows
AVAILABLE_MODELS = {}
SELECTED_MODEL = "framepack" # FramePack para img2vid en VRAM baja

# Configuración de VRAM
VRAM_8GB_MODE = True # Optimizado para la GPU del usuario

# Referencias para restauración facial en vídeo
face_restoration_enabled = True
original_image_faces = []
