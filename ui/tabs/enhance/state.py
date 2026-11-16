"""Gestión de estado para la pestaña de mejora de fotos."""


class EnhanceSessionState:
    """Estado de sesión para la pestaña de mejora de fotos."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.is_generating = False
        self.processing_queue = []
        self.current_preset = "Retrato"
        self.enhancement_settings = {
            "denoise_strength": 65,
            "sharpen_strength": 40,
            "recover_faces_strength": 0.8,
            "upscale_factor": 2,
            "exposure": 1.0,
            "shadow": 1.0,
            "highlight": 1.0,
            "temperature": 50,
            "grain_strength": 20,
            "grain_size": 10,
        }
        self.current_file = None
        self.result_image = None
        self.before_image = None
        self.uploaded_files = []
        self.output_dir = "enhance"
        self.auto_mode = False
        self.auto_analysis = None
        self.export_format = "png"
        self.apply_filters_to_filename = True
        self.allow_overwrite = True
        self.progress = 0.0
        self.status_message = "Listo"


# Estado global del módulo (para compatibilidad con Gradio)
session_state = EnhanceSessionState()


def get_session_state() -> EnhanceSessionState:
    return session_state


def set_generating(value: bool) -> None:
    session_state.is_generating = value


def reset_session_state() -> None:
    session_state._reset()
