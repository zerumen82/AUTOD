def suggest_resolution(frames, vram_gb):
    """
    AN-3: Sugerir resolución según VRAM disponible.
    - VRAM > 8 GB: 720p (1280×704) con 97 fotogramas
    - VRAM <= 8 GB: 540p (960x544) con 97 fotogramas
    """
    if vram_gb > 8:
        return "720p (1280×704)", 97
    else:
        return "540p (960x544)", 97


def get_vram_gb():
    """Obtener VRAM disponible en GB usando torch."""
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / (1024**3))
        return 0
    except Exception:
        return 0