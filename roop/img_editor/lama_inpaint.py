"""Inpainting determinista con LaMa (ONNX), para la categoría «remove».

LaMa (big-lama, WACV 2022) rellena un hueco a partir de la textura y la
estructura del alrededor. No inventa emblemas ni objetos nuevos: solo
continúa lo que ya estaba. Se usa como alternativa al muestreo difuso a
fuerza 1,00, que arranca de ruido dentro de la máscara y pinta una escena
nueva (medido en el astronauta: manga nueva, bandera y insignia duplicadas).
"""

import os

import numpy as np
import onnxruntime as ort
from PIL import Image

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_MODEL_PATH = os.path.join(_ROOT, "models", "lama_fp32.onnx")

_session = None


def is_available() -> bool:
    return os.path.isfile(_MODEL_PATH)


def _get_session():
    global _session
    if _session is None:
        if not is_available():
            raise FileNotFoundError(_MODEL_PATH)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            _MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"],
        )
    return _session


def lama_inpaint(image: Image.Image, mask: Image.Image) -> Image.Image:
    """Rellena la zona marcada (mask: L, 255 = quitar) sin inventar detalles.

    Devuelve la imagen completa con la zona rellenada y el resto intacto.
    """
    original = image.convert("RGB")
    orig_size = original.size
    img512 = original.resize((512, 512), Image.LANCZOS)
    mask512 = mask.convert("L").resize((512, 512), Image.NEAREST)

    arr = np.asarray(img512, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))[None, ...].astype(np.float32)
    marr = (np.asarray(mask512, dtype=np.float32) / 255.0)[None, None, ...].astype(np.float32)

    sess = _get_session()
    out = sess.run(None, {"image": arr, "mask": marr})[0]
    out = np.clip(out[0], 0.0, 255.0).astype(np.uint8)
    out = np.transpose(out, (1, 2, 0))
    filled = Image.fromarray(out).resize(orig_size, Image.LANCZOS)

    mask_full = mask.convert("L").resize(orig_size, Image.NEAREST)
    return Image.composite(filled, original, mask_full)
