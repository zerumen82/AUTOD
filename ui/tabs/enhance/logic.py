"""Lógica de la pestaña «✨ Mejorar fotos» (local, 100 % local, GPU-first).

Optimización para RTX 3060 Ti 8 GB:
- Los modelos pesados (Real-ESRGAN, GFPGAN) corren en ONNX Runtime + CUDA EP
  con io_binding y tiles (ya lo hacen los procesadores de roop).
- Singletons por modelo: no se recarga la sesión entre imágenes.
- Procesamiento secuencial (1 hilo) para no saturar la VRAM de 8 GB.
- Liberación de caché entre imágenes.
- Gate de calidad por paso: si una mejora empeora la métrica, se revierte.
"""

import json
import time
import threading
import zlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    from PIL import ImageCms
except ImportError:  # littlecms no disponible: se guarda el PPP igualmente
    ImageCms = None

import roop.globals
from roop.processors.Frame_Upscale import Frame_Upscale
from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
_ENHANCE_CONFIG: Optional[Dict[str, Any]] = None
_CONFIG_LOCK = threading.Lock()

# Cancelación cooperativa: el botón «Cancelar» pone la bandera y
# process_batch deja de empezar imágenes nuevas.
_CANCEL_EVENT = threading.Event()


def request_cancel() -> None:
    _CANCEL_EVENT.set()


def clear_cancel() -> None:
    _CANCEL_EVENT.clear()


def is_cancel_requested() -> bool:
    return _CANCEL_EVENT.is_set()


def load_enhance_config() -> Dict[str, Any]:
    """Carga config/enhance/enhance.json (con caché)."""
    global _ENHANCE_CONFIG
    if _ENHANCE_CONFIG is not None:
        return _ENHANCE_CONFIG
    with _CONFIG_LOCK:
        if _ENHANCE_CONFIG is not None:
            return _ENHANCE_CONFIG
        path = Path(__file__).resolve().parents[3] / "config" / "enhance" / "enhance.json"
        try:
            with open(path, "r", encoding="utf-8") as f:
                _ENHANCE_CONFIG = json.load(f)
        except Exception as e:
            print(f"[Enhance] No se pudo cargar config: {e}")
            _ENHANCE_CONFIG = {}
    return _ENHANCE_CONFIG


# ---------------------------------------------------------------------------
# Exportación: perfil de color y resolución desde configuración (no hardcode)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _srgb_icc_profile() -> bytes:
    """Perfil ICC sRGB creado en local (sin ficheros externos ni red)."""
    if ImageCms is None:
        return b""
    return ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()


def _export_save_kwargs(fmt: str) -> Dict[str, Any]:
    """Opciones de guardado del lote: sRGB + PPP y compresión por formato."""
    cfg = load_enhance_config().get("export", {})
    dpi = int(cfg.get("dpi", 254) or 254)
    kwargs: Dict[str, Any] = {"dpi": (dpi, dpi)}
    profile = str(cfg.get("color_profile", "sRGB") or "").strip().casefold()
    if profile == "srgb":
        try:
            icc = _srgb_icc_profile()
        except Exception as exc:
            icc = b""
            print(f"[Enhance] No se pudo crear el perfil sRGB: {exc}", flush=True)
        if icc:
            kwargs["icc_profile"] = icc
    if fmt in ("jpg", "jpeg"):
        kwargs["quality"] = 95
        kwargs["subsampling"] = 0
    elif fmt == "png":
        kwargs["compress_level"] = 3
    return kwargs


# ---------------------------------------------------------------------------
# Utilidades de medición (las mismas que usa el Image Editor)
# ---------------------------------------------------------------------------
def _to_gray_u8(image: Image.Image) -> np.ndarray:
    g = np.array(image.convert("L"), dtype=np.uint8)
    m = max(g.shape[:2])
    if m > 1280:
        s = 1280.0 / m
        g = cv2.resize(g, (max(1, int(g.shape[1] * s)), max(1, int(g.shape[0] * s))),
                       interpolation=cv2.INTER_AREA)
    return g


def measure_sharpness_laplacian(image: Image.Image) -> float:
    """Varianza del Laplaciano (mayor = más nítido)."""
    g = _to_gray_u8(image)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def measure_noise_laplacian(image: Image.Image) -> float:
    """Proxy de ruido: varianza del Laplaciano normalizada 0-1."""
    g = _to_gray_u8(image)
    var = float(cv2.Laplacian(g, cv2.CV_64F).var())
    return var / 255.0


def _get_face_analyser():
    """Analizador facial CUDA si está disponible; si no, CPU."""
    import roop.face_util as fu
    try:
        analyser = fu.get_face_analyser()
        if analyser is not None:
            return analyser
    except Exception:
        pass
    return fu.get_face_analyser_cpu()


def detect_faces_count(image: Image.Image) -> int:
    """Cuenta caras con InsightFace (CUDA si está disponible)."""
    try:
        analyser = _get_face_analyser()
        if analyser is None:
            return 0
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = analyser.get(bgr) or []
        return len(faces)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Mejoras clásicas (OpenCV vectorizado)
# ---------------------------------------------------------------------------
def apply_denoise_cv2(image: Image.Image, strength: float) -> Image.Image:
    """Denoise que conserva el poro y el detalle fino. strength 0-100.

    v5.1437 — un bilateral único a fuerza alta (d=14, sigma=36) borraba el
    poro y el pelo fino y, con el sharpen posterior, dejaba piel de cera
    («muy muñeco»). Ahora se separa croma de luminancia: el croma se suaviza
    (manchas de color) y en la luminancia se devuelve una fracción del detalle
    que el filtro se llevó, así la textura de la piel sobrevive.
    """
    if strength <= 0:
        return image
    s = float(np.clip(strength, 0, 100)) / 100.0
    img = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    luma, a, b = cv2.split(lab)
    lf = luma.astype(np.float32)

    # Croma: aquí sí conviene un suavizado fuerte (ruido de color, blotches).
    d_c = 5 + int(round(9 * s))
    sigma_c = 12 + 38 * s
    space_c = 8 + 18 * s
    a = cv2.bilateralFilter(a, d_c, sigma_c, space_c)
    b = cv2.bilateralFilter(b, d_c, sigma_c, space_c)

    # Luminancia: filtro moderado + reinyección de textura.
    d_l = 5 + int(round(5 * s))
    sigma_l = 8 + 22 * s
    space_l = 6 + 10 * s
    l_dn = cv2.bilateralFilter(lf, d_l, sigma_l, space_l)
    keep = 0.55 - 0.23 * s  # 55% de detalle a fuerza baja, 32% a máxima
    l_out = np.clip(l_dn + keep * (lf - l_dn), 0, 255).astype(np.uint8)

    out = cv2.cvtColor(cv2.merge([l_out, a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def apply_sharpen_cv2(image: Image.Image, strength: float) -> Image.Image:
    """Unsharp mask con umbral y recorte de halo. strength 0-100.

    v5.1437 — el unsharp clásico (amount hasta 1.5) amplificaba el ruido de la
    piel lisa y dejaba bordes duros con halo, que es la mitad del aspecto de
    muñeco. Ahora se descarta el microdetalle por debajo de un umbral (piel,
    cielo) y se recorta el exceso del halo antes de sumar.
    """
    if strength <= 0:
        return image
    s = float(np.clip(strength, 0, 100)) / 100.0
    img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    sigma = 0.8 + 1.2 * s
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    detail = img - blur
    threshold = 0.004 + 0.012 * s
    detail = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
    limit = 0.05 + 0.05 * s
    detail = np.clip(detail, -limit, limit)
    amount = 0.25 + 0.85 * s
    sharpened = np.clip(img + amount * detail, 0, 1)
    return Image.fromarray((sharpened * 255).astype(np.uint8))


def apply_lighting_color_cv2(
    image: Image.Image,
    exposure: float,
    shadow: float,
    highlight: float,
    temperature: float,
) -> Image.Image:
    """Exposición, sombras, luces y temperatura (LAB)."""
    img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    if exposure != 1.0:
        img = img * exposure
    if shadow != 1.0:
        mask = np.clip((0.5 - img) / 0.5, 0.0, 1.0)
        img = img * (1.0 + (shadow - 1.0) * mask)
    if highlight != 1.0:
        mask = np.clip((img - 0.5) / 0.5, 0.0, 1.0)
        img = img * (1.0 + (highlight - 1.0) * mask)
    if temperature != 50:
        lab = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_RGB2LAB)
        temp_factor = (temperature - 50) / 50.0
        lab[:, :, 1] = lab[:, :, 1] - temp_factor * 20
        lab[:, :, 2] = lab[:, :, 2] + temp_factor * 15
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img = np.clip(img, 0, 1)
    return Image.fromarray((img * 255).astype(np.uint8))


def apply_grain_cv2(
    image: Image.Image,
    strength: float,
    grain_size: float,
    grain_type: str = "gaussian",
    seed: Optional[int] = None,
) -> Image.Image:
    """Añade grano fotográfico (determinista si se pasa `seed`)."""
    if strength <= 0:
        return image
    rng = np.random.default_rng(seed)
    img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    if grain_type == "grey":
        g = rng.normal(0, strength / 100.0 * 0.04, (h, w))
        grain = np.stack([g, g, g], axis=2)
    elif grain_type == "silver_rich":
        grain = rng.normal(0, strength / 100.0 * 0.06, (h, w, 3))
        grain[:, :, 0] *= 1.1
        grain[:, :, 2] *= 0.9
    else:
        grain = rng.normal(0, strength / 100.0 * 0.05, (h, w, 3))
    if grain_size > 1.0:
        gh, gw = max(1, int(h / grain_size)), max(1, int(w / grain_size))
        small_grain = cv2.resize(grain, (gw, gh), interpolation=cv2.INTER_LINEAR)
        grain = cv2.resize(small_grain, (w, h), interpolation=cv2.INTER_LINEAR)
    # v5.1437 — el grano pesa más en medios tonos: en luces y sombras apenas
    # se nota, como en una foto real, en vez de ensuciar toda la imagen.
    luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    weight = 1.0 - 0.55 * np.abs(luma - 0.5) * 2.0
    if grain.ndim == 3:
        weight = weight[..., None]
    out = np.clip(img + grain * weight, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def apply_deband_cv2(image: Image.Image, strength: float, seed: Optional[int] = None) -> Image.Image:
    """Suaviza bandas de cuantización en LAB y añade dither fino.

    strength 0-100. No reemplaza un denoise fuerte: elimina la posterización
    de degradados (cielos, sombras) sin destruir el detalle.
    """
    if strength <= 0:
        return image
    rng = np.random.default_rng(seed)
    img = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    luma, a, b = cv2.split(lab)
    d = 3 + int(strength / 100 * 9)
    sigma_color = 8 + int(strength / 100 * 30)
    sigma_space = 8 + int(strength / 100 * 30)
    a = cv2.bilateralFilter(a, d, sigma_color, sigma_space)
    b = cv2.bilateralFilter(b, d, sigma_color, sigma_space)
    noise = rng.normal(0.0, strength / 100 * 1.5, luma.shape).astype(np.float32)
    luma = np.clip(luma.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(cv2.merge([luma, a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)


def apply_vibrance_cv2(image: Image.Image, vibrance: float) -> Image.Image:
    """Ajusta la saturación; el extremo 0 es blanco y negro real."""
    if vibrance <= 0:
        return ImageOps.grayscale(image).convert("RGB")
    if abs(vibrance - 100.0) < 1e-6:
        return image
    img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    # OpenCV usa H en 0-180: la piel cae en ~0-30 y ~165-180, no en 330.
    skin = ((h <= 30.0) | (h >= 165.0)).astype(np.float32)
    factor = 1.0 + (vibrance - 100.0) / 100.0 * (1.0 - s)
    factor = 1.0 + (factor - 1.0) * (1.0 - 0.6 * skin)
    hsv[:, :, 1] = np.clip(s * factor, 0.0, 1.0)
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return Image.fromarray((np.clip(out, 0.0, 1.0) * 255.0).astype(np.uint8))


# ---------------------------------------------------------------------------
# Procesadores ONNX (singletons, CUDA EP)
# ---------------------------------------------------------------------------
_UPSCALE_PROCESSORS: Dict[str, Frame_Upscale] = {}
_FACE_PROCESSORS: Dict[str, Any] = {}
_PROCESSORS_LOCK = threading.Lock()

_UPSCALE_SUBTYPES = {"esrganx2": 2, "esrganx4": 4, "lsdirx4": 4}

_FACE_MODEL_ALIASES = {
    "gfpgan": "GFPGAN",
    "codeformer": "CodeFormer",
    "restoreformer": "RestoreFormer++",
    "restoreformer++": "RestoreFormer++",
    "restoreformerpp": "RestoreFormer++",
    "gpen": "GPEN",
}


def _normalize_face_model(name: str) -> str:
    key = str(name or "GFPGAN").strip().lower()
    return _FACE_MODEL_ALIASES.get(key, "GFPGAN")


def _ensure_execution_providers() -> None:
    if not getattr(roop.globals, "execution_providers", None):
        import onnxruntime
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            roop.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            roop.globals.execution_providers = ["CPUExecutionProvider"]


def _cuda_available() -> bool:
    return "CUDAExecutionProvider" in getattr(roop.globals, "execution_providers", [])


def get_upscale_processor(subtype: str = "esrganx4") -> Frame_Upscale:
    """Devuelve (y cachea) un Frame_Upscale para el subtype pedido."""
    global _UPSCALE_PROCESSORS
    if subtype not in _UPSCALE_SUBTYPES:
        subtype = "esrganx4"
    with _PROCESSORS_LOCK:
        proc = _UPSCALE_PROCESSORS.get(subtype)
        if proc is None:
            _ensure_execution_providers()
            dev = "cuda" if _cuda_available() else "cpu"
            proc = Frame_Upscale()
            proc.Initialize({"devicename": dev, "subtype": subtype})
            _UPSCALE_PROCESSORS[subtype] = proc
            print(f"[Enhance] Frame_Upscale listo ({subtype}, {dev})", flush=True)
    return proc


def get_face_processor(name: str = "GFPGAN"):
    """Devuelve un procesador de caras (GFPGAN/CodeFormer/RestoreFormer++/GPEN).

    Solo hay uno activo: al cambiar de modelo se libera el anterior para no
    acumular VRAM en una RTX 3060 Ti de 8 GB.
    """
    model = _normalize_face_model(name)
    with _PROCESSORS_LOCK:
        active = _FACE_PROCESSORS.get("active")
        if active and active[0] == model:
            return active[1]
        if active:
            try:
                active[1].Release()
            except Exception:
                pass
            _FACE_PROCESSORS.pop("active", None)
        _ensure_execution_providers()
        dev = "cuda" if _cuda_available() else "cpu"
        if model == "GFPGAN":
            proc = Enhance_GFPGAN()
        elif model == "CodeFormer":
            from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
            proc = Enhance_CodeFormer()
        elif model == "RestoreFormer++":
            from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
            proc = Enhance_RestoreFormerPPlus()
        else:
            from roop.processors.Enhance_GPEN import Enhance_GPEN
            proc = Enhance_GPEN()
        proc.Initialize({"devicename": dev})
        _FACE_PROCESSORS["active"] = (model, proc)
        print(f"[Enhance] Motor de caras listo ({model}, {dev})", flush=True)
        return proc


def get_gfpgan_processor():
    """Compatibilidad: devuelve el motor GFPGAN."""
    return get_face_processor("GFPGAN")


def apply_upscale_onnx(image: Image.Image, factor: int, progress_cb=None, resample_mix: float = 0.25) -> Image.Image:
    """Upscale con Real-ESRGAN por tiles (CUDA). Fallback Lanczos si falla.

    v5.1438 — Real-ESRGAN sintetiza detalle pero se come el grano real de la
    foto: medido a 2x nativo, la barba y el poro desaparecen y la piel queda
    con brillo de plástico. Se conserva una fracción (``resample_mix``) del
    escalado Lanczos del original, que sí trae esa textura, manteniendo los
    bordes del modelo. Sin mezcla (0.0) se comporta como antes.
    """
    if factor == 6:
        out = apply_upscale_onnx(image, 4, resample_mix=resample_mix)
        return out.resize((out.width * 3 // 2, out.height * 3 // 2), Image.LANCZOS)
    subtype = "esrganx2" if factor == 2 else "esrganx4"
    try:
        proc = get_upscale_processor(subtype)
        if progress_cb:
            progress_cb({"phase": "upscale", "progress": 0.5, "detail": f"{factor}x"})
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        out_bgr = proc.Run(bgr)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        if resample_mix > 0:
            # OJO: bgr está en BGR y out_rgb en RGB; mezclar sin convertir
            # intercambia rojo y azul (la piel sale azul).
            plain = cv2.resize(bgr, (out_rgb.shape[1], out_rgb.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            plain = cv2.cvtColor(plain, cv2.COLOR_BGR2RGB)
            plain_f = plain.astype(np.float32)
            mixed = out_rgb.astype(np.float32) * (1.0 - resample_mix) + plain_f * resample_mix
            # El modelo también aclara y desplaza el color (medido: +1.5 pp de
            # blancos). Se le devuelve la luz/color a gran escala del escalado
            # limpio y se queda con su detalle: sin tinte ni velo.
            tone_sigma = float(np.clip(0.004 * max(out_rgb.shape[:2]), 3.0, 12.0))
            mixed = mixed + cv2.GaussianBlur(plain_f, (0, 0), tone_sigma) - cv2.GaussianBlur(mixed, (0, 0), tone_sigma)
            out_rgb = np.clip(mixed, 0, 255).astype(np.uint8)
        return Image.fromarray(out_rgb)
    except Exception as e:
        print(f"[Enhance] Upscale falló ({e}); fallback Lanczos", flush=True)
        return image.resize((image.width * factor, image.height * factor), Image.LANCZOS)


def apply_recover_faces_onnx(
    image: Image.Image,
    blend: float,
    model_name: str = "GFPGAN",
    progress_cb=None,
    cancel_check=None,
    tone_match: bool = True,
) -> Image.Image:
    """Recupera caras con el motor elegido (GFPGAN/CodeFormer/RestoreFormer++/GPEN).

    v5.1438 — con ``tone_match`` se conserva la luz y el color del original
    dentro de la cara y solo se toma del modelo la estructura: el modelo ya no
    puede imponer su propio brillo (en caras con otra iluminación aclara la
    piel y los dientes, que es la otra mitad del aspecto «muñeco»). En la
    muestra del probe no había blanqueo medible; esto lo previene.
    """
    try:
        proc = get_face_processor(model_name)
        analyser = _get_face_analyser()
        if analyser is None:
            return image
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = analyser.get(bgr) or []
        if not faces:
            return image
        out_bgr = bgr.copy()
        total_faces = len(faces)
        for fi, face in enumerate(faces):
            if cancel_check and cancel_check():
                return image
            if progress_cb:
                progress_cb({"phase": "recover_faces", "progress": 0.3 + 0.1 * (fi / max(1, total_faces)),
                            "detail": f"{model_name} cara {fi + 1}/{total_faces}"})
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            margin = int(max(x2 - x1, y2 - y1) * 0.3)
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(bgr.shape[1], x2 + margin), min(bgr.shape[0], y2 + margin)
            crop = out_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            enhanced, _sf = proc.Run(None, None, crop)
            if enhanced.shape[:2] != crop.shape[:2]:
                enhanced = cv2.resize(enhanced, (crop.shape[1], crop.shape[0]),
                                      interpolation=cv2.INTER_LANCZOS4)
            det = float(getattr(face, "det_score", 0.9) or 0.9)
            blend_factor = float(np.clip(det * 0.35 * blend, 0.08, 0.50))
            mask = _create_face_mask(crop, blend_factor)
            if mask is None:
                continue
            m3 = mask[..., None]
            roi = out_bgr[y1:y2, x1:x2].astype(np.float32)
            blended = roi * (1.0 - m3) + enhanced.astype(np.float32) * m3
            # v5.1437 — el modelo repinta la piel lisa (aspecto de muñeco). Se
            # devuelve el poro/microdetalle del original dentro de la máscara.
            texture = roi - cv2.GaussianBlur(roi, (0, 0), 1.0)
            blended = blended + texture * m3 * 0.6
            if tone_match:
                # Intercambio de frecuencias: el modelo aporta detalle, el
                # original aporta la luz y el color a gran escala.
                tone_sigma = max(4.0, 0.04 * min(crop.shape[:2]))
                low_blend = cv2.GaussianBlur(blended, (0, 0), tone_sigma)
                low_roi = cv2.GaussianBlur(roi, (0, 0), tone_sigma)
                blended = blended - low_blend + low_roi
            out_bgr[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)
    except Exception as e:
        print(f"[Enhance] Recover faces falló ({e})", flush=True)
        return image


def _create_face_mask(crop_bgr: np.ndarray, blend: float) -> Optional[np.ndarray]:
    """Máscara facial: MediaPipe si está disponible, si no elipse suave."""
    try:
        from roop.img_editor.top_tier_finish import _face_mesh_mask
        mask = _face_mesh_mask(crop_bgr, blend)
        if mask is not None:
            return mask
    except Exception:
        pass
    fh, fw = crop_bgr.shape[:2]
    mask = np.zeros((fh, fw), dtype=np.float32)
    cv2.ellipse(mask, (fw // 2, fh // 2),
                (max(8, int(fw * 0.46)), max(8, int(fh * 0.52))), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), max(6.0, min(fh, fw) * 0.08))
    return mask * blend


# ---------------------------------------------------------------------------
# Gate de calidad
# ---------------------------------------------------------------------------
# Pasos con guardia de nitidez (caída >25% = desastre, p. ej. marco negro).
_SHARP_GUARD_STEPS = {"sharpen", "recover_faces", "lighting_color"}
# Pasos con guardia de ruido: solo se revierte si el ruido AUMENTA.
_NOISE_GUARD_STEPS = {"denoise"}


def quality_gate_check(before: Image.Image, after: Image.Image, name: str) -> Tuple[Image.Image, str]:
    """Gate de calidad por paso. Revierte solo daño real, no mejoras legítimas.

    - Recorte de tonos: se revierte si empeora >0.8 pp (2.5 pp en lighting_color,
      que sube contraste a propósito y no debe descartarse por un recorte menor).
    - Nitidez: solo en pasos que no cambian escala ni suavizan por diseño;
      se revierte si cae >25% (desastre). La comparación es a la misma escala.
    - Ruido: en denoise, se revierte si el ruido aumenta >15%.
    - upscale / grain: solo guardia de recorte (cambian escala / añaden ruido
      por diseño, respectivamente).
    """
    def clip(img):
        g = _to_gray_u8(img)
        return float((g <= 4).mean() + (g >= 251).mean())

    if after.size != before.size:
        after_cmp = after.resize(before.size, Image.LANCZOS)
    else:
        after_cmp = after

    c_before, c_after = clip(before), clip(after_cmp)
    # v5.1438 — el upscale no puede caer por el recorte del propio modelo: en
    # fotos con fondo claro (cielo, pared blanca) ESRGAN sube ~1.5 pp de
    # blancos y el paso se revertía EN SILENCIO, así que «up2x» no escalaba.
    if name == "upscale":
        clip_limit = 0.04
    elif name == "lighting_color":
        clip_limit = 0.025
    else:
        clip_limit = 0.008
    if c_after - c_before > clip_limit:
        return before, f"{name} revertido (recorte de tonos +{c_after - c_before:.1%})"

    if name in _SHARP_GUARD_STEPS:
        s_before, s_after = measure_sharpness_laplacian(before), measure_sharpness_laplacian(after_cmp)
        if s_before > 25.0 and (1.0 - s_after / s_before) > 0.25:
            return before, f"{name} revertido (nitidez -{1.0 - s_after / s_before:.0%})"

    if name in _NOISE_GUARD_STEPS:
        n_before, n_after = measure_noise_laplacian(before), measure_noise_laplacian(after_cmp)
        if n_before > 0.01 and (n_after / n_before - 1.0) > 0.15:
            return before, f"{name} revertido (ruido +{n_after / n_before - 1.0:.0%})"

    return after, ""


# ---------------------------------------------------------------------------
# Autopilot: mide y propone la pila (como IA local)
# ---------------------------------------------------------------------------
def autopilot_propose(image: Image.Image) -> Dict[str, Any]:
    """Mide la foto y devuelve los ajustes propuestos + explicación."""
    noise = measure_noise_laplacian(image)
    sharp = measure_sharpness_laplacian(image)
    faces = detect_faces_count(image)
    w, h = image.size
    gray = _to_gray_u8(image)
    brightness = float(gray.mean() / 255.0)

    cfg = load_enhance_config().get("autopilot", {})
    noise_th = float(cfg.get("noise_threshold", 0.15))
    sharp_th = float(cfg.get("sharpness_threshold", 0.10))
    min_size = int(cfg.get("min_image_size", 640))

    # v5.1437 — menos denoise/sharpen y recuperación de caras más suave: la
    # receta anterior (65/55/80) era la que dejaba la piel de cera.
    denoise = 45 if noise > noise_th else 15
    sharpen = 35 if sharp < sharp_th * 100 else 18
    recover = 55 if faces > 0 else 0
    upscale = 2 if max(w, h) < min_size * 2 else 1
    exposure = 1.3 if brightness < 0.35 else (0.9 if brightness > 0.75 else 1.0)
    shadow = 1.3 if brightness < 0.35 else 1.0
    highlight = 0.7 if brightness > 0.75 else 1.0
    deband = 30 if noise > noise_th else 12
    vibrance = 103 if 0.35 <= brightness <= 0.75 else 100
    face_model = "GFPGAN"

    reasons = []
    if noise > noise_th:
        reasons.append(f"ruido σ={noise:.3f} → denoise {denoise}")
    if sharp < sharp_th * 100:
        reasons.append(f"nitidez={sharp:.1f} → sharpen {sharpen}")
    if faces > 0:
        reasons.append(f"{faces} cara(s) → recover faces {recover}")
    if upscale > 1:
        reasons.append(f"tamaño {w}×{h} → upscale {upscale}x")
    if brightness < 0.35:
        reasons.append(f"oscura (L={brightness:.2f}) → exposición {exposure}")
    elif brightness > 0.75:
        reasons.append(f"quemada (L={brightness:.2f}) → exposición {exposure}")
    if not reasons:
        reasons.append("foto dentro de rangos normales → ajustes suaves")

    return {
        "denoise_strength": denoise,
        "sharpen_strength": sharpen,
        "recover_faces_strength": recover / 100.0,
        "upscale_factor": upscale,
        "exposure": exposure,
        "shadow": shadow,
        "highlight": highlight,
        "temperature": 50,
        "grain_strength": 6,
        "grain_size": 2,
        "deband_strength": deband,
        "vibrance": vibrance,
        "face_model": face_model,
        "metrics": {
            "noise": noise,
            "sharpness": sharp,
            "faces": faces,
            "brightness": brightness,
            "size": [w, h],
        },
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Pipeline por imagen
# ---------------------------------------------------------------------------
def process_single_image(
    input_path: Path,
    settings: Dict[str, Any],
    progress_cb: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Tuple[Image.Image, str, Dict[str, float]]:
    """Procesa una imagen completa con la pila de mejoras."""
    start = time.time()
    notes: List[str] = []
    metrics: Dict[str, Any] = {}

    with Image.open(input_path) as _img:
        original = ImageOps.exif_transpose(_img).convert("RGB")
    seed = zlib.crc32(original.tobytes()) & 0xFFFFFFFF
    if progress_cb:
        progress_cb({"phase": "cargando", "progress": 0.0, "detail": input_path.name})

    metrics["sharpness_before"] = measure_sharpness_laplacian(original)
    metrics["noise_before"] = measure_noise_laplacian(original)
    rfs_check = float(settings.get("recover_faces_strength", 0) or 0)
    metrics["faces"] = detect_faces_count(original) if rfs_check > 0 else 0
    metrics["size_before"] = list(original.size)

    current = original

    # 1. Denoise
    ds = float(settings.get("denoise_strength", 0) or 0)
    if ds > 0:
        if progress_cb:
            progress_cb({"phase": "denoise", "progress": 0.1, "detail": f"strength={ds}"})
        step_before = current
        current = apply_denoise_cv2(current, ds)
        current, note = quality_gate_check(step_before, current, "denoise")
        notes.append(note if note else f"denoise {ds:g}")

    # 1b. Deband (posterización de degradados)
    db = float(settings.get("deband_strength", 0) or 0)
    if db > 0:
        if progress_cb:
            progress_cb({"phase": "deband", "progress": 0.15, "detail": f"strength={db}"})
        step_before = current
        current = apply_deband_cv2(current, db, seed)
        current, note = quality_gate_check(step_before, current, "deband")
        notes.append(note if note else f"deband {db:g}")

    # 2. Sharpen
    # v5.1438 — si además se escala, Real-ESRGAN ya aporta microcontraste:
    # afinar antes duplica el borde (medido: banda 2.0× frente a un escalado
    # limpio) y saca halo. Se afina al 60% para compensarlo.
    uf = int(settings.get("upscale_factor", 1) or 1)
    ss = float(settings.get("sharpen_strength", 0) or 0) * (0.6 if uf > 1 else 1.0)
    if ss > 0:
        if progress_cb:
            progress_cb({"phase": "sharpen", "progress": 0.2, "detail": f"strength={ss}"})
        step_before = current
        current = apply_sharpen_cv2(current, ss)
        current, note = quality_gate_check(step_before, current, "sharpen")
        notes.append(note if note else f"sharpen {ss:g}")

    # 3. Recover Faces
    rfs = float(settings.get("recover_faces_strength", 0) or 0)
    model_name = str(settings.get("face_model", "GFPGAN") or "GFPGAN")
    if rfs > 0:
        if progress_cb:
            progress_cb({"phase": "recover_faces", "progress": 0.3, "detail": f"{model_name} blend={rfs}"})
        step_before = current
        current = apply_recover_faces_onnx(current, rfs, model_name, progress_cb=progress_cb, cancel_check=cancel_check)
        current, note = quality_gate_check(step_before, current, "recover_faces")
        notes.append(note if note else f"recover_faces {model_name} {rfs:.0%}")

    # 4. Upscale
    if uf > 1:
        if progress_cb:
            progress_cb({"phase": "upscale", "progress": 0.5, "detail": f"{uf}x"})
        step_before = current
        current = apply_upscale_onnx(current, uf, progress_cb=progress_cb)
        current, note = quality_gate_check(step_before, current, "upscale")
        notes.append(note if note else f"upscale {uf}x")

    # 5. Luz y color
    exp = float(settings.get("exposure", 1.0) or 1.0)
    shd = float(settings.get("shadow", 1.0) or 1.0)
    hlg = float(settings.get("highlight", 1.0) or 1.0)
    temp = float(settings.get("temperature", 50) or 50)
    if exp != 1.0 or shd != 1.0 or hlg != 1.0 or temp != 50:
        if progress_cb:
            progress_cb({"phase": "lighting_color", "progress": 0.7,
                         "detail": f"exp={exp} shd={shd} hlg={hlg} temp={temp}"})
        step_before = current
        current = apply_lighting_color_cv2(current, exp, shd, hlg, temp)
        current, note = quality_gate_check(step_before, current, "lighting_color")
        notes.append(note if note else "lighting_color")

    # 5b. Vibrance (color)
    vib = float(settings.get("vibrance", 100))
    if abs(vib - 100.0) > 1e-6:
        if progress_cb:
            progress_cb({"phase": "vibrance", "progress": 0.75, "detail": f"vibrance={vib:g}"})
        step_before = current
        current = apply_vibrance_cv2(current, vib)
        current, note = quality_gate_check(step_before, current, "vibrance")
        notes.append(note if note else f"color {vib:g}")

    # 6. Grano
    gs = float(settings.get("grain_strength", 0) or 0)
    gsize = float(settings.get("grain_size", 2) or 2)
    if gs > 0:
        if progress_cb:
            progress_cb({"phase": "grain", "progress": 0.85, "detail": f"strength={gs}"})
        step_before = current
        current = apply_grain_cv2(
            current, gs, gsize,
            grain_type="grey" if vib <= 0 else "gaussian", seed=seed + 1,
        )
        current, note = quality_gate_check(step_before, current, "grain")
        notes.append(note if note else f"grain {gs:g}")

    metrics["sharpness_after"] = measure_sharpness_laplacian(current)
    metrics["noise_after"] = measure_noise_laplacian(current)
    metrics["size_after"] = list(current.size)
    metrics["elapsed"] = time.time() - start

    # Liberar VRAM entre imágenes
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return current, " + ".join(notes), metrics


# ---------------------------------------------------------------------------
# Procesamiento por lotes
# ---------------------------------------------------------------------------
def process_batch(
    file_paths: List[Path],
    settings: Dict[str, Any],
    output_dir: Path,
    export_format: str,
    apply_filters_to_filename: bool,
    allow_overwrite: bool,
    progress_cb: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
    autopilot: bool = False,
    settings_by_path: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Procesa una lista de archivos de forma secuencial (VRAM segura).

    Si autopilot=True, cada imagen se mide individualmente y la pila propuesta
    sustituye a «settings» (como el modo automático de IA local).
    """
    results: List[Dict[str, Any]] = []
    total = len(file_paths)
    output_dir.mkdir(parents=True, exist_ok=True)

    def process_one(idx: int, path: Path):
        if cancel_check and cancel_check():
            return None
        try:
            if progress_cb:
                progress_cb({"phase": "batch", "progress": idx / total,
                             "detail": f"{path.name} ({idx + 1}/{total})"})

            img_settings = (settings_by_path or {}).get(str(path), settings)
            if autopilot:
                try:
                    with Image.open(path) as _img:
                        probe_img = ImageOps.exif_transpose(_img).convert("RGB")
                    img_settings = autopilot_propose(probe_img)
                except Exception as exc:
                    if progress_cb:
                        progress_cb({"phase": "autopilot", "progress": idx / total,
                                     "detail": f"autopilot fallback ({exc})"})

            def _cb(payload):
                if not progress_cb:
                    return
                p = dict(payload or {})
                phase = str(p.get("phase", ""))
                detail = str(p.get("detail", ""))
                if phase not in ("batch",):
                    try:
                        frac = float(p.get("progress", 0.0))
                    except (TypeError, ValueError):
                        frac = 0.0
                    p["progress"] = (idx + max(0.0, min(1.0, frac))) / max(1, total)
                    p["detail"] = f"{path.name} ({idx + 1}/{total}) · {detail}".strip(" ·")
                progress_cb(p)

            out_img, notes, metrics = process_single_image(path, img_settings, _cb, cancel_check=cancel_check)

            stem = path.stem
            if apply_filters_to_filename:
                filters = []
                if img_settings.get("denoise_strength", 0) > 0:
                    filters.append("dn")
                if img_settings.get("sharpen_strength", 0) > 0:
                    filters.append("sh")
                if img_settings.get("recover_faces_strength", 0) > 0:
                    filters.append("rf")
                if img_settings.get("upscale_factor", 1) > 1:
                    filters.append(f"up{img_settings['upscale_factor']}x")
                if img_settings.get("deband_strength", 0) > 0:
                    filters.append("db")
                if abs(float(img_settings.get("vibrance", 100)) - 100.0) > 1e-6:
                    filters.append("co")
                if filters:
                    stem += "_" + "-".join(filters)

            ext = f".{export_format.lower()}"
            base_stem = stem
            n = 0
            while True:
                cand = output_dir / f"{stem}{ext}"
                if cand not in used_outputs and (allow_overwrite or not cand.exists()):
                    out_path = cand
                    break
                n += 1
                stem = f"{base_stem}_{n}"
            used_outputs.add(out_path)

            fmt = export_format.lower()
            out_img.save(out_path, **_export_save_kwargs(fmt))

            return {
                "input": str(path),
                "output": str(out_path),
                "notes": notes,
                "metrics": metrics,
                "success": True,
            }
        except Exception as e:
            return {
                "input": str(path),
                "output": None,
                "error": str(e),
                "success": False,
            }

    max_workers = 1  # Secuencial por VRAM (RTX 3060 Ti 8 GB)
    used_outputs: set = set()  # Evita que dos entradas con mismo stem se pisen
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, i, p): i for i, p in enumerate(file_paths)}
        for future in as_completed(futures):
            if cancel_check and cancel_check():
                break
            res = future.result()
            if res:
                results.append(res)
                if progress_cb:
                    progress_cb({"phase": "batch", "progress": len(results) / total,
                                 "detail": f"Completado {len(results)}/{total}"})

    return results


# Persistencia de la carpeta de destino entre sesiones.
_PERSISTED_OUTPUT_FILE = Path(__file__).resolve().parents[3] / "config" / "enhance" / "last_output_dir.txt"


def load_persisted_output_dir() -> str:
    try:
        text = _PERSISTED_OUTPUT_FILE.read_text(encoding="utf-8").strip()
        if text:
            p = Path(text)
            if p.is_dir():
                return str(p)
    except Exception:
        pass
    try:
        from roop.output_paths import get_enhance_output_dir
        return get_enhance_output_dir()
    except Exception:
        return str(Path(__file__).resolve().parents[3] / 'enhance')


def save_persisted_output_dir(path: str) -> None:
    try:
        _PERSISTED_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PERSISTED_OUTPUT_FILE.write_text(str(path or "").strip(), encoding="utf-8")
    except Exception:
        pass
