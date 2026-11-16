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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PIL import Image

import roop.globals
from roop.processors.Frame_Upscale import Frame_Upscale
from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN

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


def detect_faces_count(image: Image.Image) -> int:
    """Cuenta caras con InsightFace (CPU)."""
    try:
        import roop.face_util as fu
        analyser = fu.get_face_analyser_cpu()
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
    """Denoise bilateral (OpenCV). strength 0-100."""
    if strength <= 0:
        return image
    img = np.array(image.convert("RGB"))
    d = 5 + int(strength / 100 * 15)
    sigma_color = 10 + int(strength / 100 * 40)
    sigma_space = 10 + int(strength / 100 * 40)
    denoised = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    return Image.fromarray(denoised)


def apply_sharpen_cv2(image: Image.Image, strength: float) -> Image.Image:
    """Unsharp mask adaptativo. strength 0-100."""
    if strength <= 0:
        return image
    img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    sigma = 1.0 + strength / 100 * 2.0
    blur = cv2.GaussianBlur(img, (3, 3), sigma)
    amount = 0.3 + strength / 100 * 1.2
    sharpened = img + amount * (img - blur)
    sharpened = np.clip(sharpened, 0, 1)
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
        img = np.where(img < 0.5, img * shadow, img)
    if highlight != 1.0:
        img = np.where(img > 0.5, img * highlight, img)
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
) -> Image.Image:
    """Añade grano fotográfico."""
    if strength <= 0:
        return image
    img = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = img.shape[:2]
    if grain_type == "grey":
        g = np.random.normal(0, strength / 100.0 * 0.04, (h, w))
        grain = np.stack([g, g, g], axis=2)
    elif grain_type == "silver_rich":
        grain = np.random.normal(0, strength / 100.0 * 0.06, (h, w, 3))
        grain[:, :, 0] *= 1.1
        grain[:, :, 2] *= 0.9
    else:
        grain = np.random.normal(0, strength / 100.0 * 0.05, (h, w, 3))
    if grain_size > 1.0:
        gh, gw = max(1, int(h / grain_size)), max(1, int(w / grain_size))
        small_grain = cv2.resize(grain, (gw, gh), interpolation=cv2.INTER_LINEAR)
        grain = cv2.resize(small_grain, (w, h), interpolation=cv2.INTER_LINEAR)
    out = np.clip(img + grain, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Procesadores ONNX (singletons, CUDA EP)
# ---------------------------------------------------------------------------
_UPSCALE_PROCESSORS: Dict[str, Frame_Upscale] = {}
_GFPGAN_PROCESSOR: Optional[Enhance_GFPGAN] = None
_PROCESSORS_LOCK = threading.Lock()

_UPSCALE_SUBTYPES = {"esrganx2": 2, "esrganx4": 4, "lsdirx4": 4}


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
            dev = "cuda" if _cuda_available() else "cpu"
            proc = Frame_Upscale()
            proc.Initialize({"devicename": dev, "subtype": subtype})
            _UPSCALE_PROCESSORS[subtype] = proc
            print(f"[Enhance] Frame_Upscale listo ({subtype}, {dev})", flush=True)
    return proc


def get_gfpgan_processor() -> Enhance_GFPGAN:
    global _GFPGAN_PROCESSOR
    with _PROCESSORS_LOCK:
        if _GFPGAN_PROCESSOR is None:
            dev = "cuda" if _cuda_available() else "cpu"
            _GFPGAN_PROCESSOR = Enhance_GFPGAN()
            _GFPGAN_PROCESSOR.Initialize({"devicename": dev})
            print(f"[Enhance] GFPGAN listo ({dev})", flush=True)
    return _GFPGAN_PROCESSOR


def apply_upscale_onnx(image: Image.Image, factor: int) -> Image.Image:
    """Upscale con Real-ESRGAN por tiles (CUDA). Fallback Lanczos si falla."""
    subtype = "esrganx2" if factor == 2 else "esrganx4"
    try:
        proc = get_upscale_processor(subtype)
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        out_bgr = proc.Run(bgr)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb)
    except Exception as e:
        print(f"[Enhance] Upscale falló ({e}); fallback Lanczos", flush=True)
        return image.resize((image.width * factor, image.height * factor), Image.LANCZOS)


def apply_recover_faces_onnx(image: Image.Image, blend: float) -> Image.Image:
    """Recupera caras con GFPGAN v1.4 (CUDA). blend 0-1."""
    try:
        proc = get_gfpgan_processor()
        import roop.face_util as fu
        analyser = fu.get_face_analyser_cpu()
        if analyser is None:
            return image
        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        faces = analyser.get(bgr) or []
        if not faces:
            return image
        out_bgr = bgr.copy()
        for face in faces:
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
            blend_factor = float(np.clip(det * 0.18 * blend, 0.06, 0.30))
            mask = _create_face_mask(crop, blend_factor)
            if mask is None:
                continue
            m3 = mask[..., None]
            roi = out_bgr[y1:y2, x1:x2].astype(np.float32)
            out_bgr[y1:y2, x1:x2] = (roi * (1.0 - m3) + enhanced.astype(np.float32) * m3).astype(np.uint8)
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

    - Recorte de tonos: se revierte si empeora >0.8 pp (aplica a todos).
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
    if c_after - c_before > 0.008:
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

    denoise = 65 if noise > noise_th else 20
    sharpen = 55 if sharp < sharp_th * 100 else 25
    recover = 80 if faces > 0 else 0
    upscale = 2 if max(w, h) < min_size * 2 else 1
    exposure = 1.3 if brightness < 0.35 else (0.9 if brightness > 0.75 else 1.0)
    shadow = 1.3 if brightness < 0.35 else 1.0
    highlight = 0.7 if brightness > 0.75 else 1.0

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
        "grain_strength": 0,
        "grain_size": 10,
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
) -> Tuple[Image.Image, str, Dict[str, float]]:
    """Procesa una imagen completa con la pila de mejoras."""
    start = time.time()
    notes: List[str] = []
    metrics: Dict[str, Any] = {}

    original = Image.open(input_path).convert("RGB")
    if progress_cb:
        progress_cb({"phase": "cargando", "progress": 0.0, "detail": input_path.name})

    metrics["sharpness_before"] = measure_sharpness_laplacian(original)
    metrics["noise_before"] = measure_noise_laplacian(original)
    metrics["faces"] = detect_faces_count(original)
    metrics["size_before"] = list(original.size)

    current = original

    # 1. Denoise
    ds = float(settings.get("denoise_strength", 0) or 0)
    if ds > 0:
        if progress_cb:
            progress_cb({"phase": "denoise", "progress": 0.1, "detail": f"strength={ds}"})
        current = apply_denoise_cv2(current, ds)
        current, note = quality_gate_check(original, current, "denoise")
        notes.append(note if note else f"denoise {ds:g}")

    # 2. Sharpen
    ss = float(settings.get("sharpen_strength", 0) or 0)
    if ss > 0:
        if progress_cb:
            progress_cb({"phase": "sharpen", "progress": 0.2, "detail": f"strength={ss}"})
        current = apply_sharpen_cv2(current, ss)
        current, note = quality_gate_check(original, current, "sharpen")
        notes.append(note if note else f"sharpen {ss:g}")

    # 3. Recover Faces
    rfs = float(settings.get("recover_faces_strength", 0) or 0)
    if rfs > 0:
        if progress_cb:
            progress_cb({"phase": "recover_faces", "progress": 0.3, "detail": f"blend={rfs}"})
        current = apply_recover_faces_onnx(current, rfs)
        current, note = quality_gate_check(original, current, "recover_faces")
        notes.append(note if note else f"recover_faces {rfs:.0%}")

    # 4. Upscale
    uf = int(settings.get("upscale_factor", 1) or 1)
    if uf > 1:
        if progress_cb:
            progress_cb({"phase": "upscale", "progress": 0.5, "detail": f"{uf}x"})
        current = apply_upscale_onnx(current, uf)
        current, note = quality_gate_check(original, current, "upscale")
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
        current = apply_lighting_color_cv2(current, exp, shd, hlg, temp)
        current, note = quality_gate_check(original, current, "lighting_color")
        notes.append(note if note else "lighting_color")

    # 6. Grano
    gs = float(settings.get("grain_strength", 0) or 0)
    gsize = float(settings.get("grain_size", 10) or 10)
    if gs > 0:
        if progress_cb:
            progress_cb({"phase": "grain", "progress": 0.85, "detail": f"strength={gs}"})
        current = apply_grain_cv2(current, gs, gsize)
        current, note = quality_gate_check(original, current, "grain")
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

            img_settings = settings
            if autopilot:
                try:
                    probe_img = Image.open(path).convert("RGB")
                    img_settings = autopilot_propose(probe_img)
                except Exception as exc:
                    if progress_cb:
                        progress_cb({"phase": "autopilot", "progress": idx / total,
                                     "detail": f"autopilot fallback ({exc})"})

            out_img, notes, metrics = process_single_image(path, img_settings, progress_cb)

            stem = path.stem
            if apply_filters_to_filename:
                filters = []
                if settings.get("denoise_strength", 0) > 0:
                    filters.append("dn")
                if settings.get("sharpen_strength", 0) > 0:
                    filters.append("sh")
                if settings.get("recover_faces_strength", 0) > 0:
                    filters.append("rf")
                if settings.get("upscale_factor", 1) > 1:
                    filters.append(f"up{settings['upscale_factor']}x")
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

            save_kwargs: Dict[str, Any] = {}
            fmt = export_format.lower()
            if fmt in ("jpg", "jpeg"):
                save_kwargs["quality"] = 95
                save_kwargs["subsampling"] = 0
            elif fmt == "png":
                save_kwargs["compress_level"] = 3
            out_img.save(out_path, **save_kwargs)

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
