"""Verificación medida de una edición renderizada (P6, P7, P8, P9).

Ninguna comprobación de este módulo usa listas de palabras, prendas o tipos
de edición, ni la opinión de un VLM: todo se mide sobre píxeles, landmarks
faciales o segmentación. Los umbrales viven en
`config/editor_verification.json` (regla 1: nada de valores mágicos en el
código).

Motivo (medido el 2026-11-15):
- P9: `_camera_change_score` devolvía 0.00 en un cambio de cámara REAL porque
  la malla 468 no encontró cara en la salida y eso se confundía con «no
  cambió». Aquí la ausencia de malla NO se confunde con «sin cambio»: se
  informa de la firma de encuadre (posición, tamaño y roll de la cara).
- P6/P8: una persona o un objeto «añadidos» no se comprobaban; solo se
  miraba si la foto había cambiado.
- P7: la verificación era global («cambió algo»); no exigía que el cambio
  estuviera en la zona medida.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image


CONFIG_NAME = "editor_verification.json"

DEFAULT_SETTINGS: Dict[str, Any] = {
    "camera": {
        # Desplazamiento vertical del centro de la cara dentro del encuadre.
        "min_face_center_shift": 0.012,
        # Cambio relativo del tamaño de la cara (encuadre más cercano/lejano).
        "min_face_area_ratio_change": 0.15,
        # Cambio de la proporción frente/cara (ángulo alto/bajo).
        "min_forehead_ratio_change": 0.02,
        # Normalizador histórico de la distancia de landmarks.
        "landmark_shift_reference": 0.1,
        # Distancia de landmarks que se considera un cambio real.
        "landmark_score_threshold": 0.15,
        # Umbral del score combinado para dar el cambio por medido.
        "min_change_score": 0.35,
    },
    "addition": {
        # Ganancia mínima de presencia del concepto pedido (CLIPSeg).
        "min_presence_gain": 0.010,
        # Ganancia mínima de área binaria del concepto pedido.
        "min_area_gain": 0.004,
        # Una persona más debe medirse como una persona más.
        "min_person_count_gain": 1,
        # O como una superficie de persona mayor (persona pequeña o lejana).
        "min_person_area_gain": 0.015,
        # Evidencia nueva del concepto pedido (píxeles donde el segmentador
        # lo marca en la salida y no en la entrada). Calibrado el
        # 2026-11-15: objeto añadido 0,00248 frente a 0,00011 (caso sin
        # objeto) y 0,0 (pose).
        "min_fresh_area": 0.001,
    },
    "zone": {
        # Fracción mínima de la zona medida que debe cambiar.
        "min_zone_change": 0.02,
        # Fracción máxima que puede cambiar FUERA de la zona medida.
        "max_outside_change": 0.12,
    },
}


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_verification_settings() -> Dict[str, Any]:
    """Umbrales de verificación (configuración; nunca constantes en el código)."""
    settings = json.loads(json.dumps(DEFAULT_SETTINGS))
    path = os.path.join(_project_root(), "config", CONFIG_NAME)
    try:
        with open(path, encoding="utf-8") as stream:
            stored = json.load(stream)
    except (OSError, ValueError):
        return settings
    if not isinstance(stored, dict):
        return settings
    for section, values in stored.items():
        if isinstance(values, dict) and isinstance(settings.get(section), dict):
            settings[section].update(values)
    return settings


# ---------------------------------------------------------------------------
# Presencia de un concepto (CLIPSeg)
# ---------------------------------------------------------------------------

def concept_presence(masker, image: Image.Image, query: str) -> Optional[Dict[str, float]]:
    """Mide cuánto de la foto es el concepto pedido.

    `presence` es la media del mapa de probabilidad en su parte alta (2 % de
    píxeles): responde «¿hay esto en la foto?» sin depender de un umbral
    binario. `area` es la fracción de píxeles por encima del umbral del
    segmentador.
    """
    if masker is None or not query or not str(query).strip():
        return None
    try:
        heat = masker.probability(image, str(query).strip())
    except Exception:
        return None
    if heat is None or getattr(heat, "size", 0) == 0:
        return None
    try:
        values = np.asarray(heat, dtype=np.float32).ravel()
    except Exception:
        return None
    if values.size == 0:
        return None
    top = max(1, int(values.size * 0.02))
    highest = np.partition(values, -top)[-top:]
    return {
        "presence": float(highest.mean()),
        "area": float((values > 0.4).mean()),
    }


def person_measurement(masker, image: Image.Image,
                       min_area_ratio: float = 0.005) -> Optional[Dict[str, float]]:
    """Cuántas personas se miden y qué superficie ocupan (segmentación)."""
    if masker is None:
        return None
    try:
        heat = masker.probability(image, "person")
    except Exception:
        return None
    if heat is None or getattr(heat, "size", 0) == 0:
        return None
    try:
        import cv2

        binary = (np.asarray(heat, dtype=np.float32) > 0.4).astype(np.uint8)
        total = float(binary.size)
        if total <= 0:
            return None
        if binary.max() == 0:
            return {"count": 0, "area": 0.0}
        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
        people = 0
        area = 0.0
        for index in range(1, count):
            component = float(stats[index, cv2.CC_STAT_AREA]) / total
            if component >= min_area_ratio:
                people += 1
                area += component
        return {"count": people, "area": round(area, 4)}
    except Exception:
        return None


def person_count(masker, image: Image.Image, min_area_ratio: float = 0.01) -> Optional[int]:
    """Cuántas personas se miden en la foto (segmentación, sin palabras)."""
    measured = person_measurement(masker, image, min_area_ratio)
    if measured is None:
        return None
    return int(measured["count"])


def new_concept_evidence(masker, before: Image.Image, after: Image.Image,
                          query: str) -> Dict[str, Any]:
    """Evidencia NUEVA de un concepto: aparece en la salida y no estaba antes.

    Medido el 2026-11-15: la presencia global del concepto NO sirve con una
    ruta de fotograma completo (bajó en los tres casos reales medidos:
    personas -0,044; objeto -0,049; pose -0,172). Lo que discrimina es la
    evidencia nueva: píxeles donde el segmentador marca el concepto en la
    salida y no en la entrada.

    `available: False` cuando el segmentador no ve el concepto con confianza
    (pico < 0.4): entonces la señal no dice nada y no se puede afirmar ni
    negar que el objeto esté.
    """
    report: Dict[str, Any] = {"available": False}
    if masker is None or not query:
        report["reason"] = "sin segmentador o sin consulta"
        return report
    try:
        heat_before = masker.probability(before, query)
        heat_after = masker.probability(after, query)
    except Exception:
        report["reason"] = "error al medir"
        return report
    if heat_before is None or heat_after is None:
        report["reason"] = "mapa no disponible"
        return report
    before_map = np.asarray(heat_before, dtype=np.float32)
    after_map = np.asarray(heat_after, dtype=np.float32)
    if before_map.shape != after_map.shape:
        after_map = np.asarray(
            Image.fromarray((after_map * 255).astype(np.uint8)).resize(
                (before_map.shape[1], before_map.shape[0])),
            dtype=np.float32,
        ) / 255.0
    fresh = (after_map > 0.4) & (before_map < 0.2)
    total = float(fresh.size)
    report.update({
        "available": True,
        "fresh_area": round(float(fresh.sum()) / max(total, 1.0), 6),
        "peak_after": round(float(after_map.max()), 4),
        "peak_before": round(float(before_map.max()), 4),
        "query": str(query),
    })
    return report


# ---------------------------------------------------------------------------
# Firma de encuadre (P9)
# ---------------------------------------------------------------------------

# Índices de MediaPipe Face Mesh (468).
_FOREHEAD_TOP = 10
_BROW_LEFT, _BROW_RIGHT = 105, 334
_CHIN = 152
_MOUTH_LEFT, _MOUTH_RIGHT = 13, 14


def _landmarks(image: Image.Image):
    """Malla 468 de la cara (None si no hay cara detectable)."""
    try:
        from roop.img_editor.top_tier_finish import _get_face_landmarks

        return _get_face_landmarks(image)
    except Exception:
        return None


def face_framing(image: Image.Image) -> Optional[Dict[str, float]]:
    """Firma de encuadre de la cara: posición, tamaño, roll y proporción frente."""
    points = _landmarks(image)
    if points is None:
        return None
    try:
        forehead = points[_FOREHEAD_TOP]
        brow = points[[_BROW_LEFT, _BROW_RIGHT]].mean(axis=0)
        chin = points[_CHIN]
        mouth = points[[_MOUTH_LEFT, _MOUTH_RIGHT]].mean(axis=0)
        total = float(abs(chin[1] - forehead[1]))
        brow_dx = float(points[_BROW_LEFT][0] - points[_BROW_RIGHT][0])
        brow_dy = float(points[_BROW_LEFT][1] - points[_BROW_RIGHT][1])
        return {
            "center_y": float(points[:, 1].mean()),
            "area": float(np.abs(brow_dx * (chin[1] - brow[1]) - brow_dy * (chin[0] - brow[0]))),
            "roll": float(np.degrees(np.arctan2(brow_dy, brow_dx))),
            "forehead_ratio": float(abs(brow[1] - forehead[1]) / max(total, 1e-6)),
            "upper_lower_ratio": float(
                abs(chin[1] - brow[1]) / max(abs(chin[1] - mouth[1]), 1e-6),
            ),
        }
    except Exception:
        return None


def framing_change(before: Image.Image, after: Image.Image) -> Dict[str, Any]:
    """Mide el cambio de encuadre/ángulo entre dos imágenes.

    Devuelve un dict con la medida y su veredicto. Si la malla facial no está
    disponible en una de las dos imágenes, el resultado es `medible: False`:
    eso NO es «no cambió» (el fallo medido de P9), es «no se pudo medir».
    """
    settings = load_verification_settings()["camera"]
    report: Dict[str, Any] = {"medible": False, "score": 0.0, "reason": ""}
    if before is None or after is None:
        report["reason"] = "faltan imágenes"
        return report
    if before.size != after.size:
        after = after.convert("RGB").resize(before.size, Image.Resampling.LANCZOS)
    before_frame = face_framing(before)
    after_frame = face_framing(after)
    if before_frame is None or after_frame is None:
        report["reason"] = (
            "malla facial no disponible en "
            + ("la entrada" if before_frame is None else "la salida")
        )
        return report

    before_points = _landmarks(before)
    after_points = _landmarks(after)
    landmark_shift = 0.0
    if before_points is not None and after_points is not None:
        landmark_shift = float(
            np.linalg.norm(before_points - after_points, axis=1).mean()
        )
    center_shift = abs(after_frame["center_y"] - before_frame["center_y"])
    area_ratio = (
        after_frame["area"] / before_frame["area"] if before_frame["area"] > 1e-9 else 1.0
    )
    area_change = abs(area_ratio - 1.0)
    forehead_change = abs(after_frame["forehead_ratio"] - before_frame["forehead_ratio"])
    roll_change = abs(after_frame["roll"] - before_frame["roll"])

    indicators = {
        "center_shift": center_shift / max(settings["min_face_center_shift"], 1e-6),
        "area_change": area_change / max(settings["min_face_area_ratio_change"], 1e-6),
        "forehead_change": forehead_change / max(settings["min_forehead_ratio_change"], 1e-6),
        "roll_change": roll_change / 10.0,
        "landmark_shift": landmark_shift / max(settings["landmark_shift_reference"], 1e-6),
    }
    score = float(min(1.0, max(indicators.values())))
    report.update({
        "medible": True,
        "score": round(score, 4),
        "center_shift": round(center_shift, 4),
        "area_ratio": round(area_ratio, 4),
        "forehead_change": round(forehead_change, 4),
        "roll_change": round(roll_change, 2),
        "landmark_shift": round(landmark_shift, 4),
        "landmark_changed": bool(landmark_shift >= settings["landmark_score_threshold"]),
        "indicators": {key: round(value, 3) for key, value in indicators.items()},
        "threshold": settings["min_change_score"],
        "changed": bool(score >= settings["min_change_score"]),
    })
    return report


# ---------------------------------------------------------------------------
# Zona medida (P7)
# ---------------------------------------------------------------------------

def zone_change(before: Image.Image, after: Image.Image,
                 zone: Optional[Image.Image]) -> Dict[str, Any]:
    """Mide cuánto cambió la zona medida y cuánto se movió fuera de ella."""
    settings = load_verification_settings()["zone"]
    report: Dict[str, Any] = {"available": False}
    if before is None or after is None or zone is None:
        report["reason"] = "faltan imágenes o zona"
        return report
    if before.size != after.size:
        after = after.convert("RGB").resize(before.size, Image.Resampling.LANCZOS)
    try:
        mask = np.asarray(
            zone.convert("L").resize(before.size, Image.Resampling.NEAREST)
        ) >= 96
    except Exception:
        report["reason"] = "zona ilegible"
        return report
    if not mask.any():
        report["reason"] = "zona vacía"
        return report
    a = np.asarray(before.convert("RGB"), dtype=np.int16)
    b = np.asarray(after.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        report["reason"] = "tamaños distintos"
        return report
    changed = np.abs(a - b).max(axis=2) > 8
    inside = float((changed & mask).sum()) / max(int(mask.sum()), 1)
    outside = float((changed & ~mask).sum()) / max(int((~mask).sum()), 1)
    report.update({
        "available": True,
        "inside_ratio": round(inside, 4),
        "outside_ratio": round(outside, 4),
        "zone_px": int(mask.sum()),
        "min_zone_change": settings["min_zone_change"],
        "max_outside_change": settings["max_outside_change"],
        "ok": bool(
            inside >= settings["min_zone_change"]
            and outside <= settings["max_outside_change"]
        ),
    })
    return report
