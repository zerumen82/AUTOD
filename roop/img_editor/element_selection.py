"""Visual element selection for the Image Editor (H-EDIT-SELECT).

Analyzes a photo with local CPU models only (ONNX face detector, Moondream
description, Qwen noun-phrase extraction, CLIPSeg segmentation) and returns
measured regions the user can select for a focused edit.

Design rules (see ``plans/editor_object_selection_20261018.md``):

* No hardcoded object catalogue and no hardcoded classes: labels come from the
  visual description of the photo.
* A VLM only proposes labels; the region always comes from CLIPSeg segmentation
  and a face detector, never from VLM coordinates.
* No GPU inference: every analysis model runs on CPU and is released after use.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from roop.img_editor.vision_guard import VisionUnavailableError

_ROOT = Path(__file__).resolve().parents[2]
_SETTINGS_PATH = _ROOT / "config" / "editor_selection.json"
_DLL_HANDLES = ExitStack()
_DLL_DIRECTORIES = set()


@dataclass
class SelectionRegion:
    """One measured region the user can select."""

    id: str
    label: str
    mask: Image.Image  # mode "L"
    bbox: Tuple[int, int, int, int]
    kind: str  # "head" | "object"
    coverage: float = 0.0
    components: int = 1
    peak: float = 0.0


@dataclass
class SelectionAnalysis:
    """The measured elements of one photo, tied to its fingerprint."""

    fingerprint: str
    size: Tuple[int, int]
    regions: List[SelectionRegion]
    message: str = ""


def load_settings() -> dict:
    return json.loads(_SETTINGS_PATH.read_text(encoding="utf-8"))


def image_fingerprint(image: Image.Image) -> str:
    """Canonical RGB fingerprint: a new/reused photo invalidates the analysis."""
    rgb = image.convert("RGB")
    digest = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def region_by_id(analysis: SelectionAnalysis, region_id: str) -> Optional[SelectionRegion]:
    for region in analysis.regions:
        if region.id == region_id:
            return region
    return None


def union_mask(analysis: SelectionAnalysis, ids: Sequence[str]) -> Optional[Image.Image]:
    """Union of the selected regions (one mask, one edit pass)."""
    chosen = set(ids)
    masks = [region.mask for region in analysis.regions if region.id in chosen]
    if not masks or len(masks) != len(chosen):
        return None
    width, height = analysis.size
    combined = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        if not isinstance(mask, Image.Image) or mask.size != analysis.size:
            raise ValueError("La máscara no corresponde al tamaño del análisis")
        arr = np.asarray(mask.convert("L"))
        combined = np.maximum(combined, (arr >= 128).astype(np.uint8) * 255)
    if not combined.any():
        return None
    return Image.fromarray(combined, "L")


def _selected_object_support(analysis: SelectionAnalysis,
                             selected_ids: Sequence[str]) -> Optional[Image.Image]:
    """Measured support of the selected objects (measured heads excluded)."""
    object_ids = [region.id for region in analysis.regions
                  if region.id in set(selected_ids) and region.kind != "head"]
    if not object_ids:
        return None
    return union_mask(analysis, object_ids)


def _head_inside_selection(head_mask: Image.Image, support: Optional[Image.Image],
                           inclusion: float) -> bool:
    """True when a measured head belongs to the selected subject.

    A selected measured body can include its own head. Both the protection
    and the blind have to agree on that: the head is editable, so it must
    stay visible in the reference the editor receives.
    """
    if support is None:
        return False
    if not isinstance(head_mask, Image.Image):
        return False
    arr = np.asarray(head_mask.convert("L")) >= 128
    if not arr.any():
        return False
    support_arr = np.asarray(support) > 0
    return float((arr & support_arr).sum()) / int(arr.sum()) >= inclusion


def protection_mask(analysis: SelectionAnalysis, selected_ids: Sequence[str]) -> Optional[Image.Image]:
    """Heads the user did NOT select, so the finish cannot overwrite them."""
    selected = set(selected_ids)
    selected_mask = union_mask(analysis, selected_ids)
    if selected_mask is None:
        raise ValueError("La selección no tiene una máscara válida")
    support = _selected_object_support(analysis, selected_ids)
    inclusion = float(load_settings()["edit"]["head_inclusion_coverage"])
    masks = [
        region.mask for region in analysis.regions
        if region.kind == "head" and region.id not in selected
    ]
    if not masks:
        return None
    width, height = analysis.size
    combined = np.zeros((height, width), dtype=np.uint8)
    for mask in masks:
        if not isinstance(mask, Image.Image) or mask.size != analysis.size:
            raise ValueError("La cabeza protegida no corresponde al tamaño del análisis")
        # A selected measured body/object can include its own head. Protect
        # the remaining heads, not those explicitly contained in the choice.
        if _head_inside_selection(mask, support, inclusion):
            continue
        arr = np.asarray(mask.convert("L")) >= 128
        combined = np.maximum(combined, arr.astype(np.uint8) * 255)
    if not combined.any():
        return None
    return Image.fromarray(combined, "L")


def blind_mask_in_image(image: Image.Image, mask: Image.Image,
                        dilate_radius: int = 0, fill=None) -> Image.Image:
    """Gray out the measured region so the editor cannot target it.

    The blind patch is what FLUX actually receives for the unselected
    elements: it removes the identity of a nearby person without touching
    the original photo. Dilation keeps a halo of the original skin out of
    the reference so the model cannot copy the unselected face.
    """
    if not isinstance(image, Image.Image) or not isinstance(mask, Image.Image):
        raise ValueError("La máscara y la foto deben ser imágenes")
    if mask.size != image.size:
        raise ValueError("La máscara no corresponde al tamaño de la foto")
    arr = np.asarray(mask.convert("L")) >= 128
    if not arr.any():
        raise ValueError("La máscara no tiene píxeles medidos")
    if dilate_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(dilate_radius) * 2 + 1, int(dilate_radius) * 2 + 1))
        arr = cv2.dilate(arr.astype(np.uint8), kernel).astype(bool)
    blind = image.convert("RGB").copy()
    if fill is None:
        fill = tuple(load_settings()["edit"]["blind_fill"])
    blind.paste(Image.new("RGB", image.size, tuple(int(v) for v in fill)),
                Image.fromarray(arr.astype(np.uint8) * 255, "L"))
    return blind


def blind_unselected_in_reference(analysis: SelectionAnalysis, selected_ids: Sequence[str],
                                  image: Image.Image, dilate_radius: int = 0,
                                  keep_mask: Optional[Image.Image] = None) -> Image.Image:
    """Return the reference with every unselected head blinded.

    A crop that contains two people is ambiguous for the editor: it may
    apply the instruction to the wrong one. Blinding the unselected heads
    leaves only the selected subject as a valid target, and the original
    pixels of those heads are restored during composition.

    ``keep_mask`` holds the pixels that belong to the selected subject.
    Measured head ellipses of nearby people overlap, so the blind patch
    stops at that boundary instead of erasing part of the selected head.

    A head that belongs to the selected subject (its measured mask lies
    mostly inside the selected object support) is NOT blinded: the editor
    has to see the face it is asked to edit, and the same measured rule
    already decides that such a head is editable in the protection mask.
    """
    selected = set(selected_ids)
    width, height = analysis.size
    support = _selected_object_support(analysis, selected_ids)
    inclusion = float(load_settings()["edit"]["head_inclusion_coverage"])
    blind = np.zeros((height, width), dtype=bool)
    for region in analysis.regions:
        if region.kind != "head" or region.id in selected:
            continue
        if not isinstance(region.mask, Image.Image) or region.mask.size != analysis.size:
            raise ValueError("La cabeza protegida no corresponde al tamaño del análisis")
        if _head_inside_selection(region.mask, support, inclusion):
            continue
        blind |= np.asarray(region.mask.convert("L")) >= 128
    if not blind.any():
        return image.convert("RGB").copy()
    keep = None
    if keep_mask is not None:
        if keep_mask.size != analysis.size:
            raise ValueError("La máscara de conservación no corresponde al tamaño del análisis")
        keep = np.asarray(keep_mask.convert("L")) >= 128
        blind &= ~keep
        if not blind.any():
            return image.convert("RGB").copy()
    blind_image = Image.fromarray(blind.astype(np.uint8) * 255, "L")
    if keep is None:
        return blind_mask_in_image(image, blind_image, dilate_radius)
    # The dilation halo must not creep back into the selected subject, so the
    # dilated region is measured here and the selected pixels are restored.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (int(dilate_radius) * 2 + 1, int(dilate_radius) * 2 + 1)
    ) if dilate_radius > 0 else None
    dilated = cv2.dilate(blind.astype(np.uint8), kernel).astype(bool) if kernel is not None else blind
    result = image.convert("RGB").copy()
    fill = tuple(int(v) for v in load_settings()["edit"]["blind_fill"])
    result.paste(Image.new("RGB", image.size, fill),
                 Image.fromarray(dilated.astype(np.uint8) * 255, "L"))
    restore = dilated & keep
    if restore.any():
        result.paste(image.convert("RGB"),
                     Image.fromarray(restore.astype(np.uint8) * 255, "L"))
    return result


def regions_at_point(analysis: SelectionAnalysis, x: int, y: int) -> List[SelectionRegion]:
    """Return all masks actually covering a point, never bounding-box hits."""
    if not (0 <= x < analysis.size[0] and 0 <= y < analysis.size[1]):
        return []
    covering = []
    for region in analysis.regions:
        if not isinstance(region.mask, Image.Image) or region.mask.size != analysis.size:
            continue
        arr = np.asarray(region.mask.convert("L"))
        if arr[y, x] >= 128:
            covering.append(region)
    return covering


def choose_at_point(analysis: SelectionAnalysis, x: int, y: int) -> Optional[SelectionRegion]:
    """Choose a uniquely nested region; competing overlapping masks abstain."""
    covering = regions_at_point(analysis, x, y)
    if not covering:
        return None
    ordered = sorted(covering, key=lambda region: np.count_nonzero(np.asarray(region.mask) >= 128))
    chosen = ordered[0]
    core = np.asarray(chosen.mask) >= 128
    for other in ordered[1:]:
        other_core = np.asarray(other.mask) >= 128
        if np.array_equal(core, other_core) or np.any(core & ~other_core):
            return None
    return chosen


def validate_selection(image: Image.Image, analysis: SelectionAnalysis,
                       ids: Sequence[str]) -> Tuple[bool, str]:
    if not isinstance(analysis, SelectionAnalysis) or not isinstance(image, Image.Image):
        return False, "no hay análisis"
    if image.size != analysis.size:
        return False, "el tamaño de la foto cambió"
    if image_fingerprint(image) != analysis.fingerprint:
        return False, "la foto cambió; el análisis ya no corresponde"
    if not ids:
        return False, "no hay regiones seleccionadas"
    known = {region.id for region in analysis.regions}
    if len(known) != len(analysis.regions):
        return False, "el análisis contiene identificadores ambiguos"
    if isinstance(ids, str) or any(not isinstance(value, str) for value in ids):
        return False, "identificadores de selección inválidos"
    missing = [region_id for region_id in ids if region_id not in known]
    if missing:
        return False, f"selección caducada: {', '.join(missing)}"
    for region in analysis.regions:
        if region.id not in ids and region.kind != "head":
            continue
        if not isinstance(region.mask, Image.Image) or region.mask.size != analysis.size:
            return False, "una máscara no corresponde al tamaño de la foto"
        if not np.any(np.asarray(region.mask.convert("L")) >= 128):
            return False, "una región no tiene píxeles medidos"
    return True, ""


def render_selection(image: Image.Image, analysis: SelectionAnalysis,
                     selected_ids: Optional[Sequence[str]] = None) -> Image.Image:
    """Visualization overlay: outlines for every region, highlight for selected."""
    settings = load_settings()
    preview = settings["preview"]
    outline_color = tuple(preview["outline_color"])
    selected_color = tuple(preview["selected_color"])
    width_fraction = float(preview["outline_width_fraction"])
    min_width = int(preview["min_outline_width"])
    alpha = float(preview["selected_alpha"])

    selected = set(selected_ids or [])
    base = image.convert("RGB").copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    line_width = max(min_width, int(round(max(base.size) * width_fraction)))

    for region in analysis.regions:
        is_selected = region.id in selected
        color = selected_color if is_selected else outline_color
        if region.mask.size != image.size or analysis.size != image.size:
            raise ValueError("La vista de selección no corresponde a la foto")
        mask = region.mask.convert("L")
        arr = np.asarray(mask) >= 128
        if not arr.any():
            continue
        if is_selected:
            fill = np.zeros((*arr.shape, 4), dtype=np.uint8)
            fill[arr] = (*color, int(255 * alpha))
            overlay = Image.alpha_composite(overlay, Image.fromarray(fill, "RGBA"))
            draw = ImageDraw.Draw(overlay)
        contours, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            points = [tuple(point) for point in contour.reshape(-1, 2).tolist()]
            if len(points) > 1:
                draw.line(points + points[:1], fill=(*color, 255), width=line_width)
    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")


# --------------------------------------------------------------------------- #
# Analysis (CPU models, released after use)
# --------------------------------------------------------------------------- #

def _head_mask_for_box(image: Image.Image, box, kps=None) -> Optional[Image.Image]:
    """Reuse the editor's measured head geometry, including profile landmarks."""
    from roop.img_editor.img_editor_manager import ImgEditorManager

    face = SimpleNamespace(bbox=box[:4], kps=kps)
    manager = object.__new__(ImgEditorManager)
    return manager._head_mask_from_faces(image, [face])


def _detect_faces(image: Image.Image, settings: dict) -> list:
    from insightface.model_zoo import get_model

    det = settings["detector"]
    model_path = _ROOT / det["model_path"]
    if not model_path.is_file():
        raise FileNotFoundError(f"Detector local ausente: {model_path}")
    detector = get_model(str(model_path), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(det["input_size"], det["input_size"]),
                     det_thresh=det["threshold"])
    try:
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        boxes, landmarks = detector.detect(bgr)
        if boxes is None:
            return []
        return [(box[:4], landmarks[index] if landmarks is not None else None)
                for index, box in enumerate(boxes)]
    finally:
        del detector


def _add_llama_dll_directories(settings: dict) -> None:
    """llama_cpp links the CUDA backend; a bare shell has no CUDA on PATH."""
    if not hasattr(os, "add_dll_directory"):
        return
    for raw in settings["runtime"]["dll_directories"]:
        directory = Path(os.path.expandvars(raw))
        if not directory.is_absolute():
            directory = _ROOT / directory
        if directory.is_dir() and str(directory) not in _DLL_DIRECTORIES:
            try:
                _DLL_HANDLES.enter_context(os.add_dll_directory(str(directory)))
                _DLL_DIRECTORIES.add(str(directory))
            except OSError:
                pass


def _image_data_url(image: Image.Image, max_side: int) -> str:
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    preview.save(buffer, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _moondream_description(image: Image.Image, settings: dict) -> str:
    _add_llama_dll_directories(settings)
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MoondreamChatHandler

    inv = settings["inventory"]
    guard = settings.get("vision_guard") or {}
    if guard.get("enabled", True):
        # The projector has no device selector: the ggml backend decides, and
        # the CPU backend crashes the process natively.  Refuse instead.
        from roop.img_editor.vision_guard import (
            VisionUnavailableError,
            check_vision_preflight,
        )

        preflight = check_vision_preflight(settings)
        print(f"[ElementSelection] {preflight.as_message()}", flush=True)
        if not preflight.ok:
            raise VisionUnavailableError(preflight.as_message())
    handler = None
    llm = None
    try:
        handler = MoondreamChatHandler(clip_model_path=str(_ROOT / inv["projector_path"]), verbose=False)
        llm = Llama(model_path=str(_ROOT / inv["model_path"]), chat_handler=handler,
                    n_ctx=inv["context_size"], n_gpu_layers=0, n_threads=inv["threads"], verbose=False)
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _image_data_url(image, inv["max_image_side"])}},
                {"type": "text", "text": inv["describe_prompt"]},
            ]}],
            temperature=float(inv["temperature"]), max_tokens=int(inv["max_tokens"]),
            stop=["\n\nQuestion:"],
        )
        return response["choices"][0]["message"]["content"].strip()
    finally:
        try:
            if llm is not None:
                llm.close()
        except Exception:
            pass
        try:
            if handler is not None:
                handler._exit_stack.close()
        except Exception:
            pass
        del llm, handler


def _extract_noun_phrases(description: str, settings: dict) -> List[str]:
    """Rewrite a free description into short concrete labels (local LLM)."""
    _add_llama_dll_directories(settings)
    from llama_cpp import Llama

    ext = settings["extractor"]
    llm = Llama(model_path=str(_ROOT / ext["model_path"]), n_ctx=ext["context_size"],
                n_gpu_layers=0, n_threads=ext["threads"], verbose=False)
    try:
        response = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": ext["system_prompt"]},
                {"role": "user", "content": description},
            ],
            temperature=0.0, max_tokens=ext["max_tokens"],
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response["choices"][0]["message"]["content"].strip())
        labels = parsed.get("objects", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(labels, list):
            return []
        cleaned: List[str] = []
        for label in labels:
            if not isinstance(label, str):
                continue
            label = label.strip().strip('."').lower()
            if not label or len(label.split()) > ext["max_label_words"]:
                continue
            if label not in cleaned:
                cleaned.append(label)
        return cleaned[: ext["max_items"]]
    except Exception as exc:  # noqa: BLE001 - evidence probe
        print(f"[Selection] Extractor de etiquetas falló: {exc}", flush=True)
        return []
    finally:
        try:
            llm.close()
        except Exception:
            pass
        del llm


def _measure_labels(image: Image.Image, labels: Sequence[str], settings: dict) -> List[SelectionRegion]:
    from roop.img_editor.clipseg_masker import CLIPSegMasker
    seg = settings["segmenter"]
    backbone = Path(os.path.expanduser(seg["backbone_path"]))
    if not backbone.is_file():
        print(f"[Selection] Backbone CLIP ausente: {backbone}", flush=True)
        return []

    masker = CLIPSegMasker(backbone_path=str(backbone))
    masker.device = "cpu"
    masker._WEIGHT_CANDIDATES = [(str(_ROOT / seg["weights_path"]), "uni")]
    ok, message = masker.initialize()
    if not ok or "SIN pesos" in message:
        print(f"[Selection] CLIPSeg no disponible: {message}", flush=True)
        return []
    regions: List[SelectionRegion] = []
    try:
        threshold = float(seg["probability_threshold"])
        min_coverage = float(seg["min_coverage"])
        max_coverage = float(seg["max_coverage"])
        min_component = float(seg["min_component_coverage"])
        duplicate_iou = float(seg["duplicate_iou"])
        for index, label in enumerate(labels):
            heat = masker.probability(image, label)
            if heat is None:
                continue
            binary = heat >= threshold
            count, labels_img = cv2.connectedComponents(binary.astype(np.uint8))
            components = 0
            filtered = np.zeros(binary.shape, dtype=bool)
            for component in range(1, count):
                if float((labels_img == component).mean()) >= min_component:
                    components += 1
                    filtered |= labels_img == component
            if components == 0:
                continue
            binary = filtered
            coverage = float(binary.mean())
            if coverage < min_coverage or coverage > max_coverage:
                continue
            mask = Image.fromarray(binary.astype(np.uint8) * 255, "L")
            ys, xs = np.where(binary)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
            regions.append(SelectionRegion(
                id=f"object-{index}", label=label, mask=mask, bbox=bbox,
                kind="object", coverage=coverage, components=components,
                peak=float(heat.max()),
            ))
        # Drop labels whose mask is essentially a duplicate of an earlier one.
        unique: List[SelectionRegion] = []
        for region in regions:
            arr = np.asarray(region.mask) >= 128
            duplicate = False
            for kept in unique:
                kept_arr = np.asarray(kept.mask) >= 128
                intersection = float((arr & kept_arr).sum())
                union = float((arr | kept_arr).sum())
                if union and intersection / union >= duplicate_iou:
                    duplicate = True
                    break
            if not duplicate:
                unique.append(region)
        return unique
    finally:
        masker.model = None


def analyze_elements(image: Image.Image, settings: Optional[dict] = None) -> SelectionAnalysis:
    """Measure the selectable regions of one photo on CPU."""
    settings = settings or load_settings()
    image = image.convert("RGB")
    fingerprint = image_fingerprint(image)
    regions: List[SelectionRegion] = []

    work = image.copy()
    work.thumbnail((settings["max_analysis_side"], settings["max_analysis_side"]))
    faces = _detect_faces(work, settings)
    scale = np.array([image.width / work.width, image.height / work.height])
    for index, (box, kps) in enumerate(faces):
        full_box = np.asarray(box) * np.tile(scale, 2)
        full_kps = np.asarray(kps) * scale if kps is not None else None
        mask = _head_mask_for_box(image, full_box, full_kps)
        if mask is None:
            continue
        arr = np.asarray(mask) >= 128
        if not arr.any():
            continue
        ys, xs = np.where(arr)
        label_template = str(settings["detector"]["label"])
        regions.append(SelectionRegion(
            id=f"head-{index}", label=label_template.format(index=index + 1),
            mask=mask, bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            kind="head", coverage=float(arr.mean()),
        ))

    try:
        head_count = len(regions)
        description = _moondream_description(work, settings)
        labels = _extract_noun_phrases(description, settings)
        regions.extend(_measure_labels(image, labels, settings))
        message = f"{head_count} cabeza(s), {len(regions) - head_count} región(es) segmentada(s)"
        if not labels:
            message += "; el inventario visual no propuso etiquetas"
    except VisionUnavailableError as exc:
        # The vision path cannot run safely: report it instead of pretending
        # the inventory was measured without it.
        message = exc.message
        raise
    except Exception as exc:  # noqa: BLE001 - evidence probe
        message = f"análisis parcial: {exc}"

    return SelectionAnalysis(
        fingerprint=fingerprint, size=image.size, regions=regions, message=message,
    )
