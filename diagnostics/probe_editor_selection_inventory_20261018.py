"""Real CPU preflight: resolve an OPEN inventory of visible elements.

The previous capability probe asked Moondream for a JSON object and it ignored
the schema, answering in prose (``{"objects":[]}``), so CLIPSeg had no labels to
measure.  This probe keeps the two-step, evidence-based design approved in
``plans/editor_object_selection_20261018.md``:

    photo -> Moondream prose description -> local LLM noun-phrase extraction
          -> CLIPSeg real masks -> coverage / components / peak

No downloads, no packages, no GPU inference, no hardcoded object list.  Faces
are measured with the same local ONNX detector as the selector contract.
"""
import argparse
import base64
import io
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

# llama_cpp links ggml-cuda.dll -> cudart64_12.dll/cublas64_12.dll.  The CUDA
# runtime is not on PATH in a bare shell, so the library cannot load.  Adding
# the CUDA bin and the package lib dir to the DLL search path fixes it without
# touching the environment of other processes.
_CUDA_BIN = Path(os.environ.get(
    "CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4")) / "bin"
for _dll_dir in (_CUDA_BIN, Path(__file__).resolve().parents[1] / "venv/Lib/site-packages/llama_cpp/lib"):
    if _dll_dir.is_dir():
        try:
            os.add_dll_directory(str(_dll_dir))
        except OSError:
            pass


def _image_data_url(image: Image.Image, max_side: int = 768) -> str:
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    preview.save(buf, "JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def detect_faces(image: Image.Image, settings: dict) -> list:
    from insightface.model_zoo import get_model

    det = settings["detector"]
    detector = get_model(str(ROOT / det["model_path"]), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(det["input_size"], det["input_size"]),
                     det_thresh=det["threshold"])
    try:
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        boxes, _ = detector.detect(bgr)
        return [] if boxes is None else boxes.tolist()
    finally:
        del detector


def moondream_description(image: Image.Image, settings: dict) -> str:
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MoondreamChatHandler

    inv = settings["inventory"]
    handler = MoondreamChatHandler(clip_model_path=str(ROOT / inv["projector_path"]), verbose=False)
    llm = Llama(model_path=str(ROOT / inv["model_path"]), chat_handler=handler,
                n_ctx=inv["context_size"], n_gpu_layers=0, n_threads=inv["threads"], verbose=False)
    try:
        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": _image_data_url(image)}},
                {"type": "text", "text": inv["describe_prompt"]},
            ]}],
            temperature=0.0, max_tokens=inv["max_tokens"],
            stop=["\n\nQuestion:"],
        )
        return response["choices"][0]["message"]["content"].strip()
    finally:
        try:
            llm.close()
        except Exception:
            pass
        try:
            handler._exit_stack.close()
        except Exception:
            pass
        del llm, handler


def extract_noun_phrases(description: str, settings: dict) -> list:
    """Turn a free description into short concrete noun phrases with a local LLM.

    No object catalogue is used: the extractor only rewrites what the visual
    model described into segmentable labels.
    """
    from llama_cpp import Llama

    ext = settings["extractor"]
    llm = Llama(model_path=str(ROOT / ext["model_path"]), n_ctx=ext["context_size"],
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
        raw = response["choices"][0]["message"]["content"].strip()
        parsed = json.loads(raw)
        labels = parsed.get("objects", parsed) if isinstance(parsed, dict) else parsed
        if not isinstance(labels, list):
            return []
        cleaned = []
        for label in labels:
            if not isinstance(label, str):
                continue
            label = label.strip().strip('."').lower()
            if not label or len(label.split()) > ext["max_label_words"]:
                continue
            if label not in cleaned:
                cleaned.append(label)
        return cleaned[:ext["max_items"]]
    except Exception as exc:  # noqa: BLE001 - evidence probe
        print(f"[INVENTORY] extractor failed: {exc}", flush=True)
        return []
    finally:
        try:
            llm.close()
        except Exception:
            pass
        del llm


def measure_labels(image: Image.Image, labels: list, settings: dict, out_dir: Path, name: str) -> list:
    from roop.img_editor.clipseg_masker import CLIPSegMasker
    import clip

    original_load = clip.load
    backbone = Path(os.path.expanduser(settings["segmenter"]["backbone_path"]))

    def local_load(_name, *args, **kwargs):
        return original_load(str(backbone), *args, **kwargs)

    clip.load = local_load
    masker = CLIPSegMasker()
    masker.device = "cpu"
    masker._WEIGHT_CANDIDATES = [(str(ROOT / settings["segmenter"]["weights_path"]), "uni")]
    ok, message = masker.initialize()
    assert ok and "SIN pesos" not in message, message
    measured = []
    try:
        for index, label in enumerate(labels):
            heat = masker.probability(image, label)
            if heat is None:
                continue
            binary = heat >= settings["segmenter"]["probability_threshold"]
            coverage = float(binary.mean())
            count, _ = cv2.connectedComponents(binary.astype(np.uint8))
            measured.append({
                "label": label,
                "coverage": coverage,
                "components": count - 1,
                "peak": float(heat.max()),
            })
            Image.fromarray(binary.astype(np.uint8) * 255).save(out_dir / f"{name}_{index}.png")
    finally:
        masker.model = None
    return measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()

    settings = json.loads((ROOT / "config/editor_selection.json").read_text(encoding="utf-8"))
    out_dir = ROOT / "diagnostics/out/editor_selection_20261018" / f"inventory_{time.time_ns()}"
    out_dir.mkdir(parents=True)

    photos = {
        "two_people": ROOT / "diagnostics/out/kontext_review_20261018/two_people_source.jpg",
        "coffee": ROOT / "venv/Lib/site-packages/skimage/data/coffee.png",
        "astronaut": ROOT / "venv/Lib/site-packages/skimage/data/astronaut.png",
    }
    if args.only:
        photos = {args.only: photos[args.only]}
    images = {key: Image.open(path).convert("RGB") for key, path in photos.items()}

    results = {}
    for name, photo in images.items():
        entry = {"photo": str(photos[name])}
        entry["faces"] = detect_faces(photo, settings)
        print(f"[INVENTORY] {name}: {len(entry['faces'])} faces", flush=True)

        started = time.perf_counter()
        description = moondream_description(photo, settings)
        entry["description"] = description
        entry["description_seconds"] = time.perf_counter() - started
        print(f"[INVENTORY] {name} description: {description}", flush=True)

        started = time.perf_counter()
        labels = extract_noun_phrases(description, settings)
        entry["labels"] = labels
        entry["extractor_seconds"] = time.perf_counter() - started
        print(f"[INVENTORY] {name} labels: {labels}", flush=True)

        entry["measured"] = measure_labels(photo, labels, settings, out_dir, name)
        for item in entry["measured"]:
            print(f"[INVENTORY] {name} {item['label']}: coverage={item['coverage']:.2%} "
                  f"components={item['components']}", flush=True)
        results[name] = entry

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("EVIDENCE", out_dir, flush=True)


if __name__ == "__main__":
    main()
