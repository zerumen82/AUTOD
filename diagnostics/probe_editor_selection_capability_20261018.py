"""Real CPU capability preflight, before integrating element-selection UI.

Uses only installed local model files and non-explicit bundled photographs.
No ComfyUI requests, packages, model downloads or GPU inference are involved.
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    settings = json.loads((ROOT / "config/editor_selection.json").read_text(encoding="utf-8"))
    output = ROOT / "diagnostics/out/editor_selection_20261018" / f"capability_{time.time_ns()}"
    output.mkdir(parents=True)
    photos = {
        "two_people": ROOT / "diagnostics/out/kontext_review_20261018/two_people_source.jpg",
        "coffee": ROOT / "venv/Lib/site-packages/skimage/data/coffee.png",
        "astronaut": ROOT / "venv/Lib/site-packages/skimage/data/astronaut.png",
    }
    if args.only:
        photos = {args.only: photos[args.only]}
    images = {name: Image.open(path).convert("RGB") for name, path in photos.items()}
    from insightface.model_zoo import get_model
    det = settings["detector"]
    detector = get_model(str(ROOT / det["model_path"]), providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(det["input_size"], det["input_size"]), det_thresh=det["threshold"])
    results = {}
    for name, photo in images.items():
        boxes, _ = detector.detect(cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2BGR))
        results[name] = {"photo": str(photos[name]), "boxes": boxes.tolist()}
        print(name, "faces", len(boxes), flush=True)
    del detector

    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MoondreamChatHandler
    inv = settings["inventory"]
    handler = MoondreamChatHandler(clip_model_path=str(ROOT / inv["projector_path"]), verbose=False)
    llm = Llama(model_path=str(ROOT / inv["model_path"]), chat_handler=handler,
                n_ctx=inv["context_size"], n_gpu_layers=0, n_threads=inv["threads"], verbose=False)
    for name, photo in images.items():
        preview = photo.copy()
        preview.thumbnail((768, 768))
        buf = io.BytesIO()
        preview.save(buf, "JPEG")
        data = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
        started = time.perf_counter()
        response = llm.create_chat_completion(messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data}},
            {"type": "text", "text": inv["prompt"]},
        ]}], temperature=inv["temperature"], max_tokens=inv["max_tokens"],
            response_format={"type": "json_object", "schema": inv["schema"]},
            stop=["\n\nQuestion:"])
        raw = response["choices"][0]["message"]["content"]
        results[name].update(inventory=raw, inventory_seconds=time.perf_counter() - started)
        print(name, "inventory", raw, flush=True)
    llm.close()
    handler._exit_stack.close()
    del llm, handler

    from roop.img_editor.clipseg_masker import CLIPSegMasker
    import clip
    original_load = clip.load
    backbone = Path(os.path.expanduser(settings["segmenter"]["backbone_path"]))
    assert backbone.is_file(), "Missing local backbone; downloading is forbidden"
    # Use the existing loader against a concrete local filename, never a URL.
    def local_load(_name, *args, **kwargs):
        return original_load(str(backbone), *args, **kwargs)
    clip.load = local_load
    masker = CLIPSegMasker()
    masker.device = "cpu"
    masker._WEIGHT_CANDIDATES = [(str(ROOT / settings["segmenter"]["weights_path"]), "uni")]
    ok, message = masker.initialize()
    assert ok and "SIN pesos" not in message, message
    try:
        for name, photo in images.items():
            labels = json.loads(results[name]["inventory"])["objects"]
            measured = []
            for index, label in enumerate(labels[:inv["max_items"]]):
                if not label or len(label.split()) > inv["max_label_words"]:
                    continue
                heat = masker.probability(photo, label)
                assert heat is not None
                binary = heat >= settings["segmenter"]["probability_threshold"]
                coverage = float(binary.mean())
                count, _ = cv2.connectedComponents(binary.astype(np.uint8))
                measured.append({"label": label, "coverage": coverage, "components": count - 1,
                                 "peak": float(heat.max())})
                Image.fromarray(binary.astype(np.uint8) * 255).save(output / f"{name}_{index}.png")
                print(name, label, f"coverage={coverage:.3%}", flush=True)
            results[name]["measured"] = measured
    finally:
        masker.model = None
    (output / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("EVIDENCE", output, flush=True)


if __name__ == "__main__":
    main()
