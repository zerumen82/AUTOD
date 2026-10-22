"""Real local Kontext edit with two clothed people; no inference doubles.

Supply an inspected non-explicit source and an appearance instruction.
Instrumentation saves the actual reference, raw model output, measured mask,
final output, trace, hashes and changed-pixel geometry. Pixel differences are
protection evidence only; adherence still requires viewing the result.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.comfy_workflows import get_comfyui_url  # noqa: E402
from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("prompt")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    session = requests.Session()
    session.trust_env = False
    response = session.get(f"{get_comfyui_url()}/queue", timeout=5)
    response.raise_for_status()
    queue = response.json()
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI ocupado; no se encola la prueba")
    destination = args.output / str(time.time_ns())
    destination.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.source).convert("RGB")
    source.save(destination / "source.png")
    client = get_flux_edit_comfy_client()
    original_generate = client.generate
    original_load = client.load
    generations, loads = [], []

    def observe_load(*positional, **keywords):
        result = original_load(*positional, **keywords)
        loads.append({"requested": str(positional or keywords),
                      "loaded": client._flux_version, "ok": result[0]})
        return result

    def observe_generate(*positional, **keywords):
        index = len(generations)
        reference = keywords["image"]
        reference.save(destination / f"reference_{index}.png")
        record = {"size": reference.size, "steps": keywords.get("num_inference_steps"),
                  "guidance": keywords.get("guidance_scale"),
                  "mask_is_none": keywords.get("mask_image") is None,
                  "prompt": keywords.get("prompt")}
        generations.append(record)
        result, message = original_generate(*positional, **keywords)
        record["message"] = message
        if result is not None:
            result.image.save(destination / f"raw_{index}.png")
        return result, message

    client.load, client.generate = observe_load, observe_generate
    started = time.monotonic()
    try:
        output, message, mask = ImgEditorManager().generate_intelligent(
            image=source, prompt=args.prompt, engine="kontext", seed=args.seed,
            use_rewriter=True, use_semantic=True, enhance_faces=False,
        )
    finally:
        client.load, client.generate = original_load, original_generate
    if output is None:
        raise RuntimeError(message)
    output.save(destination / "result.png")
    if mask is not None:
        mask.save(destination / "mask.png")
    before = np.asarray(source, dtype=np.int16)
    after = np.asarray(output, dtype=np.int16)
    assert after.shape == before.shape, (after.shape, before.shape)
    changed = np.abs(after - before).max(axis=2) > 0
    ys, xs = np.where(changed)
    metrics = {
        "source": str(args.source.resolve()), "instruction": args.prompt,
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "result_sha256": hashlib.sha256((destination / "result.png").read_bytes()).hexdigest(),
        "seed": args.seed, "elapsed_s": time.monotonic() - started,
        "message": message, "generations": generations, "loads": loads,
        "source_size": source.size, "changed_pixels": int(changed.sum()),
        "changed_bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        if xs.size else None,
        "right_half_changed_pixels": int(changed[:, source.width // 2:].sum()),
        "bottom_third_changed_pixels": int(changed[source.height * 2 // 3:].sum()),
        "scope": "Real inference and protection metrics; requires visual inspection for adherence",
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"EVIDENCE: {destination}", flush=True)
    assert generations, f"No real generation: {message}"
    assert all(load["ok"] and "kontext" in load["loaded"].lower() for load in loads), loads
    assert all(item["mask_is_none"] for item in generations), generations
    assert all(size["size"][0] * size["size"][1] < source.width * source.height
               for size in generations), "full-frame generation instead of a local crop"
    assert changed.any(), "No edited pixels (does not establish instruction adherence)"
    assert metrics["right_half_changed_pixels"] == 0, "other person changed"
    assert metrics["bottom_third_changed_pixels"] == 0, "nonrequested clothing changed"


if __name__ == "__main__":
    main()
