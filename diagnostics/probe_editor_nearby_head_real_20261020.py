"""Real Kontext edit on a photo with two nearby people; no inference doubles.

The selected head is measured on CPU, the crop is tightened and the nearby
unselected head is blinded before the model runs. The evidence saves the real
reference, the raw model output, the final result and the per head change
geometry. Pixel metrics are protection evidence only; adherence still requires
viewing the result.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor import element_selection as selection  # noqa: E402


def measure_heads(source, settings):
    """Measure the heads exactly like the production analyser does."""
    work = source.copy()
    work.thumbnail((settings["max_analysis_side"], settings["max_analysis_side"]))
    faces = selection._detect_faces(work, settings)
    scale = np.array([source.width / work.width, source.height / work.height])
    regions = []
    for index, (box, kps) in enumerate(faces):
        full_box = np.asarray(box) * np.tile(scale, 2)
        full_kps = np.asarray(kps) * scale if kps is not None else None
        mask = selection._head_mask_for_box(source, full_box, full_kps)
        if mask is None:
            continue
        arr = np.asarray(mask.convert("L")) >= 128
        if not arr.any():
            continue
        ys, xs = np.where(arr)
        regions.append(selection.SelectionRegion(
            id=f"head-{index}", label=f"Cabeza/rostro {index + 1}", mask=mask,
            bbox=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            kind="head", coverage=float(arr.mean())))
    return selection.SelectionAnalysis(
        selection.image_fingerprint(source), source.size, regions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--select", required=True,
                        help="Exact measured head id, for example head-2")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", type=int, default=424242)
    args = parser.parse_args()
    destination = ROOT / "diagnostics/out/editor_nearby_head_20261020" / str(time.time_ns())
    destination.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.source).convert("RGB")
    source.save(destination / "source.png")
    settings = selection.load_settings()
    analysis = measure_heads(source, settings)
    if not analysis.regions:
        raise RuntimeError("No head measured on the photo")
    ids = [args.select]
    if ids[0] not in {region.id for region in analysis.regions}:
        raise ValueError(f"Unknown head id: {ids[0]}")
    selection.render_selection(source, analysis, ids).save(destination / "selected.png")

    from roop.comfy_workflows import get_comfyui_url
    session = requests.Session()
    session.trust_env = False
    queue_response = session.get(f"{get_comfyui_url()}/queue", timeout=5)
    queue_response.raise_for_status()
    queue = queue_response.json()
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI is busy; the test does not enqueue")
    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    from roop.img_editor.img_editor_manager import ImgEditorManager
    client = get_flux_edit_comfy_client()
    real_generate = client.generate
    generations = []

    def observe_generate(**kwargs):
        index = len(generations)
        kwargs["image"].save(destination / f"reference_{index}.png")
        generations.append({"size": kwargs["image"].size, "seed": kwargs["seed"],
                            "steps": kwargs["num_inference_steps"],
                            "guidance": kwargs["guidance_scale"],
                            "prompt": kwargs["prompt"],
                            "mask_is_none": kwargs["mask_image"] is None})
        result, note = real_generate(**kwargs)
        if result is not None:
            result.image.save(destination / f"raw_{index}.png")
        return result, note

    client.generate = observe_generate
    started = time.monotonic()
    try:
        output, note, mask = ImgEditorManager().generate_intelligent(
            image=source, prompt=args.prompt, engine="kontext", seed=args.seed,
            selection_analysis=analysis, selected_region_ids=ids, enhance_faces=False,
        )
    finally:
        client.generate = real_generate
    if output is None or mask is None:
        raise RuntimeError(note)
    output.save(destination / "result.png")
    mask.save(destination / "effective_mask.png")


    before = np.asarray(source)
    after = np.asarray(output)
    changed = np.any(before != after, axis=2)
    gray = np.array([int(v) for v in settings["edit"]["blind_fill"]], dtype=np.uint8)
    grey_now = np.all(after == gray, axis=2)
    grey_leak = grey_now & ~np.all(before == gray, axis=2)
    per_head = {}
    for region in analysis.regions:
        arr = np.asarray(region.mask.convert("L")) >= 128
        per_head[region.id] = {
            "changed_pixels": int((changed & arr).sum()),
            "head_pixels": int(arr.sum()),
            "grey_pixels": int((grey_now & arr).sum()),
        }
    metrics = {
        "source": str(args.source.resolve()), "instruction": args.prompt,
        "selected_ids": ids, "edit_seconds": time.monotonic() - started,
        "message": note, "generations": generations,
        "changed_pixels": int(changed.sum()),
        "changed_exterior_pixels": int(changed[np.asarray(mask) == 0].sum()),
        "grey_leak_pixels": int(grey_leak.sum()),
        "per_head": per_head,
        "scope": "Real inference and protection metrics; adherence requires visual inspection",
    }
    (destination / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"EVIDENCE {destination}", flush=True)
    assert len(generations) == 1 and generations[0]["mask_is_none"], generations
    assert changed.any(), "No pixels changed"
    assert not changed[np.asarray(mask) == 0].any(), "Exterior changed"
    assert not grey_leak.any(), "The grey blind patch reached the final photo"
    for other in [region.id for region in analysis.regions if region.id not in ids]:
        assert per_head[other]["changed_pixels"] == 0, f"{other} changed"
        assert per_head[other]["grey_pixels"] == 0, f"{other} shows the blind patch"


if __name__ == "__main__":
    main()
