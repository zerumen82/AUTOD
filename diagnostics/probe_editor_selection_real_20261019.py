"""Execute the production selector and optional Kontext edit on inspected photos.

No inference doubles, downloads, or overlays passed to the model. CLI labels
only choose from the actual measured inventory; they are not routing rules.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor import element_selection as selection  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--select", action="append", default=[])
    parser.add_argument("--prompt", default="")
    parser.add_argument("--seed", type=int, default=424242)
    parser.add_argument("--reuse-analysis", type=Path,
                        help="Reuse measured masks from a previous real run on the identical photo")
    parser.add_argument("--canvas-side", type=int, default=0,
                        help="A/B only: opt into isotropic aligned reference canvas")
    parser.add_argument("--direct-prompt", action="store_true",
                        help="A/B only: forward the full user instruction without generic wrapper")
    parser.add_argument("--guidance", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--text-encoder", default="",
                        help="A/B only: existing local second text encoder, keeping model and CLIP-L")
    parser.add_argument("--crop-context-fraction", type=float, default=None,
                        help="A/B only: base context (objects also use the object-context maximum)")
    parser.add_argument("--object-context-fraction", type=float, default=None,
                        help="A/B only: object context; set both context flags to recreate an old crop")
    args = parser.parse_args()
    destination = ROOT / "diagnostics/out/editor_selection_20261019" / str(time.time_ns())
    destination.mkdir(parents=True, exist_ok=False)
    source = Image.open(args.source).convert("RGB")
    source.save(destination / "source.png")
    started = time.monotonic()
    if args.reuse_analysis:
        stored = json.loads(args.reuse_analysis.read_text(encoding="utf-8"))
        if stored["fingerprint"] != selection.image_fingerprint(source) or tuple(stored["size"]) != source.size:
            raise ValueError("Saved real analysis belongs to a different photo")
        regions = []
        for record in stored["regions"]:
            with Image.open(args.reuse_analysis.parent / f"{record['id']}.png") as mask:
                regions.append(selection.SelectionRegion(**{**record, "mask": mask.convert("L")}))
        analysis = selection.SelectionAnalysis(stored["fingerprint"], source.size, regions, stored["message"])
    else:
        analysis = selection.analyze_elements(source)
    selection.render_selection(source, analysis).save(destination / "inventory.png")
    measured = []
    for region in analysis.regions:
        region.mask.save(destination / f"{region.id}.png")
        measured.append({"id": region.id, "label": region.label, "kind": region.kind,
                         "bbox": region.bbox, "coverage": region.coverage,
                         "components": region.components, "peak": region.peak})
    metrics = {"source": str(args.source.resolve()), "size": source.size,
               "fingerprint": analysis.fingerprint, "message": analysis.message,
               "analysis_seconds": time.monotonic() - started, "regions": measured}
    if args.reuse_analysis:
        metrics["analysis_reused_from"] = str(args.reuse_analysis.resolve())
        if "sam2_provenance" in stored:
            metrics["sam2_provenance"] = stored["sam2_provenance"]
    evidence = destination / "results.json"
    evidence.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)
    print(f"EVIDENCE {destination}", flush=True)
    if not analysis.regions:
        raise RuntimeError("No measured regions; no selector capability demonstrated")
    if not args.select:
        return
    ids = []
    for chosen in args.select:
        matches = [region.id for region in analysis.regions if chosen in (region.id, region.label)]
        if len(matches) != 1:
            raise ValueError(f"Selection must match exactly one measured region: {chosen}")
        ids.extend(matches)
    valid, reason = selection.validate_selection(source, analysis, ids)
    assert valid, reason
    selection.render_selection(source, analysis, ids).save(destination / "selected.png")
    if not args.prompt:
        return
    from roop.comfy_workflows import get_comfyui_url
    session = requests.Session()
    session.trust_env = False
    queue_response = session.get(f"{get_comfyui_url()}/queue", timeout=5)
    queue_response.raise_for_status()
    queue = queue_response.json()
    if queue.get("queue_running") or queue.get("queue_pending"):
        raise RuntimeError("ComfyUI occupied; the test does not enqueue")
    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    from roop.img_editor.img_editor_manager import ImgEditorManager
    client = get_flux_edit_comfy_client()
    real_generate = client.generate
    generations = []

    def observe_generate(**kwargs):
        index = len(generations)
        kwargs["image"].save(destination / f"reference_{index}.png")
        if args.canvas_side:
            kwargs["reference_canvas"] = {"max_side": args.canvas_side, "multiple": 64}
        if args.direct_prompt:
            from roop.img_editor.prompt_translator import translate_prompt
            kwargs["prompt"] = translate_prompt(args.prompt)
        old_encoder = client._clip_name2
        if args.text_encoder:
            from roop.img_editor.flux_edit_comfy_client import get_models_base
            if Path(args.text_encoder).name != args.text_encoder:
                raise ValueError("Encoder name must be a local registered filename")
            if not (Path(get_models_base()) / "text_encoders" / args.text_encoder).is_file():
                raise FileNotFoundError(args.text_encoder)
            client._clip_name2 = args.text_encoder
        generations.append({"size": kwargs["image"].size,
                            "seed": kwargs["seed"],
                            "steps": kwargs["num_inference_steps"],
                            "guidance": kwargs["guidance_scale"],
                            "prompt": kwargs["prompt"],
                            "text_encoder": client._clip_name2,
                            "mask_is_none": kwargs["mask_image"] is None})
        try:
            result, note = real_generate(**kwargs)
        finally:
            client._clip_name2 = old_encoder
        mapping = getattr(client, "_last_reference_canvas", None)
        if mapping is not None:
            generations[-1]["reference_canvas"] = asdict(mapping)
        if result is not None:
            result.image.save(destination / f"raw_{index}.png")
        return result, note

    client.generate = observe_generate
    original_settings_loader = selection.load_settings
    if args.crop_context_fraction is not None or args.object_context_fraction is not None:
        trial_settings = original_settings_loader()
        if args.crop_context_fraction is not None:
            trial_settings["edit"]["crop_context_fraction"] = args.crop_context_fraction
            metrics["crop_context_fraction_override"] = args.crop_context_fraction
        if args.object_context_fraction is not None:
            trial_settings["edit"]["object_context_fraction"] = args.object_context_fraction
            metrics["object_context_fraction_override"] = args.object_context_fraction
        selection.load_settings = lambda: trial_settings
    metrics["requested_seed"] = args.seed
    metrics["selection_edit_settings"] = selection.load_settings()["edit"]
    started = time.monotonic()
    try:
        output, note, mask = ImgEditorManager().generate_intelligent(
            image=source, prompt=args.prompt, engine="kontext", seed=args.seed,
            selection_analysis=analysis, selected_region_ids=ids, enhance_faces=False,
            guidance_scale=args.guidance, num_inference_steps=args.steps,
        )
    finally:
        client.generate = real_generate
        selection.load_settings = original_settings_loader
    if output is None or mask is None:
        raise RuntimeError(note)
    output.save(destination / "result.png")
    mask.save(destination / "effective_mask.png")
    changed = np.any(np.asarray(source) != np.asarray(output), axis=2)
    exterior = np.asarray(mask) == 0
    metrics.update({"edit_seconds": time.monotonic() - started, "instruction": args.prompt,
                    "selected_ids": ids, "generations": generations, "edit_message": note,
                    "changed_pixels": int(changed.sum()),
                    "exterior_changed_pixels": int(changed[exterior].sum()),
                    "scope": "Real inference/protection; visual adherence requires inspection"})
    evidence.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in metrics.items() if key != "regions"},
                     indent=2, ensure_ascii=False), flush=True)
    print(f"EVIDENCE {destination}", flush=True)
    assert len(generations) == 1 and generations[0]["mask_is_none"], generations
    assert changed.any(), "No pixels changed"
    assert not changed[exterior].any(), "Exterior changed"


if __name__ == "__main__":
    main()
