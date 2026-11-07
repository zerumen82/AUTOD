"""Real UI callback on the installed, non-explicit skimage astronaut sample.

No inference doubles. Source and raw reference/output are retained per run.
Only a ComfyUI process started by this probe may be stopped, and only idle.
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

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client  # noqa: E402
from ui.tabs import comfy_launcher  # noqa: E402
from ui.tabs.img_editor.events import on_generate  # noqa: E402
from ui.tabs.img_editor.state import SelectionSession  # noqa: E402


def run_case(args):
    samples = {
        "astronaut": "venv/Lib/site-packages/skimage/data/astronaut.png",
        "grace": "venv/Lib/site-packages/matplotlib/mpl-data/sample_data/grace_hopper.jpg",
    }
    source = ROOT / samples[args.sample]
    photo = Image.open(source).convert("RGB")
    out = ROOT / "diagnostics/out/appearance_reference_20261023" / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    photo.save(out / "source.png")
    session = None
    if args.mode == "selected":
        analysis = es.analyze_elements(photo)
        (out / "analysis.json").write_text(json.dumps({
            "message": analysis.message,
            "regions": [{"id": r.id, "kind": r.kind, "label": r.label,
                         "bbox": r.bbox, "coverage": r.coverage} for r in analysis.regions],
        }, indent=2), encoding="utf-8")
        heads = [r for r in analysis.regions if r.kind == "head"]
        assert len(heads) == 1, [(r.id, r.label) for r in heads]
        candidates = [r for r in analysis.regions if r.kind == "object"
                      and es._head_inside_selection(heads[0].mask, r.mask,
                          es.load_settings()["edit"]["head_inclusion_coverage"])]
        assert candidates, "inventory has no person region containing the measured head"
        selected = max(candidates, key=lambda region: region.coverage)
        selected.mask.save(out / "selection.png")
        heads[0].mask.save(out / "head.png")
        session = SelectionSession()
        session.replace_image(analysis.fingerprint)
        session.analysis = analysis
        session.selected_ids = [selected.id]
        print(f"REAL selection={selected.id}: {selected.label}", flush=True)
    client = get_flux_edit_comfy_client()
    original = client.generate
    calls = []

    def observe(**kwargs):
        index = len(calls)
        original_prompt = kwargs["prompt"]
        if getattr(args, "prompt_template", None):
            kwargs["prompt"] = args.prompt_template.format(instruction=args.prompt)
        if getattr(args, "seed", None) is not None:
            kwargs["seed"] = args.seed
        kwargs["image"].save(out / f"reference_{index}.png")
        calls.append({"reference_size": kwargs["image"].size,
                      "prompt": kwargs["prompt"], "steps": kwargs["num_inference_steps"],
                      "seed": kwargs.get("seed")})
        (out / f"dispatch_{index}.json").write_text(json.dumps({
            "original_prompt": original_prompt,
            "requested_engine": getattr(args, "engine", "kontext"),
            "prompt_template_override": getattr(args, "prompt_template", None),
            "client_arguments": {key: kwargs.get(key) for key in (
                "prompt", "negative_prompt", "num_inference_steps", "guidance_scale",
                "seed", "denoise", "no_kontext", "reference_latents_method",
            )},
        }, indent=2), encoding="utf-8")
        result, note = original(**kwargs)
        calls[index]["note"] = note
        if result is not None:
            result.image.save(out / f"raw_{index}.png")
        return result, note

    client.generate = observe
    started = time.monotonic()
    try:
        result, message, mask = on_generate(
            photo, args.prompt, getattr(args, "engine", "kontext"), 0.0,
            False, 1, False, "hd", source="diagnostic_neutral_sample",
            selection_state=session,
        )
        assert result is not None, message
        result.save(out / "result.png")
        if mask is not None:
            mask.save(out / "mask.png")
        before, after = np.asarray(photo), np.asarray(result)
        changed = np.any(before != after, axis=2)
        exterior = np.asarray(mask) == 0 if mask is not None else np.ones(changed.shape, bool)
        metrics = {"mode": args.mode, "engine_requested": getattr(args, "engine", "kontext"),
                   "source": str(source), "message": message,
                   "prompt": args.prompt,
                   "generation_prompt_override": getattr(args, "prompt_template", None),
                   "elapsed_s": time.monotonic() - started, "calls": calls,
                   "changed_pixels": int(changed.sum()),
                   "exterior_changed_pixels": int((changed & exterior).sum()),
                   "scope": "Real UI callback/inference; visual adherence requires inspection"}
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        print(f"EVIDENCE: {out}", flush=True)
        assert len(calls) == 1, "expected exactly one real model pass"
        assert changed.any(), "result is identical"
        assert not (changed & exterior).any(), "pixels outside measured region changed"
    finally:
        client.generate = original
        process = comfy_launcher.comfy_process
        if process is not None and process.poll() is None:
            queue = requests.get(f"{comfy_launcher.get_comfy_url()}/queue", timeout=5).json()
            if not queue.get("queue_running") and not queue.get("queue_pending"):
                process.terminate()
                process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("selected", "global"), default="selected")
    parser.add_argument("--sample", choices=("astronaut", "grace"), default="astronaut")
    parser.add_argument("--engine", choices=("kontext", "auto", "longcat"), default="kontext")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt-template", help="Explicit diagnostic override; recorded in metrics")
    parser.add_argument("--prompt", default="La persona debe ser calva")
    parser.add_argument("--batch-global", action="store_true")
    args = parser.parse_args()
    if args.batch_global:
        args.mode = "global"
        for sample, prompt in (("astronaut", "La persona debe ser calva"),
                               ("grace", "La persona debe tener el pelo azul")):
            args.sample, args.prompt = sample, prompt
            run_case(args)
    else:
        run_case(args)


if __name__ == "__main__":
    main()
