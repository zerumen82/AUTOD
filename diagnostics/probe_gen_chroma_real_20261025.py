"""Real neutral Chroma render and workflow evidence; never installs models."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from roop.img_editor.flux_gen_comfy_client import FluxGenComfyClient
    from ui.tabs import comfy_launcher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=(
        "A red ceramic teapot beside a blue cup on a wooden table, "
        "natural window light, photograph."
    ))
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=123456)
    args = parser.parse_args()
    out = ROOT / "diagnostics/out/chroma_real_20261025" / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    client = FluxGenComfyClient()
    loaded, message = client.load("chroma")
    assert loaded, message
    workflow = {}
    real_post = requests.post

    def observe_post(url, **kwargs):
        body = kwargs.get("json") or {}
        if str(url).endswith("/prompt") and "prompt" in body:
            workflow.update(body["prompt"])
            (out / "workflow.json").write_text(json.dumps(workflow, indent=2), encoding="utf-8")
        return real_post(url, **kwargs)

    started = time.monotonic()
    previous_process = comfy_launcher.comfy_process
    try:
        running, message, _port = comfy_launcher.start(directly_run=True, auto_update=False)
        assert running, message
        queue = requests.get(f"{comfy_launcher.get_comfy_url()}/queue", timeout=10).json()
        assert not queue.get("queue_running") and not queue.get("queue_pending"), "ComfyUI is busy"
        with patch("roop.img_editor.flux_gen_comfy_client.requests.post", side_effect=observe_post):
            result, message = client.generate_ai(
                prompt=args.prompt, width=args.width, height=args.height,
                seed=args.seed, inject_nsfw=False, enhance_quality=False,
                lora_name="None", prompt_modifiers={
                    "image_type": "photoreal", "lighting": "natural",
                    "framing": "auto", "skin_finish": "auto",
                    "color_grade": "auto", "use_rewriter": False,
                },
            )
        assert result is not None, message
        result.image.save(out / "result.png")
        samplers = [node["inputs"] for node in workflow.values()
                    if node.get("class_type") == "KSampler"]
        loaders = [node["inputs"] for node in workflow.values()
                   if node.get("class_type") == "CLIPLoaderGGUF"]
        assert len(samplers) == 1 and samplers[0]["cfg"] == client._get_model_conf()["cfg"]
        assert len(loaders) == 1 and loaders[0]["type"] == "chroma"
        assert sum("CLIPLoader" in node.get("class_type", "") for node in workflow.values()) == 1
        assert loaders[0]["clip_name"] == client._get_model_conf()["text_encoder"]
        tokenizers = [(key, node["inputs"]) for key, node in workflow.items()
                      if node.get("class_type") == "T5TokenizerOptions"]
        assert len(tokenizers) == 1
        tokenizer_id, tokenizer = tokenizers[0]
        for key in ("min_padding", "min_length"):
            assert tokenizer[key] == int(client._get_model_conf()["tokenizer"][key])
        encoders = [node["inputs"] for node in workflow.values()
                    if node.get("class_type") == "CLIPTextEncode"]
        assert len(encoders) == 2 and all(node["clip"] == [tokenizer_id, 0] for node in encoders)
        assert result.image.size == (args.width, args.height)
        assert not any(node.get("class_type") == "FluxGuidance" for node in workflow.values())
        metrics = {
            "scope": "real GPU Chroma render, single T5 and configured CFG; inspect image separately",
            "prompt": args.prompt, "seed": args.seed, "size": list(result.image.size),
            "model": client._flux_version, "cfg": samplers[0]["cfg"],
            "text_encoder": loaders[0], "message": message,
            "elapsed_s": time.monotonic() - started,
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(json.dumps(metrics, indent=2), flush=True)
        print(f"EVIDENCE {out}", flush=True)
    finally:
        process = comfy_launcher.comfy_process
        if process is not None and process is not previous_process and process.poll() is None:
            try:
                queue = requests.get(f"{comfy_launcher.get_comfy_url()}/queue", timeout=10).json()
            except requests.RequestException as error:
                print(f"Cleanup skipped: cannot establish that the owned server is idle: {error}")
            else:
                if not queue.get("queue_running") and not queue.get("queue_pending"):
                    process.terminate()
                    process.wait(timeout=15)


if __name__ == "__main__":
    main()
