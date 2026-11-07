"""Real neutral LongCat timing probe; uses the existing idle backend only.

No inference doubles, installations, restarts or server teardown. The outgoing
workflow and response are recorded while forwarding the real HTTP request.
Memory samples describe this run, never the historical incident.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import requests  # noqa: E402
from PIL import Image  # noqa: E402
from skimage import data  # noqa: E402

from roop.img_editor.edit_execution import begin_execution, end_execution  # noqa: E402
from roop.img_editor.comfy_progress import stop_comfy_prompt  # noqa: E402
from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient  # noqa: E402
from roop.img_editor.gen_prompt_config import get_longcat_model_config  # noqa: E402
from roop.comfy_workflows import get_comfyui_url  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="neutral")
    parser.add_argument("--seed", type=int, default=424242)
    args = parser.parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = ROOT / "diagnostics/out/editor_stall_20261105/fix" / f"real_{args.label}_{stamp}"
    out.mkdir(parents=True)

    def save(name, payload):
        (out / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    base = get_comfyui_url()

    def get(endpoint):
        response = requests.get(f"{base}/{endpoint}", timeout=10)
        response.raise_for_status()
        return response.json()

    def assert_idle():
        queue = get("queue")
        if queue.get("queue_running") or queue.get("queue_pending"):
            raise RuntimeError("Backend busy; no render submitted")
        return queue

    save("queue_before.json", assert_idle())
    save("system_before.json", get("system_stats"))
    save("sources_sha256.json", {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in (
            "roop/img_editor/comfy_progress.py", "roop/img_editor/flux_edit_comfy_client.py",
            "roop/img_editor/selected_region_edit.py", "config/generation_prompt.json",
            "config/model_configs.json", "diagnostics/probe_editor_wait_real_20261105.py",
        )
    })
    # Match the reported local crop's input shape; the client keeps its normal
    # effective resolution policy. This resize is a timing fixture, not a
    # photographic quality reference or an exact A/B with the user's image.
    source = Image.fromarray(data.coffee()).resize((768, 672), Image.Resampling.LANCZOS)
    source.save(out / "source.png")
    mask = Image.new("L", source.size, 255)
    mask.save(out / "mask.png")
    prompt = "Make the ceramic coffee cup cobalt blue. Keep the coffee, plate, table, spoon, lighting and exact framing unchanged."
    model_file = get_longcat_model_config()["model_file"]
    save("dispatch.json", {
        "label": args.label, "model": model_file, "prompt": prompt,
        "seed": args.seed, "steps": 12, "guidance_scale": 1.0, "denoise": 1.0,
        "input_size": source.size, "mask": "all photo", "no_kontext": True,
        "keep_source_pixels": True, "preserve_frame": True,
        "purpose": "neutral timing reproduction; no claim of cold cache or historical A/B",
    })
    stop = threading.Event()
    sampling_errors = []
    t0 = time.monotonic()

    def sample_gpu():
        with (out / "gpu.jsonl").open("w", encoding="utf-8") as stream:
            while not stop.is_set():
                try:
                    result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.free,power.draw",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True, timeout=10,
                    )
                    stream.write(json.dumps({"elapsed_s": time.monotonic() - t0,
                                             "csv": result.stdout.strip()}) + "\n")
                    stream.flush()
                except (OSError, subprocess.SubprocessError) as error:
                    sampling_errors.append(str(error))
                if stop.wait(15):
                    break

    progress_file = (out / "progress.jsonl").open("w", encoding="utf-8")

    def progress(prog):
        progress_file.write(json.dumps({"probe_elapsed_s": time.monotonic() - t0,
                                       **prog}, ensure_ascii=False) + "\n")
        progress_file.flush()

    actual_post = requests.post
    prompt_ids = []

    def record_post(url, *values, **kwargs):
        is_prompt = url == f"{base}/prompt"
        if is_prompt:
            assert_idle()
            save("workflow_submitted.json", kwargs.get("json"))
        response = actual_post(url, *values, **kwargs)
        if is_prompt:
            save("submission.json", response.json())
            if response.ok and response.json().get("prompt_id"):
                prompt_ids.append(response.json()["prompt_id"])
        return response

    report, token = begin_execution("longcat")
    monitor = threading.Thread(target=sample_gpu, name="neutral-gpu-sampler", daemon=True)
    monitor.start()
    metrics = {"started_utc": datetime.now(timezone.utc).isoformat(), "output_dir": str(out)}
    requests.post = record_post
    try:
        client = FluxEditComfyClient()
        load_start = time.monotonic()
        loaded, note = client.load(model_file, allow_vision_engine=True)
        metrics.update({"loaded": loaded, "load_note": note,
                        "load_seconds": time.monotonic() - load_start})
        if not loaded:
            raise RuntimeError(note)
        generate_start = time.monotonic()
        result, note = client.generate(
            image=source, mask_image=mask, prompt=prompt, seed=args.seed,
            num_inference_steps=12, guidance_scale=1.0, denoise=1.0,
            prompt_authority=True, preserve_frame=True, keep_source_pixels=True,
            no_kontext=True, progress_callback=progress,
        )
        metrics.update({"generate_seconds": time.monotonic() - generate_start,
                        "result_note": note, "image_returned": result is not None,
                        "policy_rejection": report.rejection})
        if result is None:
            raise RuntimeError(note)
        result.image.save(out / "result.png")
        metrics["output_size"] = result.image.size
        metrics["image_valid"] = result.image.width > 0 and result.image.height > 0
    except KeyboardInterrupt:
        metrics["interrupted"] = True
        metrics["own_prompt_stopped"] = {
            prompt_id: stop_comfy_prompt(base, prompt_id) for prompt_id in prompt_ids
        }
        # If interruption happened before a response supplied an ID, do not
        # interrupt an unknown job on the shared backend.
        raise
    except Exception as error:
        metrics["error"] = repr(error)
        raise
    finally:
        requests.post = actual_post
        stop.set()
        monitor.join(timeout=12)
        progress_file.close()
        end_execution(token)
        metrics["elapsed_monotonic_seconds"] = time.monotonic() - t0
        metrics["finished_utc"] = datetime.now(timezone.utc).isoformat()
        metrics["sampling_errors"] = sampling_errors
        metrics["prompt_ids"] = prompt_ids
        for prompt_id in prompt_ids:
            try:
                history = get(f"history/{prompt_id}")
                save(f"history_{prompt_id}.json", history)
                status = history[prompt_id]["status"]
                events = {kind: payload.get("timestamp") for kind, payload in status["messages"]}
                metrics["server_status"] = status["status_str"]
                if events.get("execution_start") and events.get("execution_success"):
                    metrics["server_wall_seconds"] = (
                        events["execution_success"] - events["execution_start"]
                    ) / 1000.0
            except Exception as error:
                metrics["history_error"] = repr(error)
        try:
            save("queue_after.json", get("queue"))
            save("system_after.json", get("system_stats"))
        except Exception as error:
            metrics["final_state_error"] = repr(error)
        save("metrics.json", metrics)
        print(json.dumps(metrics, indent=2), flush=True)
    if (not metrics.get("image_valid") or metrics.get("server_status") != "success"
            or len(prompt_ids) != 1 or metrics.get("policy_rejection")
            or metrics.get("history_error") or metrics.get("final_state_error")):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
