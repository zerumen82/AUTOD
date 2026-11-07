"""Observe the REAL Editor UI callback against an already running idle backend.

No inference/semantic/recipe overrides and no seed override: on_generate does not
accept a seed. Source, actual dispatches, queued workflows and raw/final images
are retained. Finishing a callback is not a visual PASS. This probe never starts,
stops, closes or interrupts ComfyUI.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import functools
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import time
import traceback
from urllib.parse import urlparse
from unittest.mock import patch

import numpy as np
from PIL import Image
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SOURCES = (
    "AGENTS.md", "run.py", "roop/core.py", "MEMORY.md",
    "config/editor_selection.json", "config/generation_prompt.json", "config/model_configs.json",
    "roop/img_editor/img_editor_manager.py", "roop/img_editor/qwen_camera_comfy_client.py",
    "roop/img_editor/flux_edit_comfy_client.py", "roop/img_editor/selected_region_edit.py",
    "roop/img_editor/nlp/semantic_analyzer.py", "roop/img_editor/edit_execution.py",
    "roop/img_editor/prompt_translator.py", "ui/tabs/img_editor/events.py",
    "roop/img_editor/top_tier_finish.py", "roop/img_editor/image_quality_pipeline.py",
    "config/quality_mode.json",
    "diagnostics/probe_editor_versatility_real_20261105.py",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def source_hashes():
    return {name: sha256_file(ROOT / name) for name in SOURCES if (ROOT / name).is_file()}


def image_hash(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def describe(value, out, stem):
    """Persist image/array arguments without changing objects sent to production."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Image.Image):
        filename = f"{stem}.png"
        value.save(out / filename)
        return {"image": filename, "mode": value.mode, "size": list(value.size),
                "pixel_sha256": image_hash(value)}
    if isinstance(value, np.ndarray):
        filename = f"{stem}.npy"
        np.save(out / filename, value, allow_pickle=False)
        return {"array": filename, "shape": list(value.shape), "dtype": str(value.dtype),
                "sha256": sha256_file(out / filename)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): describe(item, out, f"{stem}_{safe_name(key)}")
                for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        sequence = sorted(value) if isinstance(value, set) else value
        return [describe(item, out, f"{stem}_{index}") for index, item in enumerate(sequence)]
    if is_dataclass(value):
        return {field.name: describe(getattr(value, field.name), out, f"{stem}_{field.name}")
                for field in fields(value)}
    if callable(value):
        return {"callable": f"{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__name__)}"}
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(value))


def queue_snapshot(base_url):
    response = requests.get(f"{base_url}/queue", timeout=10)
    response.raise_for_status()
    payload = response.json()
    if (not isinstance(payload, dict)
            or not isinstance(payload.get("queue_running"), list)
            or not isinstance(payload.get("queue_pending"), list)):
        raise RuntimeError("El backend no devuelve una cola ComfyUI válida")
    return {"at_utc": utc_now(), "base_url": base_url, "queue": payload,
            "idle": not payload["queue_running"] and not payload["queue_pending"]}


class Recorder:
    def __init__(self, out, engine, base_url):
        self.out = out
        self.engine = engine
        self.base_url = base_url
        self.calls = []
        self.workflows = []
        self.executions = []

    def observe_generate(self, kind, original):
        @functools.wraps(original)
        def observed(client, *args, **kwargs):
            from roop.img_editor.edit_execution import current_execution

            index = len(self.calls)
            stem = f"dispatch_{index:03d}"
            bound = inspect.signature(original).bind(client, *args, **kwargs)
            arguments = {name: value for name, value in bound.arguments.items() if name != "self"}
            call = {"index": index, "kind": kind, "at_utc": utc_now(),
                    "engine_requested": self.engine,
                    "client_model": str(getattr(client, "_flux_version", "")),
                    "arguments": describe(arguments, self.out, stem),
                    "execution_before": describe(current_execution(), self.out, f"{stem}_execution_before")}
            self.calls.append(call)
            # Written BEFORE invoking the real client, including actual references/masks.
            write_json(self.out / f"{stem}.json", call)
            started = time.monotonic()
            completion = {"index": index}
            try:
                returned = original(client, *args, **kwargs)
                result, note = returned
                completion.update(note=str(note), returned_image=result is not None)
                if result is not None:
                    raw = result.image
                    raw.save(self.out / f"raw_{index:03d}.png")
                    completion.update(raw_image=f"raw_{index:03d}.png", size=list(raw.size),
                                      pixel_sha256=image_hash(raw))
                return returned
            except BaseException as exc:
                completion.update(error_type=type(exc).__name__, error=str(exc))
                raise
            finally:
                completion.update(elapsed_s=time.monotonic() - started,
                                  execution_after=describe(current_execution(), self.out,
                                                           f"{stem}_execution_after"))
                call["completion"] = completion
                write_json(self.out / f"completion_{index:03d}.json", completion)
        return observed

    def observe_request(self, original):
        @functools.wraps(original)
        def observed(session, method, url, *args, **kwargs):
            parsed = urlparse(str(url))
            base = urlparse(self.base_url)
            is_queue = (str(method).upper() == "POST" and parsed.netloc == base.netloc
                        and parsed.path.rstrip("/") == "/prompt")
            if not is_queue:
                return original(session, method, url, *args, **kwargs)
            index = len(self.workflows)
            queue = queue_snapshot(self.base_url)
            write_json(self.out / f"queue_before_workflow_{index:03d}.json", queue)
            if not queue["idle"]:
                raise RuntimeError("Cola ocupada antes de inferir; no se encola ni interrumpe trabajo ajeno")
            workflow = kwargs.get("json")
            if workflow is None:
                workflow = kwargs.get("data")
                if isinstance(workflow, (str, bytes)):
                    try:
                        workflow = json.loads(workflow)
                    except (ValueError, TypeError):
                        workflow = {"unparsed_body_type": type(workflow).__name__}
            record = {"index": index, "at_utc": utc_now(),
                      "request": describe(workflow, self.out, f"workflow_{index:03d}")}
            self.workflows.append(record)
            write_json(self.out / f"workflow_{index:03d}.json", record)
            response = original(session, method, url, *args, **kwargs)
            try:
                queued = response.json()
            except ValueError:
                queued = {"response_not_json": True}
            record.update(http_status=response.status_code,
                          prompt_id=queued.get("prompt_id") if isinstance(queued, dict) else None)
            record["accepted"] = bool(200 <= response.status_code < 300 and record["prompt_id"])
            write_json(self.out / f"workflow_response_{index:03d}.json", {
                "status_code": response.status_code, "response": queued,
            })
            return response
        return observed


def pixel_metrics(source, result, mask):
    if result is None:
        return {"result_available": False, "changed_pixels": None,
                "exterior_changed_pixels": None, "exterior_max_delta": None}
    if source.size != result.size:
        return {"result_available": True, "same_dimensions": False,
                "source_size": list(source.size), "result_size": list(result.size),
                "changed_pixels": None, "exterior_changed_pixels": None,
                "exterior_max_delta": None}
    delta = np.abs(np.asarray(source.convert("RGB"), dtype=np.int16)
                   - np.asarray(result.convert("RGB"), dtype=np.int16))
    changed = np.any(delta != 0, axis=2)
    metrics = {"result_available": True, "same_dimensions": True,
               "changed_pixels": int(changed.sum()), "max_delta": int(delta.max()),
               "exterior_changed_pixels": None, "exterior_max_delta": None,
               "mask_available": mask is not None}
    if mask is not None and mask.size == source.size:
        exterior = np.asarray(mask.convert("L")) == 0
        metrics.update(exterior_pixels=int(exterior.sum()),
                       exterior_changed_pixels=int((changed & exterior).sum()),
                       exterior_max_delta=int(delta[exterior].max()) if exterior.any() else 0,
                       mask_support_pixels=int((~exterior).sum()))
    else:
        metrics["exterior_scope_note"] = "Sin máscara final compatible: no se deduce protección del exterior"
    return metrics


def run_case(args):
    source_path = Path(args.source).resolve(strict=True)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    with Image.open(source_path) as opened:
        source = opened.convert("RGB")
    source.save(out / "source.png")
    before_hashes = source_hashes()
    core_text = (ROOT / "roop/core.py").read_text(encoding="utf-8-sig")
    version = re.search(r'^APP_VERSION\s*=\s*"([^"]+)"', core_text, flags=re.M)
    manifest = {"at_utc": utc_now(), "source": str(source_path),
                "source_file_sha256": sha256_file(source_path),
                "source_rgb_sha256": image_hash(source), "source_size": list(source.size),
                "prompt": args.prompt, "engine_requested": args.engine,
                "application_version": version.group(1) if version else None,
                "hashes_before": before_hashes,
                "seed_override": None, "recipe_override": None,
                "operational_guard": "Backend startup forbidden; existing callback and model methods execute unchanged",
                "scope": "Real on_generate callback; no inference or semantic doubles; visual verdict pending"}
    write_json(out / "run.json", manifest)

    # Lazy production imports keep --help free of app initialization.
    from roop.img_editor.edit_execution import current_execution
    from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient
    from roop.img_editor.qwen_camera_comfy_client import QwenCameraComfyClient
    from ui.tabs import comfy_launcher
    from ui.tabs.img_editor import events

    base_url = comfy_launcher.get_comfy_url()
    recorder = Recorder(out, args.engine, base_url)
    result = mask = None
    message = "Callback no ejecutado"
    error = None
    started = time.monotonic()
    final_queue = None
    original_end = events.end_execution

    def observe_end(token):
        report = describe(current_execution(), out, f"execution_{len(recorder.executions):03d}")
        recorder.executions.append(report)
        write_json(out / f"execution_{len(recorder.executions) - 1:03d}.json", report)
        return original_end(token)

    def forbid_start(*_args, **_kwargs):
        raise RuntimeError("Este probe requiere ComfyUI ya activo; no arranca ni reinicia servicios")

    try:
        initial_queue = queue_snapshot(base_url)
        write_json(out / "queue_before.json", initial_queue)
        if not initial_queue["idle"]:
            raise RuntimeError("ComfyUI está ocupado; se conserva su trabajo y no se ejecuta el callback")
        with ExitStack() as stack:
            stack.enter_context(patch.object(comfy_launcher, "start", new=forbid_start))
            stack.enter_context(patch.object(FluxEditComfyClient, "generate",
                                             new=recorder.observe_generate("flux_edit", FluxEditComfyClient.generate)))
            stack.enter_context(patch.object(QwenCameraComfyClient, "generate",
                                             new=recorder.observe_generate("qwen_camera", QwenCameraComfyClient.generate)))
            stack.enter_context(patch.object(requests.sessions.Session, "request",
                                             new=recorder.observe_request(requests.sessions.Session.request)))
            stack.enter_context(patch.object(events, "end_execution", new=observe_end))
            result, message, mask = events.on_generate(
                source, args.prompt, args.engine, 0.0, False, 1,
                args.quality, args.quality_tier,
                quality_preserve_faces=not args.no_face_restore,
                quality_upscale=not args.no_upscale,
                source="diagnostic_versatility_real_20261105",
            )
        if result is not None:
            result.save(out / "result.png")
        if mask is not None:
            mask.save(out / "mask.png")
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        (out / "exception.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        try:
            final_queue = queue_snapshot(base_url)
        except Exception as exc:
            final_queue = {"at_utc": utc_now(), "error": str(exc), "idle": None}
        write_json(out / "queue_after.json", final_queue)
        after_hashes = source_hashes()
        accepted = sum(bool(workflow.get("accepted")) for workflow in recorder.workflows)
        completed = sum(bool(call.get("completion", {}).get("returned_image")) for call in recorder.calls)
        metrics = {**manifest, **pixel_metrics(source, result, mask),
                   "message": str(message), "elapsed_s": time.monotonic() - started,
                   "error": error, "calls": recorder.calls,
                   "workflow_submission_attempts": len(recorder.workflows),
                   "accepted_workflows": accepted, "completed_model_images": completed,
                   "workflows": recorder.workflows,
                   "executions": recorder.executions, "queue_after": final_queue,
                   "hashes_after": after_hashes,
                   "sources_unchanged_during_run": before_hashes == after_hashes,
                   "visual_verdict": "NOT_EVALUATED",
                   "no_inference_observed": accepted == 0 and completed == 0}
        write_json(out / "metrics.json", metrics)
        print(json.dumps({"evidence": str(out), "result_available": result is not None,
                          "calls": len(recorder.calls), "accepted_workflows": accepted,
                          "elapsed_s": metrics["elapsed_s"], "message": str(message),
                          "error": error, "visual_verdict": "NOT_EVALUATED"}, indent=2), flush=True)
    return 1 if error is not None or result is None else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Existing input photo; preserved as source.png")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--engine", default="auto", choices=("auto", "kontext", "longcat", "qwen_camera"))
    parser.add_argument("--out", required=True, help="New evidence directory; must not exist")
    parser.add_argument("--quality", action="store_true", help="Enable the actual UI quality checkbox")
    parser.add_argument("--quality-tier", default="hd", choices=("hd", "4k"))
    parser.add_argument("--no-face-restore", action="store_true")
    parser.add_argument("--no-upscale", action="store_true")
    args = parser.parse_args()
    if not args.prompt.strip():
        parser.error("--prompt no puede estar vacío")
    try:
        return run_case(args)
    except Exception as exc:
        print(f"No se pudo preparar el probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
