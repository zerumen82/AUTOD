"""Compare an actual local call with full-window noise, without production edits."""
import argparse
import json
from pathlib import Path
import time

from PIL import Image
import requests

from roop.comfy_workflows import get_comfyui_url
from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
from ui.tabs import comfy_launcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("call_directory")
    args = parser.parse_args()
    call = Path(args.call_directory)
    window = Image.open(call / "window.png").convert("RGB")
    metadata = json.loads((call / "meta.json").read_text(encoding="utf-8"))
    assert metadata["prompt_authority"] is True and metadata["no_kontext"] is False
    assert metadata["denoise"] == 1.0, "This comparison changes only the noise scope"
    process = None
    if comfy_launcher.is_comfyui_running():
        response = requests.get(f"{get_comfyui_url()}/queue", timeout=10)
        response.raise_for_status()
        queue = response.json()
        assert not queue["queue_running"] and not queue["queue_pending"], "Do not interrupt another render"
    else:
        ok, message, _ = comfy_launcher.start(directly_run=True)
        assert ok, message
        process = comfy_launcher.comfy_process
    try:
        client = get_flux_edit_comfy_client()
        ok, message = client.load(flux_version="LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
        assert ok, message
        keys = ("prompt", "negative_prompt", "num_inference_steps", "guidance_scale", "denoise",
                "seed", "keep_source_pixels", "preserve_frame", "prompt_authority", "no_kontext", "step_cap")
        kwargs = {key: metadata[key] for key in keys}
        started = time.perf_counter()
        result, message = client.generate(image=window, mask_image=None, **kwargs)
        assert result is not None, message
        result = result.image if hasattr(result, "image") else result
        output = Path("diagnostics/out") / f"global_noise_replay_{time.time_ns()}"
        output.mkdir()
        result.save(output / "sample.png")
        window.save(output / "window.png")
        before = window.resize(result.size, Image.Resampling.LANCZOS)
        strip = Image.new("RGB", (result.width * 2, result.height))
        strip.paste(before)
        strip.paste(result, (result.width, 0))
        strip.save(output / "compare.png")
        metadata.update({"parent_call": str(call), "noise_scope": "full window; no sampling mask",
                         "seconds": time.perf_counter() - started, "status": message,
                         "production_modified": False})
        (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(output, metadata, flush=True)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


if __name__ == "__main__":
    main()
