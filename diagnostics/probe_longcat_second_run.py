"""Short same-model ComfyUI run to measure the second edit on an 8 GB GPU."""
import time
import sys
from pathlib import Path

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
from ui.tabs import comfy_launcher


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "ui/tob/ComfyUI/input/fast_1790198919.png"
MODEL = "LongCat-Image-Edit-Q4_K_S.gguf"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def main():
    assert INPUT.is_file(), INPUT
    ok, message, port = comfy_launcher.start(directly_run=True)
    assert ok and port == 8188, message
    process = comfy_launcher.comfy_process
    try:
        client = get_flux_edit_comfy_client()
        image = Image.open(INPUT).convert("RGB")
        for run in (1, 2):
            started = time.monotonic()
            loaded, detail = client.load(MODEL)
            assert loaded, detail
            load_seconds = time.monotonic() - started
            result, detail = client.generate(
                image=image,
                prompt="Edit this photo: make the man bald. Preserve all other content.",
                num_inference_steps=STEPS,
                guidance_scale=2.5,
                denoise=0.74,
                seed=12345 + run,
                reference_latents_method="index",
            )
            assert result is not None, detail
            print(
                f"[PROBE] run={run} steps={STEPS} load_s={load_seconds:.1f} "
                f"generate_s={time.monotonic()-started-load_seconds:.1f}",
                flush=True,
            )
    finally:
        # Only stop the subprocess this probe itself started. Never sweep port 8188.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except Exception:
                process.kill()


if __name__ == "__main__":
    main()
