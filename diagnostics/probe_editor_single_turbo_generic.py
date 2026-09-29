"""Replay the translated compound instruction with a generic all-changes wrapper."""
from pathlib import Path

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client


root = Path(__file__).resolve().parents[1]
source = Image.open(root / "ui/tob/ComfyUI/input/fast_1790338257.png").convert("RGB")
request = (
    "the woman on her knees goes barefoot and without stockings, "
    "the man on his knees must be bald"
)
client = get_flux_edit_comfy_client()
loaded, message = client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
assert loaded, message
result, message = client.generate(
    image=source,
    prompt=("Instruction: Edit this exact photo. Apply every requested change: "
            f"{request}. Make all requested changes clearly visible."),
    num_inference_steps=12,
    guidance_scale=1.0,
    denoise=1.0,
    no_kontext=True,
)
assert result is not None, message
out = root / "diagnostics/editor_single_turbo_generic.png"
result.image.save(out)
print(f"output={out} size={result.image.size} message={message}")
