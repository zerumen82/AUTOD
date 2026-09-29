"""Measure one-pass LongCat Turbo adherence for the user's compound image."""
from pathlib import Path

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client


root = Path(__file__).resolve().parents[1]
source = Image.open(root / "ui/tob/ComfyUI/input/fast_1790338257.png").convert("RGB")
client = get_flux_edit_comfy_client()
loaded, message = client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")
assert loaded, message
result, message = client.generate(
    image=source,
    prompt=(
        "Instruction: Edit this exact photo. The woman on her knees must be barefoot "
        "without stockings, and the man on his knees must be bald. Apply both changes "
        "completely while keeping the same people and scene."
    ),
    num_inference_steps=12,
    guidance_scale=1.0,
    denoise=1.0,
    no_kontext=True,
)
assert result is not None, message
out = root / "diagnostics/editor_single_turbo_compound.png"
result.image.save(out)
print(f"output={out} size={result.image.size} message={message}")
