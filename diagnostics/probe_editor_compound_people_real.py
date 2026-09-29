"""Replay the two-person compound edit without changing the live UI state."""
from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]
source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1790338257.png").convert("RGB")
source = source.resize((800, 533), Image.Resampling.LANCZOS)
prompt = ("la mujer de rodillas va descalza y sin medias, "
          "el hombre de rodillas debe ser calvo")
result, message, mask = get_img_editor_manager().generate_intelligent(
    image=source, prompt=prompt, engine="longcat", use_rewriter=True,
    enhance_faces=True, auto_upscale=False,
)
assert result is not None, message
out = ROOT / "diagnostics/editor_compound_people_after.png"
result.save(out)
print(f"output={out} size={result.size} mask={mask is not None} message={message}")
