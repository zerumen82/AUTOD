"""Exercise the manager's generic one-pass compound route on a saved input."""
from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


root = Path(__file__).resolve().parents[1]
source = Image.open(root / "ui/tob/ComfyUI/input/fast_1790338257.png").convert("RGB")
prompt = ("la mujer de rodillas va descalza y sin medias, "
          "el hombre de rodillas debe ser calvo")
result, message, mask = get_img_editor_manager().generate_intelligent(
    image=source, prompt=prompt, engine="longcat", use_rewriter=True,
    enhance_faces=True, auto_upscale=False,
)
assert result is not None, message
assert "un solo pase Turbo" in message, message
assert "cada cambio por separado" not in message, message
path = root / "diagnostics/editor_compound_people_fast_route.png"
result.save(path)
print(f"output={path} size={result.size} mask={mask is not None} message={message}")
