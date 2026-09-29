"""Check a person-grounded first clause from the compound request."""
from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]
source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1790338257.png").convert("RGB")
source = source.resize((800, 533), Image.Resampling.LANCZOS)
output, message, _ = get_img_editor_manager().generate_intelligent(
    image=source,
    prompt="la mujer de rodillas va descalza y sin medias",
    engine="longcat", use_rewriter=True, enhance_faces=False,
    auto_upscale=False, _compound_depth=1,
    _compound_region_hint="woman",
)
assert output is not None, message
path = ROOT / "diagnostics/editor_woman_region_after.png"
output.save(path)
print(f"output={path} size={output.size} message={message}")
