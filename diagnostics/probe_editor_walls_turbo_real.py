"""Compare a localized automatic mask on the saved slow second-pass input."""
from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1790324742.png").convert("RGB")
    source = source.resize((800, 533), Image.Resampling.LANCZOS)
    output, message, mask = get_img_editor_manager().generate_intelligent(
        image=source,
        prompt="gray walls",
        engine="longcat",
        use_rewriter=True,
        enhance_faces=False,
        auto_upscale=False,
        _compound_depth=1,
        _compound_region_hint="gray walls",
    )
    assert output is not None, message
    output_path = ROOT / "diagnostics/editor_walls_turbo_after.png"
    output.save(output_path)
    print(f"output={output_path} size={output.size} mask={mask is not None} message={message}")


if __name__ == "__main__":
    main()
