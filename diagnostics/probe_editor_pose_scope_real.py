"""Reproduce the user's localized attribute edit after the pose scope fix."""
from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1790332402.png").convert("RGB")
    source = source.resize((400, 600), Image.Resampling.LANCZOS)
    output, message, mask = get_img_editor_manager().generate_intelligent(
        image=source,
        prompt="EL HOMBRE DE RODILLAS DEBE SER CALVO",
        engine="longcat",
        use_rewriter=True,
        enhance_faces=True,
        auto_upscale=False,
    )
    assert output is not None, message
    output_path = ROOT / "diagnostics/editor_pose_scope_after.png"
    output.save(output_path)
    print(f"output={output_path} size={output.size} mask={mask is not None} message={message}")


if __name__ == "__main__":
    main()
