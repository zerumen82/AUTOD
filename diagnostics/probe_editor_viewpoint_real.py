"""Run one benign office viewpoint edit through the real manager and ComfyUI."""

from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Image.open(ROOT / "diagnostics/img_editor_scene_office_final.png").convert("RGB")
    output, message, _ = get_img_editor_manager().generate_intelligent(
        image=source,
        prompt="Show this same office scene from a clearly overhead camera angle. Keep the same people, desks, chairs, windows and plant recognizable.",
        engine="longcat",
        use_semantic=False,
        use_rewriter=False,
        enhance_faces=False,
        auto_upscale=False,
    )
    assert output is not None, message
    path = ROOT / "diagnostics/editor_viewpoint_full_real.png"
    output.save(path)
    print(f"SAVED {path} {output.size} {message}", flush=True)


if __name__ == "__main__":
    main()
