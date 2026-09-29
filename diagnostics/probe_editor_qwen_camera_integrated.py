"""Run the real editor's two-stage camera/color route on the office scene."""

from pathlib import Path

from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]


def main():
    source = Image.open(ROOT / "diagnostics/img_editor_scene_office_final.png")
    result, note, _ = ImgEditorManager().generate_intelligent(
        source,
        "Show the room from an overhead camera angle, make the wall blue",
        engine="qwen_camera",
        use_semantic=False,
        use_rewriter=False,
        enhance_faces=False,
        seed=123456,
    )
    assert result is not None, note
    assert "Qwen Camera" in note and "LongCat Turbo" in note, note
    target = ROOT / "diagnostics/editor_qwen_camera_integrated.png"
    result.save(target)
    print(f"SAVED {target} {result.size} {note}", flush=True)


if __name__ == "__main__":
    main()
