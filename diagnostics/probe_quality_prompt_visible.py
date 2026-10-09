"""Run the desktop Quality + prompt route on a real office photo."""
from pathlib import Path
import time

from PIL import Image

from ui.tabs.img_editor.events import on_generate


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics" / "img_editor_scene_office_final.png"
OUTPUT = ROOT / "diagnostics" / "quality_prompt_visible_current.png"


def main():
    source = Image.open(SOURCE).convert("RGB")
    start = time.monotonic()
    result, message, _ = on_generate(
        source, "Make only the office walls blue", "longcat", 0.0,
        False, 1, True, "hd", False,
    )
    elapsed = time.monotonic() - start
    assert result is not None, message
    result.save(OUTPUT)
    print(f"[PROBE] elapsed={elapsed:.1f}s size={result.size} output={OUTPUT}", flush=True)
    print(f"[PROBE] message={message}", flush=True)


if __name__ == "__main__":
    main()
