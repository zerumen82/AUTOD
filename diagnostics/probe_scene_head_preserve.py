"""Compare head-only preservation for a scene replacement."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.face_util import get_face_analyser, detect_faces_robust
from roop.img_editor.clipseg_masker import get_clipseg_masker
from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1790203567.png"
GENERATED = ROOT / "ui/tob/ComfyUI/output/FastEdit_00595_.png"
OUTPUT = ROOT / "diagnostics/scene_head_preserve.png"


def main():
    source = Image.open(SOURCE).convert("RGB")
    generated = Image.open(GENERATED).convert("RGB").resize(source.size)
    bgr = cv2.cvtColor(np.asarray(source), cv2.COLOR_RGB2BGR)
    faces = detect_faces_robust(bgr, get_face_analyser(), 0.20)
    print(f"[PROBE] faces={len(faces)}", flush=True)
    assert len(faces) == 2
    manager = ImgEditorManager()
    masker = get_clipseg_masker()
    background_mask = manager._background_mask_from_people(source, masker, faces)
    assert background_mask is not None
    final, status, head_mask = manager._compose_scene_preserving_heads(
        source, generated, background_mask, faces, masker,
    )
    assert final is not None, status
    assert head_mask is not None
    final.save(OUTPUT)
    masker.unload()
    print(f"[PROBE] output={OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
