"""Replay the recorded first LongCat sample on its recorded input crop.

No model inference: this isolates region selection and pasting from generation.
"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

from PIL import Image
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

OUT = ROOT / "diagnostics/out/editor_head_paste_latest_replay_20261127"


class RecordedClient:
    def __init__(self, image):
        self.image = image
        self.calls = 0

    def load(self, *_args, **_kwargs):
        return True, "recorded"

    def generate(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(image=self.image), "recorded first pass"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1795733009.png").convert("RGB")
    raw = Image.open(ROOT / "ui/tob/ComfyUI/output/FastEdit_01438_.png").convert("RGB")
    client = RecordedClient(raw)
    manager = ImgEditorManager()
    integrate = manager._integrate_local_generation
    measure_request = manager._request_region_on_subject
    measure_head = manager._head_zone_for_subject
    measure_hair = manager._head_with_its_hair
    paste_args = []

    def recorded_request(*args, **kwargs):
        found = measure_request(*args, **kwargs)
        if found is not None:
            found.save(OUT / "request_seed.png")
        return found

    def recorded_head(*args, **kwargs):
        found = measure_head(*args, **kwargs)
        if found is not None:
            found.save(OUT / "head_measured.png")
        return found

    def recorded_hair(*args, **kwargs):
        subject = args[4] if len(args) > 4 else kwargs.get("subject_mask")
        if subject is not None and args[0].size == source.size:
            subject.save(OUT / "subject.png")
        found = measure_hair(*args, **kwargs)
        if found is not None and args[0].size == source.size:
            found.save(OUT / "head_refined.png")
        return found

    def recorded_paste(*args, **kwargs):
        paste_args.append(kwargs)
        print("source face boxes:", [tuple(map(float, f.bbox)) for f in kwargs.get("source_faces", ())])
        args[2].save(OUT / f"mask_{len(paste_args)}.png")
        return integrate(*args, **kwargs)

    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Man must be bald."),
        patch.object(manager, "_appearance_work_region", side_effect=lambda image, mask, person=None: (
            image, mask, (0, 0, image.width, image.height)
        )),
        patch.object(manager, "_integrate_local_generation", side_effect=recorded_paste),
        patch.object(manager, "_request_region_on_subject", side_effect=recorded_request),
        patch.object(manager, "_head_zone_for_subject", side_effect=recorded_head),
        patch.object(manager, "_head_with_its_hair", side_effect=recorded_hair),
    ):
        result, message, _ = manager.generate_intelligent(
            image=source, prompt="EL HOMBRE DEBE SER CALVO", engine="auto",
            scope_hint="features", use_rewriter=False, enhance_faces=False,
            auto_upscale=False,
        )
    print(f"result={result is not None}, calls={client.calls}, pastes={len(paste_args)}")
    print(f"paste flags: {[p.get('expand_head_support') for p in paste_args]}")
    print(f"message={message}")
    if result is not None:
        result.save(OUT / "replay.png")
        assert client.calls == 1 and len(paste_args) == 1
        assert paste_args[0].get("expand_head_support") is True
        source_pixels = np.asarray(source, dtype=np.int16)
        raw_pixels = np.asarray(raw, dtype=np.int16)
        result_pixels = np.asarray(result, dtype=np.int16)
        head = np.asarray(Image.open(OUT / "mask_1.png").convert("L")) >= 96
        interior = cv2.erode(head.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        exterior = cv2.dilate(head.astype(np.uint8), np.ones((81, 81), np.uint8)) == 0
        assert int(np.max(np.abs(result_pixels - raw_pixels)[interior])) == 0
        assert int(np.max(np.abs(result_pixels - source_pixels)[exterior])) == 0
        print("PASS: full generated head inside; original pixels outside measured support")
        print(OUT / "replay.png")


if __name__ == "__main__":
    main()
