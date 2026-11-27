"""Replay the saved neutral sample without running a generative model."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

SAMPLE = ROOT / "diagnostics/out/appearance_reference_20261023/1795764767693508900"
OUT = ROOT / "diagnostics/out/editor_grace_replay_20261127"


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
    source = Image.open(SAMPLE / "source.png").convert("RGB")
    raw = Image.open(SAMPLE / "raw_0.png").convert("RGB")
    client = RecordedClient(raw)
    manager = ImgEditorManager()
    integrate = manager._integrate_local_generation
    measure_head = manager._head_zone_for_subject
    measure_hair = manager._head_with_its_hair
    measure_request = manager._request_region_on_subject
    calls = []

    def recorded_paste(*args, **kwargs):
        calls.append(kwargs)
        print("source face boxes:", [tuple(map(float, f.bbox)) for f in kwargs.get("source_faces", ())])
        args[2].save(OUT / f"mask_{len(calls)}.png")
        result = integrate(*args, **kwargs)
        result.save(OUT / f"paste_{len(calls)}.png")
        return result

    def recorded_head(*args, **kwargs):
        mask = measure_head(*args, **kwargs)
        if mask is not None:
            mask.save(OUT / "head.png")
        return mask

    def recorded_hair(*args, **kwargs):
        subject = args[4] if len(args) > 4 else kwargs.get("subject_mask")
        if subject is not None:
            subject.save(OUT / "subject.png")
        mask = measure_hair(*args, **kwargs)
        if mask is not None:
            mask.save(OUT / "head_refined.png")
        return mask

    def recorded_request(*args, **kwargs):
        mask = measure_request(*args, **kwargs)
        if mask is not None:
            mask.save(OUT / "request_seed.png")
        return mask

    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=client),
        patch("roop.img_editor.prompt_translator.translate_prompt",
              return_value="The person must be bald."),
        patch.object(manager, "_appearance_work_region", side_effect=lambda image, mask, person=None: (
            image, mask, (0, 0, image.width, image.height)
        )),
        patch.object(manager, "_integrate_local_generation", side_effect=recorded_paste),
        patch.object(manager, "_head_zone_for_subject", side_effect=recorded_head),
        patch.object(manager, "_head_with_its_hair", side_effect=recorded_hair),
        patch.object(manager, "_request_region_on_subject", side_effect=recorded_request),
    ):
        result, message, _ = manager.generate_intelligent(
            image=source, prompt="La persona debe ser calva", engine="auto",
            scope_hint="features", use_rewriter=False, enhance_faces=False,
            auto_upscale=False,
        )
    print(f"result={result is not None}, calls={client.calls}, pastes={len(calls)}, flags={[c.get('expand_head_support') for c in calls]}")
    print(message)
    if result is not None:
        result.save(OUT / "replay.png")
        assert client.calls == 1, "an unchanged emblem must not start another model pass"
        baseline = manager._detect_edit_faces(source)
        selected = [max(baseline, key=lambda face: float(getattr(face, "det_score", 0.0)))]
        assert manager._added_face_mask(source, result, selected) is None, (
            "an unchanged low-score emblem must not trigger an added-face retry"
        )
        print("PASS: saved first pass has no new face against the original detector baseline")
        print(OUT / "replay.png")


if __name__ == "__main__":
    main()
