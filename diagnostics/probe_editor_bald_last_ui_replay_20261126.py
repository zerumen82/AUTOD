"""Replay the last UI bald edit with its actual LongCat output, without inference."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
from roop.img_editor.img_editor_manager import get_img_editor_manager
from ui.tabs.img_editor.events import on_generate


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "diagnostics/out/editor_bald_last_ui_replay_20261126"
INPUT = ROOT / "ui/tob/ComfyUI/input/fast_1795731247.png"
RAW = ROOT / "ui/tob/ComfyUI/output/FastEdit_01436_.png"
ACTUAL = ROOT / "img_editor/edit_1795731399.png"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = Image.open(INPUT).convert("RGB").resize(
        (800, 533), Image.Resampling.LANCZOS,
    )
    raw = Image.open(RAW).convert("RGB")
    actual = Image.open(ACTUAL).convert("RGB")
    client = get_flux_edit_comfy_client()
    manager = get_img_editor_manager()
    generate_original = client.generate
    integrate_original = manager._integrate_local_generation
    work_original = manager._appearance_work_region
    calls = []

    def use_recorded_generation(**kwargs):
        kwargs["image"].save(OUTPUT / "reference.png")
        calls.append(kwargs)
        return SimpleNamespace(image=raw), "recorded real LongCat output"

    def observe_integration(*args, **kwargs):
        source_image, generated_image, mask, box = args[:4]
        source_image.save(OUTPUT / "source.png")
        generated_image.save(OUTPUT / "raw.png")
        mask.save(OUTPUT / "paste_mask.png")
        limit = kwargs.get("limit_mask")
        if limit is not None:
            limit.save(OUTPUT / "limit_mask.png")
        (OUTPUT / "box.txt").write_text(str(box), encoding="utf-8")
        return integrate_original(*args, **kwargs)

    client.generate = use_recorded_generation
    manager._integrate_local_generation = observe_integration
    # The recorded UI call sampled the full 800x533 photo. Reproduce that
    # window even if resampling its saved 1248x832 input changes CLIPSeg's
    # person bbox enough to choose a smaller window on this replay.
    manager._appearance_work_region = lambda image, mask, person=None: (
        image, mask, (0, 0, image.width, image.height)
    )
    try:
        result, message, _mask = on_generate(
            source, "EL HOMBRE DEBE SER CALVO", "auto", 0.0,
            False, 1, False, "hd", scope_hint="features",
            source="diagnostic_recorded_ui_replay",
        )
    finally:
        client.generate = generate_original
        manager._integrate_local_generation = integrate_original
        manager._appearance_work_region = work_original
    assert result is not None, message
    result.save(OUTPUT / "replay.png")
    assert len(calls) == 1, f"expected one generation, got {len(calls)}"
    replay = np.asarray(result.convert("RGB"), dtype=np.int16)
    logged = np.asarray(actual.resize(result.size), dtype=np.int16)
    print(
        f"replay-vs-UI mean absolute difference: "
        f"{np.abs(replay - logged).mean():.2f}; "
        f"mask pixels: {int((np.asarray(Image.open(OUTPUT / 'paste_mask.png')) > 0).sum())}",
        flush=True,
    )
    print(f"EVIDENCE: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
