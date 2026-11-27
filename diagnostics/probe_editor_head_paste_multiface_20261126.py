"""A nearby second head must stay untouched by local appearance support growth."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def main() -> None:
    manager = ImgEditorManager()
    rng = np.random.default_rng(20261126)
    source_array = rng.integers(45, 190, (150, 180, 3), dtype=np.uint8)
    source = Image.fromarray(source_array, "RGB")
    generated = Image.new("RGB", source.size, (80, 130, 210))
    first = SimpleNamespace(bbox=(42, 40, 82, 90), det_score=0.9)
    second = SimpleNamespace(bbox=(65, 40, 105, 90), det_score=0.31)
    own = np.asarray(manager._head_mask_from_faces(source, [first])) >= 96
    other = np.asarray(manager._head_mask_from_faces(source, [second])) >= 96
    measured = Image.fromarray(own.astype(np.uint8) * 255, "L")
    overlap_ratio = float((own & other).sum()) / max(1, int(own.sum()))
    assert overlap_ratio > 0.55, "fixture must stress two overlapping heads"

    result = manager._integrate_local_generation(
        source, generated, measured, (0, 0, *source.size),
        keep_measured_mask=True, expand_head_support=True,
        source_faces=[first, second],
    )
    out = np.asarray(result.convert("RGB"))
    assert np.array_equal(out[other], source_array[other]), (
        "the nearby second head was repainted"
    )
    own_only = own & ~other
    assert np.any(out[own_only] != source_array[own_only]), (
        "the selected head was not edited"
    )
    print(
        f"PASS: low-score other head intact at {overlap_ratio:.1%} overlap; "
        f"selected head changed",
        flush=True,
    )


if __name__ == "__main__":
    main()
