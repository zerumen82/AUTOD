"""Check that recolour chroma follows a displaced local generation window."""

import cv2
import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def main() -> None:
    rng = np.random.default_rng(20261126)
    source_array = rng.integers(60, 190, (120, 200, 3), dtype=np.uint8)
    source = Image.fromarray(source_array, "RGB")
    local_box = (60, 20, 160, 100)
    generated_array = np.empty((80, 100, 3), dtype=np.uint8)
    generated_array[:40] = (165, 100, 90)
    generated_array[40:] = (90, 110, 165)
    generated = Image.fromarray(generated_array, "RGB")
    mask_array = np.zeros((120, 200), dtype=np.uint8)
    mask_array[26:94, 66:154] = 255
    mask = Image.fromarray(mask_array, "L")

    result = ImgEditorManager._recolor_source_texture(
        source, generated, mask, local_box,
    )
    assert result is not None, "heterogeneous measured surface was skipped"
    out = np.asarray(result)
    exterior = cv2.dilate(mask_array, np.ones((13, 13), np.uint8)) == 0
    assert np.array_equal(out[exterior], source_array[exterior]), (
        "pixels beyond the measured contour changed"
    )
    for global_y, generated_y in ((40, 20), (80, 60)):
        global_x, generated_x = 90, 30
        original_lab = cv2.cvtColor(
            source_array[global_y:global_y + 1, global_x:global_x + 1],
            cv2.COLOR_RGB2LAB,
        )
        generated_lab = cv2.cvtColor(
            generated_array[
                generated_y:generated_y + 1, generated_x:generated_x + 1,
            ],
            cv2.COLOR_RGB2LAB,
        )
        original_lab[..., 1:] = generated_lab[..., 1:]
        expected = cv2.cvtColor(original_lab, cv2.COLOR_LAB2RGB)[0, 0]
        assert np.array_equal(out[global_y, global_x], expected), (
            "the generated colour was sampled at the wrong crop coordinate"
        )
    print("PASS: displaced crop chroma aligns; exterior unchanged")


if __name__ == "__main__":
    main()
