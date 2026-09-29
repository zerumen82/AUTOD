"""Compare full-context Turbo with source-grounded head and hair composites."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager

ROOT = Path(__file__).resolve().parents[1]
source = Image.open(ROOT / "diagnostics/profile_reconstructed_source.png").convert("RGB")
generated = Image.open(ROOT / "diagnostics/profile_full_longcat_raw.png").convert("RGB")
generated = generated.resize(source.size, Image.Resampling.LANCZOS)
detector = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models"),
                        providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=-1, det_size=(1024, 1024))
faces = detector.get(np.asarray(source)[:, :, ::-1].copy())
masker = CLIPSegMasker()
subject = masker.generate_mask(source, "man")
hair = masker.generate_mask(source, "hair")
editor = ImgEditorManager()
head = editor._head_mask_from_region_plan(source, faces, subject)
scalp = editor._grounded_hair_mask(source, head, subject, hair)
for name, mask in (("head", head), ("hair", scalp)):
    assert mask is not None, name
    soft = editor._feather_mask(mask, amount=6)
    Image.composite(generated, source, soft).save(
        ROOT / f"diagnostics/profile_full_{name}_composite.png")
masker.unload()
