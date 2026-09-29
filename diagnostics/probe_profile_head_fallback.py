"""Preview full-head fallback from already generated Turbo results."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from pathlib import Path
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager

ROOT = Path(__file__).resolve().parents[1]
CASES = [("first", "first_reconstructed_source.png", 622),
         ("profile", "profile_reconstructed_source.png", 621)]
detector = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models"),
                        providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=-1, det_size=(1024, 1024))
masker = CLIPSegMasker()
editor = ImgEditorManager()

for name, source_name, run in CASES:
    source = Image.open(ROOT / "diagnostics" / source_name).convert("RGB")
    raw = Image.open(ROOT / "ui/tob/ComfyUI/output" / f"FastEdit_{run:05d}_.png").convert("RGB")
    faces = detector.get(np.asarray(source)[:, :, ::-1].copy())
    subject = masker.generate_mask(source, "man")
    head = editor._head_mask_from_region_plan(source, faces, subject)
    assert head is not None, name
    crop = editor._crop_focal_context(source, head, max_area=.55)
    assert crop is not None, name
    original, _, box = crop
    candidate = raw.resize(original.size, Image.Resampling.LANCZOS)
    soft = editor._feather_mask(head.crop(box), amount=6)
    out = source.copy()
    out.paste(Image.composite(candidate, original, soft), box)
    out.save(ROOT / "diagnostics" / f"{name}_head_fallback.png")
    print(name, box, flush=True)
masker.unload()
