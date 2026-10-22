"""Measure which part of the measured text reaches the model on Kontext.

Measured on Kontext: the negative prompt is INERT (the engine samples at
CFG=1 and the guidance lives in the FluxGuidance node), so the
direction-based negative cannot change the result here. What can act is the
POSITIVE text.

This measures the positive text on its own, on the production architecture
(measured zone as the noise mask, no anchor):
  A  the production regional template;
  B  the measured end state only (no template, no enforcement);
  C  the measured end state plus the enforcement clause.

The metric is the skin share inside the measured garment zone, computed with
a plain luminance/saturation rule (the segmenter's skin detector needs the
GPU model and is not used here), against the original photo.

Nothing is downloaded.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
COMFY_OUT = ROOT / "ui" / "tob" / "ComfyUI" / "output"
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"
PROMPT = "la mujer esta desnuda, sin zapatos"
SEED = 424242

TEMPLATES = {
    "A_produccion": None,
    "B_solo_estado": "{instruction}",
    "C_estado_mas_enforcement": "{instruction}",
}


def newest_fastedit(after):
    candidates = [path for path in COMFY_OUT.glob("FastEdit_*.png") if path.stat().st_mtime > after]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def skin_share(image, zone):
    """Share of skin-like pixels inside the zone.

    Skin is bright and not saturated like the garments; the rule is a
    luminance range with a loose saturation bound. It is a rough measure,
    used only to compare the runs against each other.
    """
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    maximum = arr.max(axis=2)
    minimum = arr.min(axis=2)
    value = maximum
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
    skin = (value > 90) & (value < 235) & (saturation < 0.42)
    return float((skin & zone).sum()) / max(1, int(zone.sum()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    objects = [region for region in analysis.regions if region.kind == "object"]
    region = max(objects, key=lambda item: item.coverage)
    support = np.asarray(es.union_mask(analysis, [region.id])) > 0
    segmenter = manager._get_clothing_segmenter()
    cloth, _ = segmenter.segment_clothing(photo, threshold=0.45)
    zone = (np.asarray(cloth.convert("L")) > 127) & support
    for head in analysis.regions:
        if head.kind == "head":
            zone = zone & ~(np.asarray(head.mask.convert("L")) > 127)
    before = skin_share(photo, zone)
    print(f"[Skin] original skin share in the measured garment zone={before:.4f}", flush=True)

    base = manager.measured_body_regime(
        PROMPT, "The woman is naked, without shoes...", engine="kontext")
    state = base["state_suffix"]
    enforce = base["enforce_clause"]
    print(f"[Skin] measured state sentence={state!r}", flush=True)
    print(f"[Skin] measured enforcement={enforce!r}", flush=True)

    for name, template in TEMPLATES.items():
        settings = es.load_settings()
        if template is not None:
            settings["edit"]["prompt_template"] = template
        if name == "C_estado_mas_enforcement":
            settings["edit"]["prompt_template"] = "{instruction}"
        started = time.time()
        with patch.object(es, "load_settings", return_value=settings):
            output, message, _mask = manager.generate_intelligent(
                image=photo, prompt=PROMPT, engine="kontext",
                selection_analysis=analysis, selected_region_ids=[region.id],
                use_semantic=True, use_rewriter=True, enhance_faces=False,
                auto_upscale=False, seed=SEED,
            )
        print(f"[Skin {name}] {time.time() - started:.1f}s: {message}", flush=True)
        if output is None:
            continue
        output = output.convert("RGB")
        output.save(OUT / f"skin2_{name}.png")
        raw = newest_fastedit(started - 1)
        if raw is not None:
            shutil.copy(raw, OUT / f"skin2_{name}_raw.png")
        after = skin_share(output, zone)
        changed = np.any(source_arr != np.asarray(output), axis=2)
        print(f"[Skin {name}] skin share={after:.4f} (original {before:.4f}), "
              f"changed inside={int((changed & support).sum())} "
              f"outside={int((changed & ~support).sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
