"""Measure whether pose descriptions ground the subjects in the user's photo."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


root = Path(__file__).resolve().parents[1]
img = Image.open(root / "ui/tob/ComfyUI/input/fast_1790338257.png").convert("RGB")
img = img.resize((800, 533), Image.Resampling.LANCZOS)
manager = get_img_editor_manager()
masker = manager._get_clipseg_masker()
for query in (
    "woman on her knees", "woman", "stockings",
    "man on his knees", "man", "man on his knees must be bald",
):
    mask = masker.generate_mask(img, query)
    cov = float((np.asarray(mask.convert("L")) > 128).mean()) if mask else 0.0
    print(f"{query!r}: coverage={cov:.3%}")
    if mask:
        mask.save(root / "diagnostics" / ("descriptor_" + query.replace(" ", "_") + ".png"))

# Log real 22:25 (edit_1792182345): 'Man' midió 0,01% y 'el hombre' 6,6%
# → el grupo quedó con el texto EN ESPAÑOL y la instrucción despachada a
# LongCat viajó sin traducir (cumplimiento casi nulo, el hombre siguió con
# pelo). El texto del grupo debe ser el TRADUCIDO aunque la máscara la
# haya ganado la consulta cruda.
region = np.zeros((60, 60), np.uint8)
region[10:40, 10:40] = 255
groups = manager._group_measured_subjects(
    [{"query": "el hombre", "region": region}],
    "el hombre debe ser calvo", "Man must be bald",
)
assert groups and groups[0]["text"] == "Man must be bald", (
    "el sujeto medido con la consulta cruda despachó texto sin traducir", groups,
)
print("OK: el sujeto medido con la consulta cruda despacha la instrucción traducida")

# Log real 22:51 (edit_1792184313): el hombre a cuatro patas se detectó con
# det 0.31 frente al 0.89 de la mujer → _credible_faces descartó SU cara, la
# máscara 'Man' (que cubría su cabeza al 93,9%) no encontró cara que la
# confirmara → «Sujeto no medido» y la foto salió SIN EDITAR. La puerta del
# solape debe aceptar cualquier cara DETECTADA; el suelo de credibilidad
# sigue vigente para los anclajes de calor/head (speck del pie, 27×22 px).
synth = Image.new("RGB", (80, 60), (128, 128, 128))
blob_px = np.zeros((60, 80), np.uint8)
blob_px[35:55, 40:70] = 255  # dónde está la cara débil, dentro de la máscara
weak_blob = Image.fromarray(blob_px, "L")


class _StubMasker:
    def generate_mask(self, image, query):
        return weak_blob


class _FaceBox:
    def __init__(self, bbox, det):
        self.bbox = bbox
        self.det_score = det


strong = _FaceBox((0, 0, 25, 25), 0.89)  # la otra persona, lejos de la máscara
weak = _FaceBox((40, 35, 70, 55), 0.31)  # el hombre, dentro de la máscara

with patch.object(manager, "_get_clipseg_masker", new=lambda *a, **k: _StubMasker()):
    only_credible = manager._measured_subject_passes(
        synth, "el hombre debe ser calvo", "Man must be bald", [strong],
    )
    with_detected = manager._measured_subject_passes(
        synth, "el hombre debe ser calvo", "Man must be bald", [strong],
        detected_faces=[strong, weak],
    )
    compound_zone = manager._compound_clause_zone(
        synth, "Man must be bald", "el hombre debe ser calvo", {}, [strong],
        detected_faces=[strong, weak],
    )
assert not only_credible, (
    "con solo las caras creíbles la máscara del sujeto debió descartarse (rojo del 22:51)",
    only_credible,
)
assert with_detected and len(with_detected) == 1, (
    "la puerta del solape debe aceptar la cara DETECTADA que el filtro de credibilidad descarta",
    with_detected,
)
assert compound_zone is not None, (
    "el compuesto debe medir la zona del sujeto con las caras detectadas",
    compound_zone,
)
print("OK: la puerta del solape acepta la cara detectada (det 0.31) que _credible_faces descarta")
