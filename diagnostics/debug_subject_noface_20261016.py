"""Reproduce el descarte de sujeto del 22:51 («no coincide con una cara»).

edit_1792184313.png: CLIPSeg midió 'Man' 18,2% y 'el hombre' 19,7%, pero
ambas consultas se descartaron por no coincidir con ninguna cara creíble →
«Sujeto no medido» y la foto salió sin editar. Este script repite la
detección y la medición del manager y guarda un overlay con las caras,
sus solapes con la máscara y el veredicto de _grow_measured_subject.
"""
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / "img_editor/edit_1792184313.png"
OUT = ROOT / "diagnostics/out/subject_noface_20261016"
QUERIES = ("Man", "el hombre")


def main():
    os.makedirs(OUT, exist_ok=True)
    img = Image.open(PHOTO).convert("RGB")
    print(f"foto: {PHOTO.name} {img.size}")

    from roop.face_util import detect_faces_robust, get_face_analyser_cpu
    from roop.img_editor.img_editor_manager import ImgEditorManager

    manager = ImgEditorManager()
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    with patch("roop.face_util.get_face_analyser", new=lambda *a, **k: get_face_analyser_cpu()):
        faces = detect_faces_robust(img_bgr, get_face_analyser_cpu(), 0.20)
    credible = manager._credible_faces(faces)
    print(f"caras detectadas: {len(faces)} | creíbles: {len(credible)}")
    for index, face in enumerate(faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        is_credible = any(face is c for c in credible)
        print(
            f"  cara {index}: bbox=({x1},{y1})-({x2},{y2}) "
            f"det={float(getattr(face, 'det_score', 0.0) or 0.0):.2f} "
            f"edad={getattr(face, 'age', None)} "
            f"creíble={'sí' if is_credible else 'no'}"
        )

    masker = manager._get_clipseg_masker()
    overlay = img.convert("RGB")
    for query in QUERIES:
        mask = masker.generate_mask(img, query)
        if mask is None:
            print(f"[{query}] máscara vacía")
            continue
        region = np.asarray(mask.convert("L"))
        coverage = float((region > 96).mean())
        print(f"[{query}] cobertura={coverage:.1%}")
        for index, face in enumerate(credible):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 or y2 <= y1:
                print(f"  cara {index}: bbox degenerado tras recorte")
                continue
            overlap = float((region[y1:y2, x1:x2] > 96).mean())
            print(
                f"  cara {index}: solape máscara/cara={overlap:.1%} "
                f"({'PASA' if overlap >= 0.15 else 'DESCARTA'} el umbral 0,15)"
            )
        grown = manager._grow_measured_subject(region, credible, img.width, img.height)
        print(f"  _grow_measured_subject (solo creíbles) → {'medido' if grown is not None else 'DESCARTADO'}")
        grown_all = manager._grow_measured_subject(region, faces, img.width, img.height)
        print(f"  _grow_measured_subject (todas las caras) → {'medido' if grown_all is not None else 'DESCARTADO'}")
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.width, x2), min(img.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            overlap = float((region[y1:y2, x1:x2] > 96).mean())
            print(
                f"  TODA cara {index}: solape máscara/cara={overlap:.1%} "
                f"({'PASA' if overlap >= 0.15 else 'DESCARTA'} el umbral 0,15)"
            )
        tint = img.convert("RGBA")
        red = Image.new("RGBA", img.size, (255, 0, 0, 0))
        red.putalpha(Image.fromarray((region > 96).astype(np.uint8) * 90))
        overlay = Image.alpha_composite(tint, red)
        draw = ImageDraw.Draw(overlay)
        for index, face in enumerate(faces):
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            colour = (0, 255, 0) if any(face is c for c in credible) else (255, 255, 0)
            draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
            draw.text((x1 + 4, max(0, y1 - 14)), f"cara {index}", fill=colour)
        out_path = OUT / f"overlay_{query.replace(' ', '_')}.png"
        overlay.convert("RGB").save(out_path)
        print(f"  overlay guardado: {out_path}")


if __name__ == "__main__":
    main()
