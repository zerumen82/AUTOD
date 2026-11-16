"""Mide la banda SIN pintar que queda entre la persona y el escenario nuevo.

La ruta de escenario repinta solo el fondo medido y devuelve encima los
píxeles de la persona. Si la zona pintada empieza a varios píxeles de la
silueta, esa franja conserva el fondo viejo y sale como halo claro pegado al
pelo (matriz P1, 2026-11-14). CPU only: no usa difusión.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager, get_img_editor_manager


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "diagnostics/out/editor_camera_neutral_20261110/before.png"


def detect_faces(image: Image.Image):
    import cv2 as _cv2
    from roop.face_util import detect_faces_robust, get_face_analyser

    analyser = get_face_analyser()
    if analyser is None:
        return []
    bgr = _cv2.cvtColor(np.array(image.convert("RGB")), _cv2.COLOR_RGB2BGR)
    return list(detect_faces_robust(bgr, analyser, 0.20) or [])


def band_thickness(zone: Image.Image, people: Image.Image) -> float:
    """Grosor medio de la franja entre la silueta y el escenario pintado."""
    painted = np.asarray(zone.convert("L")) >= 96
    silhouette = np.asarray(people.convert("L")) >= 96
    band = (~painted) & (~silhouette)
    if not band.any():
        return 0.0
    distance = cv2.distanceTransform(
        (~silhouette).astype(np.uint8), cv2.DIST_L2, 5,
    )
    return float(distance[band].mean())


def main() -> int:
    image = Image.open(SOURCE).convert("RGB")
    faces = detect_faces(image)
    print(f"[diag] caras detectadas: {len(faces)}", flush=True)
    if not faces:
        print("[diag] sin caras: la ruta de escenario no mide personas")
        return 2
    masker = get_img_editor_manager()._get_clipseg_masker()
    people = ImgEditorManager._people_foreground(image, masker, faces)
    if people is None:
        print("[diag] silueta no medible")
        return 2
    people_img = Image.fromarray(people * 255, "L")
    # ¿La silueta medida llega hasta el borde real de la persona? Si el
    # contorno interior ya es fondo claro, la máscara es generosa y la zona
    # puede empezar en la propia silueta sin comerse pelo.
    photo = np.asarray(image.convert("RGB"), dtype=np.float32)
    luma = np.clip(photo @ np.array([0.299, 0.584, 0.114], np.float32), 0, 255)
    edge = (people > 0) & ~cv2.erode(people, np.ones((3, 3), np.uint8)).astype(bool)
    bright = float((luma[edge] > 150).mean()) if edge.any() else 0.0
    print(f"[diag] contorno de la silueta ya claro (fondo): {bright:.0%}")
    print("[diag] banda sin pintar alrededor de la persona:")
    for gap, label in (
        (7, "antes (resto de rutas)"),
        (3, "ruta de escenario"),
        (1, "hasta la silueta"),
    ):
        zone = ImgEditorManager._background_mask_from_people(
            image, masker, faces, gap=gap,
        )
        if zone is None:
            print(f"  gap={gap} ({label}): zona no medible")
            return 2
        coverage = float((np.asarray(zone.convert("L")) >= 96).mean())
        thickness = band_thickness(zone, people_img)
        print(
            f"  gap={gap} ({label}): escenario pintado {coverage:.1%}, "
            f"franja {thickness:.2f} px"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
