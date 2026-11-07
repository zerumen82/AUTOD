"""probe_editor_focused_refine.py

Contrato de `_second_focused_pass` (refino focal de la zona ya pegada):

1. Sin escalado de pegado (escala <= 1.5) NO se gasta una segunda pasada.
2. Con escalado (> 1.5) se llama al cliente UNA vez con la ventana zoom de
   la zona medida y el resultado se integra ACOTADO a la zona ya pegada:
   fuera de esa zona la foto no cambia ni un pixel.
3. Sin zona medida no hay refino (None), tampoco error.

Cliente falso: nunca toca GPU. Sin PYTHONPATH: el probe fija la raíz.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import cv2  # noqa: E402
from types import SimpleNamespace  # noqa: E402
from unittest.mock import patch  # noqa: E402
from PIL import Image  # noqa: E402

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

ok = True


def check(cond, label):
    global ok
    print(f"  [{'OK' if cond else 'FALLO'}] {label}")
    if not cond:
        ok = False


class FakeClient:
    def __init__(self):
        self.calls = 0
        self.arguments = None

    def generate(self, image, prompt, mask_image=None, **kwargs):
        self.calls += 1
        self.arguments = {"image": image, "mask_image": mask_image, **kwargs}
        # Refino sintetico: textura fuerte SOLO dentro de la mascara zoom.
        arr = np.asarray(image.convert("RGB")).astype(np.int16)
        m = np.asarray(mask_image.convert("L")) >= 160
        rng = np.random.default_rng(7)
        noise = rng.integers(-45, 45, size=arr.shape, dtype=np.int16)
        arr[m] = np.clip(arr[m] + noise[m], 0, 255).astype(np.int16)
        return SimpleNamespace(image=Image.fromarray(arr.astype(np.uint8))), "ok"


def build_case(scale):
    """Foto sintetica + mascara sujeto + generacion 1 a escala dada."""
    rng = np.random.default_rng(3)
    w, h = 1200, 1600
    base = (
        120
        + 40 * np.sin(np.arange(h)[:, None] / 37.0)
        + 25 * np.sin(np.arange(w)[None, :] / 23.0)
    )
    photo = np.clip(
        np.repeat(base[:, :, None], 3, axis=2)
        + rng.integers(-18, 18, size=(h, w, 3)),
        0, 255,
    ).astype(np.uint8)
    photo_img = Image.fromarray(photo)
    subject = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(subject, (600, 640), (260, 320), 0, 0, 360, 255, -1)
    subject_img = Image.fromarray(subject, "L")
    # ventana local: la mitad de la foto
    box = (300, 300, 900, 1300)
    bw, bh = box[2] - box[0], box[3] - box[1]
    gen_w, gen_h = max(8, int(bw / scale)), max(8, int(bh / scale))
    crop = photo[box[1]:box[3], box[0]:box[2]]
    small = cv2.resize(crop, (gen_w, gen_h), interpolation=cv2.INTER_AREA)
    # edicion 1: disco brillante fuerte dentro del sujeto
    gen = small.astype(np.int16)
    yy, xx = np.mgrid[0:gen_h, 0:gen_w]
    cx, cy = int((600 - box[0]) * gen_w / bw), int((640 - box[1]) * gen_h / bh)
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (min(gen_w, gen_h) * 0.22) ** 2
    gen[disk] = np.clip(gen[disk] + 70, 0, 255)
    gen_img = Image.fromarray(gen.astype(np.uint8))
    return photo_img, subject_img, box, gen_img, (gen_w, gen_h)


def integrate(mgr, photo, gen, subject, box):
    return mgr._integrate_local_generation(photo, gen, subject, box)


def main():
    mgr = object.__new__(ImgEditorManager)

    # [1] escala 1.0 -> no hay refino y no se gasta GPU
    photo, subject, box, gen, gsize = build_case(scale=1.0)
    edited = integrate(mgr, photo, gen, subject, box)
    client = FakeClient()
    out = mgr._second_focused_pass(
        client, edited, photo, subject, box, gsize,
        "p", None, 1, False, True, None, None, 1.0,
        {"num_inference_steps": 8, "denoise": 1.0}, None, None,
    )
    check(out is None, "escala 1.0x -> sin segunda pasada")
    check(client.calls == 0, "escala 1.0x -> cero llamadas al cliente")

    # [2] escala 2.0 -> refino, acotado a la zona pegada
    photo, subject, box, gen, gsize = build_case(scale=2.0)
    edited = integrate(mgr, photo, gen, subject, box)
    zone = mgr._changed_zone_on_subject(
        gen,
        photo.crop(box),
        subject.crop(box),
    )
    check(zone is not None, "zona medida existente en la generacion 1")
    client = FakeClient()
    sample = {"num_inference_steps": 8, "denoise": 1.0, "no_kontext": False}
    original_sample = sample.copy()
    out = mgr._second_focused_pass(
        client, edited, photo, subject, box, gsize,
        "p", None, 1, False, True, None, None, 1.0,
        sample, None, None,
    )
    check(client.calls == 1, "escala 2.0x -> exactamente una llamada al cliente")
    check(client.arguments.get("keep_source_pixels") is True,
          "el refino conserva el latente de foto, sin agujero de inpaint")
    check(client.arguments.get("no_kontext") is False,
          "la referencia espacial activada se conserva")
    check(sample == original_sample, "el refino no muta la receta del llamador")
    check(out is not None, "escala 2.0x -> refino devuelto")
    if out is not None:
        check(out.size == photo.size, "refino al tamano de la foto")
        a = np.asarray(out).astype(np.int16)
        b = np.asarray(edited).astype(np.int16)
        d = np.abs(a - b).max(axis=2)
        # El refino ahora regenera la cabeza entera (no solo el delta).
        paste_region = np.asarray(subject.convert("L")) >= 160
        outside = ~paste_region
        check(
            int(d[outside].sum()) == 0,
            "fuera de la cabeza nada cambia (delta=%d)" % int(d[outside].sum()),
        )
        inside = paste_region & (d > 0)
        check(
            inside.sum() >= 0.5 * paste_region.sum(),
            "la cabeza se refina (%d%% de la zona)" % (100 * inside.sum() / max(1, paste_region.sum())),
        )
    # La configuración sin referencia también debe respetarse: el refino
    # no puede sustituir la receta por una política fija de apariencia.
    client = FakeClient()
    out = mgr._second_focused_pass(
        client, edited, photo, subject, box, gsize,
        "p", None, 1, False, True, None, None, 1.0,
        {"num_inference_steps": 8, "denoise": 0.85, "no_kontext": True},
        None, None,
    )
    check(client.calls == 1 and out is not None,
          "la receta sin referencia también ejecuta el refino")
    check(client.arguments.get("no_kontext") is True,
          "la referencia espacial desactivada se conserva")
    check(client.arguments.get("denoise") == 0.85,
          "la fuerza configurada se conserva")
    check(client.arguments.get("keep_source_pixels") is True,
          "sin referencia espacial sigue usando el latente de foto")

    # Mismo recorte focal válido, pero sin edición: no debe regenerar el
    # sujeto solo porque exista una máscara. La máscara de foto completa
    # del caso final se rechaza por área y no detectaría esta regresión.
    client = FakeClient()
    out = mgr._second_focused_pass(
        client, photo, photo, subject, box, gsize,
        "p", None, 1, False, True, None, None, 1.0,
        sample, None, None,
    )
    check(out is None and client.calls == 0,
          "máscara local válida sin cambio -> sin refino ni gasto de GPU")

    empty = Image.new("L", photo.size, 0)
    client = FakeClient()
    out = mgr._second_focused_pass(
        client, edited, photo, empty, box, gsize,
        "p", None, 1, False, True, None, None, 1.0,
        sample, None, None,
    )
    check(out is None and client.calls == 0,
          "máscara vacía -> sin refino ni gasto de GPU")

    # La corrección de receta mantiene ambas compuertas del refino.
    # Detectores controlados solo en este contrato CPU; nunca en producción.
    for added, collapsed, label in (
        (subject, None, "cara añadida"),
        (None, subject, "cara degenerada"),
    ):
        client = FakeClient()
        with (
            patch.object(mgr, "_detect_edit_faces", return_value=[object()]),
            patch.object(mgr, "_added_face_mask", return_value=added) as added_check,
            patch.object(mgr, "_degenerate_faces_mask", return_value=collapsed) as collapsed_check,
        ):
            out = mgr._second_focused_pass(
                client, edited, photo, subject, box, gsize,
                "p", None, 1, False, True, None, None, 1.0,
                sample, None, None,
            )
        check(client.calls == 1 and out is None,
              f"{label} -> se rechaza el refino")
        check(added_check.call_count == 1 and collapsed_check.call_count == 1,
              f"{label} -> ambas compuertas siguen activas")
    # [4] ventana zoom sin estructura (<128 px) -> None sin llamada
    small = np.clip(
        np.repeat((100 + 30 * np.sin(np.arange(160)[:, None] / 11.0)
                   + 20 * np.sin(np.arange(120)[None, :] / 7.0))[:, :, None],
                  3, axis=2)
        + np.random.default_rng(5).integers(-12, 12, size=(160, 120, 3)),
        0, 255,
    ).astype(np.uint8)
    small_img = Image.fromarray(small)
    small_subject = Image.fromarray(
        np.full((160, 120), 255, dtype=np.uint8), "L"
    )
    gen_small = small[0:80, 0:60].astype(np.int16)
    gen_small[30:55, 25:45] = np.clip(gen_small[30:55, 25:45] + 70, 0, 255)
    edited = integrate(
        mgr, small_img,
        Image.fromarray(gen_small.astype(np.uint8)),
        small_subject, (0, 0, 60, 80),
    )
    client = FakeClient()
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = mgr._second_focused_pass(
            client, edited, small_img, small_subject, (0, 0, 60, 80),
            (30, 40), "p", None, 1, False, True, None, None, 1.0,
            {"num_inference_steps": 8, "denoise": 1.0}, None, None,
        )
    log = buf.getvalue()
    check(out is None, "zoom <128 px -> sin refino")
    check(client.calls == 0, "zoom <128 px -> cero llamadas al cliente")
    check(
        "sin estructura aprovechable" in log,
        "el guard de estructura es el que decide (no otro fallo previo)",
    )

    # [3] sin zona medida -> None sin excepcion
    flat_photo = Image.new("RGB", (600, 800), (90, 90, 90))
    flat_subject = Image.fromarray(
        np.full((800, 600), 255, dtype=np.uint8), "L"
    )
    flat_gen = Image.new("RGB", (300, 400), (90, 90, 90))
    edited = integrate(mgr, flat_photo, flat_gen, flat_subject, (0, 0, 600, 800))
    client = FakeClient()
    out = mgr._second_focused_pass(
        client, edited, flat_photo, flat_subject, (0, 0, 600, 800),
        (300, 400), "p", None, 1, False, True, None, None, 1.0,
        {"num_inference_steps": 8, "denoise": 1.0}, None, None,
    )
    check(out is None, "sin cambio medible -> None")
    check(client.calls == 0, "sin cambio medible -> cero llamadas")

    print()
    print("GRUPOS OK" if ok else "FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
