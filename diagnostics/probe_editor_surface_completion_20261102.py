"""Contrato CPU: completar la superficie medida con el color que ya recibio.

Medido el 2026-11-02 (logs/ui_start_20261102_122200.log): LongCat pinto el muro
gris y dejo la franja naranja pegada a las personas, la cama y las persianas
(salida cruda ui/tob/ComfyUI/output/FastEdit_01057_.png). El pegado solo escribe
el cambio medido, asi que la franja conservaba el color viejo; repetir la
instruccion media 0,6 % y no la tocaba (suelo del 4 %).

Contrato:
  A. la franja del color viejo dentro de la superficie pasa al color ya medido,
     conservando el sombreado local;
  B. fuera de la superficie medida no cambia ni un pixel (persona, objeto);
  C. una mancha del mismo color viejo aislada por otro color no se toca;
  D. si el modelo no recoloreo la superficie, no se inventa nada;
  E. si ya no queda color viejo, no hay nada que completar.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

W, H = 480, 320
PERSON = (slice(60, 300), slice(300, 380))
OBJECT = (slice(200, 260), slice(40, 120))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"PASS  {name}" + (f" :: {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"FAIL  {name}" + (f" :: {detail}" if detail else ""))


def scene():
    """Pared naranja con degradado, una persona y un objeto oscuro encima."""
    yy = np.linspace(0.0, 1.0, H)[:, None]
    wall = np.zeros((H, W, 3), np.float32)
    wall[..., 0] = 200.0 + 40.0 * yy
    wall[..., 1] = 120.0 + 40.0 * yy
    wall[..., 2] = 40.0 + 40.0 * yy
    original = wall.copy()
    original[PERSON] = (215.0, 170.0, 145.0)
    original[OBJECT] = (40.0, 40.0, 90.0)
    original = np.clip(original, 0, 255).astype(np.uint8)

    surface = np.full((H, W), 255, np.uint8)
    surface[PERSON] = 0
    surface[OBJECT] = 0

    luma = (
        original[..., 0] * 0.299
        + original[..., 1] * 0.587
        + original[..., 2] * 0.114
    )
    grey = 175.0 + (luma - float(luma[surface >= 96].mean())) * 0.6
    edited = original.astype(np.float32).copy()
    inside = surface >= 96
    for channel in range(3):
        edited[..., channel] = np.where(inside, grey, edited[..., channel])
    edited = np.clip(edited, 0, 255).astype(np.uint8)
    edited[PERSON] = original[PERSON]
    edited[OBJECT] = original[OBJECT]

    blocked = np.zeros((H, W), np.uint8)
    blocked[PERSON] = 1
    blocked[OBJECT] = 1
    band = cv2.dilate(blocked, np.ones((31, 31), np.uint8), iterations=1) > 0
    band &= inside
    edited[band] = original[band]
    return original, edited, Image.fromarray(surface, "L"), band, inside


def lum(arr: np.ndarray) -> np.ndarray:
    return (
        arr[..., 0].astype(np.float32) * 0.299
        + arr[..., 1].astype(np.float32) * 0.587
        + arr[..., 2].astype(np.float32) * 0.114
    )


def main() -> int:
    manager = ImgEditorManager.__new__(ImgEditorManager)
    original, edited, surface, band, inside = scene()
    band_core = cv2.erode(band.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0

    result = manager._complete_surface_recolor(
        Image.fromarray(original, "RGB"),
        Image.fromarray(edited, "RGB"),
        surface,
    )
    check("A1 devuelve imagen", result is not None)
    if result is None:
        print(f"\n{PASS}/{PASS + FAIL} PASS")
        return 1
    out = np.asarray(result.convert("RGB"))

    old_band = original[band_core].astype(np.float32)
    new_band = out[band_core].astype(np.float32)
    spread = float(
        np.mean(np.abs(new_band[:, 0] - new_band[:, 1]))
        + np.mean(np.abs(new_band[:, 1] - new_band[:, 2]))
    )
    check(
        "A2 la franja deja de ser naranja",
        spread < 12.0, f"desvio medio={spread:.1f}",
    )
    grey_ref = float(np.median(lum(out[inside])))
    band_l = float(np.mean(lum(out[band_core])))
    check(
        "A3 el brillo de la franja coincide con el de la superficie",
        abs(band_l - grey_ref) <= 12.0,
        f"franja={band_l:.1f} superficie={grey_ref:.1f}",
    )
    std_before = float(np.std(lum(original[band_core])))
    std_after = float(np.std(lum(out[band_core])))
    check(
        "A4 conserva el sombreado local",
        std_after >= 0.6 * std_before,
        f"antes={std_before:.1f} despues={std_after:.1f}",
    )

    outside = ~inside
    diff_out = np.abs(out.astype(np.int16) - edited.astype(np.int16)).max(axis=2)
    check(
        "B fuera de la superficie no cambia nada",
        int(diff_out[outside].max()) == 0,
        f"max={int(diff_out[outside].max())}",
    )

    # C. mancha del mismo color viejo aislada por un marco oscuro.
    original_c = original.copy()
    isolation = np.zeros((H, W), np.uint8)
    original_c[20:80, 150:230] = (30, 30, 30)          # marco
    original_c[32:68, 160:220] = original[32:68, 160:220]  # naranja dentro
    edited_c = edited.copy()
    edited_c[20:80, 150:230] = original_c[20:80, 150:230]
    result_c = manager._complete_surface_recolor(
        Image.fromarray(original_c, "RGB"),
        Image.fromarray(edited_c, "RGB"),
        surface,
    )
    check("C1 devuelve imagen con una mancha aislada", result_c is not None)
    if result_c is not None:
        out_c = np.asarray(result_c.convert("RGB"))
        inner = np.zeros((H, W), bool)
        inner[36:64, 164:216] = True
        moved = int(
            (
                np.abs(
                    out_c.astype(np.int16) - edited_c.astype(np.int16)
                ).max(axis=2)[inner] > 2
            ).sum()
        )
        check("C2 la mancha aislada no se toca", moved == 0, f"px={moved}")

    # D. el modelo no recoloreo nada.
    flat = manager._complete_surface_recolor(
        Image.fromarray(original, "RGB"),
        Image.fromarray(original, "RGB"),
        surface,
    )
    check("D el no-op no inventa relleno", flat is None)

    # E. idempotente: sobre su propia salida no cambia nada.
    again = manager._complete_surface_recolor(
        Image.fromarray(original, "RGB"), result, surface,
    )
    if again is None:
        check("E idempotente: ya no queda color viejo", True)
    else:
        again_arr = np.asarray(again.convert("RGB"))
        diff_again = int(
            np.abs(again_arr.astype(np.int16) - out.astype(np.int16)).max()
        )
        check(
            "E idempotente: ya no queda color viejo",
            diff_again <= 2, f"max={diff_again}",
        )

    # F. la foto real mide 6395x4263: el trabajo a 768 tambien debe converger.
    big = 3
    original_f = cv2.resize(
        original, (W * big, H * big), interpolation=cv2.INTER_NEAREST,
    )
    edited_f = cv2.resize(
        edited, (W * big, H * big), interpolation=cv2.INTER_NEAREST,
    )
    surface_f = surface.resize((W * big, H * big), Image.Resampling.NEAREST)
    result_f = manager._complete_surface_recolor(
        Image.fromarray(original_f, "RGB"),
        Image.fromarray(edited_f, "RGB"),
        surface_f,
    )
    check("F1 a escala real devuelve imagen", result_f is not None)
    if result_f is not None:
        out_f = np.asarray(result_f.convert("RGB"))
        band_f = cv2.resize(
            band.astype(np.uint8) * 255, (W * big, H * big),
            interpolation=cv2.INTER_NEAREST,
        ) > 0
        spread_f = float(
            np.mean(np.abs(out_f[band_f, 0].astype(np.float32) - out_f[band_f, 1]))
            + np.mean(np.abs(out_f[band_f, 1].astype(np.float32) - out_f[band_f, 2]))
        )
        check("F2 a escala real la franja deja de ser naranja", spread_f < 12.0,
              f"desvio medio={spread_f:.1f}")
        diff_f = np.abs(
            out_f.astype(np.int16) - edited_f.astype(np.int16)
        ).max(axis=2)
        check("F3 a escala real fuera de la superficie no cambia nada",
              int(diff_f[~(cv2.resize(
                  (inside.astype(np.uint8) * 255), (W * big, H * big),
                  interpolation=cv2.INTER_NEAREST) > 0)].max()) == 0)
        again_f = manager._complete_surface_recolor(
            Image.fromarray(original_f, "RGB"), result_f, surface_f,
        )
        if again_f is None:
            check("F4 a escala real es idempotente", True)
        else:
            delta_f = int(
                np.abs(np.asarray(again_f.convert("RGB")).astype(np.int16)
                       - out_f.astype(np.int16)).max()
            )
            check("F4 a escala real es idempotente", delta_f <= 2, f"max={delta_f}")

    # G. con caras detectadas el guard de personas corre y no rompe el peinado.
    face = SimpleNamespace(bbox=(300.0, 60.0, 380.0, 300.0))
    with_faces = manager._complete_surface_recolor(
        Image.fromarray(original, "RGB"),
        Image.fromarray(edited, "RGB"),
        surface,
        [face],
    )
    check("G con caras el guard no rompe la superficie", with_faces is not None)
    if with_faces is not None:
        out_g = np.asarray(with_faces.convert("RGB"))
        diff_g = np.abs(out_g.astype(np.int16) - edited.astype(np.int16)).max(axis=2)
        check("G2 fuera de la superficie sigue sin cambios",
              int(diff_g[outside].max()) == 0, f"max={int(diff_g[outside].max())}")

    print("=" * 60)
    print(f"{PASS}/{PASS + FAIL} PASS")
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
