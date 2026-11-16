#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe v5.1349 — «Solo acabado» y red de seguridad del Modo Calidad.

Sin GPU ni ComfyUI: el acabado se sustituye por un stub y cualquier intento
de generación hace fallar la comprobación.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


class FinishStub:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image.size, kwargs))
        return image.copy(), "stub acabado", {"stub": True}


def run(manager, prompt, **kwargs):
    finish = FinishStub()
    with patch(
        "roop.img_editor.top_tier_finish.run_top_tier_quality", finish,
    ), patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True), patch(
        "ui.tabs.comfy_launcher.start", side_effect=AssertionError("no debe arrancar"),
    ):
        result, message, mask = manager.generate_intelligent(
            image=Image.new("RGB", (64, 48), "gray"), prompt=prompt, **kwargs,
        )
    return finish, result, message, mask


def main():
    manager = ImgEditorManager()
    ok = []

    def check(name, condition, detail=""):
        ok.append(bool(condition))
        print(f"{'PASS' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))

    finish, result, message, mask = run(
        manager, "MEJORA LA IMAGEN, REALISMO, DESPOSTERIZA",
        engine="auto", quality_mode=True, quality_enhance_style="solo_mejora",
        quality_steps={"deposter": True, "upscale": False, "color": False,
                       "realism": True, "only_finish": True},
        scope_hint="clothing",
    )
    check(
        "Solo acabado: no genera y aplica el acabado",
        len(finish.calls) == 1 and result is not None and "acabado" in message.lower(),
        message,
    )
    check(
        "Solo acabado: pasos marcados llegan al acabado",
        finish.calls and finish.calls[0][1]["quality_steps"]["only_finish"] is True
        and finish.calls[0][1]["quality_steps"]["deposter"] is True
        and finish.calls[0][1]["quality_steps"]["upscale"] is False,
    )

    no_axis = manager._prompt_measures_any_axis(
        "MEJORA LA IMAGEN, MEJORA REALISMO, DESPOSTERIZA",
        "IMPROVE THE IMAGE, IMPROVE REALISM, DEPOSTERIZE",
    )
    check("caso del log no mide ningún eje", no_axis is False)

    finish, result, message, mask = run(
        manager, "IMPROVE THE IMAGE, IMPROVE REALISM, DEPOSTERIZE",
        engine="auto", quality_mode=True,
        quality_steps={"deposter": True, "upscale": True, "color": True, "realism": True},
    )
    check(
        "Modo Calidad sin eje medido: acabado en vez de conservar la foto",
        len(finish.calls) == 1 and result is not None
        and "sin eje medido" in message.lower(),
        message,
    )

    finish, result, message, mask = run(
        manager, "MAKE THE PERSON BALD",
        engine="auto", quality_mode=True,
        quality_steps={"deposter": True, "upscale": True, "color": True, "realism": True},
    )
    check(
        "Modo Calidad con eje medido: no se salta a acabado",
        len(finish.calls) == 0,
        message[:120],
    )

    passed = sum(ok)
    print(f"\n{passed}/{len(ok)} comprobaciones")
    return 0 if passed == len(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
