#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe v5.1349 — la categoría «Cámara» fuerza el POV sin prohibir el prompt.

Con stubs: la pasada de cámara se sustituye por un cliente falso y el cliente
Turbo por otro, sin GPU ni ComfyUI.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


class FakeTurbo:
    def __init__(self):
        self.calls = []

    def load(self, version=None, **kwargs):
        self.calls.append(("load", version))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (48, 32), "blue")), "OK turbo"


class FakeCamera:
    calls = []

    def generate(self, image, pose, **kwargs):
        self.calls.append(pose)
        return SimpleNamespace(image=Image.new("RGB", (48, 32), "green")), "OK"


def run(manager, prompt, translated, hint, quality_mode=False):
    turbo = FakeTurbo()
    camera = FakeCamera()
    camera.calls.clear()
    with patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True), patch(
        "roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0,
    ), patch(
        "roop.img_editor.prompt_translator.translate_prompt", return_value=translated,
    ), patch(
        "roop.face_util.get_face_analyser", return_value=None,
    ), patch.object(
        manager, "_release_local_gpu_before_comfy",
    ), patch(
        "roop.img_editor.qwen_camera_comfy_client.QwenCameraComfyClient",
        lambda: camera,
    ), patch(
        "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
        return_value=turbo,
    ), patch(
        "roop.img_editor.selected_region_edit.edit_described_subject",
        return_value=None,
    ):
        result, message, mask = manager.generate_intelligent(
            image=Image.new("RGB", (64, 48), "gray"), prompt=prompt,
            engine="auto", scope_hint=hint, quality_mode=quality_mode,
            quality_steps=(
                {"deposter": True, "upscale": True, "color": True, "realism": True}
                if quality_mode else None
            ),
        )
    return camera, turbo, result, message


def main():
    manager = ImgEditorManager()
    ok = []

    def check(name, condition, detail=""):
        ok.append(bool(condition))
        print(f"{'PASS' if condition else 'FAIL'} {name}" + (f" — {detail}" if detail else ""))

    camera, turbo, result, message = run(
        manager, "hazlo bonito", "make it pretty", "camera",
    )
    check(
        "cámara + prompt genérico: POV forzado, sin edición generativa",
        len(camera.calls) == 1 and result is not None and "Qwen Camera" in message,
        f"{message[:90]} | pose={camera.calls}",
    )

    strong_pose = "CAMBIAR DE POSTURA A LA MUJER MOSTRANDO EL CULO A LA CAMARA"
    camera, turbo, result, message = run(
        manager, strong_pose, "change the pose of the woman showing her ass to the camera",
        "camera",
    )
    check(
        "cámara + postura medida fuerte: selector ignorado, sin pasada de cámara",
        len(camera.calls) == 0,
        f"camera_calls={len(camera.calls)} | {message[:80]}",
    )

    camera, turbo, result, message = run(
        manager, "hazlo bonito", "make it pretty", "camera", quality_mode=True,
    )
    check(
        "cámara + Modo Calidad sin eje medido: la red de acabado no se come el POV",
        len(camera.calls) == 1 and result is not None,
        f"camera_calls={len(camera.calls)} | {message[:80]}",
    )

    camera, turbo, result, message = run(
        manager, "hazlo bonito", "make it pretty", "auto",
    )
    check(
        "auto: «hazlo bonito» no activa la pasada de cámara",
        len(camera.calls) == 0,
        f"camera_calls={len(camera.calls)}",
    )

    passed = sum(ok)
    print(f"\n{passed}/{len(ok)} comprobaciones")
    return 0 if passed == len(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
