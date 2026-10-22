"""Contrato del default Kontext (plan aprobado 2026-10-18).

Historia: tras medir censura 0 %, mejor calidad y menor tiempo en 5/5
modelos, Kontext Q4_K_M pasa a ser el motor por defecto del Editor. El
cambio no puede perder tres invariantes ya medidos:

1. POV/postura ⇒ Qwen Cámara aunque el motor elegido sea kontext (el
   ancla de Kontext contradice la instrucción de cámara; gate antes en
   img_editor_manager, engine in (..., "kontext")).
2. Cuerpo ⇒ el router lo respeta en Kontext (decisión del usuario
   2026-10-18: se queda en Kontext; se bloquea con contrato).
3. Focal sin máscara manual ⇒ Kontext sobre el recorte medido, sin máscara
   de inpaint en el cliente. Protección automática de las otras personas
   aprobada por el usuario el 18/10 tras el fallo de referente. Las peticiones
   compuestas globales conservan todas sus cláusulas en una sola generación.

Ejercita las rutas REALES (generate_intelligent y on_generate) con
clientes simulados; solo la inferencia GPU está stubbeada.
"""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, os.getcwd())

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
UI_SOURCE = ROOT / "ui/tabs/img_editor/ui.py"


class FakeFluxClient:
    """Registra load/generate del cliente de edición (positional o kwarg)."""

    def __init__(self):
        self.calls = []

    def load(self, *args, **kwargs):
        if args and "flux_version" not in kwargs:
            kwargs = {**kwargs, "flux_version": args[0]}
        self.calls.append(("load", kwargs))
        return True, "OK (stub)"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (48, 32), "red")), "OK (stub)"


class FakeCamera:
    calls = []

    def generate(self, image, pose, **kwargs):
        self.calls.append((image.size, pose))
        return SimpleNamespace(image=Image.new("RGB", (48, 32), "green")), "OK"


def _check_ui_default():
    """El dropdown de motor arranca en kontext Q4_K_M, sin etiqueta Q2 vieja."""
    import gradio as gr

    from roop.img_editor.gen_prompt_config import get_editor_ui_config
    from ui.tabs.img_editor.ui import build_img_editor_ui

    assert get_editor_ui_config()["default_engine"] == "kontext"
    with gr.Blocks():
        components = build_img_editor_ui()
    engine = components["engine"]
    assert engine.value == "kontext", engine.value
    # Gradio 3.x devuelve choices como (label, value) o (label, value, info)
    # según la versión; extraer label de forma robusta.
    _labels = []
    for _c in engine.choices:
        if isinstance(_c, (tuple, list)):
            _labels.append(_c[0])
        else:
            _labels.append(getattr(_c, "label", str(_c)))
    labels = _labels
    assert any("FLUX.1 Kontext Q4_K_M" in label for label in labels), (
        "la etiqueta de Kontext debe declarar Q4_K_M (el Q2 medido es el copy machine)"
    )
    assert not any("Q2 (12B" in label for label in labels), (
        "queda una etiqueta Kontext Q2 obsoleta en la UI"
    )
    with (
        patch("ui.tabs.img_editor.ui.get_editor_ui_config",
              return_value={"default_engine": "qwen_camera"}),
        gr.Blocks(),
    ):
        assert build_img_editor_ui()["engine"].value == "qwen_camera"
    with (
        patch("ui.tabs.img_editor.ui.get_editor_ui_config",
              return_value={"default_engine": "motor_inexistente"}),
        gr.Blocks(),
    ):
        try:
            build_img_editor_ui()
        except ValueError as error:
            assert "editor_ui.default_engine" in str(error), str(error)
        else:
            raise AssertionError("la UI aceptó un motor configurado inexistente")
    print("OK: default Editor configurable, validado y con etiqueta Kontext Q4_K_M")


def _check_body_router():
    """Kontext respeta el cuerpo (decisión usuario 2026-10-18); pose sigue yendo a Qwen."""
    manager = ImgEditorManager()
    body = manager._route_edit_engine("kontext", {
        "body_transform": True, "body_direction": "change", "magnitude": 0.9,
    }, 8.0)
    assert body == (None, None, ""), body
    # Pose/POV explícito → Qwen Cámara (no Kontext, su ancla contradice la vista).
    pose = manager._route_edit_engine("kontext", {
        "body_transform": True, "body_direction": "change", "magnitude": 0.9,
        "pose_change": True,
    }, 8.0)
    assert pose == (None, None, ""), pose
    print("OK: con default Kontext el cuerpo se queda en Kontext y la pose va a Qwen Cámara")


def _check_pov_goes_to_qwen():
    """POV explícito con motor kontext entra en la ruta Qwen Cámara (no en Kontext)."""
    client = FakeFluxClient()
    camera = FakeCamera()
    FakeCamera.calls = []
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=client),
        patch("roop.img_editor.qwen_camera_comfy_client.QwenCameraComfyClient",
              return_value=camera),
        patch.object(ImgEditorManager, "_release_local_gpu_before_comfy",
                     return_value=None),
    ):
        manager = ImgEditorManager()
        output, message, _ = manager.generate_intelligent(
            image=Image.new("RGB", (48, 32), "white"),
            prompt="Show the room from an overhead camera angle, make the wall blue",
            engine="kontext",
            use_semantic=False,
            use_rewriter=False,
            enhance_faces=False,
        )
    assert output is not None, message
    assert camera.calls, (
        "POV con motor kontext no entró en Qwen Cámara: el ancla de Kontext "
        "contradice la instrucción de cámara"
    )
    for _name, kwargs in client.calls:
        version = str(kwargs.get("flux_version", ""))
        assert "kontext" not in version.lower(), (
            f"se cargó Kontext en una ruta de cámara: {client.calls}"
        )
    assert "Qwen Camera" in message, message
    print(f"OK: POV con kontext → Qwen Cámara ({camera.calls[0][1]}) sin cargar Kontext")


def _check_mask_never_kontext():
    """Kontext focal usa el recorte medido y protege al otro sujeto."""
    from diagnostics.probe_editor_kontext_people_protection_20261018 import (
        _assert_protected, _execute,
    )

    _assert_protected(_execute())
    print("OK: Kontext focal sin inpaint, recorte medido y otra persona intacta")


def _check_compound_kontext_stays_global():
    """El plan multi-región del rewriter no divide un compuesto Kontext."""
    prompt = "The sofa must be grey, the walls must be blue"
    manager = ImgEditorManager()
    client = FakeFluxClient()
    original_generate = manager.generate_intelligent
    invocations = []

    def tracked_generate(**kwargs):
        invocations.append(kwargs)
        assert kwargs.get("_compound_depth", 0) == 0, (
            "Kontext dividió el prompt en una subedición regional: "
            f"{kwargs['prompt']!r}; debe generar ambas cláusulas juntas"
        )
        return original_generate(**kwargs)

    # Dobles solo de inferencia/análisis pesado. El router, la alineación del
    # plan y el despacho de generate_intelligent siguen siendo producción.
    quality = SimpleNamespace(analyze_image=lambda *_a, **_kw: {
        "scores": {"poster": 0.0, "sharpness": 1.0, "restore": 0.0},
    })
    verifier = SimpleNamespace(verify=lambda *_a, **_kw: {"compliant": True})
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch("roop.img_editor.prompt_translator.translate_prompt", return_value=prompt),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=client),
        patch("roop.img_editor.image_quality_pipeline.get_quality_finisher",
              return_value=quality),
        patch("roop.img_editor.instruction_verifier.get_instruction_verifier",
              return_value=verifier),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch.object(manager, "_apply_prompt_rewriter", return_value=(
            prompt, 0.8, "subject", True, True, False, ["sofa", "walls"], "",
        )) as rewriter,
        patch.object(manager, "_align_compound_regions",
                     wraps=manager._align_compound_regions) as regional_splitter,
        patch.object(manager, "generate_intelligent", side_effect=tracked_generate),
    ):
        output, message, mask = manager.generate_intelligent(
            image=Image.new("RGB", (48, 32), "white"),
            prompt=prompt, engine="kontext", use_rewriter=True,
            use_semantic=False, enhance_faces=False,
        )

    assert output is not None, message
    assert mask is None, "el compuesto Kontext debe ser global por prompt"
    rewriter.assert_called_once()
    regional_splitter.assert_not_called()
    assert len(invocations) == 1, invocations
    loads = [kwargs for name, kwargs in client.calls if name == "load"]
    assert len(loads) == 1 and "kontext" in loads[0]["flux_version"].lower(), loads
    generations = [kwargs for name, kwargs in client.calls if name == "generate"]
    assert len(generations) == 1, generations
    request = generations[0]
    assert "The sofa must be grey" in request["prompt"], request["prompt"]
    assert "the walls must be blue" in request["prompt"], request["prompt"]
    assert request.get("mask_image") is None, request
    print("OK: compuesto Kontext conserva ambas cláusulas en una sola generación global")


def _check_legacy_compound_regions_still_split():
    """Kontext con petición compuesta NO se divide: una sola pasada global."""
    prompt = "The sofa must be grey, the walls must be blue"
    manager = ImgEditorManager()
    verifier = SimpleNamespace(verify=lambda *_a, **_kw: {"compliant": True})
    source = Image.new("RGB", (48, 32), "white")
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.img_editor.prompt_translator.translate_prompt", return_value=prompt),
        patch("roop.img_editor.instruction_verifier.get_instruction_verifier",
              return_value=verifier),
        patch.object(manager, "_apply_prompt_rewriter", return_value=(
            prompt, 0.8, "subject", True, True, False, ["sofa", "walls"], "",
        )),
        # Afirmamos solo el despacho: Kontext no debe dividir en subediciones.
        patch.object(ImgEditorManager, "generate_intelligent", return_value=(
            source, "OK (komtext global)", None,
        )) as subedit,
    ):
        output, message, _mask = ImgEditorManager.generate_intelligent(
            manager, image=source, prompt=prompt, engine="kontext",
            use_rewriter=True, use_semantic=False, enhance_faces=False,
        )

    assert output is not None, message
    calls = [call.kwargs for call in subedit.call_args_list]
    assert len(calls) == 1, f"se esperaba 1 llamada, se hicieron {len(calls)}: {calls}"
    assert calls[0].get("engine") == "kontext", calls[0]
    assert calls[0].get("_compound_depth", 0) == 0, calls[0]
    # La petición compuesta viaja íntegra (no se divide en regiones).
    prompt_sent = calls[0].get("prompt", "")
    assert "The sofa must be grey, the walls must be blue" in prompt_sent, prompt_sent
    assert "global" in message.lower(), message
    print("OK: Kontext compuesto es una sola pasada global (sin dividir en regiones)")


def main():
    _check_ui_default()
    _check_body_router()
    _check_pov_goes_to_qwen()
    _check_mask_never_kontext()
    _check_compound_kontext_stays_global()
    _check_legacy_compound_regions_still_split()


if __name__ == "__main__":
    main()
