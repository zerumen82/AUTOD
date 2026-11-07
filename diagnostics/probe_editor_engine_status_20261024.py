"""CPU contracts for strict Kontext selection and honest Editor status.

No image-model inference. In callback cases only the manager is substituted;
real Gradio components, clients' rejection paths and status formatting run.
"""
from __future__ import annotations

import sys
import io
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
import pydantic.v1  # noqa: E402,F401 -- preload before worker threads / Gradio schema
from PIL import Image  # noqa: E402

from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient  # noqa: E402
from roop.img_editor.gen_prompt_config import get_kontext_model_config  # noqa: E402
from ui.tabs.img_editor import events, state  # noqa: E402
from ui.tabs.img_editor.ui import build_img_editor_ui  # noqa: E402


class EngineStatusContracts(unittest.TestCase):
    def test_dropdown_values_are_unique(self):
        with gr.Blocks():
            ui = build_img_editor_ui()
        choices = ui["engine"].choices
        values = [value for _label, value in choices]
        self.assertEqual(len(values), len(set(values)), choices)
        self.assertIn("auto", values)
        self.assertEqual(ui["engine"].value, "auto")
        context_labels = [label for label, value in choices if value == "kontext"]
        self.assertEqual(len(context_labels), 1)
        self.assertNotIn("Auto:", context_labels[0])

    def test_auto_status_reports_actual_engine_scope_and_verification(self):
        from roop.img_editor.edit_execution import (
            begin_execution, end_execution, format_execution_status,
            record_execution, record_execution_scope,
        )
        from roop.img_editor.gen_prompt_config import get_longcat_model_config

        report, token = begin_execution("auto")
        try:
            record_execution("LongCat Turbo", get_longcat_model_config()["model_file"])
            record_execution_scope("máscara/región medida (8.8% del fotograma)")
            status = format_execution_status(
                report, {"compliant": None}, has_instruction=True,
            )
            self.assertIn("Auto eligió: LongCat Turbo", status)
            self.assertIn("8.8%", status)
            self.assertIn("sin verificar", status.lower())
        finally:
            end_execution(token)

    def test_auto_routes_measured_camera_request_to_qwen_camera(self):
        from roop.img_editor.edit_execution import (
            begin_execution, end_execution, record_execution,
        )
        from roop.img_editor.img_editor_manager import ImgEditorManager

        manager = ImgEditorManager()
        calls = []

        class Camera:
            def generate(self, image, pose, **_kwargs):
                calls.append((image.size, pose))
                record_execution("Qwen Cámara", "qwen-image-edit-2511-Q2_K.gguf")
                return SimpleNamespace(image=image.copy()), "OK"

        report, token = begin_execution("auto")
        try:
            with (
                patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
                patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
                patch("roop.img_editor.prompt_translator.translate_prompt",
                      return_value="show the scene from above"),
                patch("roop.img_editor.nlp.semantic_analyzer.is_camera_view_request",
                      return_value=True),
                patch("roop.img_editor.qwen_camera_comfy_client.camera_pose",
                      return_value="<sks> overhead view"),
                patch("roop.img_editor.qwen_camera_comfy_client.remaining_edit",
                      return_value=""),
                patch("roop.img_editor.qwen_camera_comfy_client.QwenCameraComfyClient",
                      return_value=Camera()),
                patch.object(manager, "_release_local_gpu_before_comfy"),
            ):
                output, message, _mask = manager.generate_intelligent(
                    image=Image.new("RGB", (48, 32), "white"),
                    prompt="show the scene from above",
                    engine="auto",
                    use_semantic=False,
                    use_rewriter=False,
                    enhance_faces=False,
                )
            self.assertIsNotNone(output, message)
            self.assertIn("Qwen Camera", message)
            self.assertEqual(len(calls), 1)
            self.assertEqual([model.engine for model in report.models], ["Qwen Cámara"])
        finally:
            end_execution(token)

    def test_auto_rejects_unselected_model_before_server_access(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution

        report, token = begin_execution("auto")
        try:
            client = FluxEditComfyClient()
            with patch.object(client, "is_available", side_effect=AssertionError("server touched")):
                loaded, note = client.load("unmeasured-model.gguf")
            self.assertFalse(loaded)
            self.assertIn("Auto detuvo", note)
            self.assertTrue(report.rejection)
            self.assertFalse(client._loaded)
        finally:
            end_execution(token)

    def test_auto_hands_visual_selection_to_longcat_route(self):
        from roop.img_editor.img_editor_manager import ImgEditorManager

        manager = ImgEditorManager()
        image = Image.new("RGB", (48, 32), "white")
        with patch(
            "roop.img_editor.selected_region_edit.edit_selected_region",
            return_value=(image, "selection route", None),
        ) as selected_edit:
            output, message, _mask = manager.generate_intelligent(
                image=image,
                prompt="change this selected area",
                engine="auto",
                selection_analysis=SimpleNamespace(),
                selected_region_ids=["region-1"],
            )
        self.assertIsNotNone(output, message)
        self.assertEqual(selected_edit.call_args.kwargs["engine"], "longcat")

    def test_auto_routes_body_uncover_to_measured_subject_path(self):
        from roop.img_editor.img_editor_manager import ImgEditorManager

        manager = ImgEditorManager()
        image = Image.new("RGB", (48, 32), "white")
        edited = Image.new("RGB", image.size, "blue")
        with (
            patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
            patch("roop.face_util.get_face_analyser", return_value=None),
            patch("roop.img_editor.prompt_translator.translate_prompt",
                  return_value="The woman in red must go without shoes"),
            patch.object(manager, "_explicit_viewpoint_request", return_value=False),
            patch.object(manager, "_prompt_authority_facts", return_value={
                "body": True, "pose": False, "body_direction": "uncover",
            }),
            patch(
                "roop.img_editor.selected_region_edit.edit_described_subject",
                return_value=(edited, "measured subject route", None),
            ) as described_edit,
        ):
            output, message, _mask = manager.generate_intelligent(
                image=image,
                prompt="la mujer de rojo debe ir sin zapatos",
                engine="auto",
                use_rewriter=False,
                enhance_faces=False,
            )
        self.assertIsNotNone(output, message)
        self.assertEqual(message, "measured subject route")
        self.assertEqual(described_edit.call_args.kwargs["engine"], "longcat")
        self.assertTrue(described_edit.call_args.kwargs["automatic"])

    def test_automatic_described_subject_uses_allowed_longcat_model(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution
        from roop.img_editor.gen_prompt_config import get_longcat_model_config
        from roop.img_editor.img_editor_manager import ImgEditorManager
        from roop.img_editor.selected_region_edit import edit_described_subject

        manager = ImgEditorManager()
        image = Image.new("RGB", (48, 32), "white")
        subject = np.ones((32, 48), dtype=bool)
        target = np.zeros_like(subject)
        target[27:, 20:28] = True
        client = SimpleNamespace(load=Mock(return_value=(True, "loaded")))
        _report, token = begin_execution("auto")
        try:
            with (
                patch.object(manager, "_prompt_authority_facts", return_value={
                    "body": True, "pose": False, "body_direction": "uncover",
                }),
                patch.object(manager, "_split_compound_instruction", return_value=[]),
                patch.object(manager, "_subject_queries", return_value=["woman"]),
                patch.object(manager, "measured_body_regime", return_value={
                    "body": True, "direction": "uncover", "magnitude": 1.0,
                    "intensity": 1.0, "enforce_clause": "",
                    "negative_prompt": "", "engine": {},
                }),
                patch.object(manager, "measured_full_body_exposure", return_value=False),
                patch.object(manager, "_requested_removal_target", return_value="shoes"),
                patch.object(manager, "_compose_generation_prompt", return_value="Body prompt"),
                patch.object(manager, "_compose_local_target_prompt", return_value="Remove shoes"),
                patch.object(manager, "_release_local_gpu_before_comfy"),
                patch("roop.img_editor.selected_region_edit._described_subject_mask",
                      return_value=(subject, "woman")),
                patch("roop.img_editor.selected_region_edit._measured_clothing_zone",
                      return_value=None),
                patch("roop.img_editor.selected_region_edit._measured_target_zone",
                      return_value=target),
                patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
                patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                      return_value=client),
                patch("roop.img_editor.element_selection.load_settings",
                      return_value={"edit": {}}),
                patch("roop.img_editor.selected_region_edit._vision_zone_pass",
                      return_value=(image.copy(), "vision pass")),
            ):
                result = edit_described_subject(
                    manager, image, "la mujer de rojo debe ir sin zapatos",
                    "The woman in red must go without shoes", [],
                    engine="longcat", automatic=True,
                )
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(np.asarray(result[2]) > 0, target)
            self.assertEqual(client.load.call_args.kwargs["flux_version"],
                             get_longcat_model_config()["model_file"])
        finally:
            end_execution(token)

    def test_described_local_targets_keep_upper_and_lower_mask_geometry(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution
        from roop.img_editor.img_editor_manager import ImgEditorManager
        from roop.img_editor import selected_region_edit as selected

        height, width = 128, 96
        yy, xx = np.indices((height, width))
        pixels = np.stack((70 + xx % 20, 110 + yy % 30, 170 + (xx + yy) % 25), axis=-1).astype(np.uint8)
        image = Image.fromarray(pixels)
        subject = np.zeros((height, width), dtype=bool)
        subject[10:116, 24:72] = True
        cases = (("amber scarf", (slice(18, 36), slice(30, 58)), False),
                 ("striped boots", (slice(96, 114), slice(42, 76)), True))
        scorer = SimpleNamespace(
            score_concepts=lambda _text: {"attribute_body": 0.6, "body_exposure_local": 0.5},
            detect_target=lambda _text: "clothes",
        )
        detector = SimpleNamespace(prepare=Mock(), detect=Mock(return_value=(None, None)))
        for target_name, slices, lower in cases:
            with self.subTest(target=target_name, lower=lower):
                manager = ImgEditorManager()
                target = np.zeros_like(subject)
                target[slices] = True
                instruction = f"The person must go without the {target_name}"
                generated = []

                def infer(**kwargs):
                    generated.append(kwargs)
                    # Declared inference double; real crop/composition runs.
                    return SimpleNamespace(image=Image.new("RGB", kwargs["image"].size, (210, 55, 180))), "CPU inference double"

                client = SimpleNamespace(load=Mock(return_value=(True, "CPU loaded")), generate=Mock(side_effect=infer))
                _report, token = begin_execution("auto")
                try:
                    with ExitStack() as stack:
                        local_prompt = stack.enter_context(patch.object(manager, "_compose_local_target_prompt", wraps=manager._compose_local_target_prompt))
                        measured = stack.enter_context(patch.object(selected, "_measured_target_zone", return_value=target))
                        vision = stack.enter_context(patch.object(selected, "_vision_zone_pass", wraps=selected._vision_zone_pass))
                        contexts = (
                            patch.object(manager, "_prompt_authority_facts", return_value={"body": True, "pose": False, "body_direction": "uncover"}),
                            patch.object(manager, "_split_compound_instruction", return_value=[]),
                            patch.object(manager, "_subject_queries", return_value=["the person"]),
                            patch.object(manager, "measured_body_regime", return_value={
                                "body": True, "direction": "uncover", "magnitude": 1.0,
                                "intensity": 1.0, "enforce_clause": "", "negative_prompt": "", "engine": {},
                            }),
                            patch.object(manager, "measured_full_body_exposure", return_value=False),
                            patch.object(manager, "_compose_generation_prompt", return_value="CPU body instruction"),
                            patch.object(manager, "_get_clipseg_masker", return_value=None),
                            patch.object(manager, "_release_local_gpu_before_comfy"),
                            patch("roop.img_editor.nlp.semantic_analyzer.get_semantic_analyzer", return_value=scorer),
                            patch.object(selected, "_described_subject_mask", return_value=(subject, "the person")),
                            patch.object(selected, "_measured_clothing_zone", return_value=None),
                            patch("insightface.model_zoo.get_model", return_value=detector),
                            patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
                            patch("ui.tabs.comfy_launcher.start", side_effect=AssertionError("backend must not start")),
                            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
                        )
                        for context in contexts:
                            stack.enter_context(context)
                        result = selected.edit_described_subject(
                            manager, image, instruction, instruction, [], engine="longcat", automatic=True,
                        )
                    self.assertIsNotNone(result)
                    self.assertEqual(result[0].size, image.size)
                    np.testing.assert_array_equal(np.asarray(result[2]) > 0, target)
                    self.assertEqual(measured.call_args.args[3], target_name)
                    self.assertEqual(local_prompt.call_args.args[1], target_name)
                    self.assertEqual(local_prompt.call_args.kwargs["lower_body"], lower)
                    np.testing.assert_array_equal(vision.call_args.args[3], target)
                    np.testing.assert_array_equal(vision.call_args.args[4], subject | target)
                    self.assertEqual(len(generated), 1)
                    self.assertEqual(generated[0]["mask_image"].size, generated[0]["image"].size)
                    self.assertGreater(int(np.count_nonzero(np.asarray(generated[0]["mask_image"]))), 0)
                finally:
                    end_execution(token)

    def test_local_removal_grammar_remains_generic(self):
        from roop.img_editor.img_editor_manager import ImgEditorManager

        manager = ImgEditorManager()
        for instruction, target in (
            ("without the amber scarf", "amber scarf"),
            ("sin el broche triangular", "broche triangular"),
            ("Remove the striped boots", "striped boots"),
            ("remove the cerulean widget", "cerulean widget"),
            ("", ""),
        ):
            with self.subTest(instruction=instruction):
                self.assertEqual(manager._extract_removal_target(instruction), target)

    def test_unchanged_described_fixture_is_still_rejected(self):
        from roop.img_editor.img_editor_manager import ImgEditorManager

        manager = ImgEditorManager()
        image = Image.new("RGB", (48, 32), "white")
        facts = {
            "body": True, "pose": False, "body_direction": "uncover",
            "authority": True, "local_appearance": False, "local_region": False,
            "scenario": False, "structural_remove": False, "structural_add": False,
        }
        with (
            patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
            patch("roop.face_util.get_face_analyser", return_value=None),
            patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
            patch("roop.img_editor.prompt_translator.translate_prompt", return_value="The person must go without the scarf"),
            patch.object(manager, "_explicit_viewpoint_request", return_value=False),
            patch.object(manager, "_prompt_authority_facts", return_value=facts),
            patch.object(manager, "_split_compound_instruction", return_value=[]),
            patch.object(manager, "_measured_subject_passes", return_value=[]),
            patch.object(manager, "_validate_visual_edit", wraps=manager._validate_visual_edit) as validation,
            patch("roop.img_editor.selected_region_edit.edit_described_subject", return_value=(image.copy(), "unchanged fixture", None)),
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", side_effect=AssertionError("no measured region, no inference")) as inference,
        ):
            output, message, _ = manager.generate_intelligent(
                image=image, prompt="The person must go without the scarf", engine="auto", enhance_faces=False,
            )
        validation.assert_called_once()
        self.assertNotEqual(message, "unchanged fixture")
        self.assertIn("conserva", message.lower())
        self.assertIsNotNone(output)
        np.testing.assert_array_equal(np.asarray(output), np.asarray(image))
        inference.assert_not_called()

    def test_auto_does_not_save_unapplied_camera_result(self):
        image = Image.new("RGB", (48, 32), "white")
        manager = SimpleNamespace(
            _last_instruction_compliance=None,
            _release_local_gpu_before_comfy=lambda: None,
            generate_intelligent=lambda **kwargs: (
                kwargs["image"],
                "POV no aplicado: Qwen Camera falló; se conserva el ángulo original.",
                None,
            ),
        )
        with tempfile.TemporaryDirectory(prefix="auto_not_applied_") as directory:
            with (
                patch.object(events, "get_img_editor_manager", return_value=manager),
                patch.object(events, "get_img_editor_output_dir", return_value=directory),
                patch.object(events, "_cleanup_vram"),
            ):
                output, status, _mask = events.on_generate(
                    image, "show the room from above", "auto", 0,
                    False, 1, False, "hd",
                )
            self.assertIsNotNone(output)
            self.assertIn("POV no aplicado", status)
            self.assertEqual(list(Path(directory).glob("*.png")), [])

    def test_forbidden_load_stops_before_server_or_gpu(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution

        report, token = begin_execution("kontext")
        try:
            client = FluxEditComfyClient()
            with patch.object(client, "is_available", side_effect=AssertionError("server touched")):
                # Without the measured-route opt-in, another model is still a
                # forbidden substitution. Stale LongCat names normalize to
                # Kontext, so the rejected request names a different engine.
                loaded, note = client.load("T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf")
            self.assertFalse(loaded)
            self.assertIn("Kontext", note)
            self.assertTrue(report.rejection)
            self.assertFalse(client._loaded)
        finally:
            end_execution(token)

    def test_measured_vision_route_is_the_explicit_exception(self):
        from roop.img_editor.edit_execution import (
            allow_model, begin_execution, end_execution,
        )
        from roop.img_editor.selected_region_edit import VISION_BODY_MODEL

        report, token = begin_execution("kontext")
        try:
            client = FluxEditComfyClient()
            with patch.object(client, "is_available", return_value=False):
                loaded, note = client.load(VISION_BODY_MODEL, allow_vision_engine=True)
            self.assertFalse(loaded, "availability check is still reached")
            self.assertNotIn("sustitución automática", note)
            self.assertEqual(report.rejection, "")
            self.assertEqual(report.permitted_vision_model, VISION_BODY_MODEL)
            self.assertTrue(allow_model(VISION_BODY_MODEL)[0])
        finally:
            end_execution(token)

    def test_described_body_route_asks_the_measured_vision_exception(self):
        """v5.1252 — la ruta sin selección, con la zona de ropa medida, pide el
        motor con visión por la excepción explícita y estrecha.

        Decisión del usuario 2026-10-26 tras nueve renders medidos sobre su
        foto: Kontext responde a la zona de ropa con OTRA prenda (bañador) con
        denoise 0,80-1,00, guidance 3,5/4,6, estado afirmativo y a doble
        resolución; el motor con visión quita la prenda y conserva la pose.
        """
        from roop.img_editor.edit_execution import begin_execution, end_execution
        from roop.img_editor.selected_region_edit import (
            VISION_BODY_MODEL, edit_described_subject,
        )

        manager = SimpleNamespace(
            _prompt_authority_facts=lambda *_: {
                "pose": False, "body": True, "body_direction": "uncover"},
            _split_compound_instruction=lambda text: [text],
            _subject_queries=lambda *_: ["the woman"],
            measured_full_body_exposure=lambda *_: True,
            measured_uncover_is_whole_person=lambda *_: False,
            _clothing_zone_min_ratio=lambda: 0.5,
            measured_body_regime=lambda *_, **__: {
                "magnitude": 0.75, "direction": "uncover", "intensity": 0.7,
                "enforce_clause": "", "negative_prompt": "",
                "engine": {"img2img_denoise": 0.90}, "body": True,
            },
            _release_local_gpu_before_comfy=lambda: None,
        )
        subject = np.zeros((64, 64), dtype=bool)
        subject[8:60, 16:48] = True
        loads = []

        class RecordingClient:
            def load(self, flux_version=None, allow_vision_engine=False):
                loads.append((flux_version, allow_vision_engine))
                return False, "diagnostic client"

        zone = np.zeros((64, 64), dtype=bool)
        zone[8:40, 16:48] = True
        report, token = begin_execution("kontext")
        try:
            with (patch("roop.img_editor.selected_region_edit._described_subject_mask",
                        return_value=(subject, "the woman")),
                  patch("roop.img_editor.selected_region_edit._measured_clothing_zone",
                        return_value=zone),
                  patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
                  patch("roop.img_editor.flux_edit_comfy_client."
                        "get_flux_edit_comfy_client", return_value=RecordingClient())):
                result = edit_described_subject(
                    manager, Image.new("RGB", (64, 64)), "la mujer debe ir desnuda",
                    "the woman must go naked", [], engine="kontext")
            self.assertIsNotNone(result)
            self.assertEqual(report.rejection, "")
            # La zona de ropa medida se pide al motor con visión; el recorte
            # de la persona es lo que le da resolución (el doble de carga
            # devuelve False, así que la ruta conserva la foto).
            self.assertEqual(loads, [(VISION_BODY_MODEL, True)])
        finally:
            end_execution(token)

    def test_vision_exception_is_not_a_general_permission(self):
        from roop.img_editor.edit_execution import (
            allow_model, begin_execution, end_execution, permit_vision_model,
        )
        from roop.img_editor.selected_region_edit import VISION_BODY_MODEL

        report, token = begin_execution("kontext")
        try:
            permit_vision_model(VISION_BODY_MODEL)
            self.assertFalse(allow_model("other-model.gguf")[0])
            self.assertTrue(report.rejection)
            # A rejected request cannot resume through the exception.
            permit_vision_model(VISION_BODY_MODEL)
            self.assertFalse(allow_model(VISION_BODY_MODEL)[0])
        finally:
            end_execution(token)

    def test_already_loaded_alternative_cannot_generate(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution

        report, token = begin_execution("kontext")
        try:
            client = FluxEditComfyClient()
            client._loaded = True
            client._flux_version = "other-model.gguf"
            with patch("requests.post", side_effect=AssertionError("prompt submitted")):
                result, _note = client.generate(image=Image.new("RGB", (64, 64)), prompt="blue jacket")
            self.assertIsNone(result)
            self.assertTrue(report.rejection)
        finally:
            end_execution(token)

    def test_camera_substitution_stops_before_availability(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution
        from roop.img_editor.qwen_camera_comfy_client import QwenCameraComfyClient

        report, token = begin_execution("kontext")
        try:
            client = QwenCameraComfyClient()
            with patch.object(client, "available", side_effect=AssertionError("camera checked")):
                result, note = client.generate(Image.new("RGB", (64, 64)), "side view")
            self.assertIsNone(result)
            self.assertIn("Kontext", note)
            self.assertTrue(report.rejection)
        finally:
            end_execution(token)

    def test_rejection_prevents_later_fallback(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution, allow_model

        _report, token = begin_execution("kontext")
        try:
            self.assertFalse(allow_model("other-model.gguf")[0])
            allowed, _note = allow_model(get_kontext_model_config()["model_file"])
            self.assertFalse(allowed, "a rejected request resumed through fallback")
        finally:
            end_execution(token)

    def test_current_config_is_authority(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution, allow_model

        configured = get_kontext_model_config()["model_file"]
        _report, token = begin_execution("kontext")
        try:
            self.assertTrue(allow_model(configured)[0])
        finally:
            end_execution(token)

    def test_declined_mixed_route_does_not_reject_an_unselected_model(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution
        from roop.img_editor.selected_region_edit import edit_described_subject

        # Controlled semantic facts exercise the eligibility veto without
        # segmentation or inference. A declined route never selects a model.
        def facts(_raw, translated):
            return {"body": True, "body_direction": "uncover",
                    "local_region": translated == "the wall must be gray"}

        manager = SimpleNamespace(
            _prompt_authority_facts=facts,
            _split_compound_instruction=lambda text: [part.strip() for part in text.split(",")],
        )
        report, token = begin_execution("kontext")
        try:
            prompt = "change the jacket, the wall must be gray"
            result = edit_described_subject(
                manager, Image.new("RGB", (64, 64)), prompt, prompt, [], engine="kontext")
            self.assertIsNone(result)
            self.assertEqual(report.rejection, "", "an ineligible route latched a false engine rejection")
        finally:
            end_execution(token)

    def test_client_success_records_the_model_actually_sent(self):
        from roop.img_editor import flux_edit_comfy_client as backend
        from roop.img_editor.edit_execution import begin_execution, end_execution

        yy, xx = np.indices((128, 128))
        pixels = np.stack((xx * 2, yy * 2, (xx + yy) % 256), axis=-1).astype(np.uint8)
        image = Image.fromarray(pixels)
        encoded = io.BytesIO()
        image.save(encoded, "PNG")
        configured = get_kontext_model_config()["model_file"]
        response = SimpleNamespace(status_code=200, text="", content=encoded.getvalue(),
                                   json=lambda: {"UnetLoaderGGUF": {}, "prompt_id": "diagnostic-only"})
        submitted = []
        client = FluxEditComfyClient()

        def post(_url, **kwargs):
            if "json" in kwargs:
                submitted.append(kwargs["json"]["prompt"])
            return response

        def wait(*_args, **_kwargs):
            # Another user of the shared singleton changed its version while
            # the first request was waiting. Telemetry must describe the sent
            # workflow, not this later value.
            client._flux_version = "later-unrelated-model.gguf"
            return {"filename": "diagnostic-only.png"}, "OK (inference substituted)"

        report, token = begin_execution("kontext")
        try:
            with (patch.object(client, "is_available", return_value=True),
                  patch.object(client, "_model_exists", return_value=True),
                  patch.object(backend, "_server_vram_free_mb", return_value=(8192, 8192)),
                  patch.object(backend.requests, "get", return_value=response),
                  patch.object(backend.requests, "post", side_effect=post),
                  patch.object(backend, "wait_for_comfy_image", side_effect=wait)):
                loaded, note = client.load(configured)
                self.assertTrue(loaded, note)
                result, note = client.generate(image=image, prompt="make the jacket blue")
            self.assertIsNotNone(result, note)
            self.assertEqual(len(submitted), 1)
            self.assertEqual(submitted[0]["2"]["inputs"]["unet_name"], configured)
            self.assertEqual([(model.engine, model.model_file) for model in report.models],
                             [("Kontext", configured)])
        finally:
            end_execution(token)

    def test_context_isolation_and_reset(self):
        from roop.img_editor.edit_execution import (
            begin_execution, end_execution, current_execution, record_execution,
        )

        def run(engine):
            report, token = begin_execution(engine)
            try:
                self.assertIs(current_execution(), report)
                if engine != "kontext":
                    record_execution("Qwen Cámara", "camera.gguf")
                return list(report.models)
            finally:
                end_execution(token)

        with ThreadPoolExecutor(max_workers=2) as pool:
            first, second = list(pool.map(run, ("kontext", "qwen_camera")))
        self.assertEqual(first, [])
        self.assertEqual(len(second), 1)
        self.assertIsNone(current_execution())

    def callback(self, verdict, *, reject=False, crash=False, prompt="make the jacket blue"):
        from roop.img_editor.edit_execution import allow_model, record_execution

        image = Image.new("RGB", (48, 64), "navy")
        manager = SimpleNamespace(_last_instruction_compliance=None,
                                  _release_local_gpu_before_comfy=lambda: None)

        def generate(**kwargs):
            manager._last_instruction_compliance = verdict
            if reject:
                allow_model("other-model.gguf")
            if crash:
                raise RuntimeError("controlled diagnostic error")
            if not reject:
                record_execution("Kontext", get_kontext_model_config()["model_file"])
            return kwargs["image"].copy(), "OK", None

        manager.generate_intelligent = generate
        with tempfile.TemporaryDirectory(prefix="engine_status_") as directory:
            with (patch.object(events, "get_img_editor_manager", return_value=manager),
                  patch.object(events, "get_img_editor_output_dir", return_value=directory),
                  patch.object(events, "_cleanup_vram")):
                result = events.on_generate(image, prompt, "kontext", 0, False, 1, False, "hd")
            files = list(Path(directory).glob("*.png"))
        self.assertFalse(state.is_generating)
        return result, files

    def test_unknown_verdict_is_visible(self):
        (image, status, _), files = self.callback({"compliant": None})
        self.assertIsNotNone(image)
        self.assertEqual(len(files), 1)
        self.assertIn("sin verificar", status.lower())
        self.assertIn("Motor ejecutado: Kontext", status)

    def test_negative_verdict_is_visible(self):
        (_, status, _), _files = self.callback({"compliant": False, "unmet": "jacket still red"})
        self.assertIn("no confirmada", status.lower())
        self.assertIn("jacket still red", status)

    def test_positive_is_a_signal_not_a_quality_certificate(self):
        (_, status, _), _files = self.callback({"compliant": True})
        self.assertIn("señal positiva", status.lower())
        self.assertIn("revisar", status.lower())

    def test_rejected_request_neither_returns_nor_saves_candidate(self):
        (image, status, _), files = self.callback(None, reject=True)
        self.assertIsNone(image)
        self.assertIn("Kontext", status)
        self.assertEqual(files, [])

    def test_exception_releases_report_and_generation_lock(self):
        from roop.img_editor.edit_execution import current_execution

        (image, status, _), files = self.callback(None, crash=True)
        self.assertIsNone(image)
        self.assertIn("controlled diagnostic error", status)
        self.assertEqual(files, [])
        self.assertIsNone(current_execution())
        (_, next_status, _), _files = self.callback(None)
        self.assertIn("sin verificar", next_status.lower())

    def test_classical_quality_does_not_claim_semantic_verification(self):
        from roop.img_editor.edit_execution import begin_execution, end_execution, format_execution_status

        report, token = begin_execution("kontext")
        try:
            status = format_execution_status(report, None, has_instruction=False)
            self.assertEqual(status, "")
        finally:
            end_execution(token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
