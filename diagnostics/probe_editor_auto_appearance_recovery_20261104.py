"""CPU contracts for Auto's configured pure-appearance recovery.

Dispatch, model policy and execution recording are real. Image inference,
semantic/segmentation fixtures and translation are explicit test doubles.
No downloads, backend requests, UI startup or GPU inference are performed.
"""
from __future__ import annotations

from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from diagnostics import probe_editor_kontext_people_protection_20261018 as fixture  # noqa: E402
from roop.img_editor import gen_prompt_config as config  # noqa: E402
from roop.img_editor.edit_execution import (  # noqa: E402
    allow_model, begin_execution, current_execution, end_execution, record_execution,
)
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


PURE_FACTS = {
    "authority": True, "local_appearance": True, "pose": False, "body": False,
    "scenario": False, "structural_add": False, "structural_remove": False,
    "region_ambiguous": False, "surface_recolor": False, "local_region": False,
    "body_direction": "change",
}
BALD = "The person wearing the amber scarf must be bald"


class PolicyRecordingClient(fixture.RecordingClient):
    """Inference double uses the real permission gate before recording pixels."""

    def __init__(self):
        super().__init__()
        self.model_file = None
        self.permission_at_load = []

    def load(self, *args, **kwargs):
        model = args[0] if args else kwargs.get("flux_version", kwargs.get("qwen_version"))
        report = current_execution()
        self.permission_at_load.append((model, bool(report and model in report.automatic_models)))
        allowed, note = allow_model(model)
        if not allowed:
            return False, note
        self.model_file = model
        return super().load(*args, **kwargs)

    def generate(self, **kwargs):
        allowed, note = allow_model(self.model_file)
        if not allowed:
            return None, note
        result, note = super().generate(**kwargs)
        label = "Kontext" if "kontext" in str(self.model_file).lower() else "LongCat"
        record_execution(label, self.model_file)
        return result, note


class AutoAppearanceRecovery(unittest.TestCase):
    def settings(self, engine="kontext"):
        settings = deepcopy(config.get_generation_prompt_config())
        settings.setdefault("editor_ui", {})["automatic_appearance_engine"] = engine
        return patch.object(config, "get_generation_prompt_config", return_value=settings)

    def helper(self, manager):
        helper = getattr(manager, "_automatic_appearance_engine", None)
        self.assertTrue(callable(helper), "Missing _automatic_appearance_engine: Auto appearance recovery is not implemented")
        return helper

    def execute(self, engine="auto", configured="kontext", extra_kwargs=None):
        report, token = begin_execution(engine)
        try:
            if engine == "auto":
                self.assertNotIn(config.get_kontext_model_config()["model_file"], report.automatic_models)
            with ExitStack() as stack:
                if extra_kwargs:
                    real_dispatch = ImgEditorManager.generate_intelligent

                    def dispatch(manager, **kwargs):
                        return real_dispatch(manager, **{**kwargs, **extra_kwargs})

                    stack.enter_context(patch.object(ImgEditorManager, "generate_intelligent", new=dispatch))
                stack.enter_context(self.settings(configured))
                stack.enter_context(patch.object(fixture, "RecordingClient", PolicyRecordingClient))
                stack.enter_context(patch("roop.img_editor.selected_region_edit.edit_described_subject", return_value=None))
                run = fixture._execute(engine=engine, prompt=BALD)
            return run, report
        finally:
            end_execution(token)

    def test_auto_pure_appearance_dispatch_policy_and_recording(self):
        run, report = self.execute()
        expected = config.get_kontext_model_config()["model_file"]
        self.assertIsNotNone(run.output, run.message)
        self.assertTrue(run.client.generations, run.message)
        self.assertEqual([entry.model_file for entry in report.models], [expected])
        self.assertEqual(report.rejection, "")
        self.assertTrue(run.client.permission_at_load)
        self.assertTrue(all(model == expected and permitted for model, permitted in run.client.permission_at_load))

    def test_explicit_longcat_is_preserved(self):
        run, report = self.execute(engine="longcat")
        expected = config.get_longcat_model_config()["model_file"]
        self.assertIsNotNone(run.output, run.message)
        self.assertTrue(run.client.generations, run.message)
        self.assertEqual([entry.model_file for entry in report.models], [expected])

    def test_config_can_retain_longcat(self):
        run, report = self.execute(configured="longcat")
        self.assertIsNotNone(run.output, run.message)
        self.assertEqual([entry.model_file for entry in report.models], [config.get_longcat_model_config()["model_file"]])

    def test_painted_manual_mask_keeps_longcat(self):
        mask = Image.new("L", (640, 420), 0)
        mask.paste(255, (350, 80, 460, 230))
        run, report = self.execute(extra_kwargs={"mask_image": mask, "mask_mode": "manual"})
        self.assertIsNotNone(run.output, run.message)
        self.assertTrue(run.client.generations, run.message)
        self.assertEqual([entry.model_file for entry in report.models], [config.get_longcat_model_config()["model_file"]])

    def test_pure_appearance_categories_use_measured_facts(self):
        manager = ImgEditorManager()
        helper = self.helper(manager)
        with self.settings(), patch.object(manager, "_prompt_authority_facts", return_value=PURE_FACTS):
            for prompt in ("The person must be bald", "The person must look older",
                           "The person must have long hair", "The person must have green eyes"):
                with self.subTest(prompt=prompt):
                    self.assertEqual(helper(prompt, prompt, "longcat"), "kontext")

    def test_other_measured_intents_are_excluded(self):
        manager = ImgEditorManager()
        helper = self.helper(manager)
        excluded = ("pose", "body", "scenario", "structural_add", "structural_remove",
                    "region_ambiguous", "surface_recolor", "local_region")
        with self.settings():
            for flag in excluded:
                with self.subTest(flag=flag), patch.object(manager, "_prompt_authority_facts", return_value={**PURE_FACTS, flag: True}):
                    self.assertEqual(helper(BALD, BALD, "longcat"), "longcat")
            with patch.object(manager, "_prompt_authority_facts", return_value={**PURE_FACTS, "local_appearance": False}):
                self.assertEqual(helper(BALD, BALD, "longcat"), "longcat")

    def test_compounds_in_either_language_are_excluded(self):
        manager = ImgEditorManager()
        helper = self.helper(manager)
        with self.settings(), patch.object(manager, "_prompt_authority_facts", return_value=PURE_FACTS):
            compound = "The person must be bald, the wall must be grey"
            self.assertTrue(manager._split_compound_instruction(compound), "compound fixture is not split")
            for raw, translated in ((compound, BALD), (BALD, compound)):
                with self.subTest(raw=raw, translated=translated):
                    self.assertEqual(helper(raw, translated, "longcat"), "longcat")

    def test_explicit_current_engine_is_not_substituted(self):
        manager = ImgEditorManager()
        helper = self.helper(manager)
        with self.settings(), patch.object(manager, "_prompt_authority_facts", return_value=PURE_FACTS):
            for engine in ("kontext", "qwen_camera", "klein", "longcat_full"):
                with self.subTest(engine=engine):
                    self.assertEqual(helper(BALD, BALD, engine), engine)

    def test_invalid_configuration_rejected_before_inference(self):
        manager = ImgEditorManager()
        helper = self.helper(manager)
        with self.settings("unregistered-engine"), patch.object(manager, "_prompt_authority_facts", return_value=PURE_FACTS):
            with self.assertRaises(ValueError):
                helper(BALD, BALD, "longcat")

    def test_invalid_configuration_dispatch_never_loads_model(self):
        self.helper(ImgEditorManager())
        run, report = self.execute(configured="unregistered-engine")
        self.assertIsNone(run.output, run.message)
        self.assertEqual(run.client.loads, [])
        self.assertEqual(run.client.generations, [])
        self.assertEqual(report.models, [])

    def test_policy_rejects_kontext_without_measured_decision(self):
        report, token = begin_execution("auto")
        try:
            allowed, note = allow_model(config.get_kontext_model_config()["model_file"])
            self.assertFalse(allowed)
            self.assertTrue(note)
            self.assertTrue(report.rejection)
            self.assertEqual(report.models, [])
        finally:
            end_execution(token)

    def test_described_subject_returns_before_recovery(self):
        image = Image.new("RGB", (48, 32), (110, 130, 150))
        manager = ImgEditorManager()
        helper = self.helper(manager)
        with (
            self.settings(),
            patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
            patch("roop.face_util.get_face_analyser", return_value=None),
            patch("roop.img_editor.prompt_translator.translate_prompt", return_value=BALD),
            patch.object(manager, "_explicit_viewpoint_request", return_value=False),
            patch.object(manager, "_validate_visual_edit", return_value=(True, "CPU changed fixture")),
            patch.object(manager, "_automatic_appearance_engine", wraps=helper) as recovery,
            patch("roop.img_editor.selected_region_edit.edit_described_subject", return_value=(image, "described fixture", None)) as described,
        ):
            output, message, _ = manager.generate_intelligent(image=image, prompt=BALD, engine="auto")
        self.assertIsNotNone(output)
        self.assertEqual(message, "described fixture")
        self.assertEqual(described.call_args.kwargs["engine"], "longcat")
        self.assertTrue(described.call_args.kwargs["automatic"])
        recovery.assert_not_called()

    def test_camera_route_returns_before_recovery(self):
        from diagnostics.probe_editor_engine_status_20261024 import EngineStatusContracts

        self.helper(ImgEditorManager())
        with patch.object(ImgEditorManager, "_automatic_appearance_engine", side_effect=AssertionError("camera reached appearance recovery")) as recovery:
            EngineStatusContracts().test_auto_routes_measured_camera_request_to_qwen_camera()
        recovery.assert_not_called()

    def test_compound_recursion_and_generative_quality_do_not_recover(self):
        self.helper(ImgEditorManager())
        for overrides in ({"_compound_depth": 1}, {"quality_mode": True, "quality_use_generative": True}):
            with self.subTest(overrides=overrides), patch.object(ImgEditorManager, "_automatic_appearance_engine", side_effect=AssertionError("excluded route reached recovery")) as recovery:
                run, report = self.execute(extra_kwargs=overrides)
            recovery.assert_not_called()
            self.assertIsNotNone(run.output, run.message)
            self.assertTrue(run.client.generations, run.message)
            self.assertEqual([entry.model_file for entry in report.models], [config.get_longcat_model_config()["model_file"]])

    def test_manual_visual_selection_cancel_and_pure_quality_bypass_recovery(self):
        image = Image.new("RGB", (48, 32), (110, 130, 150))
        for kind in ("selection", "cancel", "pure_quality"):
            with self.subTest(kind=kind), ExitStack() as stack:
                manager = ImgEditorManager()
                helper = self.helper(manager)
                spy = stack.enter_context(patch.object(manager, "_automatic_appearance_engine", wraps=helper))
                stack.enter_context(patch("ui.tabs.comfy_launcher.start", side_effect=AssertionError("backend must not start")))
                kwargs = {"image": image, "prompt": BALD, "engine": "auto"}
                if kind == "selection":
                    selected = stack.enter_context(patch("roop.img_editor.selected_region_edit.edit_selected_region", return_value=(image, "selected fixture", None)))
                    kwargs.update(selection_analysis=SimpleNamespace(), selected_region_ids=["region-1"])
                elif kind == "cancel":
                    kwargs["cancel_check"] = lambda: True
                else:
                    stack.enter_context(patch("roop.img_editor.top_tier_finish.run_top_tier_quality", return_value=(image, "quality fixture", {})))
                    kwargs.update(prompt="", quality_mode=True)
                output, message, _ = manager.generate_intelligent(**kwargs)
                spy.assert_not_called()
                if kind == "selection":
                    self.assertEqual(selected.call_args.kwargs["engine"], "longcat")
                elif kind == "cancel":
                    self.assertIsNone(output)
                    self.assertIn("Cancelado", message)
                else:
                    self.assertIsNotNone(output)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AutoAppearanceRecovery)
    assert suite.countTestCases() > 0, "No tests collected"
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.testsRun > 0, "No tests executed"
    raise SystemExit(0 if result.wasSuccessful() else 1)
