"""CPU contracts for H-EDIT-VERSATILITY-INTEGRITY.

Synthetic masks, translation and diffusion are explicit test doubles. Production
dispatch, parsing, composition and pixel protection run unchanged. No backend,
downloads or GPU inference; these tests do not certify image quality.
"""
from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor.qwen_camera_comfy_client import remaining_edit  # noqa: E402


class CPUContracts(unittest.TestCase):
    """Block external effects while retaining the production dispatcher."""

    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for name, value in (
            ("ui.tabs.comfy_launcher.is_comfyui_running", True),
            ("roop.face_util.get_face_analyser", None),
            ("roop.face_util.get_face_analyser_cpu", None),
            ("roop.img_editor.img_editor_manager.get_vram_gb", 8.0),
            ("roop.comfy_workflows.get_comfyui_url", "http://127.0.0.1:8188"),
            ("onnxruntime.get_available_providers", ["CPUExecutionProvider"]),
            ("roop.img_editor.instruction_verifier.get_instruction_verifier", SimpleNamespace(
                verify=lambda *_a, **_kw: {"compliant": True, "source": "CPU verifier double"},
            )),
            ("roop.img_editor.top_tier_finish.should_edit_top_finish", False),
            ("roop.img_editor.image_quality_pipeline.get_quality_finisher", SimpleNamespace(
                analyze_image=lambda *_a, **_kw: {"scores": {
                    "poster": 0.0, "sharpness": 1.0, "restore": 0.0,
                }},
            )),
        ):
            self.stack.enter_context(patch(name, return_value=value))
        self.stack.enter_context(patch(
            "ui.tabs.comfy_launcher.start",
            side_effect=AssertionError("CPU contract forbids backend startup"),
        ))
        self.stack.enter_context(patch(
            "requests.sessions.Session.request",
            side_effect=AssertionError("CPU contract forbids HTTP"),
        ))
        self.manager = ImgEditorManager()
        self.stack.enter_context(patch.object(self.manager, "_release_local_gpu_before_comfy"))
        self.stack.enter_context(patch.object(self.manager, "_get_clothing_segmenter", return_value=None))
        self.stack.enter_context(patch.object(self.manager, "_get_clipseg_masker", return_value=None))


class SentenceContracts(CPUContracts):
    def test_coordinated_addition_list_keeps_terminal_punctuation(self):
        for text in ("Add a man, a woman and a dog", "Add red cup, blue vase"):
            for ending in ("", ".", ";", "\n"):
                with self.subTest(text=text, ending=ending):
                    self.assertEqual(self.manager._split_compound_instruction(text + ending), [])

    def test_independent_imperatives_split_even_when_addition_dominates(self):
        for clauses in (
            ("Make the woman bald", "Add a person near the door"),
            ("Remove the cup", "Add a vase near the door"),
        ):
            for separator in (". ", "\n"):
                with self.subTest(clauses=clauses, separator=separator):
                    self.assertEqual(
                        self.manager._split_compound_instruction(separator.join(clauses)),
                        list(clauses),
                    )

    def test_sentences_and_newlines_keep_all_auto_clauses(self):
        for separator in (". ", "\n", "; ", ", "):
            with self.subTest(separator=separator):
                prompt = separator.join(("The woman must be bald", "The wall must be blue"))
                self.assertEqual(self.manager._split_compound_instruction(prompt), [
                    "The woman must be bald", "The wall must be blue",
                ])
                self.assertEqual(
                    self.manager._automatic_appearance_engine(prompt, prompt, "longcat"),
                    "longcat",
                )

    def test_decimal_and_sentence_after_number(self):
        self.assertEqual(self.manager._split_compound_instruction(
            "Apply 0.5 intensity. Make the wall blue.",
        ), ["Apply 0.5 intensity", "Make the wall blue"])
        self.assertEqual(self.manager._split_compound_instruction(
            "Move the chair by 2. Make the wall blue.",
        ), ["Move the chair by 2", "Make the wall blue"])

    def test_relative_description_is_one_request(self):
        text = "The person who is standing, must have blue hair"
        self.assertEqual(self.manager._instruction_clauses(text), [
            "The person who is standing must have blue hair",
        ])
        self.assertEqual(self.manager._split_compound_instruction(text), [])


class RemovalContracts(CPUContracts):
    def test_auto_removal_pastes_entire_measured_target_including_small_changes(self):
        self._assert_full_support_removal(proposal=False)

    def test_auto_proposed_removal_keeps_measured_request_and_entire_support(self):
        self._assert_full_support_removal(proposal=True)

    def _assert_full_support_removal(self, proposal):
        source = Image.new("RGB", (240, 160), (100, 120, 140))
        support = np.zeros((160, 240), np.uint8)
        support[60:100, 100:140] = 255
        target = Image.fromarray(support)
        spoon_queries = []

        def segment_target(_image, query):
            if query != "spoon":
                return None
            spoon_queries.append(query)
            # Direct grounding fails once; a subsequent planner proposal is
            # measured by the same segmenter, never accepted on text alone.
            return target if not proposal or len(spoon_queries) > 1 else None

        segment = Mock(side_effect=segment_target)
        measured_results = []
        original_measure = self.manager._measured_region_passes

        def observe_measure(*args, **kwargs):
            result = original_measure(*args, **kwargs)
            measured_results.extend(result)
            return result

        class UnevenChangeClient(RecordingDiffusion):
            """A valid removal can expose material very close to its old color."""

            def generate(self, **kwargs):
                self.generations.append(kwargs)
                reference = np.asarray(kwargs["image"], dtype=np.int16)
                region = np.asarray(kwargs["mask_image"].convert("L")) > 0
                yy, _xx = np.indices(region.shape)
                rows = np.nonzero(region)[0]
                upper = region & (yy < (int(rows.min()) + int(rows.max()) + 1) // 2)
                # Also repaint unrequested context, which must remain protected.
                generated = reference + 20
                generated[region] = reference[region] + 2
                generated[upper] = reference[upper] + 100
                return SimpleNamespace(image=Image.fromarray(generated.astype(np.uint8))), "CPU uneven-change double"

        client = UnevenChangeClient()
        with (
            patch.object(self.manager, "_get_clipseg_masker",
                         return_value=SimpleNamespace(generate_mask=segment)),
            patch.object(self.manager, "_subject_queries", return_value=["unmeasured direct request"]),
            patch.object(self.manager, "_region_proposals", return_value=["spoon"]) as planner,
            patch.object(self.manager, "_measured_region_passes", side_effect=observe_measure),
            patch.object(self.manager, "_continue_measured_region", return_value=target),
            patch.object(self.manager, "_second_focused_pass", return_value=None),
            patch.object(self.manager, "_edit_is_nearly_unchanged", return_value=False),
            patch.object(self.manager, "_edit_leaves_a_ghost", return_value=False),
            patch.object(self.manager, "_added_face_mask", return_value=None),
            patch.object(self.manager, "_validate_visual_edit", return_value=(True, "CPU validation double")),
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        ):
            output, message, _ = self.manager.generate_intelligent(
                source, "Remove the spoon beside the cup", engine="auto",
                use_semantic=False, use_rewriter=False, enhance_faces=False,
            )
        self.assertIsNotNone(output, message)
        self.assertEqual(len(client.generations), 1, message)
        if proposal:
            planner.assert_called_once()
            self.assertEqual(spoon_queries, ["spoon", "spoon"])
        else:
            planner.assert_not_called()
        before = np.asarray(source, dtype=np.int16)
        after = np.asarray(output, dtype=np.int16)
        delta = after - before
        high = np.zeros_like(support, dtype=bool)
        high[60:80, 100:140] = True
        subtle = (support > 0) & ~high
        self.assertTrue(np.all(delta[high] == 100), "strongly changed target pixels were lost")
        self.assertTrue(np.all(delta[subtle] == 2),
                        "source was reintroduced inside the measured target by delta filtering; "
                        f"measured_request={[item.get('measured_request') for item in measured_results]}")
        self.assertTrue(np.all(delta[support == 0] == 0), "unrequested context escaped the measured removal support")
        self.assertEqual(len(measured_results), 1)
        self.assertTrue(measured_results[0].get("measured_request"),
                        "grounded proposal must carry the same removal-support contract as direct grounding")

    def test_generic_removal_helper_does_not_invent_a_body_surface(self):
        text = self.manager._compose_local_target_prompt(
            "Remove the spoon beside the cup", "spoon",
            lower_body=True, body_transform=False,
        )
        self.assertIn("spoon", text.lower())
        for invented_surface in ("bare skin", "legs", "feet"):
            self.assertNotIn(invented_surface, text.lower())

    def test_existing_body_removal_helper_keeps_measured_body_wording(self):
        text = self.manager._compose_local_target_prompt(
            "Remove the covering", "covering", lower_body=True,
        )
        self.assertIn("covering", text.lower())
        self.assertIn("bare skin", text.lower())
        self.assertIn("legs and feet", text.lower())

    def test_auto_regional_object_removal_prompt_reconstructs_neutral_scene(self):
        source = Image.new("RGB", (240, 160), (92, 114, 136))
        pixels = np.zeros((160, 240), np.uint8)
        pixels[60:100, 100:140] = 255
        target = Image.fromarray(pixels)
        prompt = "Remove the spoon beside the cup"
        segment = Mock(side_effect=lambda _image, query: target if query == "spoon" else None)
        client = RecordingDiffusion()
        with (
            patch.object(self.manager, "_get_clipseg_masker",
                         return_value=SimpleNamespace(generate_mask=segment)),
            patch.object(self.manager, "_continue_measured_region", return_value=target),
            patch.object(self.manager, "_second_focused_pass", return_value=None),
            patch.object(self.manager, "_validate_visual_edit", return_value=(True, "CPU validation double")),
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        ):
            output, message, _ = self.manager.generate_intelligent(
                source, prompt, engine="auto", use_semantic=False,
                use_rewriter=False, enhance_faces=False,
            )
        self.assertIsNotNone(output, message)
        self.assertEqual(len(client.generations), 1, message)
        generated_prompt = client.generations[0]["prompt"].lower()
        self.assertIn("spoon", generated_prompt)
        for invented_surface in ("bare skin", "legs", "feet"):
            self.assertNotIn(invented_surface, generated_prompt)
        self.assertTrue(segment.call_args_list)
        self.assertTrue(all(call.args[1] == "spoon" for call in segment.call_args_list))

    def test_ordinary_measured_removal_reaches_generator_with_exact_target_mask(self):
        source = Image.new("RGB", (240, 160), (92, 114, 136))
        pixels = np.zeros((160, 240), np.uint8)
        pixels[60:100, 100:140] = 255
        target = Image.fromarray(pixels)
        prompt = "Remove the spoon beside the cup"
        segment = Mock(return_value=target)
        analyzer = SimpleNamespace(
            get_magnitude=lambda *_: 0.8,
            detect_target=lambda *_: "object",
            get_axis_scores=lambda *_: {"attribute": 0.0, "pose": 0.0, "structural": 0.9},
            score_concepts=lambda *_: {"structural_remove": 0.9},
            get_body_transform_intensity=lambda *_: 0.0,
            get_body_direction=lambda *_: "change",
            is_structural_dominant=lambda *_: True,
            get_structural_bias=lambda *_: "remove",
            is_quality_dominant=lambda *_: False,
        )
        client = RecordingDiffusion()
        with (
            patch("roop.img_editor.selected_region_edit.edit_described_subject", return_value=None),
            patch.object(self.manager, "_get_semantic_analyzer", return_value=analyzer),
            patch.object(self.manager, "_get_clipseg_masker",
                         return_value=SimpleNamespace(generate_mask=segment)),
            patch.object(self.manager, "_build_object_removal_mask",
                         wraps=self.manager._build_object_removal_mask) as target_mask,
            patch.object(self.manager, "_build_person_removal_mask") as other_person,
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
            patch("roop.img_editor.controlnet_utils.get_controlnet_utils",
                  return_value=SimpleNamespace(detect_pose=lambda *_: None)),
        ):
            output, message, returned_mask = self.manager.generate_intelligent(
                source, prompt, engine="kontext", use_semantic=True,
                use_rewriter=False, enhance_faces=False,
            )
        target_mask.assert_called_once()
        self.assertEqual(target_mask.call_args.kwargs.get("target"), "spoon")
        other_person.assert_not_called()
        self.assertTrue(segment.call_args_list)
        self.assertTrue(all(call.args[1] == "spoon" for call in segment.call_args_list))
        self.assertIsNotNone(output, message)
        self.assertEqual(len(client.generations), 1, message)
        request = client.generations[0]
        self.assertIsNotNone(request.get("mask_image"))
        self.assertTrue(np.array_equal(np.asarray(request["mask_image"]), pixels))
        self.assertIsNotNone(returned_mask)
        self.assertTrue(np.array_equal(np.asarray(returned_mask), pixels))

    def test_ordinary_missing_object_never_falls_back_to_unrelated_person(self):
        source = Image.new("RGB", (240, 160), (92, 114, 136))
        prompt = "Remove the spoon beside the cup"
        analyzer = SimpleNamespace(
            get_magnitude=lambda *_: 0.8,
            detect_target=lambda *_: "object",
            get_axis_scores=lambda *_: {"attribute": 0.0, "pose": 0.0, "structural": 0.9},
            score_concepts=lambda *_: {"structural_remove": 0.9},
            get_body_transform_intensity=lambda *_: 0.0,
            get_body_direction=lambda *_: "change",
            is_structural_dominant=lambda *_: True,
            get_structural_bias=lambda *_: "remove",
            is_quality_dominant=lambda *_: False,
        )
        client = RecordingDiffusion()
        with (
            patch("roop.img_editor.selected_region_edit.edit_described_subject", return_value=None),
            patch.object(self.manager, "_get_semantic_analyzer", return_value=analyzer),
            patch.object(self.manager, "_build_object_removal_mask", return_value=None) as target_mask,
            patch.object(self.manager, "_build_person_removal_mask",
                         return_value=Image.new("L", source.size, 255)) as other_person,
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
            patch("roop.img_editor.controlnet_utils.get_controlnet_utils",
                  return_value=SimpleNamespace(detect_pose=lambda *_: None)),
        ):
            output, message, _ = self.manager.generate_intelligent(
                source, prompt, engine="kontext", use_semantic=True,
                use_rewriter=False, enhance_faces=False,
            )
        target_mask.assert_called_once()
        self.assertEqual(target_mask.call_args.kwargs.get("target"), "spoon")
        other_person.assert_not_called()
        self.assertIsNotNone(output, message)
        self.assertEqual(output.tobytes(), source.tobytes())
        self.assertEqual(client.generations, [])
        self.assertIn("no medid", message.lower())

    def test_object_mask_queries_only_requested_target_and_keeps_exterior(self):
        pixels = np.zeros((160, 240), np.uint8)
        pixels[60:100, 100:140] = 255
        neighbors = pixels.copy()
        neighbors[25:70, 175:215] = 255
        segment = Mock(return_value=Image.fromarray(pixels))
        generic_queries = []

        def segment_prompts(_image, prompts, **_kwargs):
            # The old clothing API unions its concepts. A generic object query
            # therefore legitimately returns a second object beside the cup.
            generic_queries.append(list(prompts))
            mask = pixels if prompts == ["cup"] else neighbors
            return Image.fromarray(mask), "CPU segmentation double"

        with (
            patch.object(self.manager, "_get_clipseg_masker", return_value=SimpleNamespace(
                generate_mask=segment,
            )),
            patch.object(self.manager, "_get_clothing_segmenter", return_value=SimpleNamespace(
                is_available=lambda: True, segment_with_prompt=segment_prompts,
            )),
        ):
            for color in ("white", "black"):
                image = Image.new("RGB", (240, 160), color)
                result = self.manager._build_object_removal_mask(image, target="cup")
                self.assertIsNotNone(result)
                self.assertTrue(np.array_equal(np.asarray(result), pixels))
        self.assertEqual([call.args[1] for call in segment.call_args_list], ["cup", "cup"])
        self.assertFalse(generic_queries, "removal must not union a generic object inventory")

    def test_missing_target_never_guesses_a_person_or_object(self):
        with patch.object(self.manager, "_get_clipseg_masker") as get_masker:
            self.assertIsNone(self.manager._build_object_removal_mask(Image.new("RGB", (120, 80))))
        get_masker.assert_not_called()

    def test_unmeasured_empty_and_saturated_target_cannot_authorize_removal(self):
        image = Image.new("RGB", (120, 80))
        for mask in (None, Image.new("L", image.size, 0), Image.new("L", image.size, 255)):
            with self.subTest(mask=mask), patch.object(
                self.manager, "_get_clipseg_masker",
                return_value=SimpleNamespace(generate_mask=Mock(return_value=mask)),
            ):
                self.assertIsNone(self.manager._build_object_removal_mask(image, target="cup"))

    def test_person_closure_keeps_target_and_excludes_adjacent_neighbor(self):
        target = np.zeros((200, 200), np.uint8)
        target[30:170, 90:110] = 255
        target[60:75, 30:170] = 255
        neighbor = np.zeros_like(target)
        neighbor[76:101, 120:148] = 255
        people = np.maximum(target, neighbor)
        source = Image.new("RGB", (200, 200), "white")
        mask = Image.fromarray(target)
        with (
            patch.object(self.manager, "_get_clipseg_masker", return_value=object()),
            patch.object(self.manager, "_background_mask_from_people",
                         return_value=Image.fromarray(255 - people)),
        ):
            result = self.manager._closed_structural_zone(
                source, mask, faces=[SimpleNamespace(bbox=(90, 30, 110, 55))],
            )
        self.assertIsNotNone(result)
        allowed = np.asarray(result) >= 96
        self.assertEqual(int(allowed[target > 0].sum()), 4600)
        self.assertEqual(int(allowed[neighbor > 0].sum()), 0)
        outside_box = np.ones_like(allowed)
        outside_box[30:170, 30:170] = False
        self.assertFalse(allowed[outside_box].any())
        self.assertTrue(np.array_equal(np.asarray(mask), target))
        self.assertEqual(source.getextrema(), ((255, 255),) * 3)

    def test_small_target_same_mask_in_simple_and_compound(self):
        image = Image.new("RGB", (200, 100), "white")
        pixels = np.zeros((100, 200), np.uint8)
        pixels[70:78, 30:60] = 255
        target = Image.fromarray(pixels)
        segment = Mock(side_effect=lambda _image, query: target if query == "spoon" else None)
        prompt = "Remove the spoon beside the cup"
        with (
            patch.object(self.manager, "_get_clipseg_masker",
                         return_value=SimpleNamespace(generate_mask=segment)),
            patch.object(self.manager, "_continue_measured_region", return_value=target),
        ):
            simple = self.manager._measured_region_passes(
                image, prompt, prompt, structural_remove=True,
            )
            self.assertEqual(len(simple), 1)
            self.assertTrue(simple[0].get("measured_request"),
                            "measured removal support must not be reduced by output delta")
            for facts in ({"structural_remove": True},
                          {"local_region": True, "structural_remove": True}):
                with self.subTest(facts=facts):
                    compound = self.manager._compound_clause_zone(
                        image, prompt, prompt, facts, faces=[],
                    )
                    self.assertIsNotNone(compound)
                    self.assertTrue(np.array_equal(np.asarray(compound), pixels))
                    self.assertTrue(np.array_equal(np.asarray(simple[0]["mask"]), pixels))
        self.assertTrue(segment.call_args_list)
        self.assertTrue(all(call.args[1] == "spoon" for call in segment.call_args_list))


class RecordingDiffusion:
    """Explicit adversarial inference double replaces all sampled pixels."""

    def __init__(self):
        self.loads = []
        self.generations = []

    def load(self, *args, **kwargs):
        self.loads.append((args, kwargs))
        return True, "CPU inference double"

    def generate(self, **kwargs):
        self.generations.append(kwargs)
        return SimpleNamespace(image=Image.new("RGB", kwargs["image"].size, "red")), "CPU inference double"


class AdditionContracts(CPUContracts):
    def test_addition_vetoes_unmeasured_clause_even_without_local_facts(self):
        source = Image.new("RGB", (240, 160), (92, 114, 136))
        first_clause = "Make the cup blue"
        addition = "Add a small red sugar packet on the table beside the cup"
        prompt = f"{first_clause}. {addition}."
        support = Image.new("L", source.size, 0)
        support.paste(255, (120, 60, 220, 140))
        client = RecordingDiffusion()

        def measure(_image, _raw, clause, **_kwargs):
            return [{"mask": support, "text": clause}] if clause == addition else []

        with (
            patch.object(self.manager, "_measured_subject_passes", return_value=[]),
            patch.object(self.manager, "_measured_region_passes", side_effect=measure) as measurement,
            patch.object(self.manager, "_second_focused_pass", return_value=None),
            patch.object(self.manager, "_validate_visual_edit", return_value=(True, "CPU validation double")),
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        ):
            # The production semantic facts remain real: this first clause
            # currently lacks a strong local flag, which cannot authorize its omission.
            output, message, _ = self.manager.generate_intelligent(
                source, prompt, engine="auto", use_semantic=False,
                use_rewriter=False, enhance_faces=False,
            )
        self.assertIsNotNone(output, message)
        self.assertTrue(any(call.args[2] == addition for call in measurement.call_args_list),
                        "the addition support was not measured")
        self.assertEqual(client.generations, [],
                         "an unmeasured non-global clause must veto the complete addition compound")
        self.assertEqual(output.tobytes(), source.tobytes())
        self.assertIn(first_clause, message)
        self.assertIn("no medida", message.lower())

    def dispatch(self, support, prompt=None, first_region_clause=None):
        source = Image.new("RGB", (240, 160), (92, 114, 136))
        head_pixels = np.zeros((160, 240), np.uint8)
        head_pixels[25:65, 25:65] = 255
        head = Image.fromarray(head_pixels)
        prompt = prompt or "The woman must be bald, Add a person near the door"
        client = RecordingDiffusion()

        def measure(_image, _raw, clause, **_kwargs):
            if clause == first_region_clause:
                return [{"mask": head, "text": clause, "coverage": 1600 / 38400}]
            return [] if support is None else [{"mask": support, "text": clause}]

        with (
            patch.object(self.manager, "_measured_subject_passes", return_value=[{
                "mask": head, "text": prompt, "coverage": 1600 / 38400,
            }]),
            patch.object(self.manager, "_measured_region_passes",
                         side_effect=measure) as measurement,
            patch.object(self.manager, "_second_focused_pass", return_value=None),
            patch.object(self.manager, "_validate_visual_edit", return_value=(True, "CPU double")),
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        ):
            output, message, _ = self.manager.generate_intelligent(
                source, prompt, engine="auto", use_semantic=False,
                use_rewriter=False, enhance_faces=False,
            )
        return SimpleNamespace(source=source, output=output, message=message,
                               client=client, head=head_pixels, measurement=measurement)

    def test_missing_addition_after_imperative_preserves_every_clause(self):
        for clauses in (
            ("Make the woman bald", "Add a person near the door"),
            ("Remove the cup", "Add a vase near the door"),
        ):
            for separator in (". ", "\n"):
                with self.subTest(clauses=clauses, separator=separator):
                    run = self.dispatch(None, separator.join(clauses), clauses[0])
                    self.assertIsNotNone(run.output, run.message)
                    self.assertEqual(run.output.tobytes(), run.source.tobytes())
                    self.assertEqual(run.client.generations, [], run.message)
                    self.assertIn(clauses[1], run.message)
                    self.assertIn("no medida", run.message.lower())

    def test_missing_addition_support_preserves_photo_and_reports_clause(self):
        run = self.dispatch(None)
        self.assertIsNotNone(run.output, run.message)
        self.assertEqual(run.output.tobytes(), run.source.tobytes())
        self.assertEqual(run.client.generations, [])
        self.assertIn("no medida", run.message.lower())
        self.assertIn("Add a person near the door", run.message)
        run.measurement.assert_called_once()

    def test_measured_addition_is_in_sampler_and_paste_union(self):
        pixels = np.zeros((160, 240), np.uint8)
        pixels[30:130, 155:215] = 255
        run = self.dispatch(Image.fromarray(pixels))
        self.assertIsNotNone(run.output, run.message)
        self.assertEqual(len(run.client.generations), 1, run.message)
        request = run.client.generations[0]
        self.assertIn("The woman must be bald", request["prompt"])
        self.assertIn("Add a person near the door", request["prompt"])
        self.assertIsNotNone(request["mask_image"])
        self.assertTrue(request["keep_source_pixels"])
        changed = np.any(np.asarray(run.output) != np.asarray(run.source), axis=2)
        union = np.maximum(run.head, pixels) > 0
        self.assertTrue(changed[run.head > 0].any(), "head clause was not pasted")
        self.assertTrue(changed[pixels > 0].any(), "addition clause was not pasted")
        self.assertFalse(changed[~union].any(), "adversarial generator leaked outside measured union")


class CameraContracts(CPUContracts):
    def dispatch(self, prompt):
        camera_calls = []
        followup_calls = []
        source = Image.new("RGB", (120, 80), "white")
        support = Image.new("L", source.size, 0)
        support.paste(255, (10, 10, 60, 50))

        def camera_generate(image, text, **kwargs):
            camera_calls.append(text)
            return SimpleNamespace(image=Image.new("RGB", image.size, "green")), "CPU camera double"

        original = self.manager.generate_intelligent

        def dispatch(image, text, **kwargs):
            if kwargs.get("mask_image") is not None:
                followup_calls.append((text, kwargs))
                result = image.copy()
                result.paste("red", (0, 0, image.width, image.height), kwargs["mask_image"])
                return result, "CPU local follow-up", kwargs["mask_image"]
            return original(image, text, **kwargs)

        with (
            patch("roop.img_editor.qwen_camera_comfy_client.QwenCameraComfyClient",
                  return_value=SimpleNamespace(generate=camera_generate)),
            patch.object(self.manager, "_measure_camera_followup_mask", return_value=support),
            patch.object(self.manager, "generate_intelligent", side_effect=dispatch),
        ):
            result, message, _ = self.manager.generate_intelligent(
                source, prompt, engine="auto", use_semantic=False,
                use_rewriter=False, enhance_faces=False,
            )
        self.assertIsNotNone(result, message)
        self.assertEqual(len(camera_calls), 1, message)
        return camera_calls[0], followup_calls

    def test_observer_is_retained_and_distinct_in_real_dispatch(self):
        dispatched = []
        for observer in ("man", "woman"):
            prompt = f"First person POV through the {observer}'s eyes"
            text, followup = self.dispatch(prompt)
            self.assertTrue(text.startswith("<sks> "))
            self.assertIn(prompt, text)
            self.assertEqual(followup, [])
            dispatched.append(text)
        self.assertNotEqual(*dispatched)

    def test_mixed_pov_preserves_observer_and_local_followup(self):
        for separator in (", ", ". ", "\n"):
            with self.subTest(separator=separator):
                camera = "First person POV through the man's eyes"
                extra = "make the wall blue"
                prompt = camera + separator + extra
                self.assertEqual(remaining_edit(prompt, prompt), extra)
                text, followup = self.dispatch(prompt)
                self.assertIn(camera, text)
                self.assertNotIn(extra, text, "local edit must not run unmasked in camera pass")
                self.assertEqual(len(followup), 1)
                self.assertEqual(followup[0][0], extra)
                self.assertIsNotNone(followup[0][1]["mask_image"])


if __name__ == "__main__":
    unittest.main()
