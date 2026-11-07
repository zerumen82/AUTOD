"""CPU reference contracts: real local semantics and observed mask dispatch.

The final routing contracts use inference doubles; they do not certify image
quality. Real CPU mask measurements are in probe_editor_referent_cpu_20261024.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


class RelativeSubjectContracts(unittest.TestCase):
    def setUp(self):
        self.manager = ImgEditorManager()

    def test_exact_user_instruction_keeps_description_out_of_action(self):
        raw = "EL HOMBRE QUE ESTA DE PIE, DEBE SER CALVO"
        translated = "The man standing must be bald"
        self.assertEqual(self.manager._requested_change_scope(raw), "SER CALVO")
        facts = self.manager._prompt_authority_facts(raw, translated)
        self.assertFalse(facts["pose"], facts)
        self.assertTrue(facts["local_appearance"], facts)

    def test_subject_queries_do_not_truncate_relative_at_comma(self):
        subjects = self.manager._visual_subject_descriptions("EL HOMBRE QUE ESTA DE PIE, DEBE SER CALVO")
        self.assertEqual(subjects, ["EL HOMBRE QUE ESTA DE PIE"])

    def test_appositive_relative_does_not_create_pose(self):
        raw = "The person, who is standing, must have blue hair"
        self.assertEqual(self.manager._requested_change_scope(raw), "have blue hair")
        facts = self.manager._prompt_authority_facts(raw, raw)
        self.assertFalse(facts["pose"], facts)

    def test_requested_pose_is_preserved(self):
        raw = "The person who is standing, must be kneeling"
        self.assertEqual(self.manager._requested_change_scope(raw), "be kneeling")
        self.assertTrue(self.manager._prompt_authority_facts(raw, raw)["pose"])

    def test_independent_subjects_keep_independent_changes(self):
        raw = "The person who is standing, must be bald, the person who is seated, must have blue hair"
        clauses = self.manager._split_compound_instruction(raw)
        self.assertEqual(len(clauses), 2, clauses)
        self.assertEqual(self.manager._requested_change_scope(raw), "be bald, have blue hair")
        facts = self.manager._prompt_authority_facts(raw, raw)
        self.assertFalse(facts["pose"], facts)

    def test_independent_requested_pose_is_not_removed(self):
        raw = "The person must be bald, the other person must be standing"
        self.assertEqual(len(self.manager._split_compound_instruction(raw)), 2)
        self.assertTrue(self.manager._prompt_authority_facts(raw, raw)["pose"])

    def test_descriptive_comma_does_not_create_two_edit_clauses(self):
        raw = "EL HOMBRE QUE ESTA DE PIE, DEBE SER CALVO"
        self.assertEqual(self.manager._split_compound_instruction(raw), [])

    def test_subject_template_preserves_every_descriptor(self):
        from roop.img_editor.clipseg_masker import CLIPSegMasker

        masker = object.__new__(CLIPSegMasker)
        subject = "the person wearing an amber scarf"
        with patch.object(masker, "generate_mask", return_value="measured mask") as generate:
            result = masker.generate_subject_mask(None, subject, "a photo of {subject}")
        self.assertEqual(result, "measured mask")
        generate.assert_called_once_with(None, "a photo of the person wearing an amber scarf")

    def test_invalid_templates_do_not_call_segmenter(self):
        from roop.img_editor.clipseg_masker import CLIPSegMasker
        from roop.img_editor.gen_prompt_config import SubjectPromptConfigError

        masker = object.__new__(CLIPSegMasker)
        for template in (None, "", "photo", "{other}", "{subject:.3}", "{subject!r}", "{subject}{subject}"):
            with self.subTest(template=template), patch.object(masker, "generate_mask") as generate:
                with self.assertRaises(SubjectPromptConfigError):
                    masker.generate_subject_mask(None, "a visual subject", template)
                generate.assert_not_called()

    def test_manager_conditions_subject_but_not_head_completion(self):
        from diagnostics import probe_editor_kontext_people_protection_20261018 as contract
        from roop.img_editor.clipseg_masker import CLIPSegMasker
        from roop.img_editor import element_selection as es

        class SubjectMasker(contract.MeasuredMasker):
            generate_subject_mask = CLIPSegMasker.generate_subject_mask

        with patch.object(contract, "MeasuredMasker", SubjectMasker):
            run = contract._execute()
        template = es.load_settings()["segmenter"]["subject_prompt_template"]
        contract._assert_protected(run, expected_query=template.format(subject=contract.REGION.lower()))
        self.assertIn(template.format(subject=contract.REGION.lower()), run.masker.queries)
        self.assertIn("head", run.masker.queries)
        self.assertIn("hair", run.masker.queries)
        self.assertNotIn(template.format(subject="head"), run.masker.queries)
        self.assertNotIn(template.format(subject="hair"), run.masker.queries)

    def test_bad_configuration_cannot_fall_back_to_another_head(self):
        from diagnostics import probe_editor_kontext_people_protection_20261018 as contract
        from roop.img_editor.clipseg_masker import CLIPSegMasker
        from roop.img_editor import element_selection as es

        class SubjectMasker(contract.MeasuredMasker):
            generate_subject_mask = CLIPSegMasker.generate_subject_mask

        settings = es.load_settings()
        settings["segmenter"]["subject_prompt_template"] = "photo without its subject"
        with (patch.object(contract, "MeasuredMasker", SubjectMasker),
              patch.object(es, "load_settings", return_value=settings)):
            run = contract._execute(single_subject=True)
        self.assertIsNone(run.output)
        self.assertFalse(run.client.generations)
        self.assertIn("configuración", run.message)


if __name__ == "__main__":
    unittest.main(verbosity=2)
