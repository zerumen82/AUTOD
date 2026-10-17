"""Contratos de controles Generate, sin servidor ni inferencias GPU."""
import unittest
from unittest.mock import patch

from roop.img_editor.gen_prompt_modifiers import _resolve_modifier_parts, assemble_generation_prompt
from roop.img_editor.flux_gen_comfy_client import FluxGenComfyClient


class Controls(unittest.TestCase):
    def test_sampler_uses_config_defaults_and_explicit_controls(self):
        import roop.img_editor.flux_gen_comfy_client as mod
        for alias in ("flux_dev_q4", "juggernaut_xl", "longcat_full"):
            for overrides in (False, True):
                with self.subTest(alias=alias, overrides=overrides):
                    client = FluxGenComfyClient()
                    self.assertTrue(client.load(alias)[0])
                    conf = client._get_model_conf()
                    captured = []

                    def capture(url, **kwargs):
                        captured.append(kwargs["json"]["prompt"])
                        raise RuntimeError("workflow capturado sin enviarlo")

                    with patch.object(client, "is_available", return_value=True), \
                            patch.object(client, "_prepare_prompt", side_effect=lambda p: p), \
                            patch.object(mod.requests, "post", side_effect=capture):
                        client.generate("a white house", width=640, height=512, seed=123456,
                                        steps=9 if overrides else None,
                                        guidance_scale=6.0 if overrides else None,
                                        negative_prompt="watermark", batch_size=3, lora_name="None")
                    wf = captured[0]
                    sampler = wf["8"]["inputs"]
                    self.assertEqual(sampler["seed"], 123456)
                    self.assertEqual(sampler["steps"], 9 if overrides else conf["steps"])
                    self.assertEqual(sampler["sampler_name"], conf["sampler"])
                    self.assertEqual(sampler["scheduler"], conf["scheduler"])
                    self.assertEqual(wf["1"]["inputs"], {"width": 640, "height": 512, "batch_size": 3})
                    cfg = 6.0 if overrides else conf["cfg"]
                    if alias == "flux_dev_q4":
                        self.assertEqual(sampler["cfg"], 1.0)
                        self.assertEqual(wf["14"]["inputs"]["guidance"], cfg)
                    else:
                        self.assertEqual(sampler["cfg"], cfg)
                    self.assertIn("watermark", wf["7"]["inputs"]["text"])

    def test_modifiers_survive_final_prompt(self):
        client = FluxGenComfyClient()
        selected = {"image_type": "photoreal", "lighting": "natural",
                    "skin_finish": "detailed", "framing": "full_body", "color_grade": "warm"}
        for alias, sdxl in (("flux_dev_q4", False), ("juggernaut_xl", True)):
            for scene in ("a white house", "a detailed white house beside a full moon in a warm valley",
                          "a white house " + "beside a green tree " * 18):
                with self.subTest(alias=alias, long=len(scene) > 100):
                    text, _, _ = assemble_generation_prompt(alias, scene, selected,
                                                            client._model_configs, is_sdxl=sdxl,
                                                            inject_nsfw=False, adult_intent=False)
                    self.assertIn("full body shot", text)
                    self.assertIn("warm tones", text)
                    self.assertIn("detailed skin pores", text)

    def test_model_switch_failure_does_not_keep_loaded_state(self):
        client = FluxGenComfyClient()
        self.assertTrue(client.load("flux_dev_q4")[0])
        loaded, msg = client.load("unknown_model_for_contract")
        self.assertFalse(loaded, msg)
        self.assertFalse(client._loaded)

    def test_all_selected_modifiers_reach_assembly(self):
        selected = {"image_type": "photoreal", "lighting": "natural",
                    "skin_finish": "detailed", "framing": "full_body", "color_grade": "warm"}
        parts, capped = _resolve_modifier_parts(selected)
        self.assertEqual({category for category, _ in parts}, set(selected))
        self.assertEqual(capped, [])

    def test_no_style_does_not_drop_existing_extra(self):
        selected = {"image_type": "auto", "lighting": "natural",
                    "skin_finish": "detailed", "framing": "wide", "color_grade": "warm"}
        parts, _ = _resolve_modifier_parts(selected)
        self.assertIn("wide angle shot", [value for _, value in parts])
        self.assertIn("warm tones", [value for _, value in parts])

    def test_longcat_keeps_real_cfg(self):
        import roop.img_editor.flux_gen_comfy_client as mod
        client = FluxGenComfyClient()
        client._loaded = True
        client._is_hart = False
        client._is_longcat = True
        client._is_longcat_turbo = False
        client._alias = "longcat_full"
        client._clip_type = "longcat_image"
        captured = []

        def capture(url, **kwargs):
            captured.append(kwargs["json"]["prompt"])
            raise RuntimeError("workflow capturado sin enviarlo")

        with patch.object(client, "is_available", return_value=True), \
                patch.object(client, "_prepare_prompt", side_effect=lambda p: p), \
                patch.object(mod.requests, "post", side_effect=capture):
            client.generate("a white house", guidance_scale=2.5, lora_name="None")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["8"]["inputs"]["cfg"], 2.5)
        self.assertFalse(any(n["class_type"] == "FluxGuidance" for n in captured[0].values()))


if __name__ == "__main__":
    unittest.main()
