"""Construcción y callbacks reales de Generate, HTTP local TestClient sin GPU."""
import unittest
from unittest.mock import patch

import gradio as gr
from fastapi.testclient import TestClient
from gradio.routes import App

from ui.tabs.gen.ui import build_gen_ui
from ui.tabs.gen.events import on_generate_image, wire_events


class GenUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with gr.Blocks() as cls.demo:
            cls.ui = build_gen_ui()
            wire_events(cls.ui)
        cls.callbacks = {fn.fn.__name__: fn.fn for fn in cls.demo.fns.values() if fn.fn}

    def test_http_config_and_legacy_contract(self):
        with TestClient(App.create_app(self.demo)) as client:
            response = client.get("/config")
        self.assertEqual(response.status_code, 200)
        deps = response.json()["dependencies"]
        legacy = next(d for d in deps if d.get("api_name") == "generate_image")
        self.assertEqual(len(legacy["inputs"]), 10)
        self.assertTrue(self.ui["modifier_preview"].value)
        self.assertIn("CFG=1", self.ui["negative_prompt"].info)
        self.assertIn("no fija", self.ui["enhance_tier"].info)

    def test_engine_switch_updates_available_loras(self):
        fn = self.callbacks["_on_engine_change"]
        pony = fn("pony_realism", "Amateur_Flash_v2.safetensors")
        self.assertEqual(pony["value"], "Amateur_Flash_v2.safetensors")
        flux = fn("flux_dev_q4", "Amateur_Flash_v2.safetensors")
        self.assertEqual(flux["value"], "Auto")
        self.assertEqual({value for _, value in flux["choices"]}, {"Auto", "None"})

    def test_incompatible_style_options_are_sanitized(self):
        updates = self.callbacks["_on_style_change"]("smartphone", "studio", "wide", "smooth", "vibrant")
        self.assertEqual([u["value"] for u in updates[:4]], ["auto", "wide", "auto", "auto"])

    def test_guidance_description_changes_with_engine(self):
        fn = self.callbacks["_on_guidance_change"]
        self.assertIn("CFG=1", fn("flux_dev_q4", 0)[1]["info"])
        # El alias retirado LongCat Full ya carga Kontext (CFG distilled=1).
        self.assertIn("no influye", fn("longcat_full", 0)[1]["info"])

    def test_editor_route_rejects_manual_lora_without_render_or_unload(self):
        from PIL import Image
        with patch("ui.tabs.gen.events.get_flux_gen_client") as client, \
                patch("ui.tabs.gen.events._post_generation_unload") as unload:
            client.return_value.is_available.return_value = True
            rows = list(on_generate_image("a white house", 512, 512, "pony_realism",
                                           "auto", "auto", "auto", "auto", False, False,
                                           ref_image=Image.new("RGB", (64, 64)),
                                           route="editor", lora_choice="Amateur_Flash_v2.safetensors"))
        self.assertIn("no aplica", rows[-1][1])
        client.return_value.generate_ai.assert_not_called()
        unload.assert_not_called()


if __name__ == "__main__":
    unittest.main()
