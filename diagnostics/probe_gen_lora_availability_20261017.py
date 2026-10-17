"""Contratos CPU de LoRAs: archivos reales pequeños, familias y preflight."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from safetensors.numpy import save_file

from roop.img_editor import gen_lora_resolver as resolver
from roop.img_editor.flux_gen_comfy_client import FluxGenComfyClient
from ui.tabs.gen.ui import _lora_dropdown_choices


class LoraAvailability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.tmp.name)
        save_file({"lora_unet_test.lora_down.weight": np.zeros((2, 2048), np.float32)},
                  self.root / "style.safetensors",
                  metadata={"ss_base_model_version": "sdxl_base_v1-0"})
        save_file({"lora_unet_test.lora_down.weight": np.zeros((2, 768), np.float32)},
                  self.root / "sd1.safetensors",
                  metadata={"ss_base_model_version": "sd_v1"})
        (self.root / "broken.safetensors").write_bytes(b"not a safetensors file")
        self.catalog = [{"file": "style.safetensors", "label": "Style SDXL",
                         "strength": 0.35, "manual_only": True}]
        self.stack.enter_context(patch.object(resolver, "_loras_dir", return_value=str(self.root)))
        self.stack.enter_context(patch.object(resolver, "get_lora_catalog", return_value=self.catalog))
        self.stack.enter_context(patch("roop.img_editor.gen_prompt_config.get_lora_catalog",
                                return_value=self.catalog))
        self.client = FluxGenComfyClient()
        self.client._loaded = True
        self.client._is_sdxl = True
        self.client._is_hart = False
        self.client._alias = "juggernaut_xl"

    def resolve(self, name):
        return self.client._resolve_lora_stack(name, None, {}, prompt_en="a white house")

    def test_missing_rejected(self):
        with self.assertRaisesRegex(ValueError, "no.*instalad"):
            self.resolve("missing.safetensors")

    def test_sdxl_on_flux_rejected(self):
        self.client._is_sdxl = False
        self.client._alias = "flux_dev_q4"
        with self.assertRaisesRegex(ValueError, "SDXL"):
            self.resolve("style.safetensors")

    def test_invalid_files_rejected(self):
        for name in ("../style.safetensors", "broken.safetensors", "sd1.safetensors",
                     str(self.root / "style.safetensors")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.resolve(name)

    def test_manual_valid_uses_catalog_strength(self):
        self.assertEqual(self.resolve("style.safetensors")[0], [("style.safetensors", 0.35)])

    def test_manual_only_never_auto_added(self):
        self.catalog[0]["always"] = True
        picks, _ = resolver.resolve_scene_loras("a white house", "juggernaut_xl")
        self.assertEqual(picks, [])

    def test_pony_selection_rejected_on_standard_sdxl(self):
        self.catalog[0]["pony_only"] = True
        with self.assertRaisesRegex(ValueError, "Pony"):
            self.resolve("style.safetensors")

    def test_filename_case_cannot_bypass_catalog_family(self):
        self.catalog[0]["pony_only"] = True
        with self.assertRaisesRegex(ValueError, "Pony"):
            self.resolve("STYLE.SAFETENSORS")

    def test_dropdown_filters_by_file_and_engine(self):
        self.catalog.append({"file": "missing.safetensors", "label": "Missing"})
        sdxl = {v for _, v in _lora_dropdown_choices("juggernaut_xl")}
        flux = {v for _, v in _lora_dropdown_choices("flux_dev_q4")}
        self.assertEqual(sdxl, {"Auto", "None", "style.safetensors"})
        self.assertEqual(flux, {"Auto", "None"})

    def test_invalid_manual_fails_before_upload(self):
        with patch.object(self.client, "is_available", return_value=True), \
                patch("roop.img_editor.flux_gen_comfy_client.requests.post") as post:
            from PIL import Image
            res, msg = self.client.generate("a white house", lora_name="missing.safetensors",
                                            init_image=Image.new("RGB", (64, 64)))
        self.assertIsNone(res)
        self.assertIn("no", msg.lower())
        post.assert_not_called()

    def test_auto_incompatible_engine_explained(self):
        self.client._is_sdxl = False
        self.client._alias = "flux_dev_q4"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(self.resolve("Auto")[0], [])
        self.assertIn("SDXL", output.getvalue())


class InstalledLoras(unittest.TestCase):
    def test_actual_local_adapters_and_loader_nodes(self):
        for alias, filename, loader in (
                ("pony_realism", "Amateur_Flash_v2.safetensors", "LoraLoader"),
                ("juggernaut_xl", "sdsFILM_v40-sdxl-V1.safetensors", "LoraLoaderModelOnly")):
            with self.subTest(alias=alias):
                client = FluxGenComfyClient()
                loaded, msg = client.load(alias)
                self.assertTrue(loaded, msg)
                stack, _, log = client._resolve_lora_stack(filename, None, client._get_model_conf())
                self.assertEqual(stack[0][0], filename)
                self.assertIn(filename, log)
                wf = {}
                model, clip = client._chain_lora_loaders(wf, ["2", 0], ["3", 0], stack)
                self.assertEqual(wf["20"]["class_type"], loader)
                self.assertEqual(model, ["20", 0])
                self.assertEqual(clip, ["20", 1] if loader == "LoraLoader" else ["3", 0])


if __name__ == "__main__":
    unittest.main()
