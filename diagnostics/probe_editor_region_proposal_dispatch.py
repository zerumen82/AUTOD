"""CPU fallback contracts; fixture masks/LLM do not validate real image quality."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def main():
    image = Image.new("RGB", (200, 100), "gray")
    pixels = np.zeros((100, 200), np.uint8)
    pixels[70:78, 30:60] = 255
    table = Image.fromarray(pixels)
    instruction = "Add a cup on the table"
    manager = ImgEditorManager()
    calls = []

    def segment(_image, query):
        calls.append(query)
        return table if query == "table" else None

    proposals = ["person", "table"]
    with (
        patch.object(manager, "_subject_queries", return_value=[instruction]),
        patch.object(manager, "_get_clipseg_masker", return_value=SimpleNamespace(generate_mask=segment)),
        patch.object(manager, "_region_proposals", return_value=proposals),
        patch.object(manager, "_continue_measured_region", return_value=table),
    ):
        result = manager._measured_region_passes(image, instruction, instruction)
    assert len(result) == 1 and result[0]["text"] == instruction
    assert np.array_equal(np.asarray(result[0]["mask"]), pixels)
    assert result[0]["coverage"] < 0.04 and "table" in calls and "person" not in calls
    print("OK: empty direct query falls back to grounded small support, full instruction retained")

    for measured in (None, Image.new("L", image.size, 0), Image.new("L", image.size, 255)):
        with patch.object(manager, "_region_proposals", return_value=proposals):
            rejected = manager._proposed_region_passes(
                image, instruction, SimpleNamespace(generate_mask=lambda *_: measured),
            )
        assert rejected == []
    print("OK: absent, empty and saturated proposals cannot authorize editing")

    large = Image.new("L", image.size, 0)
    large.paste(255, (10, 10, 100, 80))
    with (
        patch.object(manager, "_subject_queries", return_value=[instruction]),
        patch.object(manager, "_get_clipseg_masker", return_value=SimpleNamespace(generate_mask=lambda *_: large)),
        patch.object(manager, "_proposed_region_passes", side_effect=AssertionError("replaced direct mask")),
        patch.object(manager, "_continue_measured_region", return_value=large),
    ):
        result = manager._measured_region_passes(image, instruction, instruction)
    assert len(result) == 1
    print("OK: direct measured region retains its existing path")

    with (
        patch.object(manager, "_subject_queries", return_value=[instruction]),
        patch.object(manager, "_get_clipseg_masker", return_value=SimpleNamespace(generate_mask=lambda *_: None)),
        patch.object(manager, "_proposed_region_passes", side_effect=AssertionError("bypassed recolor guard")),
    ):
        assert manager._measured_region_passes(image, instruction, instruction, surface_recolor=True) == []
    print("OK: missing recolor never uses structural proposal fallback")

    with patch.object(manager, "_region_proposals", return_value=[]):
        assert manager._proposed_region_passes(image, instruction, SimpleNamespace(generate_mask=segment)) == []
    print("OK: no local language model means abstention")

    with patch.object(manager, "_region_proposals", return_value=["planta", "ventana"]):
        assert manager._proposed_region_passes(
            image, "Quita la planta junto a la ventana", SimpleNamespace(generate_mask=segment),
        ) == []
    print("OK: several grounded proposals require abstention rather than choosing the largest")

    woman = Image.new("L", image.size, 0)
    woman.paste(255, (60, 10, 130, 90))
    face = SimpleNamespace(bbox=(75, 15, 105, 40))
    with patch.object(manager, "_region_proposals", return_value=["woman"]):
        assert manager._proposed_region_passes(
            image, "Add a dog next to the woman",
            SimpleNamespace(generate_mask=lambda *_: woman), [face],
        ) == []
    with (
        patch.object(manager, "_region_proposals", return_value=proposals),
        patch.object(manager, "_continue_measured_region", return_value=woman),
    ):
        assert manager._proposed_region_passes(
            image, instruction, SimpleNamespace(generate_mask=segment), [face],
        ) == []
    print("OK: person anchor and continuation across a measured face cannot authorize insertion")

    unchanged = manager._route_edit_engine("kontext", {
        "body_transform": True, "body_direction": "change", "magnitude": 0.9,
    }, 8.0)
    assert unchanged == (None, None, "")
    changed = manager._route_edit_engine("kontext", {
        "body_transform": True, "body_direction": "change", "magnitude": 0.9,
        "pose_change": True,
    }, 8.0)
    assert changed[0] == "longcat_full", changed
    print("OK: Kontext body router receives measured pose flags without NameError")

    # A configured missing file abstains locally; there is no download call.
    assert manager._region_proposals(instruction, {
        "model_path": "diagnostics/out/absent_region_proposal_fixture.gguf",
    }) == []
    loaded = SimpleNamespace(close=Mock())
    with (
        patch("llama_cpp.Llama", return_value=loaded),
        patch("roop.img_editor.prompt_rewriter.PromptRewriter._classify_edit_support",
              side_effect=RuntimeError("fixture inference error")),
    ):
        try:
            manager._region_proposals(instruction, {
                "model_path": "models/llm/qwen2.5-3b-instruct-q4_k_m.gguf",
                "context_size": 8192, "gpu_layers": 0,
            })
        except RuntimeError as error:
            assert str(error) == "fixture inference error"
        else:
            raise AssertionError("inference exception disappeared")
    loaded.close.assert_called_once()
    print("OK: missing planner abstains; inference error closes the configured model")

    from diagnostics.probe_editor_compound_fast_dispatch import FakeClient
    for proposed, translation in (([], "Remove the plant next to the window"),
                                  (["planta"], "Remove the plant next to the window"),
                                  (["planta"], None)):
        client = FakeClient()
        with (
            patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
            patch("roop.face_util.get_face_analyser", return_value=None),
            patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
            patch.object(manager, "_release_local_gpu_before_comfy"),
            patch.object(manager, "_get_clipseg_masker", return_value=SimpleNamespace(generate_mask=lambda *_: table)),
            patch.object(manager, "_region_proposals", return_value=proposed) as called,
            patch.object(manager, "_ambiguous_region_translation", return_value=translation) as translated,
            patch.object(manager, "_continue_measured_region", return_value=table),
            patch.object(manager, "_repair_inserted_surface", side_effect=lambda _c, _i, _m, result, *_: (result, None)),
            patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Remove the floor by the window"),
            patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        ):
            output, message, _ = manager.generate_intelligent(
                image=image, prompt="Quita la planta junto a la ventana", engine="longcat",
            )
        assert output is not None, message
        assert called.call_args.args[0] == "Quita la planta junto a la ventana"
        if not proposed:
            assert client.calls == [] and "no se sustituye" in message
            assert np.array_equal(np.asarray(output), np.asarray(image))
            translated.assert_not_called()
        elif translation is None:
            assert client.calls == [] and "Traducción local no disponible" in message
            assert np.array_equal(np.asarray(output), np.asarray(image))
        else:
            sent = [kwargs for name, kwargs in client.calls if name == "generate"]
            assert len(sent) == 1
            translated.assert_called_once_with("Quita la planta junto a la ventana")
            assert translation in sent[0]["prompt"]
            assert "floor" not in sent[0]["prompt"]
            assert sent[0]["preserve_frame"] and sent[0]["mask_image"] is not None
            from roop.img_editor.flux_edit_comfy_client import measured_region_sample
            for key, value in measured_region_sample(True).items():
                assert sent[0][key] == value
    print("OK: ambiguous RAW region stays measured; complete local translation alone reaches generation")

    from roop.img_editor.prompt_rewriter import PromptRewriter
    import json
    from pathlib import Path

    config = json.loads((Path(__file__).resolve().parents[1] / "config/model_configs.json").read_text())
    translation_config = config["longcat"]["region_proposal"]["ambiguous_translation"]
    for text, reason in (("", "stop"), ("Partial translation", "length"),
                         ("Remove the book without changing the table", "stop")):
        fake = SimpleNamespace(create_completion=Mock(return_value={
            "choices": [{"text": text, "finish_reason": reason}],
        }))
        actual = PromptRewriter._translate_edit_instruction("Quita el libro sin cambiar la mesa", fake, translation_config)
        assert actual == (text if text and reason == "stop" else None)
    print("OK: empty or truncated translations cannot reach generation")

    fake = SimpleNamespace(close=Mock(), create_completion=Mock(side_effect=RuntimeError("fixture translation failure")))
    with patch.object(manager, "_load_region_model", return_value=fake):
        assert manager._ambiguous_region_translation("Quita la planta junto a la ventana") is None
    fake.close.assert_called_once()
    print("OK: local translation failure closes model and abstains")


if __name__ == "__main__":
    main()
