"""CPU fallback contracts; fixture masks/LLM do not validate real image quality."""
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def main():
    from roop.img_editor.prompt_rewriter import PromptRewriter

    planner = SimpleNamespace(create_completion=Mock(return_value={
        "choices": [{"text": "1. table\n- shelf\nTABLE"}],
    }))
    assert PromptRewriter._classify_edit_support(
        "Add a cup on the table and a book on the shelf", planner,
    ) == ["table", "shelf"]
    planner.create_completion.assert_called_once()
    print("OK: local support planner returns every distinct measured anchor")

    image = Image.new("RGB", (200, 100), "gray")
    pixels = np.zeros((100, 200), np.uint8)
    pixels[70:78, 30:60] = 255
    table = Image.fromarray(pixels)
    instruction = "Add a cup on the table"
    manager = ImgEditorManager()
    add_request = "Add a small blue sugar packet on the table beside the cup."
    add_facts = manager._prompt_authority_facts(add_request, add_request)
    assert add_facts["structural_add"] and add_facts["region_ambiguous"], add_facts
    assert not add_facts["pose"] and not add_facts["structural_remove"], add_facts
    blouse_facts = manager._prompt_authority_facts(
        "Change the person's blouse to red", "Change the person's blouse to red",
    )
    assert not blouse_facts["structural_add"], blouse_facts
    person_add_facts = manager._prompt_authority_facts(
        "Add a person near the door", "Add a person near the door",
    )
    assert person_add_facts["structural_add"] and not person_add_facts["pose"], (
        person_add_facts)
    print("OK: generic additions route to measured support without stealing clothing or pose")
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
    direct_instruction = "Change the shirt to red"
    with (
        patch.object(manager, "_subject_queries", return_value=[direct_instruction]),
        patch.object(manager, "_get_clipseg_masker", return_value=SimpleNamespace(generate_mask=lambda *_: large)),
        patch.object(manager, "_proposed_region_passes", side_effect=AssertionError("replaced direct mask")),
        patch.object(manager, "_continue_measured_region", return_value=large),
    ):
        result = manager._measured_region_passes(image, direct_instruction, direct_instruction)
    assert len(result) == 1
    print("OK: direct measured region retains its existing path")

    target_removal = "Remove the spoon beside the cup."
    removal_queries = []

    def removal_segment(_image, query):
        removal_queries.append(query)
        return table if query == "spoon" else large

    with (
        patch.object(manager, "_subject_queries", return_value=[]),
        patch.object(manager, "_visual_request_queries", return_value=[target_removal]),
        patch.object(manager, "_get_clipseg_masker",
                     return_value=SimpleNamespace(generate_mask=removal_segment)),
        patch.object(manager, "_continue_measured_region", return_value=table),
    ):
        result = manager._measured_region_passes(
            image, target_removal, target_removal, structural_remove=True,
        )
    assert removal_queries == ["spoon"], removal_queries
    assert len(result) == 1 and result[0]["coverage"] < 0.04
    assert np.array_equal(np.asarray(result[0]["mask"]), pixels)
    print("OK: object removal prefers its measured target over a broad sentence mask")

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

    wall_pixels = np.zeros_like(pixels)
    wall_pixels[15:35, 130:155] = 255
    wall = Image.fromarray(wall_pixels)
    wall_queries = []

    def wall_segment(_image, query):
        wall_queries.append(query)
        return wall if query == "wall" else None

    with (
        patch.object(manager, "_get_clipseg_masker",
                     return_value=SimpleNamespace(generate_mask=wall_segment)),
        patch.object(manager, "_region_proposals", return_value=["wall-support"]),
        patch.object(manager, "_continue_measured_region", return_value=wall),
    ):
        wall_mask = manager._measure_camera_followup_mask(image, "make the wall blue")
    assert wall_mask is not None
    assert wall_queries == ["wall"]
    assert np.array_equal(np.asarray(wall_mask), wall_pixels)
    print("OK: generated support labels are grounded to requested words before measurement")

    with patch.object(manager, "_region_proposals", return_value=["planta", "ventana"]):
        assert manager._proposed_region_passes(
            image, "Quita la planta junto a la ventana", SimpleNamespace(generate_mask=segment),
        ) == []
    print("OK: several grounded proposals require abstention rather than choosing the largest")

    multi_prompt = "Add a cup on the table and a book on the shelf"
    assert ImgEditorManager._split_compound_instruction(multi_prompt) == []
    assert ImgEditorManager._split_compound_instruction(
        "Add a cup on the table, and a book on the shelf",
    ) == []
    assert len(ImgEditorManager._split_compound_instruction(
        "The walls are blue, Add a person near the door",
    )) == 2
    shelf_pixels = np.zeros_like(pixels)
    shelf_pixels[15:30, 140:180] = 255
    shelf = Image.fromarray(shelf_pixels)

    def multi_segment(_image, query):
        return {"table": table, "shelf": shelf}.get(query)

    with (
        patch.object(manager, "_subject_queries", return_value=[]),
        patch.object(manager, "_visual_request_queries", return_value=[]),
        patch.object(manager, "_get_clipseg_masker",
                     return_value=SimpleNamespace(generate_mask=multi_segment)),
        patch.object(manager, "_region_proposals", return_value=["table", "shelf"]),
        patch.object(manager, "_continue_measured_region", side_effect=lambda measured, *_: measured),
    ):
        result = manager._measured_region_passes(image, multi_prompt, multi_prompt)
    assert len(result) == 1 and result[0]["text"] == multi_prompt
    expected_union = np.maximum(pixels, shelf_pixels)
    assert np.array_equal(np.asarray(result[0]["mask"]), expected_union)
    print("OK: coordinated additions remain one instruction with the union of measured anchors")

    door_pixels = np.zeros_like(pixels)
    door_pixels[20:80, 145:185] = 255
    door = Image.fromarray(door_pixels)
    person_prompt = "Add a person near the door"
    with (
        patch.object(manager, "_subject_queries", return_value=[]),
        patch.object(manager, "_visual_request_queries", return_value=[]),
        patch.object(manager, "_get_clipseg_masker",
                     return_value=SimpleNamespace(
                         generate_mask=lambda _image, query: door if query == "door" else None,
                     )),
        patch.object(manager, "_region_proposals", return_value=["door"]),
        patch.object(manager, "_continue_measured_region", side_effect=lambda measured, *_: measured),
    ):
        result = manager._measured_region_passes(
            image, person_prompt, person_prompt,
            faces=[SimpleNamespace(bbox=(20, 10, 40, 35))],
        )
    assert len(result) == 1 and result[0]["text"] == person_prompt
    assert np.array_equal(np.asarray(result[0]["mask"]), door_pixels)
    print("OK: person addition is authorized only by a measured spatial anchor")

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

    from diagnostics.probe_editor_compound_fast_dispatch import FakeClient

    client = FakeClient()
    segment_calls = []

    def addition_segment(_image, query):
        segment_calls.append(query)
        return table if query == "table" else None

    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch.object(
            manager, "_get_clipseg_masker",
            return_value=SimpleNamespace(generate_mask=addition_segment),
        ),
        patch.object(manager, "_region_proposals", return_value=["table"]),
        patch.object(manager, "_continue_measured_region", return_value=table),
        patch.object(
            manager, "_repair_inserted_surface",
            side_effect=lambda _c, _i, _m, result, *_a: (result, None),
        ),
        patch("roop.img_editor.prompt_translator.translate_prompt",
              return_value="Add a cup on the table"),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=client),
    ):
        output, message, _ = manager.generate_intelligent(
            image=image, prompt="Add a cup on the table", engine="longcat",
        )
    generation = [kwargs for name, kwargs in client.calls if name == "generate"]
    assert output is not None and generation, message
    assert "table" in segment_calls
    assert generation[0]["mask_image"] is not None
    assert generation[0]["preserve_frame"]
    assert np.asarray(output)[0, 0].tolist() == np.asarray(image)[0, 0].tolist()
    print("OK: an anchored object addition reaches one masked, frame-preserving edit")

    client = FakeClient()
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch.object(manager, "_measured_region_passes", return_value=[]) as measured,
        patch.object(manager, "_proposed_region_passes", return_value=[]) as proposed,
        patch("roop.img_editor.prompt_translator.translate_prompt", return_value=add_request),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=client),
    ):
        output, message, _ = manager.generate_intelligent(
            image=image, prompt=add_request, engine="longcat",
        )
    assert output is not None and np.array_equal(np.asarray(output), np.asarray(image)), message
    measured.assert_called_once()
    proposed.assert_called_once()
    assert not any(name == "generate" for name, _ in client.calls), client.calls
    assert "no se sustituye" in message, message
    print("OK: an addition with no measured location preserves the source and never runs global generation")

    unchanged = manager._route_edit_engine("kontext", {
        "body_transform": True, "body_direction": "change", "magnitude": 0.9,
    }, 8.0)
    assert unchanged == (None, None, "")
    changed = manager._route_edit_engine("kontext", {
        "body_transform": True, "body_direction": "change", "magnitude": 0.9,
        "pose_change": True,
    }, 8.0)
    assert changed == (None, None, ""), changed
    print("OK: measured pose flags keep the current Kontext weights unchanged")

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
            patch.object(manager, "_measured_region_passes", return_value=[]),
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
            assert "Remove the plant completely" in sent[0]["prompt"]
            assert "nothing of it may remain" in sent[0]["prompt"]
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

    client = FakeClient()
    removal = "Remove the spoon beside the cup."
    with (
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=None),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch.object(
            manager, "_measured_region_passes",
            return_value=[{"text": removal, "mask": table, "coverage": 0.02}],
        ),
        patch.object(manager, "_closed_structural_zone", return_value=table),
        patch.object(manager, "_ambiguous_region_translation", return_value=removal),
        patch.object(
            manager, "_repair_inserted_surface",
            side_effect=lambda _c, _i, _m, result, *_a: (result, None),
        ),
        patch("roop.img_editor.prompt_translator.translate_prompt", return_value=removal),
        patch(
            "roop.img_editor.icedit_comfy_client.ICEditComfyClient._resolve_flux_fill",
            return_value="missing-flux-fill.gguf",
        ) as resolve_fill,
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=client),
    ):
        output, message, _ = manager.generate_intelligent(
            image=image, prompt=removal, engine="longcat",
        )
    generation = [kwargs for name, kwargs in client.calls if name == "generate"]
    assert output is not None and generation, message
    dispatched_prompt = generation[0]["prompt"]
    assert "Remove the spoon completely" in dispatched_prompt
    assert "nothing of it may remain" in dispatched_prompt
    assert "Naturally reconstruct the area it occupied" in dispatched_prompt
    assert generation[0]["preserve_frame"] and generation[0]["mask_image"] is not None
    assert any(
        name == "load" and "LongCat-Image-Edit-Turbo-Q4_K_S.gguf" in str(version)
        for name, version in client.calls
    ), client.calls
    resolve_fill.assert_not_called()
    print("OK: measured removal uses LongCat Turbo, not the unavailable Fill fallback")


if __name__ == "__main__":
    main()
