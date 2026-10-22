"""Selector UI/session contracts, using real callbacks and Gradio components.

CPU model analysis, backend availability and final inference are substituted.
Controlled threads reproduce photo replacement during an ongoing analysis.
This diagnostic does not certify model adherence or browser rendering.
"""
from __future__ import annotations

import copy
import sys
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from diagnostics.probe_editor_selection_contract_20261019 import fixture  # noqa: E402
from roop.img_editor import element_selection as es  # noqa: E402
from ui.tabs.img_editor import events, selection_events as callbacks, state  # noqa: E402
from ui.tabs.img_editor.state import SelectionSession  # noqa: E402
from ui.tabs.img_editor.ui import build_img_editor_ui  # noqa: E402


def analyze(session=None):
    photo, analysis = fixture()
    session = session if session is not None else SelectionSession()
    with (patch.object(es, "analyze_elements", return_value=analysis),
          patch("roop.img_editor.flux_edit_comfy_client._queue_is_idle", return_value=True)):
        result = callbacks.on_analyze(photo, session)
    assert result[2] is session, "analysis replaced shared revision state"
    return photo, session, result


def session_isolation():
    first = SelectionSession()
    second = copy.deepcopy(first)
    assert first is not second and first.lock is not second.lock
    photo, first, _result = analyze(first)
    callbacks.on_select(["head-right", "small-object"], photo, first)
    assert set(first.snapshot()[1]) == {"head-right", "small-object"}
    assert second.snapshot() == (None, []), "another session inherited selection"
    snapshot = first.snapshot()
    snapshot[1].append("corrupted")
    snapshot[0].regions.clear()
    assert first.snapshot()[0].regions and "corrupted" not in first.snapshot()[1], "snapshot leaked mutation"


def list_and_clean_input():
    photo, session, result = analyze()
    original = np.asarray(photo).copy()
    assert isinstance(result[0], Image.Image), "no preview returned"
    assert np.any(np.asarray(result[0]) != original), "preview did not outline measured regions"
    preview, returned, _message = callbacks.on_select(["head-right", "small-object"], photo, session)
    assert returned is session
    assert preview.size == photo.size and np.any(np.asarray(preview) != original)
    assert np.array_equal(np.asarray(photo), original), "preview painted the actual input"
    callbacks.on_select([], photo, session)
    assert session.snapshot()[1] == [], "list deselection retained old IDs"
    callbacks.on_select(["missing"], photo, session)
    assert not session.snapshot()[1], "invalid ID retained"


def point_toggle_and_ambiguity():
    photo, session, _result = analyze()
    callbacks.on_point_select(photo, session, SimpleNamespace(index=(410, 100)))
    assert session.snapshot()[1] == ["head-right"]
    callbacks.on_point_select(photo, session, SimpleNamespace(index=(280, 240)))
    assert set(session.snapshot()[1]) == {"head-right", "small-object"}
    callbacks.on_point_select(photo, session, SimpleNamespace(index=(410, 100)))
    assert session.snapshot()[1] == ["small-object"], "click did not toggle region off"
    with session.lock:
        duplicate = copy.deepcopy(session.analysis.regions[3])
        duplicate.id = "ambiguous-copy"
        session.analysis.regions.append(duplicate)
    before = session.snapshot()[1]
    result = callbacks.on_point_select(photo, session, SimpleNamespace(index=(280, 240)))
    assert session.snapshot()[1] == before, "ambiguous click selected a region by order"
    message = str(result[-1]).lower()
    assert any(word in message for word in ("ambigu", "lista", "varias", "coincid")), message


def replacement_and_reuse():
    photo, session, _result = analyze()
    callbacks.on_select(["head-right"], photo, session)
    altered = photo.copy()
    altered.putpixel((0, 0), (0, 0, 0))
    callbacks.on_select(["head-right"], altered, session)
    assert not session.snapshot()[1], "stale photo kept selected IDs"
    photo, session, _result = analyze(session)
    callbacks.on_select(["head-right"], photo, session)
    revision = session.revision
    callbacks.on_new_image(photo, session)
    assert session.snapshot() == (None, []) and session.revision > revision
    photo, session, _result = analyze(session)
    callbacks.on_select(["head-right"], photo, session)
    result = callbacks.use_result_as_input(photo, session)
    assert np.array_equal(np.asarray(result[0]), np.asarray(photo))
    assert session.snapshot() == (None, []), "reuse retained old selection"


def numpy_result_reuse_preserves_pixels_and_binding():
    photo, session, _result = analyze()
    callbacks.on_select(["head-right"], photo, session)
    revision = session.revision
    pixels = np.random.default_rng(19).integers(0, 256, (71, 93, 3), dtype=np.uint8)
    original = pixels.copy()
    image, overlay, choices, returned, _message = callbacks.use_result_as_input(pixels, session)
    assert isinstance(image, Image.Image) and image.mode == "RGB"
    assert np.array_equal(np.asarray(image), original), "NumPy result lost pixels during reuse"
    assert np.array_equal(pixels, original), "reuse mutated its NumPy input"
    assert overlay is None and choices["value"] == [] and choices["visible"] is False
    assert returned is session and session.snapshot() == (None, [])
    assert session.revision > revision and session.image_bound
    assert session.current_fingerprint == es.image_fingerprint(Image.fromarray(original)), (
        "reuse bound the wrong source fingerprint"
    )
    # An empty result must preserve the last valid binding and refuse reuse.
    fingerprint, revision = session.current_fingerprint, session.revision
    empty = callbacks.use_result_as_input(None, session)
    assert all(isinstance(item, dict) and item.get("__type__") == "update" for item in empty[:4])
    assert session.current_fingerprint == fingerprint and session.revision == revision


def failed_analysis_clears_previous():
    photo, session, _result = analyze()
    callbacks.on_select(["head-right"], photo, session)
    with (patch.object(es, "analyze_elements", side_effect=RuntimeError("CPU injected analysis failure")),
          patch("roop.img_editor.flux_edit_comfy_client._queue_is_idle", return_value=True)):
        result = callbacks.on_analyze(photo, session)
    assert session.snapshot() == (None, []), "failed analysis kept old proposals"
    assert "error" in str(result[-1]).lower(), result[-1]


def active_jobs_block_analysis():
    photo, _analysis = fixture()
    session = SelectionSession()
    with (patch.object(state, "is_generating", True),
          patch.object(es, "analyze_elements") as inference):
        result = callbacks.on_analyze(photo, session)
        inference.assert_not_called()
        assert state.is_generating, "refused analysis released another job's reservation"
        assert any(word in str(result[-1]).lower() for word in ("proceso", "ocup", "esper"))
    with (patch("roop.img_editor.flux_edit_comfy_client._queue_is_idle", return_value=False),
          patch.object(es, "analyze_elements") as inference):
        result = callbacks.on_analyze(photo, session)
        inference.assert_not_called()
        assert not state.is_generating, "backend-busy refusal left application locked"
        assert any(word in str(result[-1]).lower() for word in ("cola", "ocup"))


def late_analysis_discarded():
    photo, analysis = fixture()
    session = SelectionSession()
    started, release = threading.Event(), threading.Event()

    def slow_analysis(_image):
        started.set()
        assert release.wait(10), "test synchronization timed out"
        return analysis

    with (patch.object(es, "analyze_elements", side_effect=slow_analysis),
          patch("roop.img_editor.flux_edit_comfy_client._queue_is_idle", return_value=True)):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(callbacks.on_analyze, photo, session)
            try:
                assert started.wait(10), "analysis did not start"
                callbacks.on_new_image(photo, session)
            finally:
                release.set()
            result = future.result(timeout=10)
    assert session.snapshot() == (None, []), "late analysis repopulated replaced photo"
    assert not isinstance(result[0], Image.Image), "late analysis published stale preview"


def delayed_old_request(kind):
    photo, analysis = fixture()
    replacement = photo.copy()
    replacement.putpixel((0, 0), (1, 2, 3))
    session = SelectionSession()
    if kind == "reuse":
        callbacks.use_result_as_input(replacement, session)
    else:
        callbacks.on_new_image(None if kind == "clear" else replacement, session)
    for current in (session, copy.deepcopy(session)):
        with (patch.object(es, "analyze_elements", return_value=analysis) as inference,
              patch("roop.img_editor.flux_edit_comfy_client._queue_is_idle", return_value=True)):
            result = callbacks.on_analyze(photo, current)
            inference.assert_not_called()
        assert current.snapshot() == (None, []), "old queued request repopulated current image"
        assert not isinstance(result[0], Image.Image), "old queued request published stale preview"
        assert not state.is_generating, "rejected delayed request leaked reservation"


def delayed_request_after_replacement():
    delayed_old_request("replacement")


def delayed_request_after_clear():
    delayed_old_request("clear")


def delayed_request_after_reuse():
    delayed_old_request("reuse")


def concurrent_analysis_is_rejected():
    photo, old_analysis = fixture()
    newer = photo.copy()
    newer.putpixel((0, 0), (1, 2, 3))
    new_analysis = copy.deepcopy(old_analysis)
    new_analysis.fingerprint = es.image_fingerprint(newer)
    session = SelectionSession()
    started, release = threading.Event(), threading.Event()

    def measured(image):
        if es.image_fingerprint(image) == old_analysis.fingerprint:
            started.set()
            assert release.wait(10), "test synchronization timed out"
            return old_analysis
        return new_analysis

    with (patch.object(es, "analyze_elements", side_effect=measured) as inference,
          patch("roop.img_editor.flux_edit_comfy_client._queue_is_idle", return_value=True)):
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(callbacks.on_analyze, photo, session)
            try:
                assert started.wait(10)
                rejected = callbacks.on_analyze(newer, session)
                assert not isinstance(rejected[0], Image.Image), "concurrent analysis was published"
                assert any(word in str(rejected[-1]).lower()
                           for word in ("proceso", "ocup", "esper", "foto", "imagen", "caduc")), rejected[-1]
            finally:
                release.set()
            result = future.result(timeout=10)
    assert inference.call_count == 1, "busy analysis loaded models a second time"
    assert session.snapshot()[0].fingerprint == old_analysis.fingerprint
    assert isinstance(result[0], Image.Image), "first valid analysis was lost"


def desktop_forwards_clean_selection():
    photo, session, _result = analyze()
    callbacks.on_select(["head-right", "small-object"], photo, session)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return kwargs["image"], "CPU inference", None

    manager = SimpleNamespace(generate_intelligent=generate,
                              _last_instruction_compliance=None,
                              _release_local_gpu_before_comfy=lambda: None)
    with tempfile.TemporaryDirectory(prefix="editor_selector_") as directory:
        with (
            patch.object(events, "get_img_editor_manager", return_value=manager),
            patch.object(events, "get_img_editor_output_dir", return_value=directory),
            patch.object(events, "_cleanup_vram"),
        ):
            output, message, _mask = events._on_generate_desktop(
                photo, "Make selected elements silver", "kontext", 0,
                False, 1, False, "hd", True, True, True, True, True, session,
            )
            callbacks.on_select([], photo, session)
            automatic, automatic_message, _mask = events._on_generate_desktop(
                photo, "Make the wall blue", "kontext", 0,
                False, 1, False, "hd", True, True, True, True, True, session,
            )
    assert output is not None and "guardado" in message.lower(), message
    assert automatic is not None and "guardado" in automatic_message.lower(), automatic_message
    assert len(calls) == 2
    assert np.array_equal(np.asarray(calls[0]["image"]), np.asarray(photo)), "overlay reached inference"
    assert calls[0]["selection_analysis"].fingerprint == es.image_fingerprint(photo)
    assert set(calls[0]["selected_region_ids"]) == {"head-right", "small-object"}
    assert calls[0]["mask_image"] is None, "manual mask reintroduced"
    assert calls[1].get("selection_analysis") is None and not calls[1].get("selected_region_ids"), (
        "analysis without selected IDs disabled automatic editing"
    )


def ui_endpoint_contract():
    with gr.Blocks() as app:
        ui = build_img_editor_ui()
        events.wire_events(ui)
    config = app.get_config_file()
    endpoints = {item["api_name"]: item for item in config["dependencies"]}
    ordered = [ui[key]._id for key in (
        "input_img", "prompt", "engine", "denoise", "autoregressive", "auto_iterations",
        "quality_mode", "enhance_tier", "quality_preserve_faces", "quality_deposter",
        "quality_upscale", "quality_color", "quality_realism",
    )]
    for name, count in (("edit", 8), ("edit_options", 9), ("edit_select", 13), ("edit_region", 14)):
        assert len(endpoints[name]["inputs"]) == count, (name, endpoints[name])
        expected = ordered[:count] if count <= 13 else ordered + [ui["selection_state"]._id]
        assert endpoints[name]["inputs"] == expected, f"{name}: argument order changed"
    assert isinstance(ui["input_img"], gr.Image)
    assert ui["output_img"].type == "pil" and ui["output_img"].format == "png", (
        "output must round-trip through reuse as a lossless PIL image"
    )
    pixels = np.random.default_rng(191).integers(0, 256, (37, 49, 3), dtype=np.uint8)
    transferred = ui["output_img"].preprocess(ui["output_img"].postprocess(Image.fromarray(pixels)))
    assert isinstance(transferred, Image.Image) and np.array_equal(np.asarray(transferred), pixels), (
        "actual Gradio result preprocessing lost pixels or returned a NumPy array"
    )
    assert isinstance(ui["selection_state"].value, SelectionSession)
    assert not any(item["type"] == "imageeditor" for item in config["components"])
    overlay_id = ui["overlay_img"]._id
    assert any([overlay_id, "select"] in item["targets"] or (overlay_id, "select") in item["targets"]
               for item in config["dependencies"]), "point selection not wired"
    bound = [(item, app.fns[item["id"]].fn) for item in config["dependencies"]]
    replacements = [item for item, fn in bound if fn is callbacks.on_new_image]
    assert replacements and all(item["queue"] is False for item in replacements), "image invalidation queued behind analysis"
    reuse = [item for item, fn in bound if fn is callbacks.use_result_as_input]
    assert reuse and reuse[0]["inputs"] == [ui["output_img"]._id, ui["selection_state"]._id], "reuse lost session input"


def main():
    cases = [session_isolation, list_and_clean_input, point_toggle_and_ambiguity,
             replacement_and_reuse, numpy_result_reuse_preserves_pixels_and_binding,
             failed_analysis_clears_previous, active_jobs_block_analysis,
             late_analysis_discarded, delayed_request_after_replacement,
             delayed_request_after_clear, delayed_request_after_reuse,
             concurrent_analysis_is_rejected,
             desktop_forwards_clean_selection, ui_endpoint_contract]
    failures = []
    for check in cases:
        try:
            check()
            print(f"PASS {check.__name__}", flush=True)
        except Exception as error:
            failures.append(check.__name__)
            print(f"FAIL {check.__name__}: {error}", flush=True)
            traceback.print_exc()
    print(f"H-EDIT-SELECT UI: {len(cases) - len(failures)}/{len(cases)} PASS; failures={failures}", flush=True)
    return bool(failures)


if __name__ == "__main__":
    raise SystemExit(main())
