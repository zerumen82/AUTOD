"""Keep the current FaceSwap destination when direct-path loading fails."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr

import roop.globals
import ui.globals as ui_globals
from ui.tabs.faceswap import events, logic, state, ui as faceswap_ui


def main():
    with patch.object(events, "_warmup_face_analyser_async"):
        with gr.Blocks(analytics_enabled=False) as demo:
            components = faceswap_ui.build_faceswap_ui()
            events.wire_events(components)

    add_target = next(
        block.fn for block in demo.fns.values()
        if getattr(block.fn, "__name__", "") == "on_add_target_path"
    )
    previous = SimpleNamespace(filename="previous.mp4", endframe=100)
    selected_face = object()
    state.list_files_process = [previous]
    state.selected_preview_index = 0
    roop.globals.TARGET_FACES = [selected_face]
    roop.globals.selected_face_references = {"selected_face_ref_previous.mp4": [{"embedding": [1]}]}
    ui_globals.ui_target_thumbs = ["previous thumb"]

    with patch.object(logic, "process_target_file_async", return_value=(None, None, 0)):
        missing = add_target(str(Path(tempfile.gettempdir()) / "autoauto_missing_target.mp4"))
        assert all(value == gr.update() for value in missing), missing

        with tempfile.NamedTemporaryFile(suffix=".mp4") as unreadable:
            failed = add_target(unreadable.name)
            assert all(value == gr.update() for value in failed), failed

    assert state.list_files_process == [previous]
    assert roop.globals.TARGET_FACES == [selected_face]
    assert ui_globals.ui_target_thumbs == ["previous thumb"]
    assert "selected_face_ref_previous.mp4" in roop.globals.selected_face_references
    print("[PROBE] invalid and unreadable target preserve current destination: OK")


if __name__ == "__main__":
    main()
