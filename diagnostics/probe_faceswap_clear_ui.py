"""Exercise both clear buttons against multiple loaded files and faces."""
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import numpy as np

import roop.globals
import ui.globals as ui_globals
from ui.tabs.faceswap import events, state, ui as faceswap_ui


def main():
    with patch.object(events, '_warmup_face_analyser_async'):
        with gr.Blocks(analytics_enabled=False) as demo:
            components = faceswap_ui.build_faceswap_ui()
            events.wire_events(components)
    callbacks = {getattr(block.fn, '__name__', ''): block for block in demo.fns.values()}
    assert 'bt_clear_source' in components and 'bt_clear_target' in components
    assert not any(name in components for name in ('delete_mode', 'delete_scope', 'bt_delete_selection'))
    faces = [object(), object(), object()]
    thumbs = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in faces]
    roop.globals.INPUT_FACESETS = list(faces)
    ui_globals.ui_input_thumbs = list(thumbs)
    state.source_filenames = ['one.jpg', 'two.jpg']
    state.source_load_summary = '3 faces'
    state.current_input_page = 4
    events.selected_input_face_idx = 2
    roop.globals.TARGET_FACES = list(faces)
    ui_globals.ui_target_thumbs = list(thumbs)
    state.list_files_process = [SimpleNamespace(filename='photo.jpg'), SimpleNamespace(filename='video.mp4')]
    roop.globals.selected_face_references = {'photo': [1, 2], 'video': [3]}
    state.CURRENT_DETECTED_KEY = ('video.mp4', 99)
    state.SELECTION_FACES_DATA = list(faces)
    state.ALL_DETECTED_FACES_DATA = list(faces)

    result = callbacks['clear_source'].fn()
    assert len(result) == len(callbacks['clear_source'].outputs)
    assert result[0] is None and result[2] == []
    assert not roop.globals.INPUT_FACESETS and not ui_globals.ui_input_thumbs
    assert not state.source_filenames and not state.source_load_summary
    assert state.current_input_page == 0 and events.selected_input_face_idx is None
    assert len(roop.globals.TARGET_FACES) == 3 and len(state.list_files_process) == 2

    # Destination cleanup must leave a newly loaded source bank intact.
    roop.globals.INPUT_FACESETS = list(faces)
    ui_globals.ui_input_thumbs = list(thumbs)
    result = callbacks['clear_target'].fn()
    assert len(result) == len(callbacks['clear_target'].outputs)
    assert result[0] is None and result[3] is None
    assert result[7] == [] and result[8] == []
    assert not state.list_files_process and not roop.globals.TARGET_FACES
    assert not ui_globals.ui_target_thumbs and not roop.globals.selected_face_references
    assert not state.SELECTION_FACES_DATA and not state.ALL_DETECTED_FACES_DATA
    assert state.CURRENT_DETECTED_KEY is None and roop.globals.target_path is None
    assert len(roop.globals.INPUT_FACESETS) == 3
    # Returning no value preserves the user's random-photo checkbox choice.
    assert 'value' not in callbacks['update_source_random_state'].fn()
    print('[PROBE] clear source and destination: all files/faces/previews reset, opposite bank preserved')


if __name__ == '__main__':
    main()
