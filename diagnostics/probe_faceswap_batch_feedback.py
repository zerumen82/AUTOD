"""Check that a missing video target or an unchanged photo is visible in the UI."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import roop.globals
from ui.tabs.faceswap import logic, state


def run_start():
    return list(logic.start_swap(
        enhancer="None", keep_frames=False, wait_after_extraction=False,
        skip_audio=True, face_distance=0.2, blend_ratio=1.0,
        blend_mode="blend", selected_mask_engine="None",
        processing_method="Inswapper 128", no_face_action="skip",
        vr_mode=False, use_single_source_all=False, autorotate=True,
        temporal_smoothing=False, num_swap_steps=1,
    ))


with tempfile.TemporaryDirectory(prefix="faceswap_feedback_") as temp:
    video = Path(temp) / "missing.mp4"
    photo = Path(temp) / "unchanged.jpg"
    video.touch()
    photo.touch()
    with patch.object(state, 'list_files_process', [SimpleNamespace(filename=str(video))]), \
         patch.object(roop.globals, 'INPUT_FACESETS', [object()], create=True), \
         patch.object(roop.globals, 'selected_face_references', {}, create=True), \
         patch.object(roop.globals, 'face_swap_mode', 'selected_faces', create=True), \
         patch.object(roop.globals.CFG, 'clear_output', False), \
         patch.object(logic, 'cleanup_temp_files'), \
         patch('roop.core.batch_process_regular', side_effect=AssertionError('render started')):
        outputs = run_start()
        assert 'missing.mp4' in outputs[-1][2]

    with patch.object(state, 'list_files_process', [SimpleNamespace(filename=str(photo))]), \
         patch.object(roop.globals, 'INPUT_FACESETS', [object()], create=True), \
         patch.object(roop.globals, 'TARGET_FACES', [object()], create=True), \
         patch.object(roop.globals, 'selected_face_references', {}, create=True), \
         patch.object(roop.globals, 'face_swap_mode', 'selected_faces', create=True), \
         patch.object(roop.globals.CFG, 'clear_output', False), \
         patch.object(logic, 'cleanup_temp_files'), \
         patch('ui.main.prepare_environment'), \
         patch('roop.core.batch_process_regular', return_value=iter([
             (100, 'Procesados 1 archivos; 1 foto(s) sin swap: unchanged.jpg'),
         ])), \
         patch.object(logic.gr, 'Warning') as warning, \
         patch.object(logic.gr, 'Info') as info:
        outputs = run_start()
        assert '1 foto(s) sin swap: unchanged.jpg' in outputs[-1][2]
        warning.assert_called_once()
        info.assert_not_called()

print('FaceSwap batch feedback: missing video blocked; unchanged photo shown as warning')
