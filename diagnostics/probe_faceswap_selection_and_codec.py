"""Check destination persistence and lossless-to-H.264 video delivery."""
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import roop.globals
import ui.globals
from ui.tabs.faceswap.events import _get_file_bank, _purge_bank_by_face, _register_destination_face
from roop.destination_selection import missing_video_references, video_reference_bank
from roop.ProcessMgr import ProcessMgr
from roop.core import batch_process_regular
from roop.util_ffmpeg import encode_face_swap_video


def face(box, embedding):
    x1, y1, x2, y2 = box
    return SimpleNamespace(
        bbox=np.asarray(box, dtype=np.float32),
        kps=np.asarray([
            [x1 + 8, y1 + 10], [x2 - 8, y1 + 10],
            [(x1 + x2) / 2, (y1 + y2) / 2],
            [x1 + 12, y2 - 8], [x2 - 12, y2 - 8],
        ], dtype=np.float32),
        embedding=np.asarray(embedding, dtype=np.float32),
    )


def main():
    old_refs = getattr(roop.globals, 'selected_face_references', None)
    old_targets = getattr(roop.globals, 'TARGET_FACES', None)
    old_thumbs = getattr(ui.globals, 'ui_target_thumbs', None)
    roop.globals.selected_face_references = {}
    roop.globals.TARGET_FACES = []
    ui.globals.ui_target_thumbs = []
    try:
        one = SimpleNamespace(filename=r'D:\set_a\same.jpg')
        two = SimpleNamespace(filename=r'D:\set_b\same.jpg')
        a = face([10, 10, 60, 60], [1, 0, 0])
        b = face([90, 10, 140, 60], [0, 1, 0])
        duplicate_identity_other_file = face([10, 10, 60, 60], [1, 0, 0])
        with patch('ui.tabs.faceswap.events.util.convert_to_gradio', return_value='thumb'):
            assert _register_destination_face(one, a, None)[1:] == (1, 'ok')
            assert _register_destination_face(one, b, None)[1:] == (2, 'ok')
            assert _register_destination_face(one, a, None)[1:] == (2, 'dup')
            assert _register_destination_face(two, duplicate_identity_other_file, None)[1:] == (1, 'ok')
        assert len(roop.globals.TARGET_FACES) == 3
        assert len(_get_file_bank('same.jpg', one.filename)) == 2
        assert len(_get_file_bank('same.jpg', two.filename)) == 1
        _purge_bank_by_face(a)
        assert len(_get_file_bank('same.jpg', one.filename)) == 1
        assert len(_get_file_bank('same.jpg', two.filename)) == 1
        # Restore it for the process-manager selection probe.
        with patch('ui.tabs.faceswap.events.util.convert_to_gradio', return_value='thumb'):
            assert _register_destination_face(one, a, None)[2] == 'ok'
        manager = ProcessMgr.__new__(ProcessMgr)
        bank = _get_file_bank('same.jpg')
        selected_a = manager._selected_photo_faces(
            one.filename, bank, [a, b], (100, 150, 3)
        )
        selected_b = manager._selected_photo_faces(
            two.filename, bank, [duplicate_identity_other_file], (100, 150, 3)
        )
        assert len(selected_a) == 2 and len(selected_b) == 1
        print('destination selections: 2/2 first photo, 1/1 second photo')

        video_one = r'D:\set_a\clip.mp4'
        video_two = r'D:\set_b\clip.mp4'
        manager.options = SimpleNamespace(current_video_path=video_one)
        roop.globals.selected_face_references['selected_face_ref_clip.mp4'] = [
            {'embedding': np.asarray([1, 0, 0], dtype=np.float32),
             'source_path': video_one},
            {'embedding': np.asarray([0, 1, 0], dtype=np.float32),
             'source_path': video_one},
            {'embedding': np.asarray([0, 0, 1], dtype=np.float32),
             'source_path': video_two},
        ]
        assert len(manager._video_reference_bank('clip.mp4', video_one)) == 2
        assert len(manager._video_reference_bank('clip.mp4', video_two)) == 1
        assert len(video_reference_bank(roop.globals.selected_face_references, video_one)) == 2
        assert len(video_reference_bank(roop.globals.selected_face_references, video_two)) == 1
        missing_path = r'D:\set_c\last.mp4'
        eight_videos = [video_one] * 7 + [missing_path]
        assert missing_video_references(eight_videos, roop.globals.selected_face_references) == [missing_path]
        # A same-named file elsewhere must not borrow the first video's selection.
        assert missing_video_references([r'D:\set_c\clip.mp4'],
                                        roop.globals.selected_face_references) == [r'D:\set_c\clip.mp4']
        with tempfile.TemporaryDirectory(prefix='faceswap_missing_target_') as temp:
            real_missing_path = str(Path(temp) / 'last.mp4')
            Path(real_missing_path).touch()
            with patch.object(roop.globals, 'face_swap_mode', 'selected_faces'), \
                 patch.object(roop.globals, 'INPUT_FACESETS', [], create=True), \
                 patch('roop.ProcessMgr.ProcessMgr.__init__', side_effect=AssertionError('render started')):
                status = list(batch_process_regular(
                    [SimpleNamespace(filename=real_missing_path)], None, '', True,
                    {'layers': []}, 1, None, 0, False,
                ))
            assert len(status) == 1 and status[0][0] == 0 and 'last.mp4' in status[0][1]
        print('video batch preflight: missing eighth target and same-name path detected')
        sources = [face([10, 10, 60, 60], [0.9, 0.1, 0]),
                   face([10, 10, 60, 60], [0, 1, 0])]
        picked = manager._select_source_face(
            face([10, 10, 60, 60], [1, 0, 0]), sources,
            'selected_faces', video_one,
        )
        assert picked is sources[1], 'second destination view must participate in source selection'
        print('video source: best match among selected destination views')
        selected_track = face([10, 10, 60, 60], [0, 1, 0])
        unrelated_track = face([80, 10, 130, 60], [0, 0, 1])
        roop.globals.selected_face_references['selected_face_ref_clip.mp4'][0]['face_obj'] = selected_track
        assert manager._selected_target_is_unique(
            selected_track, [selected_track, unrelated_track], 'clip.mp4'
        )
        ambiguous_track = face([80, 10, 130, 60], [0.02, 1, 0])
        assert not manager._selected_target_is_unique(
            selected_track, [selected_track, ambiguous_track], 'clip.mp4'
        )
        print('video tracking: selected identity rescues a unique low-score face, rejects ambiguity')
        manager._setup_selected_faces_frame_for_video(
            video_one, [unrelated_track, face([140, 10, 190, 60], [0, 0, -1])],
        )
        assert not hasattr(manager, '_reference_face_selected_clip.mp4')
        manager._setup_selected_faces_frame_for_video(video_one, [selected_track, unrelated_track])
        assert getattr(manager, '_reference_face_selected_clip.mp4')
        print('video tracking: waits for selected destination to appear')
        manager.options.current_video_path = None
        assert len(manager._video_reference_bank('clip.mp4')) == 3
        photo_source = manager._select_source_face(
            face([10, 10, 60, 60], [1, 0, 0]), sources,
            'selected_faces', video_one,
        )
        assert photo_source is sources[0], 'photos must select source per target face'
        print('photo source: selected per target face')

        current = np.zeros((64, 64, 3), dtype=np.uint8)
        previous = np.full_like(current, 220)
        no_mask = manager._hold_previous_face_region(
            current, previous, (16, 16, 48, 48), prev_mask=None,
        )
        assert np.array_equal(no_mask, current), 'missing mask must not create an ellipse'
        real_mask = np.zeros((64, 64), dtype=np.float32)
        real_mask[24:40, 24:40] = 1
        held = manager._hold_previous_face_region(
            current, previous, (16, 16, 48, 48), prev_mask=real_mask,
        )
        assert np.any(held[28:36, 28:36] > 0)
        assert np.array_equal(held[real_mask == 0], current[real_mask == 0]), (
            'blurred hold mask must not copy old pixels outside the applied swap'
        )
        temporal = manager._masked_temporal_blend(current, previous, real_mask, 0.4)
        assert np.any(temporal[28:36, 28:36] > 0)
        assert np.array_equal(temporal[real_mask == 0], current[real_mask == 0])
        print('video hold and temporal blend: confined to applied face mask')
    finally:
        roop.globals.selected_face_references = old_refs
        roop.globals.TARGET_FACES = old_targets
        ui.globals.ui_target_thumbs = old_thumbs

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix='faceswap_codec_') as temp:
        lossless = Path(temp) / 'working.avi'
        delivered = Path(temp) / 'delivered.mp4'
        source_audio = Path(temp) / 'source_audio.mp4'
        writer = cv2.VideoWriter(str(lossless), cv2.VideoWriter_fourcc(*'FFV1'), 30, (64, 48))
        assert writer.isOpened()
        for index in range(8):
            frame = np.full((48, 64, 3), index * 25, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        ffmpeg = root / 'ffmpeg' / 'ffmpeg.exe'
        subprocess.run([str(ffmpeg), '-y', '-f', 'lavfi', '-i',
                        'sine=frequency=440:duration=2', '-c:a', 'aac',
                        str(source_audio)], check=True, capture_output=True)
        encode_face_swap_video(str(lossless), str(source_audio), str(delivered),
                               start_frame=30, end_frame=38, fps=30)
        cap = cv2.VideoCapture(str(delivered))
        assert cap.isOpened()
        fourcc = ''.join(chr((int(cap.get(cv2.CAP_PROP_FOURCC)) >> (8 * i)) & 255) for i in range(4))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        assert fourcc.lower() in ('avc1', 'h264') and frames == 8, (fourcc, frames)
        subprocess.run([str(ffmpeg), '-v', 'error', '-i', str(delivered),
                        '-map', '0:a:0', '-f', 'null', '-'],
                       check=True, capture_output=True)
        print(f'video codec={fourcc}, frames={frames}, trimmed audio included')


if __name__ == '__main__':
    main()
