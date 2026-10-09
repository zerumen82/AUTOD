"""Check mouth state isolation with supplied normal portraits and real CPU models.

Seeded history is a regression fixture, not evidence that a user's photograph
has an object over the lips. Face measurements and all model outputs are real.
"""

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

from diagnostics.probe_faceswap_measured_composition_real import (
    cpu_sessions_only, face_record, load_baseline, persist_metadata,
    verify_processors, write_image,
)
import cv2
import numpy as np
import torch

import roop.face_util as face_util
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.types import FaceSet


STATE_NAMES = (
    '_mouth_ratio_smooth', '_mouth_state_history', '_mouth_hold_count',
    '_mouth_paste_ema_by_person', '_mouth_blend_ema_by_person',
    '_mouth_force_paste_by_person',
)


def run_case(cls, source, target, frame, photo, output, name, photo_mode,
             smoothing, stale):
    cls._swap_call_count = 0
    manager = cls()
    captured = io.StringIO()
    key = f'video_{photo.name}'
    person_key = f'{key}_p0'
    try:
        with redirect_stdout(captured):
            manager.initialize(
                [FaceSet([deepcopy(source)])], [deepcopy(target)],
                SimpleNamespace(blend_ratio=1.0, current_video_path=None),
                face_swap_mode='selected_faces',
            )
            engine, masker, enhancer = verify_processors(manager)
            sessions = [engine.model.session, masker.model_xseg, enhancer.model_gfpgan]
            calls_before = [s.diagnostic_calls for s in sessions]
            manager._is_single_photo = photo_mode
            manager._current_video_key = key
            for attr in STATE_NAMES:
                setattr(manager, attr, {})
            if stale:
                manager._mouth_ratio_smooth[person_key] = 0.99
                manager._mouth_state_history[person_key] = True
                manager._mouth_hold_count[person_key] = 4
                manager._mouth_paste_ema_by_person[person_key] = 0.95
                manager._mouth_blend_ema_by_person[person_key] = 0.91
                manager._mouth_force_paste_by_person[person_key] = False
            before = {attr: deepcopy(getattr(manager, attr)) for attr in STATE_NAMES}
            started = time.monotonic()
            result = manager._process_face_swap_v21(
                deepcopy(source), deepcopy(target), frame.copy(), frame,
                enable_temporal_smoothing=smoothing, photo_path=str(photo),
            )
            elapsed = time.monotonic() - started
            after = {attr: deepcopy(getattr(manager, attr)) for attr in STATE_NAMES}
            calls = [s.diagnostic_calls - n for s, n in zip(sessions, calls_before)]
        assert result is not None and result.shape == frame.shape
        assert all(n > 0 for n in calls), 'Required model did not run'
        changed = int((cv2.absdiff(result, frame).max(axis=2) > 0).sum())
        assert changed > 0, 'Returning the original is not a successful swap'
        decision = [line for line in captured.getvalue().splitlines()
                    if line.startswith('[MOUTH_DECISION]')]
        write_image(output / f'{name}.png', result)
        (output / f'{name}.log').write_text(captured.getvalue(), encoding='utf-8')
        return result, {
            'name': name, 'photo': photo_mode, 'smoothing': smoothing,
            'seeded_history': stale, 'state_before': before, 'state_after': after,
            'state_unchanged': before == after, 'seconds': elapsed,
            'model_calls': dict(zip(('inswapper128', 'xseg', 'gfpgan'), calls)),
            'changed_pixels': changed, 'decision': decision,
        }
    finally:
        for processor in manager.processors.values():
            processor.Release()
        manager.processors.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('target', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert all(p.is_file() for p in (args.source, args.target, args.baseline))
    assert not torch.cuda.is_available()
    output = args.report / f'mouth_cpu_{time.time_ns()}'
    output.mkdir(parents=True)
    metadata = {
        'completed': False, 'cpu_only': True,
        'source': str(args.source), 'target': str(args.target),
        'history_fixture': 'Seeded previous ratio/hold/blend; current faces and inference real',
        'current_sha256': hashlib.sha256(Path('roop/ProcessMgr.py').read_bytes()).hexdigest(),
        'cases': [],
    }
    print(output, flush=True)
    with persist_metadata(metadata, output), cpu_sessions_only():
        analyser = face_util.get_face_analyser_cpu()
        assert analyser is not None
        assert all(m.session.get_providers() == ['CPUExecutionProvider']
                   for m in analyser.models.values())
        face_util.FACE_ANALYSER = analyser
        roop.globals.execution_providers = ['CPUExecutionProvider']
        roop.globals.selected_enhancer = 'GFPGAN'
        roop.globals.use_enhancer = True
        roop.globals.enhancer_blend_factor = 0.70
        sources = face_util.extract_face_images(str(args.source), is_source_face=True)
        targets = face_util.extract_face_images(str(args.target))
        assert sources and targets
        source, source_crop = sources[0]
        target, target_crop = targets[0]
        metadata['source_face'] = face_record(source, source_crop)
        metadata['target_face'] = face_record(target, target_crop)
        frame = cv2.imdecode(np.fromfile(args.target, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert frame is not None
        write_image(output / 'original.png', frame)
        baseline = load_baseline(args.baseline)
        results = {}
        for name, cls, photo, smoothing, stale in (
            ('before_photo_clean', baseline, True, False, False),
            ('before_photo_stale', baseline, True, False, True),
            ('photo_clean', ProcessMgr, True, False, False),
            ('photo_stale', ProcessMgr, True, False, True),
            ('photo_clean_smoothing', ProcessMgr, True, True, False),
            ('photo_stale_smoothing', ProcessMgr, True, True, True),
            ('instant_frame_stale', ProcessMgr, False, False, True),
            ('video_stale', ProcessMgr, False, True, True),
        ):
            pixels, row = run_case(cls, source, target, frame, args.target,
                                   output, name, photo, smoothing, stale)
            results[name] = pixels
            metadata['cases'].append(row)
            print(json.dumps(row), flush=True)
            if cls is ProcessMgr:
                assert row['decision'], 'Missing decision evidence'
                expected_temporal = smoothing and not photo
                assert f'temporal={expected_temporal}' in row['decision'][0]
                assert row['state_unchanged'] != expected_temporal
                if not expected_temporal:
                    raw = float(re.search(r'ratio=([0-9.]+)', row['decision'][0])[1])
                    smoothed = float(re.search(r'smooth=([0-9.]+)', row['decision'][0])[1])
                    assert raw == smoothed
        assert not metadata['cases'][1]['state_unchanged'], 'Baseline failed to reproduce state mutation'
        # Match the pipeline flag: generic smoothing also performs LAB color
        # conversions. That preexisting operation is not an oral history effect.
        for clean, candidate in (
            ('photo_clean', 'before_photo_clean'),
            ('photo_clean', 'photo_stale'),
            ('photo_clean_smoothing', 'photo_stale_smoothing'),
        ):
            count = int((cv2.absdiff(results[clean], results[candidate]).max(axis=2) > 0).sum())
            metadata[f'{clean}_vs_{candidate}_changed_pixels'] = count
            assert count == 0, f'Photo result depends on oral history: {candidate}'
        metadata['generic_smoothing_changed_pixels'] = int((cv2.absdiff(
            results['photo_clean'], results['photo_clean_smoothing'],
        ).max(axis=2) > 0).sum())
        write_image(output / 'compare_photo.png', np.hstack((
            frame, results['photo_clean'], results['photo_stale'],
        )))
        assert face_record(source, source_crop) == metadata['source_face']
        assert face_record(target, target_crop) == metadata['target_face']
        metadata['completed'] = True
    print(output, flush=True)


if __name__ == '__main__':
    main()
