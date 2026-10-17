"""Real CPU compositor regression on two copies of an inspected portrait.

The canvas is a labelled diagnostic fixture, not a natural two-person scene.
Detections, embeddings, swap, segmentation and enhancement all use local models.
"""
import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

from diagnostics.probe_faceswap_measured_composition_real import (
    cpu_sessions_only, load_baseline, persist_metadata, verify_processors, write_image,
)
import cv2
import numpy as np
import torch

import roop.face_util as face_util
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.types import FaceSet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('portrait', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    for path in (args.source, args.portrait, args.baseline):
        assert path.is_file(), path
    output = args.report / f'multiface_cpu_{time.time_ns()}'
    output.mkdir(parents=True)
    meta = dict(completed=False, cpu_only=True,
                fixture='two spatial copies of one real inspected adult portrait',
                source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                portrait_sha256=hashlib.sha256(args.portrait.read_bytes()).hexdigest(),
                current_sha256=hashlib.sha256(Path('roop/ProcessMgr.py').read_bytes()).hexdigest(),
                baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
                variants=[])
    print(output, flush=True)
    with persist_metadata(meta, output), cpu_sessions_only(), (output / 'run.log').open('w', encoding='utf-8') as log, redirect_stdout(log):
        roop.globals.execution_providers = ['CPUExecutionProvider']
        roop.globals.selected_enhancer = 'GFPGAN'
        roop.globals.use_enhancer = True
        roop.globals.enhancer_blend_factor = .70
        analyser = face_util.get_face_analyser_cpu()
        assert analyser is not None
        assert all(m.session.get_providers() == ['CPUExecutionProvider'] for m in analyser.models.values())
        face_util.FACE_ANALYSER = analyser
        source = face_util.extract_face_images(str(args.source), is_source_face=True)[0][0]
        photo = cv2.imdecode(np.fromfile(args.portrait, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert photo is not None
        photo = cv2.resize(photo, (round(photo.shape[1]*512/photo.shape[0]), 512))
        padding = np.zeros((512, 128, 3), dtype=np.uint8)
        frame = np.hstack([photo, padding, photo])
        fixture = output / 'input.png'
        write_image(fixture, frame)
        targets = sorted(analyser.get(frame), key=lambda f: f.bbox[0])
        assert len(targets) == 2, len(targets)
        meta['bboxes'] = [np.asarray(f.bbox).tolist() for f in targets]
        left = np.s_[:, :photo.shape[1]]
        right = np.s_[:, photo.shape[1]+padding.shape[1]:]
        comparisons = [frame]
        for label, cls in [('before', load_baseline(args.baseline)), ('current', ProcessMgr)]:
            cls._swap_call_count = 0
            manager = cls()
            try:
                manager.initialize([FaceSet([deepcopy(source)])], deepcopy(targets),
                                   SimpleNamespace(blend_ratio=1., current_video_path=None,
                                                   face_distance_threshold=.20),
                                   face_swap_mode='selected_faces')
                manager._is_single_photo = True
                engine, masker, enhancer = verify_processors(manager)
                result = frame.copy()
                steps = []
                for index, target in enumerate(targets):
                    result = manager._process_face_swap_v21(
                        deepcopy(source), deepcopy(target), result, frame,
                        enable_temporal_smoothing=False, person_idx=index, photo_path=str(fixture))
                    assert result is not None
                    steps.append(result.copy())
                    write_image(output / f'{label}_after_{index+1}.png', result)
                calls = dict(faceswap=engine.model.session.diagnostic_calls,
                             xseg=masker.model_xseg.diagnostic_calls,
                             gfpgan=enhancer.model_gfpgan.diagnostic_calls)
                assert all(value >= 2 for value in calls.values()), calls
                first_changed = np.any(steps[0][left] != frame[left], axis=2)
                second_changed = np.any(steps[1][right] != frame[right], axis=2)
                assert first_changed.any() and second_changed.any()
                preserved = bool(np.array_equal(steps[0][left], steps[1][left]))
                erased = bool(np.array_equal(frame[left], steps[1][left]))
                meta['variants'].append(dict(label=label, model_calls=calls,
                                             first_changed_pixels=int(first_changed.sum()),
                                             second_changed_pixels=int(second_changed.sum()),
                                             first_preserved_after_second=preserved,
                                             first_erased_after_second=erased))
                assert preserved if label == 'current' else erased
                comparisons.append(result)
            finally:
                for processor in manager.processors.values():
                    processor.Release()
                manager.processors.clear()
        write_image(output / 'original_before_current.png', np.vstack(comparisons))
        meta['completed'] = True
    print(json.dumps(meta, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
