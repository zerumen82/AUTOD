"""Measure oral decisions on explicitly supplied normal photos with local CPU models.

Interceptors only record actual inference and masks; they never replace them.
Optional quality snapshots allow a before/after run with identical faces/settings.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
import importlib
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
import io
import json
from pathlib import Path
import time

from diagnostics.probe_faceswap_measured_composition_real import (
    cpu_sessions_only, face_record, load_baseline, persist_metadata, run_variant, write_image,
)
import cv2
import numpy as np
import torch

import roop.face_util as face_util
import roop.globals
import roop.mouth_detector as mouth_module
from roop.ProcessMgr import ProcessMgr


def load_quality(path):
    loader = SourceFileLoader(f'quality_snapshot_{time.time_ns()}', str(path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module.detect_foreground_occlusion


def load_mouth_detector(path):
    loader = SourceFileLoader(f'mouth_snapshot_{time.time_ns()}', str(path))
    spec = spec_from_loader(loader.name, loader)
    module = module_from_spec(spec)
    loader.exec_module(module)
    return module.MouthDetector()


def texture_stats(target):
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3))
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    thresholds = (
        max(25, min(40, sobel.mean() * 1.5)),
        max(35, lap[lap > 0].mean() * 0.6 if np.any(lap > 0) else 40),
    )
    return {
        name: {
            'maximum': float(values.max()), 'threshold': float(threshold),
            'overflow_pixels': int((values > 255).sum()),
            'strong_edges_lost_by_uint8_wrap': int(
                ((values > threshold) & (values.astype(np.uint8) <= threshold)).sum()
            ),
        }
        for name, values, threshold in zip(('sobel', 'laplacian'), (sobel, lap), thresholds)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('targets', type=Path, nargs='+')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--baseline-quality', type=Path)
    parser.add_argument('--baseline-manager', type=Path)
    parser.add_argument('--baseline-mouth', type=Path)
    parser.add_argument('--target-detection-mode', choices=('verified', 'target'), default='verified')
    args = parser.parse_args()
    assert all(p.is_file() for p in (args.source, *args.targets))
    assert not torch.cuda.is_available()
    output = args.report / f'oral_evidence_cpu_{time.time_ns()}'
    output.mkdir(parents=True)
    metadata = {
        'completed': False, 'cpu_only': True, 'source': str(args.source), 'cases': [],
        'process_sha256': hashlib.sha256(Path('roop/ProcessMgr.py').read_bytes()).hexdigest(),
        'quality_sha256': hashlib.sha256(Path('roop/quality_enhancements.py').read_bytes()).hexdigest(),
        'mouth_sha256': hashlib.sha256(Path('roop/mouth_detector.py').read_bytes()).hexdigest(),
    }
    print(output, flush=True)
    module = importlib.import_module('roop.ProcessMgr')
    original_occ = module.detect_foreground_occlusion
    original_mouth = ProcessMgr._mouth_mask_in_aligned_face
    variants = [('current', ProcessMgr, original_occ)]
    if args.baseline_quality:
        variants.insert(0, ('before', ProcessMgr, load_quality(args.baseline_quality)))
        metadata['baseline_sha256'] = hashlib.sha256(args.baseline_quality.read_bytes()).hexdigest()
    elif args.baseline_manager:
        variants.insert(0, ('before', load_baseline(args.baseline_manager), original_occ))
        metadata['baseline_sha256'] = hashlib.sha256(args.baseline_manager.read_bytes()).hexdigest()
    metadata['target_detection_mode'] = args.target_detection_mode
    original_detector = mouth_module._mouth_detector_instance
    detectors = {}
    if args.baseline_mouth:
        detectors['before'] = load_mouth_detector(args.baseline_mouth)
        detectors['current'] = mouth_module.MouthDetector()
        assert all(detector.is_initialized for detector in detectors.values())
        metadata['baseline_mouth_sha256'] = hashlib.sha256(args.baseline_mouth.read_bytes()).hexdigest()
        if len(variants) == 1:
            variants.insert(0, ('before', ProcessMgr, original_occ))
    try:
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
            assert sources, 'Measured source unavailable'
            source, source_crop = sources[0]
            metadata['source_face'] = face_record(source, source_crop)
            for index, photo in enumerate(args.targets):
                captured = io.StringIO()
                with redirect_stdout(captured):
                    # Normal single-person portraits use the same verified crop
                    # path as the source bank; target mode can select a clipped
                    # background person when the principal face needs recovery.
                    targets = face_util.extract_face_images(
                        str(photo), is_source_face=args.target_detection_mode == 'verified',
                    )
                case = {'target': str(photo), 'runs': [], 'available_faces': len(targets)}
                metadata['cases'].append(case)
                (output / f'{index}_detection.log').write_text(captured.getvalue(), encoding='utf-8')
                if not targets:
                    continue
                target, target_crop = max(targets, key=lambda entry: float(
                    np.prod(np.asarray(entry[0].bbox)[2:] - np.asarray(entry[0].bbox)[:2])))
                case['target_face'] = face_record(target, target_crop)
                frame = cv2.imread(str(photo))
                assert frame is not None
                write_image(output / f'{index}_original.png', frame)
                results = []
                for label, manager_class, occ_function in variants:
                    if detectors:
                        mouth_module._mouth_detector_instance = detectors[label]
                    prefix = f'{index}_{label}'
                    records = []

                    def measure_occ(face_img, target_region, *positional, **kwargs):
                        mask = occ_function(face_img, target_region, *positional, **kwargs)
                        n = len(records)
                        write_image(output / f'{prefix}_aligned_{n}.png', target_region)
                        write_image(output / f'{prefix}_occlusion_{n}.png',
                                    (np.clip(mask, 0, 1) * 255).astype(np.uint8))
                        records.append({'texture': texture_stats(target_region),
                                        'mask_mean': float(mask.mean()),
                                        'mask_fraction': float((mask > 0.1).mean())})
                        return mask

                    def measure_mouth(*positional, **kwargs):
                        mask = original_mouth(*positional, **kwargs)
                        write_image(output / f'{prefix}_lips.png',
                                    (np.clip(mask, 0, 1) * 255).astype(np.uint8))
                        return mask

                    variant_globals = manager_class._process_face_swap_v21.__globals__
                    variant_globals['detect_foreground_occlusion'] = measure_occ
                    manager_class._mouth_mask_in_aligned_face = staticmethod(measure_mouth)
                    captured = io.StringIO()
                    with redirect_stdout(captured):
                        result, info = run_variant(manager_class, source, target, frame, photo,
                                                   output, prefix, 1.0, 0.70)
                    log = captured.getvalue()
                    (output / f'{prefix}.log').write_text(log, encoding='utf-8')
                    info['occlusion_measurements'] = records
                    info['decision'] = [line for line in log.splitlines()
                                        if line.startswith('[MOUTH_DECISION]')]
                    assert info['successful'] and all(info['model_calls'].values())
                    assert info['decision'], 'Missing oral decision'
                    case['runs'].append(info)
                    results.append(result)
                if len(results) == 2:
                    diff = cv2.absdiff(*results)
                    case['changed_pixels'] = int((diff.max(axis=2) > 0).sum())
                    case['mean_difference'] = float(diff.mean())
                thumbs = [cv2.resize(im, (320, 320)) for im in (frame, *results)]
                write_image(output / f'{index}_comparison.png', np.hstack(thumbs))
                print(f'{photo.name}: {case["runs"][-1]["decision"]}', flush=True)
            metadata['completed'] = True
    finally:
        module.detect_foreground_occlusion = original_occ
        ProcessMgr._mouth_mask_in_aligned_face = staticmethod(original_mouth)
        if detectors:
            mouth_module._mouth_detector_instance = original_detector
            for detector in detectors.values():
                detector.release()
    print(json.dumps({'completed': metadata['completed'], 'output': str(output)}), flush=True)


if __name__ == '__main__':
    main()
