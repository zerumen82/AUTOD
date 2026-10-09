"""Run a fully decoded, previously inspected normal video through local models.

Both variants call the actual process_frame detector, tracker and compositor.
Inference interceptors record measurements and always return the real output.
No downloads, frame skipping, UI restart or original file changes.
"""

import argparse
from contextlib import contextmanager, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import time
from types import MethodType, SimpleNamespace

from diagnostics.probe_faceswap_measured_composition_real import (
    cpu_sessions_only, digest_array, load_baseline, persist_metadata,
    verify_processors, write_image,
)
import cv2
import numpy as np
import onnxruntime as ort
import torch

import roop.face_util as face_util
import roop.globals
import roop.mouth_detector as mouth_module
from roop.ProcessMgr import ProcessMgr
from roop.types import FaceSet


@contextmanager
def cuda_sessions():
    original = ort.InferenceSession

    class RecordedSession(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # FaceSwap inspects the source tensor shape with an explicit CPU
            # session before constructing its actual CUDA inference session.
            requested = kwargs.get('providers')
            if requested == ['CPUExecutionProvider']:
                assert self.get_providers() == requested
            else:
                assert 'CUDAExecutionProvider' in self.get_providers()
            self.diagnostic_calls = 0

        def run(self, *args, **kwargs):
            self.diagnostic_calls += 1
            return super().run(*args, **kwargs)

        def run_with_iobinding(self, *args, **kwargs):
            self.diagnostic_calls += 1
            return super().run_with_iobinding(*args, **kwargs)

    ort.InferenceSession = RecordedSession
    try:
        yield
    finally:
        ort.InferenceSession = original


def run_video(cls, label, frames, fps, source, reference, video, output, device):
    cls._swap_call_count = 0
    detector_before = mouth_module._mouth_detector_instance
    mouth_module._mouth_detector_instance = mouth_module.MouthDetector()
    manager = cls()
    options = SimpleNamespace(
        blend_ratio=1.0, current_video_path=str(video),
        face_swap_mode='selected_faces', use_enhancer=True,
    )
    roop.globals.selected_face_references = {
        f'selected_face_ref_{video.name}': [dict(
            embedding=reference.embedding.copy(), face_obj=deepcopy(reference),
            bbox=np.asarray(reference.bbox).copy(), source_path=str(video),
        )],
    }
    manager.initialize([FaceSet([deepcopy(source)])], [deepcopy(reference)],
                       options, face_swap_mode='selected_faces')
    if device == 'cpu':
        engine, masker, enhancer = verify_processors(manager)
    else:
        engine = manager.processors['faceswap']
        masker = manager.processors['mask_xseg']
        enhancer = manager.processors['enhance_gfpgan']
        for processor, session in ((engine, engine.model.session),
                                   (masker, masker.model_xseg),
                                   (enhancer, enhancer.model_gfpgan)):
            assert processor.devicename == 'cuda'
            assert 'CUDAExecutionProvider' in session.get_providers()
    swaps = []
    real_run = engine.Run

    def record_swap(_engine, origin, target, *args, **kwargs):
        result = real_run(origin, target, *args, **kwargs)
        if result is not None:
            pixels, matrix = result
            swaps.append(dict(
                target_bbox=np.asarray(target.bbox).tolist(),
                target_similarity=manager._calculate_similarity(
                    reference.embedding, target.embedding),
                source_hash=digest_array(origin.embedding),
                raw_hash=digest_array(pixels), matrix=np.asarray(matrix).tolist(),
            ))
        return result

    engine.Run = MethodType(record_swap, engine)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(output / f'{label}.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    assert writer.isOpened(), 'Diagnostic video writer unavailable'
    results, rows = [], []
    started = time.monotonic()
    try:
        with (output / f'{label}.log').open('w', encoding='utf-8') as logfile:
            for index, frame in enumerate(frames):
                previous = len(swaps)
                captured = io.StringIO()
                with redirect_stdout(captured):
                    result = manager.process_frame(frame.copy(), enable_temporal_smoothing=True)
                logged = captured.getvalue()
                logfile.write(logged)
                logfile.flush()
                assert result is not None and result.shape == frame.shape
                assert '[MOUTH_ERR]' not in logged, logged
                assert '[ERROR]' not in logged, logged
                assert 'Exception' not in logged and 'Traceback' not in logged, logged
                diff = cv2.absdiff(result, frame).max(axis=2)
                rows.append(dict(
                    decoded_frame=index, tracking_clock=getattr(
                        manager, f'_frame_count_{video.name}', None),
                    swaps=deepcopy(swaps[previous:]),
                    changed_pixels=int((diff > 0).sum()),
                    pixel_hash=digest_array(result),
                    decisions=[line for line in logged.splitlines()
                               if '[MOUTH_DECISION]' in line],
                    tracking=[line for line in logged.splitlines()
                              if any(tag in line for tag in ('[TRACK]', '[SCENE_CUT]', '[ZOOM_SKIP]'))],
                ))
                results.append(result.copy())
                writer.write(result)
                if index % 20 == 0 or index + 1 == len(frames):
                    print(f'{label}: {index + 1}/{len(frames)} frames, '
                          f'{len(swaps)} real swaps, {time.monotonic()-started:.1f}s', flush=True)
        calls = dict(inswapper128=engine.model.session.diagnostic_calls,
                     xseg=masker.model_xseg.diagnostic_calls,
                     gfpgan=enhancer.model_gfpgan.diagnostic_calls)
        assert all(count > 0 for count in calls.values()), calls
        return results, dict(label=label, frames=rows, model_calls=calls,
                             providers=engine.model.session.get_providers(),
                             seconds=time.monotonic()-started,
                             swaps_total=len(swaps))
    finally:
        writer.release()
        for processor in manager.processors.values():
            processor.Release()
        manager.processors.clear()
        detector = mouth_module._mouth_detector_instance
        if detector is not None and detector is not detector_before:
            detector.face_mesh.close()
        mouth_module._mouth_detector_instance = detector_before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('video', type=Path)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda':
        os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        assert torch.cuda.is_available()
    else:
        assert not torch.cuda.is_available()
    assert all(path.is_file() for path in (args.source, args.video, args.baseline))
    output = args.report / f'real_video_{args.device}_{time.time_ns()}'
    output.mkdir(parents=True)
    metadata = dict(completed=False, cpu_only=args.device == 'cpu', device=args.device,
                    video=str(args.video), source=str(args.source),
                    video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                    current_sha256=hashlib.sha256(Path('roop/ProcessMgr.py').read_bytes()).hexdigest(),
                    baseline_sha256=hashlib.sha256(args.baseline.read_bytes()).hexdigest(), variants=[])
    print(output, flush=True)
    sessions = cpu_sessions_only() if args.device == 'cpu' else cuda_sessions()
    with persist_metadata(metadata, output), sessions:
        capture = cv2.VideoCapture(str(args.video))
        assert capture.isOpened()
        frames = []
        fps = capture.get(cv2.CAP_PROP_FPS)
        expected = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
        capture.release()
        assert frames and len(frames) == expected and fps > 0
        metadata.update(decoded=len(frames), fps=fps, seconds=len(frames)/fps)
        providers = ['CPUExecutionProvider'] if args.device == 'cpu' else [
            'CUDAExecutionProvider', 'CPUExecutionProvider']
        roop.globals.execution_providers = providers
        analyser = (face_util.get_face_analyser_cpu() if args.device == 'cpu'
                    else face_util.get_face_analyser())
        assert analyser is not None
        assert all(args.device != 'cpu' or model.session.get_providers() == providers
                   for model in analyser.models.values())
        assert all(args.device != 'cuda' or 'CUDAExecutionProvider' in model.session.get_providers()
                   for model in analyser.models.values())
        face_util.FACE_ANALYSER = analyser
        roop.globals.selected_enhancer = 'GFPGAN'
        roop.globals.use_enhancer = True
        roop.globals.enhancer_blend_factor = 0.70
        origins = face_util.extract_face_images(str(args.source), is_source_face=True)
        assert origins
        source = origins[0][0]
        from roop.face_util_rotation import get_all_faces_smart
        targets = get_all_faces_smart(frames[0], for_target=True, prefer_fast=True, include_180=False)
        assert targets
        reference = max(targets, key=lambda face: (face.bbox[2]-face.bbox[0])*(face.bbox[3]-face.bbox[1]))
        metadata['reference'] = dict(bbox=np.asarray(reference.bbox).tolist(),
                                     kps=np.asarray(reference.kps).tolist())
        variants = [('before', load_baseline(args.baseline)), ('current', ProcessMgr)]
        pixels = {}
        for label, cls in variants:
            results, row = run_video(cls, label, frames, fps, source, reference, args.video, output, args.device)
            pixels[label] = results
            metadata['variants'].append(row)
        metadata['different_pixels_per_frame'] = [int((np.any(a != b, axis=2)).sum())
                                                  for a, b in zip(pixels['before'], pixels['current'])]
        chosen = np.unique(np.linspace(0, len(frames)-1, 8, dtype=int))
        tiles = []
        for index in chosen:
            tiles.append(np.concatenate([cv2.resize(images[index], (176, 320))
                                         for images in (frames, pixels['before'], pixels['current'])], axis=1))
        write_image(output / 'original_before_current.jpg', np.concatenate(tiles, axis=0))
        metadata['comparison_frames'] = chosen.tolist()
        metadata['limitations'] = ['Complete short normal video; not a long-video certification.',
                                    'No deliberate cut or zoom in this native video.',
                                    'Real pixel changes do not prove identity quality.']
        metadata['completed'] = True
    print(json.dumps(dict(completed=True, output=str(output), decoded=metadata['decoded'],
                          variants=[dict(label=r['label'], swaps=r['swaps_total'], calls=r['model_calls'])
                                    for r in metadata['variants']]), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
