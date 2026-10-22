"""Local Moondream descriptions of measured SAM2 masks, diagnostic only.

The model sees an isolated image and an open descriptive prompt, never source
labels, candidate quality annotations, object catalogues or reference answers.
No downloads, coordinates from VLM or production integration. Llama text
requests CPU; the installed Llava projector has no exposed CPU selector and was
observed on GPU during the interrupted first attempt. Do not run beside renders.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from roop.img_editor import element_selection as es

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'diagnostics/editor_sam2_moondream_20261019.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'diagnostics/out/editor_sam2_moondream_20261019')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    settings = es.load_settings()
    inv = settings['inventory']
    sam_path = args.candidates.resolve()
    sam = json.loads(sam_path.read_text(encoding='utf-8-sig'))
    photo_path = resolve(sam['source'])
    photo = Image.open(photo_path).convert('RGB')
    if sam['fingerprint'] != es.image_fingerprint(photo):
        raise ValueError('Source/SAM2 fingerprint mismatch')
    model_path, projector_path = resolve(inv['model_path']), resolve(inv['projector_path'])
    if not model_path.is_file() or not projector_path.is_file():
        raise FileNotFoundError('Installed local Moondream model/projector required; no downloads')
    if cfg['gpu_layers'] != 0:
        raise ValueError('This diagnostic requires zero GPU layers for the text model')
    output = args.output / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    print('EVIDENCE', output, flush=True)
    report = {
        'source': str(photo_path), 'source_sha256': sha256(photo_path),
        'source_fingerprint': sam['fingerprint'], 'sam2_results': str(sam_path),
        'sam2_results_sha256': sha256(sam_path), 'config': cfg,
        'config_sha256': sha256(args.config), 'script_sha256': sha256(Path(__file__)),
        'model_path': str(model_path), 'model_sha256': sha256(model_path),
        'projector_path': str(projector_path), 'projector_sha256': sha256(projector_path),
        'text_device_requested': 'cpu',
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'projector_device': 'not controlled by installed Llava handler; GPU observed in interrupted attempt',
        'scope': 'Open descriptions only; no semantic/completeness guarantee',
        'status': 'running',
        'candidates': []}

    def save_report():
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save_report()
    es._add_llama_dll_directories(settings)
    from llama_cpp import Llama
    from llama_cpp.llama_chat_format import MoondreamChatHandler
    handler = llm = None
    started = time.perf_counter()
    try:
        handler = MoondreamChatHandler(clip_model_path=str(projector_path),
                                      verbose=cfg['projector_load_verbose'])
        # Keep backend-load diagnostics, without printing the image data URL
        # during the subsequent caption requests.
        handler.verbose = False
        llm = Llama(model_path=str(model_path), chat_handler=handler,
                    n_ctx=cfg['context_size'], n_threads=cfg['threads'],
                    n_gpu_layers=cfg['gpu_layers'], verbose=False)
        report['load_seconds'] = time.perf_counter() - started
        save_report()
        for index in cfg['candidates']:
            candidate = sam['candidates'][index]
            mask_path = sam_path.parent / candidate['mask']
            mask = np.asarray(Image.open(mask_path).convert('L')) >= 128
            if mask.shape != (photo.height, photo.width) or not mask.any():
                raise ValueError(f'Invalid measured mask for candidate {index}')
            ys, xs = np.where(mask)
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            pixels = np.asarray(photo).copy()
            pixels[~mask] = cfg['neutral_rgb']
            isolated = Image.fromarray(pixels).crop(box)
            input_path = output / f'candidate_{index:04d}_masked_crop.png'
            isolated.save(input_path)
            data_url = es._image_data_url(isolated, cfg['max_image_side'])
            sent_path = output / f'candidate_{index:04d}_sent.jpg'
            sent_path.write_bytes(base64.b64decode(data_url.split(',', 1)[1]))
            tick = time.perf_counter()
            response = llm.create_chat_completion(
                messages=[{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': data_url}},
                    {'type': 'text', 'text': cfg['prompt']},
                ]}], temperature=cfg['temperature'], max_tokens=cfg['max_tokens'],
                stop=cfg['stop'])
            description = response['choices'][0]['message']['content'].strip()
            entry = {'index': index, 'mask': str(mask_path), 'mask_sha256': sha256(mask_path),
                     'source_label_not_sent': candidate.get('label'),
                     'method': candidate['method'], 'bbox': box,
                     'coverage': float(mask.mean()), 'input_crop': input_path.name,
                     'input_crop_sha256': sha256(input_path), 'sent_jpeg': sent_path.name,
                     'sent_jpeg_sha256': sha256(sent_path), 'prompt': cfg['prompt'],
                     'description': description, 'seconds': time.perf_counter() - tick,
                     'finish_reason': response['choices'][0].get('finish_reason'),
                     'usage': response.get('usage')}
            report['candidates'].append(entry)
            save_report()
            print(f'DESCRIPTION {index:04d} ({entry["seconds"]:.2f}s): {description}', flush=True)
        report['total_seconds'] = time.perf_counter() - started
        report['status'] = 'completed'
        save_report()
    except BaseException as error:
        report['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
        report['error'] = f'{type(error).__name__}: {error}'
        report['total_seconds'] = time.perf_counter() - started
        save_report()
        raise
    finally:
        if llm is not None:
            llm.close()
        if handler is not None:
            handler._exit_stack.close()
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
