"""Control A/B for the Moondream CPU crash observed on 2026-10-19 23:42.

The isolated-caption probe died with a native STATUS_STACK_BUFFER_OVERRUN
(0xc0000409, ucrtbase.dll) right after the context was created, while the
projector had been proven to load on CPU. This control separates three
possibilities with real executions instead of assumptions:

  phase_text  : text model alone, no vision handler -> is the CPU text path OK?
  phase_vision : the same handler used by production element_selection ->
                 does the vision path work on CPU?

Everything runs on the installed local models, CPU only, no downloads and no
production code changes.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from PIL import Image

from roop.img_editor import element_selection as es

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path,
                        default=Path('venv/Lib/site-packages/skimage/data/coffee.png'))
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'diagnostics/out/moondream_cpu_control_20261019')
    args = parser.parse_args()

    image_path = args.image if args.image.is_absolute() else ROOT / args.image
    output = args.output / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=False)
    print('EVIDENCE', output, flush=True)

    settings = es.load_settings()
    inv = settings['inventory']
    report = {
        'image': str(image_path),
        'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'threads': inv['threads'],
        'context_size': inv['context_size'],
        'phases': {},
    }

    def save():
        (output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    es._add_llama_dll_directories(settings)
    from llama_cpp import Llama

    photo = Image.open(image_path).convert('RGB')

    # Phase 1: text model only, no vision handler.
    phase = {'name': 'text_only', 'status': 'running'}
    report['phases']['text_only'] = phase
    save()
    llm = None
    try:
        tick = time.perf_counter()
        llm = Llama(model_path=str(ROOT / inv['model_path']), n_ctx=inv['context_size'],
                    n_gpu_layers=0, n_threads=inv['threads'], verbose=False)
        phase['load_seconds'] = time.perf_counter() - tick
        tick = time.perf_counter()
        response = llm.create_chat_completion(
            messages=[{'role': 'user', 'content': 'Reply with the single word: ready'}],
            temperature=0.0, max_tokens=8)
        phase['text'] = response['choices'][0]['message']['content'].strip()
        phase['seconds'] = time.perf_counter() - tick
        phase['status'] = 'completed'
    except BaseException as error:
        phase['status'] = 'failed'
        phase['error'] = f'{type(error).__name__}: {error}'
    finally:
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass
    save()
    print(f'PHASE text_only -> {phase["status"]}: {phase.get("text", phase.get("error"))}',
          flush=True)

    # Phase 2: the production vision path, CPU only.
    phase = {'name': 'vision', 'status': 'running'}
    report['phases']['vision'] = phase
    save()
    try:
        tick = time.perf_counter()
        phase['description'] = es._moondream_description(photo, settings)
        phase['seconds'] = time.perf_counter() - tick
        phase['status'] = 'completed'
    except BaseException as error:
        phase['status'] = 'failed'
        phase['error'] = f'{type(error).__name__}: {error}'
    save()
    print(f'PHASE vision -> {phase["status"]}: {phase.get("description", phase.get("error"))}',
          flush=True)

    report['status'] = 'completed'
    save()
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
