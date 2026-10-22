"""Runtime detection of the Moondream vision backend (diagnostic only).

Scenarios:
  order_hazard : import llama_cpp BEFORE adding the CUDA DLL directories and
                 check whether the CUDA backend is still reachable afterwards.
  fresh        : add the CUDA DLL directories first (production order) and
                 report the backend state.
  hidden_cuda  : hide CUDA and confirm the detection reports it as unusable.

No production code is modified and no model is loaded.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor import element_selection as es  # noqa: E402


def report(tag: str, sink: dict) -> None:
    try:
        from llama_cpp import llama_cpp as L
        sink[tag] = bool(L.llama_supports_gpu_offload())
    except BaseException as exc:
        sink[tag] = f'ERR {type(exc).__name__}: {exc}'


def main() -> None:
    settings = es.load_settings()
    sink: dict = {'scenario': sys.argv[1]}

    if sys.argv[1] == 'order_hazard':
        # llama_cpp imported first: the CUDA DLL may already be unresolvable.
        from llama_cpp import llama_cpp as L
        report('BEFORE_DIRS', sink)
        es._add_llama_dll_directories(settings)
        report('AFTER_DIRS', sink)
        try:
            L.llama_backend_init()
            report('AFTER_BACKEND_INIT', sink)
        except BaseException as exc:
            sink['AFTER_BACKEND_INIT'] = f'ERR {type(exc).__name__}: {exc}'
    elif sys.argv[1] == 'fresh':
        es._add_llama_dll_directories(settings)
        report('FRESH', sink)
    elif sys.argv[1] == 'hidden_cuda':
        es._add_llama_dll_directories(settings)
        report('HIDDEN', sink)
    else:
        raise SystemExit('scenario required')

    out = ROOT / 'diagnostics/out/moondream_gpu_backend_20261020' / f'{sys.argv[1]}.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sink, indent=2), encoding='utf-8')
    print(json.dumps(sink))


if __name__ == '__main__':
    main()
