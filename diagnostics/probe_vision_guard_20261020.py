"""Verify the production vision guard with real executions, no mocks.

Scenarios executed, each in its own process so the ggml backend state is real:

  allow    : CUDA visible and enough VRAM -> the guard must allow the run.
  no_cuda  : CUDA hidden -> the guard must refuse (the CPU projector crashes
             the process natively).
  low_vram : CUDA visible but the requirement raised above the free VRAM ->
             the guard must refuse with a VRAM message.
  disabled : the guard disabled in the settings -> the vision path is allowed.

The allow and disabled scenarios run the real production description function,
so a native crash in the CPU projector would be visible here.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor import vision_guard as vg  # noqa: E402

IMAGE = Path('venv/Lib/site-packages/skimage/data/coffee.png')
OUT = ROOT / 'diagnostics/out/vision_guard_20261020'


def child(mode: str) -> dict:
    """Run one scenario in a clean process and return its evidence."""
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT)
    if mode == 'no_cuda':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    code = (
        "import json, sys\n"
        "sys.path.insert(0, r'%s')\n"
        "from roop.img_editor import element_selection as es\n"
        "from roop.img_editor import vision_guard as vg\n"
        "from PIL import Image\n"
        "settings = es.load_settings()\n"
        "mode = %r\n"
        "if mode == 'low_vram':\n"
        "    settings['vision_guard']['vram_margin_mb'] = 10**9\n"
        "if mode == 'disabled':\n"
        "    settings['vision_guard']['enabled'] = False\n"
        "pre = vg.check_vision_preflight(settings)\n"
        "out = {'ok': pre.ok, 'reason': pre.reason, 'gpu_offload': pre.gpu_offload,\n"
        "       'max_devices': pre.max_devices, 'free_vram_mb': pre.free_vram_mb,\n"
        "       'required_vram_mb': pre.required_vram_mb}\n"
        "if mode in ('allow', 'disabled'):\n"
        "    photo = Image.open(r'%s').convert('RGB')\n"
        "    photo.thumbnail((settings['max_analysis_side'], settings['max_analysis_side']))\n"
        "    try:\n"
        "        out['description'] = es._moondream_description(photo, settings)\n"
        "    except BaseException as exc:\n"
        "        out['vision_error'] = f'{type(exc).__name__}: {exc}'\n"
        "print('EVIDENCE_JSON ' + json.dumps(out))\n"
    ) % (ROOT, mode, ROOT / IMAGE)
    proc = subprocess.run(
        [sys.executable, '-c', code], env=env, cwd=str(ROOT),
        capture_output=True, text=True, timeout=900,
    )
    evidence = {'mode': mode, 'returncode': proc.returncode}
    for line in proc.stdout.splitlines():
        if line.startswith('EVIDENCE_JSON '):
            evidence['result'] = json.loads(line[len('EVIDENCE_JSON '):])
    evidence['stdout_tail'] = proc.stdout[-2000:]
    evidence['stderr_tail'] = proc.stderr[-2000:]
    return evidence


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for mode in ('allow', 'no_cuda', 'low_vram', 'disabled'):
        results[mode] = child(mode)
        print(f"--- {mode} ---")
        print(json.dumps(results[mode].get('result'), indent=2, ensure_ascii=False))
    (OUT / 'results.json').write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                      encoding='utf-8')
    print(f'EVIDENCE {OUT}')


if __name__ == '__main__':
    main()
