"""Verify the vision guard when another process holds the GPU.

A helper process reserves VRAM with torch, exactly like the external
application that was blocking the earlier runs.  Then the real production
preflight and the real production description function are executed in a
separate process: the guard must refuse with a VRAM message and the process
must survive (the CPU projector path would abort it natively).

No production code is changed and no model is downloaded.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / 'diagnostics/out/vision_guard_busy_gpu_20261020'

HOLDER = """
import sys, time
import torch
free, total = torch.cuda.mem_get_info(0)
reserve = int(float(sys.argv[1]) * (1024 ** 3))
block = torch.empty(reserve, dtype=torch.uint8, device='cuda')
print(f'RESERVED {block.numel() * block.element_size() // (1024 * 1024)} MiB', flush=True)
sys.stdout.flush()
time.sleep(float(sys.argv[2]))
"""

CHILD = """
import json, sys
sys.path.insert(0, r'%s')
from roop.img_editor import element_selection as es
from roop.img_editor import vision_guard as vg
from PIL import Image
settings = es.load_settings()
pre = vg.check_vision_preflight(settings)
out = {'ok': pre.ok, 'reason': pre.reason, 'gpu_offload': pre.gpu_offload,
       'free_vram_mb': pre.free_vram_mb, 'required_vram_mb': pre.required_vram_mb}
try:
    photo = Image.open(r'%s').convert('RGB')
    photo.thumbnail((settings['max_analysis_side'], settings['max_analysis_side']))
    out['description'] = es._moondream_description(photo, settings)
except BaseException as exc:
    out['vision_error'] = f'{type(exc).__name__}: {exc}'
print('EVIDENCE_JSON ' + json.dumps(out))
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reserve_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    holder = subprocess.Popen(
        [sys.executable, '-c', HOLDER, str(reserve_gb), '600'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        line = holder.stdout.readline().strip()
        print(f'HOLDER {line}', flush=True)
        time.sleep(3)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT)
        code = CHILD % (ROOT, ROOT / 'venv/Lib/site-packages/skimage/data/coffee.png')
        proc = subprocess.run([sys.executable, '-c', code], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, timeout=600)
        result = {'returncode': proc.returncode}
        for raw in proc.stdout.splitlines():
            if raw.startswith('EVIDENCE_JSON '):
                result['result'] = json.loads(raw[len('EVIDENCE_JSON '):])
        result['stderr_tail'] = proc.stderr[-1500:]
        (OUT / 'results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False),
                                          encoding='utf-8')
        print(json.dumps(result.get('result'), indent=2, ensure_ascii=False))
        print(f'RETURNCODE {proc.returncode}')
        print(f'EVIDENCE {OUT}')
    finally:
        holder.kill()
        holder.wait(timeout=30)


if __name__ == '__main__':
    main()
