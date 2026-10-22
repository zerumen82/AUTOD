"""Exercise the UI analysis callback while the GPU is held by another process.

The UI path must show the guard message and must not raise, and the session
must stay usable afterwards.  A holder process reserves VRAM so the guard
refuses the vision run, exactly like the external application that caused the
earlier native crash.

Real callback, real Gradio update objects, no mocks and no downloads.
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

OUT = ROOT / 'diagnostics/out/vision_guard_20261020'

HOLDER = """
import sys, time
import torch
reserve = int(float(sys.argv[1]) * (1024 ** 3))
block = torch.empty(reserve, dtype=torch.uint8, device='cuda')
block.fill_(1)
torch.cuda.synchronize()
print(f'RESIDENT {block.numel() * block.element_size() // (1024 * 1024)} MiB', flush=True)
time.sleep(float(sys.argv[2]))
"""

CHILD = """
import json, sys
sys.path.insert(0, r'%s')
from PIL import Image
from ui.tabs.img_editor import selection_events as callbacks
from ui.tabs.img_editor import state
photo = Image.open(r'%s').convert('RGB')
session = state.SelectionSession()
result = callbacks.on_analyze(photo, session)
overlay, choices, out_session, message = result
print('EVIDENCE_JSON ' + json.dumps({
    'message': message,
    'has_overlay': overlay is not None,
    'choices_visible': getattr(choices, 'visible', None),
    'session_type': type(out_session).__name__,
}))
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    reserve_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 6.5
    holder = subprocess.Popen(
        [sys.executable, '-c', HOLDER, str(reserve_gb), '600'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        print('HOLDER', holder.stdout.readline().strip(), flush=True)
        time.sleep(4)
        env = dict(os.environ)
        env['PYTHONPATH'] = str(ROOT)
        code = CHILD % (ROOT, ROOT / 'venv/Lib/site-packages/skimage/data/coffee.png')
        proc = subprocess.run([sys.executable, '-c', code], env=env, cwd=str(ROOT),
                              capture_output=True, text=True, timeout=600)
        result = {'returncode': proc.returncode}
        for raw in proc.stdout.splitlines():
            if raw.startswith('EVIDENCE_JSON '):
                result['result'] = json.loads(raw[len('EVIDENCE_JSON '):])
        result['stderr_tail'] = proc.stderr[-1200:]
        (OUT / 'ui_busy_gpu.json').write_text(
            json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps(result.get('result'), indent=2, ensure_ascii=False))
        print(f'RETURNCODE {proc.returncode}')
    finally:
        holder.kill()
        holder.wait(timeout=30)


if __name__ == '__main__':
    main()
