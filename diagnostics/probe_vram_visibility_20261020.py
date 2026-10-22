"""Check whether torch sees VRAM held by another process on this driver.

A holder process allocates a block and writes to it, so the pages become
resident.  The observer process then compares its own mem_get_info reading
with nvidia-smi, which is the only reliable source for other processes.

WDDM can accept an allocation that is not resident in VRAM, so the resident
write is what makes the comparison meaningful.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def nvidia_used_mb() -> int:
    out = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=30,
    )
    return int(out.stdout.strip().splitlines()[0])


def main() -> None:
    reserve_gb = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
    import torch

    before_nvidia = nvidia_used_mb()
    before_torch = int(torch.cuda.mem_get_info(0)[0] // (1024 * 1024))
    holder = subprocess.Popen(
        [sys.executable, '-c', HOLDER, str(reserve_gb), '300'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        print('HOLDER', holder.stdout.readline().strip(), flush=True)
        time.sleep(4)
        after_nvidia = nvidia_used_mb()
        after_torch = int(torch.cuda.mem_get_info(0)[0] // (1024 * 1024))
        report = {
            'reserve_gb': reserve_gb,
            'nvidia_used_before_mb': before_nvidia,
            'nvidia_used_after_mb': after_nvidia,
            'nvidia_delta_mb': after_nvidia - before_nvidia,
            'torch_free_before_mb': before_torch,
            'torch_free_after_mb': after_torch,
            'torch_delta_mb': before_torch - after_torch,
        }
        out = ROOT / 'diagnostics/out/vision_guard_busy_gpu_20261020' / f'vram_probe_{reserve_gb}.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))
    finally:
        holder.kill()
        holder.wait(timeout=30)


if __name__ == '__main__':
    main()
