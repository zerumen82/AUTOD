"""Two real local projector loads/closes, run only after renders have finished."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.element_selection import _add_llama_dll_directories, load_settings  # noqa: E402
from roop.img_editor.instruction_verifier import InstructionVerifier  # noqa: E402


def free_mb():
    data = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
    return float(data.splitlines()[0].strip())


def main():
    _add_llama_dll_directories(load_settings())
    out = ROOT / "diagnostics/out/verifier_cleanup_20261024" / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    records = []
    verifier = InstructionVerifier()
    for _index in range(2):
        before = free_mb()
        try:
            assert verifier._load(), verifier._unavailable_reason
            loaded = free_mb()
        finally:
            verifier.unload()
        time.sleep(0.5)
        after = free_mb()
        record = {"before_free_mb": before, "loaded_free_mb": loaded, "after_free_mb": after,
                  "allocated_mb": before - loaded, "released_mb": after - loaded}
        records.append(record)
        (out / "metrics.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(json.dumps(record), flush=True)
        assert before > loaded, "projector did not allocate real GPU memory"
        assert after - loaded > (before - loaded) / 2, "most of the GPU allocation remained resident"
    allocation = records[0]["allocated_mb"]
    assert records[0]["after_free_mb"] - records[1]["after_free_mb"] < allocation / 4, (
        "VRAM accumulated across consecutive loads", records)
    print(f"PASS native projector cleanup; evidence={out}", flush=True)


if __name__ == "__main__":
    main()
