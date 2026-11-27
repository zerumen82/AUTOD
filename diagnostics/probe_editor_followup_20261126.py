"""Run a second real UI edit on an already rendered, non-explicit image."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from ui.tabs import comfy_launcher
from ui.tabs.img_editor.events import on_generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    source = Image.open(args.source).convert("RGB")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    try:
        result, message, mask = on_generate(
            source, args.prompt, "auto", 0.0, False, 1, False, "hd",
            source="diagnostic_neutral_followup",
        )
        if result is None:
            raise RuntimeError(message)
        result.save(args.output / "result.png")
        if mask is not None:
            mask.save(args.output / "mask.png")
        before = np.asarray(source, dtype=np.int16)
        after = np.asarray(result.convert("RGB"), dtype=np.int16)
        changed = np.max(np.abs(before - after), axis=2) > 8
        report = {
            "source": str(args.source.resolve()),
            "output": str((args.output / "result.png").resolve()),
            "prompt": args.prompt,
            "message": message,
            "elapsed_s": time.monotonic() - start,
            "changed_px": int(changed.sum()),
        }
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    finally:
        process = comfy_launcher.comfy_process
        if process is not None and process.poll() is None:
            try:
                queue = requests.get(
                    f"{comfy_launcher.get_comfy_url()}/queue", timeout=5,
                ).json()
            except requests.RequestException:
                queue = None
            if queue is not None and not queue.get("queue_running") and not queue.get("queue_pending"):
                process.terminate()
                process.wait(timeout=15)


if __name__ == "__main__":
    main()
