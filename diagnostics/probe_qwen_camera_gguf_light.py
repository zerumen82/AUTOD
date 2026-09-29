"""Measure Qwen camera with the installed Q4 vision encoder at 768px."""

import io
import sys
import time
from pathlib import Path

import requests
from PIL import Image

from roop.img_editor.qwen_camera_comfy_client import QwenCameraComfyClient
from roop.img_editor.flux_edit_comfy_client import _sync_unload
from roop.comfy_workflows import get_comfyui_url
from roop.img_editor.comfy_progress import wait_for_comfy_image


ROOT = Path(__file__).resolve().parents[1]


def main():
    side = int(sys.argv[1]) if len(sys.argv) > 1 else 768
    url = get_comfyui_url()
    _sync_unload(url, "prueba Qwen codificador GGUF")
    source = Image.open(ROOT / "diagnostics/img_editor_scene_office_final.png").convert("RGB")
    source.thumbnail((side, side), Image.Resampling.LANCZOS)
    memory = io.BytesIO()
    source.save(memory, "PNG")
    memory.seek(0)
    uploaded = requests.post(f"{url}/upload/image", files={
        "image": ("qwen_camera_gguf_light.png", memory, "image/png")}, timeout=60)
    uploaded.raise_for_status()
    workflow = QwenCameraComfyClient._workflow(
        uploaded.json()["name"], "<sks> front view high-angle shot wide shot", 123456)
    # Bypass FluxKontextImageScale: it expands even a 640px source to ~1MP.
    workflow["10"]["inputs"]["pixels"] = ["1", 0]
    workflow["11"]["inputs"]["image1"] = ["1", 0]
    workflow["12"]["inputs"]["image1"] = ["1", 0]
    del workflow["9"]
    workflow["7"] = {"class_type": "CLIPLoaderGGUF", "inputs": {
        "clip_name": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf", "type": "qwen_image"}}
    started = time.time()
    queued = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=30)
    queued.raise_for_status()
    data = queued.json()
    assert "prompt_id" in data, data
    print(f"QUEUED {data['prompt_id']} {source.size}", flush=True)
    meta, note = wait_for_comfy_image(url, data["prompt_id"], timeout=300, steps_hint=4)
    assert meta is not None, note
    response = requests.get(f"{url}/view", params={
        "filename": meta["filename"], "subfolder": meta.get("subfolder", ""),
        "type": meta.get("type", "output")}, timeout=60)
    response.raise_for_status()
    result = ROOT / f"diagnostics/editor_qwen_camera_native_{side}.png"
    result.write_bytes(response.content)
    with Image.open(result) as rendered:
        print(f"SAVED {result} {rendered.size} {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
