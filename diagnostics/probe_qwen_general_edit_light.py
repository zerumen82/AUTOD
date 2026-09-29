"""Measure ordinary Qwen image editing without the camera LoRA on a benign photo."""

import io
import time
from pathlib import Path

import requests
from PIL import Image

from roop.comfy_workflows import get_comfyui_url
from roop.img_editor.comfy_progress import wait_for_comfy_image
from roop.img_editor.flux_edit_comfy_client import _sync_unload
from roop.img_editor.qwen_camera_comfy_client import QwenCameraComfyClient


ROOT = Path(__file__).resolve().parents[1]


def main():
    url = get_comfyui_url()
    source = Image.open(ROOT / "diagnostics/img_editor_scene_office_final.png").convert("RGB")
    source.thumbnail((768, 768), Image.Resampling.LANCZOS)
    _sync_unload(url, "prueba Qwen edición general")
    memory = io.BytesIO()
    source.save(memory, "PNG")
    memory.seek(0)
    uploaded = requests.post(f"{url}/upload/image", files={
        "image": ("qwen_general_edit_office.png", memory, "image/png")}, timeout=60)
    uploaded.raise_for_status()
    prompt = (
        "Change only the office walls to blue. Keep the same people, their faces, "
        "clothes, body positions, furniture, windows, plant and camera view."
    )
    workflow = QwenCameraComfyClient._workflow(uploaded.json()["name"], prompt, 123456)
    workflow["4"]["inputs"]["model"] = ["2", 0]
    del workflow["3"]
    started = time.time()
    queued = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=30)
    queued.raise_for_status()
    payload = queued.json()
    assert "prompt_id" in payload, payload
    print(f"QUEUED {payload['prompt_id']} {source.size}", flush=True)
    meta, note = wait_for_comfy_image(url, payload["prompt_id"], timeout=300, steps_hint=4)
    assert meta is not None, note
    response = requests.get(f"{url}/view", params={
        "filename": meta["filename"], "subfolder": meta.get("subfolder", ""),
        "type": meta.get("type", "output")}, timeout=60)
    response.raise_for_status()
    result = ROOT / "diagnostics/editor_qwen_general_edit_light.png"
    result.write_bytes(response.content)
    print(f"SAVED {result} {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
