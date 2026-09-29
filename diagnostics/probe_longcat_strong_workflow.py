"""Inspect strong LongCat workflow without contacting ComfyUI or using the GPU."""
from unittest.mock import patch

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient


class Response:
    def __init__(self, status_code):
        self.status_code = status_code
        self.text = "probe stopped before enqueue"

    def json(self):
        return {"error": {"message": self.text}}


client = FluxEditComfyClient()
client._loaded = True
client._flux_version = "LongCat-Image-Edit-Q4_K_S.gguf"
client._clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
client._clip_name2 = None
client._clip_type = "longcat_image"
client._vae_name = "ae.safetensors"
client._dual_clip = False
client._is_gguf_clip = False
captured = {}


def fake_post(url, **kwargs):
    if url.endswith("/prompt"):
        captured.update(kwargs["json"]["prompt"])
        return Response(400)
    return Response(200)


with patch("roop.img_editor.flux_edit_comfy_client.requests.post", side_effect=fake_post):
    result, _ = client.generate(
        Image.new("RGB", (800, 533), (120, 130, 140)),
        prompt="Edit this exact photo",
        num_inference_steps=36,
        guidance_scale=2.5,
        no_kontext=True,
        reference_latents_method="index",
    )

assert result is None
assert captured["6"]["inputs"]["image1"] == ["1", 0]
assert "image1" not in captured["7"]["inputs"]
assert captured["8"]["inputs"]["latent_image"] == ["5", 0]
assert captured["8"]["inputs"]["steps"] == 36
print("[PROBE] strong LongCat: imagen positiva única, negativo textual, foto como latente inicial", flush=True)
