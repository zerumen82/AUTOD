"""CPU regression: Chroma uses its T5 tokenizer and true classifier-free guidance.

Reference: lodestones/Chroma1-HD/ComfyUI_Chroma1-HD_T2I-workflow.json.
No model weights are loaded and no request is sent to ComfyUI.
"""
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor.flux_gen_comfy_client import FluxGenComfyClient  # noqa: E402


def capture(alias, cfg=None, tokenizer=None):
    client = FluxGenComfyClient()
    with patch.object(client, "_model_exists", return_value=True):
        loaded, message = client.load(alias)
    assert loaded, message
    if tokenizer is not None:
        client._model_configs[alias]["tokenizer"] = tokenizer
    captured = []

    def intercept(_url, **kwargs):
        captured.append(kwargs["json"]["prompt"])
        raise RuntimeError("workflow intercepted without inference")

    with (
        patch.object(client, "is_available", return_value=True),
        patch.object(client, "_prepare_prompt", side_effect=lambda prompt: prompt),
        patch("roop.img_editor.flux_gen_comfy_client.get_comfyui_url", return_value="http://localhost"),
        patch("roop.img_editor.flux_gen_comfy_client.requests.post", side_effect=intercept),
    ):
        client.generate("a wooden cabin in a pine forest", use_ai=False,
                        guidance_scale=cfg, lora_name="None", seed=1234)
    assert captured, "workflow not built"
    return captured[0], client


def main():
    client = FluxGenComfyClient()
    checked = []

    def without_clip_l(subdir, filename):
        checked.append((subdir, filename))
        return filename != "clip_l.safetensors"

    with patch.object(client, "_model_exists", side_effect=without_clip_l):
        loaded, message = client.load("chroma")
    assert loaded, f"Chroma incorrectly requires CLIP-L: {message}; {checked}"

    for cfg in (None, 5.2, 1.0):
        wf, client = capture("chroma", cfg)
        assert wf["3"]["class_type"] == "CLIPLoaderGGUF", wf["3"]
        assert wf["3"]["inputs"] == {
            "clip_name": client._clip_name, "type": "chroma",
        }, wf["3"]
        tokenizer = next((node for node in wf.values()
                          if node["class_type"] == "T5TokenizerOptions"), None)
        assert tokenizer is not None, "missing Chroma tokenizer options"
        assert tokenizer["inputs"] == {"clip": ["3", 0], "min_padding": 1, "min_length": 0}, tokenizer
        tokenizer_id = next(key for key, node in wf.items() if node is tokenizer)
        assert wf["6"]["inputs"]["clip"] == [tokenizer_id, 0], wf["6"]
        assert wf["7"]["inputs"]["clip"] == [tokenizer_id, 0], wf["7"]
        assert wf["8"]["inputs"]["cfg"] == (3.8 if cfg is None else cfg), wf["8"]
        assert not any(node["class_type"] == "FluxGuidance" for node in wf.values()), wf
    print("PASS Chroma: T5 only, tokenizer, configured CFG and explicit overrides")

    wf, _ = capture("chroma", tokenizer={"min_padding": 2, "min_length": 64})
    tokenizer = next(node for node in wf.values() if node["class_type"] == "T5TokenizerOptions")
    assert tokenizer["inputs"]["min_padding"] == 2, tokenizer
    assert tokenizer["inputs"]["min_length"] == 64, tokenizer
    print("PASS Chroma: tokenizer settings come from model configuration")

    wf, _ = capture("flux_dev_q4", 4.5)
    assert wf["3"]["class_type"] == "DualCLIPLoaderGGUF", wf["3"]
    assert wf["3"]["inputs"]["type"] == "flux", wf["3"]
    assert wf["14"]["inputs"]["guidance"] == 4.5, wf["14"]
    assert wf["8"]["inputs"]["cfg"] == 1.0, wf["8"]
    assert not any(node["class_type"] == "T5TokenizerOptions" for node in wf.values())
    print("PASS FLUX: dual loader and distilled guidance preserved")

    wf, _ = capture("lustify4", 5.5)
    assert wf["2"]["class_type"] == "CheckpointLoaderSimple", wf["2"]
    assert wf["8"]["inputs"]["cfg"] == 5.5, wf["8"]
    assert not any(node["class_type"] in ("FluxGuidance", "T5TokenizerOptions") for node in wf.values())
    print("PASS SDXL: checkpoint and sampler guidance preserved")


if __name__ == "__main__":
    main()
