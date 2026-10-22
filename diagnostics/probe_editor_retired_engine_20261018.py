"""CPU contracts for retired LongCat identifiers and real Kontext workflow.

Only HTTP I/O is intercepted. Model inventory, load selection and workflow
construction execute production code; no inference or UI changes occur.
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import flux_edit_comfy_client as editor  # noqa: E402
from roop.img_editor import gen_prompt_config as prompts  # noqa: E402


class WorkflowCaptured(Exception):
    pass


def http_get(url, **_kwargs):
    if url.endswith("/system_stats"):
        data = {"devices": [{"vram_free": 8 * 1024**3, "vram_total": 8 * 1024**3}]}
    elif url.endswith("/object_info/UnetLoaderGGUF"):
        data = {"UnetLoaderGGUF": {}}
    else:
        raise AssertionError(f"Unexpected HTTP read: {url}")
    return SimpleNamespace(status_code=200, json=lambda: data)


def main():
    aliases = (
        None, "kontext", "imagine",
        "LongCat-Image-Edit-Turbo-Q4_K_S.gguf",
        "LongCat-Image-Edit-Q4_K_S.gguf",
        "longcat", "longcat_turbo", "longcat_full", "LongCat Turbo",
        "LONGCAT-Image-Edit-Q8_0.gguf",
    )
    with (
        patch.object(editor, "get_comfyui_url", return_value="http://cpu-probe.invalid"),
        patch.object(editor.requests, "get", side_effect=http_get),
        patch.object(editor.requests, "post") as post,
    ):
        post.return_value = SimpleNamespace(status_code=200, json=lambda: {"ok": True})
        client = editor.FluxEditComfyClient()
        for alias in aliases:
            loaded, message = client.load(alias)
            assert loaded, message
            assert client._is_kontext_dev, f"Retired engine still active: {alias} -> {client._flux_version}"
            assert not client._is_longcat_turbo and not client._is_qwen_image_edit
            assert client._clip_type == "flux" and client._dual_clip
        configured = prompts.get_kontext_model_config()
        assert client._flux_version == configured["model_file"]
        assert client._model_exists("diffusion_models", configured["model_file"])
        print(f"PASS: {len(aliases)} retired identifiers select installed {client._flux_version}")

        captured = []

        def intercept(url, **kwargs):
            if url.endswith("/prompt"):
                captured.append(kwargs["json"]["prompt"])
                raise WorkflowCaptured
            assert url.endswith("/upload/image"), url
            return SimpleNamespace(status_code=200, json=lambda: {"name": "cpu_probe.png"})

        with patch.object(editor.requests, "post", side_effect=intercept):
            try:
                client.generate(
                    image=Image.new("RGB", (128, 128), "blue"),
                    prompt="Make the jacket green", seed=42,
                    num_inference_steps=configured["steps"],
                    guidance_scale=configured["cfg"],
                )
            except WorkflowCaptured:
                pass
        assert len(captured) == 1
        workflow = captured[0]
        assert workflow["2"]["inputs"]["unet_name"] == configured["model_file"]
        assert workflow["3"]["class_type"] == "DualCLIPLoaderGGUF"
        assert workflow["3"]["inputs"]["type"] == "flux"
        assert workflow["6g"]["inputs"]["guidance"] == configured["cfg"]
        assert workflow["8"]["inputs"]["steps"] == configured["steps"]
        assert not any("qwen" in node["class_type"].lower() for node in workflow.values())
        assert not any("longcat" in json.dumps(node).lower() for node in workflow.values())
        print(f"PASS: native Kontext workflow uses FluxGuidance={configured['cfg']}, steps={configured['steps']}")

        for version in (
            "Qwen_Image_Edit-Q2_K.gguf", "flux-2-klein-base-4b-Q4_K_S.gguf",
            "flux-2-klein-9b-Q4_K_S.gguf",
            "ggml-model-Q4_K_M.gguf",
        ):
            loaded, message = client.load(version)
            assert loaded, message
            assert not client._is_kontext_dev
            loaded, message = client.load("longcat_full")
            assert loaded, message
            assert client._is_kontext_dev and client._dual_clip
            assert not any((client._is_longcat_turbo, client._is_qwen_image_edit,
                            client._is_qwen_q5, client._is_klein_4b, client._is_klein_9b))
        print("PASS: Qwen/Klein/Fill remain loadable; switching back clears all engine flags")

        with tempfile.TemporaryDirectory(prefix="editor_kontext_config_") as directory:
            config_dir = Path(directory) / "config"
            config_dir.mkdir()
            configured["model_file"] = "flux1-kontext-dev-Q2_K.gguf"
            configured["steps"] = 24
            configured["cfg"] = 4.1
            (config_dir / "model_configs.json").write_text(
                json.dumps({"kontext": configured}), encoding="utf-8",
            )
            with patch.object(prompts, "_project_root", return_value=directory):
                loaded, message = client.load("longcat")
                assert loaded, message
                assert client._flux_version == configured["model_file"]
                assert prompts.get_kontext_model_config()["cfg"] == configured["cfg"]
                assert editor._appearance_sampling_config()[0] == configured["steps"]
        print("PASS: model file and sampling follow configuration changes")

    with patch.object(editor, "get_kontext_model_config", return_value={}):
        try:
            editor._appearance_sampling_config()
        except KeyError:
            pass
        else:
            raise AssertionError("Missing Kontext configuration silently used a legacy recipe")
    print("PASS: missing Kontext recipe cannot fall back to LongCat")


if __name__ == "__main__":
    main()
