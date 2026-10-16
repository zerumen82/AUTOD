"""CPU regressions for the approved 2026-10-13 repairs; no real sampler.

--baseline executes the actual HEAD class definitions against the same tests.
Network, GPU release, face discovery and sampling are scripted fixtures only.
"""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
editor_module = importlib.import_module("roop.img_editor.img_editor_manager")
process_module = importlib.import_module("roop.ProcessMgr")
body_fixture = importlib.import_module("diagnostics.probe_editor_body_clause_guard")


def selected_class(module, path, name):
    source = ((subprocess.check_output(["git", "show", "HEAD:" + path], cwd=ROOT)
               .decode("utf-8")) if "--baseline" in sys.argv
              else (ROOT / path).read_text(encoding="utf-8"))
    hashes[path] = hashlib.sha256(source.encode()).hexdigest()
    tree = ast.parse(source)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    # Use the real module namespace so patches of network/VRAM helpers affect
    # extracted methods too. The replacement lives only in this probe process.
    namespace = vars(module)
    exec(compile(ast.Module(body=[node], type_ignores=[]), path, "exec"), namespace)
    return namespace[name]


hashes = {}
Editor = selected_class(editor_module, "roop/img_editor/img_editor_manager.py", "ImgEditorManager")
Process = selected_class(process_module, "roop/ProcessMgr.py", "ProcessMgr")


def mask(box, size=(240, 160)):
    image = Image.new("L", size)
    image.paste(255, box)
    return image


def compound_all_regions():
    manager = Editor()
    left, right = mask((20, 30, 50, 70)), mask((180, 30, 210, 70))
    with patch.object(manager, "_measured_subject_passes", return_value=[{"mask": left}, {"mask": right}]):
        union = manager._compound_clause_zone(Image.new("RGB", left.size), "people change", "", {}, [])
    px = np.asarray(union)
    assert px[45, 30] == 255 and px[45, 190] == 255, "second measured person discarded"
    assert px[45, 100] == 0, "unmeasured gap must stay excluded"


class InputCopyClient:
    def __init__(self, color=None):
        self.calls = []
        self.color = color
    def load(self, _version):
        return True, "fixture load"
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        output = kwargs["image"].copy()
        if self.color:
            output.paste(self.color, (0, 0, output.width, output.height))
        return SimpleNamespace(image=output), "fixture generation"


def body_anchor():
    manager = Editor()
    manager._prompt_authority_facts = body_fixture._facts
    manager._compose_generation_prompt = lambda text, **_kw: "EDIT " + text
    photo = body_fixture._photo()
    passes, faces = body_fixture._passes(photo)
    client = InputCopyClient()
    body_fixture._run(manager, photo, passes[:1], faces, client)
    assert client.calls and all(c["mask_image"] is not None for c in client.calls)
    assert all(c["no_kontext"] is False for c in client.calls), "measured body part lost spatial anchor"


def body_honest_noop():
    manager = Editor()
    manager._prompt_authority_facts = body_fixture._facts
    manager._compose_generation_prompt = lambda text, **_kw: "EDIT " + text
    photo = body_fixture._photo()
    passes, faces = body_fixture._passes(photo)
    client = InputCopyClient()
    output, message, produced = body_fixture._run(manager, photo, passes[:1], faces, client)
    assert produced and len(client.calls) == 2
    assert np.array_equal(np.asarray(output), np.asarray(photo)), "no-op changed source pixels"
    assert "sin cambio medido en la zona pedida" in message, "no-op falsely reported applied"


def body_guard_complete():
    manager = Editor()
    manager._prompt_authority_facts = body_fixture._facts
    manager._compose_generation_prompt = lambda text, **_kw: "EDIT " + text
    body_fixture.scenario_collateral_and_retry(manager)
    body_fixture.scenario_request_never_lands(manager)
    body_fixture.scenario_no_op(manager)


def compound_pose_geometry():
    manager = Editor()
    source = Image.new("RGB", (240, 160), "white")
    old_person = mask((15, 30, 65, 100))
    client = InputCopyClient("blue")
    def facts(raw, translated):
        pose = "stand" in (raw + translated).lower()
        return {"authority": True, "pose": pose, "body": False,
                "body_direction": "change", "body_intensity": 0.0,
                "local_appearance": False, "local_region": not pose,
                "region_ambiguous": False, "surface_recolor": not pose}
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=None),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_prompt_authority_facts", side_effect=facts),
          patch.object(manager, "_compound_clause_zone", return_value=old_person),
          patch("roop.img_editor.prompt_translator.translate_prompt", return_value="the man stands up, the wall is blue"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client)):
        output, message, _ = manager.generate_intelligent(
            image=source, prompt="the man stands up, the wall is blue", engine="longcat")
    assert output is not None, message
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["preserve_frame"] is False, "local sibling clause freezes compound pose"
    assert call["mask_image"] is None, "old person silhouette clips requested new geometry"
    assert call["keep_source_pixels"] is False
    assert output.getpixel((190, 45)) == (0, 0, 255), "new geometry outside old silhouette was discarded"


def cut_temporal_masks():
    for soft in (False, True):
        manager = Process.__new__(Process)
        manager._mark_unassigned = lambda *a, **kw: None
        manager._open_reacquire_window = lambda *a, **kw: None
        one, other = "video_first.mp4", "video_second.mp4"
        manager._prev_final_mask = {one: np.ones((3, 3)), other: np.full((3, 3), 2)}
        manager._mask_boost_ema_first = "unrelated attribute"
        setattr(manager, "_mask_boost_ema_first.mp4", 1.4)
        setattr(manager, "_mask_boost_ema_second.mp4", 1.8)
        manager._reset_tracking_on_scene_cut("first.mp4", 20, soft=soft)
        assert (one in manager._prev_final_mask) is soft, "hard cut reuses final mask / soft cut loses it"
        assert hasattr(manager, "_mask_boost_ema_first.mp4") is soft, "hard cut reuses boost / soft cut loses it"
        assert np.all(manager._prev_final_mask[other] == 2), "another video's mask was removed"
        assert getattr(manager, "_mask_boost_ema_second.mp4") == 1.8
        assert manager._mask_boost_ema_first == "unrelated attribute"


results = []
for test in (compound_all_regions, body_anchor, body_honest_noop, compound_pose_geometry,
             cut_temporal_masks, body_guard_complete):
    try:
        test()
        row = {"test": test.__name__, "pass": True}
    except Exception as error:
        row = {"test": test.__name__, "pass": False, "error": repr(error)}
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
print(json.dumps({"baseline": "--baseline" in sys.argv, "source_hashes": hashes,
                  "results": results}, ensure_ascii=False, indent=2))
sys.exit(0 if all(r["pass"] for r in results) else 1)
