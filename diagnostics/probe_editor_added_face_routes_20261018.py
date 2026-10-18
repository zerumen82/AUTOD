"""CPU flow regressions for extra faces in measured compounds/subject passes."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


SIZE = (240, 160)
BASE_FACE = SimpleNamespace(bbox=[15, 90, 45, 130], det_score=0.95)
EXTRA_FACE = SimpleNamespace(bbox=[170, 20, 210, 60], det_score=0.95)


def mask(rect):
    arr = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    x1, y1, x2, y2 = rect
    arr[y1:y2, x1:x2] = 255
    return Image.fromarray(arr)


def photo():
    return Image.fromarray(np.random.default_rng(12).integers(40, 200, (160, 240, 3), dtype=np.uint8))


def facts(text):
    pose = "dancing" in text
    addition = "Add a person" in text
    recolor = "blue" in text or "green" in text
    return {
        "authority": pose, "pose": pose, "body": False,
        "body_direction": "change", "body_intensity": 0.0,
        "local_appearance": not (pose or addition or recolor),
        "local_region": addition or recolor,
        "surface_recolor": recolor, "scenario": False,
        "structural_remove": False, "region_ambiguous": False,
    }


def detected(image):
    arr = np.asarray(image)
    # Sentinel lies in full-photo coordinates, including for local crops.
    added = np.any((arr[:, :, 0] == 251) & (arr[:, :, 1] == 1))
    return [BASE_FACE, EXTRA_FACE] if added else [BASE_FACE]


class Client:
    def __init__(self, persist=False, fail_retry=False, resize=False, first_clean=False):
        self.calls = []
        self.persist = persist
        self.fail_retry = fail_retry
        self.resize = resize
        self.first_clean = first_clean

    def load(self, *_args, **_kwargs):
        return True, "loaded"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        retry = "no face may be added" in kwargs["prompt"]
        if retry and self.fail_retry:
            return None, "inference failed"
        arr = np.asarray(kwargs["image"].convert("RGB")).copy()
        arr[:, :, 2] = 220
        if (not retry or self.persist) and not (self.first_clean and len(self.calls) == 1):
            # Full-frame test uses a right-hand crop for the second subject.
            if arr.shape[1] == SIZE[0]:
                arr[20:60, 170:210] = [251, 1, 200]
            elif arr.shape[1] == 120:
                arr[20:60, 50:90] = [251, 1, 200]
        image = Image.fromarray(arr)
        if self.resize:
            image = image.resize((image.width // 2, image.height // 2), Image.Resampling.NEAREST)
        return SimpleNamespace(image=image), "rendered"


def integrate(source, generated, region, box, **_kwargs):
    """Deterministic measured paste; generator and detection stay controlled."""
    candidate = source.copy()
    candidate.paste(generated.resize((box[2] - box[0], box[3] - box[1])), box[:2])
    return Image.composite(candidate, source, region)


def patches(manager):
    stack = ExitStack()
    for name, value in (
        ("_prompt_authority_facts", facts),
        ("_detect_edit_faces", detected),
        ("_release_local_gpu_before_comfy", lambda: None),
        ("_get_clipseg_masker", lambda: None),
        ("_edit_is_nearly_unchanged", lambda *_a: False),
        ("_second_focused_pass", lambda *_a, **_k: None),
        ("_repair_inserted_surface", lambda _c, _i, _m, updated, *_a: (updated, "")),
        ("_integrate_local_generation", integrate),
    ):
        if name == "_prompt_authority_facts":
            value = lambda raw, translated: facts(translated or raw)
        stack.enter_context(patch.object(manager, name, side_effect=value))
    return stack


def compound(client, prompt="The man is bald, the walls are blue", cancelled=None):
    manager = ImgEditorManager()
    source = photo()
    region = mask((120, 0, 240, 160))
    with (
        patches(manager),
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", return_value=SimpleNamespace(get=lambda _i: [BASE_FACE])),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8),
        patch("roop.img_editor.prompt_translator.translate_prompt", side_effect=lambda text: text),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
        patch.object(manager, "_explicit_viewpoint_request", return_value=False),
        patch.object(manager, "_compound_clause_zone", return_value=region),
        patch.object(manager, "_appearance_work_region", side_effect=lambda image, zone: (image, zone, (0, 0, *SIZE))),
        patch.object(manager, "_degenerate_faces_mask", return_value=None),
    ):
        output, message, _ = manager.generate_intelligent(
            source, prompt, engine="longcat", use_semantic=False,
            enhance_faces=False, cancel_check=cancelled,
        )
    return source, output, message


def assert_restored(source, output):
    src, dst = np.asarray(source), np.asarray(output)
    assert np.array_equal(src[28:52, 178:202], dst[28:52, 178:202]), "extra face not restored"
    assert np.any(src[100:130, 140:165] != dst[100:130, 140:165]), "legitimate edit lost"
    assert np.array_equal(src[:, :100], dst[:, :100]), "unrequested region changed"


def subjects(client, text="The walls are blue", two=False, cancelled=None, no_op=False):
    manager = ImgEditorManager()
    source = photo()
    passes = [{"text": text, "mask": mask((120, 0, 240, 160))}]
    if two:
        passes.insert(0, {"text": "The left wall is green", "mask": mask((0, 0, 100, 160))})

    def work(image, zone, _person=None):
        box = zone.getbbox()
        return image.crop(box), zone.crop(box), box

    with (
        patches(manager),
        patch.object(manager, "_appearance_work_region", side_effect=work),
        patch.object(manager, "_edit_is_nearly_unchanged", return_value=no_op),
    ):
        output, message, produced = manager._apply_subject_passes(
            client, source, passes, 12, "", None, 1.0, None,
            cancelled, localize=True, faces=[BASE_FACE],
        )
    assert produced or output is None, message
    return source, output, message


def main():
    manager = ImgEditorManager()
    assert manager._instruction_preserves_face_layout("The man is bald")
    assert manager._instruction_preserves_face_layout("Recolor the wall blue")
    for instruction in (
        "The man is dancing", "Add a person near the door",
        "The man is bald and add a person near the door",
        "Change the background to a beach",
    ):
        assert not manager._instruction_preserves_face_layout(instruction), instruction
    print("PASS actual semantic gates: appearance/recolor versus pose, add and scenario")
    for runner in (compound, subjects):
        client = Client()
        _, output, message = runner(client)
        assert output is not None, message
        assert len(client.calls) == 2, f"{runner.__name__}: extra face did not trigger retry ({len(client.calls)})"
        assert "no face may be added" in client.calls[1]["prompt"]
        assert len(detected(output)) == 1
        for failed in (False, True):
            client = Client(persist=True, fail_retry=failed)
            source, output, message = runner(client)
            assert len(client.calls) == 2, (runner.__name__, len(client.calls), message)
            assert_restored(source, output)
        print(f"PASS {runner.__name__}: retry, persistent restoration and failed retry cleanup")

    for prompt in (
        "The man is dancing, the walls are blue",
        "The walls are blue, Add a person near the door",
    ):
        client = Client()
        _, output, _ = compound(client, prompt)
        assert len(client.calls) == 1, (prompt, len(client.calls))
        assert len(detected(output)) == 2, "legitimate changed composition was rejected"
    client = Client()
    subjects(client, "Add a person near the door")
    assert len(client.calls) == 1
    print("PASS exemptions: pose, mixed add-person compound and structural subject pass")

    client = Client(persist=True)
    source, output, _ = subjects(client, two=True)
    arr, src = np.asarray(output), np.asarray(source)
    assert len(client.calls) == 3, len(client.calls)
    assert np.all(arr[:, :100, 2] == 220), "second subject overwrote the first edit"
    assert np.array_equal(arr[28:52, 178:202], src[28:52, 178:202])
    print("PASS multiple subjects: previous edits and full-frame restoration preserved")

    client = Client(persist=True, first_clean=True)
    source, output, _ = subjects(client, no_op=True)
    assert len(client.calls) == 2, "no-op retry must not trigger a third render"
    assert_restored(source, output)
    client = Client(persist=True, resize=True)
    source, output, _ = subjects(client)
    assert len(client.calls) == 2
    assert_restored(source, output)
    print("PASS subject retry: new face created during no-op retry and resized crop cleaned")

    for runner in (compound, subjects):
        client = Client(persist=True)
        _, output, message = runner(client, cancelled=lambda: bool(client.calls))
        assert output is None and "Cancelado" in message, (runner.__name__, message)
        assert len(client.calls) == 1, "cancellation must prevent the extra render"
    print("PASS cancellation: no extra generation or delivered candidate")


if __name__ == "__main__":
    main()
