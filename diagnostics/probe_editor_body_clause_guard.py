"""Body clauses paste only the change the request asks for, and verify it.

Regression for edit_1791454352 (2026-10-08): the compound body pass pasted
the whole measured person, so the model's collateral (a dress added to the
barefoot woman, a swimsuit on another person) entered the photo while the
requested change (bare feet) never arrived, and the whole-frame no-op gate
never retried because the collateral itself counted as a change.

Contract checked here, with a scripted client and a scripted segmenter
(no generator, no GPU):
  1. The paste keeps measured change only inside the requested zone
     (region the segmenter gave the clause on the original photo, plus the
     pixels where the request exists in the generated frame).
  2. A first attempt whose change misses the requested zone triggers the
     retry, even when the frame changed elsewhere.
  3. When the request never lands, the message says so instead of silently
     delivering the collateral.
"""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager

WIDTH, HEIGHT = 240, 160
BG = (250, 250, 250)
SKIN = (214, 172, 140)
HEEL = (20, 20, 20)
DRESS = (200, 30, 160)
WOMAN_CLOTH = (60, 90, 200)
MAN_CLOTH = (40, 160, 60)
DARK_CLOTH = (90, 60, 40)
SWIM = (220, 40, 40)

# Production now sends local windows. Keep fixture frames and segmenter
# coordinates in the same window, rather than squeezing a whole photo into it.
_IMAGE_BOXES = {}


def _image_box(image):
    entry = _IMAGE_BOXES.get(id(image))
    if entry is not None:
        registered, box = entry
        assert registered is image
        return box
    assert image.size == (WIDTH, HEIGHT), ("unregistered fixture crop", image.size)
    return (0, 0, WIDTH, HEIGHT)

HEELS_BOX = (40, 132, 70, 150)
DRESS_BOX = (30, 60, 80, 115)
MAN_TORSO_BOX = (115, 60, 180, 140)
DARK_TORSO_BOX = (195, 65, 233, 145)


def _fill(img, box, color):
    px = img.load()
    x1, y1, x2, y2 = box
    for y in range(y1, y2):
        for x in range(x1, x2):
            px[x, y] = color
    return img


def _photo():
    img = Image.new("RGB", (WIDTH, HEIGHT), BG)
    _fill(img, (20, 30, 90, 150), WOMAN_CLOTH)
    _fill(img, (35, 30, 70, 60), SKIN)
    _fill(img, HEELS_BOX, HEEL)
    _fill(img, (110, 25, 185, 150), MAN_CLOTH)
    _fill(img, (130, 25, 165, 55), SKIN)
    _fill(img, (190, 35, 235, 150), DARK_CLOTH)
    _fill(img, (200, 35, 230, 62), SKIN)
    return img


def _region(img, box, color):
    out = img.copy()
    return _fill(out, box, color)


def _region2(img, box1, color1, box2, color2):
    out = _region(img, box1, color1)
    return _fill(out, box2, color2)


def _box_all(img, box, color, tol=2):
    arr = np.asarray(img.convert("RGB"), dtype=np.int16)
    x1, y1, x2, y2 = box
    target = np.array(color, dtype=np.int16)
    return bool((np.abs(arr[y1:y2, x1:x2] - target).max(axis=2) <= tol).all())


def _box_dominant(img, box, target, other):
    """The box mean sits closer to target than to other.

    The paste feathers with a Gaussian blur, so a narrow zone blends a few
    percent of the source color. The requested state still has to dominate
    the result clearly.
    """
    arr = np.asarray(img.convert("RGB"), dtype=np.float64)
    x1, y1, x2, y2 = box
    mean = arr[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)
    to_target = float(np.abs(mean - np.array(target, dtype=np.float64)).max())
    to_other = float(np.abs(mean - np.array(other, dtype=np.float64)).max())
    if to_target >= to_other:
        print(f"DEBUG box={box} mean={mean.tolist()} d_target={to_target:.1f} d_other={to_other:.1f}")
        return False
    return True


class _Face:
    def __init__(self, bbox):
        self.bbox = bbox


class _Masker:
    """The segmenter answers any query with pixels, content dependent."""

    def generate_mask(self, image, prompt, threshold=0.4):
        arr = np.zeros((image.height, image.width), np.uint8)
        text = (prompt or "").strip(" .,!?\"'").lower()
        if "barefoot" in text or "descalz" in text:
            x1, y1, x2, y2 = HEELS_BOX
            left, top, right, bottom = _image_box(image)
            local_x1, local_y1 = max(0, x1-left), max(0, y1-top)
            local_x2, local_y2 = min(right-left, x2-left), min(bottom-top, y2-top)
            if local_x2 > local_x1 and local_y2 > local_y1:
                arr[local_y1:local_y2, local_x1:local_x2] = 255
        elif "naked" in text or "desnud" in text:
            rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
            skin = (
                (rgb[..., 0] > 180)
                & (rgb[..., 1] > 130) & (rgb[..., 1] < 210)
                & (rgb[..., 2] > 100) & (rgb[..., 2] < 190)
                & ((rgb[..., 0] - rgb[..., 2]) > 25)
            )
            arr[skin] = 255
        return Image.fromarray(arr, "L")


class _ScriptedClient:
    """Returns pre-drawn frames per clause and attempt."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.attempts = {}
        self.calls = []

    def load(self, version):
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        prompt = str(kwargs.get("prompt", "")).lower()
        if "barefoot" in prompt or "descalz" in prompt:
            key = "barefoot"
        elif "naked" in prompt or "desnud" in prompt:
            key = "naked"
        else:
            raise AssertionError(f"prompt sin clave conocida: {prompt}")
        step = self.attempts.get(key, 0)
        self.attempts[key] = step + 1
        frames = self.behavior[key]
        frame = frames[min(step, len(frames) - 1)]
        box = _image_box(kwargs["image"])
        generated = frame.crop(box)
        assert generated.size == kwargs["image"].size
        _IMAGE_BOXES[id(generated)] = (generated, box)
        return SimpleNamespace(image=generated), f"OK {key} intento {step}"


def _facts(_raw, _translated):
    return {
        "authority": True,
        "pose": False,
        "body": True,
        "body_direction": "change",
        "body_intensity": 0.9,
        "local_appearance": False,
        "local_region": False,
        "surface_recolor": False,
    }


def _passes(photo):
    woman_mask = Image.new("L", photo.size, 0)
    woman_mask.paste(255, (20, 25, 90, 150))
    man_mask = Image.new("L", photo.size, 0)
    # The group measurement for "the man" reaches both men, as in the real
    # incident where the other person's torso was repainted too.
    man_mask.paste(255, (110, 20, 235, 150))
    faces = [
        _Face((35, 30, 70, 60)),
        _Face((130, 25, 165, 55)),
        _Face((200, 35, 230, 62)),
    ]
    return [
        {"text": "the woman goes barefoot", "mask": woman_mask},
        {"text": "the man goes naked", "mask": man_mask},
    ], faces


def _run(manager, photo, passes, faces, client):
    _IMAGE_BOXES.clear()
    original_window = manager._appearance_work_region

    def fixture_window(*args, **kwargs):
        window, mask, box = original_window(*args, **kwargs)
        if box is not None:
            _IMAGE_BOXES[id(window)] = (window, box)
        return window, mask, box

    with (patch.object(manager, "_get_clipseg_masker", return_value=_Masker()),
          patch.object(manager, "_appearance_work_region", side_effect=fixture_window)):
        return manager._apply_subject_passes(
            client, photo, passes,
            seed=1, negative_prompt="", lora_name="", lora_strength=0,
            progress_callback=None, cancel_check=None, faces=faces,
        )


def scenario_collateral_and_retry(manager):
    photo = _photo()
    dress_only = _region(photo, DRESS_BOX, DRESS)
    dressed_and_bare = _region2(photo, DRESS_BOX, DRESS, HEELS_BOX, SKIN)
    naked = _region2(
        photo, MAN_TORSO_BOX, SKIN, DARK_TORSO_BOX, SWIM,
    )
    client = _ScriptedClient({
        "barefoot": [dress_only, dressed_and_bare],
        "naked": [naked],
    })
    passes, faces = _passes(photo)
    output, message, produced = _run(manager, photo, passes, faces, client)
    assert produced, message
    assert output is not None and output.size == photo.size, message

    # The requested changes arrived: retry for bare feet, first pass for naked.
    # The asserted cores sit beyond the 6 px feather of the paste.
    assert client.attempts.get("barefoot") == 2, client.attempts
    assert client.attempts.get("naked") == 1, client.attempts
    assert _box_dominant(output, (50, 137, 60, 141), SKIN, HEEL), (
        "los pies no quedaron descalzos"
    )
    assert _box_dominant(output, (130, 75, 165, 130), SKIN, MAN_CLOTH), (
        "el torso desnudo no se pegó"
    )

    # The collateral never enters: her dress, the other person's swimsuit.
    assert _box_all(output, DRESS_BOX, WOMAN_CLOTH), (
        "el vestido añadido por el modelo se pegó"
    )
    assert _box_all(output, DARK_TORSO_BOX, DARK_CLOTH), (
        "el bañador pintado en la otra persona se pegó"
    )
    assert _box_all(output, (5, 5, 6, 6), BG), "el fondo cambió"
    assert "sin cambio medido en la zona pedida" not in message, message
    print("OK: cuerpo pegado solo en la zona pedida, con reintento medido")


def scenario_request_never_lands(manager):
    photo = _photo()
    dress_only = _region(photo, DRESS_BOX, DRESS)
    client = _ScriptedClient({"barefoot": [dress_only, dress_only]})
    passes, faces = _passes(photo)
    output, message, produced = _run(
        manager, photo, passes[:1], faces, client,
    )
    assert produced, message
    assert client.attempts.get("barefoot") == 2, client.attempts
    assert _box_all(output, DRESS_BOX, WOMAN_CLOTH), (
        "el vestido añadido se pegó aunque la petición no llegó"
    )
    assert _box_all(output, HEELS_BOX, HEEL), (
        "los tacones no cambiaron: no hay nada que entregar"
    )
    assert "sin cambio medido en la zona pedida" in message, message
    print("OK: petición sin cumplir -> foto conservada y nota honesta")


def scenario_no_op(manager):
    photo = _photo()
    client = _ScriptedClient({"barefoot": [photo, photo]})
    passes, faces = _passes(photo)
    output, message, produced = _run(
        manager, photo, passes[:1], faces, client,
    )
    assert produced, message
    assert client.attempts.get("barefoot") == 2, client.attempts
    assert (np.asarray(output) == np.asarray(photo)).all(), (
        "un pase sin cambios alteró la foto"
    )
    assert "sin cambio medido en la zona pedida" in message, message
    print("OK: pase sin cambios -> reintento y nota honesta")


def main():
    manager = ImgEditorManager()
    manager._prompt_authority_facts = _facts
    manager._compose_generation_prompt = lambda text, **_kw: f"EDIT {text}"
    scenario_collateral_and_retry(manager)
    scenario_request_never_lands(manager)
    scenario_no_op(manager)
    print("PASS: probe_editor_body_clause_guard")


if __name__ == "__main__":
    main()
