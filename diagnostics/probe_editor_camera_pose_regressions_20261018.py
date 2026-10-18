"""CPU regression contract for trained camera axes, without GPU or a UI."""

from itertools import product
from unittest.mock import patch

from roop.img_editor.nlp.semantic_analyzer import CAMERA_POSES, camera_pose_config
from roop.img_editor.qwen_camera_comfy_client import camera_pose


def main():
    cases = [
        ("low angle shot", "low-angle shot"),
        ("high angle shot", "high-angle shot"),
        ("close-up shot", "close-up"),
        ("full body shot", "wide shot"),
        ("atrás izquierda", "back-left quarter view"),
        ("back left", "back-left quarter view"),
        ("atrás derecha", "back-right quarter view"),
        ("back right", "back-right quarter view"),
        ("front left", "front-left quarter view"),
        ("front right", "front-right quarter view"),
        ("camera angle", "front-right quarter view elevated shot medium shot"),
        ("view", "front-right quarter view elevated shot medium shot"),
        ("shot", "front-right quarter view elevated shot medium shot"),
    ]
    failed = []
    for prompt, expected in cases:
        pose = camera_pose(prompt)
        ok = expected in pose
        print(f"{'PASS' if ok else 'FAIL'} {prompt!r}: {pose}")
        if not ok:
            failed.append((prompt, expected, pose))
    assert not failed, failed
    print(f"OK: {len(cases)} camera pose regression checks")

    # The input is the LoRA's real trained vocabulary. Check every combination
    # to detect cross-axis contamination, rather than just the reported inputs.
    combinations = list(product(*(CAMERA_POSES[axis] for axis in (
        "azimuth", "elevation", "distance"))))
    for azimuth, elevation, distance in combinations:
        trained_pose = f"{azimuth} {elevation} {distance}"
        actual = camera_pose(trained_pose)
        assert actual == f"<sks> {trained_pose}", (trained_pose, actual)
    print(f"OK: {len(combinations)} trained camera pose combinations")

    # A configured measurement floor must actually affect selection. Restore
    # the production configuration on leaving the block.
    config = camera_pose_config()
    with patch.dict(config, {"signal_threshold": 1.0}):
        assert camera_pose("low angle shot") == (
            "<sks> front-right quarter view elevated shot medium shot")
    assert "low-angle shot" in camera_pose("low angle shot")
    print("OK: configured measurement threshold is applied")


if __name__ == "__main__":
    main()
