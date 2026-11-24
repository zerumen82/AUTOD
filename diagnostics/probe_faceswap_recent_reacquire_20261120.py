"""Reproduce la readquisición errónea del lote real del 20/11/2026.

El banco UI no se conservó: usa la cara elegida en f0 como referencia fija.
No modifica los vídeos de entrada ni las salidas entregadas.
"""

import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import roop.globals  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.face_util_rotation import get_all_faces_smart  # noqa: E402

LOGS = (
    ROOT / "logs/process_20261120_021602.log",
    ROOT / "logs/process_20261120_013428.log",
)
OUT = ROOT / "diagnostics/out/faceswap_latest_20261120/reacquire_probe.json"


def video_from_log(log):
    content = log.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"^Video: (.+)$", content, re.M)
    if match is None:
        raise RuntimeError(f"Log sin vídeo: {log}")
    video = Path(match.group(1).strip())
    if not video.is_file():
        raise RuntimeError(f"Vídeo no disponible: {video}")
    return video, content


def logged_bbox(content, frame, kind="Success"):
    if kind == "Success":
        pattern = rf"\[TRACK\] Frame {frame}: Success[^\n]*?bbox=\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)"
    else:
        pattern = rf"\[TRACK\] RE-BUSQUEDA closeup frame {frame}[^\n]*?bbox=\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)"
    match = re.search(pattern, content)
    if match is None:
        raise RuntimeError(f"Sin bbox {kind} del frame {frame}")
    return np.array([int(value) for value in match.groups()], dtype=np.float32)


def detected_face(video, index, logged_box):
    capture = cv2.VideoCapture(str(video))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"No se leyó {video.name} f{index}")
    faces = get_all_faces_smart(
        image, min_score=None, for_target=True, prefer_fast=True, include_180=False,
    )
    faces = [face for face in faces if getattr(face, "embedding", None) is not None]
    if not faces:
        raise RuntimeError(f"No se detectó una cara en {video.name} f{index}")
    center = (logged_box[:2] + logged_box[2:]) * 0.5
    return min(faces, key=lambda face: float(np.linalg.norm(
        (np.asarray(face.bbox[:2]) + np.asarray(face.bbox[2:])) * 0.5 - center
    )))


def manager_with_anchor(video, reference, anchor, frame, extra_references=(), fixed_face=None):
    manager = ProcessMgr.__new__(ProcessMgr)
    manager.options = None
    manager.sim_threshold = 0.28
    manager._recovery_anchor_bbox = {video.name: anchor}
    manager._last_swap_bbox = {video.name: anchor}
    if fixed_face is not None:
        embedding = np.asarray(fixed_face.embedding, dtype=np.float32)
        manager._selected_target_fixed_embedding = {
            video.name: embedding / np.linalg.norm(embedding),
        }
    setattr(manager, f"_frame_count_{video.name}", frame)
    setattr(manager, f"_last_visual_cut_frame_{video.name}", 0)
    roop.globals.selected_face_references[f"selected_face_ref_{video.name}"] = [
        {"embedding": face.embedding} for face in (reference, *extra_references)
    ]
    return manager


def main():
    roop.globals.selected_face_references = {}
    main_video, main_log = video_from_log(LOGS[0])
    second_video, second_log = video_from_log(LOGS[1])
    selected = detected_face(main_video, 0, logged_bbox(main_log, 1))
    profile = detected_face(main_video, 1631, logged_bbox(main_log, 1631, "RE-BUSQUEDA"))
    wrong = detected_face(main_video, 1699, logged_bbox(main_log, 1699, "RE-BUSQUEDA"))
    true_logged_box = logged_bbox(main_log, 1329, "RE-BUSQUEDA")
    true_return = copy.copy(detected_face(main_video, 1329, true_logged_box))
    # El detector del probe puede devolver una caja distinta a la de la corrida
    # original. Conservar la geometría registrada y el embedding facial real.
    true_return.bbox = true_logged_box
    later_return = detected_face(main_video, 1729, logged_bbox(main_log, 1730))
    second_selected = detected_face(second_video, 0, logged_bbox(second_log, 1))
    second_occluded = detected_face(
        second_video, 1661, logged_bbox(second_log, 1661, "RE-BUSQUEDA"),
    )

    wrong_manager = manager_with_anchor(
        main_video, selected, logged_bbox(main_log, 1664), 1699,
        extra_references=(profile,), fixed_face=selected,
    )
    wrong_rejected = wrong_manager._unverified_reacquire_jump(wrong, main_video.name, 0.33)
    wrong_route_rejected = wrong_manager._apply_reacquired_face(
        wrong, "_probe_assigned", "_probe_lost", "_probe_history", main_video.name,
    ) is None
    wrong_bank_best = wrong_manager._bank_best_sim(wrong.embedding, main_video.name)
    reversed_manager = manager_with_anchor(
        main_video, profile, logged_bbox(main_log, 1664), 1699,
        extra_references=(selected,), fixed_face=selected,
    )
    reversed_rejected = reversed_manager._unverified_reacquire_jump(
        wrong, main_video.name, 0.33,
    )
    reversed_route_rejected = reversed_manager._apply_reacquired_face(
        wrong, "_probe_assigned", "_probe_lost", "_probe_history", main_video.name,
    ) is None
    true_manager = manager_with_anchor(
        main_video, selected, logged_bbox(main_log, 1048), 1329,
    )
    true_kept = not true_manager._unverified_reacquire_jump(
        true_return, main_video.name, 0.30,
    )
    later_manager = manager_with_anchor(
        main_video, selected, logged_bbox(main_log, 1664), 1729,
    )
    later_kept = not later_manager._unverified_reacquire_jump(
        later_return, main_video.name, 0.35,
    )
    second_manager = manager_with_anchor(
        second_video, second_selected, logged_bbox(second_log, 1646), 1661,
    )
    occluded_kept = not second_manager._unverified_reacquire_jump(
        second_occluded, second_video.name, 0.41,
    )
    report = {
        "video": main_video.name,
        "reference_note": "Cara elegida f0 reconstruida; el banco UI original no se conserva",
        "wrong_frame": 1699,
        "wrong_detector_score": float(wrong.det_score),
        "wrong_fixed_similarity": wrong_bank_best,
        "wrong_primary_similarity": wrong_manager._calculate_similarity(
            selected.embedding, wrong.embedding,
        ),
        "true_1329_fixed_similarity": true_manager._bank_best_sim(
            true_return.embedding, main_video.name,
        ),
        "true_1729_fixed_similarity": later_manager._bank_best_sim(
            later_return.embedding, main_video.name,
        ),
        "second_1661_fixed_similarity": second_manager._bank_best_sim(
            second_occluded.embedding, second_video.name,
        ),
        "wrong_rejected": wrong_rejected,
        "wrong_apply_route_rejected": wrong_route_rejected,
        "wrong_reversed_bank_rejected": reversed_rejected,
        "wrong_reversed_route_rejected": reversed_route_rejected,
        "true_1329_kept": true_kept,
        "true_1729_kept": later_kept,
        "second_1661_kept": occluded_kept,
    }
    report["verdict"] = "PASS" if all((
        wrong_rejected, wrong_route_rejected, reversed_rejected,
        reversed_route_rejected, true_kept, later_kept, occluded_kept,
    )) else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
