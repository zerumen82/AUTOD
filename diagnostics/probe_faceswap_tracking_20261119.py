"""Comprueba con detecciones reales los dos fallos de tracking del lote 2026-11-19.

Lee los vídeos y los logs originales; no altera los MP4 del usuario.
"""

import copy
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import roop.globals  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.face_util import extract_face_images  # noqa: E402
from roop.face_util_rotation import get_all_faces_smart  # noqa: E402
from roop.faceswap_config import get_faceswap_config  # noqa: E402

FALSE_LOG = ROOT / "logs/process_20261119_015653.log"
PARTIAL_LOG = ROOT / "diagnostics/out/faceswap_tracking_20261119/baseline_partial_process.log"
RETURN_LOG = ROOT / "diagnostics/out/faceswap_tracking_20261119/baseline_video7_process.log"
BASELINE = json.loads((ROOT / "diagnostics/faceswap_tracking_20261119_baseline.json").read_text(encoding="utf-8"))
VIDEO5_LOG = ROOT / "logs/ui_start_20261118_235534.log"
VIDEO5_NAME = "Aggressivebbc_20260313__2032353863555260759_1_20323536768541696000.mp4"
OUT = ROOT / "diagnostics/out/faceswap_tracking_20261119/report.json"


def video_from_log(log):
    if not log.is_file() and log == FALSE_LOG:
        matches = list(ROOT.glob(f".autodeep_temp/*/{BASELINE['video']}"))
        if len(matches) != 1:
            raise RuntimeError(f"Se esperaba un vídeo original y hay {len(matches)}")
        return matches[0]
    match = re.search(r"^Video: (.+)$", log.read_text(encoding="utf-8", errors="replace"), re.M)
    if not match:
        raise RuntimeError(f"Log sin vídeo: {log}")
    path = Path(match.group(1).strip())
    if not path.is_file():
        raise RuntimeError(f"Vídeo no disponible: {path}")
    return path


def logged_bbox(log, frame):
    if not log.is_file() and log == FALSE_LOG:
        box = BASELINE["bboxes"].get(str(frame))
        if box is None:
            raise RuntimeError(f"Sin bbox preservada para el frame {frame}")
        return np.array(box, dtype=np.float32)
    pattern = rf"\[TRACK\].*?[Ff]rame {frame}\b[^\n]*?bbox=\(\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\)"
    match = re.search(pattern, log.read_text(encoding="utf-8", errors="replace"))
    if not match:
        raise RuntimeError(f"Sin bbox del frame {frame} en {log}")
    return np.array([int(match.group(i)) for i in range(1, 5)], dtype=np.float32)


def detected_face(video, index, target_bbox):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer {video.name} frame {index}")
    faces = [item[0] for item in extract_face_images(image, (False, index), is_source_face=False)]
    faces = [face for face in faces if getattr(face, "embedding", None) is not None]
    if not faces:
        raise RuntimeError(f"Sin detección facial en {video.name} frame {index}")
    target_center = (target_bbox[:2] + target_bbox[2:]) * 0.5
    face = min(faces, key=lambda f: np.linalg.norm(
        (np.asarray(f.bbox[:2]) + np.asarray(f.bbox[2:])) * 0.5 - target_center
    ))
    return face, image.shape


def video5_false_candidate():
    """Re-detect the background patch accepted by the old 640-first path."""
    text = VIDEO5_LOG.read_text(encoding="utf-8", errors="replace")
    start = text.index(f"[VIDEO] {VIDEO5_NAME}")
    segment = text[start:]
    match = re.search(
        r"RE-BUSQUEDA closeup frame 432 \(sim=([0-9.]+),[^\n]*?"
        r"bbox=\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)",
        segment,
    )
    anchor_match = re.search(
        r"\[TRACK\] RE-BUSQUEDA closeup frame 430 [^\n]*?"
        r"bbox=\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)",
        segment,
    )
    if match is None or anchor_match is None:
        raise RuntimeError("Faltan la recuperación falsa o la ancla en el log original")
    logged_similarity = float(match.group(1))
    logged_bbox = np.array([int(match.group(i)) for i in range(2, 6)], dtype=np.float32)
    anchor_bbox = np.array([int(anchor_match.group(i)) for i in range(1, 5)], dtype=np.float32)
    originals = list((ROOT / ".autodeep_temp").glob(f"*/{VIDEO5_NAME}"))
    if len(originals) != 1:
        raise RuntimeError(f"Se esperaba un vídeo 5 original, encontrados {len(originals)}")
    video = originals[0]
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 431)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("No se pudo leer el frame 431 del vídeo 5")
    faces = get_all_faces_smart(
        image, min_score=None, for_target=True,
        prefer_fast=True, include_180=False,
    )
    if not faces:
        raise RuntimeError("El detector rápido no encontró el falso candidato")
    center = (logged_bbox[:2] + logged_bbox[2:]) * 0.5
    face = min(faces, key=lambda item: float(np.linalg.norm(
        (np.asarray(item.bbox[:2]) + np.asarray(item.bbox[2:])) * 0.5 - center
    )))
    if float(np.linalg.norm(np.asarray(face.bbox) - logged_bbox)) > 8:
        raise RuntimeError("La detección ya no coincide con el bbox original")
    return video.name, face, logged_similarity, logged_bbox, anchor_bbox


def measured_visual_cuts(video, start, end):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    detector = ProcessMgr.__new__(ProcessMgr)
    cuts = []
    for index in range(start, end + 1):
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"No se pudo leer {video.name} frame {index}")
        is_cut, _difference = detector._detect_visual_scene_cut(
            image, video.name, index + 1,
        )
        if is_cut:
            cuts.append(index + 1)
    cap.release()
    return cuts


def main():
    false_video = video_from_log(FALSE_LOG)
    partial_video = video_from_log(PARTIAL_LOG)
    false_bbox = logged_bbox(FALSE_LOG, 270)
    last_false_bbox = logged_bbox(FALSE_LOG, 195)
    last_partial_bbox = logged_bbox(PARTIAL_LOG, 1753)
    partial_bbox = logged_bbox(PARTIAL_LOG, 2121)

    false_face, _ = detected_face(false_video, 269, false_bbox)
    original_face, _ = detected_face(false_video, 194, last_false_bbox)
    anchor_face, _ = detected_face(partial_video, 2114, partial_bbox)
    partial_face, frame_shape = detected_face(partial_video, 2120, partial_bbox)
    nearby_face, _ = detected_face(partial_video, 2121, partial_bbox)
    intermittent_faces = [
        detected_face(partial_video, frame, partial_bbox)
        for frame in (2112, 2114)
    ]
    false_visual_cuts = measured_visual_cuts(false_video, 185, 249)

    manager = ProcessMgr.__new__(ProcessMgr)
    manager.options = None
    manager.sim_threshold = 0.28
    manager._last_swap_bbox = {
        false_video.name: last_false_bbox,
        partial_video.name: last_partial_bbox,
    }
    # Reproduce el estado tras un swap real y los hard cuts del lote original.
    manager._recovery_anchor_bbox = dict(manager._last_swap_bbox)
    setattr(manager, f"_tracking_embedding_{partial_video.name}", anchor_face.embedding)
    setattr(manager, f"_position_history_{partial_video.name}", [
        tuple((last_partial_bbox[:2] + last_partial_bbox[2:]) * 0.5)
    ])
    setattr(manager, f"_reacquire_fail_count_{partial_video.name}", 267)
    roop.globals.selected_face_references = {
        f"selected_face_ref_{partial_video.name}": [{"embedding": nearby_face.embedding}],
    }
    manager._reset_tracking_on_scene_cut(str(false_video), 196, reason="probe hard cut")
    manager._reset_tracking_on_scene_cut(str(partial_video), 1754, reason="probe hard cut")
    setattr(manager, f"_frame_count_{false_video.name}", 270)
    setattr(manager, f"_last_visual_cut_frame_{false_video.name}",
            false_visual_cuts[-1] if false_visual_cuts else 0)
    false_cut_expired = all(
        250 - cut > int(get_faceswap_config()['reacquire_visual_cut_window_frames'])
        for cut in false_visual_cuts
    )

    # La similitud procede del log original: el banco de la UI de aquel lote
    # no se conserva. La detección y las bboxes sí se miden de nuevo en los MP4.
    if FALSE_LOG.is_file():
        similarity_match = re.search(
            r"RE-BUSQUEDA closeup frame 270 \(sim=([0-9.]+)",
            FALSE_LOG.read_text(encoding="utf-8", errors="replace"),
        )
        if similarity_match is None:
            raise RuntimeError("Sin similitud de la falsa recuperación en el log")
        logged_similarity = float(similarity_match.group(1))
    else:
        logged_similarity = float(BASELINE["similarity_270"])
    far_jump_rejected = manager._unverified_reacquire_jump(
        false_face, false_video.name, logged_similarity,
    )
    true_continuity_kept = not manager._unverified_reacquire_jump(
        original_face, false_video.name, logged_similarity,
    )
    partial_allowed = not manager._recovery_face_clipped(
        partial_face, frame_shape, partial_video.name,
    )
    intermittent_allowed = [
        not manager._recovery_face_clipped(face, shape, partial_video.name)
        for face, shape in intermittent_faces
    ]
    reacquired = manager._reacquire_face_global(
        [partial_face], partial_video.name, 2121,
        closeup_mode=True, frame_shape=frame_shape,
    )
    unrelated = copy.copy(partial_face)
    unrelated.embedding = false_face.embedding
    unrelated_rejected = manager._recovery_face_clipped(
        unrelated, frame_shape, partial_video.name,
    )
    invalid = copy.copy(partial_face)
    invalid.embedding = np.full_like(np.asarray(partial_face.embedding), np.nan)
    invalid_partial_rejected = manager._recovery_face_clipped(
        invalid, frame_shape, partial_video.name,
    )
    invalid_jump_rejected = manager._unverified_reacquire_jump(
        invalid, partial_video.name, 1.0,
    )
    return_video = video_from_log(RETURN_LOG)
    return_face, _ = detected_face(return_video, 810, logged_bbox(RETURN_LOG, 811))
    initially_selected_face, _ = detected_face(return_video, 0, logged_bbox(RETURN_LOG, 1))
    absent_scene_face, absent_shape = detected_face(return_video, 807, logged_bbox(RETURN_LOG, 811))
    roop.globals.selected_face_references[f"selected_face_ref_{return_video.name}"] = [
        {"face_obj": initially_selected_face,
         "embedding": initially_selected_face.embedding,
         "bbox": initially_selected_face.bbox},
    ]
    initial_match = manager._calculate_similarity(
        initially_selected_face.embedding, absent_scene_face.embedding,
    )
    unseeded_manager = ProcessMgr.__new__(ProcessMgr)
    unseeded_manager.options = None
    unseeded_manager._setup_selected_faces_frame_for_video(
        str(return_video), [absent_scene_face],
    )
    unrelated_single_not_anchored = (
        initial_match < unseeded_manager._destination_match_threshold(0.35)
        and not hasattr(unseeded_manager, f"_target_face_assigned_{return_video.name}")
        and not hasattr(unseeded_manager, f"_reference_face_selected_{return_video.name}")
    )
    unseeded_manager._setup_selected_faces_frame_for_video(
        str(return_video), [initially_selected_face],
    )
    selected_face_later_anchored = (
        getattr(unseeded_manager, f"_target_face_assigned_{return_video.name}", None)
        is initially_selected_face
    )
    loose_manager = ProcessMgr.__new__(ProcessMgr)
    loose_manager.options = SimpleNamespace(face_distance_threshold=0.9)
    loose_manager.sim_threshold = 0.28
    loose_manager._last_swap_bbox = {}
    loose_manager._setup_selected_faces_frame_for_video(
        str(return_video), [absent_scene_face],
    )
    loose_single_not_anchored = not hasattr(
        loose_manager, f"_target_face_assigned_{return_video.name}",
    )
    loose_single_not_reacquired = loose_manager._reacquire_face_global(
        [absent_scene_face], return_video.name, 808,
        after_scene_cut=True, frame_shape=absent_shape,
    ) is None
    loose_selected_reacquired = loose_manager._reacquire_face_global(
        [initially_selected_face], return_video.name, 809,
        after_scene_cut=True, frame_shape=absent_shape,
    ) is initially_selected_face
    roop.globals.selected_face_references.pop(f"selected_face_ref_{return_video.name}")
    return_visual_cuts = measured_visual_cuts(return_video, 780, 810)
    manager._recovery_anchor_bbox[return_video.name] = logged_bbox(RETURN_LOG, 181)
    setattr(manager, f"_frame_count_{return_video.name}", 811)
    setattr(manager, f"_last_visual_cut_frame_{return_video.name}",
            return_visual_cuts[-1] if return_visual_cuts else 0)
    return_sim_match = re.search(
        r"RE-BUSQUEDA closeup frame 811 \(sim=([0-9.]+)",
        RETURN_LOG.read_text(encoding="utf-8", errors="replace"),
    )
    if return_sim_match is None:
        raise RuntimeError("Sin similitud del retorno visual en el log")
    return_similarity = float(return_sim_match.group(1))
    visual_return_allowed = not manager._unverified_reacquire_jump(
        return_face, return_video.name, return_similarity,
    )
    setattr(manager, f"_last_visual_cut_frame_{return_video.name}", 0)
    same_jump_without_cut_rejected = manager._unverified_reacquire_jump(
        return_face, return_video.name, return_similarity,
    )
    manager.options = SimpleNamespace(face_distance_threshold=0.9)
    loose_ui_far_rejected = manager._unverified_reacquire_jump(
        false_face, false_video.name, logged_similarity,
    )
    loose_ui_unrelated_rejected = manager._recovery_face_clipped(
        unrelated, frame_shape, partial_video.name,
    )
    manager.options = None
    video5_name, video5_face, video5_similarity, video5_bbox, video5_anchor = (
        video5_false_candidate()
    )
    video5_manager = ProcessMgr.__new__(ProcessMgr)
    video5_manager.options = None
    video5_manager.sim_threshold = 0.28
    video5_manager._recovery_anchor_bbox = {video5_name: video5_anchor}
    setattr(video5_manager, f"_frame_count_{video5_name}", 432)
    setattr(video5_manager, f"_last_visual_cut_frame_{video5_name}", 0)
    # El banco UI se perdió: reproducir exactamente la similitud 0,56 del log,
    # mientras la bbox y la confianza proceden de la detección real actual.
    video5_manager._bank_best_sim = lambda _embedding, _name: video5_similarity
    video5_far_rejected = video5_manager._unverified_reacquire_jump(
        video5_face, video5_name, video5_similarity,
    )
    video5_reacquire_route_rejected = video5_manager._apply_reacquired_face(
        video5_face, "_probe_assigned", "_probe_lost", "_probe_history",
        video5_name,
    ) is None
    video5_near = copy.copy(video5_face)
    video5_near.bbox = video5_anchor.copy()
    video5_near_kept = not video5_manager._unverified_reacquire_jump(
        video5_near, video5_name, video5_similarity,
    )
    video5_invalid = copy.copy(video5_face)
    video5_invalid.det_score = float("nan")
    video5_invalid_rejected = video5_manager._unverified_reacquire_jump(
        video5_invalid, video5_name, video5_similarity,
    )
    setattr(video5_manager, f"_last_visual_cut_frame_{video5_name}", 431)
    video5_recent_cut_unchanged = not video5_manager._unverified_reacquire_jump(
        video5_face, video5_name, video5_similarity,
    )
    setattr(video5_manager, f"_last_visual_cut_frame_{video5_name}", 0)
    report = {
        "false_detection": {
            "frame": 270,
            "detected_bbox": np.asarray(false_face.bbox).tolist(),
            "last_real_bbox": last_false_bbox.tolist(),
            "logged_similarity": logged_similarity,
            "rejected_far_jump": far_jump_rejected,
            "near_real_kept": true_continuity_kept,
            "active_bbox_cleared_by_cut": false_video.name not in manager._last_swap_bbox,
            "loose_ui_still_rejected": loose_ui_far_rejected,
            "measured_visual_cuts_185_249": false_visual_cuts,
            "cut_expired_before_false_detection": false_cut_expired,
        },
        "partial_face": {
            "frame": 2121,
            "detected_bbox": np.asarray(partial_face.bbox).tolist(),
            "last_swap_bbox": last_partial_bbox.tolist(),
            "geometry_identity_allowed": partial_allowed,
            "intermittent_frames_2112_2114_allowed": intermittent_allowed,
            "intermittent_bboxes": [
                np.asarray(face.bbox).tolist() for face, _ in intermittent_faces
            ],
            "global_reacquire_accepted": reacquired is partial_face,
            "unrelated_identity_rejected": unrelated_rejected,
            "loose_ui_unrelated_rejected": loose_ui_unrelated_rejected,
            "invalid_embedding_rejected": invalid_partial_rejected and invalid_jump_rejected,
            "anchor_similarity": manager._calculate_similarity(anchor_face.embedding, partial_face.embedding),
            "nearby_similarity": manager._calculate_similarity(nearby_face.embedding, partial_face.embedding),
            "landmarks_inside": int(((np.asarray(partial_face.kps)[:, 0] >= 0)
                                      & (np.asarray(partial_face.kps)[:, 0] < frame_shape[1])
                                      & (np.asarray(partial_face.kps)[:, 1] >= 0)
                                      & (np.asarray(partial_face.kps)[:, 1] < frame_shape[0])).sum()),
            "geometry": manager._coherent_face_geometry(partial_face.kps, partial_face.bbox)[0],
        },
        "visual_cut_return": {
            "frame": 811,
            "logged_similarity": return_similarity,
            "detected_bbox": np.asarray(return_face.bbox).tolist(),
            "detector_score": float(return_face.det_score),
            "measured_visual_cuts_780_811": return_visual_cuts,
            "real_visual_cut_allowed": visual_return_allowed,
            "same_jump_without_cut_rejected": same_jump_without_cut_rejected,
            "single_unrelated_face_not_anchored": unrelated_single_not_anchored,
            "single_unrelated_similarity": initial_match,
            "selected_face_later_anchored": selected_face_later_anchored,
            "loose_ui_single_not_anchored": loose_single_not_anchored,
            "loose_ui_single_not_reacquired": loose_single_not_reacquired,
            "loose_ui_selected_reacquired": loose_selected_reacquired,
        },
        "video5_background_false_reacquire": {
            "video_frame": 431,
            "process_frame": 432,
            "logged_bbox": video5_bbox.tolist(),
            "detected_bbox": np.asarray(video5_face.bbox).tolist(),
            "anchor_bbox": video5_anchor.tolist(),
            "logged_bank_similarity": video5_similarity,
            "measured_detection_score": float(video5_face.det_score),
            "far_background_rejected": video5_far_rejected,
            "reacquire_route_rejected": video5_reacquire_route_rejected,
            "near_low_score_kept": video5_near_kept,
            "invalid_score_rejected": video5_invalid_rejected,
            "high_similarity_recent_cut_unchanged": video5_recent_cut_unchanged,
            "bank_note": "Original UI bank lost; similarity replayed from original log",
        },
    }
    report["verdict"] = "PASS" if all((
        far_jump_rejected, true_continuity_kept, loose_ui_far_rejected,
        false_video.name not in manager._last_swap_bbox,
        false_cut_expired,
        partial_allowed, all(intermittent_allowed), reacquired is partial_face,
        unrelated_rejected,
        loose_ui_unrelated_rejected, invalid_partial_rejected,
        invalid_jump_rejected, return_visual_cuts and return_visual_cuts[-1] == 811,
        visual_return_allowed,
        same_jump_without_cut_rejected,
        unrelated_single_not_anchored,
        selected_face_later_anchored,
        loose_single_not_anchored, loose_single_not_reacquired,
        loose_selected_reacquired,
        video5_far_rejected, video5_reacquire_route_rejected,
        video5_near_kept, video5_invalid_rejected,
        video5_recent_cut_unchanged,
    )) else "FAIL"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
