"""Measure the real composite mask around brief FaceSwap changes in video 5.

This is a diagnostic render with a reconstructed target reference and a
specified source face. It never overwrites the user's output.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roop.globals  # noqa: E402
from diagnostics.probe_faceswap_video_recovery_render_20261107 import load_source_face  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.ProcessOptions import ProcessOptions  # noqa: E402
from roop.face_util import extract_face_images  # noqa: E402
from roop.types import FaceSet  # noqa: E402

VIDEO_NAME = "Aggressivebbc_20260313__2032353863555260759_1_20323536768541696000.mp4"
OUT = ROOT / "diagnostics/out/faceswap_tracking_20261119"


def frame_at(cap, index):
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {index}")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=425)
    parser.add_argument("--end", type=int, default=436)
    parser.add_argument("--reference-frame", type=int, default=425)
    parser.add_argument("--tag", default="inframe")
    parser.add_argument("--direct", action="store_true",
                        help="Medir la composición con detección real sin el gate de tracking")
    parser.add_argument("--source", type=Path,
                        default=ROOT / "testdata/test_face.webp")
    parser.add_argument("--scan", nargs="+", type=int,
                        help="Medir similitud de referencias candidatas sin render")
    args = parser.parse_args()
    if args.start < 0 or args.end <= args.start:
        raise ValueError("Rango de frames inválido")

    originals = list((ROOT / ".autodeep_temp").glob(f"*/{VIDEO_NAME}"))
    if len(originals) != 1:
        raise RuntimeError(f"Se esperaba un original, encontrados: {len(originals)}")
    video = originals[0]
    cap = cv2.VideoCapture(str(video))
    reference_image = frame_at(cap, args.reference_frame)
    faces = [entry[0] for entry in extract_face_images(
        reference_image, (False, args.reference_frame), is_source_face=False,
    )]
    faces = [face for face in faces if getattr(face, "embedding", None) is not None]
    if not faces:
        raise RuntimeError("No se detectó una cara de referencia")
    print("[DIAG] Caras detectadas:", [
        [float(v) for v in face.bbox] for face in faces
    ], flush=True)
    frame_h, frame_w = reference_image.shape[:2]
    in_frame = [face for face in faces if (
        face.bbox[0] >= 0 and face.bbox[1] >= 0
        and face.bbox[2] <= frame_w and face.bbox[3] <= frame_h
    )]
    if not in_frame and not args.direct:
        raise RuntimeError("Ninguna cara de referencia completa en el frame")
    target = max(faces if args.direct else in_frame, key=lambda face: float(
        (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
    ))
    if args.scan:
        scan = []
        scan_faces = {}
        comparer = ProcessMgr.__new__(ProcessMgr)
        for index in args.scan:
            frame = frame_at(cap, index)
            candidates = [entry[0] for entry in extract_face_images(
                frame, (False, index), is_source_face=False,
            )]
            scan_faces[index] = candidates
            scan.append({
                "frame": index,
                "faces": [{
                    "bbox": [float(v) for v in face.bbox],
                    "detection_score": float(getattr(face, "det_score", 0.0)),
                    "similarity": comparer._calculate_similarity(
                        target.embedding, getattr(face, "embedding", None),
                    ),
                } for face in candidates],
            })
        pairwise = []
        for index, candidates in scan_faces.items():
            matches = []
            for reference_index, reference_candidates in scan_faces.items():
                if reference_index == index:
                    continue
                scores = [
                    comparer._calculate_similarity(
                        getattr(face, "embedding", None),
                        getattr(reference, "embedding", None),
                    )
                    for face in candidates for reference in reference_candidates
                ]
                finite_scores = [float(score) for score in scores
                                 if score is not None and np.isfinite(score)]
                if finite_scores:
                    matches.append({
                        "reference_frame": reference_index,
                        "similarity": max(finite_scores),
                    })
            pairwise.append({
                "frame": index,
                "max_any_face_pairs": sorted(
                    matches, key=lambda item: item["similarity"], reverse=True,
                )[:5],
            })
        cap.release()
        OUT.mkdir(parents=True, exist_ok=True)
        report_path = OUT / f"video5_ref_scan_{args.reference_frame}.json"
        report_path.write_text(json.dumps({
            "reference_frame": args.reference_frame,
            "reference_bbox": [float(v) for v in target.bbox],
            "pairwise_note": "Maximum over every detected face pair; not identity confirmation",
            "scan": scan,
            "pairwise": pairwise,
        }, indent=2), encoding="utf-8")
        print(f"[DIAG] {report_path}", flush=True)
        return
    cap.release()

    roop.globals.selected_face_references = {
        f"selected_face_ref_{video.name}": [{
            "face_obj": target, "embedding": target.embedding,
            "bbox": target.bbox, "kps": getattr(target, "kps", None),
        }],
    }
    roop.globals.face_swap_mode = "selected_faces"
    roop.globals.processing = True
    source_path = args.source.resolve()
    if not source_path.is_file():
        raise RuntimeError(f"Fuente no disponible: {source_path}")
    source = load_source_face(source_path)
    options = ProcessOptions(
        {"face_swapper": {}, "mask_xseg": {}},
        getattr(roop.globals, "distance_threshold", 0.6),
        getattr(roop.globals, "blend_ratio", 0.95),
        "selected_faces", 0, "", None, 1, False, False,
        getattr(roop.globals, "use_enhancer", True),
        getattr(roop.globals, "blend_mode", "seamless"),
    )
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [], options)
    original_swap = manager._process_face_swap_v21
    records = []
    current = {"frame": None}

    def trace_swap(frame, event, value):
        if frame.f_code.co_name != "_process_face_swap_v21":
            return None
        if event != "return":
            return trace_swap
        local = frame.f_locals
        mask = local.get("composite_mask")
        box = getattr(local.get("target_face"), "bbox", None)
        record = {
            "call": len(records),
            "frame": current["frame"],
            "identity_status": (
                "unverified_forced_composition" if args.direct
                else "selected_mode_tracking"
            ),
            "partial_gate_rejected": current.get("partial_gate_rejected"),
            "detection_score": current.get("detection_score"),
            "bank_similarity": current.get("bank_similarity"),
            "frame_count": getattr(manager, f"_frame_count_{video.name}", None),
            "bbox": None if box is None else [float(v) for v in box],
            "has_composite": mask is not None,
            "occ_fraction": local.get("occ_frac"),
            "occ_applied": local.get("occ_applied"),
            "mouth_paste_strength": local.get("mouth_paste_strength"),
            "mouth_blend_factor": local.get("mouth_blend_factor_ref"),
            "mouth_force_paste": local.get("mouth_force_paste"),
        }
        if mask is not None and box is not None:
            alpha = np.asarray(mask[:, :, 0], dtype=np.float32)
            frame_h, frame_w = alpha.shape
            x1 = max(0, min(frame_w, int(np.floor(box[0]))))
            y1 = max(0, min(frame_h, int(np.floor(box[1]))))
            x2 = max(0, min(frame_w, int(np.ceil(box[2]))))
            y2 = max(0, min(frame_h, int(np.ceil(box[3]))))
            visible_area = max(0, x2 - x1) * max(0, y2 - y1)
            record["visible_bbox_area"] = visible_area
            if visible_area:
                visible_alpha = alpha[y1:y2, x1:x2]
                record["pre_paste_alpha_mean_visible_bbox"] = float(
                    np.mean(visible_alpha)
                )
                record["pre_paste_alpha_support_visible_bbox"] = float(
                    np.count_nonzero(visible_alpha > 0.15) / visible_area
                )
        records.append(record)
        return trace_swap

    def traced_swap(*swap_args, **swap_kwargs):
        previous = sys.gettrace()
        sys.settrace(trace_swap)
        try:
            return original_swap(*swap_args, **swap_kwargs)
        finally:
            sys.settrace(previous)

    manager._process_face_swap_v21 = traced_swap
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / f"video5_diag_{args.start}_{args.end}_{args.tag}.mp4"
    if args.direct:
        manager.options.current_video_path = str(video)
        manager._current_video_key = f"video_{video.name}"
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (frame_w, frame_h),
        )
        if not writer.isOpened():
            raise RuntimeError("No se pudo abrir el vídeo de diagnóstico")
        for index in range(args.start, args.end):
            frame = frame_at(cap, index)
            detected = [entry[0] for entry in extract_face_images(
                frame, (False, index), is_source_face=False,
            )]
            detected = [face for face in detected if getattr(face, "embedding", None) is not None]
            if detected:
                face = max(detected, key=lambda item: float(
                    (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
                ))
                current["frame"] = index
                current["detection_score"] = float(getattr(face, "det_score", 0.0))
                current["bank_similarity"] = manager._bank_best_sim(
                    face.embedding, video.name,
                )
                current["partial_gate_rejected"] = manager._recovery_face_clipped(
                    face, frame.shape, video.name,
                )
                result = traced_swap(source, face, frame.copy(), frame,
                                     enable_temporal_smoothing=True)
                records[-1]["changed_pixels"] = int(np.count_nonzero(
                    np.max(cv2.absdiff(result, frame), axis=2) > 12
                ))
            else:
                result = frame
            writer.write(result)
        writer.release()
        cap.release()
    else:
        for _percent, _message in manager.run_batch_inmem(
            str(video), str(output), start_frame=args.start,
            end_frame=args.end, skip_audio=True, temporal_smoothing=True,
        ):
            pass
    manager.Release()
    report = {
        "video": str(video),
        "output": str(output),
        "reference_frame": args.reference_frame,
        "reference_bbox": [float(v) for v in target.bbox],
        "source": str(source_path),
        "mode": "direct" if args.direct else "tracked",
        "limitations": (
            "Direct mode forces composition on the largest detected face per frame; "
            "identity and tracking are not verified. Alpha is measured before "
            "any oral paste, inside the visible part of the detected bbox."
            if args.direct else
            "Target bank is reconstructed, not the original UI selection. "
            "Alpha is measured before any oral paste."
        ),
        "records": records,
    }
    report_path = OUT / f"video5_diag_{args.start}_{args.end}_{args.tag}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[DIAG] {report_path}", flush=True)


if __name__ == "__main__":
    main()
