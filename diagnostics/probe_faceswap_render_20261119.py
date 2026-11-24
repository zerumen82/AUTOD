"""Renderiza segmentos reales de los dos vídeos con fallos de tracking 2026-11-19."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roop.globals  # noqa: E402
from diagnostics.probe_faceswap_tracking_20261119 import (  # noqa: E402
    FALSE_LOG, PARTIAL_LOG, detected_face, logged_bbox, video_from_log,
)
from diagnostics.probe_faceswap_video_recovery_render_20261107 import load_source_face  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.ProcessOptions import ProcessOptions  # noqa: E402
from roop.types import FaceSet  # noqa: E402

OUT = ROOT / "diagnostics/out/faceswap_tracking_20261119"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("false", "partial", "return"))
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--trace-jumps", action="store_true")
    parser.add_argument("--trace-swap-returns", action="store_true")
    parser.add_argument("--trace-target", action="store_true")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--unseeded", action="store_true")
    parser.add_argument("--distance", type=float, default=None)
    args = parser.parse_args()
    if args.case == "false":
        log, start, end = FALSE_LOG, 185, 310
        reference_frames = (0, 189)
    elif args.case == "partial":
        log, start, end = PARTIAL_LOG, 2109, 2142
        reference_frames = (2114, 2121)
    else:
        from diagnostics.probe_faceswap_tracking_20261119 import RETURN_LOG
        log, start, end = RETURN_LOG, 800, 831
        reference_frames = (0, 1)
    if args.end is not None:
        end = args.end

    video = video_from_log(log)
    if args.case == "false":
        reference_bboxes = (logged_bbox(log, 1), logged_bbox(log, 190))
    elif args.case == "partial":
        reference_bboxes = (logged_bbox(log, 2121), logged_bbox(log, 2121))
    else:
        reference_bboxes = (logged_bbox(log, 1), logged_bbox(log, 2))
    reference_faces = [
        detected_face(video, i, bbox)[0]
        for i, bbox in zip(reference_frames, reference_bboxes)
    ]
    roop.globals.selected_face_references = {
        f"selected_face_ref_{video.name}": [
            {"face_obj": face, "embedding": face.embedding, "bbox": face.bbox,
             "kps": getattr(face, "kps", None)}
            for face in reference_faces
        ],
    }
    roop.globals.face_swap_mode = "selected_faces"
    roop.globals.processing = True
    source = load_source_face(ROOT / "testdata/test_face.webp")
    options = ProcessOptions(
        {"face_swapper": {}, "mask_xseg": {}},
        args.distance if args.distance is not None else getattr(roop.globals, "distance_threshold", 0.6),
        getattr(roop.globals, "blend_ratio", 0.95),
        "selected_faces", 0, "", None, 1, False, False,
        getattr(roop.globals, "use_enhancer", True),
        getattr(roop.globals, "blend_mode", "seamless"),
    )
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [], options)
    if args.trace_target:
        original_find = manager._find_target_face_for_selected_mode
        original_clipped = manager._recovery_face_clipped
        watched_lines = {3146, 3203, 3215, 3361, 3410, 3472, 3486, 3560,
                         3601, 3610, 3861, 3895, 3923, 3947, 4031}

        def trace_target(frame, event, value):
            if frame.f_code.co_name != '_find_target_face_for_selected_mode':
                return None
            local = frame.f_locals
            count = local.get('frame_count', 0)
            if count in (4, 6) and event == 'line' and frame.f_lineno in watched_lines:
                candidate = local.get('best_match', local.get('_f'))
                bbox = getattr(candidate, 'bbox', None)
                print(
                    f"[TRACE_TARGET] frame={count} line={frame.f_lineno} "
                    f"bbox={bbox} score={local.get('best_score')} "
                    f"min={local.get('min_track_score')} identity={local.get('identity_ok')} "
                    f"spatial={local.get('spatial_lock')} clipped={local.get('partial_unverified')} "
                    f"weak={local.get('weak_partial_track')} near_edge={local.get('partial_near_edge')} "
                    f"det={getattr(candidate, 'det_score', None)} "
                    f"bank={manager._bank_best_sim(getattr(candidate, 'embedding', None), video.name) if candidate is not None else None}",
                    flush=True,
                )
            return trace_target

        def traced_find(*find_args, **find_kwargs):
            previous_trace = sys.gettrace()
            sys.settrace(trace_target)
            try:
                return original_find(*find_args, **find_kwargs)
            finally:
                sys.settrace(previous_trace)

        manager._find_target_face_for_selected_mode = traced_find

        def trace_clipped(frame, event, value):
            if frame.f_code.co_name == '_recovery_face_clipped' and event == 'return':
                count = getattr(manager, f'_frame_count_{video.name}', 0)
                if count in (4, 6):
                    local = frame.f_locals
                    print(
                        f"[TRACE_CLIPPED] frame={count} line={frame.f_lineno} result={value} "
                        f"bbox={getattr(local.get('face'), 'bbox', None)} "
                        f"visible={local.get('visible_fraction')} overflow={local.get('overflow_w'), local.get('overflow_h')} "
                        f"inside={local.get('inside')} coherent={local.get('coherent')} "
                        f"bank={local.get('bank_sim')} track={local.get('track_sim')} "
                        f"anchor={local.get('anchor')} limit={local.get('limit')}",
                        flush=True,
                    )
            return trace_clipped

        def traced_clipped(*clip_args, **clip_kwargs):
            previous_trace = sys.gettrace()
            sys.settrace(trace_clipped)
            try:
                return original_clipped(*clip_args, **clip_kwargs)
            finally:
                sys.settrace(previous_trace)

        manager._recovery_face_clipped = traced_clipped
    if args.trace_swap_returns:
        original_swap = manager._process_face_swap_v21
        recent_lines = []

        def trace_return(frame, event, value):
            if frame.f_code.co_name == '_process_face_swap_v21':
                if event == 'line':
                    recent_lines.append(frame.f_lineno)
                    if len(recent_lines) > 20:
                        recent_lines.pop(0)
                if event == 'return':
                    print(f"[TRACE_SWAP_RETURN] lines={recent_lines}", flush=True)
                return trace_return
            return None

        def traced_swap(*swap_args, **swap_kwargs):
            previous_trace = sys.gettrace()
            recent_lines.clear()
            sys.settrace(trace_return)
            try:
                return original_swap(*swap_args, **swap_kwargs)
            finally:
                sys.settrace(previous_trace)

        manager._process_face_swap_v21 = traced_swap
    if args.trace_jumps:
        original_guard = manager._unverified_reacquire_jump

        def traced_guard(face, name, similarity):
            result = original_guard(face, name, similarity)
            bbox = getattr(face, 'bbox', None)
            if bbox is not None and bbox[0] > 500:
                print(
                    f"[TRACE_JUMP] bbox={tuple(int(v) for v in bbox)} "
                    f"anchor={manager._recovery_anchor(name)} sim={similarity:.3f} "
                    f"bank={manager._bank_best_sim(face.embedding, name)} "
                    f"frame={getattr(manager, f'_frame_count_{name}', None)} "
                    f"visual_cut={getattr(manager, f'_last_visual_cut_frame_{name}', None)} "
                    f"rejected={result}", flush=True,
                )
            return result

        manager._unverified_reacquire_jump = traced_guard
    if args.case == "partial":
        anchor = logged_bbox(log, 1753)
        manager._last_swap_bbox = {video.name: anchor}
        setattr(manager, f"_tracking_embedding_{video.name}", reference_faces[0].embedding)
        setattr(manager, f"_position_history_{video.name}", [
            tuple((anchor[:2] + anchor[2:]) * 0.5)
        ])
        setattr(manager, f"_reacquire_fail_count_{video.name}", 267)
    elif args.case == "return" and not args.unseeded:
        manager._recovery_anchor_bbox = {video.name: logged_bbox(log, 181)}
        setattr(manager, f"_reference_face_selected_{video.name}", True)
        setattr(manager, f"_user_selected_face_{video.name}", True)
        setattr(manager, f"_original_embedding_{video.name}", reference_faces[0].embedding)
        setattr(manager, f"_tracking_embedding_{video.name}", reference_faces[0].embedding)
        setattr(manager, f"_reacquire_fail_count_{video.name}", 586)

    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{args.case}_{start}_{end}{args.tag}.mp4"
    for _percent, _message in manager.run_batch_inmem(
        str(video), str(target), start_frame=start, end_frame=end,
        skip_audio=True, temporal_smoothing=True,
    ):
        pass
    manager.Release()
    print(f"[RENDER] {target} ({target.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
