"""Render real acotado para verificar los rescates de vídeo v5.1319.

Reprocesa un tramo del vídeo real del lote 2026-11-07 con el código nuevo y
compara las etiquetas de guardas contra el log original de ese lote:

  - Fix A (kps degenerados): [ALIGN_REJECT] → [ALIGN_RECOVER] (swap continúa)
  - Fix B (máscara insuficiente): [MASK_SKIP] → [MASK_HOLD] (hold del frame anterior)

El lote original usó el banco de 634 facesets y la selección de destino de la
UI. Aquí se inyecta una referencia de destino detectada en el frame 0 (la misma
cara que ancló el track original, bbox (413,380,482,482)) y una sola cara de
origen de testdata. Las guardas viven en el detector/tracker, no en el banco,
así que los frames que fallaron en el lote vuelven a fallar aquí sin rescate.

Uso (desde la raíz del proyecto):
    python diagnostics/probe_faceswap_video_recovery_render_20261107.py --start 0 --end 900

Salidas en diagnostics/out/video_recovery_render_20261107/:
    <video>_<start>_<end>.mp4   render con el código nuevo
    report.json                 conteos baseline vs nuevo + PASS/FAIL
    sample_f*.png               frames de muestra (origen | salida | diff)
"""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_VIDEO = ROOT / (
    ".autodeep_temp/16c4ffe9724d5cc9409799cb96be691a5be1b81f2f69a07ff0f64d2ab7569283"
    "/VID_20261027_174936_599.mp4"
)
DEFAULT_BASELINE = ROOT / "logs/process_20261107_070133.log"
DEFAULT_SOURCE = ROOT / "testdata/test_face.webp"
# Cara anclada en el track del lote original (log: [TRACK] Frame 1).
ANCHOR_BBOX = (413.0, 380.0, 482.0, 482.0)
OUT_DIR = ROOT / "diagnostics/out/video_recovery_render_20261107"


def parse_log_counts(path: Path, lo: int, hi: int) -> dict:
    """Cuenta las etiquetas de guardas del log, acotadas por frame de TRACK."""
    counts = {"rej": 0, "recover": 0, "msk": 0, "hold": 0}
    frames = {"rej": [], "recover": [], "msk": [], "hold": []}
    # bbox real del tracker por frame: la cara SE MUEVE (frame 59 del vídeo 9
    # ya está en (525,396,591,497), lejos del ancla del frame 0), así que la
    # comprobación de píxeles no puede usar el bbox estático del arranque.
    track_bbox = {}
    last_track = -1
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = re.search(r"\[TRACK\] Frame (\d+):.*bbox=\((-?\d+),(-?\d+),(-?\d+),(-?\d+)\)", line)
            if match:
                last_track = int(match.group(1))
                track_bbox[last_track] = tuple(int(match.group(i)) for i in range(2, 6))
            if "[ALIGN_REJECT]" in line:
                key = "rej"
            elif "[ALIGN_RECOVER]" in line:
                key = "recover"
            elif "[MASK_SKIP]" in line:
                key = "msk"
            elif "[MASK_HOLD]" in line:
                key = "hold"
            else:
                continue
            if lo <= last_track < hi:
                counts[key] += 1
                frames[key].append(last_track)
    return {"counts": counts, "frames": frames, "track_bbox": track_bbox}


def build_destination_ref(video_path: Path, frame_idx: int = 0) -> "object":
    """Detecta la cara del frame 0 más cercana al ancla del lote original."""
    from roop.face_util import extract_face_images

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No se pudo leer el frame {frame_idx} de {video_path}")
    detected = extract_face_images(frame, (False, frame_idx), is_source_face=False)
    faces = [item[0] for item in detected]
    faces = [f for f in faces if getattr(f, "embedding", None) is not None]
    if not faces:
        raise RuntimeError("Frame 0 sin caras con embedding; no hay ancla de destino")
    anchor_cx = (ANCHOR_BBOX[0] + ANCHOR_BBOX[2]) / 2.0
    anchor_cy = (ANCHOR_BBOX[1] + ANCHOR_BBOX[3]) / 2.0

    def _dist(face):
        x1, y1, x2, y2 = face.bbox
        return ((x1 + x2) / 2.0 - anchor_cx) ** 2 + ((y1 + y2) / 2.0 - anchor_cy) ** 2

    face = min(faces, key=_dist)
    print(
        f"[REF] Ancla de destino: bbox={face.bbox} dist_centro={_dist(face) ** 0.5:.1f}px "
        f"de {len(faces)} cara(s) en frame {frame_idx}",
        flush=True,
    )
    return face


def load_source_face(source_path: Path):
    from roop.face_util import extract_face_images

    candidates = [source_path] + [
        ROOT / "testdata" / name
        for name in ("test_face.webp", "cpu_detected.jpg", "test_swap_result.jpg")
        if (ROOT / "testdata" / name).is_file()
    ]
    seen = set()
    for candidate in candidates:
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        try:
            faces = extract_face_images(str(candidate), (False, 0), is_source_face=True)
        except Exception as error:  # noqa: BLE001 - siguiente candidato
            print(f"[SOURCE] {candidate.name} omitido: {error}", flush=True)
            continue
        if faces and getattr(faces[0][0], "embedding", None) is not None:
            if candidate != source_path:
                print(f"[SOURCE] usando {candidate.name} como cara de origen",
                      flush=True)
            return faces[0][0]
        print(f"[SOURCE] {candidate.name} sin cara de origen válida", flush=True)
    raise RuntimeError(f"Sin cara de origen válida en {candidates}")


def sample_frames(frames: list, limit: int = 4) -> list:
    if not frames:
        return []
    picks = sorted(set([frames[0], frames[len(frames) // 2], frames[-1]]))
    return picks[:limit]


def pixel_check(video_path: Path, out_path: Path, start: int, end: int,
                samples: dict, anchor_bbox, track_bbox: dict) -> dict:
    """Compara salida vs origen en los frames muestreados (diff > 12 px).

    Usa el bbox del tracker para ese frame (TRACK N ↔ vídeo N-1); sin él,
    cae al bbox ancla del frame 0.
    """
    src = cv2.VideoCapture(str(video_path))
    dst = cv2.VideoCapture(str(out_path))
    results = {}
    m = 12
    for frame_idx in sorted({f for v in samples.values() for f in v}):
        bbox = track_bbox.get(frame_idx, anchor_bbox)
        bx1, by1, bx2, by2 = [int(v) for v in bbox]
        bx1, by1, bx2, by2 = bx1 - m, by1 - m, bx2 + m, by2 + m
        src.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        dst.set(cv2.CAP_PROP_POS_FRAMES, frame_idx - start)
        ok_s, frame_s = src.read()
        ok_d, frame_d = dst.read()
        if not (ok_s and ok_d) or frame_s is None or frame_d is None:
            results[str(frame_idx)] = {"error": "no se pudo leer el frame"}
            continue
        if frame_s.shape != frame_d.shape:
            results[str(frame_idx)] = {"error": f"shape {frame_s.shape} vs {frame_d.shape}"}
            continue
        diff = cv2.absdiff(frame_s, frame_d).max(axis=2)
        h, w = diff.shape
        clip = (max(0, bx1), max(0, by1), min(w, bx2), min(h, by2))
        in_bbox = diff[clip[1]:clip[3], clip[0]:clip[2]]
        results[str(frame_idx)] = {
            "bbox_used": [int(v) for v in bbox],
            "changed_gt12_total": int((diff > 12).sum()),
            "changed_gt12_anchor_bbox": int((in_bbox > 12).sum()) if in_bbox.size else 0,
        }
    src.release()
    dst.release()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=900)
    parser.add_argument("--skip-render", action="store_true",
                        help="Solo analizar el log más reciente (sin renderizar)")
    args = parser.parse_args()

    if not args.skip_render:
        if not args.video.is_file():
            raise SystemExit(f"Vídeo de entrada no encontrado: {args.video}")
        if not args.baseline.is_file():
            raise SystemExit(f"Log baseline no encontrado: {args.baseline}")

    baseline = parse_log_counts(args.baseline, args.start, args.end)
    print(f"[BASELINE] {args.baseline.name} rango {args.start}-{args.end}: "
          f"{baseline['counts']}", flush=True)

    track_bbox_baseline = baseline.get("track_bbox", {})

    if not args.skip_render:
        import roop.globals
        from roop.ProcessMgr import ProcessMgr
        from roop.ProcessOptions import ProcessOptions
        from roop.types import FaceSet

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        video_name = args.video.name
        ref_face = build_destination_ref(args.video, 0)
        source_face = load_source_face(args.source)

        roop.globals.selected_face_references = {
            f"selected_face_ref_{video_name}": [{
                "face_obj": ref_face,
                "embedding": ref_face.embedding,
                "bbox": ref_face.bbox,
                "kps": getattr(ref_face, "kps", None),
            }],
        }
        roop.globals.face_swap_mode = "selected_faces"
        roop.globals.processing = True

        options = ProcessOptions(
            {"face_swapper": {}, "mask_xseg": {}},
            getattr(roop.globals, "distance_threshold", 0.6),
            getattr(roop.globals, "blend_ratio", 0.95),
            "selected_faces",
            0, "", None, 1, False, False,
            getattr(roop.globals, "use_enhancer", True),
            getattr(roop.globals, "blend_mode", "seamless"),
        )
        manager = ProcessMgr()
        manager.initialize([FaceSet([source_face])], [], options)

        out_path = OUT_DIR / f"{args.video.stem}_{args.start}_{args.end}.mp4"
        run_started = time.time()
        for _pct, _msg in manager.run_batch_inmem(
            str(args.video), str(out_path),
            start_frame=args.start, end_frame=args.end,
            fps=29.0, skip_audio=True, temporal_smoothing=True,
        ):
            pass
        manager.Release()
        print(f"[RENDER] completado en {time.time() - run_started:.0f}s → {out_path}",
              flush=True)
        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise SystemExit("[RENDER] el vídeo de salida no existe")

    # El log de sesión lo crea run_batch_inmem; el más reciente es el nuestro.
    logs = sorted(ROOT.glob("logs/process_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise SystemExit("No hay logs de sesión para analizar")
    new_log = logs[-1]
    current = parse_log_counts(new_log, args.start, args.end)
    print(f"[NUEVO] {new_log.name} rango {args.start}-{args.end}: "
          f"{current['counts']}", flush=True)

    rej_frames = baseline["frames"]["rej"]
    msk_frames = baseline["frames"]["msk"]
    samples = {"rej": sample_frames(rej_frames), "msk": sample_frames(msk_frames)}
    pixels = {}
    out_path = OUT_DIR / f"{args.video.stem}_{args.start}_{args.end}.mp4"
    track_bbox_new = current.get("track_bbox", {})
    if out_path.is_file():
        pixels = pixel_check(args.video, out_path, args.start, args.end,
                             samples, ANCHOR_BBOX, track_bbox_new)
        print(f"[PIXELES] {json.dumps(pixels, indent=2)}", flush=True)

    b, c = baseline["counts"], current["counts"]
    failures = []
    if b["rej"] > 0:
        if c["recover"] < max(1, b["rej"] // 2):
            failures.append(
                f"fix A: solo {c['recover']}/{b['rej']} frames con ALIGN_RECOVER")
        for fidx in samples["rej"]:
            info = pixels.get(str(fidx), {})
            if info.get("changed_gt12_anchor_bbox", 0) <= 50:
                bbox_used = info.get("bbox_used", "desconocido")
                failures.append(
                    f"fix A: frame {fidx} sin swap medido en bbox={bbox_used} "
                    f"({info})")
    if b["msk"] > 0:
        if c["msk"] >= b["msk"]:
            failures.append(
                f"fix B: MASK_SKIP no baja ({c['msk']} >= {b['msk']})")
        for fidx in samples["msk"]:
            info = pixels.get(str(fidx), {})
            if info.get("changed_gt12_total", 0) <= 50:
                failures.append(
                    f"fix B: frame {fidx} sin píxeles cambiados ({info})")

    report = {
        "video": str(args.video),
        "range": [args.start, args.end],
        "baseline_log": str(args.baseline),
        "new_log": str(new_log),
        "baseline": baseline,
        "current": current,
        "samples": samples,
        "pixels": pixels,
        "failures": failures,
        "verdict": "PASS" if not failures else "FAIL",
        "generated": datetime.now().isoformat(timespec="seconds"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[REPORTE] {report_path}", flush=True)
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}", flush=True)
        return 1
    print("[PROBE] PASS — rescates A y B verificados en render real", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
