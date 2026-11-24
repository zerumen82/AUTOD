"""Render integrado corto del retorno erróneo f1699 del lote 20/11/2026.

Usa caras reales del vídeo como banco reconstruido; no reproduce el banco UI.
La salida va a diagnostics y nunca sobrescribe el MP4 entregado.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import roop.globals  # noqa: E402
from diagnostics.probe_faceswap_recent_reacquire_20261120 import (  # noqa: E402
    LOGS, detected_face, logged_bbox, video_from_log,
)
from diagnostics.probe_faceswap_video_recovery_render_20261107 import (  # noqa: E402
    load_source_face,
)
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.ProcessOptions import ProcessOptions  # noqa: E402
from roop.types import FaceSet  # noqa: E402

OUT = ROOT / "diagnostics/out/faceswap_latest_20261120"
SOURCE_REPORT = ROOT / (
    "diagnostics/out/faceswap_tracking_20261119/"
    "video5_diag_425_436_real_source_direct_v2.json"
)


def main():
    reverse_bank = "--reverse-bank" in sys.argv[1:]
    video, content = video_from_log(LOGS[0])
    source_path = Path(json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))["source"])
    if not source_path.is_file():
        raise RuntimeError(f"Fuente real no disponible: {source_path}")
    refs = [
        detected_face(video, 0, logged_bbox(content, 1)),
        detected_face(video, 1329, logged_bbox(content, 1329, "RE-BUSQUEDA")),
        detected_face(video, 1631, logged_bbox(content, 1631, "RE-BUSQUEDA")),
    ]
    if reverse_bank:
        refs.reverse()
    roop.globals.selected_face_references = {
        f"selected_face_ref_{video.name}": [
            {"face_obj": face, "embedding": face.embedding, "bbox": face.bbox,
             "kps": getattr(face, "kps", None)}
            for face in refs
        ],
    }
    roop.globals.face_swap_mode = "selected_faces"
    roop.globals.processing = True
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
    OUT.mkdir(parents=True, exist_ok=True)
    suffix = "refs3_reverse" if reverse_bank else "refs3"
    output = OUT / f"video5_1640_1736_v51402_{suffix}.mp4"
    if output.exists():
        raise FileExistsError(output)
    for percent, message in manager.run_batch_inmem(
        str(video), str(output), start_frame=1640, end_frame=1736,
        skip_audio=True, temporal_smoothing=True,
    ):
        if percent >= 100:
            print(f"[DIAG] {message}", flush=True)
    process_log = next(
        (path for path in sorted((ROOT / "logs").glob("process_*.log"),
                                 key=lambda path: path.stat().st_mtime, reverse=True)
         if f"Output: {output}" in path.read_text(encoding="utf-8", errors="replace")[:1000]),
        None,
    )
    if process_log is None:
        raise RuntimeError("El render terminó sin log de proceso localizable")
    manager.Release()
    report = {
        "video": str(video),
        "source": str(source_path),
        "output": str(output),
        "process_log": str(process_log),
        "range": [1640, 1736],
        "reference_frames": [1631, 1329, 0] if reverse_bank else [0, 1329, 1631],
        "limitations": "Banco destino reconstruido, no equivale al banco UI original",
    }
    report_path = OUT / f"video5_1640_1736_v51402_{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
