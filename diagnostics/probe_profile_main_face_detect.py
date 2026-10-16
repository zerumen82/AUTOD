"""Mide qué detecta el pipeline actual de AutoAuto en una foto concreta.

Objetivo (caso real 02457): la sujeto principal en perfil casi 90° NO se
detecta, mientras sí se detecta una persona de fondo de 50x63 px.

Este probe NO cambia producción. Solo mide, con los mismos analizadores que
usa el producto:
  - RetinaFace (insightface buffalo_l) con det_size 640 / 1024 y espejo.
  - MediaPipe (fallback del producto) con la foto tal cual.
Salva un JSON con las cajas detectadas y un overlay PNG para revisión.

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_profile_main_face_detect.py [ruta_foto]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DEFAULT_PHOTO = r"D:\PROJECTS\uee\FACES\MARETA\02457.jpg"
OUT_DIR = os.path.join(ROOT, "diagnostics", "out", "profile_detect_20261013")


def _boxes_from(faces, label):
    out = []
    for f in faces:
        bbox = [round(float(v), 1) for v in f.bbox]
        out.append({
            "from": label,
            "bbox": bbox,
            "w": round(bbox[2] - bbox[0], 1),
            "h": round(bbox[3] - bbox[1], 1),
            "det_score": round(float(f.get("det_score", f.get("score", 0.0))), 4),
        })
    return out


def run():
    photo = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PHOTO
    os.makedirs(OUT_DIR, exist_ok=True)
    img_bgr = cv2.imread(photo)
    if img_bgr is None:
        print(f"FAIL: no se pudo leer {photo}")
        return 1
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    frame_area = w * h
    print(f"Foto: {photo}  {w}x{h}")

    import roop.globals  # noqa: F401
    from roop.face_util import get_face_analyser

    analyser = get_face_analyser()
    orig_det = analyser.det_size
    orig_input = analyser.det_model.input_size

    report = {"photo": photo, "size": [w, h], "runs": []}

    for det_size in ((640, 640), (1024, 1024)):
        for mirror in (False, True):
            analyser.det_size = det_size
            analyser.det_model.input_size = det_size
            frame = img_rgb[:, :, ::-1] if mirror else img_rgb  # espejo simple
            try:
                faces = analyser.get(np.ascontiguousarray(frame))
            except Exception as exc:  # noqa: BLE001
                print(f"  det_size={det_size[0]} mirror={mirror}: ERROR {exc}")
                report["runs"].append({
                    "det_size": det_size[0], "mirror": mirror, "error": str(exc),
                })
                continue
            boxes = []
            for f in faces:
                bbox = [round(float(v), 1) for v in f.bbox]
                if mirror:
                    bbox = [round(w - bbox[2], 1), bbox[1], round(w - bbox[0], 1), bbox[3]]
                boxes.append({
                    "bbox": bbox,
                    "w": round(bbox[2] - bbox[0], 1),
                    "h": round(bbox[3] - bbox[1], 1),
                    "area_frac": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / frame_area, 5),
                    "det_score": round(float(f.get("det_score", 0.0)), 4),
                })
            print(f"  det_size={det_size[0]} mirror={mirror}: {len(boxes)} cara(s) -> {boxes}")
            report["runs"].append({
                "det_size": det_size[0], "mirror": mirror, "faces": boxes,
            })

    # MediaPipe (fallback del producto, solo se usa hoy con 0 caras)
    analyser.det_size = orig_det
    analyser.det_model.input_size = orig_input

    # Ruta REAL del producto (ProcessMgr.process_frame foto):
    # get_all_faces_smart(frame, min_score=None, for_target=True,
    #                     prefer_fast=False, include_180=True)
    import roop.face_util_rotation as _fur
    _orig_filter = _fur._filter_false_positive_faces

    def _spy_filter(faces):
        in_boxes = [
            [round(float(v)) for v in f.bbox] + [round(float(getattr(f, "det_score", 0.0) or 0.0), 3)]
            for f in faces
        ]
        out = _orig_filter(faces)
        print(f"  SPY filtro_final: entra {len(faces)} -> sale {len(out)}; in={in_boxes}")
        return out

    _fur._filter_false_positive_faces = _spy_filter
    _orig_convert = _fur.convert_faces_to_face_objects

    def _spy_convert(faces):
        conv = _orig_convert(faces)
        boxes = [
            [round(float(v)) for v in f.bbox] + [round(float(getattr(f, "det_score", 0.0) or 0.0), 3)]
            for f in conv
        ]
        print(f"  SPY convert: {len(faces)} raw -> {len(conv)} objetos; {boxes}")
        return conv

    _fur.convert_faces_to_face_objects = _spy_convert
    from roop.face_util_rotation import get_all_faces_smart
    for prefer_fast in (False, True):
        try:
            prod_faces = get_all_faces_smart(
                img_bgr, min_score=None, for_target=True,
                prefer_fast=prefer_fast, include_180=True,
            )
            boxes = []
            for f in prod_faces:
                bbox = [round(float(v), 1) for v in f.bbox]
                boxes.append({
                    "bbox": bbox,
                    "w": round(bbox[2] - bbox[0], 1),
                    "h": round(bbox[3] - bbox[1], 1),
                    "area_frac": round((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / frame_area, 5),
                    "det_score": round(float(getattr(f, "det_score", 0.0) or 0.0), 4),
                })
            print(f"  PRODUCTO get_all_faces_smart(prefer_fast={prefer_fast}): {len(boxes)} cara(s) -> {boxes}")
            report["runs"].append({
                "engine": f"producto_smart_prefer_fast_{prefer_fast}",
                "faces": boxes,
            })
        except Exception as exc:  # noqa: BLE001
            print(f"  PRODUCTO get_all_faces_smart(prefer_fast={prefer_fast}): ERROR {exc}")
            report["runs"].append({
                "engine": f"producto_smart_prefer_fast_{prefer_fast}", "error": str(exc),
            })

    # Instrumentación: por qué la sujeto principal (visible a 640) no sobrevive
    # al filtro final `_filter_false_positive_faces` en la ruta de foto.
    try:
        import numpy as _np
        from roop.face_util import _has_face_geometry
        from roop.face_util_rotation import convert_faces_to_face_objects
        from roop.faceswap_config import landmarks_fit_bbox

        analyser.det_size = (640, 640)
        analyser.det_model.input_size = (640, 640)
        raw640 = analyser.get(img_rgb)
        conv640 = convert_faces_to_face_objects(
            [f for f in raw640 if float(f.get("det_score", 0.0)) >= 0.10]
        )
        gates = []
        for f in conv640:
            bbox = [round(float(v), 1) for v in f.bbox]
            det = round(float(getattr(f, "det_score", 0.0) or 0.0), 4)
            kps_arr = _np.asarray(f.kps, dtype=_np.float32)
            fit_ok = bool(landmarks_fit_bbox(kps_arr, f.bbox))
            orig_kps = f.kps
            f.kps = kps_arr
            try:
                geo_ok = bool(_has_face_geometry(f))
            finally:
                f.kps = orig_kps
            kept = fit_ok and (det >= 0.50 or geo_ok)
            gates.append({
                "bbox": bbox, "w": round(bbox[2] - bbox[0], 1), "h": round(bbox[3] - bbox[1], 1),
                "det_score": det, "landmarks_fit_bbox": fit_ok, "has_face_geometry": geo_ok,
                "survives_final_filter": kept,
            })
            print(f"  GATE det640 bbox={bbox} det={det} fit={fit_ok} geo={geo_ok} -> kept={kept}")
        analyser.det_size = orig_det
        analyser.det_model.input_size = orig_input
        report["runs"].append({"engine": "gates_det640", "faces": gates})
    except Exception as exc:  # noqa: BLE001
        print(f"  GATE det640: ERROR {exc}")
        report["runs"].append({"engine": "gates_det640", "error": str(exc)})

    try:
        from roop.face_util_rotation import get_all_faces_mediapipe
        mp_faces = get_all_faces_mediapipe(cv2.imread(photo))
        boxes = _boxes_from(mp_faces, "mediapipe")
        for b in boxes:
            b["area_frac"] = round(b["w"] * b["h"] / frame_area, 5)
        print(f"  mediapipe: {len(boxes)} cara(s) -> {boxes}")
        report["runs"].append({"det_size": None, "mirror": None, "engine": "mediapipe", "faces": boxes})
    except Exception as exc:  # noqa: BLE001
        print(f"  mediapipe: ERROR {exc}")
        report["runs"].append({"engine": "mediapipe", "error": str(exc)})

    json_path = os.path.join(OUT_DIR, "detect_report.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")

    # Overlay con la mejor corrida (más caras, y la mayor como referencia)
    overlay = img_bgr.copy()
    for run_data in report["runs"]:
        for f in run_data.get("faces", []) or []:
            x1, y1, x2, y2 = (int(v) for v in f["bbox"])
            color = (0, 255, 0) if f["w"] * f["h"] > 0.05 * frame_area else (0, 0, 255)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            cv2.putText(overlay, f"{run_data.get('engine') or run_data['det_size']}",
                        (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    png_path = os.path.join(OUT_DIR, "detect_overlay.png")
    cv2.imwrite(png_path, overlay)
    print(f"PNG: {png_path}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
