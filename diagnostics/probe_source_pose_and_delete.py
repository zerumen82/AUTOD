# -*- coding: utf-8 -*-
"""Selección por pose y borrado de un archivo de origen."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class Face:
    def __init__(self, emb, bbox, kps, det=0.9, seed=1):
        vec = np.asarray(emb, dtype=np.float32)
        vec = vec / np.linalg.norm(vec)
        self.embedding = vec
        self.normed_embedding = vec
        self.bbox = bbox
        self.kps = kps
        self.det_score = det
        self.score = det
        rng = np.random.default_rng(seed)
        self.face_img_ref = rng.integers(40, 220, size=(420, 420, 3), dtype=np.uint8)
        self.source_image_size = (900, 900)


FRONT = [[80, 100], [220, 102], [150, 160], [110, 210], [190, 212]]
PROFILE = [[40, 110], [70, 108], [18, 160], [48, 210], [95, 200]]


def _unit(values):
    vec = np.asarray(values, dtype=np.float32)
    return vec / np.linalg.norm(vec)


def main():
    fails = 0

    def check(name, cond, extra=""):
        nonlocal fails
        print(("OK  " if cond else "FAIL") + f" {name}" + (f" — {extra}" if extra else ""))
        if not cond:
            fails += 1

    from roop.ProcessMgr import ProcessMgr
    from roop.types import FaceSet
    import roop.globals
    import ui.globals
    from ui.tabs.faceswap.events import _drop_source_file_faces, _purge_bank_by_face

    person_a = _unit([1, 0, 0, 0])
    person_b = _unit([0, 1, 0, 0])
    person_a_profile = _unit([0.96, 0.28, 0, 0])
    target = Face(person_b, [10, 10, 180, 220], FRONT, seed=9)
    sharp_front = Face(person_a, [0, 0, 420, 520], FRONT, det=0.95, seed=2)
    matching_profile = Face(person_a_profile, [0, 0, 300, 390], PROFILE, det=0.88, seed=3)
    other = Face(person_b, [0, 0, 360, 450], FRONT, det=0.93, seed=4)
    tiny = Face(_unit([0.0, 0.40, 0.92, 0.0]), [0, 0, 30, 32], FRONT, det=0.80, seed=5)
    manager = ProcessMgr.__new__(ProcessMgr)
    manager._source_face_name_by_id = {
        id(sharp_front): "front.jpg",
        id(matching_profile): "profile.jpg",
        id(other): "other.jpg",
        id(tiny): "tiny.jpg",
    }
    manager._auto_source_consensus_cache = {}

    photo = manager._select_source_face_by_similarity(
        target, [sharp_front, matching_profile, other], is_profile=True, quality_first=True,
    )
    check("foto de perfil elige el perfil de su identidad", photo is matching_profile, getattr(photo, "bbox", None))

    many_a = [Face(person_a, [0, 0, 280, 340], FRONT, seed=10 + i) for i in range(4)]
    video = manager._select_source_face_by_similarity(
        target, [tiny, *many_a], is_profile=False, quality_first=False,
    )
    check("vídeo con similitud baja no fija la cara diminuta", video is not tiny and video in many_a)

    roop.globals.INPUT_FACESETS = [
        FaceSet(faces=[sharp_front], name="same.jpg"),
        FaceSet(faces=[matching_profile], name="same.jpg"),
        FaceSet(faces=[other], name="keep.jpg"),
    ]
    ui.globals.ui_input_thumbs = ["a", "b", "c"]
    removed = _drop_source_file_faces(0)
    check(
        "borrar un origen quita las caras de ese archivo",
        removed == 2 and len(roop.globals.INPUT_FACESETS) == 1
        and roop.globals.INPUT_FACESETS[0].name == "keep.jpg"
        and ui.globals.ui_input_thumbs == ["c"],
        f"removed={removed} left={[fs.name for fs in roop.globals.INPUT_FACESETS]}",
    )

    kept = Face(person_a, [1, 1, 40, 40], FRONT, seed=8)
    roop.globals.selected_face_references = {
        "selected_face_ref_a.jpg": [{"face_obj": other}],
        "selected_face_ref_b.jpg": [{"face_obj": kept}, {"face_obj": other}],
    }
    _purge_bank_by_face(other)
    left = roop.globals.selected_face_references
    check(
        "borrar un destino quita solo esa cara del banco",
        "selected_face_ref_a.jpg" not in left
        and left["selected_face_ref_b.jpg"] == [{"face_obj": kept}],
        str(left),
    )
    print(f"TOTAL FAIL={fails}")
    return fails


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
