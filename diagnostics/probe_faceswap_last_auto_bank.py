"""Compare old and quality-first source choice on the last automatic photo bank."""
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import cv2

from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.face_util_rotation import get_all_faces_smart
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / '.autodeep_temp'
PHOTO = TEMP / 'e80f75b083cb24fe430d742ea9bdcd2db5e44f00cddf1463456b945f771177b5' / '713373234.jpg'
START = datetime(2026, 9, 27, 1, 27).timestamp()
END = datetime(2026, 9, 27, 1, 28).timestamp()


def main():
    files = sorted(
        p for p in TEMP.rglob('*')
        if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}
        and START <= p.stat().st_mtime < END
    )
    print(f'[PROBE] uploaded_bank_files={len(files)}', flush=True)
    assert len(files) >= 300
    bank, accepted = [], 0
    with (ROOT / 'diagnostics' / 'last_auto_bank_extract.log').open('w', encoding='utf-8') as log:
        for index, path in enumerate(files, 1):
            with redirect_stdout(log):
                extracted = extract_face_images(str(path), is_source_face=True)
            for face, image in extracted:
                if image is None or image.size == 0:
                    continue
                face.face_img = image
                face.face_img_ref = image
                bank.append(FaceSet(faces=[face], name=path.name))
                accepted += 1
            if index % 50 == 0:
                print(f'[PROBE] scanned={index}/{len(files)} accepted={accepted}', flush=True)
    print(f'[PROBE] accepted_sources={accepted}', flush=True)
    assert accepted > 100

    frame = cv2.imread(str(PHOTO))
    assert frame is not None
    faces = get_all_faces_smart(frame, for_target=True, include_180=True)
    target = max(
        (f for f in faces if (f.bbox[0] + f.bbox[2]) * 0.5 > frame.shape[1] * 0.55),
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    manager = ProcessMgr.__new__(ProcessMgr)
    manager.input_facesets = bank
    manager.source_embeddings_cache = {}
    manager._cache_source_embeddings()
    sources = [entry.faces[0] for entry in bank]
    profile = manager._is_profile_face(target.kps, target.det_score)
    old = manager._select_source_face_by_similarity(target, sources, is_profile=profile, quality_first=False)
    new = manager._select_source_face_by_similarity(target, sources, is_profile=profile, quality_first=True)
    print(f'[PROBE] target_bbox={target.bbox} old={manager._source_face_name_by_id[id(old)]} '
          f'old_quality={manager._score_source_face_quality(old):.2f} '
          f'new={manager._source_face_name_by_id[id(new)]} '
          f'new_quality={manager._score_source_face_quality(new):.2f}', flush=True)


if __name__ == '__main__':
    main()
