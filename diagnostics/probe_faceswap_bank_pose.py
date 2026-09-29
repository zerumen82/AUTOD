"""Measure pose/quality distribution of the current source folder."""
from contextlib import redirect_stdout
from pathlib import Path

from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images

SOURCE_DIR = Path(r'D:\PROJECTS\uee\FACES\MARETA')
ROOT = Path(__file__).resolve().parents[1]


def main():
    files = sorted(p for p in SOURCE_DIR.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    mgr = ProcessMgr.__new__(ProcessMgr)
    data = []
    with (ROOT / 'diagnostics' / 'faceswap_bank_pose_extract.log').open('w', encoding='utf-8') as log:
        for idx, path in enumerate(files, 1):
            with redirect_stdout(log):
                faces = extract_face_images(str(path), is_source_face=True)
            for face, _ in faces:
                data.append((path.name, mgr._is_profile_face(getattr(face, 'kps', None)),
                             mgr._score_source_face_quality(face)))
            if idx % 100 == 0:
                print(f'[POSE] {idx}/{len(files)} loaded={len(data)}', flush=True)
    profiles = sorted((d for d in data if d[1]), key=lambda d: d[2], reverse=True)
    frontals = sorted((d for d in data if not d[1]), key=lambda d: d[2], reverse=True)
    print(f'[POSE] total={len(data)} profiles={len(profiles)} frontal={len(frontals)}', flush=True)
    for label, rows in [('profile', profiles), ('frontal', frontals)]:
        for name, _, quality in rows[:12]:
            print(f'[POSE] {label} {name} quality={quality:.2f}', flush=True)


if __name__ == '__main__':
    main()
