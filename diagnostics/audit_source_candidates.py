"""Inspect source candidates that a previous bank scan accepted at low confidence."""
from contextlib import redirect_stdout
from pathlib import Path
import json
import re
import sys

import numpy as np
from PIL import Image, ImageDraw

from roop.face_util import extract_face_images
from roop.ProcessMgr import ProcessMgr


def main():
    root = Path(__file__).resolve().parents[1]
    bank = Path(sys.argv[1])
    report = Path(sys.argv[2]) if len(sys.argv) > 2 else root / 'diagnostics/source_audit_output'
    report.mkdir(exist_ok=True)
    history = (root / 'diagnostics/faceswap_bank_pose_extract.log').read_text(encoding='utf-8', errors='replace')
    blocks = re.split(r'\[SOURCE_DETECT\] Cargando cara de origen: ', history)[1:]
    names = [b.splitlines()[0].rsplit(' (', 1)[0] for b in blocks
             if 'No se detectó ninguna cara real' not in b
             and ('thresh=0.12' in b or 'thresh=0.22' in b)]
    if '--all' in sys.argv:
        names = sorted(p.name for p in bank.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'})
    manager = ProcessMgr.__new__(ProcessMgr)
    rows, tiles = [], []
    with (report / 'extract.log').open('w', encoding='utf-8') as log:
        for number, name in enumerate(names, 1):
            with redirect_stdout(log):
                faces = extract_face_images(str(bank / name), is_source_face=True)
            for face, crop in faces:
                row = dict(name=name, confidence=float(face.det_score),
                           quality=manager._score_source_face_quality(face),
                           bbox=np.asarray(face.bbox).tolist(),
                           embedding=np.asarray(face.embedding).tolist(),
                           kps=np.asarray(face.kps).tolist(),
                           profile=bool(manager._is_profile_face(face.kps)))
                rows.append(row)
                tile = Image.new('RGB', (220, 260), '#eeeeee')
                thumbnail = Image.fromarray(crop)
                thumbnail.thumbnail((220, 220))
                tile.paste(thumbnail, ((220-thumbnail.width)//2, (220-thumbnail.height)//2))
                draw = ImageDraw.Draw(tile)
                draw.text((3, 222), f"{len(rows)} {name[:30]}", fill='black')
                draw.text((3, 240), f"det={row['confidence']:.2f} quality={row['quality']:.2f}", fill='black')
                tiles.append(tile)
            print(f'[AUDIT] {number}/{len(names)} {name}: {len(faces)} accepted', flush=True)
    for start in range(0, len(tiles), 24):
        chunk = tiles[start:start+24]
        sheet = Image.new('RGB', (220*6, 260*((len(chunk)+5)//6)), '#888888')
        for index, tile in enumerate(chunk):
            sheet.paste(tile, ((index%6)*220, (index//6)*260))
        sheet.save(report / f'contact_{start//24+1}.jpg')
    (report / 'candidates.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(f'[AUDIT] {len(rows)} candidates, contact sheets in {report}', flush=True)


if __name__ == '__main__':
    main()
