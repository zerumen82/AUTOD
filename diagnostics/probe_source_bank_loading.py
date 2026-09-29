"""Summarize why a sample of source images does or does not enter the bank."""
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys

from roop.face_util import extract_face_images


if len(sys.argv) != 2:
    raise SystemExit('Usage: python -m diagnostics.probe_source_bank_loading SOURCE_DIRECTORY')
bank = Path(sys.argv[1])
files = sorted(list(bank.glob('*.jpg')) + list(bank.glob('*.jpeg')))
sample = files[::max(1, len(files) // 24)][:24]
counts = Counter()
recovery = Counter()
print(f'[PROBE] source bank files={len(files)}, sampled={len(sample)}', flush=True)
for index, path in enumerate(sample, 1):
    output = StringIO()
    with redirect_stdout(output):
        faces = extract_face_images(str(path), (False, 0), is_source_face=True)
    if faces:
        counts['accepted'] += len(faces)
        log = output.getvalue()
        if 'Segunda escala confirma cara' in log:
            recovery['second_scale_verified'] += 1
        if 'Recuperadas' in log:
            recovery['closeup_detector'] += 1
    else:
        log = output.getvalue()
        if 'Cara INCOMPLETA' in log:
            counts['incomplete'] += 1
        elif 'Sin geometría' in log:
            counts['geometry'] += 1
        elif 'Cara borrosa' in log:
            counts['blur'] += 1
        elif 'Falló extracción de embedding' in log:
            counts['embedding'] += 1
        elif 'No se detectó' in log:
            counts['no_detection'] += 1
        else:
            counts['other_rejection'] += 1
    if index % 6 == 0:
        print(f'[PROBE] processed={index}/{len(sample)}, outcomes={dict(counts)}, recovered={dict(recovery)}', flush=True)
