"""Compare actual CLIPSeg measurement in source/captured windows, CPU only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402

from roop.img_editor.clipseg_masker import CLIPSegMasker  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('window', type=Path)
    parser.add_argument('query')
    parser.add_argument('exclusion', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--seed', type=Path)
    parser.add_argument('--margins', type=float, nargs='+', default=[])
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    source = Image.open(args.source).convert('RGB')
    window = Image.open(args.window).convert('RGB')
    excluded = np.asarray(Image.open(args.exclusion).convert('L')) >= 128
    assert excluded.shape == (source.height, source.width)
    backbone = Path.home()/'.cache/clip/ViT-B-16.pt'
    assert backbone.is_file(), 'CLIP backbone must already exist; no implicit download allowed'
    assert any(Path(path).is_file() for path, _ in CLIPSegMasker._WEIGHT_CANDIDATES)
    error, _, (x, y), _ = cv2.minMaxLoc(cv2.matchTemplate(np.asarray(source), np.asarray(window), cv2.TM_SQDIFF))
    box = (x, y, x+window.width, y+window.height)
    assert np.array_equal(np.asarray(source.crop(box)), np.asarray(window)), error
    plans = [('full', source, (0,0,source.width,source.height)), ('crop', window, box)]
    if args.seed:
        measured_seed = np.asarray(Image.open(args.seed).convert('L')) >= 96
        assert measured_seed.shape == excluded.shape and measured_seed.any()
        yy,xx = np.where(measured_seed & ~excluded)
        left,top,right,bottom = int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
        for margin in args.margins:
            assert margin >= 0
            dx,dy = int(np.ceil((right-left)*margin)),int(np.ceil((bottom-top)*margin))
            bounds=(max(0,left-dx),max(0,top-dy),min(source.width,right+dx),min(source.height,bottom+dy))
            plans.append((f'margin_{margin}',source.crop(bounds),bounds))
    output = args.report/f'crop_measurement_{time.time_ns()}'
    output.mkdir(parents=True)
    masker = CLIPSegMasker()
    assert masker.device == 'cpu'
    metadata = dict(completed=False, device='cpu', query=args.query, window_box=box,
                    source=str(args.source), window=str(args.window),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(), variants=[])
    try:
        for label, image, bounds in plans:
            measured = masker.generate_mask(image, args.query)
            probability = masker.probability(image, args.query)
            assert probability is not None and np.isfinite(probability).all()
            heat = np.zeros(excluded.shape, np.float32)
            lx,ty,rx,by = bounds
            heat[ty:by,lx:rx] = probability
            Image.fromarray((heat*255).astype(np.uint8)).save(output/f'{label}_probability.png')
            selected = np.zeros(excluded.shape, bool)
            if measured is not None:
                selected[ty:by,lx:rx] = np.asarray(measured) >= 96
            overlap = int((selected & excluded).sum())
            selected &= ~excluded
            Image.fromarray(selected.astype(np.uint8)*255).save(output/f'{label}_mask.png')
            overlay = np.asarray(source).copy()
            overlay[selected] = (overlay[selected]*.55+np.array([255,0,0])*.45).astype(np.uint8)
            Image.fromarray(overlay).save(output/f'{label}_overlay.png')
            metadata['variants'].append(dict(label=label,source_coverage=float(selected.mean()),
                                             box=bounds, raw_exclusion_overlap=overlap,
                                             crop_boundary_foreground=bool(np.any(probability[0]>.4)
                                                 or np.any(probability[-1]>.4) or np.any(probability[:,0]>.4)
                                                 or np.any(probability[:,-1]>.4)),
                                             max_probability=float(heat.max())))
        metadata.update(completed=True,seconds=time.monotonic()-started)
    finally:
        masker.unload()
        (output/'meta.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(output,json.dumps(metadata),flush=True)


if __name__ == '__main__':
    main()
