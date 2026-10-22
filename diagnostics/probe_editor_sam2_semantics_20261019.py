"""Diagnostic local CLIP semantics for actual SAM2 masks and open visual labels.

No ground truth, catalogue, model downloads or production mutations. Cosines
measure compatibility among the supplied labels; they certify neither semantic
correctness nor completeness. All label text is read from source evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import clip
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def rank(scores, labels):
    order = np.argsort(-np.asarray(scores)).tolist()
    return {'ranking': [{'label': labels[index], 'cosine': float(scores[index])}
                        for index in order],
            'top_label': labels[order[0]],
            'top_margin': float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else None}


def views_for(image, mask, cfg):
    ys, xs = np.where(mask)
    if not len(xs):
        return None, {}
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    padding = float(cfg['crop_context_fraction'])
    dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
    box = [max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)),
           min(image.width, math.ceil(x1 + dx)), min(image.height, math.ceil(y1 + dy))]
    rgb = np.asarray(image).copy()
    rgb[~mask] = cfg['neutral_rgb']
    available = {'tight_crop': image.crop(box), 'masked_crop': Image.fromarray(rgb).crop(box)}
    return box, {name: available[name] for name in cfg['views']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidates', type=Path, help='SAM2 results.json with saved mask filenames')
    parser.add_argument('--config', type=Path, default=ROOT / 'diagnostics/editor_sam2_semantics_20261019.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'diagnostics/out/editor_sam2_semantics_20261019')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8-sig'))
    if cfg['batch_size'] < 1 or not cfg['views']:
        raise ValueError('A positive batch size and nonempty views are required')
    if len(set(cfg['views'])) != len(cfg['views']) or any(
            name not in {'tight_crop', 'masked_crop'} for name in cfg['views']):
        raise ValueError('Views must be distinct supported transforms')
    if cfg['crop_context_fraction'] < 0:
        raise ValueError('Context fraction must be nonnegative')
    sam_path = args.candidates.resolve()
    sam = json.loads(sam_path.read_text(encoding='utf-8-sig'))
    inventory_path = resolve(sam['analysis'])
    inventory = json.loads(inventory_path.read_text(encoding='utf-8-sig'))
    labels = list(dict.fromkeys(region['label'] for region in inventory['regions']))
    if not labels:
        raise ValueError('No source visual labels; no replacement catalogue is permitted')
    source_path = resolve(sam['source'])
    photo = Image.open(source_path).convert('RGB')
    if tuple(inventory['size']) != photo.size:
        raise ValueError('Source inventory image size differs from source')
    from roop.img_editor.element_selection import image_fingerprint
    if not sam['fingerprint'] == inventory['fingerprint'] == image_fingerprint(photo):
        raise ValueError('Source photograph/inventory/SAM2 fingerprints differ')
    model_path = resolve(cfg['model_path'])
    if not model_path.is_file():
        raise FileNotFoundError('Installed local CLIP checkpoint required; this probe never downloads')
    target = args.output / str(time.time_ns())
    target.mkdir(parents=True, exist_ok=False)
    print('EVIDENCE', target, flush=True)
    torch.set_num_threads(cfg['threads'])
    started = time.perf_counter()
    model, preprocess = clip.load(str(model_path), device='cpu', jit=False)
    model.eval()
    templates = cfg['text_templates']
    if not templates:
        raise ValueError('Text templates are required')
    texts = [template.format(label=label) for label in labels for template in templates]
    with torch.inference_mode():
        features = model.encode_text(clip.tokenize(texts)).float()
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.reshape(len(labels), len(templates), -1).mean(dim=1)
        text_features = features / features.norm(dim=-1, keepdim=True)
    print('MODEL CPU loaded; labels', labels, flush=True)
    result = {
        'source': str(source_path), 'source_sha256': sha256(source_path),
        'source_fingerprint': sam['fingerprint'], 'sam2_results': str(sam_path),
        'sam2_results_sha256': sha256(sam_path), 'inventory': str(inventory_path),
        'inventory_sha256': sha256(inventory_path), 'labels': labels,
        'labels_sha256': hashlib.sha256(json.dumps(labels, ensure_ascii=False).encode()).hexdigest(),
        'model_path': str(model_path), 'model_sha256': sha256(model_path),
        'config': cfg, 'config_path': str(args.config.resolve()), 'config_sha256': sha256(args.config),
        'script_sha256': sha256(Path(__file__)), 'device': 'cpu',
        'scope': 'Open-label compatibility only; no ground truth, semantic guarantee or object-completeness guarantee',
        'candidates': []}
    pending = []
    for index, candidate in enumerate(sam['candidates']):
        mask_path = sam_path.parent / candidate['mask']
        mask = np.asarray(Image.open(mask_path).convert('L')) >= 128
        if mask.shape != (photo.height, photo.width):
            raise ValueError(f'Candidate mask size mismatch: {mask_path}')
        box, views = views_for(photo, mask, cfg)
        entry = {'index': index, 'mask': str(mask_path), 'mask_sha256': sha256(mask_path),
                 'method': candidate['method'], 'source_label': candidate.get('label'),
                 'source_region_id': candidate.get('region_id'), 'bbox': box,
                 'coverage': float(mask.mean()), 'model_iou': candidate['model_iou'],
                 'stability': candidate['stability'], 'views': {}}
        result['candidates'].append(entry)
        if not views:
            entry['omitted_reason'] = 'empty mask'
            continue
        for name, view in views.items():
            if cfg['save_views']:
                filename = f'candidate_{index:04d}_{name}.png'
                view.save(target / filename)
                entry['views'][name] = {'image': filename}
            else:
                entry['views'][name] = {}
            pending.append((index, name, preprocess(view)))
    with torch.inference_mode():
        for offset in range(0, len(pending), cfg['batch_size']):
            batch = pending[offset:offset + cfg['batch_size']]
            images = torch.stack([tensor for _, _, tensor in batch])
            image_features = model.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            scores = (image_features @ text_features.T).cpu().numpy()
            for (index, name, _), values in zip(batch, scores):
                result['candidates'][index]['views'][name].update(rank(values, labels))
            print(f'ENCODED {offset + len(batch)}/{len(pending)} views', flush=True)
    for entry in result['candidates']:
        if not entry['views']:
            continue
        matrix = np.array([[next(item['cosine'] for item in view['ranking'] if item['label'] == label)
                            for label in labels] for view in entry['views'].values()])
        entry['mean'] = rank(matrix.mean(axis=0), labels)
        entry['view_agreement'] = len({view['top_label'] for view in entry['views'].values()}) == 1
        if entry['source_label'] in labels:
            ranking = entry['mean']['ranking']
            source_index = next(i for i, item in enumerate(ranking) if item['label'] == entry['source_label'])
            entry['source_label_rank'] = source_index + 1
            entry['source_label_gap_to_top'] = ranking[0]['cosine'] - ranking[source_index]['cosine']
        print(f'RANK {entry["index"]:04d} source={entry["source_label"]!r} '
              f'top={entry["mean"]["top_label"]!r} margin={entry["mean"]["top_margin"]:.4f} '
              f'agree={entry["view_agreement"]}', flush=True)
    result['seconds'] = time.perf_counter() - started
    (target / 'scores.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('COMPLETE', target, flush=True)


if __name__ == '__main__':
    main()
