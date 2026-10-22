"""Real local CPU comparison of CLIPSeg context crops and semantic noun aliases.

Diagnostic only: saved visual inventory is evidence, not ground truth. All masks
use the configured production threshold. No downloads or production mutations.
"""
import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from roop.img_editor import element_selection as es
from roop.img_editor.clipseg_masker import CLIPSegMasker

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = ROOT / "diagnostics/out/editor_selection_20261018/inventory_1792360222004739700/results.json"
ALIAS_PROMPT = (
    'For the target visual label, return its simple concrete noun core and at most '
    'one equivalent short visual phrase. Remove color, material, contents, and '
    'relationships where they obscure the physical item. Preserve the item identity. '
    'Do not add parts, other items, or unseen things. Reply as JSON with an '
    '"objects" array of one or two short labels. Use the image description only '
    'to disambiguate the target label. Do not list other objects in the description.'
)


def bbox(mask):
    ys, xs = np.where(mask)
    return None if not len(xs) else [int(xs.min()), int(ys.min()),
                                    int(xs.max()) + 1, int(ys.max()) + 1]


def crop_box(box, size, fraction):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * fraction, (y1 - y0) * fraction
    return [max(0, math.floor(x0 - dx)), max(0, math.floor(y0 - dy)),
            min(size[0], math.ceil(x1 + dx)), min(size[1], math.ceil(y1 + dy))]


def aliases_for(inventory, settings, prompt):
    es._add_llama_dll_directories(settings)
    from llama_cpp import Llama
    ext = settings['extractor']
    llm = Llama(model_path=str(ROOT / ext['model_path']), n_ctx=ext['context_size'],
                n_gpu_layers=0, n_threads=ext['threads'], verbose=False)
    output = {}
    try:
        for name, entry in inventory.items():
            output[name] = {}
            for label in entry['labels']:
                started = time.perf_counter()
                response = llm.create_chat_completion(
                    messages=[{'role': 'system', 'content': prompt},
                              {'role': 'user', 'content': json.dumps({
                                  'target_label': label, 'image_description': entry['description']})}],
                    temperature=0, max_tokens=ext['max_tokens'],
                    response_format={'type': 'json_object'})
                raw = response['choices'][0]['message']['content']
                try:
                    values = json.loads(raw).get('objects', [])
                    values = [item.strip().lower() for item in values if isinstance(item, str)]
                    values = list(dict.fromkeys(item for item in values
                                               if item and len(item.split()) <= ext['max_label_words']))
                except (ValueError, AttributeError, TypeError):
                    values = []
                output[name][label] = {'aliases': values, 'raw': raw,
                                       'seconds': time.perf_counter() - started}
                print(f'ALIASES {name} {label!r}: {values}', flush=True)
    finally:
        llm.close()
    return output


def overlay(image, mask):
    pixels = np.asarray(image).copy()
    color = np.array([255, 170, 0], dtype=np.uint8)
    pixels[mask] = (pixels[mask] * 0.7 + color * 0.3).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(pixels, contours, -1, (0, 220, 255), 2)
    return Image.fromarray(pixels)


def save_candidate(image, heat, base, threshold, directory, stem, metadata):
    mask = heat >= threshold
    count, components = cv2.connectedComponents(mask.astype(np.uint8))
    sizes = [int((components == index).sum()) for index in range(1, count)]
    baseline = base >= threshold
    union = (mask | baseline).sum()
    metrics = dict(metadata, threshold=threshold, bbox=bbox(mask),
                   coverage=float(mask.mean()), peak=float(heat.max()),
                   components=count - 1, component_pixels=sizes,
                   baseline_iou=float((mask & baseline).sum() / union) if union else 1.0,
                   added_pixels=int((mask & ~baseline).sum()),
                   removed_pixels=int((baseline & ~mask).sum()))
    np.save(directory / f'{stem}_heat.npy', heat.astype(np.float32))
    Image.fromarray(mask.astype(np.uint8) * 255).save(directory / f'{stem}_mask.png')
    preview = overlay(image, mask)
    preview.save(directory / f'{stem}_overlay.png')
    metrics['heat'] = f'{stem}_heat.npy'
    metrics['overlay'] = f'{stem}_overlay.png'
    print(f'MASK {stem} coverage={metrics["coverage"]:.3%} bbox={metrics["bbox"]}', flush=True)
    return metrics, preview


def montage(items, directory, name, width):
    tiles = []
    for title, picture in items:
        picture = picture.copy()
        picture.thumbnail((width, width))
        tile = Image.new('RGB', (width, width + 55), 'white')
        tile.paste(picture, ((width - picture.width) // 2, 55))
        ImageDraw.Draw(tile).text((5, 5), title, fill='black')
        tiles.append(tile)
    columns = min(4, len(tiles))
    canvas = Image.new('RGB', (width * columns, (width + 55) * math.ceil(len(tiles) / columns)), 'white')
    for index, tile in enumerate(tiles):
        canvas.paste(tile, ((index % columns) * width, (index // columns) * (width + 55)))
    canvas.save(directory / f'{name}_comparison.jpg', quality=95)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument('--only', nargs='+', default=['coffee', 'two_people', 'astronaut'])
    parser.add_argument('--fractions', nargs='+', type=float, default=[0.35, 0.75, 1.5])
    parser.add_argument('--alias-prompt', default=ALIAS_PROMPT)
    parser.add_argument('--skip-aliases', action='store_true')
    parser.add_argument('--preview-side', type=int, default=400)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if any(value < 0 for value in args.fractions):
        parser.error('Context fractions must be nonnegative')
    settings = es.load_settings()
    inventory = json.loads(args.inventory.read_text(encoding='utf-8'))
    inventory = {name: inventory[name] for name in args.only}
    output = args.output or ROOT / 'diagnostics/out/editor_selection_multiscale_20261019' / str(time.time_ns())
    output.mkdir(parents=True, exist_ok=True)
    print('EVIDENCE', output, flush=True)
    results = {'inventory': str(args.inventory), 'device': 'cpu',
               'fractions': args.fractions, 'threshold': settings['segmenter']['probability_threshold'],
               'alias_prompt': args.alias_prompt, 'photos': {}}
    aliases = {} if args.skip_aliases else aliases_for(inventory, settings, args.alias_prompt)
    results['aliases'] = aliases
    (output / 'aliases.json').write_text(json.dumps(aliases, indent=2), encoding='utf-8')
    import torch
    torch.set_num_threads(args.threads)
    seg = settings['segmenter']
    backbone = Path(seg['backbone_path']).expanduser()
    weights = ROOT / seg['weights_path']
    if not backbone.is_file() or not weights.is_file():
        raise FileNotFoundError('Installed local CLIPSeg model files are required; no downloads')
    masker = CLIPSegMasker(backbone_path=str(backbone))
    masker.device = 'cpu'
    masker._WEIGHT_CANDIDATES = [(str(weights), 'uni')]
    ok, message = masker.initialize()
    if not ok or 'SIN pesos' in message:
        raise RuntimeError(message)
    threshold = float(seg['probability_threshold'])
    try:
        for name, entry in inventory.items():
            image = Image.open(entry['photo']).convert('RGB')
            directory = output / name
            directory.mkdir(exist_ok=True)
            image.save(directory / 'source.png')
            report = {'photo': entry['photo'], 'description': entry['description'], 'labels': {}}
            results['photos'][name] = report
            heat_cache = {}
            for index, label in enumerate(entry['labels']):
                labels = list(dict.fromkeys([label] + aliases.get(name, {}).get(label, {}).get('aliases', [])))
                candidates = []
                pictures = [('Original', image)]
                original_heat = None
                for alias_index, query in enumerate(labels):
                    if query not in heat_cache:
                        heat_cache[query] = masker.probability(image, query)
                    heat = heat_cache[query]
                    if heat is None:
                        raise RuntimeError(f'No real heat for {query}')
                    if original_heat is None:
                        original_heat = heat
                    stem = f'label{index}_query{alias_index}_global'
                    record, preview = save_candidate(image, heat, original_heat, threshold, directory,
                                                     stem, {'query': query, 'method': 'global'})
                    candidates.append(record)
                    pictures.append((f'{query}: global', preview))
                    measured_box = bbox(heat >= threshold)
                    if measured_box is None:
                        continue
                    seen_boxes = set()
                    for fraction in args.fractions:
                        box = crop_box(measured_box, image.size, fraction)
                        if tuple(box) in seen_boxes:
                            continue
                        seen_boxes.add(tuple(box))
                        local_heat = masker.probability(image.crop(box), query)
                        if local_heat is None:
                            raise RuntimeError(f'No real crop heat for {query}')
                        reprojected = np.zeros_like(heat)
                        x0, y0, x1, y1 = box
                        reprojected[y0:y1, x0:x1] = local_heat
                        for method, candidate in [('crop', reprojected),
                                                  ('max_global_crop', np.maximum(heat, reprojected))]:
                            stem = f'label{index}_query{alias_index}_context{fraction}_{method}'
                            record, preview = save_candidate(
                                image, candidate, original_heat, threshold, directory, stem,
                                {'query': query, 'method': method, 'crop_bbox': box,
                                 'measured_global_bbox': measured_box, 'context_fraction': fraction})
                            candidates.append(record)
                            if method == 'crop':
                                pictures.append((f'{query}: crop context={fraction}', preview))
                report['labels'][label] = candidates
                montage(pictures, directory, f'label{index}', args.preview_side)
                (output / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    finally:
        masker.model = None
    print('COMPLETE', output, flush=True)


if __name__ == '__main__':
    main()
