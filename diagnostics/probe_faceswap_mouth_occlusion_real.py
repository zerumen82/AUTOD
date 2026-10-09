"""Inspect occlusion and mouth masks on a real failed photo without touching UI output."""
import copy
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import roop.globals
import roop.ProcessMgr as process_module
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.face_util_rotation import get_all_faces_smart
from roop.types import FaceSet

ROOT = Path(__file__).resolve().parents[1]
PHOTO = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ.get('AUTOAUTO_PROBE_PHOTO', str(
    ROOT / '.autodeep_temp' / 'f4ee748acb611ae543649b5631baf569bdd1495da5aee5b7b834747ff61fc7e5' / '1897592076.jpg'
)))
SOURCE = Path(r'D:\PROJECTS\uee\FACES\MARETA\IMG_20200424_192331.jpg')
OUT = ROOT / 'diagnostics'


def main():
    frame = cv2.imread(str(PHOTO))
    faces = get_all_faces_smart(frame, for_target=True, include_180=True)
    target = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    points = np.asarray(target.landmark_106, dtype=np.float32)
    marked = frame.copy()
    for index, (x, y) in enumerate(points):
        cv2.circle(marked, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.putText(marked, str(index), (int(x) + 2, int(y) - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.30, (0, 0, 255), 1, cv2.LINE_AA)
    x1, y1, x2, y2 = [int(v) for v in target.bbox]
    cv2.imwrite(str(OUT / 'faceswap_mouth_106_points.jpg'), marked[max(0,y1-80):y2+80, max(0,x1-80):x2+80])
    source = extract_face_images(str(SOURCE), is_source_face=True)[0][0]
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = 'GFPGAN'
    roop.globals.use_enhancer = True
    original_occlusion = process_module.detect_foreground_occlusion

    def record_occlusion(swapped, reference, **kwargs):
        mask = original_occlusion(swapped, reference, **kwargs)
        cv2.imwrite(str(OUT / 'faceswap_mouth_occ_target.jpg'), reference)
        cv2.imwrite(str(OUT / 'faceswap_mouth_occ_mask.png'), np.uint8(np.clip(mask, 0, 1) * 255))
        print(f'[PROBE] occlusion fraction={np.mean(mask > .1):.3f}, mean={np.mean(mask):.3f}', flush=True)
        return mask

    process_module.detect_foreground_occlusion = record_occlusion
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [target], SimpleNamespace(blend_ratio=1.0, current_video_path=None), face_swap_mode='selected_faces')
    manager._is_single_photo = True
    original_mouth = manager._mouth_mask_in_aligned_face

    def record_mouth(*args, **kwargs):
        mask = original_mouth(*args, **kwargs)
        cv2.imwrite(str(OUT / 'faceswap_mouth_lips_mask.png'), np.uint8(np.clip(mask, 0, 1) * 255))
        print(f'[PROBE] mouth mask fraction={np.mean(mask > .5):.3f}', flush=True)
        return mask

    manager._mouth_mask_in_aligned_face = record_mouth
    result = manager._process_face_swap_v21(source, copy.deepcopy(target), frame.copy(), frame)
    cv2.imwrite(str(OUT / 'faceswap_mouth_repro.jpg'), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print('[PROBE] output=', OUT / 'faceswap_mouth_repro.jpg', flush=True)

    inspect_dark_component()


def inspect_dark_component():
    ref = cv2.imread(str(OUT / 'faceswap_mouth_occ_target.jpg'))
    lip = cv2.imread(str(OUT / 'faceswap_mouth_lips_mask.png'), cv2.IMREAD_GRAYSCALE) > 127
    gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(lip)
    lx1, lx2, ly1, ly2 = xs.min(), xs.max(), ys.min(), ys.max()
    reach = int(max(lx2-lx1, ly2-ly1) * 2.5)
    rx1, ry1 = max(0, lx1-reach), max(0, ly1-reach)
    rx2, ry2 = min(ref.shape[1], lx2+reach), min(ref.shape[0], ly2+reach)
    skin_bright = np.percentile(gray[ry1:ry2, rx1:rx2], 80)
    threshold = min(125, skin_bright * 0.62)
    dark = np.uint8(gray < threshold)
    n, comps, stats, cents = cv2.connectedComponentsWithStats(dark, 8)
    found = np.zeros_like(gray, dtype=np.uint8)
    for label in range(1, n):
        component = comps == label
        overlap = np.sum(component & cv2.dilate(np.uint8(lip), np.ones((17, 17), np.uint8)).astype(bool))
        if overlap > max(20, lip.sum() * .05) and stats[label, cv2.CC_STAT_AREA] > lip.sum() * 2:
            found[component] = 255
            print(f'[PROBE] dark component area={stats[label, cv2.CC_STAT_AREA]} overlap={overlap}', flush=True)
    cv2.imwrite(str(OUT / 'faceswap_mouth_dark_component.png'), found)
    print(f'[PROBE] dark threshold={threshold:.1f} skin80={skin_bright:.1f}', flush=True)


if __name__ == '__main__':
    main()
