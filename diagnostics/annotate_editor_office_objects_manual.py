"""Manual diagnostic annotation of visible office objects; never production.

Coordinates were traced from the source at 2x with a labelled pixel grid.
They describe this fixture only, not object detection rules or inference.
Fine leaves, inter-leaf holes and antialiasing retain manual uncertainty.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / 'diagnostics' / 'img_editor_scene_office_final.png'
OUT = ROOT / 'diagnostics' / 'out' / 'editor_object_complete_20261006' / 'annotations'

# Fixed polygons are a declared human-labelled fixture, never model output.
ARM = [(837,252),(864,274),(902,315),(957,365),(982,394),(996,418),
       (995,438),(978,460),(953,487),(920,518),(883,537),(849,545),
       (835,533),(871,512),(901,484),(925,457),(945,431),(949,416),
       (929,396),(903,371),(875,348),(851,328)]
PLANT = {
    'crown': [
        [(938,276),(945,272),(957,277),(975,266),(991,271),(1008,255),
         (1016,253),(1008,266),(1020,270),(1036,258),(1050,263),(1038,268),
         (1048,277),(1060,279),(1058,286),(1072,287),(1085,296),(1087,306),
         (1078,299),(1062,298),(1055,309),(1073,321),(1083,335),(1084,346),
         (1076,332),(1070,332),(1060,326),(1054,325),(1048,330),(1042,323),
         (1036,335),(1028,341),(1018,332),(1009,344),(994,348),(984,359),
         (970,361),(952,352),(936,339),(919,322),(909,321),(921,306),
         (932,302),(932,291),(939,283)],
        [(974,349),(989,353),(1006,349),(1022,347),(1037,346),(1053,355),
         (1069,357),(1075,367),(1081,376),(1089,387),(1086,399),(1078,386),
         (1071,378),(1068,393),(1061,385),(1053,376),(1049,394),(1042,389),
         (1035,383),(1029,401),(1024,406),(1015,401),(1011,415),(1005,416),
         (998,402),(988,397),(982,386),(970,376),(968,365)],
        [(893,374),(889,388),(900,390),(906,407),(915,416),(919,428),
         (915,440),(925,434),(934,427),(937,445),(945,438),(953,425),
         (958,414),(948,400),(940,391),(921,379),(907,373)],
    ],
    'trunk': [
        [(1018,397),(1023,397),(1021,426),(1019,453),(1018,480),
         (1013,502),(1008,515),(1003,516),(1006,499),(1010,479),
         (1011,452),(1014,425)],
        [(996,527),(1002,529),(998,539),(991,548),(980,549),(987,540)],
        [(970,490),(978,485),(979,514),(980,535),(987,545),
         (976,546),(971,531),(970,516)],
        [(1008,356),(1014,353),(1021,365),(1024,390),(1018,397),(1015,377)],
    ],
    'pot': [
        [(970,558),(991,559),(1009,557),(1007,565),(1004,575),
         (1008,585),(1027,593),(1034,596),(1027,602),(1005,608),
         (978,604),(968,598),(965,584),(964,565)],
        [(981,540),(1000,541),(1027,541),(1030,549),(1020,553),(985,552)],
    ],
}
# Inter-leaf background holes individually judged visible in the source.
PLANT_HOLES = [
    [(944,301),(950,295),(952,307),(946,315),(940,313)],
    [(968,293),(974,295),(974,305),(969,308)],
    [(996,312),(1003,311),(1005,321),(999,328),(995,322)],
    [(1027,302),(1034,304),(1034,315),(1029,317)],
    [(1021,325),(1027,322),(1029,330),(1025,338)],
    [(991,365),(998,362),(1003,371),(999,379),(994,376)],
    [(1020,370),(1026,368),(1027,380),(1021,385)],
    [(939,415),(941,406),(947,414),(946,423)],
]
CHAIR = {
    'back': [[(1105,436),(1122,434),(1174,434),(1192,438),(1200,444),
              (1198,456),(1192,479),(1183,503),(1174,526),(1168,538),
              (1153,548),(1100,551),(1076,548),(1070,542),(1064,530),
              (1068,504),(1077,478),(1092,450)]],
    'arms': [
        [(1010,516),(1027,514),(1064,516),(1076,521),(1072,527),
         (1042,525),(1028,525),(1014,527),(1008,522)],
        [(1026,525),(1035,525),(1047,562),(1036,565)],
        [(1057,526),(1065,526),(1064,560),(1057,564)],
        [(1110,531),(1130,531),(1154,533),(1171,535),(1174,540),
         (1169,546),(1141,545),(1113,544),(1103,540),(1102,535)],
        [(1161,544),(1170,542),(1174,559),(1180,584),(1182,597),
         (1177,603),(1163,602),(1161,594),(1173,593),(1168,572)],
    ],
    'seat': [[(1030,558),(1063,555),(1103,557),(1130,561),(1145,570),
              (1149,583),(1143,594),(1127,601),(1076,596),(1038,587),
              (1015,578),(1007,571),(1009,564)]],
    'base_wheels': [
        [(1094,600),(1111,601),(1111,646),(1093,648)],
        [(1070,598),(1081,599),(1082,615),(1068,615),(1065,608)],
        [(1092,644),(1108,642),(1133,650),(1189,665),(1191,676),
         (1177,676),(1110,657),(1107,687),(1095,689),(1095,658),
         (1061,663),(1021,672),(1018,661),(1062,648)],
        [(1103,637),(1132,634),(1163,640),(1168,651),(1151,651),(1113,647)],
        [(1075,638),(1098,637),(1091,646),(1060,650),(1052,645)],
    ],
}
WHEELS = [(1018,672,10,12),(1103,694,11,14),(1190,687,11,12)]


def polygon_mask(shape, polygons):
    mask = np.zeros(shape, np.uint8)
    for polygon in polygons:
        cv2.fillPoly(mask, [np.asarray(polygon, np.int32)], 255)
    return mask


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    image = np.asarray(Image.open(PHOTO).convert('RGB'))
    shape = image.shape[:2]
    arm = polygon_mask(shape, [ARM])
    Image.fromarray(arm).save(OUT / 'manual_occluding_arm.png')
    metadata = {
        'source': str(PHOTO.relative_to(ROOT)),
        'source_sha256': hashlib.sha256(PHOTO.read_bytes()).hexdigest(),
        'width': shape[1], 'height': shape[0],
        'provenance': 'MANUAL DIAGNOSTIC FIXTURE: visually traced polygons at 2x; no segmentation model or inference',
        'limitations': [
            'approximate visible boundary, especially fine leaves and antialiasing; not laboratory-grade IoU truth',
            'inter-leaf holes manually sampled; small unlabelled holes may remain',
            'occluded object surfaces are excluded, never inferred behind people/chair/table',
            'cast shadows excluded from object mask and require separate final-result visual assessment',
            '7px boundary band marks approximate annotation uncertainty; some wispy leaf errors may exceed it; neither core nor outer band implies exact border truth',
        ],
        'coordinates': {'plant': PLANT, 'plant_holes': PLANT_HOLES, 'chair': CHAIR, 'chair_wheels': WHEELS, 'occluding_arm': ARM},
        'objects': {},
    }
    panels = [image]
    for name, annotation in (('plant', PLANT), ('chair', CHAIR)):
        combined = np.zeros(shape, np.uint8)
        parts = {}
        for part, polygons in annotation.items():
            mask = polygon_mask(shape, polygons)
            if name == 'plant':
                mask[arm > 0] = 0
                if part == 'crown':
                    mask[polygon_mask(shape, PLANT_HOLES) > 0] = 0
            if name == 'chair' and part == 'base_wheels':
                for x, y, radius_x, radius_y in WHEELS:
                    cv2.ellipse(mask, (x, y), (radius_x, radius_y), 0, 0, 360, 255, -1)
            combined |= mask
            Image.fromarray(mask).save(OUT / f'{name}_{part}.png')
            parts[part] = {'pixels': int(np.count_nonzero(mask)), 'mask': f'{name}_{part}.png'}
        Image.fromarray(combined).save(OUT / f'{name}_visible_object.png')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        core = cv2.erode(combined, kernel)
        outer = cv2.dilate(combined, kernel)
        Image.fromarray(core).save(OUT / f'{name}_certain_core.png')
        Image.fromarray(outer - core).save(OUT / f'{name}_boundary_uncertainty.png')
        overlay = image.copy()
        overlay[combined > 0] = (.60 * overlay[combined > 0] + .40 * np.array([255, 0, 0])).astype(np.uint8)
        panels.append(overlay)
        Image.fromarray(overlay).save(OUT / f'{name}_overlay.png')
        metadata['objects'][name] = {'mask': f'{name}_visible_object.png', 'pixels': int(np.count_nonzero(combined)), 'coverage': float((combined > 0).mean()), 'parts': parts, 'manual_arm_overlap_pixels': int(np.count_nonzero((combined > 0) & (arm > 0)))}
    Image.fromarray(np.concatenate(panels, axis=1)).save(OUT / 'manual_objects_compare.png')
    (OUT / 'annotation.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata['objects'], indent=2))


if __name__ == '__main__':
    main()
