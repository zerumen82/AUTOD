"""Measure independent MediaPipe support for accepted source detections."""
from pathlib import Path
import json
import sys

import cv2
import mediapipe as mp
import numpy as np


def iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(overlap / max(1, np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap))


def main():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads((root/'diagnostics/source_audit_output/candidates.json').read_text())
    bank = Path(sys.argv[1])
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=3,
                                       min_detection_confidence=.5) as mesh, \
         mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=.5) as detector:
        for row in rows:
            img = cv2.imread(str(bank/row['name']))
            h,w = img.shape[:2]
            x1,y1,x2,y2 = row['bbox']
            px,py = int((x2-x1)*.5),int((y2-y1)*.5)
            cx1,cy1,cx2,cy2=max(0,x1-px),max(0,y1-py),min(w,x2+px),min(h,y2+py)
            crop = cv2.cvtColor(img[cy1:cy2,cx1:cx2], cv2.COLOR_BGR2RGB)
            if max(crop.shape[:2]) > 640:
                factor=640/max(crop.shape[:2])
                crop=cv2.resize(crop,None,fx=factor,fy=factor)
            result=mesh.process(crop)
            boxes=[]
            for face in result.multi_face_landmarks or []:
                points=np.array([[p.x,p.y] for p in face.landmark])
                lo=points.min(axis=0)*[cx2-cx1,cy2-cy1]+[cx1,cy1]
                hi=points.max(axis=0)*[cx2-cx1,cy2-cy1]+[cx1,cy1]
                boxes.append([*lo,*hi])
            overlap=max((iou(row['bbox'],box) for box in boxes),default=0)
            detected=detector.process(crop)
            support=0.0
            for face in detected.detections or []:
                b=face.location_data.relative_bounding_box
                box=[cx1+b.xmin*(cx2-cx1),cy1+b.ymin*(cy2-cy1),
                     cx1+(b.xmin+b.width)*(cx2-cx1),cy1+(b.ymin+b.height)*(cy2-cy1)]
                support=max(support,iou(row['bbox'],box))
            print(f"{row['name']}: det={row['confidence']:.2f} mesh_iou={overlap:.2f} full_detector_iou={support:.2f}",flush=True)


if __name__=='__main__':
    main()
