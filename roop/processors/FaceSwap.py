import os
import numpy as np
import insightface
import cv2
import roop.globals
from typing import Any
from roop.types import Face, Frame


class FaceSwap:
    def __init__(self) -> None:
        self.model = None
        self.model_path = None
        self.devicename = None

    def Initialize(self, options: dict) -> None:
        devnm = options.get('devicename', 'cpu')
        model_path = options.get('model', 'inswapper_128.onnx')

        if not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.getcwd(), 'models', model_path))

        if self.model is None or self.model_path != model_path or self.devicename != devnm:
            self.model_path = model_path
            self.devicename = devnm
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in devnm else ['CPUExecutionProvider']
            try:
                self.model = insightface.model_zoo.get_model(model_path, providers=providers)
                print(f"[FaceSwap] Modelo cargado: {model_path}")
            except Exception as e:
                print(f"[FaceSwap] Error carga: {e}")

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame, paste_back: bool = True) -> Any:
        if self.model is None:
            return None
        try:
            if not isinstance(temp_frame, np.ndarray):
                temp_frame = np.array(temp_frame)
            for attr in ['kps', 'embedding', 'normed_embedding', 'bbox']:
                for face in (target_face, source_face):
                    val = getattr(face, attr, None)
                    if isinstance(val, list):
                        setattr(face, attr, np.array(val, dtype=np.float32))
                    elif val is None and attr == 'normed_embedding':
                        emb = getattr(face, 'embedding', None)
                        if emb is not None:
                            nrm = np.array(emb, dtype=np.float32)
                            norm = np.linalg.norm(nrm)
                            if norm > 0:
                                nrm = nrm / norm
                            setattr(face, attr, nrm)
            try:
                res_data = self.model.get(temp_frame, target_face, source_face, paste_back=False)
            except Exception as e_inner:
                print(f"[FaceSwap] model.get() falló con paste_back=False: {e_inner}")
                print(f"[FaceSwap] Intentando paste_back=True como fallback...")
                try:
                    result = self.model.get(temp_frame, target_face, source_face, paste_back=True)
                    if result is not None:
                        return result
                except Exception as e2:
                    print(f"[FaceSwap] model.get() también falló con paste_back=True: {e2}")
                return None
            if res_data is None:
                return None
            if isinstance(res_data, tuple):
                swapped_face, M = res_data
            else:
                return res_data
            if not paste_back:
                return swapped_face
            if isinstance(swapped_face, list):
                swapped_face = np.array(swapped_face)
            if isinstance(M, list):
                M = np.array(M, dtype=np.float64)
            return self.paste_back_robust(temp_frame, swapped_face, M)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None

    def paste_back_robust(self, target_img, source_face_img, M):
        try:
            h, w = target_img.shape[:2]
            M_inv = cv2.invertAffineTransform(M)
            warped_face = cv2.warpAffine(source_face_img, M_inv, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
            mask_128 = np.zeros((128, 128), dtype=np.float32)
            cv2.ellipse(mask_128, (64, 64), (54, 72), 0, 0, 360, 1.0, -1)
            mask_128 = cv2.GaussianBlur(mask_128, (7, 7), 0)
            face_mask = cv2.warpAffine(mask_128, M_inv, (w, h), flags=cv2.INTER_LINEAR)
            face_mask = np.clip(face_mask * 2.0, 0, 1)
            if face_mask.ndim == 2: face_mask = np.expand_dims(face_mask, axis=2)
            blend = getattr(roop.globals, 'blend_ratio', 0.98)
            alpha = face_mask * blend
            result = target_img.astype(np.float32) * (1.0 - alpha) + warped_face.astype(np.float32) * alpha
            return np.clip(result, 0, 255).astype(np.uint8)
        except Exception as e:
            print(f"[ERROR] paste_back_robust: {e}")
            return target_img


def detect_mouth_open(target_face, landmarks_106, target_image):
    """Detecta si la boca está abierta. 3 niveles de fallback."""
    if target_image is not None:
        try:
            from roop.mouth_detector import detect_mouth_open_advanced
            is_open, ratio, mouth_data = detect_mouth_open_advanced(target_image)
            if mouth_data is not None:
                region = {
                    'upper_lip_top': mouth_data.get('upper_lip_top'),
                    'upper_lip_bottom': mouth_data.get('upper_lip_bottom'),
                    'lower_lip_top': mouth_data.get('lower_lip_top'),
                    'lower_lip_bottom': mouth_data.get('lower_lip_bottom'),
                    'mouth_left': mouth_data.get('mouth_left'),
                    'mouth_right': mouth_data.get('mouth_right'),
                    'mouth_height': mouth_data.get('mouth_height'),
                    'width': mouth_data.get('mouth_width', 0),
                    'method': 'mediapipe_468'
                }
                return is_open, region, ratio
        except Exception:
            pass

    if landmarks_106 is not None and len(landmarks_106) >= 106:
        try:
            upper = landmarks_106[52]
            lower = landmarks_106[58]
            left = landmarks_106[55]
            right = landmarks_106[61]
            mouth_h = abs(lower[1] - upper[1])
            mouth_w = abs(right[0] - left[0])
            if mouth_w > 0:
                ratio = mouth_h / mouth_w
                is_open = ratio > 0.08
                region = {
                    'mouth_left': (int(left[0]), int(left[1])),
                    'mouth_right': (int(right[0]), int(right[1])),
                    'mouth_height': mouth_h,
                    'width': mouth_w,
                    'method': 'landmarks_106'
                }
                return is_open, region, ratio
        except Exception:
            pass

    if hasattr(target_face, 'kps') and target_face.kps is not None and len(target_face.kps) >= 5:
        try:
            kps = target_face.kps
            left_mouth = kps[3]
            right_mouth = kps[4]
            nose = kps[2]
            mouth_w = abs(right_mouth[0] - left_mouth[0])
            mouth_center_y = (left_mouth[1] + right_mouth[1]) / 2
            expected_y = nose[1] + 0.6 * mouth_w
            vertical_diff = mouth_center_y - expected_y
            ratio = vertical_diff / (mouth_w if mouth_w > 0 else 1)
            is_open = ratio > 0.08
            region = {
                'width': mouth_w,
                'vertical_diff': vertical_diff,
                'method': 'kps_5'
            }
            return is_open, region, max(0, ratio)
        except Exception:
            pass

    return False, None, 0.0


def create_mouth_preservation_mask(target_img, mouth_region, blend_ratio=0.95):
    """Crea máscara elíptica para preservar la boca original."""
    try:
        h, w = target_img.shape[:2]
        mask = np.ones((h, w), dtype=np.float32)
        if mouth_region is None:
            return mask

        method = mouth_region.get('method', '')

        if method == 'mediapipe_468':
            ml = mouth_region.get('mouth_left', (0, 0))
            mr = mouth_region.get('mouth_right', (0, 0))
            ut = mouth_region.get('upper_lip_top', (0, 0))
            lb = mouth_region.get('lower_lip_bottom', (0, 0))
            cx = (ml[0] + mr[0]) // 2
            cy = (ut[1] + lb[1]) // 2
            mw = abs(mr[0] - ml[0])
            mh = abs(lb[1] - ut[1])
            axes_x = max(int(mw * 0.8), 20)
            axes_y = max(int(mh * 1.5), 15)
        elif method == 'landmarks_106':
            ml = mouth_region.get('mouth_left', (0, 0))
            mr = mouth_region.get('mouth_right', (0, 0))
            cx = (ml[0] + mr[0]) // 2
            cy = ml[1]
            mw = abs(mr[0] - ml[0])
            axes_x = max(int(mw * 0.8), 20)
            axes_y = max(int(axes_x * 0.8), 15)
        else:
            cx, cy = w // 2, h // 2
            axes_x = max(int(w * 0.15), 20)
            axes_y = max(int(h * 0.10), 15)

        ellipse_mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(ellipse_mask, (cx, cy), (axes_x, axes_y), 0, 0, 360, 1.0, -1)
        ellipse_mask = cv2.GaussianBlur(ellipse_mask, (15, 15), 0)
        preserve_strength = min(0.95, max(0.50, blend_ratio))
        mask = np.clip(ellipse_mask * preserve_strength, 0, 1)
        return mask

    except Exception as e:
        print(f"[FaceSwap] Error en create_mouth_preservation_mask: {e}")
        return np.ones_like(target_img[..., 0], dtype=np.float32) if target_img.ndim >= 2 else np.ones((1, 1), dtype=np.float32)