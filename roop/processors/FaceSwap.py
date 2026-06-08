import os
import numpy as np
import insightface
import cv2
import inspect
import roop.globals
from typing import Any, Tuple, Optional, Dict
from roop.types import Face, Frame


class FaceWarpEngine:
    """Landmark-based face warp using Delaunay triangulation for strong identity transfer"""
    
    @staticmethod
    def warp_face(source_img, target_img, src_kps, tgt_kps, src_landmarks_106=None, tgt_landmarks_106=None, out_size=256):
        from insightface.utils import face_align
        import numpy as np, cv2
        
        aimg_src, M_src = face_align.norm_crop2(source_img, np.array(src_kps, dtype=np.float32), out_size)
        aimg_tgt, M_tgt = face_align.norm_crop2(target_img, np.array(tgt_kps, dtype=np.float32), out_size)
        
        h, w = aimg_tgt.shape[:2]
        
        if src_landmarks_106 is not None and tgt_landmarks_106 is not None:
            src_lm = np.array(src_landmarks_106, dtype=np.float32)
            tgt_lm = np.array(tgt_landmarks_106, dtype=np.float32)
        else:
            src_lm = np.array(src_kps, dtype=np.float32)
            tgt_lm = np.array(tgt_kps, dtype=np.float32)
        
        n_pts = len(tgt_lm)
        ones = np.ones((n_pts, 1), dtype=np.float32)
        tgt_align = (M_tgt @ np.hstack([tgt_lm[:n_pts], ones]).T).T[:, :2]
        src_align = (M_src @ np.hstack([src_lm[:n_pts], ones]).T).T[:, :2]
        
        margin = 15
        extra = np.array([
            [0, 0], [w//2, 0], [w-1, 0],
            [0, h//2], [w-1, h//2],
            [0, h-1], [w//2, h-1], [w-1, h-1],
            [margin, margin], [w-margin, margin],
            [margin, h-margin], [w-margin, h-margin],
        ], dtype=np.float32)
        
        all_tgt = np.vstack([tgt_align, extra]).astype(np.float32)
        all_src = np.vstack([src_align, extra]).astype(np.float32)
        
        rect = (0, 0, w, h)
        subdiv = cv2.Subdiv2D(rect)
        for p in all_tgt:
            subdiv.insert((int(round(p[0])), int(round(p[1]))))
        
        tri_list = subdiv.getTriangleList()
        triangles = tri_list.reshape(-1, 3, 2)
        
        warped = np.zeros_like(aimg_tgt)
        count = 0
        for t in triangles:
            idxs = []
            for v in t:
                dists = np.sum((all_tgt - v) ** 2, axis=1)
                best = np.argmin(dists)
                if dists[best] < 25:
                    idxs.append(best)
            if len(set(idxs)) != 3:
                continue
            
            src_tri = np.ascontiguousarray(all_src[idxs].astype(np.float32))
            tgt_tri = np.ascontiguousarray(t.astype(np.float32))
            
            rx, ry, rw, rh = cv2.boundingRect(tgt_tri)
            if rw < 2 or rh < 2:
                continue
            
            rx_f, ry_f = float(rx), float(ry)
            tgt_offset = np.ascontiguousarray((tgt_tri - [rx_f, ry_f]).astype(np.float32))
            src_offset = np.ascontiguousarray((src_tri - [rx_f, ry_f]).astype(np.float32))
            
            warp_mat = cv2.getAffineTransform(src_offset, tgt_offset)
            
            src_roi = aimg_src[ry:ry+rh, rx:rx+rw]
            if src_roi.size == 0 or src_roi.shape[0] < 2 or src_roi.shape[1] < 2:
                continue
            
            warped_roi = cv2.warpAffine(src_roi, warp_mat, (rw, rh), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
            
            mask = np.zeros((rh, rw), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(tgt_offset), 255)
            
            roi_old = warped[ry:ry+rh, rx:rx+rw].copy()
            mask_3ch = np.stack([mask, mask, mask], axis=-1)
            roi_new = np.where(mask_3ch > 0, warped_roi, roi_old)
            warped[ry:ry+rh, rx:rx+rw] = roi_new
            count += 1
        
        return warped, M_tgt


# Cache: detect once per face-type whether normed_embedding is writable (no setter = read-only)
_writable_cache: dict[int, bool] = {}

def _is_normed_embedding_writable(face) -> bool:
    """
    Duck-type detect: can this Face object accept a write to .normed_embedding?
    insightface.Face  -> .normed_embedding is a read-only @property -> False
    roop.*.Face        -> .normed_embedding is a plain attribute  -> True
    Cached by class identity (type object id), not by name, so two Face classes
    with the same __name__ cannot collide.
    """
    key = type(face)
    if key not in _writable_cache:
        try:
            key.__setattr__(face, 'normed_embedding', None)
        except AttributeError:
            _writable_cache[key] = False
        except Exception:
            _writable_cache[key] = False
        else:
            _writable_cache[key] = True
    return _writable_cache[key]


def _normalize_source_embedding(source_face: Any) -> bool:
    """
    Normalize source_face.embedding into a unit-norm embedding, handling both
    writable and read-only .normed_embedding (insightface.Face).

    Returns True on success, False if embedding was absent or normalization failed.
    Works with:
      - roop.face_util.Face    (normed_embedding writable  -> sets both attrs)
      - roop.types.Face        (normed_embedding writable  -> sets both attrs)
      - insightface.Face       (normed_embedding wirter-only  -> only sets embedding so the property computes correctly)
    """
    import numpy as np
    emb = getattr(source_face, 'embedding', None)
    if emb is None:
        return False
    emb = np.array(emb, dtype=np.float32).flatten()
    if emb.shape[0] > 512:
        emb = emb[:512]
    norm = np.linalg.norm(emb)
    if norm <= 0:
        return False
    normalized = emb / norm
    source_face.embedding = normalized
    if _is_normed_embedding_writable(source_face):
        source_face.normed_embedding = normalized
    return True


class FaceSwap:
    def __init__(self) -> None:
        self.model = None
        self.model_path = None
        self.devicename = None
        self.is_256 = False

    def Initialize(self, options: dict) -> None:
        devnm = options.get('devicename', 'cpu')
        
        # Lista de candidatos en orden de preferencia
        candidates = [
            os.path.abspath(os.path.join(os.getcwd(), 'models', 'inswapper_256.onnx')),
            os.path.abspath(os.path.join(os.getcwd(), 'models', 'inswapper_128_facefusion.onnx')),
            os.path.abspath(os.path.join(os.getcwd(), 'models', 'inswapper_128.onnx'))
        ]
        
        model_path = None
        for path in candidates:
            if os.path.exists(path):
                # Verificar si el modelo es compatible con embeddings (rank 2) o requiere imagen (rank 4)
                try:
                    import onnxruntime as ort
                    temp_sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                    source_input = next(i for i in temp_sess.get_inputs() if i.name == 'source')
                    if len(source_input.shape) == 2: # [1, 512] -> Compatible
                        model_path = path
                        break
                    else:
                        print(f"[FaceSwap] Saltando {os.path.basename(path)}: requiere imagen como entrada, no compatible con ADN Maestro.")
                except Exception as e:
                    print(f"[FaceSwap] Error verificando {os.path.basename(path)}: {e}")
                    continue

        if model_path is None:
            print("[FaceSwap] ❌ ERROR: No se encontró ningún modelo inswapper compatible en /models")
            return

        self.model_path = model_path
        self.devicename = devnm
        self.is_256 = '256' in os.path.basename(model_path)
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in devnm else ['CPUExecutionProvider']
        
        try:
            print(f"[FaceSwap] Cargando modelo compatible: {os.path.basename(model_path)} (Resolución {'256px' if self.is_256 else '128px'})")
            
            from insightface.model_zoo.inswapper import INSwapper
            
            self.model = INSwapper(model_file=model_path, session=None)
            self.model.session = ort.InferenceSession(model_path, providers=providers)
            
            print(f"[FaceSwap] Modelo inicializado correctamente")
            
            if type(self.model).__name__ != 'INSwapper':
                print(f"[FaceSwap] ❌ ERROR: El modelo cargado no es INSwapper, es {type(self.model).__name__}")
                self.model = None
        except Exception as e:
            print(f"[FaceSwap] Error crítico inicializando modelo: {e}")
            self.model = None

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame, paste_back: bool = True) -> Any:
        try:
            if not isinstance(temp_frame, np.ndarray):
                temp_frame = np.array(temp_frame)
            
            for face in (target_face, source_face):
                for attr in ['kps', 'bbox', 'embedding']:
                    val = getattr(face, attr, None)
                    if val is not None and isinstance(val, (list, tuple)):
                        try:
                            setattr(face, attr, np.array(val, dtype=np.float32))
                        except (AttributeError, TypeError):
                            pass

            # 1. Intentar primero con el MODELO IA (Máxima fidelidad)
            if self.model is None or type(self.model).__name__ != 'INSwapper':
                self.Initialize({"devicename": getattr(self, 'devicename', 'cpu')})
            
            res_data = None
            if self.model is not None:
                _normalize_source_embedding(source_face)
                try:
                    # Inswapper_128 soporta paste_back=False, Inswapper_256 (FF) puede variar
                    try:
                        res_data = self.model.get(temp_frame, target_face, source_face, paste_back=False)
                    except TypeError:
                        res_data = self.model.get(temp_frame, target_face, source_face)
                except Exception as e:
                    print(f"[FaceSwap] Error modelo IA: {e}")

            # 2. FALLBACK: Si la IA falló, intentar Warp (solo si es necesario)
            if res_data is None:
                print("[FaceSwap] Modelo IA falló o no disponible, intentando Fallback Warp...")
                res_data = self._run_warp(temp_frame, target_face, source_face, paste_back=False)
            
            if res_data is None:
                return None
            
            # 3. POST-PROCESO: Normalizar a 256px para el resto del pipeline (v5.2)
            if isinstance(res_data, tuple):
                swapped_face, M = res_data
                out_h, out_w = swapped_face.shape[:2]
                
                if out_w != 256:
                    scale = 256.0 / out_w
                    swapped_face = cv2.resize(swapped_face, (256, 256), interpolation=cv2.INTER_CUBIC)
                    # Escalar solo la parte de transformación, no la de traslación si es necesario, 
                    # pero en matrices de 2x3 de insightface, multiplicar todo por el factor de escala
                    # suele ser correcto porque la traslación también está en píxeles del crop.
                    M = M * scale
                
                res_data = (swapped_face, M)

            if not paste_back:
                return res_data

            if isinstance(res_data, tuple):
                swapped_face, M = res_data
                return self.paste_back_robust(temp_frame, swapped_face, M)
            else:
                return res_data
                
        except Exception as e:
            print(f"[FaceSwap] Error en Run: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_warp(self, temp_frame, target_face, source_face, paste_back=True):
        """Landmark-based face warp using Delaunay triangulation for identity transfer"""
        try:
            src_img_bgr = None
            src_path = getattr(source_face, 'source_image', None)
            if src_path and os.path.isfile(src_path):
                src_full = cv2.imread(src_path)
                if src_full is not None and src_full.size > 0:
                    src_img_bgr = src_full
            if src_img_bgr is None:
                src_img = getattr(source_face, 'face_img_ref', None)
                if src_img is None:
                    src_img = getattr(source_face, 'face_img', None)
                if src_img is not None:
                    if src_img.shape[2] == 3:
                        src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)
                    elif src_img.shape[2] == 4:
                        src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGBA2BGR)
                    else:
                        src_img_bgr = src_img.copy()
            if src_img_bgr is None:
                print("[FaceSwap] No source image available for warp")
                return None

            src_kps = np.array(source_face.kps, dtype=np.float32)
            tgt_kps = np.array(target_face.kps, dtype=np.float32)
            
            src_106 = getattr(source_face, 'landmark_2d_106', None)
            tgt_106 = getattr(target_face, 'landmark_2d_106', None)

            warped_face, M = FaceWarpEngine.warp_face(
                src_img_bgr, temp_frame, src_kps, tgt_kps,
                src_landmarks_106=src_106, tgt_landmarks_106=tgt_106
            )

            _dbg = os.path.join(os.path.dirname(__file__), '..', 'debug_swap')
            os.makedirs(_dbg, exist_ok=True)
            _fnum = getattr(FaceSwap, '_debug_warp', 0) + 1
            FaceSwap._debug_warp = _fnum
            if _fnum <= 3:
                cv2.imwrite(os.path.join(_dbg, f'warp_result_f{_fnum}.png'), warped_face)

            if not paste_back:
                return warped_face, M

            return self.paste_back_robust(temp_frame, warped_face, M)

        except Exception as e:
            print(f"[FaceSwap] _run_warp error: {e}")
            import traceback as tb
            tb.print_exc()
            return None

    def paste_back_robust(self, target_img, source_face_img, M):
        try:
            from roop.quality_enhancements import create_soft_mask, detect_foreground_occlusion, blend_with_poisson
            
            if source_face_img is None or target_img is None:
                return target_img

            M = np.asarray(M, dtype=np.float32)
            if M.shape == (3, 3):
                M = M[:2, :]
            if M.shape != (2, 3):
                return target_img

            h, w = target_img.shape[:2]
            M_inv = cv2.invertAffineTransform(M)
            
            # Warp de la cara swappeada al frame original (M: original->aligned, usar inversa)
            warped_face = cv2.warpAffine(source_face_img, M_inv, (w, h), borderMode=cv2.BORDER_REPLICATE)
            
            # Crear máscara para el pegado
            bbox = [0, 0, source_face_img.shape[1], source_face_img.shape[0]]
            pts = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], [bbox[2], bbox[3]], [bbox[0], bbox[3]]], dtype=np.float32)
            pts_warped = cv2.transform(pts.reshape(-1, 1, 2), M_inv)
            
            x_coords = pts_warped[:, 0, 0]
            y_coords = pts_warped[:, 0, 1]
            target_bbox = [np.min(x_coords), np.min(y_coords), np.max(x_coords), np.max(y_coords)]
            
            # Detección de oclusiones
            face_w, face_h = source_face_img.shape[1], source_face_img.shape[0]
            target_region_warped = cv2.warpAffine(target_img, M_inv, (face_w, face_h), borderMode=cv2.BORDER_REPLICATE)
            occ_mask = detect_foreground_occlusion(source_face_img, target_region_warped)
            
            # Crear máscara suave (feather bajo para no afectar exterior)
            mask = create_soft_mask(target_bbox, (h, w), feather=5, occlusion_mask=occ_mask)
            
            mask_3ch = cv2.merge([mask, mask, mask])
            result = (warped_face * mask_3ch + target_img * (1 - mask_3ch)).astype(np.uint8)
            
            return result
        except Exception as e:
            print(f"[FaceSwap] Error en paste_back_robust: {e}")
            return target_img


# --- FUNCIONES DE PRESERVACIÓN DE BOCA ---

def detect_mouth_open(target_face, landmarks_106, image) -> Tuple[bool, Optional[Dict], float]:
    """Detecta boca abierta usando MouthDetector avanzado, con fallback a landmarks 106"""
    try:
        from roop.mouth_detector import get_mouth_detector
        detector = get_mouth_detector()
        if detector and detector.is_initialized:
            x1, y1, x2, y2 = target_face.bbox
            h, w = image.shape[:2]
            pw, ph = int((x2-x1)*0.2), int((y2-y1)*0.2)
            x1, y1 = max(0, int(x1-pw)), max(0, int(y1-ph))
            x2, y2 = min(w, int(x2+pw)), min(h, int(y2+ph))

            face_roi = image[y1:y2, x1:x2]
            if face_roi.size == 0:
                return False, None, 0.0

            is_open, ratio, mouth_data = detector.detect_mouth_open(face_roi)

            if mouth_data:
                landmark_keys = {'upper_lip_top', 'upper_lip_bottom', 'lower_lip_top', 'lower_lip_bottom', 'mouth_left', 'mouth_right', 'upper_lip_left_curve', 'upper_lip_right_curve', 'lower_lip_left_curve', 'lower_lip_right_curve', 'mouth_center_top', 'mouth_center_bottom', 'upper_lip_center', 'lower_lip_center'}
                for key, val in mouth_data.items():
                    if key in landmark_keys and isinstance(val, tuple) and len(val) == 2:
                        mouth_data[key] = (val[0] + x1, val[1] + y1)
                mouth_data['roi_offset'] = (x1, y1)
                mouth_data['roi_size'] = (face_roi.shape[1], face_roi.shape[0])
                return is_open, mouth_data, ratio

            # Fallback: MediaPipe no dio datos, usar landmarks_106 si están disponibles
            if landmarks_106 is not None and len(landmarks_106) >= 68:
                mouth_pts = landmarks_106[52:68]
                if len(mouth_pts) >= 4:
                    upper_y = min(p[1] for p in mouth_pts[:8])
                    lower_y = max(p[1] for p in mouth_pts[8:])
                    mouth_height = lower_y - upper_y
                    face_height = y2 - y1
                    if face_height > 0:
                        ratio_est = mouth_height / face_height
                        is_open_est = ratio_est > 0.035
                        mouth_data = {
                            'mouth_left': (int(mouth_pts[0][0]), int(mouth_pts[0][1])),
                            'mouth_right': (int(mouth_pts[6][0]), int(mouth_pts[6][1])),
                            'upper_lip_top': (int(mouth_pts[3][0]), int(upper_y)),
                            'lower_lip_bottom': (int(mouth_pts[11][0]), int(lower_y)),
                        }
                        return is_open_est, mouth_data, ratio_est

            return is_open, mouth_data, ratio
    except Exception as e:
        print(f"[MOUTH_DETECT] Error: {e}")
    return False, None, 0.0


def create_mouth_preservation_mask(image, mouth_data, blend_ratio=0.5) -> np.ndarray:
    """Crea máscara suave para la boca con polígono detallado + elipse amplia (v5.4)"""
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    if not mouth_data:
        return mask

    try:
        # ========== PARTE 1: Polígono de precisión (existente) ==========
        full_order = [
            'mouth_left',
            'upper_lip_left_curve',
            'upper_lip_top',
            'upper_lip_right_curve',
            'mouth_right',
            'lower_lip_right_curve',
            'lower_lip_bottom',
            'lower_lip_left_curve',
        ]
        
        inner_order = [
            'mouth_left',
            'upper_lip_bottom',
            'mouth_right',
            'lower_lip_top'
        ]

        pts_full = [mouth_data[k] for key in full_order if (k := key) in mouth_data]
        pts_inner = [mouth_data[k] for key in inner_order if (k := key) in mouth_data]

        if len(pts_full) >= 3:
            pts_full_arr = np.array(pts_full, dtype=np.int32)
            cv2.fillPoly(mask, [pts_full_arr], 0.80)
            if len(pts_inner) >= 3:
                pts_inner_arr = np.array(pts_inner, dtype=np.int32)
                cv2.fillPoly(mask, [pts_inner_arr], 1.0)

            # Dilatación masiva para expandir la máscara más allá de los labios exactos
            mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=5)
            mask = cv2.GaussianBlur(mask, (31, 31), 0)
            mask = np.clip(mask, 0, 1.0)

        # ========== PARTE 2: Elipse amplia centrada en la boca (NUEVO v5.4) ==========
        # Calcular centro de la boca
        cx, cy = 0, 0
        if 'mouth_center_top' in mouth_data and 'mouth_center_bottom' in mouth_data:
            cx_top, cy_top = mouth_data['mouth_center_top']
            cx_bot, cy_bot = mouth_data['mouth_center_bottom']
            cx = int((cx_top + cx_bot) / 2)
            cy = int((cy_top + cy_bot) / 2)
        elif 'mouth_left' in mouth_data and 'mouth_right' in mouth_data:
            ml = mouth_data['mouth_left']
            mr = mouth_data['mouth_right']
            cx = int((ml[0] + mr[0]) / 2)
            cy = int((ml[1] + mr[1]) / 2)

        if cx > 0 and cy > 0:
            if 'mouth_left' in mouth_data and 'mouth_right' in mouth_data:
                mouth_w = abs(mouth_data['mouth_right'][0] - mouth_data['mouth_left'][0])
            else:
                mouth_w = 100
            if 'upper_lip_top' in mouth_data and 'lower_lip_bottom' in mouth_data:
                mouth_h = abs(mouth_data['lower_lip_bottom'][1] - mouth_data['upper_lip_top'][1])
            else:
                mouth_h = 50

            # Radio amplio: 3.0x ancho de boca, 4.5x alto (cubre mejillas y mentón)
            rx = max(int(mouth_w * 3.0), 130)
            ry = max(int(mouth_h * 4.5), 110)

            ellipse_mask = np.zeros(image.shape[:2], dtype=np.float32)
            cv2.ellipse(ellipse_mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
            ellipse_mask = cv2.GaussianBlur(ellipse_mask, (31, 31), 0)

            # Combinar: máximo entre polígono expandido y elipse
            mask = np.maximum(mask, ellipse_mask * 0.75)

    except Exception as e:
        print(f"[MOUTH_MASK] Error: {e}")

    return mask * blend_ratio
