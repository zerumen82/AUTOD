import os
import numpy as np
import cv2
import roop.globals
from roop.path_helper import get_root_path
from roop.faceswap_config import get_faceswap_config, landmarks_fit_bbox
from typing import Any, Optional
from roop.types import Face, Frame


class FaceWarpEngine:
    """Landmark-based face warp using Delaunay triangulation for strong identity transfer"""

    @staticmethod
    def arcface_template(image_size_out):
        """Use the local InsightFace convention, including the 128px x shift."""
        from insightface.utils import face_align

        size = int(image_size_out)
        if size <= 0 or size != image_size_out:
            raise ValueError('Alignment output size must be a positive integer')
        if size % 112 == 0:
            ratio, shift = size / 112.0, 0.0
        elif size % 128 == 0:
            ratio = size / 128.0
            shift = (128 - 112) / 2 * ratio
        else:
            raise ValueError('Alignment output size must follow InsightFace 112/128 convention')
        template = face_align.arcface_dst.copy() * ratio
        template[:, 0] += shift
        return template

    @staticmethod
    def estimate_constrained_norm(
        image_size,
        kps,
        image_size_out=128,
        bbox=None,
        max_translate_px=None,
        max_scale_dev=None,
        crop_to_bbox_ratio=None,
        translate_bbox_fraction=None,
    ):
        """Keep InsightFace M unless its inverse crop scale/center escapes bbox.

        The inverse square crop side is compared with sqrt(bbox area), so
        both bbox dimensions contribute without treating profile eye spacing
        as face size. Outside the configured band, preserve roll, constrain
        scale and anchor the template centroid to the measured bbox center.
        Invalid geometry/configuration raises ValueError; no image is allocated.
        Points outside their bbox are rejected even if a caller bypasses
        ProcessMgr's earlier landmark validity gate.
        """
        from insightface.utils import face_align

        dimensions = np.asarray(image_size, dtype=np.float64)
        points = np.asarray(kps, dtype=np.float32)
        bounds = np.asarray(bbox, dtype=np.float64)
        if dimensions.shape != (2,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0):
            raise ValueError('Alignment requires positive finite image dimensions')
        if points.shape != (5, 2) or not np.isfinite(points).all():
            raise ValueError('Alignment requires five finite landmarks')
        if bounds.shape != (4,) or not np.isfinite(bounds).all():
            raise ValueError('Alignment requires a finite bbox')
        bbox_size = bounds[2:] - bounds[:2]
        if np.any(bbox_size <= 0):
            raise ValueError('Alignment requires a bbox with positive width and height')
        if not landmarks_fit_bbox(points, bounds):
            raise ValueError('Alignment landmarks lie outside their face bbox')
        template = FaceWarpEngine.arcface_template(image_size_out).astype(np.float64)
        config = get_faceswap_config()
        scale_dev = float(config['align_constrain_scale_dev'] if max_scale_dev is None else max_scale_dev)
        crop_ratio = float(config['align_constrain_crop_to_bbox_ratio'] if crop_to_bbox_ratio is None else crop_to_bbox_ratio)
        translate_fraction = float(config['align_constrain_translate_bbox_fraction'] if translate_bbox_fraction is None else translate_bbox_fraction)
        if not np.isfinite([scale_dev, crop_ratio, translate_fraction]).all():
            raise ValueError('Alignment constraints must be finite')
        if not 0 <= scale_dev < 1 or crop_ratio <= 0 or translate_fraction < 0:
            raise ValueError('Alignment constraint ranges are invalid')
        if max_translate_px is None:
            translate_limit = bbox_size * translate_fraction
        else:
            translate_px = float(max_translate_px)
            if not np.isfinite(translate_px) or translate_px < 0:
                raise ValueError('Alignment translation limit must be finite and nonnegative')
            translate_limit = np.full(2, translate_px)

        # estimate_norm returns the same matrix as norm_crop2 without allocating
        # a dummy full-resolution frame and a crop for every processed face.
        m_kps = face_align.estimate_norm(points, int(image_size_out))
        if not np.isfinite(m_kps).all():
            raise ValueError('Landmarks do not yield a finite alignment')
        determinant = float(np.linalg.det(m_kps[:, :2]))
        if determinant <= np.finfo(np.float64).eps:
            raise ValueError('Landmarks do not yield an invertible alignment')
        inverse = cv2.invertAffineTransform(m_kps)
        inverse_scale = 1 / np.sqrt(determinant)
        expected_side = np.sqrt(bbox_size[0] * bbox_size[1]) * crop_ratio
        minimum_side = expected_side * (1 - scale_dev)
        maximum_side = expected_side * (1 + scale_dev)
        crop_side = int(image_size_out) * inverse_scale
        center = (bounds[:2] + bounds[2:]) / 2
        template_center = template.mean(axis=0)
        projected_center = inverse[:, :2] @ template_center + inverse[:, 2]
        if minimum_side <= crop_side <= maximum_side and np.all(np.abs(projected_center - center) <= translate_limit):
            return m_kps, 'kps_ok'

        constrained_side = np.clip(crop_side, minimum_side, maximum_side)
        corrected_inverse = inverse.copy()
        corrected_inverse[:, :2] *= constrained_side / crop_side
        corrected_inverse[:, 2] = center - corrected_inverse[:, :2] @ template_center
        constrained = cv2.invertAffineTransform(corrected_inverse)
        if not np.isfinite(constrained).all() or np.linalg.det(constrained[:, :2]) <= 0:
            raise ValueError('BBox constraint did not yield a valid alignment')
        return constrained, 'bbox_anchored'

    
    @staticmethod
    def warp_face(source_img, target_img, src_kps, tgt_kps, src_landmarks_106=None, tgt_landmarks_106=None, out_size=256):
        from insightface.utils import face_align
        import numpy as np
        import cv2
        
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

            h_s, w_s = aimg_src.shape[:2]
            src_x1 = max(0, int(np.floor(np.min(src_tri[:, 0]))))
            src_y1 = max(0, int(np.floor(np.min(src_tri[:, 1]))))
            src_x2 = min(w_s, int(np.ceil(np.max(src_tri[:, 0]))))
            src_y2 = min(h_s, int(np.ceil(np.max(src_tri[:, 1]))))
            rw_s, rh_s = src_x2 - src_x1, src_y2 - src_y1
            if rw_s < 2 or rh_s < 2:
                continue

            tgt_offset = np.ascontiguousarray((tgt_tri - [float(rx), float(ry)]).astype(np.float32))
            src_offset = np.ascontiguousarray((src_tri - [float(src_x1), float(src_y1)]).astype(np.float32))
            
            warp_mat = cv2.getAffineTransform(src_offset, tgt_offset)
            src_roi = aimg_src[src_y1:src_y1+rh_s, src_x1:src_x1+rw_s]
            if src_roi.size == 0:
                continue
            
            warped_roi = cv2.warpAffine(src_roi, warp_mat, (rw, rh), None, cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101)
            mask = np.zeros((rh, rw), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(tgt_offset), 255)
            
            warped[ry:ry+rh, rx:rx+rw] = cv2.bitwise_and(warped_roi, warped_roi, mask=mask) + \
                                         cv2.bitwise_and(warped[ry:ry+rh, rx:rx+rw], warped[ry:ry+rh, rx:rx+rw], mask=cv2.bitwise_not(mask))
        
        return warped, M_tgt


def _normalize_source_embedding(source_face: Face) -> None:
    if hasattr(source_face, 'embedding') and source_face.embedding is not None:
        emb = np.array(source_face.embedding, dtype=np.float32).flatten()
        norm = np.linalg.norm(emb)
        if norm > 0:
            source_face.embedding = (emb / norm).reshape(1, -1)


class FaceSwap:
    def __init__(self) -> None:
        self.model = None
        self.model_path = None
        self.devicename = None
        self.is_256 = False
        self.source_is_image = False

    def Initialize(self, options: dict) -> None:
        devnm = options.get('devicename', 'cpu')
        
        root = get_root_path()
        candidates = [
            os.path.abspath(os.path.join(root, 'models', 'inswapper_128_facefusion.onnx')),
            os.path.abspath(os.path.join(root, 'models', 'inswapper_128.onnx')),
            os.path.abspath(os.path.join(root, 'models', 'inswapper_256.onnx'))
        ]
        
        model_path = None
        for path in candidates:
            if os.path.exists(path):
                try:
                    import onnxruntime as ort
                    temp_sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
                    inputs = temp_sess.get_inputs()
                    source_input = next((i for i in inputs if i.name == 'source'), inputs[1])
                    shape_len = len(source_input.shape)
                    if shape_len == 2 or shape_len == 4:
                        model_path = path
                        self.is_256 = '256' in os.path.basename(path)
                        self.source_is_image = (shape_len == 4)
                        print(f"[FaceSwap] Usando {os.path.basename(path)} (source shape={source_input.shape}, {'IMAGE' if self.source_is_image else 'EMBEDDING'})")
                        break
                except Exception:
                    continue

        if model_path is None:
            print("[FaceSwap] ❌ ERROR: No se encontró ningún modelo inswapper compatible en /models")
            return

        self.model_path = model_path
        self.devicename = devnm
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in devnm else ['CPUExecutionProvider']
        
        try:
            from insightface.model_zoo.inswapper import INSwapper
            session = temp_sess if providers == ['CPUExecutionProvider'] else ort.InferenceSession(model_path, providers=providers)
            self.model = INSwapper(model_file=model_path, session=session)
            print(f"[FaceSwap] Modelo {os.path.basename(model_path)} inicializado correctamente.")
        except Exception as e:
            print(f"[FaceSwap] Error crítico inicializando modelo: {e}")
            self.model = None

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame, paste_back: bool = True) -> Any:
        try:
            if self.model is None:
                self.Initialize({"devicename": getattr(self, 'devicename', 'cpu')})
            
            # v5.22: Forzar conversión a numpy arrays para evitar errores de shape (list -> array)
            for face in [source_face, target_face]:
                if hasattr(face, 'kps') and isinstance(face.kps, (list, tuple)):
                    face.kps = np.array(face.kps, dtype=np.float32)
                if hasattr(face, 'bbox') and isinstance(face.bbox, (list, tuple)):
                    face.bbox = np.array(face.bbox, dtype=np.float32)
            
            res_data = None
            if self.model is not None:
                if self.source_is_image:
                    if getattr(roop.globals, 'log_level', 'error') == 'debug':
                        print("[FaceSwap] Usando pipeline 256px (IMAGE)")
                    try:
                        from insightface.utils import face_align
                        src_img = getattr(source_face, 'face_img', getattr(source_face, 'face_img_ref', None))
                        if src_img is not None:
                            # v5.40: Revertir a Pipeline de 256px BGR 0-255 (Estándar Inswapper)
                            src_kps = np.array(source_face.kps, dtype=np.float32)
                            tgt_kps = np.array(target_face.kps, dtype=np.float32)
                            
                            # 1. Alinear (norm_crop2 devuelve BGR 0-255)
                            # H-FS-004: M constreñida por bbox — norm_crop2 solo usa
                            # 5 kps y con kps degenerados genera warp 1,6x
                            # desplazado. Si M se dispara, se re-estima anclada.
                            _tbbox = getattr(target_face, 'bbox', None)
                            try:
                                M_tgt, _m_how = FaceWarpEngine.estimate_constrained_norm(
                                    temp_frame.shape[:2], tgt_kps, 256, bbox=_tbbox,
                                )
                            except (ValueError, TypeError, OSError, KeyError) as alignment_error:
                                print(f"[FaceSwap] Invalid target alignment/config: {alignment_error}", flush=True)
                                return None
                            aimg_src, _ = face_align.norm_crop2(src_img, src_kps, 256)
                            aimg_tgt = cv2.warpAffine(
                                temp_frame, M_tgt, (256, 256),
                                borderMode=cv2.BORDER_REPLICATE,
                            )
                            
                            # 2. Preparar blobs [1, 3, 256, 256] BGR 0-255 (Sin normalización manual)
                            blob_src = aimg_src.astype(np.float32)
                            blob_tgt = aimg_tgt.astype(np.float32)
                            
                            # HWC -> NCHW
                            blob_src = blob_src.transpose(2, 0, 1)[np.newaxis, ...]
                            blob_tgt = blob_tgt.transpose(2, 0, 1)[np.newaxis, ...]
                            
                            # 3. Inferencia
                            res = self.model.session.run(None, {
                                self.model.input_names[0]: blob_tgt,
                                self.model.input_names[1]: blob_src
                            })[0]
                            
                            # 4. Des-normalizar salida (NCHW -> HWC)
                            swapped_face = res[0].transpose(1, 2, 0)
                            
                            # v5.40: Detección robusta de escala de salida (0-1 vs 0-255)
                            if swapped_face.max() < 2.0:
                                swapped_face = (swapped_face * 255.0)
                            
                            swapped_face = swapped_face.clip(0, 255).astype(np.uint8)
                            # NO convertir a BGR si ya viene del pipeline BGR
                            # swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)
                            
                            res_data = (swapped_face, M_tgt)
                        else:
                            res_data = self._run_warp(temp_frame, target_face, source_face, paste_back=False)
                    except Exception as e:
                        print(f"[FaceSwap] Error pipeline 256px (v5.40): {e}", flush=True)
                        res_data = self._run_warp(temp_frame, target_face, source_face, paste_back=False)
                else:
                    # v5.40: Pipeline manual para modelos 128px (embedding-based) BGR 0-255
                    if getattr(roop.globals, 'log_level', 'error') == 'debug':
                        print("[FaceSwap] Usando pipeline 128px (EMBEDDING)")
                    try:
                        from insightface.utils import face_align
                        src_emb = getattr(source_face, 'embedding', None)
                        if src_emb is not None:
                            src_emb = np.array(src_emb, dtype=np.float32).flatten()
                            src_emb = src_emb / np.linalg.norm(src_emb)
                            tgt_kps = np.array(target_face.kps, dtype=np.float32)

                            # H-FS-004: misma M constreñida en producción 128.
                            _tbbox128 = getattr(target_face, 'bbox', None)
                            try:
                                M_tgt, _m_how128 = FaceWarpEngine.estimate_constrained_norm(
                                    temp_frame.shape[:2], tgt_kps, 128, bbox=_tbbox128,
                                )
                            except (ValueError, TypeError, OSError, KeyError) as alignment_error:
                                print(f"[FaceSwap] Invalid target alignment/config: {alignment_error}", flush=True)
                                return None
                            aimg_tgt = cv2.warpAffine(
                                temp_frame, M_tgt, (128, 128),
                                borderMode=cv2.BORDER_REPLICATE,
                            )
                            
                            # v5.41: Ambos modelos inswapper_128 esperan input [0,1] float32.
                            # El wrapper INSwapper de insightface normaliza a [0,1] internamente,
                            # pero nuestro pipeline bypassa el wrapper y usa raw ONNX session.
                            # Si pasamos [0,255] las activaciones saturan → output blanco (>90% px >250).
                            # v5.46: Convert target crop from BGR to RGB (model expects RGB) and normalize to [0,1]
                            aimg_tgt_rgb = cv2.cvtColor(aimg_tgt, cv2.COLOR_BGR2RGB)
                            blob_tgt = aimg_tgt_rgb.astype(np.float32) / 255.0
                            blob_tgt = blob_tgt.transpose(2, 0, 1)[np.newaxis, ...]
                            
                            # v5.46: Apply emap projection matrix from the model to the source face embedding
                            if hasattr(self.model, 'emap') and self.model.emap is not None:
                                blob_src = np.dot(src_emb.reshape((1, -1)), self.model.emap)
                                blob_src /= np.linalg.norm(blob_src)
                            else:
                                blob_src = src_emb[np.newaxis, :]
                            
                            res = self.model.session.run(None, {
                                self.model.input_names[0]: blob_tgt,
                                self.model.input_names[1]: blob_src
                            })[0]
                            
                            swapped_face = res[0].transpose(1, 2, 0)
                            if swapped_face.max() < 2.0:
                                swapped_face = (swapped_face * 255.0)
                                
                            swapped_face = swapped_face.clip(0, 255).astype(np.uint8)
                            # v5.46: Convert output RGB back to BGR for the rest of the pipeline
                            swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)
                            res_data = (swapped_face, M_tgt)
                        else:
                            res_data = self._run_warp(temp_frame, target_face, source_face, paste_back=False)
                    except Exception as e:
                        print(f"[FaceSwap] Error pipeline 128px (v5.40): {e}", flush=True)
                        res_data = self._run_warp(temp_frame, target_face, source_face, paste_back=False)

            if res_data is None:
                if getattr(roop.globals, 'log_level', 'error') == 'debug':
                    print("[FaceSwap] ⚠️ Fallback a WARP GEOMÉTRICO (modelo falló o no disponible)")
                res_data = self._run_warp(temp_frame, target_face, source_face, paste_back=False)
            
            if res_data is None:
                return None
            
            if isinstance(res_data, tuple):
                swapped_face, M = res_data
                if swapped_face.shape[1] != 256:
                    scale = 256.0 / swapped_face.shape[1]
                    swapped_face = cv2.resize(swapped_face, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    M = M * scale
                res_data = (swapped_face, M)

            if not paste_back:
                return res_data
            if isinstance(res_data, tuple):
                return self.paste_back_robust(temp_frame, res_data[0], res_data[1])
            return res_data
                
        except Exception as e:
            print(f"[FaceSwap] Error en Run: {e}")
            return None

    def _run_warp(self, temp_frame, target_face, source_face, paste_back=True):
        try:
            src_img = getattr(source_face, 'face_img_ref', getattr(source_face, 'face_img', None))
            if src_img is None:
                return None
            src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR) if src_img.shape[2] >= 3 else src_img
            
            warped_face, M = FaceWarpEngine.warp_face(
                src_img_bgr, temp_frame, np.array(source_face.kps), np.array(target_face.kps)
            )
            if not paste_back:
                return warped_face, M
            return self.paste_back_robust(temp_frame, warped_face, M)
        except Exception:
            return None

    def paste_back_robust(self, target_img, source_face_img, M):
        try:
            from roop.quality_enhancements import blend_with_poisson
            h, w = target_img.shape[:2]
            M_inv = cv2.invertAffineTransform(M)
            warped_face = cv2.warpAffine(source_face_img, M_inv, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            mask = np.zeros((source_face_img.shape[0], source_face_img.shape[1]), dtype=np.float32)
            cv2.ellipse(mask, (128, 128), (95, 125), 0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (15, 15), 0)
            warped_mask = cv2.warpAffine(mask, M_inv, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            
            center = (int(np.mean(np.where(warped_mask > 0.5)[1])), int(np.mean(np.where(warped_mask > 0.5)[0])))
            return blend_with_poisson(warped_face, target_img, warped_mask, center)
        except Exception:
            return target_img

    def Release(self):
        self.model = None


def detect_mouth_open(target_face: Face, landmarks_106=None, target_image=None, face_bbox=None) -> tuple:
    """
    Detecta si la boca está abierta y devuelve la región y el ratio de apertura. (v5.65)
    Utiliza MediaPipe Face Mesh para máxima precisión (95%+).
    """
    is_open = False
    mouth_region = None
    open_ratio = 0.0
    
    # 1. Intentar MediaPipe (Máxima calidad)
    if target_image is not None:
        try:
            from roop.mouth_detector import detect_mouth_open_advanced
            is_open, open_ratio, mouth_data = detect_mouth_open_advanced(target_image)
            if mouth_data:
                mouth_region = mouth_data
                return is_open, mouth_region, open_ratio
        except Exception as e:
            if getattr(roop.globals, 'log_level', 'error') == 'debug':
                print(f"[MOUTH] MediaPipe falló: {e}")
            
    # 2. Fallback a 106 landmarks (Alta calidad)
    if landmarks_106 is not None:
        try:
            pts = np.array(landmarks_106, dtype=np.float32)
            # Puntos de boca en 106 landmarks (aprox 52-71)
            mouth_pts = pts[52:72]
            h_mouth = np.max(mouth_pts[:, 1]) - np.min(mouth_pts[:, 1])
            w_mouth = np.max(mouth_pts[:, 0]) - np.min(mouth_pts[:, 0])
            # v5.1074c: el ancho de boca proyectado se colapsa en PERFIL (la boca se ve
            # de canto) → h/w se infla 3-5× aunque la boca esté cerrada (mide el grosor
            # de labios proyectado, no la abertura real). Log real Corn1982vzla: ratio
            # 0.33-0.78 todo el vídeo → boca target "activa" 269/277 frames (97%) con
            # la boca REAL cerrada (MediaPipe con padding: 0.05, max 0.17 en 73 frames).
            # Normalizar por el ancho de cara (bbox, escala estable): cerrado 0.05-0.18,
            # boca muy abierta cantando ~0.28+. El thresh 0.28/0.30 de ProcessMgr
            # funciona igual que con MediaPipe.
            if face_bbox is not None:
                fw = max(1.0, float(face_bbox[2] - face_bbox[0]))
                open_ratio = h_mouth / fw
            else:
                open_ratio = h_mouth / (w_mouth + 1e-6)
            is_open = open_ratio > 0.15  # v5.1040: 0.20→0.15 — detecta habla (boca ligeramente abierta)
            mouth_region = mouth_pts
        except Exception:
            pass
        
    return is_open, mouth_region, open_ratio


MOUTH_OUTER_LANDMARK_INDICES = (
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
)


def mouth_region_to_frame_points(mouth_data: Any, frame_shape,
                                 crop_origin=(0, 0), crop_size=None) -> Optional[np.ndarray]:
    """Return lip contour in full-frame pixels for MediaPipe or 106 landmarks."""
    if isinstance(mouth_data, np.ndarray):
        points = np.asarray(mouth_data, dtype=np.float32).reshape(-1, 2)
        return points if len(points) >= 3 and np.isfinite(points).all() else None
    if not isinstance(mouth_data, dict) or 'landmarks_468' not in mouth_data:
        return None
    height, width = frame_shape[:2]
    if crop_size is not None:
        width, height = crop_size
    ox, oy = crop_origin
    try:
        landmarks = mouth_data['landmarks_468'].landmark
        points = np.asarray([
            [landmarks[index].x * width + ox, landmarks[index].y * height + oy]
            for index in MOUTH_OUTER_LANDMARK_INDICES
        ], dtype=np.float32)
        return points if np.isfinite(points).all() else None
    except (AttributeError, IndexError, TypeError, ValueError):
        return None


def create_mouth_preservation_mask(frame: np.ndarray, mouth_data: Any, blend_ratio: float = 1.0) -> np.ndarray:
    """
    Crea una máscara quirúrgica para la boca basada en landmarks de MediaPipe o 106 pts. (v5.65)
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    try:
        if isinstance(mouth_data, dict) and 'landmarks_468' in mouth_data:
            # Usar MediaPipe 468 landmarks (precisión sub-pixel)
            pts = []
            landmarks = mouth_data['landmarks_468'].landmark
            for idx in MOUTH_OUTER_LANDMARK_INDICES:
                lm = landmarks[idx]
                pts.append([int(lm.x * w), int(lm.y * h)])
            cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)
            # v5.1100: margen + feather — el polígono exacto de labios al 100% hasta el
            # borde = línea de costura visible entre boca source y target. Dilatación 3px
            # + blur cubre el micro-offset de la alineación 128→256.
            mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
        elif isinstance(mouth_data, np.ndarray):
            # Fallback a 106 landmarks (cascara externa de labios)
            hull = cv2.convexHull(mouth_data.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1.0)
            
    except Exception as e:
        if getattr(roop.globals, 'log_level', 'error') == 'debug':
            print(f"[MOUTH_MASK] Error: {e}")
            
    return mask * blend_ratio
