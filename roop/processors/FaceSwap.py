import os
import numpy as np
import insightface
import cv2
import inspect
import roop.globals
from roop.path_helper import get_root_path
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
            if src_roi.size == 0: continue
            
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
            self.model = INSwapper(model_file=model_path, session=None)
            self.model.session = ort.InferenceSession(model_path, providers=providers)
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
                        print(f"[FaceSwap] Usando pipeline 256px (IMAGE)")
                    try:
                        from insightface.utils import face_align
                        src_img = getattr(source_face, 'face_img', getattr(source_face, 'face_img_ref', None))
                        if src_img is not None:
                            # v5.40: Revertir a Pipeline de 256px BGR 0-255 (Estándar Inswapper)
                            src_kps = np.array(source_face.kps, dtype=np.float32)
                            tgt_kps = np.array(target_face.kps, dtype=np.float32)
                            
                            # 1. Alinear (norm_crop2 devuelve BGR 0-255)
                            aimg_src, _ = face_align.norm_crop2(src_img, src_kps, 256)
                            aimg_tgt, M_tgt = face_align.norm_crop2(temp_frame, tgt_kps, 256)
                            
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
                        print(f"[FaceSwap] Usando pipeline 128px (EMBEDDING)")
                    try:
                        from insightface.utils import face_align
                        src_emb = getattr(source_face, 'embedding', None)
                        if src_emb is not None:
                            src_emb = np.array(src_emb, dtype=np.float32).flatten()
                            src_emb = src_emb / np.linalg.norm(src_emb)
                            tgt_kps = np.array(target_face.kps, dtype=np.float32)
                            
                            aimg_tgt, M_tgt = face_align.norm_crop2(temp_frame, tgt_kps, 128)
                            
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
                    print(f"[FaceSwap] ⚠️ Fallback a WARP GEOMÉTRICO (modelo falló o no disponible)")
                res_data = self._run_warp(temp_frame, target_face, source_face, paste_back=False)
            
            if res_data is None: return None
            
            if isinstance(res_data, tuple):
                swapped_face, M = res_data
                if swapped_face.shape[1] != 256:
                    scale = 256.0 / swapped_face.shape[1]
                    swapped_face = cv2.resize(swapped_face, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                    M = M * scale
                res_data = (swapped_face, M)

            if not paste_back: return res_data
            if isinstance(res_data, tuple):
                return self.paste_back_robust(temp_frame, res_data[0], res_data[1])
            return res_data
                
        except Exception as e:
            print(f"[FaceSwap] Error en Run: {e}")
            return None

    def _run_warp(self, temp_frame, target_face, source_face, paste_back=True):
        try:
            src_img = getattr(source_face, 'face_img_ref', getattr(source_face, 'face_img', None))
            if src_img is None: return None
            src_img_bgr = cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR) if src_img.shape[2] >= 3 else src_img
            
            warped_face, M = FaceWarpEngine.warp_face(
                src_img_bgr, temp_frame, np.array(source_face.kps), np.array(target_face.kps)
            )
            if not paste_back: return warped_face, M
            return self.paste_back_robust(temp_frame, warped_face, M)
        except Exception: return None

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
        except Exception: return target_img

    def Release(self):
        self.model = None
