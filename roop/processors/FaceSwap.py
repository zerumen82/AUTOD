import os
import numpy as np
import insightface
import cv2
import inspect
import roop.globals
from typing import Any, Tuple, Optional, Dict
from roop.types import Face, Frame


class FaceSwap:
    def __init__(self) -> None:
        self.model = None
        self.model_path = None
        self.devicename = None
        self.is_256 = False

    def Initialize(self, options: dict) -> None:
        devnm = options.get('devicename', 'cpu')
        model_path = options.get('model', None)

        if model_path is None:
            # Prioridad 128 (usa embedding ArcFace, transfiere identidad correctamente)
            model_128 = os.path.abspath(os.path.join(os.getcwd(), 'models', 'inswapper_128.onnx'))
            model_256 = os.path.abspath(os.path.join(os.getcwd(), 'models', 'inswapper_256.onnx'))
            
            if os.path.exists(model_128):
                model_path = model_128
            elif os.path.exists(model_256):
                model_path = model_256
            else:
                # Buscar cualquier onnx en models que parezca un swapper
                models_dir = os.path.abspath(os.path.join(os.getcwd(), 'models'))
                if os.path.exists(models_dir):
                    for f in os.listdir(models_dir):
                        if f.endswith('.onnx') and 'inswapper' in f.lower():
                            model_path = os.path.join(models_dir, f)
                            break

        if model_path is None:
            print("[FaceSwap] ❌ ERROR: No se encontró modelo inswapper en /models")
            return

        if not os.path.isabs(model_path):
            model_path = os.path.abspath(os.path.join(os.getcwd(), 'models', model_path))

        # Always (re)load the model to ensure correct type (INSwapper)
        # This replaces any previously loaded model (which might be ArcFaceONNX from older code)
        self.model_path = model_path
        self.devicename = devnm
        self.is_256 = '256' in os.path.basename(model_path)
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'cuda' in devnm else ['CPUExecutionProvider']
        
        try:
            print(f"[FaceSwap] Cargando modelo (FORZANDO INSwapper): {os.path.basename(model_path)} (HighRes={self.is_256})")
            
            # FORZAR carga como INSwapper para evitar TypeError con ArcFaceONNX
            from insightface.model_zoo.inswapper import INSwapper
            print(f"[DEBUG] INSwapper class: {INSwapper}, from {INSwapper.__module__}")
            import onnxruntime as ort
            
            self.model = INSwapper(model_file=model_path, session=None)
            # Configurar sesión con los proveedores adecuados
            self.model.session = ort.InferenceSession(model_path, providers=providers)
            
            print(f"[FaceSwap] Modelo inicializado correctamente")
            print(f"[DEBUG] Tipo de modelo después de cargar: {type(self.model).__name__}")
            
            # Verificar que el modelo sea INSwapper; si no, cancelar
            if type(self.model).__name__ != 'INSwapper':
                print(f"[FaceSwap] ❌ ERROR: El modelo cargado no es INSwapper, es {type(self.model).__name__}")
                self.model = None
        except Exception as e:
            print(f"[FaceSwap] Error crítico inicializando modelo: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def Run(self, source_face: Face, target_face: Face, temp_frame: Frame, paste_back: bool = True) -> Any:
        # Verificar que el modelo es correcto antes de usarlo
        if self.model is None or type(self.model).__name__ != 'INSwapper':
            if self.model is not None:
                print(f"[FaceSwap] Modelo incorrecto detectado en Run: {type(self.model).__name__}, recargando...")
            # Reintentar cargar el modelo
            self.Initialize({"devicename": getattr(self, 'devicename', 'cpu')})
            if self.model is None or type(self.model).__name__ != 'INSwapper':
                print("[FaceSwap] ❌ NO SE PUDO CARGAR EL MODELO INSWAPPER")
                return None
        
        try:
            # Asegurar que el frame sea numpy array
            if not isinstance(temp_frame, np.ndarray):
                temp_frame = np.array(temp_frame)
            
            # Preparar caras: convertir listas a numpy arrays si es necesario
            for face in (target_face, source_face):
                for attr in ['kps', 'bbox', 'embedding', 'normed_embedding']:
                    val = getattr(face, attr, None)
                    if val is not None and isinstance(val, list):
                        setattr(face, attr, np.array(val, dtype=np.float32))
            
            # Preparar embedding de origen específicamente para inswapper (512-dim)
            if hasattr(source_face, 'embedding') and source_face.embedding is not None:
                emb = np.array(source_face.embedding, dtype=np.float32).flatten()
                if emb.shape[0] > 512:
                    emb = emb[:512]
                source_face.embedding = emb
                
                # Normalizar si no lo está
                norm = np.linalg.norm(emb)
                if norm > 0:
                    source_face.normed_embedding = emb / norm

            # Ejecutar swap
            try:
                if self.is_256:
                    res_data = self._run_256(temp_frame, target_face, source_face, paste_back)
                else:
                    sig = inspect.signature(self.model.get)
                    try:
                        res_data = self.model.get(temp_frame, target_face, source_face, paste_back=False)
                    except TypeError:
                        res_data = self.model.get(temp_frame, target_face, source_face)
            except Exception as e:
                print(f"[FaceSwap] Error en Run: {e}")
                import traceback
                traceback.print_exc()
                return None

            if res_data is None:
                return None
            
            if not paste_back:
                if isinstance(res_data, tuple): return res_data[0]
                return res_data

            if isinstance(res_data, tuple):
                swapped_face, M = res_data
                # Usar nuestro pegado de alta calidad
                return self.paste_back_robust(temp_frame, swapped_face, M)
            else:
                # El modelo ya hizo el paste_back internamente
                return res_data
                
        except Exception as e:
            print(f"[FaceSwap] Error en Run: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _run_256(self, temp_frame, target_face, source_face, paste_back=True):
        """Face swap para modelo 256 (image-to-image)"""
        try:
            from insightface.utils import face_align
            import traceback as tb
            
            # 1. Alinear la cara TARGET del frame actual (BGR)
            aimg, M = face_align.norm_crop2(temp_frame, np.array(target_face.kps, dtype=np.float32), 256)
            blob_target = cv2.dnn.blobFromImage(aimg, 1.0 / self.model.input_std, self.model.input_size,
                                                  (self.model.input_mean, self.model.input_mean, self.model.input_mean),
                                                  swapRB=True)
            
            # 2. Obtener y alinear la cara SOURCE
            source_img_bgr = None
            src_path = getattr(source_face, 'source_image', None)
            if src_path and os.path.isfile(src_path):
                src_full = cv2.imread(src_path)
                if src_full is not None and src_full.size > 0:
                    kps = np.array(source_face.kps, dtype=np.float32)
                    aimg_source, _ = face_align.norm_crop2(src_full, kps, 256)
                    source_img_bgr = aimg_source
            
            if source_img_bgr is None:
                source_img = getattr(source_face, 'face_img_ref', None)
                if source_img is None:
                    source_img = getattr(source_face, 'face_img', None)
                if source_img is None:
                    print("[FaceSwap] source_face sin imagen")
                    return None
                if len(source_img.shape) == 3 and source_img.shape[2] == 3:
                    source_img_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR)
                elif len(source_img.shape) == 3 and source_img.shape[2] == 4:
                    source_img_bgr = cv2.cvtColor(source_img, cv2.COLOR_RGBA2BGR)
                else:
                    source_img_bgr = source_img.copy()
                bbox = np.array(source_face.bbox, dtype=np.float32)
                kps_orig = np.array(source_face.kps, dtype=np.float32)
                face_w = bbox[2] - bbox[0]
                face_h = bbox[3] - bbox[1]
                px = face_w * 0.15
                py = face_h * 0.15
                x1_pad = max(0, int(bbox[0] - px))
                y1_pad = max(0, int(bbox[1] - py))
                kps_crop = kps_orig.copy()
                kps_crop[:, 0] -= x1_pad
                kps_crop[:, 1] -= y1_pad
                aimg_source, _ = face_align.norm_crop2(source_img_bgr, kps_crop, 256)
            
            blob_source = cv2.dnn.blobFromImage(aimg_source, 1.0 / self.model.input_std, self.model.input_size,
                                                  (self.model.input_mean, self.model.input_mean, self.model.input_mean),
                                                  swapRB=True)
            
            # 3. Ejecutar el modelo
            input_feed = {self.model.input_names[0]: blob_target, self.model.input_names[1]: blob_source}
            pred = self.model.session.run(self.model.output_names, input_feed)[0]
            
            # 4. Convertir output: CHW RGB(0-1) -> HWC BGR(0-255)
            img_fake = pred.transpose((0,2,3,1))[0]
            bgr_fake = np.clip(255 * img_fake, 0, 255).astype(np.uint8)[:, :, ::-1]
            
            if not paste_back:
                return bgr_fake, M
            
            # 5. Paste back estilo INSwapper: mÃ¡scara basada en diferencia de pÃ­xeles
            #    para evitar el efecto de doble ojo: solo se mezcla donde la cara cambiÃ³ realmente
            fake_diff = bgr_fake.astype(np.float32) - aimg.astype(np.float32)
            fake_diff = np.abs(fake_diff).mean(axis=2)
            # Solo mezclar donde el cambio es significativo (evita bordes y zonas sin cambio)
            fthresh = 10
            fake_diff[:2,:] = 0
            fake_diff[-2:,:] = 0
            fake_diff[:,:2] = 0
            fake_diff[:,-2:] = 0
            IM = cv2.invertAffineTransform(M)
            img_white = np.full((aimg.shape[0],aimg.shape[1]), 255, dtype=np.float32)
            bgr_fake_full = cv2.warpAffine(bgr_fake, IM, (temp_frame.shape[1], temp_frame.shape[0]), borderValue=0.0)
            img_white = cv2.warpAffine(img_white, IM, (temp_frame.shape[1], temp_frame.shape[0]), borderValue=0.0)
            fake_diff = cv2.warpAffine(fake_diff, IM, (temp_frame.shape[1], temp_frame.shape[0]), borderValue=0.0)
            img_white[img_white>20] = 255
            fake_diff[fake_diff<fthresh] = 0
            fake_diff[fake_diff>=fthresh] = 255
            img_mask = img_white
            mask_h_inds, mask_w_inds = np.where(img_mask==255)
            mask_h = np.max(mask_h_inds) - np.min(mask_h_inds)
            mask_w = np.max(mask_w_inds) - np.min(mask_w_inds)
            mask_size = int(np.sqrt(mask_h*mask_w))
            k = max(mask_size//10, 10)
            kernel = np.ones((k,k),np.uint8)
            img_mask = cv2.erode(img_mask,kernel,iterations = 1)
            kernel = np.ones((2,2),np.uint8)
            fake_diff = cv2.dilate(fake_diff,kernel,iterations = 1)
            k = max(mask_size//20, 5)
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            img_mask = cv2.GaussianBlur(img_mask, blur_size, 0)
            k = 5
            kernel_size = (k, k)
            blur_size = tuple(2*i+1 for i in kernel_size)
            fake_diff = cv2.GaussianBlur(fake_diff, blur_size, 0)
            img_mask /= 255
            fake_diff /= 255
            img_mask = np.reshape(img_mask, [img_mask.shape[0],img_mask.shape[1],1])
            fake_merged = img_mask * bgr_fake_full + (1-img_mask) * temp_frame.astype(np.float32)
            fake_merged = fake_merged.astype(np.uint8)
            return fake_merged
            
        except Exception as e:
            print(f"[FaceSwap] _run_256 error: {e}")
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
            
            # Crear máscara suave
            mask = create_soft_mask(target_bbox, (h, w), feather=25, occlusion_mask=occ_mask)
            
            mask_3ch = cv2.merge([mask, mask, mask])
            result = (warped_face * mask_3ch + target_img * (1 - mask_3ch)).astype(np.uint8)
            
            return result
        except Exception as e:
            print(f"[FaceSwap] Error en paste_back_robust: {e}")
            return target_img


# --- FUNCIONES DE PRESERVACIÓN DE BOCA ---

def detect_mouth_open(target_face, landmarks_106, image) -> Tuple[bool, Optional[Dict], float]:
    """Detecta boca abierta usando MouthDetector avanzado"""
    try:
        from roop.mouth_detector import get_mouth_detector
        detector = get_mouth_detector()
        if detector and detector.is_initialized:
            # Extraer región de la cara para mediapipe
            x1, y1, x2, y2 = target_face.bbox
            # Ampliar un poco el recorte para asegurar que entra toda la boca
            h, w = image.shape[:2]
            pw, ph = int((x2-x1)*0.2), int((y2-y1)*0.2)
            x1, y1 = max(0, int(x1-pw)), max(0, int(y1-ph))
            x2, y2 = min(w, int(x2+pw)), min(h, int(y2+ph))
            
            face_roi = image[y1:y2, x1:x2]
            if face_roi.size == 0:
                return False, None, 0.0
                
            is_open, ratio, mouth_data = detector.detect_mouth_open(face_roi)
            
            # Mapear coordenadas de mouth_data al frame original
            if mouth_data:
                for key, val in mouth_data.items():
                    if isinstance(val, tuple) and len(val) == 2:
                        mouth_data[key] = (val[0] + x1, val[1] + y1)
                
                # Añadir dimensiones del ROI para referencia
                mouth_data['roi_offset'] = (x1, y1)
                mouth_data['roi_size'] = (face_roi.shape[1], face_roi.shape[0])
                
            return is_open, mouth_data, ratio
    except Exception as e:
        print(f"[MOUTH_DETECT] Error: {e}")
    return False, None, 0.0


def create_mouth_preservation_mask(image, mouth_data, blend_ratio=0.5) -> np.ndarray:
    """Crea máscara suave para la boca basándose en los datos detectados"""
    mask = np.zeros(image.shape[:2], dtype=np.float32)
    if not mouth_data:
        return mask
        
    try:
        # Puntos clave para el polígono de la boca
        pts = []
        # Labio superior
        if 'upper_lip_top' in mouth_data: pts.append(mouth_data['upper_lip_top'])
        if 'mouth_right' in mouth_data: pts.append(mouth_data['mouth_right'])
        # Labio inferior
        if 'lower_lip_bottom' in mouth_data: pts.append(mouth_data['lower_lip_bottom'])
        if 'mouth_left' in mouth_data: pts.append(mouth_data['mouth_left'])
        
        if len(pts) >= 3:
            pts_arr = np.array(pts, dtype=np.int32)
            cv2.fillPoly(mask, [pts_arr], 1.0)
            
            # Aplicar desenfoque fuerte para suavizar la transición
            blur_size = 15
            mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    except Exception as e:
        print(f"[MOUTH_MASK] Error: {e}")
        
    return mask * blend_ratio
