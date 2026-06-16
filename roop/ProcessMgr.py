print("\n" + "="*60)
print(">>> [TRUE-INTEGRATION] IDENTITY-ABSOLUTE v5.70 — DNA 1.0 + enhancer 0.70 + mouth 0.85 <<<")
print(">>> (256px, Occlusion-Locked, Depth-Color, Hyper-Sharp, XSeg-PRO) <<<")
print("="*60 + "\n")

import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

import roop.globals
from roop.utils import run_ffmpeg
from roop.quality_enhancements import (
    match_color_histogram,
    create_soft_mask,
    blend_with_poisson,
    adjust_face_brightness,
    detect_foreground_occlusion
)


def is_valid_progress_callback(callback):
    """Verifica si un callback de progreso es válido sin causar errores"""
    if callback is None:
        return False
    try:
        return hasattr(callback, '__call__')
    except Exception as e:
        print(f"[WARNING] Error verificando callback: {e}")
        return False


def get_face_center(face):
    """Get center coordinates of a face"""
    if not hasattr(face, 'bbox'):
        return None
    return ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2)


def get_gender(face):
    """Obtiene el género de una cara"""
    try:
        gender_confidence = 0.0
        detected_gender = None
        validation_score = 0.0
        
        strictness_mode = getattr(roop.globals, 'gender_strictness_mode', 'balanced')
        
        if strictness_mode == 'permissive':
            threshold_low, threshold_high = getattr(roop.globals, 'gender_threshold_permissive', (0.30, 0.70))
        elif strictness_mode == 'strict':
            threshold_low, threshold_high = getattr(roop.globals, 'gender_threshold_strict', (0.45, 0.55))
        else:
            threshold_low, threshold_high = getattr(roop.globals, 'gender_threshold_balanced', (0.40, 0.60))
        
        min_confidence_threshold = getattr(roop.globals, 'gender_confidence_threshold', 0.3)
  
        if hasattr(face, 'gender') and face.gender is not None:
            gender_value = face.gender

            if isinstance(gender_value, (list, np.ndarray)):
                gender_value = gender_value[0] if len(gender_value) > 0 else None
              
            if gender_value is not None:
                gender_value = float(gender_value)
                  
                if gender_value == 0 or gender_value == 0.0:
                    detected_gender = 'female'
                    gender_confidence = 0.98
                    validation_score += 2.0
                elif gender_value == 1 or gender_value == 1.0:
                    detected_gender = 'male'
                    gender_confidence = 0.98
                    validation_score += 2.0
                elif 0.0 < gender_value < 1.0:
                    if gender_value < threshold_low:
                        detected_gender = 'female'
                        gender_confidence = 0.70 + (threshold_low - gender_value) * 0.5
                        gender_confidence = min(0.95, gender_confidence)
                        validation_score += 1.5
                    elif gender_value > threshold_high:
                        detected_gender = 'male'
                        gender_confidence = 0.70 + (gender_value - threshold_high) * 0.5
                        gender_confidence = min(0.95, gender_confidence)
                        validation_score += 1.5
                    else:
                        if strictness_mode == 'permissive':
                            if gender_value < 0.5:
                                detected_gender = 'female'
                                gender_confidence = 0.55
                            else:
                                detected_gender = 'male'
                                gender_confidence = 0.55
                            validation_score += 0.8
                        else:
                            detected_gender = None
                            gender_confidence = 0.30
                            validation_score += 0.3
                else:
                    if gender_value < 0:
                        detected_gender = 'female'
                        gender_confidence = 0.60
                    else:
                        detected_gender = 'male'
                        gender_confidence = 0.60
  
        if hasattr(face, 'sex') and face.sex is not None:
            sex_value = str(face.sex).upper()
            if sex_value in ['F', 'FEMALE', 'WOMAN', 'FEM', 'W']:
                if detected_gender == 'female':
                    validation_score += 1.0
                    gender_confidence = min(0.98, gender_confidence + 0.15)
                elif detected_gender is None or detected_gender == 'male':
                    detected_gender = 'female'
                    gender_confidence = max(gender_confidence, 0.80)
                    validation_score += 0.8
            elif sex_value in ['M', 'MALE', 'MAN', 'MASC']:
                if detected_gender == 'male':
                    validation_score += 1.0
                    gender_confidence = min(0.98, gender_confidence + 0.15)
                elif detected_gender is None or detected_gender == 'female':
                    detected_gender = 'male'
                    gender_confidence = max(gender_confidence, 0.80)
                    validation_score += 0.8
  
        if hasattr(face, 'age') and face.age is not None:
            try:
                age_value = float(face.age)
                if detected_gender and 10 <= age_value <= 90:
                    validation_score += 0.5
                    gender_confidence = min(0.98, gender_confidence + 0.05)
            except:
                pass
  
        if hasattr(face, '__setattr__'):
            try:
                face.gender_confidence = gender_confidence
            except:
                pass
  
        if detected_gender and gender_confidence >= min_confidence_threshold:
            return detected_gender
        else:
            if strictness_mode == 'permissive' and detected_gender and gender_confidence > 0.20:
                return detected_gender
            return None
  
    except Exception as e:
        print(f"[GENDER_ERROR] Exception in get_gender: {e}")
        import traceback
        traceback.print_exc()
        return None

class ProcessMgr:
    """Gestor de procesamiento de face swapping v5.70"""
    _swap_call_count = 0  # Contador global para debug
    
    def __init__(self, progress_callback=None):
        self.progress_callback = progress_callback
        self.input_facesets = []
        self.target_faces = []
        self.options = None
        self.is_initialized = False
        self.processors = {}
        self.thread_lock = threading.Lock()

        self.source_embeddings_cache = {}
        self.previous_result = None
        self.frame_count = 0

        self.face_assignment_cache = {}
        self.face_position_history = {}
        self.global_source_for_all_id = None
        self.selected_face_assignment_cache = {}

        self.selected_assignment_ttl = getattr(roop.globals, 'selected_assignment_ttl', 60)
        self.selected_top_k = getattr(roop.globals, 'selected_top_k', 3)
        self._auto_source_face_cache = {}
        
        # NUEVO: Suavizado temporal para boca - evita flickering en preservación de boca
        self._mouth_open_history = {}  # Historial de apertura de boca por video
        self._mouth_blend_smooth = {}   # Blend ratio suavizado por video
        self._occ_mask_ema = {}         # Historial de máscara de oclusión (v5.65)
        self._mouth_object_detected = {} # Flag de objeto en boca (v5.65)

        if is_valid_progress_callback(self.progress_callback):
            print(f"ProcessMgr v5.60 (FORCED QUALITY) inicializado con progress_callback (Top-K={self.selected_top_k}, TTL={self.selected_assignment_ttl})")
        else:
            print(f"ProcessMgr v5.60 (FORCED QUALITY) inicializado SIN progress_callback (Top-K={self.selected_top_k}, TTL={self.selected_assignment_ttl})")

        self._initialize_processors()

    def initialize(self, input_facesets, target_faces, options, face_swap_mode=None):
        self.input_facesets = input_facesets or []
        self.target_faces = target_faces or []
        self.options = options
        self.face_swap_mode = face_swap_mode  # Guardar modo para optimizaciones
        self.is_initialized = True

        self._cache_source_embeddings()
        
        # LIMPIAR historial de boca para nuevo video
        self._mouth_open_history.clear()
        self._mouth_blend_smooth.clear()

        print(f"ProcessMgr v5.60 (FORCED QUALITY) inicializado: {len(self.input_facesets)} facesets, {len(self.target_faces)} target faces")

        if len(self.input_facesets) == 0:
            print("[WARNING] No hay facesets de origen cargados")
        if len(self.target_faces) == 0:
            print("[WARNING] No hay caras destino seleccionadas")

        self.global_source_for_all_id = None
        self._auto_source_face_cache.clear()

    def _cache_source_embeddings(self):
        self.source_embeddings_cache.clear()
        all_embs_with_quality = []
        
        for faceset in self.input_facesets:
            if hasattr(faceset, "faces") and faceset.faces:
                for face in faceset.faces:
                    if hasattr(face, 'embedding') and face.embedding is not None:
                        quality = self._score_source_face_quality(face)
                        emb = np.array(face.embedding, dtype=np.float32)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            unit_emb = emb / norm
                            self.source_embeddings_cache[id(face)] = unit_emb
                            all_embs_with_quality.append((unit_emb, quality))
        
        # v5.71: Master Embedding = weighted blend top-3 (quality^3 weighting)
        # Mejor cara ~80%, 2da ~15%, 3ra ~5% — enriquece identidad sin diluir
        self.master_source_embedding = None
        if len(all_embs_with_quality) > 0:
            all_embs_with_quality.sort(key=lambda x: x[1], reverse=True)
            top_k = min(3, len(all_embs_with_quality))
            weights = np.array([max(0.001, float(x[1])) ** 3 for x in all_embs_with_quality[:top_k]], dtype=np.float64)
            weights /= weights.sum()
            weighted_emb = np.zeros_like(all_embs_with_quality[0][0], dtype=np.float64)
            for i in range(top_k):
                weighted_emb += all_embs_with_quality[i][0].astype(np.float64) * weights[i]
            self.master_source_embedding = weighted_emb.astype(np.float32)
            norm = np.linalg.norm(self.master_source_embedding)
            if norm > 0:
                self.master_source_embedding /= norm
            print(f"[IDENTITY] Master Embedding = weighted blend top-{top_k} (weights: {[f'{w:.2f}' for w in weights]}, best_quality={all_embs_with_quality[0][1]:.2f})")
        
        print(f"v5.60: {len(self.source_embeddings_cache)} embeddings cacheados")

    def _initialize_processors(self):
        try:
            from roop.processors.FaceSwap import FaceSwap
            from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
            from roop.processors.Enhance_RestoreFormerPPlus import Enhance_RestoreFormerPPlus
            from roop.processors.Enhance_GFPGAN import Enhance_GFPGAN
            from roop.processors.Mask_XSeg import Mask_XSeg

            try:
                # FORZAR GPU si PyTorch tiene CUDA disponible (más confiable)
                import torch
                use_cuda = torch.cuda.is_available()
                if use_cuda:
                    print(f"[ProcessMgr] GPU detectada: {torch.cuda.get_device_name(0)}")
                else:
                    print("[ProcessMgr] GPU NO disponible en PyTorch, usando CPU")
                    use_cuda = False
            except Exception as e:
                print(f"[ProcessMgr] Error detectando GPU: {e}")
                use_cuda = False

            devname = "cuda" if use_cuda else "cpu"

            print(f"[ProcessMgr] Inicializando FaceSwap con devicename={devname}")
            print(f"[ProcessMgr] torch.cuda.is_available() = {use_cuda}")
            self.processors["faceswap"] = FaceSwap()
            self.processors["faceswap"].Initialize({"devicename": devname})
            print(f"[ProcessMgr] FaceSwap inicializado correctamente, usando device: {self.processors['faceswap'].devicename}")

            # Inicializar XSeg para máscaras de alta calidad
            try:
                self.processors["mask_xseg"] = Mask_XSeg()
                self.processors["mask_xseg"].Initialize({"devicename": devname})
                print(f"XSeg Masker initialized (devicename={devname})")
            except Exception as e:
                print(f"Failed to initialize XSeg Masker: {e}")

            # Obtener el enhancer seleccionado
            selected_enhancer = getattr(roop.globals, 'selected_enhancer', 'None')
            use_enhancer = getattr(roop.globals, 'use_enhancer', True)

            if use_enhancer and selected_enhancer and selected_enhancer != "None":
                try:
                    if selected_enhancer == "CodeFormer":
                        self.processors["enhance_codeformer"] = Enhance_CodeFormer()
                        self.processors["enhance_codeformer"].Initialize({"devicename": devname})
                        print(f"CodeFormer Enhancer initialized (devicename={devname})")
                    elif selected_enhancer == "Restoreformer++":
                        self.processors["enhance_restoreformer"] = Enhance_RestoreFormerPPlus()
                        self.processors["enhance_restoreformer"].Initialize({"devicename": devname})
                        print(f"Restoreformer++ Enhancer initialized (devicename={devname})")
                    elif selected_enhancer == "GFPGAN":
                        self.processors["enhance_gfpgan"] = Enhance_GFPGAN()
                        self.processors["enhance_gfpgan"].Initialize({"devicename": devname})
                        print(f"GFPGAN Enhancer initialized (devicename={devname})")
                except Exception as e:
                    print(f"Failed to initialize {selected_enhancer} Enhancer: {e}")

            print(f"ProcessMgr: Procesadores inicializados (devicename={devname})")
        except Exception as e:
            print(f"Failed to initialize processors: {e}")
            import traceback
            traceback.print_exc()

    def run_batch(self, input_paths, output_paths, num_threads=1):
        if not self.is_initialized:
            print("ProcessMgr not initialized")
            return

        def process_single_image(input_path, output_path):
            try:
                from roop.capturer import get_image_frame
                frame = get_image_frame(input_path)
                if frame is None:
                    print(f"Could not read image: {input_path}")
                    return
                processed_frame = self.process_frame(frame, file_path=input_path)
                if processed_frame is not None:
                    cv2.imwrite(output_path, processed_frame)
                    print(f"Processed: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
            except Exception as e:
                print(f"Processing {input_path}: {e}")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for ip, op in zip(input_paths, output_paths):
                futures.append(executor.submit(process_single_image, ip, op))
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Batch processing error: {e}")

    def process_frame(self, frame, enable_temporal_smoothing=False, file_path=None):
        _log = r'D:\PROJECTS\AUTOAUTO\debug_swap.log'
        try:
            if not self.is_initialized or frame is None:
                with open(_log, 'a') as lf:
                    lf.write(f"[PROC_FRAME] No inicializado o frame None\n")
                return frame
        except:
            with open(_log, 'a') as lf:
                lf.write(f"[PROC_FRAME] Exception en check is_initialized\n")
        
        # NUEVO: Tracking de caras ya procesadas en este frame para evitar doble swap
        self._frame_processed_faces = getattr(self, '_frame_processed_faces', {})
        current_frame_key = id(frame)
        if current_frame_key != getattr(self, '_last_frame_id', None):
            self._frame_processed_faces = {}
            self._last_frame_id = current_frame_key
        
        # NUEVO: Asegurar que selected_face_references exista
        if not hasattr(roop.globals, 'selected_face_references'):
            roop.globals.selected_face_references = {}
        
        try:
            if "faceswap" in self.processors and len(self.input_facesets) > 0:
                from roop.face_util_rotation import get_all_faces_smart
                target_faces_detected = get_all_faces_smart(frame, min_score=None, for_target=True)
                with open(_log, 'a') as lf:
                    lf.write(f"[PROC_FRAME] detectadas {len(target_faces_detected)} caras\n")

                valid_faces = target_faces_detected

                if not valid_faces:
                    with open(_log, 'a') as lf:
                        lf.write(f"[PROC_FRAME] SIN valid_faces - return frame\n")
                    return frame

            # ============================================
            # MODO SMART-QUALITY v4.9.6: AJUSTES BLINDADOS
            # ============================================
            # Respetamos la selección de cara del usuario (UI)
            # pero forzamos la máxima calidad en el procesamiento
            face_swap_mode = getattr(self.options, 'face_swap_mode', 'selected')
            use_enhancer = True # Forzar siempre ON
            enhancer_blend = 0.85 # Parecido máximo
            preserve_mouth = True # Gestos naturales
            color_match_strength = 0.50 
            identity_dna_injection = 0.95
            
            video_path = getattr(self.options, 'current_video_path', None) or file_path
            is_real_video = getattr(self.options, 'current_video_path', None) is not None
            self._current_video_key = f"video_{os.path.basename(video_path)}" if video_path else "frame"
            
            faces_to_process = []
            
            if is_real_video:
                video_basename = os.path.basename(video_path)
                if not hasattr(self, f'_reference_face_selected_{video_basename}'):
                    self._setup_selected_faces_frame_for_video(video_path, valid_faces)
                
                target_face = self._find_target_face_for_selected_mode(video_path, valid_faces)
                if target_face:
                    faces_to_process = [target_face]
            else:
                # MODO FOTO: Respetar si el usuario seleccionó una cara específica en la UI
                filename = os.path.basename(file_path) if file_path else "unknown"
                video_key = f"selected_face_ref_{filename}"
                face_ref_data = roop.globals.selected_face_references.get(video_key) or roop.globals.selected_face_references.get(filename)
                
                if face_ref_data:
                    target_face_ref = face_ref_data.get('face_obj')
                    if target_face_ref:
                        best_match = None
                        max_sim = -1
                        for face in valid_faces:
                            sim = self._calculate_similarity(target_face_ref.embedding, face.embedding)
                            if sim > max_sim:
                                max_sim = sim
                                best_match = face
                        if best_match and max_sim > 0.10:
                            faces_to_process = [best_match]
                
                # Fallback: si no hay selección manual, elegir la principal
                if not faces_to_process:
                    faces_to_process = [max(valid_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]

            if not faces_to_process:
                return frame
                
            # Siempre key frame para máxima calidad
            self._is_key_frame = True
            result_frame = frame.copy()
            
            # Limitar a 1 cara si el modo es 'selected'
            faces_to_process_limited = faces_to_process[:1] if face_swap_mode in ['selected_faces_frame', 'selected_faces', 'selected'] else faces_to_process
            
            for i, target_face in enumerate(faces_to_process_limited):
                try:
                    all_faces = []
                    for faceset in self.input_facesets:
                        if hasattr(faceset, "faces") and faceset.faces:
                            all_faces.extend(faceset.faces)
                      
                    if not all_faces: continue

                    source_face = None
                    if is_real_video and video_path:
                        lock_key = f'_video_locked_source_face_{os.path.basename(video_path)}'
                        locked_face = getattr(self, lock_key, None)
                        if locked_face is not None:
                            source_face = locked_face

                    if source_face is None:
                        source_face = self._select_source_face(target_face, all_faces, face_swap_mode, video_path)
                        if source_face is not None and is_real_video and video_path:
                            setattr(self, lock_key, source_face)
                            sim_score = self._calculate_similarity(
                                getattr(source_face, 'embedding', None),
                                getattr(target_face, 'embedding', None)
                            ) if hasattr(source_face, 'embedding') else 0
                            print(f"[MATCH] Source seleccionada | similitud={sim_score:.3f}")

                    if source_face is None:
                        continue
                    
                    # PROCESAR SWAP CON CALIDAD FORZADA
                    result_frame = self._process_face_swap_v21(
                        source_face, target_face, result_frame, frame, enable_temporal_smoothing
                    )
                except Exception as e:
                    print(f"Error procesando cara {i+1}: {e}")
                    continue
                  
                # FRAME HOLD: si no se procesó ninguna cara, congelar la última región de swap
                if not faces_to_process and is_real_video and video_path:
                    video_basename = os.path.basename(video_path)
                    prev_full = getattr(self, '_prev_frame_result', {}).get(self._current_video_key)
                    last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                    if prev_full is not None and last_bbox is not None:
                        try:
                            x1, y1, x2, y2 = [int(v) for v in last_bbox]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            if y2 > y1 and x2 > x1:
                                prev_face = prev_full[y1:y2, x1:x2]
                                if prev_face.shape == result_frame[y1:y2, x1:x2].shape:
                                    result_frame[y1:y2, x1:x2] = cv2.addWeighted(
                                        result_frame[y1:y2, x1:x2].astype(np.float32), 0.3,
                                        prev_face.astype(np.float32), 0.7, 0
                                    ).astype(np.uint8)
                        except Exception as e_hold:
                            print(f"[FRAME_HOLD] Error: {e_hold}")
                
                return result_frame
            else:
                return frame
        except Exception as e:
            print(f"Frame processing error: {e}")
            return frame

    def _score_source_face_quality(self, face):
        """Score a source face for inswapper_256. Requires a valid embedding."""
        try:
            if not hasattr(face, 'embedding') or face.embedding is None:
                return float('-inf')

            emb = np.array(face.embedding, dtype=np.float32).flatten()
            if emb.size == 0 or np.linalg.norm(emb) <= 0:
                return float('-inf')

            score = 0.0

            det_score = getattr(face, 'det_score', None)
            if det_score is not None:
                score += max(0.0, min(1.0, float(det_score))) * 2.0

            face_score = getattr(face, 'score', None)
            if face_score is not None:
                score += max(0.0, min(1.0, float(face_score))) * 0.5

            if hasattr(face, 'bbox') and face.bbox is not None:
                x1, y1, x2, y2 = [float(v) for v in face.bbox]
                face_w = max(0.0, x2 - x1)
                face_h = max(0.0, y2 - y1)
                face_area = face_w * face_h

                if face_w < 30 or face_h < 30:
                    score -= 2.0
                else:
                    score += min(2.0, np.log1p(face_area) / 7.0)

                aspect = face_w / face_h if face_h > 0 else 0.0
                if 0.65 <= aspect <= 1.45:
                    score += 0.4
                else:
                    score -= 0.5

                ref_img = getattr(face, 'face_img_ref', None)
                if ref_img is None:
                    ref_img = getattr(face, 'face_img', None)
                if isinstance(ref_img, np.ndarray) and ref_img.size > 0:
                    img_h, img_w = ref_img.shape[:2]
                    img_area = max(1, img_w * img_h)
                    area_ratio = face_area / img_area
                    is_full_image_bbox = (
                        x1 <= 2 and y1 <= 2 and
                        x2 >= img_w - 2 and y2 >= img_h - 2 and
                        area_ratio > 0.85
                    )
                    if is_full_image_bbox:
                        score -= 3.0
                    elif 0.03 <= area_ratio <= 0.75:
                        score += 0.6

            kps = getattr(face, 'kps', None)
            if kps is not None and len(kps) >= 5:
                score += 0.5

            normed_embedding = getattr(face, 'normed_embedding', None)
            if normed_embedding is not None:
                score += 0.3

            return score
        except Exception:
            return float('-inf')

    def _select_best_source_face(self, candidate_faces, mode_label):
        cache_key = (mode_label, tuple(id(face) for face in candidate_faces))
        if cache_key in self._auto_source_face_cache:
            return self._auto_source_face_cache[cache_key]

        best_idx = None
        best_face = None
        best_score = float('-inf')

        for idx, face in enumerate(candidate_faces):
            score = self._score_source_face_quality(face)
            if score > best_score:
                best_idx = idx
                best_face = face
                best_score = score

        if best_face is None or best_score == float('-inf'):
            print(f"[SELECT_SOURCE] Modo {mode_label}: ninguna cara origen valida con embedding")
            return None

        print(f"[SELECT_SOURCE] Modo {mode_label}: mejor cara origen #{best_idx + 1} (quality={best_score:.2f})")
        self._auto_source_face_cache[cache_key] = best_face
        return best_face

    def _cosine_similarity(self, emb1, emb2):
        """Calcula similitud coseno entre dos embeddings"""
        try:
            if emb1 is None or emb2 is None:
                return -1.0
            emb1 = np.array(emb1).flatten()
            emb2 = np.array(emb2).flatten()
            dot = np.dot(emb1, emb2)
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            if norm1 == 0 or norm2 == 0:
                return -1.0
            return dot / (norm1 * norm2)
        except Exception:
            return -1.0

    def _select_source_face_by_similarity(self, target_face, candidate_faces):
        """
        Selecciona la cara origen que mejor coincide con el target usando embedding similarity.
        Para modo 'selected_faces_frame': hace matching real entre target y source.
        """
        try:
            if not candidate_faces:
                return None
            
            target_emb = getattr(target_face, 'embedding', None)
            if target_emb is None:
                print("[SELECT_SOURCE] Target sin embedding, usando quality fallback")
                return self._select_best_source_face(candidate_faces, 'similarity_fallback')
            
            best_face = None
            best_sim = -1.0
            best_idx = -1
            best_quality = -1.0
            
            # 1. Recopilar todos los candidatos con su similitud y calidad
            candidates = []
            for idx, src_face in enumerate(candidate_faces):
                src_emb = getattr(src_face, 'embedding', None)
                if src_emb is None:
                    continue
                sim = self._cosine_similarity(target_emb, src_emb)
                quality = self._score_source_face_quality(src_face)
                candidates.append({
                    'face': src_face,
                    'sim': sim,
                    'quality': quality,
                    'idx': idx
                })
            
            if not candidates:
                print(f"[SELECT_SOURCE] No source con embedding, usando quality")
                return self._select_best_source_face(candidate_faces, 'no_embedding')

            # 2. Encontrar la mejor similitud
            max_sim = max(c['sim'] for c in candidates)
            
            # v5.2.1: Umbral de seguridad ajustado (0.20 -> 0.15) para permitir casos difíciles
            if max_sim < 0.15:
                print(f"[SELECT_SOURCE] Similitud crítica ({max_sim:.4f}). Usando Master Embedding como fallback en lugar de omitir.")
                # En lugar de return None, se deja caer al compromise path para evitar frames sin swap
                # Forzar max_sim a 0.15 para que el compromise logic funcione
                max_sim = 0.15

            # v5.0: NUEVA LÓGICA DE INTENSIDAD
            # Si el usuario tiene muchas muestras, NO queremos la más parecida al target (eso debilita el swap).
            # Queremos la de MEJOR CALIDAD que sea del mismo sujeto.
            
            # Si hay una clara identidad (similitud decente), priorizar calidad al 100%
            if max_sim > 0.22:
                # Filtrar todos los que sean "probablemente la misma persona" (> 0.15 o 80% del max)
                threshold = max(0.15, max_sim * 0.80)
                identity_candidates = [c for c in candidates if c['sim'] >= threshold]
                best_candidate = max(identity_candidates, key=lambda c: c['quality'])
                print(f"[SELECT_SOURCE] Identidad detectada (max_sim={max_sim:.4f}). Seleccionada muestra de mejor calidad: #{best_candidate['idx'] + 1} (qual={best_candidate['quality']:.2f}, sim={best_candidate['sim']:.4f})")
                return best_candidate['face']

            # Si la similitud es baja, usar el mejor compromiso calidad/identidad
            best_candidate = max(candidates, key=lambda c: c['quality'] + (c['sim'] * 10.0))
            print(f"[SELECT_SOURCE] Similitud baja ({max_sim:.4f}), usando compromiso calidad/identidad (weight=10.0): #{best_candidate['idx'] + 1}")
            return best_candidate['face']
            
        except Exception as e:
            print(f"[ERROR] _select_source_face_by_similarity: {e}")
            import traceback; traceback.print_exc()
            return None

    def _select_source_face(self, target_face, candidate_faces, face_swap_mode, video_path=None):
        """
        Lógica de selección de cara origen:
        - Prioriza el matching por similitud para encontrar la mejor cara origen para el destino.
        - Usa el índice seleccionado como fallback si el matching no es posible.
        """
        try:
            if not candidate_faces:
                return None
            
            # 1. MATCHING POR SIMILITUD (Prioridad para encontrar "mejor cara")
            if face_swap_mode in ['selected', 'selected_faces', 'auto', 'selected_faces_frame']:
                result = self._select_source_face_by_similarity(target_face, candidate_faces)
                if result is not None:
                    return result

            # 2. FALLBACK: Índice seleccionado explícitamente
            if hasattr(self.options, 'selected_index') and self.options.selected_index is not None:
                idx = self.options.selected_index
                if 0 <= idx < len(candidate_faces):
                    # print(f"[SELECT_SOURCE] Usando cara seleccionada explícitamente: #{idx+1}")
                    return candidate_faces[idx]
                elif len(candidate_faces) > 0:
                    # Fallback al primer faceset si el índice es inválido
                    return candidate_faces[0]

            # 3. FALLBACK FINAL: Mejor cara por calidad
            return self._select_best_source_face(candidate_faces, face_swap_mode)
            
        except Exception as e:
            print(f"[ERROR] _select_source_face: {e}")
            return candidate_faces[0] if candidate_faces else None

    def _setup_selected_faces_frame_for_video(self, video_path, valid_faces):
        """
        Setup inicial de cara de referencia para modo Selected Faces Frame.
        IMPORTANTE: Una vez establecida la cara de referencia, NO debe cambiar durante todo el video.
        """
        try:
            # NUEVO: Asegurar que selected_face_references exista
            if not hasattr(roop.globals, 'selected_face_references'):
                roop.globals.selected_face_references = {}

            if not valid_faces:
                return

            video_basename = os.path.basename(video_path)
            video_key = f"selected_face_ref_{video_basename}"

            selected_face = None
            selection_method = "auto"

            face_ref_data = None
            user_has_selected_face = False  # Flag para saber si el usuario seleccionó una cara

            if hasattr(roop.globals, 'selected_face_references'):
                if video_key in roop.globals.selected_face_references:
                    face_ref_data = roop.globals.selected_face_references[video_key]
                    user_has_selected_face = True
                else:
                    clean_video_name = video_basename

                    for k, v in roop.globals.selected_face_references.items():
                        if not k.startswith("selected_face_ref_"):
                            continue

                        ref_filename = k.replace("selected_face_ref_", "")

                        if ref_filename.lower() == clean_video_name.lower():
                            face_ref_data = v
                            user_has_selected_face = True
                            break
                        if ref_filename in clean_video_name or clean_video_name in ref_filename:
                            face_ref_data = v
                            user_has_selected_face = True
                            break

            # Guardar flag para saber si el usuario seleccionó una cara
            setattr(self, f'_user_selected_face_{video_basename}', user_has_selected_face)

            # En modo selected_faces_frame:
            # - Si el usuario seleccionó cara: usar ESA cara como referencia (sin fallback)
            # - Si el usuario NO seleccionó cara: usar fallback a la cara más grande PERO mantenerla fija
            if not user_has_selected_face:
                print(f"[SELECTED_FACES_FRAME] No hay cara seleccionada por el usuario para {video_basename}, usando fallback automatico a la cara mas grande (FIJA para todo el video)")

            if face_ref_data:
                user_selected_face = face_ref_data.get('face_obj')
                user_embedding = face_ref_data.get('embedding')
                user_bbox = face_ref_data.get('bbox')

                if user_selected_face is not None:
                    best_match = None
                    best_score = -1

                    for face in valid_faces:
                        if not hasattr(face, 'embedding') or face.embedding is None:
                            continue

                        if user_embedding is not None:
                            score = self._calculate_similarity(user_embedding, face.embedding)
                        else:
                            score = self._bbox_iou(user_bbox, face.bbox) if user_bbox is not None else 0

                        if score > best_score:
                            best_score = score
                            best_match = face

                    if best_match and best_score > 0.15:
                        selected_face = best_match
                        selection_method = f"usuario (score={best_score:.2f})"
                    else:
                        selected_face = user_selected_face
                        selection_method = f"usuario-guardado (score={best_score:.2f})"

            # FALLBACK INICIAL: Solo si no hay selección del usuario
            if selected_face is None:
                selected_face = max(valid_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                selection_method = f"auto-grande ({len(valid_faces)} caras)"

            # Guardar embedding de referencia para tracking consistente
            if hasattr(selected_face, 'embedding') and selected_face.embedding is not None:
                emb = np.array(selected_face.embedding, dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    setattr(self, f'_fallback_embedding_{video_basename}', emb / norm)

            setattr(self, f'global_source_for_all_id_{video_basename}', id(selected_face))
            setattr(self, f'global_source_bbox_{video_basename}', selected_face.bbox)
            setattr(self, f'_target_face_assigned_{video_basename}', selected_face)
            
            has_embedding = False
            if hasattr(selected_face, 'embedding') and selected_face.embedding is not None:
                try:
                    emb = np.array(selected_face.embedding, dtype=np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                        setattr(self, f'global_source_embedding_{video_basename}', emb)
                        setattr(self, f'_original_embedding_{video_basename}', emb.copy())
                        has_embedding = True
                except Exception as e:
                    print(f"[SETUP] {video_basename}: Error embedding: {e}")

            setattr(self, f'_tracking_lost_count_{video_basename}', 0)
            setattr(self, f'_position_history_{video_basename}', [get_face_center(selected_face)])
            setattr(self, f'_frame_count_{video_basename}', 0)

            setattr(self, f'_reference_face_selected_{video_basename}', True)
            print(f"[SETUP] {video_basename}: {selection_method}, embedding={'YES' if has_embedding else 'NO'}")

        except Exception as e:
            print(f"[SETUP ERROR] {os.path.basename(video_path)}: {e}")
            import traceback
            traceback.print_exc()

    def _init_tracking_scene(self, video_path, valid_faces, frame_count):
        """Initialize scene understanding for intelligent tracking"""
        video_basename = os.path.basename(video_path)
        scene_attr = f'_scene_state_{video_basename}'
        
        if hasattr(self, scene_attr):
            return getattr(self, scene_attr)
        
        # Initialize scene state
        scene_state = {
            'initialized': False,
            'camera_moving': False,
            'avg_face_size': 0,
            'face_count_history': [],
            'lighting_baseline': None,
            'last_scene_cut': 0,
            'total_scene_changes': 0,
            'typical_motion_speed': 0,
            'is_close_up': False
        }
        
        if valid_faces:
            # Calculate average face size
            face_sizes = []
            for face in valid_faces:
                if hasattr(face, 'bbox') and face.bbox is not None:
                    w = face.bbox[2] - face.bbox[0]
                    h = face.bbox[3] - face.bbox[1]
                    face_sizes.append(w * h)
            
            if face_sizes:
                scene_state['avg_face_size'] = sum(face_sizes) / len(face_sizes)
                scene_state['is_close_up'] = scene_state['avg_face_size'] > 10000
        
        scene_state['initialized'] = True
        setattr(self, scene_attr, scene_state)
        return scene_state

    def _update_scene_state(self, video_path, valid_faces, frame_count, position_history):
        """Update scene understanding based on new frame"""
        video_basename = os.path.basename(video_path)
        scene_attr = f'_scene_state_{video_basename}'
        
        if not hasattr(self, scene_attr):
            return self._init_tracking_scene(video_path, valid_faces, frame_count)
        
        scene = getattr(self, scene_attr)
        
        # Update face count history
        current_face_count = len(valid_faces)
        scene['face_count_history'].append(current_face_count)
        if len(scene['face_count_history']) > 30:
            scene['face_count_history'].pop(0)
        
        # Detect camera movement based on face size changes
        if len(position_history) >= 5:
            recent_sizes = []
            for face in valid_faces[:5]:
                if hasattr(face, 'bbox') and face.bbox is not None:
                    w = face.bbox[2] - face.bbox[0]
                    h = face.bbox[3] - face.bbox[1]
                    recent_sizes.append(w * h)
            
            if recent_sizes and scene['avg_face_size'] > 0:
                size_ratio = sum(recent_sizes) / len(recent_sizes) / scene['avg_face_size']
                # If face size changes >30% rapidly, camera might be zooming
                if abs(size_ratio - 1.0) > 0.3:
                    scene['camera_moving'] = True
                else:
                    scene['camera_moving'] = False
        
        setattr(self, scene_attr, scene)
        return scene

    def _predict_next_position(self, position_history, scene_state=None):
        """Predict next position using motion modeling"""
        if not position_history or len(position_history) < 2:
            return None
        
        # Simple Kalman-like prediction
        if len(position_history) >= 3:
            # Calculate velocity (using last 3 positions)
            vx = (position_history[-1][0] - position_history[-3][0]) / 2
            vy = (position_history[-1][1] - position_history[-3][1]) / 2
            
            # Calculate acceleration (if enough history)
            if len(position_history) >= 4:
                ax = (position_history[-1][0] - 2*position_history[-2][0] + position_history[-3][0]) / 4
                ay = (position_history[-1][1] - 2*position_history[-2][1] + position_history[-3][1]) / 4
                
                # Predict with acceleration
                pred_x = position_history[-1][0] + vx + 0.5 * ax
                pred_y = position_history[-1][1] + vy + 0.5 * ay
            else:
                # Predict with velocity only
                pred_x = position_history[-1][0] + vx
                pred_y = position_history[-1][1] + vy
            
            # Adapt prediction confidence based on scene
            if scene_state and scene_state.get('camera_moving'):
                # Less confident in prediction if camera is moving
                confidence = 0.5
            else:
                confidence = 0.7
            
            return (pred_x, pred_y, confidence)
        
        elif len(position_history) >= 2:
            # Simple linear prediction
            vx = position_history[-1][0] - position_history[-2][0]
            vy = position_history[-1][1] - position_history[-2][1]
            return (position_history[-1][0] + vx, position_history[-1][1] + vy, 0.6)
        
        return None

    def _calculate_motion_consistency(self, face, position_history):
        """Check if face motion is consistent with expectations"""
        if not position_history or len(position_history) < 3:
            return 1.0
        
        center = get_face_center(face)
        if not center:
            return 0.0
        
        prediction = self._predict_next_position(position_history)
        if not prediction:
            return 0.5
        
        pred_x, pred_y, conf = prediction
        actual_distance = np.sqrt((center[0] - pred_x)**2 + (center[1] - pred_y)**2)
        
        # Calculate expected position variance
        if len(position_history) >= 5:
            recent_dists = []
            # Calcular distancias entre posiciones consecutivas (últimas 4)
            for i in range(len(position_history)-1, len(position_history)-5, -1):
                if i > 0:
                    d = np.sqrt((position_history[i][0] - position_history[i-1][0])**2 + 
                                (position_history[i][1] - position_history[i-1][1])**2)
                    recent_dists.append(d)
            
            if recent_dists:
                avg_speed = sum(recent_dists) / len(recent_dists)
                # Motion is consistent if within 2 standard deviations
                if actual_distance < avg_speed * 2:
                    return 1.0
                elif actual_distance < avg_speed * 4:
                    return 0.5
                else:
                    return 0.0
        
        return 0.5

    def _find_target_face_for_selected_mode(self, video_path, valid_faces):
        """Intelligent tracking with scene analysis and motion prediction"""
        try:
            video_basename = os.path.basename(video_path)
            
            # Verify if user selected a face
            user_selected_attr = f'_user_selected_face_{video_basename}'
            user_selected = getattr(self, user_selected_attr, True)
            use_fallback = not user_selected

            if not valid_faces:
                return None

            video_key = f"selected_face_ref_{video_basename}"
            
            # Initialize scene understanding
            frame_count_attr = f'_frame_count_{video_basename}'
            frame_count = getattr(self, frame_count_attr, 0) + 1
            setattr(self, frame_count_attr, frame_count)
            
            # Update scene state with current frame
            scene_state = self._update_scene_state(video_path, valid_faces, frame_count, [])
            
            assigned_attr = f'_target_face_assigned_{video_basename}'
            original_embedding_attr = f'_original_embedding_{video_basename}'
            tracking_lost_attr = f'_tracking_lost_count_{video_basename}'
            position_history_attr = f'_position_history_{video_basename}'
            consecutive_success_attr = f'_consecutive_tracking_success_{video_basename}'
            
            # Initialize attributes if needed
            for attr, default in [(tracking_lost_attr, 0), (position_history_attr, []), (consecutive_success_attr, 0)]:
                if not hasattr(self, attr):
                    setattr(self, attr, default)
            
            lost_count = getattr(self, tracking_lost_attr)
            position_history = getattr(self, position_history_attr)
            
            # Get original embedding
            original_embedding = getattr(self, original_embedding_attr, None)
            if original_embedding is None:
                # 1. Intentar obtener referencia específica del video
                if hasattr(roop.globals, 'selected_face_references') and video_key in roop.globals.selected_face_references:
                    face_ref_data = roop.globals.selected_face_references[video_key]
                    user_embedding = face_ref_data.get('embedding')
                    if user_embedding is not None:
                        emb = np.array(user_embedding, dtype=np.float32)
                        norm = np.linalg.norm(emb)
                        if norm > 0:
                            original_embedding = emb / norm
                            setattr(self, original_embedding_attr, original_embedding)
                
                # 2. SMART PROPAGATION FALLBACK: Si no hay específica, buscar en el lote global de target_faces
                if original_embedding is None and self.target_faces:
                    print(f"[TRACK] No hay referencia para {video_basename}, aplicando Propagación Inteligente...")
                    best_global_match = None
                    max_global_sim = -1
                    
                    # Comparar las caras detectadas en este frame con TODAS las caras seleccionadas en el lote
                    for face in valid_faces:
                        if not hasattr(face, 'embedding') or face.embedding is None: continue
                        
                        for target_face in self.target_faces:
                            if not hasattr(target_face, 'embedding') or target_face.embedding is None: continue
                            
                            sim = self._calculate_similarity(target_face.embedding, face.embedding)
                            if sim > max_global_sim:
                                max_global_sim = sim
                                best_global_match = target_face
                    
                    if best_global_match and max_global_sim > 0.45: # Umbral conservador para propagación automática
                        print(f"[TRACK] Propagación exitosa para {video_basename} (sim={max_global_sim:.2f})")
                        emb = np.array(best_global_match.embedding, dtype=np.float32)
                        norm = np.linalg.norm(emb)
                        original_embedding = emb / norm
                        setattr(self, original_embedding_attr, original_embedding)
            
            # INTELLIGENT TRACKING SECTION
            if hasattr(self, assigned_attr):
                assigned_face = getattr(self, assigned_attr)
                
                if assigned_face and hasattr(assigned_face, 'bbox'):
                    assigned_center = get_face_center(assigned_face)
                    
                    if not assigned_center:
                        lost_count += 1
                        setattr(self, tracking_lost_attr, lost_count)
                        return None
                    
                    # Get scene-aware threshold
                    base_threshold = 200
                    if scene_state.get('is_close_up'):
                        # More lenient for close-ups
                        base_threshold = max(250, 0.6 * (assigned_face.bbox[2] - assigned_face.bbox[0]))
                    elif scene_state.get('camera_moving'):
                        # More aggressive tracking if camera is moving
                        base_threshold = max(400, 0.9 * (assigned_face.bbox[2] - assigned_face.bbox[0]))
                    else:
                        base_threshold = max(250, 0.6 * (assigned_face.bbox[2] - assigned_face.bbox[0]))
                    
                    # Predict next position using motion modeling
                    prediction = self._predict_next_position(position_history, scene_state)
                    
                    # Find best match with intelligent scoring
                    best_match = None
                    best_score = -1
                    best_distance = float('inf')
                    
                    for face in valid_faces:
                        face_center = get_face_center(face)
                        if not face_center:
                            continue
                        
                        # Calculate distance
                        dist = np.sqrt((face_center[0] - assigned_center[0])**2 + 
                                     (face_center[1] - assigned_center[1])**2)
                        
                        # Skip if too far
                        if dist > base_threshold:
                            continue
                        
                        # INTELLIGENT SCORING
                        score = 0.0
                        
                        # 1. Distance component (normalized 0-1)
                        dist_normalized = 1.0 - (dist / base_threshold)
                        score += dist_normalized * 0.4  # 40% weight on distance
                        
                        # 2. Embedding similarity
                        if original_embedding is not None and hasattr(face, 'embedding') and face.embedding is not None:
                            emb_score = self._calculate_similarity(original_embedding, face.embedding)
                            
                            # Dynamic threshold based on scene (Relaxed)
                            if scene_state.get('is_close_up'):
                                emb_thresh = 0.12
                            elif scene_state.get('camera_moving'):
                                emb_thresh = 0.15
                            else:
                                emb_thresh = 0.18

                            if emb_score >= emb_thresh:
                                score += emb_score * 0.40  # 40% weight on embedding
                            else:
                                # Check motion consistency - if motion is good, still consider it
                                motion_consistency = self._calculate_motion_consistency(face, position_history)
                                if emb_score >= 0.10 and motion_consistency > 0.80:
                                    score += emb_score * 0.25 + motion_consistency * 0.15
                                    # print(f"[TRACK] Low emb ({emb_score:.2f}) but good motion")
                                else:
                                    score -= 0.1  # Reduced penalty

                        else:
                            # No embedding - rely more on distance and motion
                            score += 0.25
                        
                        # 3. Motion prediction bonus
                        if prediction:
                            pred_x, pred_y, pred_conf = prediction
                            pred_dist = np.sqrt((face_center[0] - pred_x)**2 + (face_center[1] - pred_y)**2)
                            if pred_dist < base_threshold * 0.5:  # Within 50% of threshold
                                score += pred_conf * 0.15  # Up to 15% bonus for matching prediction
                        
                        # 4. Size consistency
                        if hasattr(face, 'bbox') and hasattr(assigned_face, 'bbox'):
                            curr_w = face.bbox[2] - face.bbox[0]
                            curr_h = face.bbox[3] - face.bbox[1]
                            prev_w = assigned_face.bbox[2] - assigned_face.bbox[0]
                            prev_h = assigned_face.bbox[3] - assigned_face.bbox[1]
                            size_ratio = (curr_w * curr_h) / (prev_w * prev_h) if prev_w * prev_h > 0 else 1.0
                            if 0.7 < size_ratio < 1.4:  # Size hasn't changed dramatically
                                score += 0.1
                        
                        if score > best_score:
                            best_score = score
                            best_match = face
                            best_distance = dist
                    
                    # Decision making
                    if best_match and best_score > 0.20:  # Minimum confidence threshold (v3.6: Aumentado para evitar enganchar objetos falsos)
                        # Update state
                        setattr(self, assigned_attr, best_match)
                        new_center = get_face_center(best_match)
                        
                        if new_center:
                            position_history.append(new_center)
                            if len(position_history) > 15:  # Keep more history
                                position_history.pop(0)
                            setattr(self, position_history_attr, position_history)
                        
                        setattr(self, tracking_lost_attr, 0)
                        consec_success = getattr(self, consecutive_success_attr, 0) + 1
                        setattr(self, consecutive_success_attr, consec_success)
                        
                        # Progressive embedding update (v3.1: blend suave y frecuente)
                        if hasattr(best_match, 'embedding') and best_match.embedding is not None:
                            if consec_success >= 5 and best_score > 0.50:
                                old_emb = np.array(getattr(self, original_embedding_attr), dtype=np.float32)
                                new_emb = np.array(best_match.embedding, dtype=np.float32)
                                
                                # v4.6: Reducir update en perfiles para evitar deriva de identidad
                                is_p = ProcessMgr._is_profile_face(getattr(best_match, 'kps', None))
                                blend_alpha = 0.05 if is_p else 0.15 # 5% en perfil, 15% normal
                                
                                blend = blend_alpha * new_emb + (1.0 - blend_alpha) * old_emb
                                norm = np.linalg.norm(blend)
                                if norm > 0:
                                    setattr(self, original_embedding_attr, blend / norm)
                                    setattr(self, consecutive_success_attr, 0)
                                    if frame_count % 30 == 0:
                                        print(f"[TRACK] Frame {frame_count}: Embedding blend updated (profile={is_p})")
                        
                        # Save tracking embedding as recovery fallback
                        tracking_emb_attr = f'_tracking_embedding_{video_basename}'
                        if hasattr(best_match, 'embedding') and best_match.embedding is not None:
                            emb_arr = np.array(best_match.embedding, dtype=np.float32)
                            norm = np.linalg.norm(emb_arr)
                            if norm > 0:
                                setattr(self, tracking_emb_attr, emb_arr / norm)
                        
                        print(f"[TRACK] Frame {frame_count}: Success (score={best_score:.2f}, dist={best_distance:.1f}px)")
                        return best_match
                    
                    # ==========================================================
                    # INTELLIGENT FALLBACK: Tracking lost, try Recognition
                    # ==========================================================
                    print(f"[TRACK] Frame {frame_count}: Local tracking failed, attempting Global Recognition...")
                    
                    rec_embedding = original_embedding
                    if rec_embedding is None:
                        rec_embedding = getattr(self, f'_tracking_embedding_{video_basename}', None)
                    
                    if rec_embedding is not None:
                        best_recognition = None
                        best_rec_score = -1
                        
                        for face in valid_faces:
                            if not hasattr(face, 'embedding') or face.embedding is None:
                                continue
                            
                            rec_score = self._calculate_similarity(rec_embedding, face.embedding)
                            if rec_score > best_rec_score:
                                best_rec_score = rec_score
                                best_recognition = face
                        
                        # v5.48: Umbral más permisivo para re-adquirir perfiles
                        rec_threshold = 0.25 if lost_count < 10 else 0.35 

                        if best_recognition and best_rec_score >= rec_threshold:
                            print(f"[TRACK] RE-ACQUIRED via Global Recognition (score={best_rec_score:.2f})")

                            # Suavizar transición de bbox para evitar saltos bruscos
                            if assigned_face is not None and hasattr(assigned_face, 'bbox') and hasattr(best_recognition, 'bbox'):
                                import copy
                                smoothed_face = copy.copy(best_recognition)
                                old_bbox = np.array(assigned_face.bbox, dtype=np.float32)
                                new_bbox = np.array(best_recognition.bbox, dtype=np.float32)
                                blend_bbox = 0.5 * old_bbox + 0.5 * new_bbox
                                smoothed_face.bbox = tuple(blend_bbox.astype(int))
                                if hasattr(best_recognition, 'kps') and best_recognition.kps is not None and hasattr(assigned_face, 'kps') and assigned_face.kps is not None:
                                    old_kps = np.array(assigned_face.kps, dtype=np.float32)
                                    new_kps = np.array(best_recognition.kps, dtype=np.float32)
                                    blend_kps = 0.4 * old_kps + 0.6 * new_kps
                                    smoothed_face.kps = blend_kps.tolist()
                                best_recognition = smoothed_face

                            setattr(self, assigned_attr, best_recognition)
                            setattr(self, tracking_lost_attr, 0)
                            new_center = get_face_center(best_recognition)
                            if new_center:
                                setattr(self, position_history_attr, [new_center])
                            setattr(self, f'_reacquire_ramp_{video_basename}', 0)
                            return best_recognition
                    
                    # Tracking truly lost
                    lost_count += 1
                    setattr(self, tracking_lost_attr, lost_count)
                    setattr(self, consecutive_success_attr, 0)
                    
                    # ==========================================================
                    # GHOST TRACKING (v2.3): Predecir posición si se pierde la cara
                    # Evita parpadeos en perfiles o movimientos rápidos
                    # ==========================================================
                    if lost_count < 20 and prediction and assigned_face:
                        import copy
                        ghost_face = copy.copy(assigned_face)
                        pred_x, pred_y, pred_conf = prediction
                        
                        # Calcular desplazamiento desde la última posición conocida
                        old_center = get_face_center(assigned_face)
                        if old_center:
                            dx = pred_x - old_center[0]
                            dy = pred_y - old_center[1]
                            
                            # Dampen extreme displacements (likely wrong prediction)
                            # v5.53: max_shift aumentado 80/40→120/60 para capturar movimientos rápidos
                            max_shift = 120 if lost_count < 5 else 60
                            if abs(dx) > max_shift:
                                dx = max_shift if dx > 0 else -max_shift
                            if abs(dy) > max_shift:
                                dy = max_shift if dy > 0 else -max_shift
                            
                            # v5.48: Inercia aumentada para mejor seguimiento de perfiles
                            dx = dx * 0.85
                            dy = dy * 0.85
                            
                            # Desplazar BBox y Keypoints
                            if hasattr(ghost_face, 'bbox'):
                                b = ghost_face.bbox
                                ghost_face.bbox = [b[0]+dx, b[1]+dy, b[2]+dx, b[3]+dy]
                            
                            if hasattr(ghost_face, 'kps') and ghost_face.kps is not None:
                                k = np.array(ghost_face.kps)
                                k[:, 0] += dx
                                k[:, 1] += dy
                                ghost_face.kps = k.tolist()
                                
                            # Marcar como ghost para blending conservador
                            ghost_face.is_ghost = True
                            setattr(ghost_face, '_ghost_dx', dx)
                            setattr(ghost_face, '_ghost_dy', dy)
                            # Score progresivo: 0.9 → 0.5 según lost_count
                            decay = max(0.5, 1.0 - lost_count * 0.025)
                            ghost_face.det_score = getattr(assigned_face, 'det_score', 0.5) * decay
                            
                            # Actualizar historia con la predicción para mantener la inercia
                            position_history.append((pred_x, pred_y))
                            if len(position_history) > 15: position_history.pop(0)
                            setattr(self, position_history_attr, position_history)
                            
                            if lost_count % 3 == 0:
                                print(f"[TRACK] Ghost Tracking Activo (frame {frame_count}, lost={lost_count}, score_decay={decay:.2f})")
                            return ghost_face

                    print(f"[TRACK] Frame {frame_count}: Lost (score={best_score:.2f}), attempts {lost_count}/15")
                    
                    # Reset y preparar re-intento periódico
                    if lost_count >= 15:
                        print(f"[TRACK] Resetting state after {lost_count} failures")
                        # Guardar última posición conocida para transición suave al re-adquirir
                        if assigned_face is not None and hasattr(assigned_face, 'bbox'):
                            setattr(self, f'_last_known_bbox_{video_basename}', assigned_face.bbox)
                            if hasattr(assigned_face, 'kps'):
                                setattr(self, f'_last_known_kps_{video_basename}', assigned_face.kps)
                        setattr(self, assigned_attr, None)
                        setattr(self, position_history_attr, [])
                    return None
            
            # ==========================================================
            # PERIODIC RE-ACQUISITION after reset
            #   - Progressive thresholds as more frames pass
            #   - Falls back to tracking embedding if original is None
            # ==========================================================
            recovery_attr = f'_recovery_attempts_{video_basename}'
            if not hasattr(self, recovery_attr):
                setattr(self, recovery_attr, 0)
            recovery_attempts = getattr(self, recovery_attr) + 1
            setattr(self, recovery_attr, recovery_attempts)
            
            # Best embedding available: original > tracking > None
            reacquire_emb = original_embedding
            if reacquire_emb is None:
                reacquire_emb = getattr(self, f'_tracking_embedding_{video_basename}', None)
            
            if reacquire_emb is not None:
                best_reacquire = None
                best_reacquire_score = -1
                best_threshold = 0.15
                
                # Progressive threshold: stricter early, more lenient over time
                if recovery_attempts < 15:
                    base_thresh = 0.12
                elif recovery_attempts < 60:
                    base_thresh = 0.08
                elif recovery_attempts < 200:
                    base_thresh = 0.05
                else:
                    base_thresh = 0.03
                
                for face in valid_faces:
                    if not hasattr(face, 'embedding') or face.embedding is None:
                        continue
                    
                    emb_score = self._calculate_similarity(reacquire_emb, face.embedding)
                    
                    face_area = (face.bbox[2]-face.bbox[0]) * (face.bbox[3]-face.bbox[1]) if hasattr(face, 'bbox') else 0
                    face_thresh = base_thresh - 0.03 if face_area > 10000 else base_thresh
                    
                    if emb_score >= face_thresh and emb_score > best_reacquire_score:
                        best_reacquire_score = emb_score
                        best_reacquire = face
                        best_threshold = face_thresh
                
                if best_reacquire and best_reacquire_score >= best_threshold:
                    print(f"[TRACK] RE-ACQUIRED after {recovery_attempts} frames (score={best_reacquire_score:.2f}, thresh={best_threshold:.2f})")
                    # Suavizar transición desde la última posición conocida
                    last_bbox = getattr(self, f'_last_known_bbox_{video_basename}', None)
                    if last_bbox is not None and hasattr(best_reacquire, 'bbox'):
                        import copy
                        smoothed_face = copy.copy(best_reacquire)
                        old_bbox = np.array(last_bbox, dtype=np.float32)
                        new_bbox = np.array(best_reacquire.bbox, dtype=np.float32)
                        
                        # DETECTAR CAMBIO DE PLANO (Evitar mezclar bboxes lejanos)
                        dist = np.sqrt(np.sum(((old_bbox[:2]+old_bbox[2:])/2 - (new_bbox[:2]+new_bbox[2:])/2)**2))
                        if dist < 150:
                            blend_bbox = 0.3 * old_bbox + 0.7 * new_bbox
                            smoothed_face.bbox = tuple(blend_bbox.astype(int))
                            last_kps = getattr(self, f'_last_known_kps_{video_basename}', None)
                            if last_kps is not None and hasattr(best_reacquire, 'kps') and best_reacquire.kps is not None:
                                old_kps = np.array(last_kps, dtype=np.float32)
                                new_kps = np.array(best_reacquire.kps, dtype=np.float32)
                                blend_kps = 0.3 * old_kps + 0.7 * new_kps
                                smoothed_face.kps = blend_kps.tolist()
                            best_reacquire = smoothed_face
                        else:
                            print(f"[SCENE_CUT] Re-adquisición lejana ({dist:.0f}px), ignorando blending.")
                        
                        # Limpiar para no re-usar en futuras re-adquisiciones lejanas
                        delattr(self, f'_last_known_bbox_{video_basename}')
                        if hasattr(self, f'_last_known_kps_{video_basename}'):
                            delattr(self, f'_last_known_kps_{video_basename}')
                    setattr(self, assigned_attr, best_reacquire)
                    setattr(self, tracking_lost_attr, 0)
                    setattr(self, recovery_attr, 0)
                    center = get_face_center(best_reacquire)
                    if center:
                        setattr(self, position_history_attr, [center])
                    return best_reacquire
            
            return None
            
        except Exception as e:
            print(f"[ERROR] _find_target_face_for_selected_mode: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _bbox_iou(self, a, b):
        """Compute IoU between two bboxes (x1,y1,x2,y2). Returns 0 if invalid."""
        try:
            if not a or not b:
                return 0.0
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
            area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
            union = area_a + area_b - inter
            if union <= 0:
                return 0.0
            return inter / union
        except Exception:
            return 0.0

    def _select_intelligent_face(self, target_face, candidate_faces):
        try:
            if not candidate_faces:
                return None
                
            best_face = None
            best_similarity = -1
            
            # NO usar umbral - siempre elegir la más similar (máxima fidelidad)
            # usa_gender_filter = getattr(roop.globals, 'use_gender_filter', True)
            use_gender_filter = True  # SIEMPRE activo para mejor match
            
            target_emb = None
            if hasattr(target_face, "embedding") and target_face.embedding is not None:
                target_emb = np.array(target_face.embedding, dtype=np.float32)
                if np.linalg.norm(target_emb) > 0:
                    target_emb = target_emb / np.linalg.norm(target_emb)
            
            for face in candidate_faces:
                similarity = 0.0
                
                if target_emb is not None and hasattr(face, 'embedding') and face.embedding is not None:
                    similarity = self._calculate_similarity(target_emb, face.embedding)
                    
                    if use_gender_filter:
                        target_gender = get_gender(target_face)
                        face_gender = get_gender(face)
                        if target_gender and face_gender and target_gender == face_gender:
                            similarity += 0.1
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_face = face
            
            return best_face if best_face else candidate_faces[0]
        except Exception as e:
            return candidate_faces[0] if candidate_faces else None

    @staticmethod
    def _is_profile_face(kps):
        """Detect if face is a profile/side view based on 5-point kps landmarks"""
        if kps is None or len(kps) < 5:
            return False
        kps = np.array(kps)
        eye_dist_x = abs(kps[0][0] - kps[1][0])
        face_width = np.max(kps[:, 0]) - np.min(kps[:, 0])
        if face_width < 1:
            return False
        # Profile: eyes are close together in x relative to face width
        eye_ratio = eye_dist_x / face_width
        # Also check if one eye is much higher than the other (head tilt in profile)
        eye_dist_y = abs(kps[0][1] - kps[1][1])
        eye_y_ratio = eye_dist_y / (face_width + 1e-6)
        # v5.1: Relaxed y_ratio (0.30 -> 0.45) to avoid false profile detection on tilted faces
        return eye_ratio < 0.22 or eye_y_ratio > 0.45

    def _process_face_swap_v21(self, source_face, target_face, result_frame, original_frame, enable_temporal_smoothing=False):
        ProcessMgr._swap_call_count += 1
        call_num = ProcessMgr._swap_call_count

        # LOGS DE DIAGNÓSTICO INICIAL (v5.40)
        if call_num % 100 == 1 or call_num <= 5:
            msg = f"[SWAP_START] Frame {call_num}: Procesando intercambio..."
            print(msg, flush=True)
            with open(os.path.join(os.path.dirname(__file__), '..', 'debug_swap.log'), 'a') as lf:
                lf.write(f"{msg}\n")

        # v5.1.4: Inicialización robusta al inicio del scope
        user_blend = getattr(self.options, 'blend_ratio', 1.0)
        f_center = 1.0
        f_size = 1.0
        kps_ema = 1.0
        h_align = 256
        w_align = 256
        
        try:
            if not hasattr(source_face, 'embedding') or source_face.embedding is None:
                print(f"[SWAP_SKIP] Frame {call_num}: cara origen sin embedding; no se puede usar inswapper_256", flush=True)
                return original_frame

            if not hasattr(target_face, 'bbox') or target_face.bbox is None:
                print(f"[SWAP_SKIP] Frame {call_num}: cara destino sin bbox", flush=True)
                return original_frame

            # Detectar perfil para ajustar suavizado y blending
            target_kps = getattr(target_face, 'kps', None)
            is_profile = ProcessMgr._is_profile_face(target_kps)
            if is_profile:
                print(f"[PROFILE] Frame {call_num}: Perfil detectado")

            # ============================================
            # SUAVIZADO TEMPORAL AVANZADO (v3.5)
            # ============================================
            video_key = getattr(self, '_current_video_key', 'default')
            video_basename = video_key[6:] if video_key.startswith('video_') else video_key
            if enable_temporal_smoothing:
                prev_bbox = getattr(self, '_prev_face_bbox', {}).get(video_key)
                prev_kps = getattr(self, '_prev_face_kps', {}).get(video_key)
                prev_frame_result = getattr(self, '_prev_frame_result', {}).get(video_key)

                if prev_bbox is not None:
                    # 1. Suavizado de BBox: Separar Centro de Tamaño para evitar "pumping"
                    curr_bbox = np.array(target_face.bbox)
                    prev_bbox_arr = np.array(prev_bbox)
                    
                    # Calcular centros y tamaños
                    curr_center = (curr_bbox[:2] + curr_bbox[2:]) / 2
                    prev_center = (prev_bbox_arr[:2] + prev_bbox_arr[2:]) / 2
                    
                    velocity = np.max(np.abs(curr_center - prev_center))
                    setattr(self, '_last_velocity', velocity)
                    
                    # DETECTAR CAMBIO DE PLANO ANTES DE SUAVIZAR (v3.6)
                    last_reset = getattr(self, '_last_reset_call', {}).get(video_key, 0)
                    is_lockout = (call_num - last_reset) < 5
                    
                    # v5.1.3: Asegurar inicialización de variables de suavizado
                    f_center = 0.90
                    f_size = 0.95
                    kps_ema = 0.85

                    if velocity > 160 and not is_lockout:
                        print(f"[SCENE_CUT] Frame {call_num}: Salto detectado ({velocity:.0f}px). Reseteando historial temporal.")
                        if not hasattr(self, '_last_reset_call'): self._last_reset_call = {}
                        self._last_reset_call[video_key] = call_num

                        for attr in ['_color_ema', '_brightness_ema', '_enhancer_ema', '_mouth_state_history']:
                            if hasattr(self, attr):
                                history = getattr(self, attr)
                                if isinstance(history, dict) and video_key in history:
                                    del history[video_key]
                        
                        f_center = 1.0
                        f_size = 1.0
                        kps_ema = 1.0
                    else:
                        # v5.60: Allow more raw data if high intensity requested, but with some smoothing to avoid jitter
                        if user_blend >= 0.95:
                            f_center = 0.95
                            kps_ema = 0.92
                        else:
                            # v5.60: Perfiles con EMA más suavizado para evitar jitter (0.60->0.75, 0.70->0.82, 0.80->0.88)
                            if velocity < 4:
                                f_center = 0.85
                                kps_ema = 0.80 if not is_profile else 0.75
                            elif velocity < 12:
                                f_center = 0.90
                                kps_ema = 0.85 if not is_profile else 0.82
                            else:
                                f_center = 0.95
                                kps_ema = 0.90 if not is_profile else 0.88
                        
                        f_size = 0.95
                    
                    curr_size = curr_bbox[2:] - curr_bbox[:2]
                    prev_size = prev_bbox_arr[2:] - prev_bbox_arr[:2]
                    
                    new_center = f_center * curr_center + (1 - f_center) * prev_center
                    new_size = f_size * curr_size + (1 - f_size) * prev_size
                    
                    # Reconstruir BBox
                    smoothed_bbox = np.zeros(4)
                    smoothed_bbox[:2] = new_center - new_size / 2
                    smoothed_bbox[2:] = new_center + new_size / 2
                    target_face.bbox = tuple(smoothed_bbox.astype(int))

                    # 2. Suavizado de Landmarks (KPS) - CRÍTICO PARA ALINEACIÓN
                    if prev_kps is not None and hasattr(target_face, 'kps') and target_face.kps is not None:
                        curr_kps = np.array(target_face.kps)
                        prev_kps_arr = np.array(prev_kps)
                        
                        # APLICAR suavizado de landmarks
                        new_kps = kps_ema * curr_kps + (1 - kps_ema) * prev_kps_arr
                        target_face.kps = new_kps.tolist()

                # Guardar estados para el siguiente frame
                if video_key:
                    if not hasattr(self, '_prev_face_bbox'): self._prev_face_bbox = {}
                    if not hasattr(self, '_prev_face_kps'): self._prev_face_kps = {}
                    self._prev_face_bbox[video_key] = target_face.bbox
                    if hasattr(target_face, 'kps'): self._prev_face_kps[video_key] = target_face.kps

            # ============================================
            # NUEVO: DETECTAR BOCA ABIERTA (Con Histeresis)
            # ============================================
            preserve_mouth = getattr(roop.globals, 'preserve_mouth_expression', True)
            mouth_open = False
            mouth_region = None
            mouth_open_ratio = 0.0

            if preserve_mouth:
                try:
                    from roop.processors.FaceSwap import detect_mouth_open, create_mouth_preservation_mask
                except ImportError:
                    mouth_open = False
                else:
                    landmarks_106 = getattr(target_face, 'landmark_106', None)
                    mouth_open, mouth_region, mouth_open_ratio = detect_mouth_open(target_face, landmarks_106, result_frame)
                    
                    # Histeresis temporal para evitar parpadeo de apertura/cierre
                    video_key = getattr(self, '_current_video_key', 'default')
                    if not hasattr(self, '_mouth_state_history'): self._mouth_state_history = {}
                    prev_state = self._mouth_state_history.get(video_key, False)
                    
                    # v5.3: Umbral dinámico con histeresis — preservar boca incluso con apertura leve
                    thresh = 0.30 if is_profile else (0.05 if prev_state else 0.15)
                    if is_profile:
                        mouth_open_ratio = min(mouth_open_ratio, 0.5)
                    mouth_open = mouth_open_ratio > thresh
                    self._mouth_state_history[video_key] = mouth_open
                    
                    if mouth_open:
                        print(f"[MOUTH_PRESERVE] Frame {call_num}: Boca abierta (ratio={mouth_open_ratio:.2f})")

            # ============================================
            # 5. COORDENADAS FINALES (v3.5)
            # ============================================
            x1, y1, x2, y2 = target_face.bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(result_frame.shape[1], int(x2)), min(result_frame.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return original_frame

            # v5.59: Skip-swap solo para tracking extremadamente malo — det_score < 0.05 + velocity > 100px
            det_score = getattr(target_face, 'det_score', 1.0)
            velocity = getattr(self, '_last_velocity', 0)
            if det_score < 0.05 and velocity > 100:
                if enable_temporal_smoothing and prev_frame_result is not None:
                    print(f"[SKIP] Frame {call_num}: swap saltado (det_score={det_score:.2f}, vel={velocity:.0f}px)")
                    return prev_frame_result
                return original_frame

            # ============================================
            # 1. IDENTITY INJECTION (v5.64: ADN Maestro GOD-MIX)
            # ============================================
            # Inyectar identidad frontal (Master) para estabilizar rasgos en ángulos difíciles
            swap_source_face = source_face
            if hasattr(self, 'master_source_embedding') and self.master_source_embedding is not None:
                # v5.69: Inyección absoluta (100% ambas) — embedding puro del master para identidad máxima
                dna_mix = 1.0
                if dna_mix > 0:
                    import copy
                    swap_source_face = copy.copy(source_face)
                    mixed_emb = (1.0 - dna_mix) * np.array(source_face.embedding) + dna_mix * self.master_source_embedding
                    norm = np.linalg.norm(mixed_emb)
                    if norm > 0:
                        swap_source_face.embedding = (mixed_emb / norm).tolist()

            # ============================================
            # 2. SWAP
            # ============================================
            res_data = self.processors["faceswap"].Run(swap_source_face, target_face, result_frame, paste_back=False)
            if res_data is None:
                with open(os.path.join(os.path.dirname(__file__), '..', 'debug_swap.log'), 'a') as lf:
                    lf.write(f"[SWAP_FAIL] Frame {call_num}: FaceSwap devolvió None\n")
                if enable_temporal_smoothing and prev_frame_result is not None:
                    return prev_frame_result
                return original_frame
            
            # res_data puede ser (swapped_face, M) o solo swapped_face
            if isinstance(res_data, tuple):
                swapped_face_aligned, M = res_data
            else:
                swapped_face_aligned = res_data
                M = None # Algunos modelos lo hacen interno

            # v5.2.7: Guardar raw swap ANTES de GFPGAN para detección precisa de oclusiones
            raw_swapped_aligned = swapped_face_aligned.copy()

            # v5.57: M-EMA adaptivo con menos smoothing en perfiles para tracking más rápido
            if M is not None and enable_temporal_smoothing:
                m_attr = f'_m_ema_{video_basename}'
                prev_m = getattr(self, m_attr, None)
                if prev_m is not None:
                    # Detectar si hay un cambio brusco (escena nueva) para no suavizar
                    m_diff = np.max(np.abs(M - prev_m))
                    if m_diff < 15.0: # Umbral para jitter vs movimiento real
                        m_alpha = 0.70 if m_diff < 5.0 else 0.50
                        if is_profile:
                            m_alpha *= 0.10  # v5.65: rigidity-pro en perfiles para seguimiento instantáneo
                        M = M * (1.0 - m_alpha) + prev_m * m_alpha
                setattr(self, m_attr, M)

            # DEBUG: guardar raw model output para frame 1-3
            if call_num <= 3:
                debug_dir = os.path.join(os.path.dirname(__file__), '..', 'debug_swap')
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(os.path.join(debug_dir, f'01_raw_model_f{call_num}.png'), swapped_face_aligned)

            # ============================================
            # MODO AUTO-PILOT v5.0: INTENSIDAD DINÁMICA
            # ============================================
            # v5.0: Respetar el Blend Ratio del usuario (UI) para la intensidad total
            user_blend = getattr(self.options, 'blend_ratio', 1.0)
            
            use_enhancer = True # Forzar siempre ON
            selected_enhancer = "GFPGAN" # Forzar el mejor
            # v5.71: enhancer_blend 0.70, Master Embedding weighted top-3
            enhancer_blend = 0.70
            preserve_mouth = True # Evitar borrar gestos
            
            enhancer_key = next((k for k in ["enhance_gfpgan", "enhance_codeformer", "enhance_restoreformer"] if k in self.processors), None)
            if enhancer_key is not None:
                # v5.66: Saltar enhancer en perfiles de baja detección (evita alucinaciones GFPGAN)
                det_score_enh = getattr(target_face, 'det_score', 1.0)
                skip_enhancer = is_profile and det_score_enh < 0.45
                if skip_enhancer and call_num % 50 == 1:
                    print(f"[QUALITY] Enhancer SKIP (perfil det_score={det_score_enh:.2f})")
                if not skip_enhancer:
                    try:
                        # Aplicar enhancer directamente sobre la cara alineada (calidad superior)
                        class FaceSetMock:
                            def __init__(self, face): self.faces = [face]; self.ref_images = []
                        
                        enhanced = self.processors[enhancer_key].Run(FaceSetMock(source_face), None, swapped_face_aligned)
                        if enhanced is not None:
                            if isinstance(enhanced, tuple): enhanced = enhanced[0]
                            if enhanced.shape[:2] != swapped_face_aligned.shape[:2]:
                                enhanced = cv2.resize(enhanced, (swapped_face_aligned.shape[1], swapped_face_aligned.shape[0]))
                            
                            # Estabilización temporal EMA del enhancer
                            if enable_temporal_smoothing and video_key:
                                if not hasattr(self, '_enhancer_ema'): self._enhancer_ema = {}
                                prev_enh = self._enhancer_ema.get(video_key)
                                if prev_enh is not None and prev_enh.shape == enhanced.shape:
                                    enh_ema_alpha = 0.80 # v5.64: más responsivo para texturas dinámicas
                                    enhanced = cv2.addWeighted(enhanced, enh_ema_alpha, prev_enh, 1.0 - enh_ema_alpha, 0)
                                self._enhancer_ema[video_key] = enhanced.copy()
                            
                            # v5.60: Radial GFPGAN fade mejorado (radio 60->75) — centro 100% GFPGAN, bordes raw
                            h_f, w_f = enhanced.shape[:2]
                            Y_f, X_f = np.ogrid[:h_f, :w_f]
                            center_f = (w_f / 2, h_f / 2)
                            dist_f = np.sqrt((X_f - center_f[0])**2 + (Y_f - center_f[1])**2)
                            fade = 1.0 - 1.0 / (1.0 + np.exp(-0.10 * (dist_f - 75.0)))
                            fade_3ch = np.stack([fade, fade, fade], axis=-1).astype(np.float32)
                            alpha = enhancer_blend * fade_3ch
                            blended_face = enhanced.astype(np.float32) * alpha + swapped_face_aligned.astype(np.float32) * (1.0 - alpha)
                            swapped_face_aligned = np.clip(blended_face, 0, 255).astype(np.uint8)
                            
                            if call_num <= 3:
                                cv2.imwrite(os.path.join(debug_dir, f'02_after_enhancer_f{call_num}.png'), swapped_face_aligned)
                            
                            # v5.65: Unsharp sigma 1.0 + amount 6.8 (HYPER-SHARP)
                            blurred = cv2.GaussianBlur(swapped_face_aligned, (0, 0), 1.0)
                            swapped_face_aligned = cv2.addWeighted(swapped_face_aligned, 6.8, blurred, -5.8, 0)
                            if call_num % 50 == 1:
                                print(f"[QUALITY] Enhancer ({enhancer_blend:.2f}) + unsharp mask (v5.71)")
                    except Exception as e:
                        print(f"[AUTO_PILOT_ERR] {e}")

            # ============================================
            # 3. COLOR Y BRILLO (ESTABILIZADO EMA v2.7)
            # ============================================
            # Obtener región de referencia para color matching
            h_align, w_align = swapped_face_aligned.shape[:2]
            if M is not None:
                # Extraer región del original que corresponde a la alineación
                reference_region = cv2.warpAffine(original_frame, M, (w_align, h_align), borderMode=cv2.BORDER_REPLICATE)
            else:
                reference_region = original_frame[y1:y2, x1:x2].copy()
                reference_region = cv2.resize(reference_region, (w_align, h_align))

            # Estabilización de Brillo EMA
            # v5.45: Brillo matching fortalecido 0.20→0.25 para mejor integración con color_match_strength 0.30/0.35
            brightness_strength = 0.25
            if enable_temporal_smoothing and video_key:
                if not hasattr(self, '_brightness_ema'): self._brightness_ema = {}
                prev_bright = self._brightness_ema.get(video_key, 1.0)
                
                curr_bright_ratio = np.mean(cv2.cvtColor(reference_region, cv2.COLOR_BGR2GRAY)) / (np.mean(cv2.cvtColor(swapped_face_aligned, cv2.COLOR_BGR2GRAY)) + 1e-6)
                curr_bright_ratio = np.clip(curr_bright_ratio, 0.7, 1.3)
                
                smooth_bright = 0.2 * curr_bright_ratio + 0.8 * prev_bright
                self._brightness_ema[video_key] = smooth_bright
                
                face_lab = cv2.cvtColor(swapped_face_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
                adj = 1.0 + (smooth_bright - 1.0) * brightness_strength
                face_lab[:, :, 0] = np.clip(face_lab[:, :, 0] * adj, 0, 255)
                swapped_face_aligned = cv2.cvtColor(face_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

            # v5.61: Color matching equilibrado (0.15 frontal, 0.25 perfil) para preservar parecido
            color_match_strength = 0.25 if is_profile else 0.15
            swapped_face_aligned = match_color_histogram(swapped_face_aligned, reference_region, blend_factor=color_match_strength)

            if enable_temporal_smoothing and video_key:
                if not hasattr(self, '_color_ema'): self._color_ema = {}
                prev_color = self._color_ema.get(video_key)
                if prev_color is not None and prev_color.shape == swapped_face_aligned.shape:
                    # v5.65: Color EMA instant-photon (0.95 nuevo) para cero lag lumínico
                    color_ema_alpha = 0.95 if user_blend >= 0.95 else 0.88
                    swapped_face_aligned = cv2.addWeighted(swapped_face_aligned, color_ema_alpha, prev_color, 1.0 - color_ema_alpha, 0)
                self._color_ema[video_key] = swapped_face_aligned

            # ============================================
            # 4. WARP BACK Y MASKING (PRECISIÓN v4.7: XSeg + Profile)
            # ============================================
            h_f, w_f = result_frame.shape[:2]

            if call_num <= 3:
                cv2.imwrite(os.path.join(debug_dir, f'03_after_color_f{call_num}.png'), swapped_face_aligned)

            if M is not None:
                M_inv = cv2.invertAffineTransform(M)
                
                # A. Warp de la cara (BORDER_REFLECT para evitar halo oscuro en bordes de máscara)
                warped_face = cv2.warpAffine(swapped_face_aligned, M_inv, (w_f, h_f), borderMode=cv2.BORDER_REFLECT)

                # Safety fill: rellenar áreas negras del warp con el frame original para evitar "máscara negra"
                fb = np.max(warped_face.astype(np.float32), axis=2) < 1
                if np.any(fb):
                    fb_blur = cv2.GaussianBlur(fb.astype(np.float32), (5, 5), 0)
                    for c in range(3):
                        warped_face[:,:,c] = (warped_face[:,:,c].astype(np.float32) * (1 - fb_blur) + original_frame[:,:,c].astype(np.float32) * fb_blur).astype(np.uint8)

                # DEBUG: Diagnóstico de alineación (solo primeros frames)
                if call_num <= 3:
                    print(f"[ALIGN_DEBUG] Frame {call_num}:")
                    print(f"[ALIGN_DEBUG]   M (2x3) = [{M[0,0]:.4f}, {M[0,1]:.4f}, {M[0,2]:.2f}; {M[1,0]:.4f}, {M[1,1]:.4f}, {M[1,2]:.2f}]")
                    print(f"[ALIGN_DEBUG]   M_inv (2x3) = [{M_inv[0,0]:.4f}, {M_inv[0,1]:.4f}, {M_inv[0,2]:.2f}; {M_inv[1,0]:.4f}, {M_inv[1,1]:.4f}, {M_inv[1,2]:.2f}]")
                    print(f"[ALIGN_DEBUG]   target_face.kps = {target_face.kps}")
                    print(f"[ALIGN_DEBUG]   target_face.bbox = {target_face.bbox}")
                    # Mapear las 4 esquinas de la cara alineada (128x128) al frame original
                    h_a, w_a = swapped_face_aligned.shape[:2]
                    corners = np.array([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]], dtype=np.float32)
                    corners_warped = cv2.transform(corners.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
                    print(f"[ALIGN_DEBUG]   swapped_face_aligned size: {w_a}x{h_a}")
                    print(f"[ALIGN_DEBUG]   Corners after warp: TL({corners_warped[0,0]:.0f},{corners_warped[0,1]:.0f}) TR({corners_warped[1,0]:.0f},{corners_warped[1,1]:.0f}) BR({corners_warped[2,0]:.0f},{corners_warped[2,1]:.0f}) BL({corners_warped[3,0]:.0f},{corners_warped[3,1]:.0f})")
                    # Región no-cero en warped_face (contenido real)
                    ys, xs = np.where(np.max(warped_face.astype(np.float32), axis=2) > 5)
                    if len(ys) > 0:
                        print(f"[ALIGN_DEBUG]   warped_face non-zero bbox: x=[{xs.min()},{xs.max()}] y=[{ys.min()},{ys.max()}]")
                    else:
                        print(f"[ALIGN_DEBUG]   warped_face: ALL ZERO (warp falló)")
                    emb_log = list(swap_source_face.embedding[:3]) if isinstance(swap_source_face.embedding, list) else swap_source_face.embedding[:3].tolist()
                    print(f"[ALIGN_DEBUG]   swap_source_face embedding[:3] = {emb_log}")
                    print(f"[ALIGN_DEBUG]   Frame size: {w_f}x{h_f}")
                
                # B. NUEVO: Generar máscara XSeg de alta precisión
                final_mask = None
                if "mask_xseg" in self.processors:
                    try:
                        # La máscara XSeg se genera sobre la cara alineada (calidad nativa 256x256)
                        xseg_mask = self.processors["mask_xseg"].Run(swapped_face_aligned, "")
                        if xseg_mask is not None and np.max(xseg_mask) > 0:
                            # v5.45: Umbral bajado 0.40→0.25 para incluir más zona facial y evitar recorte elíptico visible
                            xseg_thresh = 0.25
                            _, xseg_mask = cv2.threshold(xseg_mask, xseg_thresh, 1.0, cv2.THRESH_BINARY)

                            # v5.4.3: Atenuación superior reducida al 15% para ambos modos
                            h_x, w_x = xseg_mask.shape[:2]
                            fade_pct = 0.15
                            gradient_x = np.ones((h_x, w_x), dtype=np.float32)
                            fade_h = int(h_x * fade_pct)
                            for row in range(fade_h):
                                gradient_x[row, :] = (row / fade_h) ** 1.0
                            xseg_mask = xseg_mask * gradient_x

                            # Warp de la máscara XSeg al espacio del frame
                            final_mask = cv2.warpAffine(xseg_mask, M_inv, (w_f, h_f), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    except Exception as e:
                        print(f"[MASK_ERR] XSeg falló: {e}")

                # C. FALLBACK: Si XSeg no está o falló, usar elipse dinámica mejorada
                if final_mask is None:
                    mask_align = np.zeros((h_align, w_align), dtype=np.float32)
                    if is_profile:
                        # v5.53: Elipse perfil reducida 0.65/0.70→0.50/0.50 para minimizar blur fuera de cara
                        cv2.ellipse(mask_align, (w_align//2, h_align//2),
                                    (int(w_align*0.50), int(h_align*0.50)), 0, 0, 360, 1.0, -1)
                        mask_align = cv2.GaussianBlur(mask_align, (21, 21), 0)
                    else:
                        # v5.53: Elipse frontal reducida 0.55/0.58→0.45/0.48 para eliminar blur elíptico
                        cv2.ellipse(mask_align, (w_align//2, h_align//2),
                                    (int(w_align*0.45), int(h_align*0.48)), 0, 0, 360, 1.0, -1)
                    
                    # v5.4: Atenuación superior generosa en fallback (10%)
                    h_a, w_a = mask_align.shape[:2]
                    grad_a = np.ones((h_a, w_a), dtype=np.float32)
                    fade_ha = int(h_a * 0.10)
                    for row in range(fade_ha):
                        grad_a[row, :] = (row / fade_ha) ** 0.5
                    mask_align = mask_align * grad_a

                    final_mask = cv2.warpAffine(mask_align, M_inv, (w_f, h_f), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # D. Aplicar oclusión (v5.2: Independiente del blend ratio para proteger objetos siempre)
                occ_mask_aligned = detect_foreground_occlusion(raw_swapped_aligned, reference_region)
                
                # v5.65: Estabilización temporal (EMA) de oclusión para evitar parpadeo en videos
                if enable_temporal_smoothing and video_key:
                    prev_occ = self._occ_mask_ema.get(video_key)
                    if prev_occ is not None and prev_occ.shape == occ_mask_aligned.shape:
                        occ_ema_alpha = 0.85 # v5.65: alta respuesta pero sin ruido
                        occ_mask_aligned = cv2.addWeighted(occ_mask_aligned, occ_ema_alpha, prev_occ, 1.0 - occ_ema_alpha, 0)
                    self._occ_mask_ema[video_key] = occ_mask_aligned.copy()

                # v5.4.3: Fuerza de oclusión reducida a 0.50 para no destruir la máscara facial
                occ_strength = (0.60 if is_profile else 0.50) 
                
                if np.max(occ_mask_aligned) > 0.1:
                    occ_mask_frame = cv2.warpAffine(occ_mask_aligned, M_inv, (w_f, h_f), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    final_mask = np.clip(final_mask - (occ_mask_frame * occ_strength), 0, 1.0)
                    if call_num % 100 == 1:
                        print(f"[QUALITY] Oclusión aplicada (fuerza={occ_strength:.2f})")

                # v5.65: GaussianBlur //75 (bordes microscópicos) + erosión 5×5
                blur_sz = int(max(5, min(x2-x1, y2-y1) // 75)) | 1
                kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                final_mask = cv2.erode(final_mask, kernel_e, iterations=1)
                final_mask = cv2.GaussianBlur(final_mask, (blur_sz, blur_sz), 0)

                # v5.65: Content feathering "Surgical-Seams" (offset 1) para eliminar blur residual
                content_blur_sz = int(max(31, blur_sz + 1)) | 1
                tent = np.clip(1.0 - np.abs(final_mask - 0.5) * 2.0, 0, 1.0)
                if np.max(tent) > 0.01:
                    blurred_face = cv2.GaussianBlur(warped_face, (content_blur_sz, content_blur_sz), 0)
                    tent_3ch = cv2.merge([tent, tent, tent])
                    warped_face = (warped_face.astype(np.float32) * (1.0 - tent_3ch) + blurred_face.astype(np.float32) * tent_3ch).astype(np.uint8)

                # v5.57: Tail truncation 0.10 + erosión 5×5 para máscara limpia
                final_mask = np.clip((final_mask - 0.10) / (1.0 - 0.10), 0, 1.0)
            else:
                # Fallback de emergencia
                print("[WARNING] M es None, usando fallback de elipse directa")
                warped_face = result_frame.copy()
                warped_face[y1:y2, x1:x2] = cv2.resize(swapped_face_aligned, (x2-x1, y2-y1))
                final_mask = create_soft_mask((x1, y1, x2, y2), (h_f, w_f), feather=40)

            # ============================================
            # 5. PRESERVACIÓN DE BOCA (Con detección MediaPipe 468)
            # ============================================
            if mouth_open and mouth_region is not None:
                # v5.65: Detección inteligente de objetos en boca (micros, comida, manos)
                m_blend = 0.85
                
                try:
                    # Usar la oclusión detectada en la zona de la boca para subir el blend si hay objetos
                    mouth_mask_aligned = create_mouth_preservation_mask(reference_region, mouth_region, blend_ratio=1.0)
                    mouth_occ_score = np.mean(cv2.bitwise_and(occ_mask_aligned, mouth_mask_aligned))
                    
                    if mouth_occ_score > 0.02: # Si hay oclusión en >2% de la boca
                        # v5.65: Protección dinámica (hasta 0.85) para objetos
                        m_blend = min(0.85, 0.50 + mouth_occ_score * 5.0)
                        if call_num % 50 == 1:
                            print(f"[MOUTH_OBJECT] Objeto detectado (score={mouth_occ_score:.3f}). m_blend={m_blend:.2f}")
                except:
                    pass

                mouth_mask = create_mouth_preservation_mask(original_frame, mouth_region, blend_ratio=1.0)

                # v5.69: Dilatar máscara de boca para proteger área completa de labios
                mouth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                mouth_mask = cv2.dilate(mouth_mask, mouth_kernel, iterations=2)

                # v5.2.3: Suavizado de boca más nítido (9x9) para no borrar el labio inferior
                mouth_mask = cv2.GaussianBlur(mouth_mask, (15, 15), 0)

                m_impact = (mouth_mask * m_blend).mean()
                final_mask = np.clip(final_mask - (mouth_mask * m_blend), 0, 1.0)

                if call_num % 50 == 1:
                    print(f"[QUALITY] Boca restaurada (impacto={m_impact:.3f}, fuerza={m_blend*100:.0f}%)")
            # ============================================
            # 6. DEBUG MASK
            # ============================================
            if call_num <= 3:
                cv2.imwrite(os.path.join(debug_dir, f'05_mask_final_f{call_num}.png'), (final_mask * 255).astype(np.uint8))
                cv2.imwrite(os.path.join(debug_dir, f'06_warped_face_f{call_num}.png'), warped_face)
                cv2.imwrite(os.path.join(debug_dir, f'07_original_frame_f{call_num}.png'), original_frame)
                print(f"[DEBUG_MASK] Mask mean={final_mask.mean():.3f} max={final_mask.max():.3f} applied", flush=True)
            # ============================================
            # 7. BLENDING FINAL (UNIFICADO)
            # ============================================
            mask_3ch = cv2.merge([final_mask, final_mask, final_mask])
            
            # Atenuación por perfil, velocidad o "ghosting"
            velocity = getattr(self, '_last_velocity', 0)
            is_ghost = getattr(target_face, 'is_ghost', False)
            
            # v5.1.4: Bloquear swap_weight a 1.0 si el usuario lo pide (máxima potencia)
            swap_weight = 1.0 if user_blend >= 0.98 else user_blend
            
            if is_ghost:
                # Si es una proyección (tracking perdido), reducir peso drásticamente
                lost_count = getattr(self, f'_tracking_lost_count_{video_basename}', 1)
                swap_weight *= max(0.35, 0.90 - lost_count * 0.04)
            
            if is_profile:
                # Perfil: peso completo excepto cuando tracking es inestable (bajo det_score o alta velocidad)
                if velocity > 80:
                    swap_weight *= max(0.5, 1.0 - (velocity - 80) / 200)
            
            if not is_profile and velocity > 100:
                swap_weight *= max(0.6, 1.0 - (velocity - 100) / 200)

            mask_3ch = mask_3ch * swap_weight

            result_frame = (warped_face.astype(np.float32) * mask_3ch + 
                           original_frame.astype(np.float32) * (1.0 - mask_3ch)).astype(np.uint8)

            # v5.2.3: Estabilización de GHOST (Si el tracking falla, mezclar con el frame anterior)
            if is_ghost and enable_temporal_smoothing and prev_frame_result is not None:
                if prev_frame_result.shape == result_frame.shape:
                    # Mezcla temporal 50/50 para ocultar saltos de alineación en predicción
                    result_frame = cv2.addWeighted(result_frame, 0.5, prev_frame_result, 0.5, 0)

            # v5.50: EMA con congelamiento progresivo según calidad de tracking
            if enable_temporal_smoothing and prev_frame_result is not None:
                if prev_frame_result.shape == result_frame.shape:
                    det_score = getattr(target_face, 'det_score', 1.0)
                    if det_score < 0.3:
                        alpha_prev = 0.85  # congelamiento parcial — 15% del swap nuevo
                    elif det_score < 0.4:
                        alpha_prev = 0.75  # casi congelar
                    elif det_score < 0.6:
                        alpha_prev = 0.40 if is_profile else 0.30
                    else:
                        alpha_prev = 0.25 if is_profile else 0.15
                    result_frame = cv2.addWeighted(result_frame, 1.0 - alpha_prev, prev_frame_result, alpha_prev, 0)
                    if call_num % 50 == 1:
                        print(f"[QUALITY] EMA v5.59 (det_score={det_score:.2f}, alpha_prev={alpha_prev:.2f})")

            if call_num <= 3:
                cv2.imwrite(os.path.join(debug_dir, f'04_final_result_f{call_num}.png'), result_frame)

            mask_mean = final_mask.mean() if 'final_mask' in dir() else -1
            with open(os.path.join(os.path.dirname(__file__), '..', 'debug_swap.log'), 'a') as lf:
                lf.write(f"[SWAP_END] Frame {call_num}: resultado devuelto. Mask mean={mask_mean:.3f}\n")

            # Guardar estado para el siguiente frame
            if enable_temporal_smoothing and video_key:
                if not hasattr(self, '_prev_frame_result'): self._prev_frame_result = {}
                self._prev_frame_result[video_key] = result_frame.copy()
            
            return result_frame
            
        except Exception as e:
            print(f"[DEBUG] Error en _process_face_swap_v21: {e}", flush=True)
            return original_frame
            
        finally:
            # Limpieza de memoria
            import gc
            gc.collect()

    def _get_gender_confidence(self, face):
        try:
            if hasattr(face, 'gender_confidence'):
                return float(face.gender_confidence)
            
            if hasattr(face, 'gender'):
                gender_value = face.gender
                if isinstance(gender_value, (list, np.ndarray)):
                    gender_value = gender_value[0] if len(gender_value) > 0 else None
                
                if gender_value is not None:
                    gender_value = float(gender_value)
                    
                    if gender_value == 0 or gender_value == 0.0 or gender_value == 1 or gender_value == 1.0:
                        return 0.95
                    
                    if 0.0 < gender_value < 1.0:
                        if gender_value < 0.35 or gender_value > 0.65:
                            return 0.90
                        elif gender_value < 0.45 or gender_value > 0.55:
                            return 0.75
                        else:
                            return 0.50
            
            return 0.50
            
        except Exception as e:
            return 0.50

    def _get_face_tracking_key(self, face):
        try:
            if hasattr(face, 'bbox') and face.bbox is not None:
                x1, y1, x2, y2 = face.bbox
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                return f"face_{center_x}_{center_y}"
            return f"id_{id(face)}"
        except Exception:
            return f"fallback_{id(face)}"
    
    def _get_face_key(self, face):
        """Genera una clave única para una cara basada en su posición (para evitar doble procesamiento)"""
        try:
            if hasattr(face, 'bbox') and face.bbox is not None:
                x1, y1, x2, y2 = face.bbox
                # Usar coordenadas discretizadas para tolerate small movements
                grid_size = 20
                grid_x = int((x1 + x2) / 2 / grid_size) * grid_size
                grid_y = int((y1 + y2) / 2 / grid_size) * grid_size
                return f"face_{grid_x}_{grid_y}"
            return f"id_{id(face)}"
        except Exception:
            return f"fallback_{id(face)}"

    def _calculate_similarity(self, emb1, emb2):
        try:
            if emb1 is None or emb2 is None:
                return 0.0
            
            emb1 = np.array(emb1, dtype=np.float32).flatten()
            emb2 = np.array(emb2, dtype=np.float32).flatten()
            
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            emb1_norm = emb1 / norm1
            emb2_norm = emb2 / norm2
            
            # Asegurar que sea escalar
            similarity = float(np.dot(emb1_norm, emb2_norm))
            return max(0.0, min(1.0, similarity))
        except Exception as e:
            print(f"[SIMILARITY_ERROR] {e}")
            return 0.0

    def run_batch_inmem(self, video_path, output_path, start_frame=0, end_frame=None, fps=24.0, num_threads=1, skip_audio=False):
        _log = r'D:\PROJECTS\AUTOAUTO\debug_swap.log'
        with open(_log, 'a') as lf:
            lf.write(f"[RUN_BATCH] INICIO video_path={video_path}\n")
        try:
            if not self.is_initialized:
                print("ProcessMgr not initialized")
                return

            import cv2
            import os
            import time
            from pathlib import Path
            from concurrent.futures import ThreadPoolExecutor, as_completed
            from tqdm import tqdm

            if self.options:
                self.options.current_video_path = video_path

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Could not open video: {video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if end_frame is None or end_frame > total_frames:
                end_frame = total_frames
            if start_frame < 0:
                start_frame = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Calcular total de frames a procesar
            total_frames_to_process = end_frame - start_frame
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_output = output_path + ".temp.mp4"
            out = cv2.VideoWriter(temp_output, fourcc, fps_video, (width, height))

            if not out.isOpened():
                print(f"Could not create output video: {temp_output}")
                cap.release()
                return

            print(f"[VIDEO] {os.path.basename(video_path)} | {width}x{height} | {fps_video:.2f} fps | {total_frames} frames | rango {start_frame}-{end_frame}")
            print(f"[LOAD] FaceSets: {len(self.input_facesets)} sets, {sum(len(fs.faces) for fs in self.input_facesets if hasattr(fs,'faces') and fs.faces)} caras totales")

            # Yield inicial
            yield (0, f"Iniciando: {os.path.basename(video_path)}")

            # ============================================
            # INICIALIZAR MÉTRICAS EN TIEMPO REAL
            # ============================================
            from roop.metrics_tracker import MetricsTracker, set_current_tracker

            metrics = MetricsTracker(total_frames=end_frame - start_frame)
            set_current_tracker(metrics)
            metrics.start()
            
            # ============================================
            # BATCH PROCESSING - Procesamiento en paralelo
            # ============================================
            # Leer todos los frames primero para procesamiento en batch
            batch_size = getattr(roop.globals, 'batch_processing_size', 4)
            max_workers = getattr(roop.globals, 'max_batch_threads', 4)
            
            print(f"[BATCH] Configuración: batch_size={batch_size}, max_workers={max_workers}")
            print(f"[BATCH] Leyendo frames para procesamiento en paralelo...")
            
            # Inicializar contador de frames
            current_frame = start_frame
            
            print(f"[BATCH] Iniciando procesamiento secuencial con streaming...")
            start_time = time.time()
            last_yield_time = time.time()
            yield_interval = 1.0  # Actualizar UI cada 1 segundo
            
            processed_count = 0
            
            # Procesar frames uno a uno sin cargarlos todos en memoria
            while True:
                ret, frame = cap.read()
                if not ret or current_frame >= end_frame:
                    break
                
                try:
                    # Procesar frame secuencialmente con smoothing
                    processed = self.process_frame(frame, enable_temporal_smoothing=True, file_path=video_path)
                    if processed is not None:
                        out.write(processed)
                    else:
                        out.write(frame)

                    processed_count += 1
                    
                    # Métricas
                    try:
                        from roop.metrics_tracker import _current_tracker
                        if _current_tracker:
                            _current_tracker.update_frame_processed()
                    except:
                        pass

                    # Yield de métricas periódicamente
                    current_time = time.time()
                    if current_time - last_yield_time >= yield_interval:
                        elapsed_processing = current_time - start_time
                        fps_current = processed_count / elapsed_processing if elapsed_processing > 0 else 0
                        progress_pct = (processed_count / total_frames_to_process) * 100
                        remaining_frames = total_frames_to_process - processed_count
                        eta_seconds = remaining_frames / fps_current if fps_current > 0 else 0
                        eta_str = f"{int(eta_seconds // 60):02d}:{int(eta_seconds % 60):02d}"

                        yield (progress_pct, f"{fps_current:.1f} FPS | {processed_count}/{total_frames_to_process} frames | ETA: {eta_str}")
                        last_yield_time = current_time

                except Exception as e:
                    print(f"[BATCH] Error frame {current_frame}: {e}")
                    out.write(frame)
                
                current_frame += 1
            
            cap.release()
            out.release()
            
            elapsed_time = time.time() - start_time
            fps_processed = processed_count / elapsed_time if elapsed_time > 0 else 0

            print(f"[BATCH] Procesamiento completado en {elapsed_time:.2f}s ({fps_processed:.2f} fps)")
            
            import gc
            gc.collect()
            
            print(f"[BATCH] Memoria liberada")

            # Audio: restaurar ANTES de mover el archivo temporal
            if not skip_audio:
                try:
                    from roop.util_ffmpeg import restore_audio
                    restore_audio(temp_output, video_path, None, None, output_path)
                    print(f"[BATCH] Audio restaurado desde: {video_path}")
                except Exception as e:
                    print(f"[BATCH] Error restaurando audio: {e}")
                    # Si falla el audio, mover el video sin audio
                    import shutil
                    shutil.move(temp_output, output_path)
            else:
                # Sin audio, solo mover el archivo
                import shutil
                shutil.move(temp_output, output_path)

            print(f"[BATCH] Video guardado en: {output_path}")

            # Limpiar archivo temporal
            try:
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    print(f"[CLEANUP] Temporal eliminado: {temp_output}")
            except Exception as e:
                print(f"[CLEANUP] Error eliminando temporal: {e}")

            # Yield de completado
            yield (100, f"Video completado: {os.path.basename(output_path)}")

        except Exception as e:
            print(f"[BATCH] Error en run_batch_inmem: {e}")
            import traceback
            traceback.print_exc()
            
            # Si hay error, intentar procesar sin batch
            yield (0, f"Error: {e}. Reintentando...")
            
            # Reiniciar variables para el fallback
            current_frame = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            processed_count = 0
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    processed_frame = self.process_frame(frame, enable_temporal_smoothing=True)
                    if processed_frame is not None:
                        out.write(processed_frame)
                        processed_count += 1
                    else:
                        out.write(frame)
                except Exception as e:
                    print(f"[ERROR] Frame {current_frame}: error - {e}")
                    out.write(frame)
                
                current_frame += 1
                
                if current_frame % 10 == 0 or current_frame == end_frame:
                    progress_percent = (current_frame / total_frames_to_process) * 100 if total_frames_to_process > 0 else 0
                    msg = f"Procesando frame {current_frame}/{total_frames_to_process} ({progress_percent:.1f}%)"
                    yield (progress_percent, msg)
                    
                    if is_valid_progress_callback(self.progress_callback):
                        try: self.progress_callback(progress_percent, desc=msg)
                        except: pass

            cap.release()
            out.release()
            
            yield (100, "Frames procesados, finalizando video...")

            if not skip_audio and os.path.exists(temp_output):
                yield (100, "Fusionando audio...")
                try:
                    from roop.util_ffmpeg import restore_audio
                    restore_audio(temp_output, video_path, None, None, output_path)
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    print(f"Audio merged successfully: {output_path}")
                except Exception as e:
                    print(f"Audio merge failed: {e}")
                    if os.path.exists(temp_output):
                        try: os.rename(temp_output, output_path)
                        except: pass
            else:
                if os.path.exists(temp_output):
                    try: os.rename(temp_output, output_path)
                    except: pass

            yield (100, "Video completado con éxito")

    def Release(self):
        try:
            for processor in self.processors.values():
                if hasattr(processor, "Release"):
                    processor.Release()
            self.processors.clear()
            self.source_embeddings_cache.clear()
            self.face_assignment_cache.clear()
            self.face_position_history.clear()
            self.global_source_for_all_id = None
            self.selected_face_assignment_cache.clear()
            
            # LIMPIEZA DE VRAM DESPUÉS DE PROCESAR VIDEO
            self._cleanup_vram()
        except:
            pass

    def _cleanup_vram(self):
        """Limpia caché CUDA y libera VRAM después del procesamiento"""
        import gc
        import tempfile
        import os
        
        # 1. Limpiar caché CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[ProcessMgr] VRAM después de limpiar: {allocated:.2f} GB")
        
        # 2. Forzar garbage collector
        gc.collect()
        
        # 3. Limpiar temporales
        temp_dir = tempfile.gettempdir()
        try:
            for f in os.listdir(temp_dir):
                if f.startswith("temp_frame_") or f.startswith("faceswap_") or f.startswith("roop_"):
                    try:
                        os.remove(os.path.join(temp_dir, f))
                    except:
                        pass
        except:
            pass
        
        print("[ProcessMgr] Limpieza de VRAM completada")

    def _log_audio_merge_error(self, error, video_path, temp_output):
        error_type = type(error).__name__
        error_message = str(error)
        
        audio_exists = os.path.exists(video_path)
        temp_exists = os.path.exists(temp_output)
        
        print(f"[AUDIO_ERROR] Tipo: {error_type}")
        print(f"[AUDIO_ERROR] Mensaje: {error_message}")
        print(f"[AUDIO_ERROR] Archivo de video existe: {audio_exists}")
        print(f"[AUDIO_ERROR] Archivo temporal existe: {temp_exists}")
        
        if not audio_exists:
            print(f"[AUDIO_ERROR] El archivo de video no existe: {video_path}")
        if not temp_exists:
            print(f"[AUDIO_ERROR] El archivo temporal no existe: {temp_output}")

    def release_resources(self):
        self.Release()
