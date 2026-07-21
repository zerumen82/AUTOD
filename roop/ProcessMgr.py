print("\n" + "="*70)
print(">>> [TRUE-INTEGRATION] v5.1035 - Forehead fix + mouth object detection + closeup tracking <<<")
print("="*70 + "\n")

import os
import random
import threading
import traceback
import torch
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
    """Gestor de procesamiento de face swapping v5.1035"""
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
        # v5.1022: EMA único occ_frame_ema (eliminado doble EMA que causaba ghosting)
        self._occ_frame_ema = {}        # EMA local de occ en frame space (v5.1020)
        self._mouth_object_detected = {} # Flag de objeto en boca (v5.65)

        if is_valid_progress_callback(self.progress_callback):
            print(f"ProcessMgr v5.1016 inicializado con progress_callback (Top-K={self.selected_top_k}, TTL={self.selected_assignment_ttl})")
        else:
            print(f"ProcessMgr v5.78 inicializado SIN progress_callback (Top-K={self.selected_top_k}, TTL={self.selected_assignment_ttl})")

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

        print(f"ProcessMgr v5.1016 inicializado: {len(self.input_facesets)} facesets, {len(self.target_faces)} target faces")

        if len(self.input_facesets) == 0:
            print("[WARNING] No hay facesets de origen cargados")
        if len(self.target_faces) == 0:
            print("[WARNING] No hay caras destino seleccionadas")

        self.global_source_for_all_id = None
        self._auto_source_face_cache.clear()

    def _cache_source_embeddings(self):
        self.source_embeddings_cache.clear()
        self._source_face_entries = []
        all_embs_with_quality = []
        profile_embs_with_quality = []
        
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
                            is_prof = ProcessMgr._is_profile_face(getattr(face, 'kps', None))
                            all_embs_with_quality.append((unit_emb, quality))
                            self._source_face_entries.append({
                                'face': face, 'emb': unit_emb, 'quality': quality, 'is_profile': is_prof,
                            })
                            if is_prof:
                                profile_embs_with_quality.append((unit_emb, quality))
        
        self.master_source_embedding = None
        self.profile_master_embedding = None
        if len(all_embs_with_quality) > 0:
            all_embs_with_quality.sort(key=lambda x: x[1], reverse=True)
            best_emb = all_embs_with_quality[0][0]
            self.master_source_embedding = best_emb.copy()
            norm = np.linalg.norm(self.master_source_embedding)
            if norm > 0:
                self.master_source_embedding /= norm
            print(f"[IDENTITY] Master Embedding = mejor cara individual (quality={all_embs_with_quality[0][1]:.2f}, total {len(all_embs_with_quality)})")
        if profile_embs_with_quality:
            profile_embs_with_quality.sort(key=lambda x: x[1], reverse=True)
            self.profile_master_embedding = profile_embs_with_quality[0][0].copy()
            norm = np.linalg.norm(self.profile_master_embedding)
            if norm > 0:
                self.profile_master_embedding /= norm
            print(f"[IDENTITY] Profile Master = mejor cara de perfil (quality={profile_embs_with_quality[0][1]:.2f}, n={len(profile_embs_with_quality)})")
        
        print(f"v5.78: {len(self.source_embeddings_cache)} embeddings cacheados")

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
            if "faceswap" not in self.processors or len(self.input_facesets) == 0:
                return frame

            from roop.face_util_rotation import get_all_faces_smart
            target_faces_detected = get_all_faces_smart(frame, min_score=None, for_target=True)
            with open(_log, 'a') as lf:
                lf.write(f"[PROC_FRAME] detectadas {len(target_faces_detected)} caras\n")

            valid_faces = target_faces_detected

            video_path = getattr(self.options, 'current_video_path', None) or file_path
            is_real_video = getattr(self.options, 'current_video_path', None) is not None
            self._current_video_key = f"video_{os.path.basename(video_path)}" if video_path else "frame"

            if not valid_faces:
                with open(_log, 'a') as lf:
                    lf.write(f"[PROC_FRAME] SIN valid_faces - hold o frame\n")
                if is_real_video and video_path:
                    video_basename = os.path.basename(video_path)
                    # v5.1022b: Contador de frames sin cara — tras 3 devuelve original
                    empty_attr = f'_empty_faces_count_{video_basename}'
                    empty_count = getattr(self, empty_attr, 0) + 1
                    setattr(self, empty_attr, empty_count)
                    if empty_count <= 3 and self._should_hold_previous_swap(video_basename):
                        prev_full = getattr(self, '_prev_frame_result', {}).get(self._current_video_key)
                        last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                        if prev_full is not None and last_bbox is not None:
                            # v5.1002: prev_mask=None para elipse en posicion actual (last_bbox),
                            # prev_mask de frame anterior podria estar desfasado por ghost tracking
                            return self._hold_previous_face_region(frame, prev_full, last_bbox, prev_mask=None)
                return frame

            # Track average det_score for adaptive low-quality thresholds
            video_path_q = getattr(self.options, 'current_video_path', None) or (file_path or "")
            vkey_q = os.path.basename(video_path_q)
            if valid_faces:
                # v5.1022b: Reset contador de frames vacíos
                if is_real_video and video_path:
                    empty_attr = f'_empty_faces_count_{os.path.basename(video_path)}'
                    setattr(self, empty_attr, 0)
                scores = [getattr(f, 'det_score', 0.5) for f in valid_faces if hasattr(f, 'det_score')]
                if scores:
                    frame_avg = sum(scores) / len(scores)
                    score_attr = f'_avg_det_score_{vkey_q}'
                    prev_avg = getattr(self, score_attr, 0.5)
                    new_avg = 0.92 * prev_avg + 0.08 * frame_avg  # EMA suave
                    setattr(self, score_attr, new_avg)

            # Calidad/identidad: globals + _process_face_swap_v21 (enhancer_blend_factor, GFPGAN, etc.)
            face_swap_mode = getattr(self.options, 'face_swap_mode', 'selected')

            faces_to_process = []
            
            if is_real_video:
                self._is_single_photo = False  # v5.1026: modo video
                video_basename = os.path.basename(video_path)
                frame_count_pre = getattr(self, f'_frame_count_{video_basename}', 0) + 1
                self._update_closeup_state(valid_faces, frame.shape, video_basename)
                plan_smooth_until = getattr(self, f'_plan_change_smooth_until_{video_basename}', 0)
                is_visual_cut, cut_diff = self._detect_visual_scene_cut(
                    frame, video_basename, frame_count_pre,
                )
                zoom_in = self._is_zoom_in_progress(valid_faces, video_basename)
                if is_visual_cut and (zoom_in or frame_count_pre <= plan_smooth_until):
                    setattr(
                        self, f'_plan_change_smooth_until_{video_basename}',
                        max(plan_smooth_until, frame_count_pre + 24),
                    )
                    if frame_count_pre % 35 == 1:
                        print(
                            f"[ZOOM_SKIP] Frame {frame_count_pre}: diff={cut_diff:.1f} "
                            "— zoom/primer plano, no scene cut"
                        )
                elif is_visual_cut:
                    self._reset_tracking_on_scene_cut(
                        video_path, frame_count_pre,
                        reason=f"cambio visual abrupto (diff={cut_diff:.1f})",
                    )

                plan_smooth_until = getattr(self, f'_plan_change_smooth_until_{video_basename}', 0)
                if frame_count_pre <= plan_smooth_until and frame_count_pre % 30 == 1:
                    print(
                        f"[PLAN_CHANGE] Frame {frame_count_pre}: zoom suave activo "
                        f"(hasta frame {plan_smooth_until})"
                    )
                if self._detect_closeup_plan_change(
                    valid_faces, frame.shape, video_basename, frame_count_pre,
                ):
                    self._open_reacquire_window(video_basename, frame_count_pre, frames=60)
                    print(
                        f"[PLAN_CHANGE] Frame {frame_count_pre}: primer plano detectado "
                        "— re-busqueda activa"
                    )

                if not hasattr(self, f'_reference_face_selected_{video_basename}'):
                    self._setup_selected_faces_frame_for_video(video_path, valid_faces)

                recovered = self._try_unassigned_reacquire(
                    valid_faces, video_basename, frame_count_pre, frame.shape,
                )
                if recovered is not None:
                    self._assign_recovered_face(recovered, video_basename)
                
                target_face = self._find_target_face_for_selected_mode(video_path, valid_faces, frame)
                if target_face:
                    faces_to_process = [target_face]
            else:
                # MODO FOTO: Respetar si el usuario seleccionó una cara específica en la UI
                self._is_single_photo = True  # v5.1026: flag para maximizar identidad source en fotos
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
                        if best_match and max_sim > 0.35:
                            faces_to_process = [best_match]

                # v5.87: Sin fallback — si no hay match fiable, hold o frame original

            if not faces_to_process:
                if is_real_video and video_path:
                    video_basename = os.path.basename(video_path)
                    if self._should_hold_previous_swap(video_basename) or (
                        getattr(self, f'_closeup_active_{video_basename}', False)
                        and self._has_swap_history(video_basename)
                    ):
                        prev_full = getattr(self, '_prev_frame_result', {}).get(self._current_video_key)
                        last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                        if prev_full is not None and last_bbox is not None:
                            blend = 0.95 if getattr(self, f'_closeup_active_{video_basename}', False) else 0.88
                            return self._hold_previous_face_region(
                                frame, prev_full, last_bbox, blend=blend, prev_mask=None,
                            )
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
                    is_target_profile = ProcessMgr._is_profile_face(getattr(target_face, 'kps', None))
                    if is_real_video and video_path:
                        lock_key, profile_lock_key = self._video_source_lock_keys(video_path)
                        if is_target_profile:
                            profile_locked = getattr(self, profile_lock_key, None)
                            if profile_locked is not None:
                                source_face = profile_locked
                        else:
                            locked_face = getattr(self, lock_key, None)
                            if locked_face is not None:
                                source_face = locked_face

                    if source_face is None:
                        source_face = self._select_source_face(
                            target_face, all_faces, face_swap_mode, video_path,
                            is_profile=is_target_profile,
                        )
                        if source_face is not None and is_real_video and video_path:
                            lock_key, profile_lock_key = self._video_source_lock_keys(video_path)
                            sim_score = self._calculate_similarity(
                                getattr(source_face, 'embedding', None),
                                getattr(target_face, 'embedding', None),
                            ) if hasattr(source_face, 'embedding') else 0
                            if is_target_profile:
                                setattr(self, profile_lock_key, source_face)
                                print(
                                    f"[MATCH] Perfil bloqueado | similitud={sim_score:.3f} "
                                    f"(identidad vía master+profile embedding)"
                                )
                            elif getattr(self, lock_key, None) is None:
                                setattr(self, lock_key, source_face)
                                tag = "alta" if sim_score >= 0.22 else "baja-sim"
                                print(
                                    f"[MATCH] Source bloqueada ({tag}) | "
                                    f"similitud={sim_score:.3f} — estable hasta scene cut"
                                )

                    if source_face is None:
                        if is_real_video and video_path:
                            video_basename = os.path.basename(video_path)
                            if self._should_hold_previous_swap(video_basename):
                                prev_full = getattr(self, '_prev_frame_result', {}).get(self._current_video_key)
                                last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                                if prev_full is not None and last_bbox is not None:
                                    return self._hold_previous_face_region(frame, prev_full, last_bbox, prev_mask=None)
                        continue

                    # PROCESAR SWAP CON CALIDAD FORZADA
                    result_frame = self._process_face_swap_v21(
                        source_face, target_face, result_frame, frame, enable_temporal_smoothing
                    )
                except Exception as e:
                    print(f"Error procesando cara {i+1}: {e}")
                    continue

            return result_frame
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
                if ProcessMgr._is_profile_face(kps):
                    score += 0.35

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

    def _find_best_source_embedding_for_target(self, target_emb, profile_only=False, min_sim=0.10):
        match = self._find_best_source_entry_for_target(
            target_emb, profile_only=profile_only, min_sim=min_sim,
        )
        return match['emb'] if match else None

    def _find_best_source_entry_for_target(self, target_emb, profile_only=False, min_sim=0.10):
        if target_emb is None:
            return None
        entries = getattr(self, '_source_face_entries', [])
        if not entries:
            return None
        target_arr = np.array(target_emb, dtype=np.float32).flatten()
        norm = np.linalg.norm(target_arr)
        if norm <= 0:
            return None
        target_arr = target_arr / norm
        best_entry = None
        best_sim = -1.0
        for entry in entries:
            if profile_only and not entry.get('is_profile'):
                continue
            sim = float(np.dot(target_arr, entry['emb']))
            if sim > best_sim:
                best_sim = sim
                best_entry = entry
        if best_entry is not None and best_sim >= min_sim:
            best_entry = dict(best_entry)
            best_entry['sim'] = best_sim
            return best_entry
        return None

    @staticmethod
    def _video_source_lock_keys(video_path: str):
        basename = os.path.basename(video_path)
        return (
            f'_video_locked_source_face_{basename}',
            f'_video_locked_profile_source_{basename}',
        )

    def _clear_video_source_locks(self, video_path: str):
        if not video_path:
            return
        lock_key, profile_lock_key = self._video_source_lock_keys(video_path)
        for key in (lock_key, profile_lock_key):
            if hasattr(self, key):
                delattr(self, key)

    def _get_user_reference_embedding(self, video_basename: str):
        video_key = f"selected_face_ref_{video_basename}"
        if hasattr(roop.globals, 'selected_face_references'):
            ref = roop.globals.selected_face_references.get(video_key)
            if ref and ref.get('embedding') is not None:
                emb = np.array(ref['embedding'], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    return emb / norm
        orig = getattr(self, f'_original_embedding_{video_basename}', None)
        if orig is not None:
            emb = np.array(orig, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                return emb / norm
        return getattr(self, f'_tracking_embedding_{video_basename}', None)

    def _detect_visual_scene_cut(self, frame, video_basename: str, frame_count: int = 0):
        if frame is None:
            return False, 0.0
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thumb = cv2.resize(gray, (64, 36), interpolation=cv2.INTER_AREA)
            attr = f'_prev_scene_thumb_{video_basename}'
            prev = getattr(self, attr, None)
            setattr(self, attr, thumb)
            if prev is None or prev.shape != thumb.shape:
                return False, 0.0
            diff = float(np.mean(np.abs(thumb.astype(np.float32) - prev.astype(np.float32))))

            hist_attr = f'_scene_diff_history_{video_basename}'
            history = getattr(self, hist_attr, [])
            history.append(diff)
            if len(history) > 12:
                history.pop(0)
            setattr(self, hist_attr, history)

            last_visual_cut = getattr(self, f'_last_visual_cut_frame_{video_basename}', 0)
            if frame_count > 0 and (frame_count - last_visual_cut) < 20:
                return False, diff

            baseline = (
                sum(history[:-1]) / max(1, len(history) - 1)
                if len(history) > 1 else diff
            )
            is_cut = diff > 48.0 or (
                diff > max(38.0, baseline * 2.8) and diff > 28.0
            )
            if is_cut and frame_count > 0:
                setattr(self, f'_last_visual_cut_frame_{video_basename}', frame_count)
            return is_cut, diff
        except Exception:
            return False, 0.0

    def _assign_recovered_face(self, face, video_basename: str):
        setattr(self, f'_target_face_assigned_{video_basename}', face)
        setattr(self, f'_tracking_lost_count_{video_basename}', 0)
        setattr(self, f'_consecutive_tracking_success_{video_basename}', 0)
        center = get_face_center(face)
        if center:
            setattr(self, f'_position_history_{video_basename}', [center])
        for attr in (
            f'_unassigned_since_{video_basename}',
            f'_unassigned_reacquire_tick_{video_basename}',
        ):
            if hasattr(self, attr):
                delattr(self, attr)
        return face

    def _mark_unassigned(self, video_basename: str, frame_count: int):
        if not getattr(self, f'_unassigned_since_{video_basename}', 0):
            setattr(self, f'_unassigned_since_{video_basename}', frame_count)

    def _open_reacquire_window(self, video_basename: str, frame_count: int, frames: int = 120):
        setattr(self, f'_reacquire_window_until_{video_basename}', frame_count + frames)
        self._mark_unassigned(video_basename, frame_count)

    def _largest_face_area(self, face) -> float:
        if not hasattr(face, 'bbox') or face.bbox is None:
            return 0.0
        x1, y1, x2, y2 = face.bbox
        return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

    def _update_closeup_state(self, valid_faces, frame_shape, video_basename: str):
        active = False
        if valid_faces and frame_shape is not None:
            largest = max(valid_faces, key=self._largest_face_area)
            active = self._is_closeup_face(largest.bbox, frame_shape)
        setattr(self, f'_closeup_active_{video_basename}', active)
        return active

    def _is_zoom_in_progress(self, valid_faces, video_basename: str) -> bool:
        if not valid_faces:
            return False
        largest = max(valid_faces, key=self._largest_face_area)
        area = self._largest_face_area(largest)
        if area <= 0:
            return False
        prev_attr = f'_prev_largest_face_area_{video_basename}'
        prev_area = getattr(self, prev_attr, 0.0)
        setattr(self, prev_attr, area)
        if prev_area > 0 and area / prev_area >= 1.08:
            return True
        hist = getattr(self, f'_face_area_history_{video_basename}', [])
        if len(hist) >= 2 and hist[-2] > 0 and area / hist[-2] >= 1.06:
            return True
        return False

    def _has_swap_history(self, video_basename: str) -> bool:
        video_key = f"video_{video_basename}"
        prev_full = getattr(self, '_prev_frame_result', {}).get(video_key)
        last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
        return prev_full is not None and last_bbox is not None

    def _detect_closeup_plan_change(self, valid_faces, frame_shape, video_basename: str, frame_count: int):
        if not valid_faces or frame_shape is None:
            return False
        largest = max(valid_faces, key=self._largest_face_area)
        area = self._largest_face_area(largest)
        if area <= 0 or not self._is_closeup_face(largest.bbox, frame_shape):
            return False

        hist_attr = f'_face_area_history_{video_basename}'
        history = getattr(self, hist_attr, [])
        history.append(area)
        if len(history) > 10:
            history.pop(0)
        setattr(self, hist_attr, history)

        if len(history) < 4:
            return False
        baseline = sum(history[:-1]) / max(1, len(history) - 1)
        if baseline <= 0:
            return False
        growth = area / baseline
        # Zoom gradual en 2+ frames (evita salto brusco al entrar en primer plano)
        recent_growth = 1.0
        if len(history) >= 3:
            prev_area = history[-2]
            if prev_area > 0:
                recent_growth = area / prev_area
        assigned = getattr(self, f'_target_face_assigned_{video_basename}', None)
        lost_count = getattr(self, f'_tracking_lost_count_{video_basename}', 0)
        if assigned is not None and lost_count == 0:
            # Tracking estable: solo marcar suavizado de tamaño, sin re-busqueda
            if growth >= 1.45 or recent_growth >= 1.25:
                setattr(self, f'_plan_change_smooth_until_{video_basename}', frame_count + 18)
            return False
        if growth >= 2.6 and recent_growth >= 1.18:
            return True
        return False

    def _reset_tracking_on_scene_cut(self, video_path, frame_count, reason="", soft=False):
        video_basename = os.path.basename(video_path) if video_path else ""
        if not video_basename:
            return
        video_key = f"video_{video_basename}"
        closeup_active = getattr(self, f'_closeup_active_{video_basename}', False)
        if closeup_active:
            soft = True

        setattr(self, f'_target_face_assigned_{video_basename}', None)
        setattr(self, f'_tracking_lost_count_{video_basename}', 0)
        setattr(self, f'_position_history_{video_basename}', [])
        setattr(self, f'_consecutive_tracking_success_{video_basename}', 0)
        setattr(self, f'_strict_recovery_tick_{video_basename}', 0)
        if not soft:
            self._mark_unassigned(video_basename, frame_count)
            setattr(self, f'_hard_scene_cut_frame_{video_basename}', frame_count)

        if not soft:
            if hasattr(self, '_prev_frame_result') and isinstance(self._prev_frame_result, dict):
                self._prev_frame_result.pop(video_key, None)
            if hasattr(self, '_last_swap_bbox') and isinstance(self._last_swap_bbox, dict):
                self._last_swap_bbox.pop(video_basename, None)
            if hasattr(self, '_last_swap_mask') and isinstance(self._last_swap_mask, dict):
                self._last_swap_mask.pop(video_basename, None)
        if hasattr(self, '_prev_face_bbox') and isinstance(self._prev_face_bbox, dict):
            self._prev_face_bbox.pop(video_key, None)
        if hasattr(self, '_prev_face_kps') and isinstance(self._prev_face_kps, dict):
            self._prev_face_kps.pop(video_key, None)
        for attr in (f'_last_known_bbox_{video_basename}', f'_last_known_kps_{video_basename}'):
            if hasattr(self, attr):
                delattr(self, attr)

        if not soft:
            self._clear_video_source_locks(video_path)

        scene_attr = f'_scene_state_{video_basename}'
        scene = getattr(self, scene_attr, None)
        if not isinstance(scene, dict):
            scene = {'last_scene_cut': frame_count, 'total_scene_changes': 1}
        else:
            scene['last_scene_cut'] = frame_count
            scene['total_scene_changes'] = scene.get('total_scene_changes', 0) + 1
        setattr(self, scene_attr, scene)
        suppress_frames = 2 if soft else 6
        setattr(self, f'_hold_suppressed_until_{video_basename}', frame_count + suppress_frames)
        if not soft:
            self._open_reacquire_window(video_basename, frame_count, frames=60)
        elif closeup_active:
            setattr(self, f'_plan_change_smooth_until_{video_basename}', frame_count + 24)

        if reason:
            tag = "soft, mantiene swap" if soft else "reset tracking, sin hold"
            print(f"[SCENE_CUT] Frame {frame_count}: {reason} — {tag}")

    def _should_hold_previous_swap(self, video_basename: str) -> bool:
        frame_count = getattr(self, f'_frame_count_{video_basename}', 0)
        # v5.1011: Tras scene cut duro, no hacer hold por 15 frames — evita swap congelado
        hard_cut_frame = getattr(self, f'_hard_scene_cut_frame_{video_basename}', 0)
        if hard_cut_frame > 0 and abs(frame_count - hard_cut_frame) < 15:
            return False
        closeup_active = getattr(self, f'_closeup_active_{video_basename}', False)
        lost_count = getattr(self, f'_tracking_lost_count_{video_basename}', 0)
        has_history = self._has_swap_history(video_basename)

        # Primer plano: mantener swap visible salvo pérdida muy prolongada
        if closeup_active and has_history and lost_count < 20:
            return True

        if frame_count <= getattr(self, f'_hold_suppressed_until_{video_basename}', 0):
            if closeup_active and has_history and lost_count < 8:
                return True
            return False
        scene = getattr(self, f'_scene_state_{video_basename}', {})
        if isinstance(scene, dict):
            last_cut = scene.get('last_scene_cut', 0)
            ban_frames = 3 if closeup_active else 8
            if last_cut > 0 and abs(frame_count - last_cut) < ban_frames:
                if closeup_active and has_history:
                    return True
                return False
        lost_limit = 14 if closeup_active else 6
        if lost_count >= lost_limit:
            return False
        return True

    def _reacquire_face_global(
        self, valid_faces, video_basename, frame_count,
        min_sim=0.30, after_scene_cut=False, closeup_mode=False,
    ):
        ref_emb = self._get_user_reference_embedding(video_basename)
        if ref_emb is None or not valid_faces:
            return None
        multi_face = len(valid_faces) >= 2
        if closeup_mode:
            min_sim = min(min_sim, 0.25 if multi_face else 0.22)  # v5.1034: closeup BAJA el umbral (era max, causaba rechazo en primer plano)
        elif after_scene_cut:
            min_sim = min(min_sim, 0.20 if multi_face else 0.16)  # v5.1024: más permisivo tras scene cut — acepta sim≥0.16
        else:
            min_sim = max(min_sim, 0.32 if multi_face else 0.28)

        # v5.1005: Use tracking_embedding (up-to-date) for better recovery
        tracking_emb = getattr(self, f'_tracking_embedding_{video_basename}', None)

        # v5.1005: Progressive threshold lowering after consecutive re-search failures.
        # v5.1022: También para multi-face (evita swap congelado en videos con varias caras).
        re_fail_attr = f'_reacquire_fail_count_{video_basename}'
        re_fails = getattr(self, re_fail_attr, 0)
        if re_fails >= 20:
            min_sim = max(0.02 if not multi_face else 0.05, min_sim - 0.36)  # v5.1029: multi_face floor 0.12→0.05
        elif re_fails >= 10:
            min_sim = max(0.12 if not multi_face else 0.20, min_sim - 0.16)
        elif re_fails >= 5:
            min_sim = max(0.18 if not multi_face else 0.24, min_sim - 0.08)

        # Get last known position for spatial proximity check
        position_history = getattr(self, f'_position_history_{video_basename}', [])
        last_center = position_history[-1] if position_history else None

        candidates = []
        for face in valid_faces:
            if not hasattr(face, 'embedding') or face.embedding is None:
                continue
            sim = float(self._calculate_similarity(ref_emb, face.embedding))

            # v5.1005: Use tracking_emb if it gives better match
            if tracking_emb is not None:
                track_sim = float(self._calculate_similarity(tracking_emb, face.embedding))
                sim = max(sim, track_sim)

            # v5.1005: Single-face spatial override — if only 1 face in frame
            # and near last known position, accept with low sim threshold.
            # Spatial proximity + single face + sane det_score + area consistency
            # make false positives extremely unlikely even at threshold 0.02.
            accept = sim >= min_sim - 0.005  # v5.1027: epsilon para evitar falsos negativos por precisión flotante
            if not accept and not multi_face and last_center is not None:
                center = get_face_center(face)
                if center:
                    dist = np.sqrt(
                        (center[0] - last_center[0])**2 +
                        (center[1] - last_center[1])**2
                    )
                    face_w = max(1.0, face.bbox[2] - face.bbox[0]) if (hasattr(face, 'bbox') and face.bbox is not None) else 100
                    if dist < max(300, face_w * 2.0):
                        det = getattr(face, 'det_score', 0.5)
                        if det > 0.10:
                            accept = sim >= max(0.02, min_sim - 0.20)
                            # v5.1027: Fix last_bbox_val undefined → usar _last_swap_bbox
                            last_bbox_ref = self._last_swap_bbox.get(video_basename) if hasattr(self, '_last_swap_bbox') else None
                            if not accept and last_bbox_ref is not None:
                                lw = max(1, last_bbox_ref[2] - last_bbox_ref[0])
                                lh = max(1, last_bbox_ref[3] - last_bbox_ref[1])
                                last_area = lw * lh
                                fw = face.bbox[2] - face.bbox[0] if (hasattr(face, 'bbox') and face.bbox is not None) else 1
                                fh = face.bbox[3] - face.bbox[1] if (hasattr(face, 'bbox') and face.bbox is not None) else 1
                                area_ratio = (fw * fh) / max(1, last_area)
                                if 0.18 <= area_ratio <= 6.0:
                                    accept = sim >= max(0.02, min_sim - 0.30)

            if accept:
                area = self._largest_face_area(face)
                candidates.append((sim, area, face))

        if not candidates:
            setattr(self, re_fail_attr, re_fails + 1)
            unassigned_since = getattr(self, f'_unassigned_since_{video_basename}', 0)
            should_log = (
                after_scene_cut or closeup_mode
                or (unassigned_since > 0 and frame_count - unassigned_since > 20)
            )
            if should_log and frame_count % 20 == 1:
                best_any = max(
                    (
                        float(self._calculate_similarity(ref_emb, f.embedding))
                        for f in valid_faces
                        if hasattr(f, 'embedding') and f.embedding is not None
                    ),
                    default=-1.0,
                )
                track_sim_str = ""
                if tracking_emb is not None:
                    track_best = max(
                        (
                            float(self._calculate_similarity(tracking_emb, f.embedding))
                            for f in valid_faces
                            if hasattr(f, 'embedding') and f.embedding is not None
                        ),
                        default=-1.0,
                    )
                    track_sim_str = f", track_sim={track_best:.2f}"
                spatial_info = ""
                if not multi_face and last_center is not None:
                    center_f = get_face_center(valid_faces[0])
                    if center_f:
                        d = np.sqrt((center_f[0] - last_center[0])**2 + (center_f[1] - last_center[1])**2)
                        spatial_info = f", dist={d:.0f}px"
                print(
                    f"[TRACK] RE-BUSQUEDA sin match frame {frame_count} "
                    f"(mejor_sim={best_any:.2f}{track_sim_str}, "
                    f"umbral={min_sim:.2f}, caras={len(valid_faces)}, "
                    f"re_fails={re_fails}{spatial_info})"
                )
            # v5.1027: After scene cut — fallback accept con tracking_emb mínimo 0.10
            # para evitar frames perdidos cuando el position_history se reseteó y
            # ref_emb da baja similitud (ej. frame 861 con ref=0.04 track=0.16).
            if after_scene_cut and tracking_emb is not None:
                for face in valid_faces:
                    if not hasattr(face, 'embedding') or face.embedding is None:
                        continue
                    track_sim = float(self._calculate_similarity(tracking_emb, face.embedding))
                    if track_sim >= 0.10:
                        area = self._largest_face_area(face)
                        candidates.append((track_sim, area, face))
            if not candidates and re_fails >= 10:
                # v5.1029: Fallback extremo: aceptar la mejor cara disponible aunque sea sim baja
                best_fallback = None
                best_fb_sim = -1.0
                for face in valid_faces:
                    if not hasattr(face, 'embedding') or face.embedding is None:
                        continue
                    fb_sim = float(self._calculate_similarity(ref_emb, face.embedding))
                    if fb_sim > best_fb_sim:
                        best_fb_sim = fb_sim
                        best_fallback = face
                if best_fallback is not None and best_fb_sim >= 0.03:
                    area = self._largest_face_area(best_fallback)
                    candidates.append((best_fb_sim, area, best_fallback))
            if not candidates:
                return None

        if closeup_mode:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        else:
            candidates.sort(key=lambda item: item[0], reverse=True)
        best_sim, _best_area, best_face = candidates[0]
        if multi_face and len(candidates) >= 2:
            margin = best_sim - candidates[1][0]
            need_margin = 0.04
            if margin < need_margin:
                # v5.1003: Relajado: intentar con la de mayor área si sim es aceptable
                if best_sim >= min_sim:
                    best_face = max(candidates, key=lambda item: item[1])[2]
                    best_sim = max(c[0] for c in candidates)
                else:
                    if frame_count % 25 == 1:
                        print(
                            f"[TRACK] RE-BUSQUEDA ambigua frame {frame_count} "
                            f"({best_sim:.2f} vs {candidates[1][0]:.2f})"
                        )
                    return None

        # v5.1005: Reset re-search fail counter on success
        setattr(self, re_fail_attr, 0)

        tag = "closeup" if closeup_mode else "global"
        re_search_mode = "spatial" if best_sim < min_sim else "emb"
        print(f"[TRACK] RE-BUSQUEDA {tag} frame {frame_count} (sim={best_sim:.2f}, mode={re_search_mode})")
        return best_face

    def _in_reacquire_window(self, video_basename: str, frame_count: int) -> bool:
        return frame_count <= getattr(self, f'_reacquire_window_until_{video_basename}', 0)

    def _try_unassigned_reacquire(self, valid_faces, video_basename, frame_count, frame_shape=None):
        if not valid_faces:
            return None
        if getattr(self, f'_target_face_assigned_{video_basename}', None) is not None:
            return None

        in_window = self._in_reacquire_window(video_basename, frame_count)
        unassigned_since = getattr(self, f'_unassigned_since_{video_basename}', 0)
        long_unassigned = unassigned_since > 0 and (frame_count - unassigned_since) >= 20

        has_closeup = False
        if frame_shape is not None:
            for face in valid_faces:
                if hasattr(face, 'bbox') and self._is_closeup_face(face.bbox, frame_shape):
                    has_closeup = True
                    break

        tick_attr = f'_unassigned_reacquire_tick_{video_basename}'
        tick = getattr(self, tick_attr, 0) + 1
        setattr(self, tick_attr, tick)

        if not (in_window or has_closeup or long_unassigned or tick % 4 == 0):
            return None

        return self._reacquire_face_global(
            valid_faces, video_basename, frame_count,
            after_scene_cut=in_window or long_unassigned,
            closeup_mode=has_closeup,
        )

    def _resolve_swap_embedding(self, source_face, target_face, is_profile=False, video_basename=None):
        """Embedding para inswapper: master frontal + refuerzo perfil/match en ángulos difíciles."""
        if not hasattr(self, 'master_source_embedding') or self.master_source_embedding is None:
            return source_face

        # SOURCE RANDOM: usar embedding real de la cara seleccionada, no master
        if getattr(roop.globals, 'source_random', False):
            return source_face

        target_emb = getattr(target_face, 'embedding', None)
        master = self.master_source_embedding
        profile_master = getattr(self, 'profile_master_embedding', None)

        if is_profile:
            match_entry = self._find_best_source_entry_for_target(
                target_emb, profile_only=True, min_sim=0.08,
            )
            if match_entry is None:
                match_entry = self._find_best_source_entry_for_target(
                    target_emb, profile_only=False, min_sim=0.10,
                )
            match_emb = match_entry['emb'] if match_entry else None
            match_sim = float(match_entry.get('sim', 0.0)) if match_entry else 0.0
            setattr(self, '_last_profile_match_sim', match_sim)  # v5.1033: guardar para ajuste de m_blend
            # v5.1024: Cuando match_sim < 0.20, el embedding de perfil es demasiado ruidoso
            # y diluye la identidad. Usamos 100% master (frontal de máxima calidad).
            if match_emb is not None and match_sim >= 0.20 and profile_master is not None:
                mixed = 0.25 * master + 0.30 * profile_master + 0.45 * match_emb
            elif match_emb is not None and match_sim >= 0.16:
                if profile_master is not None:
                    mixed = 0.40 * master + 0.30 * profile_master + 0.30 * match_emb
                else:
                    mixed = 0.55 * master + 0.45 * match_emb
            elif profile_master is not None:
                mixed = 0.50 * master + 0.50 * profile_master
            else:
                mixed = master.copy()

            # v5.1009: EMA profile embedding to prevent identity flicker from source match switching
            if video_basename:
                if not hasattr(self, '_profile_emb_ema'):
                    self._profile_emb_ema = {}
                prev_emb = self._profile_emb_ema.get(video_basename)
                if prev_emb is not None and prev_emb.shape == mixed.shape:
                    mixed = 0.60 * prev_emb + 0.40 * mixed  # v5.1028: 85→60% prev para adaptación más rápida de identidad en perfil
                self._profile_emb_ema[video_basename] = mixed.copy()
        else:
            mixed = master.copy()

        norm = np.linalg.norm(mixed)
        if norm > 0:
            mixed = mixed / norm

        class _SourceFaceWrap:
            embedding = mixed.tolist()

        return _SourceFaceWrap()

    def _select_source_face_by_similarity(self, target_face, candidate_faces, is_profile=False):
        """
        Selecciona la cara origen que mejor coincide con el target usando embedding similarity.
        Para modo 'selected_faces_frame': hace matching real entre target y source.
        """
        try:
            if not candidate_faces:
                return None
            
            target_emb = getattr(target_face, 'embedding', None)
            if target_emb is None:
                print("[SELECT_SOURCE] Target sin embedding — sin fallback a otra source")
                return None
            
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
                    'idx': idx,
                    'is_profile': ProcessMgr._is_profile_face(getattr(src_face, 'kps', None)),
                })
            
            if not candidates:
                print("[SELECT_SOURCE] No source con embedding — sin fallback a otra source")
                return None

            max_sim = max(c['sim'] for c in candidates)
            critical_sim = max_sim < 0.15

            if critical_sim:
                print(
                    f"[SELECT_SOURCE] Similitud crítica ({max_sim:.4f}). "
                    "master+profile embedding; eligiendo mejor sim en pool perfil."
                )

            sim_weight = 25.0 if is_profile else 10.0

            if is_profile or max_sim <= 0.22:
                profile_cands = [c for c in candidates if c['is_profile']]
                pool = profile_cands if profile_cands and (is_profile or max_sim < 0.18) else candidates
                if critical_sim:
                    best_candidate = max(pool, key=lambda c: c['sim'])
                    tag = "perfil-crítico" if is_profile else "baja-sim-crítico"
                    print(
                        f"[SELECT_SOURCE] Modo {tag} (max_sim={max_sim:.4f}). "
                        f"#{best_candidate['idx'] + 1} (sim={best_candidate['sim']:.4f}, qual={best_candidate['quality']:.2f})"
                    )
                    return best_candidate['face']
                threshold = max(0.10, max_sim * 0.65) if is_profile else max(0.12, max_sim * 0.70)
                identity_candidates = [c for c in pool if c['sim'] >= threshold] or pool
                if is_profile and max_sim < 0.22:
                    best_candidate = max(
                        identity_candidates,
                        key=lambda c: (c['sim'] * sim_weight) + (c['quality'] * 0.35),
                    )
                elif not is_profile and max_sim < 0.20:
                    best_candidate = max(
                        identity_candidates,
                        key=lambda c: (c['sim'] * 18.0) + (c['quality'] * 0.30),
                    )
                else:
                    best_candidate = max(
                        identity_candidates,
                        key=lambda c: c['quality'] + (c['sim'] * sim_weight),
                    )
                tag = "perfil" if is_profile else "baja-sim"
                print(
                    f"[SELECT_SOURCE] Modo {tag} (max_sim={max_sim:.4f}). "
                    f"#{best_candidate['idx'] + 1} (qual={best_candidate['quality']:.2f}, "
                    f"sim={best_candidate['sim']:.4f}, w={sim_weight:.0f})"
                )
                return best_candidate['face']

            if max_sim > 0.22:
                threshold = max(0.15, max_sim * 0.80)
                identity_candidates = [c for c in candidates if c['sim'] >= threshold]
                best_candidate = max(identity_candidates, key=lambda c: c['quality'])
                print(f"[SELECT_SOURCE] Identidad detectada (max_sim={max_sim:.4f}). Seleccionada muestra de mejor calidad: #{best_candidate['idx'] + 1} (qual={best_candidate['quality']:.2f}, sim={best_candidate['sim']:.4f})")
                return best_candidate['face']

            best_candidate = max(candidates, key=lambda c: c['quality'] + (c['sim'] * sim_weight))
            print(f"[SELECT_SOURCE] Similitud baja ({max_sim:.4f}), compromiso calidad/identidad (weight={sim_weight:.0f}): #{best_candidate['idx'] + 1}")
            return best_candidate['face']
            
        except Exception as e:
            print(f"[ERROR] _select_source_face_by_similarity: {e}")
            import traceback; traceback.print_exc()
            return None

    def _select_source_face(self, target_face, candidate_faces, face_swap_mode, video_path=None, is_profile=False):
        """
        Lógica de selección de cara origen:
        - Prioriza el matching por similitud para encontrar la mejor cara origen para el destino.
        - Usa el índice seleccionado como fallback si el matching no es posible.
        - Si source_random está activo, elige aleatoriamente entre todas las caras origen.
        """
        try:
            if not candidate_faces:
                return None

            # 0. SOURCE ALEATORIO: elige una cara al azar de todas las disponibles
            if getattr(roop.globals, 'source_random', False):
                import random
                idx = random.randint(0, len(candidate_faces) - 1)
                picked = candidate_faces[idx]
                print(f"[SOURCE_RANDOM] Cara #{idx+1} de {len(candidate_faces)} seleccionada")
                return picked
            
            # Solo matching por similitud + lock (sin fallback a otra source)
            if face_swap_mode in ['selected', 'selected_faces', 'auto', 'selected_faces_frame']:
                return self._select_source_face_by_similarity(
                    target_face, candidate_faces, is_profile=is_profile,
                )
            return None

        except Exception as e:
            print(f"[ERROR] _select_source_face: {e}")
            return None

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

            if not user_has_selected_face:
                print(
                    f"[SELECTED_FACES_FRAME] Sin cara seleccionada para {video_basename} "
                    "— sin fallback auto (hold hasta selección o match fuerte)"
                )

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

                    if best_match and best_score > 0.35:
                        selected_face = best_match
                        selection_method = f"usuario (score={best_score:.2f})"
                    elif best_score > 0.15:
                        print(
                            f"[SELECTED_FACES_FRAME] Match débil ({best_score:.2f}) "
                            "— sin fallback, hold hasta mejor detección"
                        )
                    else:
                        print(
                            f"[SELECTED_FACES_FRAME] Sin match fiable ({best_score:.2f}) "
                            "— sin fallback a otra cara"
                        )

            if selected_face is None:
                setattr(self, f'_reference_face_selected_{video_basename}', True)
                return

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
        
        # SCENE CUT: Detectar cambio abrupto en número de caras
        scene_cut_detected = False
        if len(scene['face_count_history']) >= 3:
            prev_2 = scene['face_count_history'][-3:-1]
            curr_count = scene['face_count_history'][-1]
            if len(prev_2) == 2 and prev_2[0] > 0 and prev_2[1] > 0:
                avg_prev = (prev_2[0] + prev_2[1]) / 2.0
                if curr_count == 0 and avg_prev >= 1.5:
                    scene_cut_detected = True
                    print(f"[SCENE_CUT] Face count drop: {avg_prev:.0f} → 0 at frame {frame_count}")
                elif curr_count >= 2 and avg_prev <= 0.5:
                    scene_cut_detected = True
                    print(f"[SCENE_CUT] Face count surge: {avg_prev:.0f} → {curr_count} at frame {frame_count}")

        if scene_cut_detected:
            scene['last_scene_cut'] = frame_count
            scene['total_scene_changes'] += 1
            scene['_pending_reset'] = True
        
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

    def _identity_margin(self, valid_faces, ref_embedding):
        sims = []
        for face in valid_faces or []:
            if not hasattr(face, 'embedding') or face.embedding is None:
                continue
            sims.append(float(self._calculate_similarity(ref_embedding, face.embedding)))
        sims.sort(reverse=True)
        if not sims:
            return 0.0, 0.0
        top = sims[0]
        second = sims[1] if len(sims) > 1 else 0.0
        return top, top - second

    def _validate_locked_identity(self, valid_faces, ref_embedding, candidate_face, multi_face=False, fallback_embedding=None, closeup_recovery=False):
        if ref_embedding is None or candidate_face is None:
            return True, 1.0
        cand_sim = float(self._calculate_similarity(
            ref_embedding, getattr(candidate_face, 'embedding', None),
        ))
        # v5.1005: Try fallback_embedding (e.g. tracking_embedding) for better match
        if fallback_embedding is not None and cand_sim < 0.40:
            fb_sim = float(self._calculate_similarity(
                fallback_embedding, getattr(candidate_face, 'embedding', None),
            ))
            if fb_sim > cand_sim:
                cand_sim = fb_sim
                ref_embedding = fallback_embedding
        top_sim, margin = self._identity_margin(valid_faces, ref_embedding)
        # v5.1033: recovery mode (lost_count>0) relaja umbrales para evitar oscilación RE-BUSQUEDA
        recovery_mode = closeup_recovery or (hasattr(self, '_tracking_recovery') and getattr(self, '_tracking_recovery') > 0)
        if recovery_mode and not multi_face:
            min_sim = 0.28
            min_margin = 0.03
        elif recovery_mode and multi_face:
            min_sim = 0.35
            min_margin = 0.03
        else:
            min_sim = 0.45 if multi_face else 0.40
            min_margin = 0.05 if multi_face else 0.05
        ok = (
            cand_sim >= min_sim
            and cand_sim >= top_sim - 0.05
            and margin >= min_margin
        )
        # v5.1022b: Reintentar más laxo para multi-face con similitudes cercanas
        if not ok and multi_face:
            ok = (
                cand_sim >= min_sim - 0.05
                and cand_sim >= top_sim - 0.08
                and margin >= 0.03
            )
        return ok, margin

    def _try_strict_same_person_reacquire(
        self,
        valid_faces,
        ref_embedding,
        spatial_anchor,
        video_basename,
        frame_count,
        min_sim=0.50,
    ):
        """Re-enganche solo misma persona: sim alta + proximidad al último bbox conocido."""
        if ref_embedding is None or not valid_faces or spatial_anchor is None:
            return None
        anchor_bbox = getattr(spatial_anchor, 'bbox', None)
        if anchor_bbox is None:
            return None

        multi_face = len(valid_faces) >= 2

        anchor_center = get_face_center(spatial_anchor)
        face_w = max(1.0, float(anchor_bbox[2] - anchor_bbox[0]))
        max_center_dist = max(220.0, face_w * 1.2) if multi_face else max(250.0, face_w * 1.5)

        candidates = []
        for face in valid_faces:
            if not hasattr(face, 'embedding') or face.embedding is None:
                continue
            sim = float(self._calculate_similarity(ref_embedding, face.embedding))
            if sim < min_sim:
                continue
            center = get_face_center(face)
            if anchor_center and center:
                dist = np.sqrt(
                    (center[0] - anchor_center[0]) ** 2 + (center[1] - anchor_center[1]) ** 2
                )
                if dist > max_center_dist:
                    continue
            candidates.append((sim, face))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_sim, best_face = candidates[0]
        if len(candidates) >= 2:
            margin = best_sim - candidates[1][0]
            need_margin = 0.06 if multi_face else 0.06
            if margin < need_margin:
                # v5.1022b: Multi-face con margen mínimo 0.03 para caras similares
                if multi_face and margin >= 0.03:
                    pass  # Accept with relaxed margin
                else:
                    if frame_count % 25 == 1:
                        print(
                            f"[TRACK_MULTI] Re-enganche ambiguo frame {frame_count} "
                            f"({best_sim:.2f} vs {candidates[1][0]:.2f}) — hold"
                        )
                    return None

        print(
            f"[TRACK] RE-ENGANCHE estricto frame {frame_count} "
            f"(sim={best_sim:.2f}, min={min_sim:.2f}, multi={multi_face})"
        )
        return best_face

    def _apply_reacquired_face(self, face, assigned_attr, tracking_lost_attr,
                               position_history_attr, video_basename, last_bbox=None, last_kps=None):
        import copy
        out = copy.copy(face)
        if last_bbox is not None and hasattr(out, 'bbox'):
            old_bbox = np.array(last_bbox, dtype=np.float32)
            new_bbox = np.array(out.bbox, dtype=np.float32)
            dist = np.sqrt(
                np.sum(
                    ((old_bbox[:2] + old_bbox[2:]) / 2 - (new_bbox[:2] + new_bbox[2:]) / 2) ** 2
                )
            )
            face_w = max(1.0, float(new_bbox[2] - new_bbox[0]))
            old_weight = max(0.05, min(0.35, 0.35 * (1.0 - dist / max(1.0, face_w * 3.0))))
            blend_bbox = old_weight * old_bbox + (1.0 - old_weight) * new_bbox
            out.bbox = tuple(blend_bbox.astype(int))
            if last_kps is not None and hasattr(out, 'kps') and out.kps is not None:
                old_kps = np.array(last_kps, dtype=np.float32)
                new_kps = np.array(out.kps, dtype=np.float32)
                out.kps = (old_weight * old_kps + (1.0 - old_weight) * new_kps).tolist()
        setattr(self, assigned_attr, out)
        setattr(self, tracking_lost_attr, 0)
        center = get_face_center(out)
        if center:
            setattr(self, position_history_attr, [center])
        # v5.1005: Update tracking_embedding so next frame's local tracking
        # can use the newly recovered face's embedding instead of stale original
        if hasattr(face, 'embedding') and face.embedding is not None:
            emb_arr = np.array(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(emb_arr)
            if norm > 0:
                setattr(self, f'_tracking_embedding_{video_basename}', emb_arr / norm)
        return out

    def _find_target_face_for_selected_mode(self, video_path, valid_faces, current_frame=None):
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

            position_history_attr = f'_position_history_{video_basename}'
            position_history = getattr(self, position_history_attr, [])
            
            # Update scene state with current frame
            scene_state = self._update_scene_state(video_path, valid_faces, frame_count, position_history)

            if scene_state.get('_pending_reset'):
                scene_state['_pending_reset'] = False
                setattr(self, f'_scene_state_{video_basename}', scene_state)
                self._reset_tracking_on_scene_cut(
                    video_path, frame_count, reason="cambio abrupto de caras en escena",
                )
                scene_state = getattr(self, f'_scene_state_{video_basename}', scene_state)
                hold_suppressed = True
                global_after_cut = self._reacquire_face_global(
                    valid_faces, video_basename, frame_count, after_scene_cut=True,
                    closeup_mode=scene_state.get('is_close_up', False),
                )
                if global_after_cut is not None:
                    return self._apply_reacquired_face(
                        global_after_cut,
                        f'_target_face_assigned_{video_basename}',
                        f'_tracking_lost_count_{video_basename}',
                        position_history_attr, video_basename,
                    )
            
            assigned_attr = f'_target_face_assigned_{video_basename}'
            original_embedding_attr = f'_original_embedding_{video_basename}'
            tracking_lost_attr = f'_tracking_lost_count_{video_basename}'
            consecutive_success_attr = f'_consecutive_tracking_success_{video_basename}'
            
            # Initialize attributes if needed
            for attr, default in [(tracking_lost_attr, 0), (position_history_attr, []), (consecutive_success_attr, 0)]:
                if not hasattr(self, attr):
                    setattr(self, attr, default)
            
            lost_count = getattr(self, tracking_lost_attr)
            position_history = getattr(self, position_history_attr)
            hold_suppressed = frame_count <= getattr(self, f'_hold_suppressed_until_{video_basename}', 0)
            
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
                
                # v5.87: Sin propagación automática a otra cara del lote — solo referencia explícita del usuario
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
                        # v5.1034: umbral mucho más amplio — closeup con movimiento rápido puede saltar 400-500px
                        base_threshold = max(450, 1.50 * (assigned_face.bbox[2] - assigned_face.bbox[0]))
                    elif scene_state.get('camera_moving'):
                        # More aggressive tracking if camera is moving
                        base_threshold = max(450, 1.0 * (assigned_face.bbox[2] - assigned_face.bbox[0]))
                    else:
                        base_threshold = max(320, 0.85 * (assigned_face.bbox[2] - assigned_face.bbox[0]))  # v5.1023: 250→320
                    if len(valid_faces) >= 2:
                        face_span = assigned_face.bbox[2] - assigned_face.bbox[0]
                        base_threshold = min(base_threshold, max(240, face_span * 1.0))

                    # Predict next position using motion modeling
                    prediction = self._predict_next_position(position_history, scene_state)
                    
                    multi_face = len(valid_faces) >= 2
                    if multi_face and frame_count % 120 == 1:
                        print(f"[TRACK_MULTI] {len(valid_faces)} caras en frame — lock identidad activo")

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
                        
                        dist_weight = 0.22 if multi_face else 0.40
                        emb_weight = 0.58 if multi_face else 0.40
                        dist_normalized = 1.0 - (dist / base_threshold)
                        score += dist_normalized * dist_weight

                        if original_embedding is not None and hasattr(face, 'embedding') and face.embedding is not None:
                            emb_score = self._calculate_similarity(original_embedding, face.embedding)

                            # v5.1005: Use tracking_embedding (up-to-date) if it gives better match
                            tracking_emb_local = getattr(self, f'_tracking_embedding_{video_basename}', None)
                            if tracking_emb_local is not None:
                                track_emb_score = self._calculate_similarity(tracking_emb_local, face.embedding)
                                emb_score = max(emb_score, track_emb_score)

                            if multi_face:
                                emb_thresh = 0.35  # v5.1021: 0.48→0.35 — acepta más candidatos en multi-face
                            elif scene_state.get('is_close_up'):
                                emb_thresh = 0.28  # v5.1021: 0.32→0.28
                            elif scene_state.get('camera_moving'):
                                emb_thresh = 0.30  # v5.1021: 0.35→0.30
                            else:
                                emb_thresh = 0.32  # v5.1021: 0.38→0.32

                            if emb_score >= emb_thresh:
                                score += emb_score * emb_weight
                            else:
                                # v5.1004: Si la cara está muy cerca del ancla espacial
                                # (dist < 1/3 del umbral), NO rechazar — usar distancia
                                # como señal principal. El embedding bajo puede ser un
                                # falso negativo en multi-face (la misma persona pero
                                # con expresión/iluminación diferente).
                                if dist < base_threshold / 3:
                                    score += max(0.20, emb_score) * emb_weight * 0.60
                                else:
                                    score -= 0.55 if multi_face else 0.35
                                    if multi_face:
                                        continue
                        else:
                            if multi_face:
                                continue
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
                    
                    min_track_score = 0.40 if multi_face else 0.30  # v5.1021: 0.50→0.40 multi, 0.38→0.30 single
                    if scene_state.get('is_close_up') and lost_count > 0:
                        min_track_score = 0.25 if multi_face else 0.22  # v5.1021: 0.30→0.25 multi, 0.28→0.22 single
                    elif lost_count > 0:
                        min_track_score = max(0.15, min_track_score * 0.50)  # v5.1033: bajar umbral tras pérdida para aceptar caras con score bajo
                    identity_ok = True
                    id_margin = 1.0
                    if best_match and original_embedding is not None:
                        # v5.1005: Pass tracking_embedding as fallback for better match
                        fb_emb = getattr(self, f'_tracking_embedding_{video_basename}', None)
                        closeup_recovery = scene_state.get('is_close_up') and lost_count > 0
                        self._tracking_recovery = lost_count  # v5.1033: pasar a validate para relajar umbrales
                        identity_ok, id_margin = self._validate_locked_identity(
                            valid_faces, original_embedding, best_match,
                            multi_face=multi_face, fallback_embedding=fb_emb,
                            closeup_recovery=closeup_recovery,
                        )

                    # v5.1004: Spatial proximity bypass — si la cara está prácticamente
                    # en la misma posición que el ancla, aceptar directamente porque
                    # es la misma persona (indetectable a 60fps). El embedding puede
                    # fallar por iluminación/expresión pero la posición no engaña.
                    spatial_lock = best_match and best_distance < min(150 if not scene_state.get('is_close_up') else 220, base_threshold * (0.35 if not scene_state.get('is_close_up') else 0.50))  # v5.1034: closeup necesita más margen espacial
                    # v5.1033: en multi-face, spatial_lock requiere score mínimo para evitar oscilación de identidad
                    if multi_face:
                        spatial_lock = spatial_lock and best_score > max(0.05, min_track_score * 0.2)
                    if best_match and ((best_score > min_track_score and identity_ok) or spatial_lock):
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
                            # v5.81: Umbral más permisivo para baja calidad
                            avg_det_emb = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
                            score_threshold = 0.35 if avg_det_emb < 0.30 else 0.50
                            if consec_success >= 5 and best_score > score_threshold:
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
                                        print(f"[TRACK] Frame {frame_count}: Embedding blend updated (profile={is_p}, score={best_score:.2f})")
                        
                        # Save tracking embedding as recovery fallback
                        tracking_emb_attr = f'_tracking_embedding_{video_basename}'
                        if hasattr(best_match, 'embedding') and best_match.embedding is not None:
                            emb_arr = np.array(best_match.embedding, dtype=np.float32)
                            norm = np.linalg.norm(emb_arr)
                            if norm > 0:
                                setattr(self, tracking_emb_attr, emb_arr / norm)
                        
                        print(
                            f"[TRACK] Frame {frame_count}: Success "
                            f"(score={best_score:.2f}, dist={best_distance:.1f}px"
                            f"{f', margin={id_margin:.2f}' if multi_face else ''})"
                        )
                        return best_match

                    if best_match and multi_face and frame_count % 25 == 1:
                        print(
                            f"[TRACK_MULTI] Frame {frame_count}: candidato rechazado "
                            f"(score={best_score:.2f}, margin={id_margin:.2f}) — ghost/hold"
                        )
                    
                    rec_emb = original_embedding
                    if rec_emb is None:
                        rec_emb = getattr(self, f'_tracking_embedding_{video_basename}', None)
                    strict_min_sim = 0.35 if (scene_state.get('is_close_up') and lost_count > 0) else 0.50
                    strict_match = self._try_strict_same_person_reacquire(
                        valid_faces, rec_emb, assigned_face, video_basename, frame_count,
                        min_sim=strict_min_sim,
                    )
                    if strict_match is not None:
                        return self._apply_reacquired_face(
                            strict_match, assigned_attr, tracking_lost_attr,
                            position_history_attr, video_basename,
                            last_bbox=getattr(assigned_face, 'bbox', None),
                            last_kps=getattr(assigned_face, 'kps', None),
                        )

                    if best_score < 0 and valid_faces and assigned_center:
                        fc0 = get_face_center(valid_faces[0])
                        if fc0:
                            d0 = np.sqrt((fc0[0] - assigned_center[0])**2 + (fc0[1] - assigned_center[1])**2)
                            if frame_count % 30 == 1:
                                print(
                                    f"[TRACK_DIST] Frame {frame_count}: best_score={best_score:.2f}, "
                                    f"1st face dist={d0:.0f}px (umbral={base_threshold:.0f}), "
                                    f"n_faces={len(valid_faces)}, base_th={base_threshold:.0f}"
                                )
                        # Fallback: reintentar con umbral ampliado
                        expanded_threshold = base_threshold * 2.5
                        for face in valid_faces:
                            face_center = get_face_center(face)
                            if not face_center:
                                continue
                            dist = np.sqrt(
                                (face_center[0] - assigned_center[0])**2 +
                                (face_center[1] - assigned_center[1])**2
                            )
                            if dist > expanded_threshold:
                                continue
                            score = 0.0
                            if original_embedding is not None and hasattr(face, 'embedding') and face.embedding is not None:
                                emb_score = float(self._calculate_similarity(original_embedding, face.embedding))
                                emb_thresh_exp = 0.18 if (scene_state.get('is_close_up') and lost_count > 0) else (0.28 if scene_state.get('is_close_up') else 0.38)
                                if emb_score >= emb_thresh_exp:
                                    score = emb_score * 0.30
                            # v5.1034: closeup+lost acepta por proximidad espacial aunque embedding sea bajo
                            if score <= 0 and scene_state.get('is_close_up') and lost_count > 0:
                                score = max(0.05, 0.25 * (1.0 - dist / expanded_threshold))
                            if score > best_score:
                                best_score = score
                                best_match = face
                                best_distance = dist
                        if best_match and best_score > 0:
                            closeup_recovery_exp = scene_state.get('is_close_up')
                            min_closeup_score = 0.06  # v5.1034: umbral más bajo para closeup+lost
                            if (closeup_recovery_exp and best_score > min_closeup_score) or best_score > 0.15:
                                if closeup_recovery_exp and best_score < 0.15:
                                    # v5.1027: Low-confidence closeup — require 3 consecutive frames
                                    confirm_attr = f'_closeup_confirm_{video_basename}'
                                    confirm_count = getattr(self, confirm_attr, 0)
                                    if confirm_count < 2:  # v5.1034: 3→2 para recuperación más rápida
                                        setattr(self, confirm_attr, confirm_count + 1)
                                        setattr(self, tracking_lost_attr, 0)
                                        if frame_count % 10 == 1:
                                            print(f"[TRACK] Frame {frame_count}: closeup pendiente ({best_score:.2f}, confirm {confirm_count+1}/2)")
                                        bb = best_match.bbox if (hasattr(best_match, 'bbox') and best_match.bbox is not None) else None
                                        if bb is not None:
                                            pfr = getattr(self, '_prev_frame_result', {}).get(getattr(self, '_current_video_key', ''))
                                            return self._hold_previous_face_region(
                                                current_frame, pfr, bb,
                                                blend=0.90, prev_mask=None,
                                            )
                                    else:
                                        setattr(self, confirm_attr, 0)
                                        if frame_count % 10 == 1:
                                            print(f"[TRACK] Frame {frame_count}: closeup recovery confirmado ({best_score:.2f})")
                                        return self._apply_reacquired_face(
                                            best_match, assigned_attr, tracking_lost_attr,
                                            position_history_attr, video_basename,
                                            last_bbox=getattr(assigned_face, 'bbox', None),
                                            last_kps=getattr(assigned_face, 'kps', None),
                                        )
                                else:
                                    # High confidence (>=0.15), accept immediately
                                    if closeup_recovery_exp:
                                        print(f"[TRACK] Frame {frame_count}: closeup recovery aceptado ({best_score:.2f}, dist={best_distance:.0f}px)")
                                    return self._apply_reacquired_face(
                                        best_match, assigned_attr, tracking_lost_attr,
                                        position_history_attr, video_basename,
                                        last_bbox=getattr(assigned_face, 'bbox', None),
                                        last_kps=getattr(assigned_face, 'kps', None),
                                    )
                    elif frame_count % 30 == 1:
                        print(
                            f"[TRACK] Frame {frame_count}: tracking local perdido "
                            f"(score={best_score:.2f}) — ghost/hold"
                        )

                    # Tracking truly lost
                    lost_count += 1
                    setattr(self, tracking_lost_attr, lost_count)
                    setattr(self, consecutive_success_attr, 0)

                    if hold_suppressed or lost_count <= 3 or self._in_reacquire_window(video_basename, frame_count):
                        global_match = self._reacquire_face_global(
                            valid_faces, video_basename, frame_count,
                            after_scene_cut=hold_suppressed or self._in_reacquire_window(video_basename, frame_count),
                            closeup_mode=scene_state.get('is_close_up', False),
                        )
                        if global_match is not None:
                            return self._apply_reacquired_face(
                                global_match, assigned_attr, tracking_lost_attr,
                                position_history_attr, video_basename,
                                last_bbox=getattr(assigned_face, 'bbox', None),
                                last_kps=getattr(assigned_face, 'kps', None),
                            )
                    
                    # ==========================================================
                    # FALLBACK FINAL v5.1022b: Aceptar best_match si está cerca
                    # aunque identidad sea baja. Evita swap congelado en videos
                    # multi-face con detecciones de baja calidad (obloog2).
                    # ==========================================================
                    if best_match is not None and assigned_center is not None:
                        fc = get_face_center(best_match)
                        if fc:
                            d = np.sqrt((fc[0] - assigned_center[0])**2 + (fc[1] - assigned_center[1])**2)
                            fbw = max(1.0, best_match.bbox[2] - best_match.bbox[0]) if (hasattr(best_match, 'bbox') and best_match.bbox is not None) else 100
                            max_dist = max(500, fbw * 3.0) if scene_state.get('is_close_up') else max(350, fbw * 2.5)  # v5.1034: closeup más permisivo
                            if d < max_dist:
                                print(f"[TRACK] Frame {frame_count}: fallback espacial (dist={d:.0f}px, closeup={scene_state.get('is_close_up', False)})")
                                return self._apply_reacquired_face(
                                    best_match, assigned_attr, tracking_lost_attr,
                                    position_history_attr, video_basename,
                                    last_bbox=getattr(assigned_face, 'bbox', None),
                                    last_kps=getattr(assigned_face, 'kps', None),
                                )

                    # ==========================================================
                    # GHOST TRACKING (v2.3): Predecir posición si se pierde la cara
                    # Evita parpadeos en perfiles o movimientos rápidos
                    # ==========================================================
                    # v5.1020: Ghost limits +2 para evitar que el tracking se quede
                    # sin swap durante RE-BUSQUEDA largas (swap congelado en Video 3).
                    avg_det_g = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
                    ghost_limit = 8 if avg_det_g < 0.30 else (7 if avg_det_g < 0.45 else 6)  # v5.1023: +2 — más persistencia en giros rápidos
                    if scene_state.get('is_close_up'):
                        ghost_limit += 1
                    if not hold_suppressed and lost_count < ghost_limit and prediction and assigned_face:
                        import copy
                        ghost_face = copy.copy(assigned_face)
                        pred_x, pred_y, pred_conf = prediction
                        
                        # Calcular desplazamiento desde la última posición conocida
                        old_center = get_face_center(assigned_face)
                        if old_center:
                            # v5.1024: OPTICAL FLOW ENHANCEMENT — usar flujo óptico de Farneback
                            # para refinar el desplazamiento en ghost tracking (captura rotación real)
                            flow_dx = 0.0
                            flow_dy = 0.0
                            if current_frame is not None and lost_count < 3:
                                prev_frame_of = getattr(self, '_prev_frame_optical', None)
                                if prev_frame_of is not None and prev_frame_of.shape == current_frame.shape:
                                    try:
                                        prev_gray = cv2.cvtColor(prev_frame_of, cv2.COLOR_BGR2GRAY)
                                        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                                        flow = cv2.calcOpticalFlowFarneback(
                                            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                                        )
                                        # Media del flujo en la región de la cara anterior
                                        bx1, by1, bx2, by2 = [int(v) for v in assigned_face.bbox]
                                        bx1, by1 = max(0, bx1), max(0, by1)
                                        bx2 = min(flow.shape[1], bx2)
                                        by2 = min(flow.shape[0], by2)
                                        if bx2 > bx1 and by2 > by1:
                                            face_flow = flow[by1:by2, bx1:bx2]
                                            flow_dx = float(np.median(face_flow[:, :, 0]))
                                            flow_dy = float(np.median(face_flow[:, :, 1]))
                                    except Exception:
                                        pass
                            setattr(self, '_prev_frame_optical', current_frame)

                            # Combinar Kalman prediction + optical flow
                            dx = pred_x - old_center[0]
                            dy = pred_y - old_center[1]
                            # Si optical flow sugiere dirección diferente, darle peso
                            if abs(flow_dx) > 5 or abs(flow_dy) > 5:
                                of_weight = 0.35 if lost_count < 2 else 0.20
                                dx = dx * (1.0 - of_weight) + flow_dx * of_weight
                                dy = dy * (1.0 - of_weight) + flow_dy * of_weight
                            
                            # Dampen extreme displacements (likely wrong prediction)
                            # v5.1034: max_shift mucho mayor para closeup (movimiento rápido en primer plano)
                            max_shift = 200 if (scene_state.get('is_close_up') and lost_count < 5) else (120 if lost_count < 5 else (100 if scene_state.get('is_close_up') else 60))
                            if abs(dx) > max_shift:
                                dx = max_shift if dx > 0 else -max_shift
                            if abs(dy) > max_shift:
                                dy = max_shift if dy > 0 else -max_shift
                            
                            # v5.48: Inercia aumentada para mejor seguimiento de perfiles
                            inertia = 0.88 if avg_det_g < 0.30 else 0.85  # Más inercia en baja calidad
                            dx = dx * inertia
                            dy = dy * inertia
                            
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
                            decay_slope = 0.020 if avg_det_g < 0.30 else 0.025  # Decaimiento más suave en baja calidad
                            decay = max(0.5, 1.0 - lost_count * decay_slope)
                            ghost_face.det_score = getattr(assigned_face, 'det_score', 0.5) * decay
                            
                            # Actualizar historia con la predicción para mantener la inercia
                            position_history.append((pred_x, pred_y))
                            if len(position_history) > 15: position_history.pop(0)
                            setattr(self, position_history_attr, position_history)
                            
                            if lost_count % 3 == 0:
                                print(f"[TRACK] Ghost Tracking Activo (frame {frame_count}, lost={lost_count}/{ghost_limit}, score_decay={decay:.2f})")
                            return ghost_face

                    print(f"[TRACK] Frame {frame_count}: Lost (score={best_score:.2f}), attempts {lost_count}/15")
                    
                    # v5.1020: Forzar reacquire agresivo tras ghost expired
                    if lost_count >= ghost_limit and not hold_suppressed:
                        global_match = self._reacquire_face_global(
                            valid_faces, video_basename, frame_count,
                            after_scene_cut=True, closeup_mode=scene_state.get('is_close_up', False),
                        )
                        if global_match is not None:
                            return self._apply_reacquired_face(
                                global_match, assigned_attr, tracking_lost_attr,
                                position_history_attr, video_basename,
                                last_bbox=getattr(assigned_face, 'bbox', None),
                                last_kps=getattr(assigned_face, 'kps', None),
                            )
                    
                    # Reset y preparar re-intento periódico
                    if lost_count >= 8:  # v5.1020: 15→8 — reacquire más rápido tras pérdida
                        print(f"[TRACK] Resetting state after {lost_count} failures")
                        setattr(self, assigned_attr, None)
                        setattr(self, position_history_attr, [])
                        setattr(self, f'_hold_suppressed_until_{video_basename}', frame_count + 8)
                        # Reset profile hysteresis para arrancar limpio en re-adquisición
                        for k in list(self.__dict__.keys()):
                            if k.startswith(f'_profile_hysteresis_video_{video_basename}'):
                                delattr(self, k)
                        self._open_reacquire_window(video_basename, frame_count, frames=90)
                        global_match = self._reacquire_face_global(
                            valid_faces, video_basename, frame_count,
                            after_scene_cut=True, closeup_mode=True,
                        )
                        if global_match is not None:
                            return self._apply_reacquired_face(
                                global_match, assigned_attr, tracking_lost_attr,
                                position_history_attr, video_basename,
                                last_bbox=getattr(assigned_face, 'bbox', None),
                                last_kps=getattr(assigned_face, 'kps', None),
                            )
                    return None

            assigned_face = getattr(self, assigned_attr, None)
            if assigned_face is None:
                # v5.1029: Marcar como no-asignado para que reacquire se active con long_unassigned
                self._mark_unassigned(video_basename, frame_count)
                recovery_attr = f'_strict_recovery_tick_{video_basename}'
                tick = getattr(self, recovery_attr, 0) + 1
                setattr(self, recovery_attr, tick)

                in_window = self._in_reacquire_window(video_basename, frame_count)
                near_cut = abs(frame_count - scene_state.get('last_scene_cut', 0)) < 60
                unassigned_since = getattr(self, f'_unassigned_since_{video_basename}', 0)
                long_unassigned = unassigned_since > 0 and (frame_count - unassigned_since) >= 15
                is_close_up = scene_state.get('is_close_up', False)
                if in_window or near_cut or long_unassigned or tick % 2 == 0:
                    global_match = self._reacquire_face_global(
                        valid_faces, video_basename, frame_count,
                        after_scene_cut=in_window or near_cut or long_unassigned,
                        closeup_mode=long_unassigned or is_close_up,
                    )
                    if global_match is not None:
                        setattr(self, recovery_attr, 0)
                        return self._apply_reacquired_face(
                            global_match, assigned_attr, tracking_lost_attr,
                            position_history_attr, video_basename,
                        )

                rec_emb = original_embedding
                if rec_emb is None:
                    rec_emb = getattr(self, f'_tracking_embedding_{video_basename}', None)
                last_bbox = getattr(self, f'_last_known_bbox_{video_basename}', None)
                last_kps = getattr(self, f'_last_known_kps_{video_basename}', None)
                if rec_emb is not None and last_bbox is not None and tick % 6 == 0:
                    class _BBoxAnchor:
                        pass
                    anchor = _BBoxAnchor()
                    anchor.bbox = last_bbox
                    strict_recovery_min_sim = 0.35 if scene_state.get('is_close_up') else 0.50
                    strict_match = self._try_strict_same_person_reacquire(
                        valid_faces, rec_emb, anchor, video_basename, frame_count,
                        min_sim=strict_recovery_min_sim,
                    )
                    if strict_match is not None:
                        setattr(self, recovery_attr, 0)
                        if hasattr(self, f'_last_known_bbox_{video_basename}'):
                            delattr(self, f'_last_known_bbox_{video_basename}')
                        if hasattr(self, f'_last_known_kps_{video_basename}'):
                            delattr(self, f'_last_known_kps_{video_basename}')
                        return self._apply_reacquired_face(
                            strict_match, assigned_attr, tracking_lost_attr,
                            position_history_attr, video_basename,
                            last_bbox=last_bbox, last_kps=last_kps,
                        )

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
        """Detect if face is a profile/side view based on 5-point kps landmarks
        v5.1024: Thresholds expandidos para capturar semi-perfiles (eye_ratio<0.25)."""
        if kps is None or len(kps) < 5:
            return False
        kps = np.array(kps)
        eye_dist_x = abs(kps[0][0] - kps[1][0])
        face_width = np.max(kps[:, 0]) - np.min(kps[:, 0])
        if face_width < 1:
            return False
        eye_ratio = eye_dist_x / face_width
        eye_dist_y = abs(kps[0][1] - kps[1][1])
        eye_y_ratio = eye_dist_y / (face_width + 1e-6)
        # v5.1024: Perfil = ojos juntos (semi-perfil 45° incluido) o inclinación fuerte
        return eye_ratio < 0.25 or (eye_ratio < 0.32 and eye_y_ratio > 0.35)

    @staticmethod
    def _source_identity_params(is_profile: bool = False, is_closeup: bool = False):
        """
        Mapea ajustes UI/globals hacia máximo parecido source.
        Slider enhancer alto (0.85-0.95) = poca mezcla GFPGAN = más identidad cruda del swap.
        """
        ui_identity = float(getattr(roop.globals, 'enhancer_blend_factor', 0.94))
        gfpgan_mix = float(np.clip((1.0 - ui_identity) * 0.35, 0.10, 0.42))  # v5.1032: floor 0.06→0.10 — más nitidez
        if is_closeup:
            gfpgan_mix = min(gfpgan_mix * 1.5, 0.30)  # v5.1024: más GFPGAN en closeups para textura de piel
        elif ui_identity >= 0.90:
            gfpgan_mix = min(gfpgan_mix, 0.28)
        base_cm = float(getattr(roop.globals, 'color_match_strength', 0.10))
        color_match = min(0.12, base_cm * 0.80) if is_profile else base_cm
        if is_closeup:
            color_match *= 0.80
        brightness = float(getattr(roop.globals, 'brightness_strength', 0.15))
        brightness *= 1.20
        return gfpgan_mix, color_match, brightness

    @staticmethod
    def _is_closeup_face(bbox, frame_shape) -> bool:
        """Primer plano en vertical, horizontal o cuadrado (talking-head / close-up)."""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fh, fw = frame_shape[:2]
            face_h = max(1, y2 - y1)
            face_w = max(1, x2 - x1)
            area_ratio = (face_h * face_w) / float(max(1, fh * fw))
            height_ratio = face_h / float(fh)
            width_ratio = face_w / float(fw)
            frame_aspect = fw / float(max(1, fh))

            if frame_aspect < 0.85:
                # Vertical (Grok 9:16, reels…): cara alta y estrecha
                return (
                    area_ratio > 0.055
                    or height_ratio > 0.16
                    or width_ratio > 0.28
                )
            if frame_aspect > 1.15:
                # Horizontal (16:9, 4:3…): priorizar altura de cara en frame ancho
                return (
                    area_ratio > 0.050
                    or height_ratio > 0.17
                    or width_ratio > 0.20
                )
            # Casi cuadrado
            return (
                area_ratio > 0.060
                or height_ratio > 0.18
                or width_ratio > 0.24
            )
        except Exception:
            return False

    @staticmethod
    def _hold_previous_face_region(current_frame, prev_frame, bbox, blend=0.88, prev_mask=None):
        """Mantiene el último swap cuando falla tracking/detección.
        v5.94: Reutiliza la máscara real del último swap si existe (evita círculo visible)."""
        if prev_frame is None or bbox is None or current_frame is None:
            return current_frame
        try:
            if prev_frame.shape != current_frame.shape:
                return current_frame
            x1, y1, x2, y2 = [int(v) for v in bbox]
            fw, fh = x2 - x1, y2 - y1
            if fw <= 0 or fh <= 0:
                return current_frame
            mix = float(np.clip(blend, 0.05, 0.92))  # v5.1028: 0.5→0.05 para fade-out completo de ghost
            h_f, w_f = current_frame.shape[:2]

            if prev_mask is not None and prev_mask.shape[:2] == (h_f, w_f):
                soft_mask = np.clip(prev_mask.astype(np.float32), 0, 1)
                blur_sz = max(21, min(fw, fh) // 5) | 1
                soft_mask = cv2.GaussianBlur(soft_mask, (blur_sz, blur_sz), 0)
            else:
                cx = int(np.clip((x1 + x2) // 2, 0, w_f - 1))
                cy = int(np.clip((y1 + y2) // 2, 0, h_f - 1))
                soft_mask = np.zeros((h_f, w_f), dtype=np.float32)
                cv2.ellipse(
                    soft_mask,
                    (cx, cy),
                    (max(8, int(fw * 0.40)), max(8, int(fh * 0.44))),
                    0, 0, 360, 1.0, -1,
                )
                blur_sz = max(21, min(fw, fh) // 4) | 1
                soft_mask = cv2.GaussianBlur(soft_mask, (blur_sz, blur_sz), 0)
            soft_mask = np.clip(soft_mask * mix, 0, 1)
            soft_3ch = cv2.merge([soft_mask, soft_mask, soft_mask])

            curr = current_frame.astype(np.float32)
            prev_matched = match_color_histogram(prev_frame, current_frame, blend_factor=0.35).astype(np.float32)
            blended = curr * (1.0 - mix) + prev_matched * mix
            return (curr * (1.0 - soft_3ch) + blended * soft_3ch).astype(np.uint8)
        except Exception:
            return current_frame

    @staticmethod
    def _build_ellipse_mask(frame_shape, cx, cy, rx, ry, tight=False):
        h_f, w_f = frame_shape[:2]
        cx = int(np.clip(cx, 0, w_f - 1))
        cy = int(np.clip(cy, 0, h_f - 1))
        if tight:
            rx = max(12, int(rx * 0.88))
            ry = max(12, int(ry * 0.88))
        rx = max(12, int(rx))
        ry = max(12, int(ry))
        mask = np.zeros((h_f, w_f), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 1.0, -1)
        blur_ratio = 0.45 if tight else 0.42  # v5.1026: 0.35→0.42 — blur extra para eliminar borde de elipse
        blur_size = max(31 if tight else 27, int(min(rx, ry) * blur_ratio)) | 1  # v5.1026: min 25/21→31/27
        mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        # v5.1024: Radial gradient — elimina el borde visible del elipse de safety recovery.
        # La elipse sólida + blur crea un perímetro visible como "marco" porque la transición
        # 1.0→0.0 en ~blur_size px es detectable. El gradiente radial decae 1.0→0.0 en 2× el
        # radio, haciendo que el borde exterior se desvanezca gradualmente sin perímetro detectable.
        ys, xs = np.ogrid[:h_f, :w_f]
        dist = np.sqrt(((xs - cx) / max(1, rx))**2 + ((ys - cy) / max(1, ry))**2)
        radial = np.clip(1.0 - dist * 0.50, 0, 1.0)
        mask = mask * radial
        return np.clip(mask, 0, 1.0)

    @staticmethod
    def _regenerate_face_safety_mask(frame_shape, swapped_face_aligned, M_inv, bbox=None, tight=False):
        """Regenera máscara elíptica; elige warp o bbox según cobertura real."""
        candidates = []
        h_f_s, w_f_s = frame_shape[:2]

        if M_inv is not None and swapped_face_aligned is not None:
            h_a_f, w_a_f = swapped_face_aligned.shape[:2]
            corners = np.array(
                [[0, 0], [w_a_f, 0], [w_a_f, h_a_f], [0, h_a_f]],
                dtype=np.float32,
            )
            corners_w = cv2.transform(corners.reshape(-1, 1, 2), M_inv).reshape(-1, 2)
            cx_w = int(np.mean(corners_w[:, 0]))
            cy_w = int(np.mean(corners_w[:, 1]))
            face_w_s = max(8, int(np.max(corners_w[:, 0]) - np.min(corners_w[:, 0])))
            face_h_s = max(8, int(np.max(corners_w[:, 1]) - np.min(corners_w[:, 1])))
            if face_w_s >= 20 and face_h_s >= 20:
                warp_mask = ProcessMgr._build_ellipse_mask(
                    frame_shape, cx_w, cy_w, face_w_s // 2, face_h_s // 2, tight=tight,
                )
                candidates.append(('warp', warp_mask))

        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cx_b, cy_b = (x1 + x2) // 2, (y1 + y2) // 2
            scale = 0.88 if tight else 1.0
            rx_b = max(16, int((x2 - x1) // 2 * scale))
            ry_b = max(16, int((y2 - y1) // 2 * scale))
            bbox_mask = ProcessMgr._build_ellipse_mask(frame_shape, cx_b, cy_b, rx_b, ry_b, tight=tight)
            candidates.append(('bbox', bbox_mask))

        if not candidates:
            return None

        best = max(candidates, key=lambda item: item[1].mean())
        return best[1]

    @staticmethod
    def _mask_face_coverage(mask, threshold=0.15):
        if mask is None:
            return 0
        return int(np.sum(mask > threshold))

    @staticmethod
    def _mask_needs_safety_recovery(mask, face_w, face_h):
        """Detecta máscaras rotas por cobertura facial real (no mean global del frame)."""
        if mask is None:
            return True
        mask_max = float(mask.max())
        if mask_max < 0.25:
            return True
        face_area = max(1, face_w * face_h)
        min_pixels = max(64, int(face_area * 0.10))
        coverage = ProcessMgr._mask_face_coverage(mask, 0.15)
        if coverage < min_pixels:
            return True
        strong = int(np.sum(mask > 0.45))
        if strong < max(24, int(face_area * 0.04)):
            return True
        return False

    @staticmethod
    def _build_bbox_safety_mask(frame_shape, x1, y1, x2, y2, scale=1.0, tight=False):
        fw, fh = max(8, x2 - x1), max(8, y2 - y1)
        scale = float(max(0.75, scale))
        return ProcessMgr._build_ellipse_mask(
            frame_shape,
            (x1 + x2) // 2, (y1 + y2) // 2,
            max(18, int(fw // 2 * scale)), max(18, int(fh // 2 * scale)),
            tight=tight,
        )

    @staticmethod
    def _feather_mask_edges(mask, feather_px=14):
        """Suaviza bordes con distance transform — elimina marco/elipse visible."""
        if mask is None or float(mask.max()) < 0.01:
            return mask
        m = np.clip(mask.astype(np.float32), 0, 1)
        adaptive_thresh = min(0.35, max(0.10, float(m.mean()) * 1.8))
        binary = (m > adaptive_thresh).astype(np.uint8) * 255
        if not np.any(binary):
            return m
        feather = max(6.0, float(feather_px))
        dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        ramp = np.clip(dist_in / feather, 0, 1).astype(np.float32)
        core = m > 0.55
        softened = np.where(core, m, m * ramp)
        return np.clip(softened, 0, 1.0)

    @staticmethod
    def _widen_mask_for_blend(mask, face_w, face_h, frame_shape=None):
        """Transición suave sin invadir pelo/fondo — un solo blur mantiene el borde en la cara.
        v5.1020: Si frame_shape se proporciona, añade gradiente en bordes de frame para
        evitar que la elipse extendida fuera del frame cree un borde visible al cortarse
        contra el límite del fotograma."""
        m = np.clip(mask.astype(np.float32), 0, 1)
        blur = max(9, (min(face_w, face_h) // 10) | 1)
        blur = min(blur, 61) | 1  # v5.1024c: 45→61 — transición más ancha para closeups extremos
        # v5.1028: Cap relativo al frame para evitar que el blur sea demasiado grande en videos pequeños (224x432)
        if frame_shape is not None:
            frame_min = min(frame_shape[0], frame_shape[1])
            blur = min(blur, max(9, (frame_min // 6) | 1))
        wide = cv2.GaussianBlur(m, (blur, blur), 0)
        # v5.1020: Taper a 0 en bordes de frame para evitar marco visible
        # cuando la cara está cerca del borde y la máscara extendida se corta.
        if frame_shape is not None:
            h, w = frame_shape[:2]
            border_px = max(5, min(blur // 2, 20))
            y_grad = np.ones(h, dtype=np.float32)
            if border_px > 0:
                y_grad[:border_px] = np.linspace(0.0, 1.0, border_px)
                y_grad[-border_px:] = np.linspace(1.0, 0.0, border_px)
            x_grad = np.ones(w, dtype=np.float32)
            if border_px > 0:
                x_grad[:border_px] = np.linspace(0.0, 1.0, border_px)
                x_grad[-border_px:] = np.linspace(1.0, 0.0, border_px)
            grad_2d = np.outer(y_grad, x_grad)
            wide = wide * grad_2d
        return np.clip(wide, 0, 1.0)

    @staticmethod
    def _apply_forehead_gradient(blend_mask, frame_shape, target_face, y1, face_h, is_profile):
        """Gradiente frontal suave: reduce el swap en la parte superior de la frente
        para preservar flequillo/cabello del target. Usa keypoints oculares (kps) como
        referencia anatómica para determinar dónde termina la frente y empieza el cabello.
        v5.1027: grad_start=0 para que el gradiente cubra TODA el área superior del frame
        sin discontinuidad. El gradiente smoothstep va de 0 (máxima reducción en top del
        frame) a 1 (sin reducción en línea de ojos). El área extendida por
        _widen_mask_for_blend (GaussianBlur sobre XSeg) queda dentro del gradiente,
        eliminando la línea visible en el borde superior del bbox."""
        h_f, w_f = frame_shape[:2]
        try:
            if hasattr(target_face, 'kps') and target_face.kps is not None and len(target_face.kps) >= 2:
                kps = np.array(target_face.kps)
                eye_y = float(min(kps[0][1], kps[1][1]))
            else:
                eye_y = y1 + face_h * 0.55
            grad_start = 0  # v5.1027: cubre desde top del frame para evitar discontinuidad de widen_mask
            grad_end = min(h_f, int(eye_y + int(face_h * 0.05)))
            if grad_end > grad_start + 3:
                grad_h = grad_end - grad_start
                t = np.linspace(0.0, 1.0, grad_h, dtype=np.float32)
                smooth_t = t * t * (3.0 - 2.0 * t)
                max_reduction = 0.50 if is_profile else 0.50  # v5.1035: max 50% reducción — evita efecto flequillo visible
                reduction = max_reduction * (1.0 - smooth_t)
                blend_mask[grad_start:grad_end, :] *= (1.0 - reduction[:, np.newaxis])
        except Exception:
            pass
        return blend_mask

    @staticmethod
    def _prime_warp_edges(warped_face, original_frame, mask, face_w, face_h):
        """Pre-mezcla warped→original en zona de borde — preserva flequillo mezclando
        más original en la transición sin endurecer el borde de máscara."""
        if warped_face is None or original_frame is None or mask is None:
            return warped_face
        m = np.clip(mask.astype(np.float32), 0, 1)
        edge = np.clip((0.72 - m) / 0.70, 0, 1).astype(np.float32)
        blur = max(11, (min(face_w, face_h) // 10) | 1)
        edge = cv2.GaussianBlur(edge, (blur, blur), 0) * 0.80
        if float(edge.max()) < 0.02:
            return warped_face
        e3 = cv2.merge([edge, edge, edge])
        out = warped_face.astype(np.float32) * (1.0 - e3) + original_frame.astype(np.float32) * e3
        return np.clip(out, 0, 255).astype(np.uint8)

    @staticmethod
    def _soothe_swap_ring(result_frame, original_frame, mask, face_w, face_h):
        """Post-paso: funde el anillo de transición hacia el original — preserva flequillo."""
        if mask is None or result_frame is None or original_frame is None:
            return result_frame
        if result_frame.shape != original_frame.shape:
            return result_frame
        m = np.clip(mask.astype(np.float32), 0, 1)
        if float(m.max()) < 0.05:
            return result_frame
        soothe = np.clip((0.97 - m) / 0.94, 0, 1).astype(np.float32)
        soothe *= (m > 0.01).astype(np.float32) * (m < 0.90).astype(np.float32)
        blur = max(17, (min(face_w, face_h) // 6) | 1)
        soothe = cv2.GaussianBlur(soothe, (blur, blur), 0)
        if float(soothe.max()) < 0.02:
            return result_frame
        soothe *= 0.95
        s3 = cv2.merge([soothe, soothe, soothe])
        blended = (
            result_frame.astype(np.float32) * (1.0 - s3)
            + original_frame.astype(np.float32) * s3
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    @staticmethod
    def _mouth_mask_full_frame(frame, mouth_region, crop_origin=(0, 0), crop_size=None):
        """Máscara oral en coordenadas del frame completo (corrige crop en close-up)."""
        from roop.processors.FaceSwap import create_mouth_preservation_mask
        if crop_size is not None and mouth_region is not None:
            ox, oy = crop_origin
            cw, ch = crop_size
            if cw > 0 and ch > 0:
                crop_canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
                crop_mask = create_mouth_preservation_mask(crop_canvas, mouth_region, blend_ratio=1.0)
                full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
                y2p = min(frame.shape[0], oy + ch)
                x2p = min(frame.shape[1], ox + cw)
                full_mask[oy:y2p, ox:x2p] = crop_mask[0:y2p - oy, 0:x2p - ox]
                return full_mask
        return create_mouth_preservation_mask(frame, mouth_region, blend_ratio=1.0)

    @staticmethod
    def _detect_mouth_content_diff(swapped_aligned, reference_aligned):
        """Diferencia swap vs original en zona oral — detecta lengua/objetos no presentes en el swap."""
        try:
            if swapped_aligned is None or reference_aligned is None:
                return 0.0
            if swapped_aligned.shape != reference_aligned.shape:
                reference_aligned = cv2.resize(
                    reference_aligned,
                    (swapped_aligned.shape[1], swapped_aligned.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            h, w = swapped_aligned.shape[:2]
            mouth_zone = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(
                mouth_zone, (w // 2, int(h * 0.72)),
                (max(8, int(w * 0.24)), max(6, int(h * 0.16))),
                0, 0, 360, 255, -1,
            )
            diff = cv2.absdiff(swapped_aligned, reference_aligned)
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            mouth_pixels = diff_gray[mouth_zone > 0]
            if len(mouth_pixels) < 40:
                return 0.0
            mean_diff = float(np.mean(mouth_pixels))
            p90_diff = float(np.percentile(mouth_pixels, 90))
            return float(np.clip((mean_diff * 0.45 + p90_diff * 0.55) / 255.0, 0.0, 1.0))
        except Exception:
            return 0.0

    @staticmethod
    def _masked_temporal_blend(current, previous, mask, alpha):
        """Mezcla temporal solo dentro de la máscara facial (evita blur en todo el frame)."""
        if previous is None or alpha <= 0.001:
            return current
        if previous.shape != current.shape:
            return current
        alpha = float(np.clip(alpha, 0.0, 0.55))
        m = np.clip(mask.astype(np.float32), 0.0, 1.0)
        if m.max() < 0.01:
            return current
        m3 = cv2.merge([m, m, m])
        blended = current.astype(np.float32) * (1.0 - alpha) + previous.astype(np.float32) * alpha
        return (current.astype(np.float32) * (1.0 - m3) + blended * m3).astype(np.uint8)

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

            # ============================================
            # SUAVIZADO TEMPORAL AVANZADO (v3.5)
            # ============================================
            video_key = getattr(self, '_current_video_key', 'default')
            video_basename = video_key[6:] if video_key.startswith('video_') else video_key

            # Detectar perfil con hysteresis para evitar parpadeo
            # raw_is_profile incrementa +2 por detección, decae -1 por no-detección
            # Así una detección aislada da 2 frames de persistencia; una racha da más
            target_kps = getattr(target_face, 'kps', None)
            raw_is_profile = ProcessMgr._is_profile_face(target_kps)
            prof_key = f'_profile_hysteresis_{video_key}'
            prof_state = getattr(self, prof_key, 0)
            if raw_is_profile:
                prof_state = min(prof_state + 2, 6)
                is_profile = True
            elif prof_state > 0:
                prof_state -= 1
                is_profile = True
            else:
                prof_state = 0
                is_profile = False
            setattr(self, prof_key, prof_state)
            if is_profile:
                print(f"[PROFILE] Frame {call_num}: Perfil detectado (raw={raw_is_profile}, sticky={prof_state})")
            video_path = getattr(self.options, 'current_video_path', None)
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

                    # Adaptive scene cut: lower threshold for low quality video
                    avg_det_scene = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
                    scene_threshold = 120 if avg_det_scene < 0.35 else 160
                    # Also check scene state for face-count-based cut
                    scene_state_obj = getattr(self, f'_scene_state_{video_basename}', {})
                    scene_cut_frame = scene_state_obj.get('last_scene_cut', 0) if isinstance(scene_state_obj, dict) else 0
                    frame_count_scene = getattr(self, f'_frame_count_{video_basename}', 0)
                    is_near_scene_cut = abs(frame_count_scene - scene_cut_frame) < 5 if scene_cut_frame > 0 else False

                    plan_smooth = frame_count_scene <= getattr(
                        self, f'_plan_change_smooth_until_{video_basename}', 0,
                    )
                    hard_scene_cut = velocity > scene_threshold and not plan_smooth
                    soft_scene_cut = is_near_scene_cut and velocity > 35 and not plan_smooth
                    if (hard_scene_cut or soft_scene_cut) and not is_lockout:
                        print(f"[SCENE_CUT] Frame {call_num}: Salto detectado (vel={velocity:.0f}px, thresh={scene_threshold}, face_cut={is_near_scene_cut}). Reseteando historial temporal.")
                        if not hasattr(self, '_last_reset_call'): self._last_reset_call = {}
                        self._last_reset_call[video_key] = call_num
                        if video_path and hard_scene_cut:
                            self._reset_tracking_on_scene_cut(
                                video_path, frame_count_scene,
                                reason=f"salto bbox vel={velocity:.0f}px",
                            )

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
                            # v5.79: det_score-aware smoothing — low-confidence jumps get smoothed more
                            det_score_kps = getattr(target_face, 'det_score', 1.0)
                            # v5.81: avg_det_score adaptive — pervasively low quality = more smoothing
                            avg_det_kps = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
                            if det_score_kps < 0.65 and velocity > 25:
                                f_center = 0.75
                                kps_ema = 0.60 if not is_profile else 0.55
                                if call_num % 30 == 1:
                                    print(f"[TRACK_SMOOTH] Frame {call_num}: det={det_score_kps:.2f}, vel={velocity:.0f}px → suavizado fuerte")
                            elif avg_det_kps < 0.30 and velocity > 15:
                                # Low quality + moderate motion → extra smoothing
                                f_center = 0.70
                                kps_ema = 0.55 if not is_profile else 0.50
                                if call_num % 30 == 1:
                                    print(f"[TRACK_LOWQ] Frame {call_num}: avg_det={avg_det_kps:.2f}, vel={velocity:.0f}px → suavizado extra")
                            elif velocity < 4:
                                f_center = 0.85
                                kps_ema = 0.80 if not is_profile else 0.75
                            elif velocity < 12:
                                f_center = 0.90
                                kps_ema = 0.85 if not is_profile else 0.82
                            else:
                                f_center = 0.95
                                kps_ema = 0.90 if not is_profile else 0.88
                        
                        f_size = 0.95
                        if plan_smooth:
                            f_size = 0.62
                            f_center = min(f_center, 0.82)
                    
                    curr_size = curr_bbox[2:] - curr_bbox[:2]
                    prev_size = prev_bbox_arr[2:] - prev_bbox_arr[:2]
                    size_ratio = float(np.max(curr_size) / max(1.0, np.max(prev_size)))
                    if size_ratio > 1.35 or size_ratio < 0.72:
                        f_size = min(f_size, 0.58 if plan_smooth else 0.72)
                        if call_num % 25 == 1:
                            print(f"[PLAN_SMOOTH] Frame {call_num}: zoom bbox ratio={size_ratio:.2f} → f_size={f_size:.2f}")
                    
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
            mouth_crop_origin = (0, 0)
            mouth_crop_size = None

            bx1, by1, bx2, by2 = target_face.bbox
            is_closeup = self._is_closeup_face(target_face.bbox, original_frame.shape)

            if preserve_mouth:
                try:
                    from roop.processors.FaceSwap import detect_mouth_open, create_mouth_preservation_mask
                except ImportError:
                    mouth_open = False
                else:
                    landmarks_106 = getattr(target_face, 'landmark_106', None)
                    # MediaPipe más fiable en crop facial (especialmente primeros planos)
                    mouth_probe = original_frame
                    if is_closeup:
                        pad = int(max(12, (by2 - by1) * 0.15))
                        cx1, cy1 = max(0, int(bx1) - pad), max(0, int(by1) - pad)
                        cx2, cy2 = min(original_frame.shape[1], int(bx2) + pad), min(original_frame.shape[0], int(by2) + pad)
                        if cx2 > cx1 and cy2 > cy1:
                            mouth_probe = original_frame[cy1:cy2, cx1:cx2]
                            mouth_crop_origin = (cx1, cy1)
                            mouth_crop_size = (mouth_probe.shape[1], mouth_probe.shape[0])
                    mouth_open, mouth_region, mouth_open_ratio = detect_mouth_open(
                        target_face, landmarks_106, mouth_probe,
                    )
                    
                    # v5.1024: EMA smoothing del ratio — elimina fluctuaciones frame-a-frame
                    if not hasattr(self, '_mouth_ratio_smooth'): self._mouth_ratio_smooth = {}
                    prev_ratio = self._mouth_ratio_smooth.get(video_key, mouth_open_ratio)
                    mouth_open_ratio_smooth = 0.75 * mouth_open_ratio + 0.25 * prev_ratio  # v5.1029: 0.60/0.40→0.75/0.25 — sigue más rápido el ratio crudo
                    self._mouth_ratio_smooth[video_key] = mouth_open_ratio_smooth
                    
                    # Histeresis temporal para evitar parpadeo de apertura/cierre
                    video_key = getattr(self, '_current_video_key', 'default')
                    if not hasattr(self, '_mouth_state_history'): self._mouth_state_history = {}
                    prev_state = self._mouth_state_history.get(video_key, False)
                    
                    # v5.1019: Histéresis más amplia — preservación solo con apertura real (elimina parpadeo)
                    if is_closeup:
                        thresh = 0.28 if is_profile else (0.18 if prev_state else 0.22)  # v5.1028: deactivate 0.10→0.18 para que pueda desactivarse
                    else:
                        thresh = 0.32 if is_profile else (0.20 if prev_state else 0.26)  # v5.1028: deactivate 0.13→0.20
                    mouth_open = mouth_open_ratio_smooth > thresh
                    # v5.1024: Hold sostenido — evita parpadeo al cerrar boca
                    if not hasattr(self, '_mouth_hold_count'): self._mouth_hold_count = {}
                    if mouth_open:
                        self._mouth_hold_count[video_key] = 4  # v5.1030: 10→4 — se desactiva tras ~0.13s de silencio
                    else:
                        hold = self._mouth_hold_count.get(video_key, 0)
                        if hold > 0:
                            self._mouth_hold_count[video_key] = hold - 1
                            mouth_open = True  # mantener activa durante el hold
                    self._mouth_state_history[video_key] = mouth_open
                    
                    if mouth_open:
                        print(f"[MOUTH_PRESERVE] Frame {call_num}: Boca activa (ratio={mouth_open_ratio:.2f} smooth={mouth_open_ratio_smooth:.2f}, closeup={is_closeup})")

            # ============================================
            # 5. COORDENADAS FINALES (v3.5)
            # ============================================
            x1, y1, x2, y2 = target_face.bbox
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(result_frame.shape[1], int(x2)), min(result_frame.shape[0], int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return original_frame

            face_w = x2 - x1
            face_h = y2 - y1
            is_small_face = min(face_w, face_h) < 120

            # v5.59: Skip-swap solo para tracking extremadamente malo — det_score < 0.05 + velocity > 100px
            det_score = getattr(target_face, 'det_score', 1.0)
            velocity = getattr(self, '_last_velocity', 0)
            is_ghost = getattr(target_face, 'is_ghost', False)
            lost_count = getattr(self, f'_tracking_lost_count_{video_basename}', 0)

            # Ghost tracking: mantener último swap (nunca parpadear a original en primer plano)
            if is_ghost:
                closeup_active = getattr(self, f'_closeup_active_{video_basename}', False)
                prev_frame_result = getattr(self, '_prev_frame_result', {}).get(video_key)
                last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                if prev_frame_result is not None and last_bbox is not None:
                    if call_num % 25 == 1:
                        print(
                            f"[TRACK] Frame {call_num}: ghost lost={lost_count} "
                            f"→ hold swap (closeup={closeup_active})"
                        )
                    hold_blend = 0.96 if closeup_active else 0.90
                    # v5.1001: usar bbox predicha de ghost_face (desplazada con dx,dy) en vez de last_bbox
                    if not hasattr(self, '_last_swap_bbox'):
                        self._last_swap_bbox = {}
                    self._last_swap_bbox[video_basename] = (x1, y1, x2, y2)
                    # v5.1002: prev_mask=None para forzar elipse en la NUEVA posicion (predicha),
                    # en vez de la mascara del frame anterior (posicion vieja que causa swap congelado)
                    return self._hold_previous_face_region(
                        original_frame, prev_frame_result, (x1, y1, x2, y2),
                        blend=hold_blend, prev_mask=None,
                    )
                fade_attr = f'_ghost_fade_alpha_{video_basename}'
                fade_alpha = getattr(self, fade_attr, None)
                if not self._should_hold_previous_swap(video_basename):
                    fade_start = 0.96 if closeup_active else 0.90
                    fade_step = 0.12 if closeup_active else 0.08  # v5.1027: 0.16/0.12→0.12/0.08 — más gradual (9-13 frames vs 6-8)
                    if fade_alpha is None:
                        fade_alpha = fade_start
                    elif fade_alpha > 0:
                        fade_alpha = max(0, fade_alpha - fade_step)
                    if fade_alpha > 0 and prev_frame_result is not None and last_bbox is not None:
                        setattr(self, fade_attr, fade_alpha)
                        if call_num % 10 == 1:
                            print(f"[TRACK] Frame {call_num}: ghost fading α={fade_alpha:.2f}")
                        return self._hold_previous_face_region(
                            original_frame, prev_frame_result, last_bbox,
                            blend=fade_alpha, prev_mask=None,
                        )
                    if call_num % 25 == 1:
                        print(f"[TRACK] Frame {call_num}: ghost post-corte → original")
                    return original_frame
                if call_num % 25 == 1:
                    print(f"[TRACK] Frame {call_num}: ghost sin histórico → original")
                return original_frame

            # v5.1012: Skip-hold con timeout reducido — después de 6 frames, fuerza swap nuevo
            skip_hold_count_attr = f'_skip_hold_count_{video_basename}'
            skip_hold_count = getattr(self, skip_hold_count_attr, 0)
            if det_score < 0.05 and velocity > 100:
                if skip_hold_count < 6:
                    prev_frame_result = getattr(self, '_prev_frame_result', {}).get(video_key)
                    last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                    if prev_frame_result is not None and last_bbox is not None:
                        setattr(self, skip_hold_count_attr, skip_hold_count + 1)
                        if call_num % 10 == 1:
                            print(f"[SKIP_HOLD] Frame {call_num}: hold #{skip_hold_count+1} (det_score={det_score:.2f})")
                        # v5.1001: usar bbox actual (la cara se movió, last_bbox está desfasada)
                        if not hasattr(self, '_last_swap_bbox'):
                            self._last_swap_bbox = {}
                        self._last_swap_bbox[video_basename] = (x1, y1, x2, y2)
                        # v5.1002: prev_mask=None para elipse en posicion actual, no mascara vieja
                        return self._hold_previous_face_region(
                            original_frame, prev_frame_result, (x1, y1, x2, y2), blend=0.82, prev_mask=None,
                        )
                    if call_num % 10 == 1:
                        print(f"[SKIP] Frame {call_num}: swap saltado (det_score={det_score:.2f}, vel={velocity:.0f}px)")
                    setattr(self, skip_hold_count_attr, skip_hold_count + 1)
                    return original_frame
                # Timeout: forzar swap nuevo aunque det_score sea bajo
                if call_num % 30 == 1:
                    print(f"[SKIP_HOLD_TIMEOUT] Frame {call_num}: timeout {skip_hold_count} holds → forzando swap")
                setattr(self, skip_hold_count_attr, 0)
                # Continuar con swap normal

            # ============================================
            # 1. IDENTITY INJECTION (v5.78: master + perfil/match)
            # ============================================
            swap_source_face = self._resolve_swap_embedding(source_face, target_face, is_profile=is_profile, video_basename=video_basename)

            # ============================================
            # 2. SWAP
            # ============================================
            res_data = self.processors["faceswap"].Run(swap_source_face, target_face, result_frame, paste_back=False)
            if res_data is None:
                with open(os.path.join(os.path.dirname(__file__), '..', 'debug_swap.log'), 'a') as lf:
                    lf.write(f"[SWAP_FAIL] Frame {call_num}: FaceSwap devolvió None\n")
                return original_frame
            
            # res_data puede ser (swapped_face, M) o solo swapped_face
            if isinstance(res_data, tuple):
                swapped_face_aligned, M = res_data
            else:
                swapped_face_aligned = res_data
                M = None # Algunos modelos lo hacen interno

            # v5.2.7: Guardar raw swap ANTES de GFPGAN para detección precisa de oclusiones
            raw_swapped_aligned = swapped_face_aligned.copy()

            # v5.1012: M-EMA responsive — mucho menos peso de matriz anterior para eliminar swap congelado
            if M is not None and enable_temporal_smoothing:
                m_attr = f'_m_ema_{video_basename}'
                prev_m = getattr(self, m_attr, None)
                if prev_m is not None:
                    m_diff = np.max(np.abs(M - prev_m))
                    # v5.1029: M-EMA responsive — 0.40→0.50 (60-50% new) para tracking vivo sin congelamiento
                    ema_thresh = max(40.0, face_w * 0.20, face_h * 0.20)
                    if m_diff > ema_thresh:
                        m_alpha = 0.25
                        if call_num % 25 == 1:
                            print(f"[M_EMA] Frame {call_num}: giro extremo (diff={m_diff:.1f}) — reset EMA")
                    elif m_diff < 5.0:
                        m_alpha = 0.40
                    elif m_diff < 15.0:
                        m_alpha = 0.40 + 0.10 * (m_diff - 5.0) / 10.0
                    else:
                        m_alpha = 0.50
                    if is_profile:
                        m_alpha *= 0.85
                    m_alpha = max(0.08, min(0.50, m_alpha))
                    M = M * (1.0 - m_alpha) + prev_m * m_alpha
                else:
                    m_diff = 0.0
                # Guardar m_diff para EMA condicional en otros subsistemas
                setattr(self, f'_m_diff_{video_basename}', m_diff)
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
            
            use_enhancer = getattr(roop.globals, 'use_enhancer', True)
            selected_enhancer = getattr(roop.globals, 'selected_enhancer', 'GFPGAN') or 'GFPGAN'
            gfpgan_mix, color_match_strength, brightness_strength = self._source_identity_params(
                is_profile=is_profile, is_closeup=is_closeup,
            )
            preserve_mouth = getattr(roop.globals, 'preserve_mouth_expression', True)
            
            enh_map = {
                "GFPGAN": "enhance_gfpgan",
                "CodeFormer": "enhance_codeformer",
                "Restoreformer++": "enhance_restoreformer",
            }
            enhancer_key = enh_map.get(selected_enhancer)
            if enhancer_key and enhancer_key not in self.processors:
                enhancer_key = next((k for k in ["enhance_gfpgan", "enhance_codeformer", "enhance_restoreformer"] if k in self.processors), None)
            if use_enhancer and enhancer_key is not None:
                # v5.66: Saltar enhancer en perfiles de baja detección (evita alucinaciones GFPGAN)
                # v5.1024: Umbral bajado 0.45→0.20 — perfiles con det_score 0.20-0.45 ahora reciben enhancer
                det_score_enh = getattr(target_face, 'det_score', 1.0)
                skip_enhancer = is_profile and det_score_enh < 0.08  # v5.1032: 0.10→0.08 — casi todos los perfiles reciben enhancer
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
                                    enh_ema_alpha = 0.50 if is_profile else 0.55
                                    # v5.1024: EMA condicional — movimiento rápido reduce inercia
                                    cur_m_diff = getattr(self, f'_m_diff_{video_basename}', 0)
                                    if cur_m_diff > 10.0:
                                        enh_ema_alpha = min(0.85, enh_ema_alpha + 0.25)
                                    enhanced = cv2.addWeighted(enhanced, enh_ema_alpha, prev_enh, 1.0 - enh_ema_alpha, 0)
                                self._enhancer_ema[video_key] = enhanced.copy()
                            
                            # v5.77: GFPGAN solo en núcleo; bordes y centro priorizan swap crudo (source)
                            h_f, w_f = enhanced.shape[:2]
                            Y_f, X_f = np.ogrid[:h_f, :w_f]
                            center_f = (w_f / 2, h_f / 2)
                            dist_f = np.sqrt((X_f - center_f[0])**2 + (Y_f - center_f[1])**2)
                            fade_radius = 52 if is_closeup else 58
                            fade = 1.0 - 1.0 / (1.0 + np.exp(-0.12 * (dist_f - fade_radius)))
                            fade_3ch = np.stack([fade, fade, fade], axis=-1).astype(np.float32)
                            alpha = gfpgan_mix * fade_3ch
                            blended_face = enhanced.astype(np.float32) * alpha + swapped_face_aligned.astype(np.float32) * (1.0 - alpha)
                            swapped_face_aligned = np.clip(blended_face, 0, 255).astype(np.uint8)
                            
                            if call_num <= 3:
                                cv2.imwrite(os.path.join(debug_dir, f'02_after_enhancer_f{call_num}.png'), swapped_face_aligned)
                            
                            # v5.1024: Unsharp adaptativo — más nitidez en closeups (más resolución disponible)
                            face_h_us = max(1, y2 - y1)
                            if is_closeup and face_h_us > 250:
                                unsharp_amt = 2.0
                                unsharp_sigma = 1.0
                            elif is_closeup:
                                unsharp_amt = 1.8
                                unsharp_sigma = 0.7
                            else:
                                unsharp_amt = 1.5
                                unsharp_sigma = 0.5
                            blurred = cv2.GaussianBlur(swapped_face_aligned, (0, 0), unsharp_sigma)
                            swapped_face_aligned = cv2.addWeighted(swapped_face_aligned, unsharp_amt, blurred, -(unsharp_amt - 1.0), 0)
                            if call_num % 50 == 1:
                                print(f"[QUALITY] Source-first GFPGAN mix={gfpgan_mix:.2f} color={color_match_strength:.2f} unsharp={unsharp_amt:.1f}")
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

            # Estabilización de Brillo EMA (baja = más tono del swap/source)
            if enable_temporal_smoothing and video_key:
                if not hasattr(self, '_brightness_ema'): self._brightness_ema = {}
                prev_bright = self._brightness_ema.get(video_key, 1.0)
                
                curr_bright_ratio = np.mean(cv2.cvtColor(reference_region, cv2.COLOR_BGR2GRAY)) / (np.mean(cv2.cvtColor(swapped_face_aligned, cv2.COLOR_BGR2GRAY)) + 1e-6)
                curr_bright_ratio = np.clip(curr_bright_ratio, 0.7, 1.3)
                
                # v5.1024: EMA condicional — movimiento rápido da más peso al brillo actual
                cur_m_diff = getattr(self, f'_m_diff_{video_basename}', 0)
                bright_ema_new = 0.80 if cur_m_diff > 10.0 else 0.45
                smooth_bright = bright_ema_new * curr_bright_ratio + (1.0 - bright_ema_new) * prev_bright
                self._brightness_ema[video_key] = smooth_bright
                
                face_lab = cv2.cvtColor(swapped_face_aligned, cv2.COLOR_BGR2LAB).astype(np.float32)
                adj = 1.0 + (smooth_bright - 1.0) * brightness_strength
                face_lab[:, :, 0] = np.clip(face_lab[:, :, 0] * adj, 0, 255)
                swapped_face_aligned = cv2.cvtColor(face_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

            swapped_face_aligned = match_color_histogram(swapped_face_aligned, reference_region, blend_factor=color_match_strength)

            if enable_temporal_smoothing and video_key:
                if not hasattr(self, '_color_ema'): self._color_ema = {}
                prev_color = self._color_ema.get(video_key)
                if prev_color is not None and prev_color.shape == swapped_face_aligned.shape:
                    # v5.95: Color EMA más responsivo — 42% nuevo (antes 28%) para reducir efecto congelado
                    ui_identity = float(getattr(roop.globals, 'enhancer_blend_factor', 0.92))
                    color_ema_alpha = 0.72 if is_profile else (0.58 if ui_identity >= 0.85 else 0.65)
                    # v5.1024: EMA condicional — movimiento rápido reduce inercia
                    cur_m_diff = getattr(self, f'_m_diff_{video_basename}', 0)
                    if cur_m_diff > 10.0:
                        color_ema_alpha = min(0.90, color_ema_alpha + 0.20)
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
                
                # A. Warp de la cara — perfil/closeup usan CONSTANT+fill (REFLECT deja marco en bordes)
                warp_border = cv2.BORDER_CONSTANT if (is_profile or is_closeup) else cv2.BORDER_REFLECT
                warped_face = cv2.warpAffine(
                    swapped_face_aligned, M_inv, (w_f, h_f), borderMode=warp_border, borderValue=0,
                )

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
                # v5.79: Caras pequeñas saltan XSeg (máscara muy reducida → mosaico)
                skip_xseg = is_small_face
                if skip_xseg and call_num <= 3:
                    print(f"[MASK_SMALL] Frame {call_num}: cara {face_w}x{face_h} <120px → elipse (evita mosaico)")
                if "mask_xseg" in self.processors and not skip_xseg:
                    try:
                        # La máscara XSeg se genera sobre la cara alineada (calidad nativa 256x256)
                        xseg_mask = self.processors["mask_xseg"].Run(swapped_face_aligned, "")
                        if xseg_mask is not None and np.max(xseg_mask) > 0:
                            xseg_thresh = 0.18 if is_profile else 0.22
                            if is_profile or is_closeup:
                                xseg_mask = np.clip((xseg_mask.astype(np.float32) - xseg_thresh) / max(0.05, 1.0 - xseg_thresh), 0, 1)
                                xseg_mask = cv2.GaussianBlur(xseg_mask, (17, 17), 0)
                            else:
                                _, xseg_mask = cv2.threshold(xseg_mask, xseg_thresh, 1.0, cv2.THRESH_BINARY)

                            # v5.1018: Eliminada atenuación superior — causaba "efecto flequillo"
                            # Warp de la máscara XSeg al espacio del frame
                            final_mask = cv2.warpAffine(xseg_mask, M_inv, (w_f, h_f), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    except Exception as e:
                        print(f"[MASK_ERR] XSeg falló: {e}")

                # C. FALLBACK: Si XSeg no está o falló, usar elipse dinámica mejorada
                if final_mask is None:
                    mask_align = np.zeros((h_align, w_align), dtype=np.float32)
                    if is_profile:
                        # v5.73: Elipse perfil expandida 0.65→0.75 para mayor identidad source
                        cv2.ellipse(mask_align, (w_align//2, h_align//2),
                                    (int(w_align*0.75), int(h_align*0.75)), 0, 0, 360, 1.0, -1)
                        mask_align = cv2.GaussianBlur(mask_align, (21, 21), 0)
                    else:
                        # v5.73: Elipse frontal expandida 0.60→0.70 para mayor identidad source
                        cv2.ellipse(mask_align, (w_align//2, h_align//2),
                                    (int(w_align*0.70), int(h_align*0.70)), 0, 0, 360, 1.0, -1)
                    
                    # v5.1018: Eliminada atenuación superior en fallback — causaba "efecto flequillo"
                    final_mask = cv2.warpAffine(mask_align, M_inv, (w_f, h_f), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

                # D. Aplicar oclusión (v5.2: Independiente del blend ratio para proteger objetos siempre)
                occ_mask_aligned = detect_foreground_occlusion(raw_swapped_aligned, reference_region)
                
                # v5.79: Oclusión adaptativa — más suave en caras pequeñas
                occ_strength = (0.35 if is_small_face else (0.50 if is_profile else 0.60))  # v5.1028: small_face 0.25→0.35
                
                if np.max(occ_mask_aligned) > 0.1:
                    occ_mask_frame = cv2.warpAffine(occ_mask_aligned, M_inv, (w_f, h_f), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    occ_mask_frame = cv2.GaussianBlur(occ_mask_frame, (31, 31), 0)
                    # v5.1022: EMA único y rápido — 0.80 nuevo elimina ghosting en 2 frames
                    prev_occ_frame = getattr(self, '_occ_frame_ema', {}).get(video_key)
                    if prev_occ_frame is not None and prev_occ_frame.shape == occ_mask_frame.shape:
                        occ_mask_frame = cv2.addWeighted(occ_mask_frame, 0.80, prev_occ_frame, 0.20, 0)
                    self._occ_frame_ema[video_key] = occ_mask_frame.copy()
                    reduction = occ_mask_frame * occ_strength
                    # v5.1022b: Cap 0.20 evita flash breve sin causar ghosting
                    reduction = np.minimum(reduction, 0.35)  # v5.1028: 0.55→0.35 — evita colapso de máscara por safety recovery
                    final_mask = np.clip(final_mask - reduction, 0, 1.0)
                    if call_num % 100 == 1:
                        print(f"[QUALITY] Oclusión aplicada (fuerza={occ_strength:.2f})")

                # v5.78: Erosión adaptativa — más suave en perfil / boca abierta / primer plano
                ekernel = max(3, min(face_w, face_h) // 32) | 1
                if is_profile:
                    ekernel = max(3, ekernel - 1)  # v5.993: -2→-1 — menos contracción = menos marco visible en perfiles
                if is_closeup and min(face_w, face_h) >= 180:
                    ekernel = max(3, ekernel - 2)
                if mouth_open and mouth_open_ratio <= 0.45:
                    ekernel = max(3, ekernel - 1)
                blur_sz = int(max(5 if is_closeup else 3, min(face_w, face_h) // (85 if is_closeup else 75))) | 1
                # v5.1029: Caras muy pequeñas (<60px) saltan erosión — la elipse warpeada se destruye
                is_very_small_face = min(face_w, face_h) < 60
                if is_profile or is_closeup or is_very_small_face:
                    # v5.995: sin erosión en perfil/closeup (causa marco). Dilatación mínima + blur + feather.
                    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    final_mask = cv2.dilate(final_mask, dilate_k, iterations=1)
                    blur_sz = max(blur_sz, (min(face_w, face_h) // (7 if is_profile else 8)) | 1)
                    blur_sz = min(blur_sz, 41)  # v5.1023: 27→41 — mayor suavizado en perfiles
                    final_mask = cv2.GaussianBlur(final_mask, (blur_sz, blur_sz), 0)
                    final_mask = self._feather_mask_edges(final_mask, feather_px=max(10, blur_sz // 2))
                else:
                    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ekernel, ekernel))
                    final_mask = cv2.erode(final_mask, kernel_e, iterations=1)
                    final_mask = cv2.GaussianBlur(final_mask, (blur_sz, blur_sz), 0)

                # v5.83/94/993: Sin content-feather en elipse pura, primer plano o perfil — evita bloque/marco
                if not skip_xseg and not is_closeup and not is_profile:
                    content_blur_sz = int(max(13, min(25, blur_sz + 3))) | 1
                    tent = np.clip(1.0 - np.abs(final_mask - 0.5) * 2.0, 0, 1.0)
                    warp_gate = (np.max(warped_face.astype(np.float32), axis=2) > 8).astype(np.float32)
                    warp_gate = cv2.GaussianBlur(warp_gate, (9, 9), 0)
                    tent = tent * warp_gate
                    if np.max(tent) > 0.01:
                        blurred_face = cv2.GaussianBlur(warped_face, (content_blur_sz, content_blur_sz), 0)
                        tent_3ch = cv2.merge([tent, tent, tent])
                        warped_face = (warped_face.astype(np.float32) * (1.0 - tent_3ch) + blurred_face.astype(np.float32) * tent_3ch).astype(np.uint8)

                # v5.75: Tail truncation mínimo 0.005 — solo elimina ruido, no mata caras pequeñas
                final_mask = np.clip((final_mask - 0.005) / (1.0 - 0.005), 0, 1.0)

                # v5.1008: Consistencia temporal de máscara — 30% del frame anterior
                # suaviza fluctuaciones de XSeg entre frames consecutivos que causan parpadeo
                if enable_temporal_smoothing and video_key:
                    prev_final_mask = getattr(self, '_prev_final_mask', {}).get(video_key)
                    if (prev_final_mask is not None and prev_final_mask.shape == final_mask.shape
                        and not self._mask_needs_safety_recovery(prev_final_mask, face_w, face_h)
                        and not self._mask_needs_safety_recovery(final_mask, face_w, face_h)):
                        final_mask = cv2.addWeighted(
                            final_mask.astype(np.float32), 0.72,
                            prev_final_mask.astype(np.float32), 0.28, 0,
                        )
            else:
                print("[WARNING] M es None, usando fallback elíptico (sin paste rectangular)")
                warped_face = result_frame.copy()
                resized_face = cv2.resize(swapped_face_aligned, (face_w, face_h))
                roi_mask = np.zeros((face_h, face_w), dtype=np.float32)
                cv2.ellipse(
                    roi_mask,
                    (face_w // 2, face_h // 2),
                    (max(4, int(face_w * 0.44)), max(4, int(face_h * 0.48))),
                    0, 0, 360, 1.0, -1,
                )
                roi_mask = cv2.GaussianBlur(roi_mask, (21, 21), 0)
                roi_mask_3 = cv2.merge([roi_mask, roi_mask, roi_mask])
                roi_orig = warped_face[y1:y2, x1:x2].astype(np.float32)
                warped_face[y1:y2, x1:x2] = (
                    resized_face.astype(np.float32) * roi_mask_3 + roi_orig * (1.0 - roi_mask_3)
                ).astype(np.uint8)
                final_mask = create_soft_mask((x1, y1, x2, y2), (h_f, w_f), feather=40)

            mask_before_mouth = final_mask.copy() if final_mask is not None else None
            mouth_force_paste = False
            mouth_paste_strength = 0.0

            # ============================================
            # 5c. PRESERVACIÓN DE BOCA — con suelo de máscara (v5.79)
            # v5.1026: En fotos, saltar preservación para máxima identidad source
            # ============================================
            if mouth_open and mouth_region is not None and mask_before_mouth is not None:
                mouth_occ_score = 0.0
                mouth_content_score = 0.0
                if is_profile:
                    m_blend = 0.46  # v5.1023: más protección oral en perfil
                elif is_closeup:
                    m_blend = 0.60  # v5.1025: 0.50→0.60 — preservación oral dominante en primer plano
                else:
                    m_blend = 0.55  # v5.1025: 0.44→0.55 — más original en boca siempre
                if mouth_open_ratio > 0.28:
                    oral_strength = min(1.0, (mouth_open_ratio - 0.28) / 0.52)
                    m_blend = min(0.88, m_blend * (1.0 + oral_strength * 0.80))  # v5.1025: cap 0.78→0.88 — más original en bocas muy abiertas
                if is_small_face:
                    m_blend *= 0.80  # v5.1024b: 0.65→0.80 — menos penalización doble (small_face × low_mask hundía a 35%)
                if mask_before_mouth.mean() < 0.04:
                    m_blend *= 0.85  # v5.1024b: 0.75→0.85 — restore mínimo ~46% vs ~35% antes
                # v5.1033: perfil con match bajo → más original (swap poco fiable)
                if is_profile:
                    prof_sim = getattr(self, '_last_profile_match_sim', 0.0)
                    if prof_sim < 0.20:
                        m_blend = max(m_blend, 0.65)

                try:
                    mouth_mask_aligned = create_mouth_preservation_mask(reference_region, mouth_region, blend_ratio=1.0)
                    mouth_occ_score = float(np.mean(cv2.bitwise_and(occ_mask_aligned, mouth_mask_aligned)))
                    # v5.998: Solo computar content_diff en boca abierta (>0.25) para evitar
                    # que la diferencia natural entre dos personas distintas dispare el paste.
                    if mouth_open_ratio > 0.25:
                        mouth_content_score = self._detect_mouth_content_diff(raw_swapped_aligned, reference_region)
                    else:
                        mouth_content_score = 0.0
                    mouth_signal = max(mouth_occ_score, mouth_content_score * 0.80)  # v5.1035: 0.50→0.80 — más peso al contenido oral
                    if mouth_signal > 0.004:
                        occ_strength = min(1.0, mouth_signal * 12.0)
                        m_blend = min(0.75, m_blend * (1.0 + occ_strength * 0.65))  # v5.1025: 0.65→0.75 — menos cap en boca con contenido oral
                        # Umbral más alto para evitar paste en diferencias faciales naturales
                        if mouth_content_score > 0.020 or mouth_occ_score > 0.015:  # v5.1035: 0.030→0.020 — detecta objetos más sutiles en boca
                            mouth_force_paste = True
                            mouth_paste_strength = min(
                                0.96,
                                0.82 + mouth_content_score * 0.55 + mouth_occ_score * 1.0,
                            )
                        if call_num % 50 == 1:
                            print(
                                f"[MOUTH_OBJECT] Contenido oral (occ={mouth_occ_score:.3f}, "
                                f"diff={mouth_content_score:.3f}) → restore={m_blend:.2f}, "
                                f"paste={mouth_paste_strength:.2f}"
                            )
                except Exception as e:
                    print(f"[MOUTH_ERR] preservación: {e}", flush=True)

                m_blend = float(np.clip(m_blend, 0.0, 0.92))
                apply_mouth_preserve = m_blend >= 0.10

                if apply_mouth_preserve:
                    mouth_mask = self._mouth_mask_full_frame(
                        original_frame, mouth_region, mouth_crop_origin, mouth_crop_size,
                    )
                    mouth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    mouth_mask = cv2.dilate(mouth_mask, mouth_kernel, iterations=1)
                    mouth_mask = cv2.GaussianBlur(mouth_mask, (15, 15), 0)

                    m_impact = (mouth_mask * m_blend).mean()
                    # v5.1024: TWO-PASS COMPOSITING — NO restamos mouth_mask de final_mask
                    # para no colapsar la máscara en caras pequeñas. En vez de eso,
                    # guardamos mouth_mask y m_blend para aplicarlos como blend en el
                    # paso de composición final (sección 7), evitando el safety recovery.
                    self._mouth_blend_mask = mouth_mask.copy()
                    self._mouth_blend_factor = m_blend
                    # Conservamos min_mouth_residual por si safety recovery se activa por otra razón
                    min_mouth_residual = 0.10 * mouth_mask
                    final_mask = np.maximum(final_mask, min_mouth_residual)

                    if call_num % 50 == 1:
                        print(
                            f"[QUALITY] Boca (ratio={mouth_open_ratio:.2f}, occ={mouth_occ_score:.3f}, "
                            f"impacto={m_impact:.3f}, restore={m_blend*100:.0f}%)"
                        )
                elif call_num % 50 == 1:
                    print(
                        f"[MOUTH_SKIP] Frame {call_num}: expresión oral fuerte "
                        f"(ratio={mouth_open_ratio:.2f}) — swap visible, oclusión activa"
                    )

            # v5.1027: Broad face floor to prevent MASK_SAFETY regeneration
            # The mouth-only residual (0.10 every mouth_mask) leaves the rest of
            # the face unboosted. XSeg masks for small faces can have max < 0.25,
            # triggering safety recovery ellipse which creates visible artifacts.
            if final_mask is not None:
                if final_mask.max() < 0.45:
                    cx_b, cy_b = (x1 + x2) // 2, (y1 + y2) // 2
                    rx_b, ry_b = max(16, (x2 - x1)//2), max(16, (y2 - y1)//2)
                    broad_floor = ProcessMgr._build_ellipse_mask(
                        final_mask.shape, cx_b, cy_b, rx_b, ry_b, tight=False,
                    )
                    broad_floor = np.clip(broad_floor * 0.65, 0, 0.65)  # v5.1028: 0.55→0.65 — piso más fuerte para caras pequeñas
                    final_mask = np.maximum(final_mask, broad_floor)

            # ============================================
            # 5b. SAFETY FALLBACK — cobertura rota → regenerar elipse (v5.83)
            # ============================================
            # v5.1008: Histeresis safety recovery — evita parpadeo ON/OFF entre frames
            hyster_attr = f'_recovery_hysteresis_{video_basename}'
            hyster_count = getattr(self, hyster_attr, 0)
            mask_needs_it = self._mask_needs_safety_recovery(final_mask, face_w, face_h) if final_mask is not None else True
            if mask_needs_it:
                hyster_count = 3  # reset: necesitamos 3 frames de recuperacion
            elif hyster_count > 0 and final_mask is not None and final_mask.max() < 0.40:
                hyster_count -= 1  # mantener recuperacion hasta que pase la histeresis
                mask_needs_it = True
            setattr(self, hyster_attr, hyster_count)
            M_inv_safe = cv2.invertAffineTransform(M) if M is not None else None
            if final_mask is not None and mask_needs_it:
                recovered = False
                if mask_before_mouth is not None and not self._mask_needs_safety_recovery(mask_before_mouth, face_w, face_h):
                    final_mask = cv2.GaussianBlur(mask_before_mouth, (31, 31), 0)
                    final_mask = np.clip(final_mask * 0.88, 0.08, 1.0)
                    recovered = True
                    if call_num % 20 == 1:
                        print(f"[MASK_RECOVER] Frame {call_num}: pre-boca (mean={final_mask.mean():.3f}, max={final_mask.max():.3f})")
                else:
                    regen = self._regenerate_face_safety_mask(
                        original_frame.shape[:2],
                        swapped_face_aligned,
                        M_inv_safe,
                        bbox=(x1, y1, x2, y2),
                        tight=is_closeup or is_profile,
                    )
                    if regen is not None:
                        prev_swap_mask = getattr(self, '_last_swap_mask', {}).get(video_basename)
                        if (
                            prev_swap_mask is not None
                            and prev_swap_mask.shape == regen.shape
                            and (is_profile or is_closeup)
                        ):
                            # v5.997: reducir peso previo (0.42→0.15) para evitar ghosting
                            # cuando el frame anterior también fue safety fallback
                            final_mask = cv2.addWeighted(
                                regen.astype(np.float32), 0.85,
                                prev_swap_mask.astype(np.float32), 0.15, 0,
                            )
                        else:
                            final_mask = regen
                        recovered = True
                if recovered and self._mask_needs_safety_recovery(final_mask, face_w, face_h):
                    if is_profile or is_closeup:
                        sb = max(15, (min(face_w, face_h) // 6) | 1)
                        final_mask = cv2.GaussianBlur(final_mask, (sb, sb), 0)
                        final_mask = np.clip(final_mask * 1.04, 0.08, 1.0)
                        final_mask = self._feather_mask_edges(final_mask, feather_px=sb // 2)
                        if (
                            self._mask_needs_safety_recovery(final_mask, face_w, face_h)
                            and mask_before_mouth is not None
                        ):
                            final_mask = cv2.GaussianBlur(mask_before_mouth, (35, 35), 0)
                            final_mask = np.clip(final_mask * 0.92, 0.10, 1.0)
                            final_mask = self._feather_mask_edges(final_mask, feather_px=sb // 2)
                            if call_num % 20 == 1:
                                print(f"[MASK_SAFETY] Frame {call_num}: perfil/closeup pre-boca blur (sin bbox)")
                    if self._mask_needs_safety_recovery(final_mask, face_w, face_h) and not is_profile:
                        boost = 1.30 if min(face_w, face_h) < 120 else 1.20  # v5.1026: 1.18/1.08→1.30/1.20 — elipse más grande para transición suave
                        final_mask = self._build_bbox_safety_mask(
                            original_frame.shape[:2], x1, y1, x2, y2, scale=boost, tight=False,
                        )
                        if self._mask_needs_safety_recovery(final_mask, face_w, face_h):
                            final_mask = self._build_bbox_safety_mask(
                                original_frame.shape[:2], x1, y1, x2, y2, scale=boost * 1.20, tight=False,
                            )
                        if call_num % 20 == 1:
                            cov = self._mask_face_coverage(final_mask)
                            print(f"[MASK_SAFETY] Frame {call_num}: fallback bbox (mean={final_mask.mean():.3f}, cov={cov}, max={final_mask.max():.3f})")
                    elif (
                        self._mask_needs_safety_recovery(final_mask, face_w, face_h)
                        and is_closeup and not is_profile
                    ):
                        boost = 0.92
                        final_mask = self._build_bbox_safety_mask(
                            original_frame.shape[:2], x1, y1, x2, y2, scale=boost, tight=True,
                        )
                        if call_num % 20 == 1:
                            cov = self._mask_face_coverage(final_mask)
                            print(f"[MASK_SAFETY] Frame {call_num}: closeup bbox tight (mean={final_mask.mean():.3f}, cov={cov})")
                elif recovered and call_num % 20 == 1:
                    cov = self._mask_face_coverage(final_mask)
                    print(f"[MASK_SAFETY] Frame {call_num}: elipse OK (mean={final_mask.mean():.3f}, cov={cov}, max={final_mask.max():.3f})")
                # v5.1002: widen ya aplicado globalmente en blend_mask (linea 3335)
                # ya no se necesita widen adicional en final_mask post-recovery
                if recovered and is_profile:
                    prev_swap_mask = getattr(self, '_last_swap_mask', {}).get(video_basename)
                    if prev_swap_mask is not None and prev_swap_mask.shape == final_mask.shape:
                        final_mask = cv2.addWeighted(final_mask, 0.84, prev_swap_mask.astype(np.float32), 0.16, 0)
                    final_mask = self._feather_mask_edges(
                        final_mask, feather_px=max(10, (min(face_w, face_h) // 10)),
                    )
                elif recovered and not is_profile:
                    final_mask = self._feather_mask_edges(
                        final_mask, feather_px=max(14, (min(face_w, face_h) // 8)),
                    )
                if recovered and mouth_open and mouth_region is not None:
                    try:
                        mouth_mask_safe = self._mouth_mask_full_frame(
                            original_frame, mouth_region, mouth_crop_origin, mouth_crop_size,
                        )
                        mouth_kernel_safe = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                        mouth_mask_safe = cv2.dilate(mouth_mask_safe, mouth_kernel_safe, iterations=1)
                        mouth_mask_safe = cv2.GaussianBlur(mouth_mask_safe, (11, 11), 0)
                        m_blend_safe = 0.22 if is_profile else 0.26
                        if mouth_open_ratio > 0.45:
                            m_blend_safe *= 0.45
                        if is_small_face:
                            m_blend_safe *= 0.65
                        trial = np.clip(final_mask - mouth_mask_safe * m_blend_safe, 0, 1.0)
                        if not self._mask_needs_safety_recovery(trial, face_w, face_h):
                            final_mask = trial
                    except Exception as e:
                        print(f"[MASK_ERR] safety mouth mask: {e}", flush=True)
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
            # v5.1002: widen_mask SIEMPRE en todas las caras — erosion+blur minimo
            # (ekernel=3, blur_sz=3) produce transicion ~1.5px que causa marco visible.
            # Widen_mask genera transicion >15px eliminando el borde.
            if final_mask is None:
                return original_frame
            blend_mask = self._widen_mask_for_blend(final_mask, face_w, face_h, original_frame.shape)
            # v5.1024: Gradiente frontal suave sobre blend_mask (post-widening)
            blend_mask = self._apply_forehead_gradient(
                blend_mask, original_frame.shape[:2], target_face, y1, face_h, is_profile,
            )
            mask_3ch = cv2.merge([blend_mask, blend_mask, blend_mask])
            
            # Atenuación por perfil, velocidad o "ghosting"
            velocity = getattr(self, '_last_velocity', 0)
            
            # v5.1.4: Bloquear swap_weight a 1.0 si el usuario lo pide (máxima potencia)
            swap_weight = 1.0 if user_blend >= 0.98 else user_blend
            
            # v5.81: Atenuación extra por baja calidad general del video
            avg_det_w = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
            if avg_det_w < 0.30:
                swap_weight *= 0.85
            elif avg_det_w < 0.40:
                swap_weight *= 0.92
            
            if is_profile:
                # Perfil: peso completo excepto cuando tracking es inestable (bajo det_score o alta velocidad)
                if velocity > 80:
                    swap_weight *= max(0.5, 1.0 - (velocity - 80) / 200)
            
            if not is_profile and velocity > 100:
                swap_weight *= max(0.6, 1.0 - (velocity - 100) / 200)

            mask_3ch = mask_3ch * swap_weight

            if blend_mask is not None:
                warped_face = self._prime_warp_edges(
                    warped_face, original_frame, blend_mask, face_w, face_h,
                )

            # v5.1027: Merge mouth blend into main composite mask (single pass)
            # para eliminar la transición visible del two-pass. La máscara 3CH
            # incluye la reducción oral: donde mouth_mask > 0, mask_3ch se reduce
            # según mouth_blend_factor, mostrando más original en la boca.
            mouth_blend_mask_ref = getattr(self, '_mouth_blend_mask', None)
            mouth_blend_factor_ref = getattr(self, '_mouth_blend_factor', 0.0)
            composite_mask = mask_3ch.copy() if mask_3ch is not None else mask_3ch
            if mouth_blend_mask_ref is not None and mouth_blend_factor_ref > 0.01:
                mouth_fr = cv2.resize(mouth_blend_mask_ref, (mask_3ch.shape[1], mask_3ch.shape[0]))
                mouth_fr_3 = cv2.merge([mouth_fr, mouth_fr, mouth_fr])
                composite_mask = composite_mask * (1.0 - mouth_fr_3 * mouth_blend_factor_ref)
                self._mouth_blend_mask = None
                self._mouth_blend_factor = 0.0

            result_frame = (warped_face.astype(np.float32) * composite_mask + 
                           original_frame.astype(np.float32) * (1.0 - composite_mask)).astype(np.uint8)

            # v5.76: Suavizado temporal SOLO en máscara facial (nunca en todo el frame)
            if enable_temporal_smoothing and prev_frame_result is not None and not is_ghost:
                if prev_frame_result.shape == result_frame.shape:
                    det_score = getattr(target_face, 'det_score', 1.0)
                    avg_det_t = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
                    # v5.95: Alpha reducido ~40% para menor inercia temporal — más responsivo a cambios de expresión/giro
                    if avg_det_t < 0.30:
                        alpha_prev = 0.22 if det_score < 0.4 else 0.16
                    elif det_score < 0.3:
                        alpha_prev = 0.18
                    elif det_score < 0.4:
                        alpha_prev = 0.14
                    elif det_score < 0.6:
                        alpha_prev = 0.10  # v5.1009: same for profile/non-profile
                    else:
                        alpha_prev = 0.10  # v5.1011: 0.10 (was 0.06)
                    # v5.1012: Profile caps raised significantly for 2-3x temporal smoothing vs non-profile
                    # v5.1024: EMA condicional — movimiento rápido reduce drásticamente alpha_prev
                    cur_m_diff = getattr(self, f'_m_diff_{video_basename}', 0)
                    if cur_m_diff > 10.0:
                        alpha_prev = min(alpha_prev, 0.06)  # solo 6% del frame anterior en movimiento rápido
                    elif is_profile and is_closeup:
                        alpha_prev = min(alpha_prev, 0.12)  # v5.1028: 0.18→0.12 — menos inercia temporal en perfiles
                    elif is_profile and velocity > 25:
                        alpha_prev = min(alpha_prev, 0.14)  # v5.1028: 0.20→0.14
                    elif is_profile:
                        alpha_prev = min(alpha_prev, 0.16)  # v5.1028: 0.28→0.16 — mismo rango que no-perfil
                    frame_count_ema = getattr(self, f'_frame_count_{video_basename}', 0)
                    if frame_count_ema <= getattr(self, f'_plan_change_smooth_until_{video_basename}', 0):
                        alpha_prev = min(alpha_prev, 0.10)  # v5.1011: 0.10 (was 0.06)
                    # v5.82: EMA alto + máscara débil = mosaico; reducir mezcla temporal
                    if final_mask is not None and self._mask_needs_safety_recovery(final_mask, face_w, face_h):
                        alpha_prev = min(alpha_prev, 0.08 if is_closeup else 0.10)
                    if is_closeup and not is_profile:
                        alpha_prev = min(alpha_prev, 0.10)  # v5.1011: 0.10 (was 0.06)
                    result_frame = self._masked_temporal_blend(
                        result_frame, prev_frame_result, final_mask, alpha_prev,
                    )
                    if call_num % 50 == 1:
                        print(f"[QUALITY] EMA masked (det={det_score:.2f}, avg_det={avg_det_t:.2f}, alpha={alpha_prev:.2f})")

            if mouth_force_paste and mouth_region is not None and mouth_paste_strength > 0.01:
                try:
                    paste_mask = self._mouth_mask_full_frame(
                        original_frame, mouth_region, mouth_crop_origin, mouth_crop_size,
                    )
                    paste_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                    paste_mask = cv2.dilate(paste_mask, paste_kernel, iterations=2)
                    paste_mask = cv2.GaussianBlur(paste_mask, (41, 41), 0)  # v5.1027: (31,31)→(41,41) — transición aún más ancha para eliminar borde de pasta
                    paste_mask = np.clip(paste_mask * mouth_paste_strength, 0, 1.0)
                    paste_3ch = cv2.merge([paste_mask, paste_mask, paste_mask])
                    result_frame = (
                        original_frame.astype(np.float32) * paste_3ch
                        + result_frame.astype(np.float32) * (1.0 - paste_3ch)
                    ).astype(np.uint8)
                    if call_num % 50 == 1:
                        print(
                            f"[MOUTH_PASTE] Frame {call_num}: píxeles originales en boca "
                            f"(strength={mouth_paste_strength:.2f})"
                        )
                except Exception as e:
                    print(f"[MOUTH_PASTE_ERR] Frame {call_num}: {e}", flush=True)

            if blend_mask is not None:
                result_frame = self._soothe_swap_ring(
                    result_frame, original_frame, blend_mask, face_w, face_h,
                )

            if call_num <= 3:
                cv2.imwrite(os.path.join(debug_dir, f'04_final_result_f{call_num}.png'), result_frame)

            mask_mean = final_mask.mean() if final_mask is not None else -1
            mask_max = final_mask.max() if final_mask is not None else -1
            if self._mask_needs_safety_recovery(final_mask, face_w, face_h) and call_num % 20 == 1:
                cov = self._mask_face_coverage(final_mask)
                print(f"[MASK_WARN] Frame {call_num}: cobertura crítica (mean={mask_mean:.4f}, cov={cov}, max={mask_max:.3f})")
            with open(os.path.join(os.path.dirname(__file__), '..', 'debug_swap.log'), 'a') as lf:
                lf.write(f"[SWAP_END] Frame {call_num}: resultado devuelto. Mask mean={mask_mean:.3f} max={mask_max:.3f}\n")

            # Guardar estado para el siguiente frame
            if enable_temporal_smoothing and video_key:
                if not hasattr(self, '_prev_frame_result'): self._prev_frame_result = {}
                self._prev_frame_result[video_key] = result_frame.copy()
            if not hasattr(self, '_last_swap_bbox'):
                self._last_swap_bbox = {}
            self._last_swap_bbox[video_basename] = (x1, y1, x2, y2)
            if final_mask is not None:
                if not hasattr(self, '_last_swap_mask'):
                    self._last_swap_mask = {}
                self._last_swap_mask[video_basename] = final_mask.copy()
                # v5.1008: Guardar para consistencia temporal de máscara
                if not hasattr(self, '_prev_final_mask'):
                    self._prev_final_mask = {}
                self._prev_final_mask[video_key] = final_mask.copy()

            return result_frame
            
        except Exception as e:
            print(f"[DEBUG] Error en _process_face_swap_v21: {e}", flush=True)
            traceback.print_exc()
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
        # ---- Log de sesión dedicado ----
        import sys
        import os as _os
        from datetime import datetime
        session_log_name = f"process_{datetime.now():%Y%m%d_%H%M%S}.log"
        session_log_path = _os.path.join(r'D:\PROJECTS\AUTOAUTO\logs', session_log_name)
        _os.makedirs(r'D:\PROJECTS\AUTOAUTO\logs', exist_ok=True)
        self._session_log_file = open(session_log_path, 'w', encoding='utf-8')
        self._session_log_file.write(f"=== AUTO-AUTO v5.1033 — M-EMA responsive + min_track dinámico + perfil m_blend boost ===\n")
        self._session_log_file.write(f"Start: {datetime.now()}\n")
        self._session_log_file.write(f"Video: {video_path}\n")
        self._session_log_file.write(f"Output: {output_path}\n\n")
        self._session_log_file.flush()
        # Limpiar logs antiguos (mantener solo últimos 10)
        try:
            log_dir = r'D:\PROJECTS\AUTOAUTO\logs'
            all_logs = sorted([
                _os.path.join(log_dir, f) for f in _os.listdir(log_dir)
                if f.startswith('process_') and f.endswith('.log')
            ], key=_os.path.getmtime)
            while len(all_logs) > 10:
                _os.remove(all_logs.pop(0))
        except:
            pass
        class _Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    try:
                        f.write(obj)
                        f.flush()
                    except:
                        pass
            def flush(self):
                for f in self.files:
                    try:
                        f.flush()
                    except:
                        pass
        self._old_stdout = sys.stdout
        sys.stdout = _Tee(sys.stdout, self._session_log_file)
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

        finally:
            if hasattr(self, '_old_stdout') and self._old_stdout:
                sys.stdout = self._old_stdout
            if hasattr(self, '_session_log_file') and self._session_log_file:
                try:
                    self._session_log_file.write(f"\n=== END ===\n")
                    self._session_log_file.close()
                except:
                    pass
                self._session_log_file = None

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
