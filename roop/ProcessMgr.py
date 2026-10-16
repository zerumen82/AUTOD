import os
import threading
import traceback
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np

import roop.globals
from roop.faceswap_config import get_faceswap_config, landmarks_fit_bbox
from roop.quality_enhancements import (
    match_color_histogram,
    detect_foreground_occlusion
)


print("\n" + "="*70)
print(">>> [TRUE-INTEGRATION] v5.1190 — FaceSwap automatic photo source selection and rotation <<<")
print("="*70 + "\n")

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
            except BaseException:
                pass
  
        if hasattr(face, '__setattr__'):
            try:
                face.gender_confidence = gender_confidence
            except BaseException:
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
    """Gestor de procesamiento de face swapping v5.1202"""
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
            print(f"ProcessMgr v5.1202 inicializado con progress_callback (Top-K={self.selected_top_k}, TTL={self.selected_assignment_ttl})")
        else:
            print(f"ProcessMgr v5.1202 inicializado SIN progress_callback (Top-K={self.selected_top_k}, TTL={self.selected_assignment_ttl})")

        self._initialize_processors()

    def initialize(self, input_facesets, target_faces, options, face_swap_mode=None):
        self.input_facesets = input_facesets or []
        self.target_faces = target_faces or []
        self.options = options
        self.face_swap_mode = face_swap_mode  # Guardar modo para optimizaciones
        self.is_initialized = True

        self._cache_source_embeddings()
        
        # LIMPIAR historial de boca para nuevo video. El mismo archivo
        # reprocesado no debe heredar el blend del último plano.
        self._mouth_open_history.clear()
        self._mouth_blend_smooth.clear()
        if isinstance(getattr(self, '_occ_frame_ema', None), dict):
            self._occ_frame_ema.clear()
        if isinstance(getattr(self, '_prev_final_mask', None), dict):
            self._prev_final_mask.clear()
        for attr in list(vars(self)):
            if attr.startswith('_mask_boost_ema_'):
                delattr(self, attr)
        for attr in (
            '_mouth_blend_ema_by_person',
            '_mouth_force_paste_by_person',
            '_mouth_paste_ema_by_person',
            '_mouth_state_history',
            '_mouth_ratio_smooth',
            '_mouth_hold_count',
        ):
            history = getattr(self, attr, None)
            if isinstance(history, dict):
                history.clear()
        self._mouth_blend_mask = None
        self._mouth_blend_factor = 0.0

        print(f"ProcessMgr v5.1202 inicializado: {len(self.input_facesets)} facesets, {len(self.target_faces)} target faces")

        if len(self.input_facesets) == 0:
            print("[WARNING] No hay facesets de origen cargados")
        if len(self.target_faces) == 0:
            print("[WARNING] No hay caras destino seleccionadas")

        self.global_source_for_all_id = None
        self._auto_source_face_cache.clear()

    def _cache_source_embeddings(self):
        self.source_embeddings_cache.clear()
        self._source_face_entries = []
        self._source_face_name_by_id = {}
        self._auto_source_consensus_cache = {}
        all_embs_with_quality = []
        profile_embs_with_quality = []
        
        for faceset in self.input_facesets:
            if hasattr(faceset, "faces") and faceset.faces:
                for face in faceset.faces:
                    self._source_face_name_by_id[id(face)] = getattr(faceset, 'name', '')
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
            # v5.1139 — Master embedding ENRIQUECIDO SIN DILUCIÓN: con un pool grande
            # (655 caras), la mejor cara individual pierde el matiz de la identidad.
            # Antes el promedio de TODAS diluía (v5.70 lo revertió); ahora se ponderan
            # SOLO las caras de la MISMA persona: las top-K por calidad (hasta 12) que
            # tengan sim >= 0.60 con la mejor (el mismo umbral _SOURCE_IDENTITY_SIM_THRESH
            # de v5.1064: por debajo es ruido entre personas distintas). Todo o nada:
            # si solo hay 1 cara de esa persona, embedding real 100% (sin tocar).
            all_embs_with_quality.sort(key=lambda x: x[1], reverse=True)
            _top_emb, _top_q = all_embs_with_quality[0]
            _pool = all_embs_with_quality[:12]
            _same_person = [(_e, _q) for (_e, _q) in _pool if float(np.dot(_top_emb, _e)) >= 0.60]
            if len(_same_person) > 1:
                _qsum = sum(_q for _e, _q in _same_person)
                _comb = sum(np.asarray(_e, dtype=np.float32) * _q for _e, _q in _same_person)
                _n = np.linalg.norm(_comb)
                if _n > 0:
                    self.master_source_embedding = (_comb / _n).astype(np.float32)
                    print(
                        f"[IDENTITY] Master embedding combinado: {len(_same_person)} caras de la "
                        f"MISMA persona (sim>=0.60 con la mejor, quality={_top_q:.2f}) "
                        f"ponderadas por calidad — sin dilución de otras identidades"
                    )
            if self.master_source_embedding is None:
                self.master_source_embedding = _top_emb.copy()
                print(f"[IDENTITY] Master Embedding = mejor cara individual (quality={_top_q:.2f}, total {len(all_embs_with_quality)})")
            norm = np.linalg.norm(self.master_source_embedding)
            if norm > 0:
                self.master_source_embedding /= norm
        if profile_embs_with_quality:
            profile_embs_with_quality.sort(key=lambda x: x[1], reverse=True)
            # v5.1152b — Profile Master ENRIQUECIDO (mismo mecanismo del master frontal
            # v5.1139): el master de perfil era SOLO la mejor cara individual; con la
            # variación de yaw entre perfiles del pool (lean/square/planos distintos)
            # el embedding individual pierde el matiz rotacional. Se ponderan por
            # calidad las top-12 de perfil con sim >= 0.60 con la mejor (el mismo
            # umbral _SOURCE_IDENTITY_SIM_THRESH v5.1064 — por debajo es OTRA persona).
            # Todo o nada: 1 solo perfil de esa persona → embedding real 100% (sin tocar).
            _ptop_emb, _ptop_q = profile_embs_with_quality[0]
            _ppool = profile_embs_with_quality[:12]
            _psame = [(_e, _q) for (_e, _q) in _ppool if float(np.dot(_ptop_emb, _e)) >= 0.60]
            if len(_psame) > 1:
                _pqsum = sum(_q for _e, _q in _psame)
                _pcomb = sum(np.asarray(_e, dtype=np.float32) * _q for _e, _q in _psame)
                _pn = np.linalg.norm(_pcomb)
                if _pn > 0:
                    self.profile_master_embedding = (_pcomb / _pn).astype(np.float32)
                    print(
                        f"[IDENTITY] Profile Master combinado: {len(_psame)} perfiles de la "
                        f"MISMA persona (sim>=0.60 con el mejor, quality={_ptop_q:.2f}) "
                        f"ponderados por calidad — sin dilución"
                    )
            if self.profile_master_embedding is None:
                self.profile_master_embedding = _ptop_emb.copy()
                print(f"[IDENTITY] Profile Master = mejor cara de perfil (quality={_ptop_q:.2f}, n={len(profile_embs_with_quality)})")
            norm = np.linalg.norm(self.profile_master_embedding)
            if norm > 0:
                self.profile_master_embedding /= norm
        
        print(f"v5.1202: {len(self.source_embeddings_cache)} embeddings cacheados")

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
                    lf.write("[PROC_FRAME] No inicializado o frame None\n")
                return frame
        except BaseException:
            with open(_log, 'a') as lf:
                lf.write("[PROC_FRAME] Exception en check is_initialized\n")
        
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
            # v5.1053: en closeup activo usar det_size 640 primero (fast path ~2.5x, fallback 1024).
            # El estado closeup es el del frame anterior (persistente entre frames).
            # v5.1053b: prefer_fast en TODOS los vídeos — el paso 3 (closeup 640 extra) añadía
            # duplicados a distinta escala (causa del parpadeo) y triplicaba el coste (0.5 FPS);
            # 640-first + fallback 1024 cubre closeups y caras diminutas. Fotos: full pass.
            _vbn_hint = os.path.basename(getattr(self.options, 'current_video_path', None) or file_path or "")
            _closeup_hint = bool(_vbn_hint and getattr(self, f'_closeup_active_{_vbn_hint}', False))
            _is_video_hint = getattr(self.options, 'current_video_path', None) is not None
            target_faces_detected = get_all_faces_smart(
                frame, min_score=None, for_target=True,
                prefer_fast=_closeup_hint or _is_video_hint,
                include_180=not _is_video_hint,
            )
            with open(_log, 'a') as lf:
                lf.write(f"[PROC_FRAME] detectadas {len(target_faces_detected)} caras\n")

            valid_faces = target_faces_detected

            video_path = getattr(self.options, 'current_video_path', None) or file_path
            is_real_video = getattr(self.options, 'current_video_path', None) is not None

            # v5.1112: MODO FOTO — paridad de detección con la UI.
            # La UI detecta caras de destino con preproceso (CLAHE+sharpen) y thresh 0.15
            # (face_util.extract_face_images → detect_faces_robust); el batch detectaba
            # solo sobre la imagen CRUDA @ 0.10 (get_all_faces_smart). Fotos donde la UI
            # encontró la cara pero el detector crudo no → 0 valid_faces → foto devuelta
            # ORIGINAL sin swap y SIN NINGÚN LOG ("se salta caras seleccionadas").
            # Reintento con el MISMO pipeline que usó la UI al seleccionar.
            if not valid_faces and not is_real_video:
                try:
                    from roop.face_util import detect_faces_robust, preprocess_image_for_detection
                    from roop.face_util_rotation import convert_faces_to_face_objects, get_face_analyser as _gfa
                    _analyser = _gfa()
                    if _analyser is not None and frame is not None:
                        _pre_rgb = cv2.cvtColor(
                            preprocess_image_for_detection(frame), cv2.COLOR_BGR2RGB
                        )
                        _raw = detect_faces_robust(_pre_rgb, _analyser, 0.15)
                        if _raw:
                            valid_faces = convert_faces_to_face_objects(_raw)
                            print(
                                f"[FOTO] Reintento detección con preproceso UI: "
                                f"{len(valid_faces)} cara(s) recuperada(s)"
                            )
                        # v5.1139: closeup extremo — la cara ocupa casi todo y el detector
                        # no la ve (anchors techo ~512px). Mismos 2 fallbacks que la UI:
                        # (a) padding gris (la cara "se aleja" del detector),
                        # (b) descalado 0.5 (la cara pasa bajo el techo de anchors).
                        if not valid_faces:
                            _h9, _w9 = frame.shape[:2]
                            _pad_h9, _pad_w9 = _h9 // 3, _w9 // 3
                            _img_pad9 = cv2.copyMakeBorder(
                                _pre_rgb, _pad_h9, _pad_h9, _pad_w9, _pad_w9,
                                cv2.BORDER_CONSTANT, value=[127, 127, 127],
                            )
                            _pad_faces9 = detect_faces_robust(_img_pad9, _analyser, 0.12)
                            if _pad_faces9:
                                _conv9 = []
                                for _f9 in _pad_faces9:
                                    _fw9 = float(_f9.bbox[2] - _f9.bbox[0])
                                    _fh9 = float(_f9.bbox[3] - _f9.bbox[1])
                                    if _fw9 < min(_w9 * 0.15, 35) or _fh9 < min(_h9 * 0.15, 35):
                                        continue
                                    _f9.bbox = np.array([
                                        _f9.bbox[0] - _pad_w9, _f9.bbox[1] - _pad_h9,
                                        _f9.bbox[2] - _pad_w9, _f9.bbox[3] - _pad_h9,
                                    ], dtype=_f9.bbox.dtype)
                                    if getattr(_f9, 'kps', None) is not None:
                                        _f9.kps = _f9.kps.copy()
                                        _f9.kps[:, 0] -= _pad_w9
                                        _f9.kps[:, 1] -= _pad_h9
                                    _conv9.append(_f9)
                                if _conv9:
                                    valid_faces = convert_faces_to_face_objects(_conv9)
                                    print(f"[FOTO] Reintento con padding: {len(valid_faces)} cara(s)")
                        if not valid_faces:
                            _h9b, _w9b = frame.shape[:2]
                            _small9 = cv2.resize(
                                _pre_rgb, (max(1, _w9b // 2), max(1, _h9b // 2)),
                                interpolation=cv2.INTER_AREA,
                            )
                            _sm_faces9 = detect_faces_robust(_small9, _analyser, 0.15)
                            if _sm_faces9:
                                valid_faces = convert_faces_to_face_objects(_sm_faces9)
                                for _vf9 in valid_faces:
                                    try:
                                        _vf9.bbox = [v * 2.0 for v in _vf9.bbox]
                                        if getattr(_vf9, 'kps', None) is not None:
                                            _vf9.kps = [[p[0] * 2.0, p[1] * 2.0] for p in _vf9.kps]
                                    except Exception:
                                        pass
                                if valid_faces:
                                    print(f"[FOTO] Reintento descalado 0.5: {len(valid_faces)} cara(s)")
                except Exception as e:
                    print(f"[FOTO] Reintento detección falló: {e}")

            if not valid_faces and not is_real_video:
                # La UI ya detectó y seleccionó esta cara sobre este mismo
                # archivo. Si un segundo análisis falla, conservar su bbox,
                # kps y embedding en vez de guardar la foto sin swap.
                _filename = os.path.basename(file_path or '')
                _stored = roop.globals.selected_face_references.get(f"selected_face_ref_{_filename}")
                _exact_refs = self._photo_refs_for_exact_file(file_path, self._as_ref_list(_stored))
                _ui_faces = [
                    face for ref in _exact_refs
                    if (face := self._photo_reference_face(ref, frame.shape)) is not None
                ]
                if _ui_faces:
                    valid_faces = _ui_faces
                    print(f"[FOTO] Detección de proceso vacía; reutilizadas {len(valid_faces)} cara(s) seleccionadas en esta misma foto", flush=True)

            if not valid_faces and not is_real_video:
                print(f"[FOTO] SKIP {os.path.basename(file_path or '')}: 0 caras detectadas (ni crudo ni preproceso) — foto original sin swap")
            self._current_video_key = f"video_{os.path.basename(video_path)}" if video_path else "frame"

            if not valid_faces:
                with open(_log, 'a') as lf:
                    lf.write("[PROC_FRAME] SIN valid_faces - hold o frame\n")
                if is_real_video and video_path:
                    video_basename = os.path.basename(video_path)
                    # Keep the tracking clock aligned with the decoded video.
                    # Frames without a detection still consume time in scene-cut,
                    # reacquisition and hold windows.
                    frame_count_attr = f'_frame_count_{video_basename}'
                    frame_count = getattr(self, frame_count_attr, 0) + 1
                    setattr(self, frame_count_attr, frame_count)
                    # A new shot can start without a detected face. Observe
                    # that frame before holding pixels from the previous shot.
                    is_visual_cut, cut_diff = self._detect_visual_scene_cut(
                        frame, video_basename, frame_count,
                    )
                    if is_visual_cut:
                        self._reset_tracking_on_scene_cut(
                            video_path, frame_count,
                            reason=f"cambio visual sin cara (diff={cut_diff:.1f})",
                        )
                        return frame
                    # v5.1022b: Contador de frames sin cara — tras 3 devuelve original
                    empty_attr = f'_empty_faces_count_{video_basename}'
                    empty_count = getattr(self, empty_attr, 0) + 1
                    setattr(self, empty_attr, empty_count)
                    if empty_count <= 3 and self._should_hold_previous_swap(video_basename):
                        prev_full = getattr(self, '_prev_frame_result', {}).get(self._current_video_key)
                        last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                        if prev_full is not None and last_bbox is not None:
                            # v5.1054c: fade-out — elipse se desvanece en vez de congelarse
                            fade_blend = self._faded_hold_blend(video_basename, 0.88)
                            if fade_blend < 0.10:
                                return frame
                            prev_mask = getattr(self, '_last_swap_mask', {}).get(video_basename)
                            return self._hold_previous_face_region(frame, prev_full, last_bbox, blend=fade_blend, prev_mask=prev_mask)
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

            # v5.1139: modo aleatorio = SOLO FOTOS (la UI bloquea el checkbox con vídeo
            # cargado). Definido aquí para TODOS los caminos (guard UnboundLocalError:
            # el truncado de faces_to_process más abajo lo lee también en vídeo, donde
            # SIEMPRE es False).
            _random_multi = False

            faces_to_process = []
            
            if is_real_video:
                self._is_single_photo = False  # v5.1026: modo video
                video_basename = os.path.basename(video_path)
                frame_count_pre = getattr(self, f'_frame_count_{video_basename}', 0) + 1
                try:
                    self._update_closeup_state(valid_faces, frame.shape, video_basename)
                except Exception as e:
                    print(f"[PREPROC_ERR] _update_closeup_state: {e}")
                    traceback.print_exc()
                plan_smooth_until = getattr(self, f'_plan_change_smooth_until_{video_basename}', 0)
                is_visual_cut, cut_diff = self._detect_visual_scene_cut(
                    frame, video_basename, frame_count_pre,
                )
                try:
                    zoom_in = self._is_zoom_in_progress(valid_faces, video_basename)
                except Exception as e:
                    print(f"[PREPROC_ERR] _is_zoom_in_progress: {e}")
                    traceback.print_exc()
                    zoom_in = False
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
                try:
                    if self._detect_closeup_plan_change(
                        valid_faces, frame.shape, video_basename, frame_count_pre,
                    ):
                        self._open_reacquire_window(video_basename, frame_count_pre, frames=60)
                        print(
                            f"[PLAN_CHANGE] Frame {frame_count_pre}: primer plano detectado "
                            "— re-busqueda activa"
                        )
                except Exception as e:
                    print(f"[PREPROC_ERR] _detect_closeup_plan_change: {e}")
                    traceback.print_exc()

                if not hasattr(self, f'_reference_face_selected_{video_basename}'):
                    try:
                        self._setup_selected_faces_frame_for_video(video_path, valid_faces)
                    except Exception as e:
                        print(f"[PREPROC_ERR] _setup_selected_faces_frame_for_video: {e}")
                        traceback.print_exc()

                # The tracker below already performs global reacquisition for
                # an unassigned target. Running it here as well counted two
                # failed attempts on one decoded frame and accelerated its
                # fallback thresholds without new visual evidence.
                target_face = self._find_target_face_for_selected_mode(video_path, valid_faces, frame)
                if isinstance(target_face, np.ndarray):
                    # v5.1048: closeup pendiente / hold devuelve IMAGEN ya blendada
                    # (no un Face). Antes `if target_face:` lanzaba ValueError (truth
                    # value of array ambiguous) → except devolvía el frame ORIGINAL
                    # sin swap → frames originales intercalados = temblor/parpadeo.
                    return target_face
                if target_face is not None:
                    faces_to_process = [target_face]
            else:
                # MODO FOTO: Respetar si el usuario seleccionó una cara específica en la UI
                self._is_single_photo = True  # v5.1026: flag para maximizar identidad source en fotos
                filename = os.path.basename(file_path) if file_path else "unknown"
                video_key = f"selected_face_ref_{filename}"
                face_ref_data = roop.globals.selected_face_references.get(video_key) or roop.globals.selected_face_references.get(filename)
                # El banco de la UI se indexa por basename. Si el lote contiene
                # dos archivos con el mismo nombre en carpetas distintas, usar
                # solo la selección hecha sobre esta foto antes de comparar caras.
                if face_ref_data:
                    _named_refs = self._as_ref_list(face_ref_data)
                    if any(ref.get('source_path') for ref in _named_refs):
                        face_ref_data = self._photo_refs_for_exact_file(file_path, _named_refs) or None

                # v5.1043: Fallback a TARGET_FACES (destino seleccionado en la UI) cuando no
                # existe referencia por archivo. Sin esto, las fotos del batch se guardaban
                # SIN swap (original re-encodeado) si el usuario seleccionó el destino en
                # otra preview (log real: todas las fotos del batch con diff ~1 = JPEG noise).
                if not face_ref_data:
                    # v5.1112: probar el MEJOR match entre TODAS las caras destino de la UI,
                    # no solo TARGET_FACES[0]. Un archivo sin referencia propia (el usuario
                    # seleccionó su cara en otra preview) quedaba atado a la primera cara
                    # del pool: sim 0.05-0.16 → skip silencioso = foto sin swap.
                    target_faces = getattr(roop.globals, 'TARGET_FACES', []) or []
                    _best_tf, _best_sim = None, -1.0
                    # v5.1140: en random source la similitud destino↔cara NO descarta
                    # (el source es aleatorio; la cara destino solo define el sitio).
                    _in_random = bool(getattr(roop.globals, 'source_random', False))
                    if target_faces and valid_faces:
                        for _tf in target_faces:
                            _tf_emb = getattr(_tf, 'embedding', None)
                            if _tf_emb is None:
                                continue
                            for _vf in valid_faces:
                                _s = self._calculate_similarity(_tf_emb, getattr(_vf, 'embedding', None))
                                if _s > _best_sim:
                                    _best_sim, _best_tf = _s, _tf
                    if _best_tf is not None and (_best_sim > 0.28 or _in_random):
                        face_ref_data = {
                            'face_obj': _best_tf,
                            'embedding': _best_tf.embedding,
                            'bbox': getattr(_best_tf, 'bbox', None),
                        }
                        print(
                            f"[FOTO] Sin referencia por archivo — mejor match de "
                            f"{len(target_faces)} destinos UI (sim={max(0.0, _best_sim):.3f}"
                            f"{', random: sim no descarta' if _in_random else ''})"
                        )
                    else:
                        print(
                            f"[FOTO] {filename}: sin referencia por archivo ni match "
                            f">=0.28 entre {len(target_faces)} destinos y {len(valid_faces)} caras "
                            f"(best={max(0.0, _best_sim):.3f}); evaluando recuperación"
                        )

                if face_ref_data:
                    # v5.1138: el banco COMPLETO participa (hasta 3 refs por archivo).
                    # Antes solo la 1ª ref matcheaba: una foto cuyo ref[0] daba sim baja
                    # salía sin swap aunque otra ref del banco matcheara bien, y en modo
                    # aleatorio el usuario quiere 2-3 destinos seleccionados swappeados.
                    _photo_bank = self._as_ref_list(face_ref_data)
                else:
                    _photo_bank = []
                if _photo_bank:
                    _random_multi = (
                        bool(getattr(roop.globals, 'source_random', False))
                        and len(_photo_bank) > 1
                    )
                    _all_pairs = []
                    for _ref in _photo_bank:
                        _ref_obj = _ref.get('face_obj')
                        _ref_emb = _ref.get('embedding')
                        if _ref_emb is None and _ref_obj is not None:
                            _ref_emb = getattr(_ref_obj, 'embedding', None)
                        if _ref_emb is None:
                            continue
                        for face in valid_faces:
                            sim = self._calculate_similarity(_ref_emb, face.embedding)
                            _all_pairs.append({'face': face, 'sim': sim, 'ref': _ref})
                    # v5.1100: 0.35→0.28 — ref de baja calidad (cara pequeña en UI) ya no pierde el match
                    _good = [p for p in _all_pairs if p['sim'] > 0.28]
                    if _good:
                        if _random_multi:
                            # MODO ALEATORIO MULTI-DESTINO: el usuario seleccionó 2-3 caras
                            # de la MISMA imagen → cada una se swappea (con su source
                            # aleatorio propio). Dedup por bbox: la misma cara detectada
                            # no se procesa dos veces aunque matchee varias refs.
                            # v5.1139: se toman TODOS los pares válidos (hasta el nº de
                            # caras destino seleccionadas) — antes `faces_to_process[:1]`
                            # TRUNCABA a 1 justo después y el multi-destino nunca
                            # ejecutaba más de un swap ("no funciona el random multi").
                            try:
                                from roop.face_util_rotation import _same_face_boxes
                            except Exception:
                                _same_face_boxes = None
                            _good.sort(key=lambda p: p['sim'], reverse=True)
                            _taken = []
                            _max_taken = max(len(_photo_bank), sum(1 for p in _good if p['sim'] > 0.45))
                            for p in _good:
                                if len(_taken) >= _max_taken:
                                    break
                                if _same_face_boxes is not None and any(
                                    _same_face_boxes(p['face'].bbox, t.bbox) for t in _taken
                                ):
                                    continue
                                _taken.append(p['face'])
                            faces_to_process = _taken
                            print(
                                f"[FOTO] {filename}: {len(faces_to_process)} destino(s) "
                                f"swappeados (modo aleatorio multi, banco={len(_photo_bank)})"
                            )
                        else:
                            # v5.1153 — MATCH INEQUÍVOCO: elegir SOLO el mejor par. Antes se
                            # elegía la cara con sim más alta, aunque OTRAS caras del frame
                            # puntuaran casi igual (sims 0.30/0.31/0.29 = ninguna supera el
                            # ruido entre personas 0.57, v5.1064) → el swap podía caer en la
                            # cara equivocada. Con varios pares cerca del máximo NO hay match
                            # fiable → no swap (coherente con vídeo: ante duda, no pintar).
                            # Umbral de ambigüedad 0.03 = el mismo margin del tracking multi.
                            # v5.1154 — el gate SOLO aplica con UNA ref. Con varias refs
                            # (2 destinos seleccionados), los dos mejores pares son de
                            # personas DISTINTAS (ref1→cara1, ref2→cara2) y compararlos
                            # entre sí era absurdo: foto de 2 personas = siempre ambigua
                            # = ninguna swap ("si tengo 2 = ninguna"). Con N refs: asignación
                            # greedy por cara (cada cara a su mejor ref, sin duplicados).
                            _good_sorted = sorted(_good, key=lambda p: p['sim'], reverse=True)
                            if len(_photo_bank) > 1:
                                _assigned_faces, _assigned_refs = [], []
                                for p in _good_sorted:
                                    _f_idx = id(p['face'])
                                    _ref_idx = id(p.get('ref'))
                                    if _f_idx in _assigned_faces or _ref_idx in _assigned_refs:
                                        continue
                                    _assigned_faces.append(_f_idx)
                                    _assigned_refs.append(_ref_idx)
                                    faces_to_process.append(p['face'])
                                print(
                                    f"[FOTO] {filename}: {len(faces_to_process)} destino(s) "
                                    f"swappeados (match por ref, banco={len(_photo_bank)})"
                                )
                            elif len(_good_sorted) >= 2 and (
                                _good_sorted[0]['sim'] - _good_sorted[1]['sim'] < 0.03
                            ):
                                print(
                                    f"[FOTO] {filename}: match ambiguo "
                                    f"(sim1={_good_sorted[0]['sim']:.3f} vs "
                                    f"sim2={_good_sorted[1]['sim']:.3f}, diff<0.03) — "
                                    f"evaluando posición de la selección UI"
                                )
                            else:
                                _best = _good_sorted[0]
                                faces_to_process = [_best['face']]
                    elif bool(getattr(roop.globals, 'source_random', False)):
                        # v5.1140: en MODO ALEATORIO el source es aleatorio por definición,
                        # así que la similitud ref↔cara NO puede ser criterio de descarte:
                        # la ref solo define DÓNDE swappear. Una foto con caras detectadas
                        # SIEMPRE sale swappeada (la cara mayor cuando no hay match de
                        # identidad). Nunca SKIP en random (log real: best=0.158 → foto
                        # original sin swap).
                        def _detected_face_area(face):
                            """Area robusta para bboxes que llegan como list o ndarray."""
                            try:
                                box = np.asarray(getattr(face, 'bbox', None), dtype=np.float32).reshape(-1)
                                if box.size < 4:
                                    return 0.0
                                return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))
                            except Exception:
                                return 0.0

                        _faces_sorted = sorted(
                            valid_faces,
                            key=_detected_face_area,
                            reverse=True,
                        )
                        faces_to_process = _faces_sorted[:len(_photo_bank)]
                        print(
                            f"[FOTO] {filename}: {len(faces_to_process)} destino(s) "
                            f"swappeados (modo aleatorio, SIN match de identidad — "
                            f"best_sim={max((p['sim'] for p in _all_pairs), default=0.0):.3f}, "
                            "el source es aleatorio: la sim no descarta)"
                        )
                    else:
                        # v5.1112: el fallo del match ya no es silencioso — ahora con el
                        # mejor sim de TODO el banco (antes solo ref[0]).
                        _weak = max((p['sim'] for p in _all_pairs), default=0.0)
                        print(
                            f"[FOTO] {filename}: match débil contra su referencia "
                            f"(mejor sim de {len(_photo_bank)} ref(s)={max(_weak, 0.0):.3f} <= 0.28, "
                            f"caras detectadas={len(valid_faces)}); evaluando recuperación"
                        )

                    if not faces_to_process and valid_faces:
                        _exact_refs = self._photo_refs_for_exact_file(file_path, _photo_bank)
                        _spatial_face, _spatial_iou = self._photo_spatial_match(_exact_refs, valid_faces)
                        if _spatial_face is not None:
                            faces_to_process = [_spatial_face]
                            print(f"[FOTO] Cara seleccionada en esta misma foto recuperada por posición (IoU={_spatial_iou:.2f})", flush=True)
                elif face_swap_mode not in ['selected_faces_frame', 'selected_faces', 'selected']:
                    # v5.1043: Modos no-selected en fotos: procesar TODAS las caras detectadas
                    # (sin referencia). Antes el bloque solo llenaba faces_to_process con la
                    # referencia, dejando 'all'/'auto' con 0 caras → original sin swap.
                    faces_to_process = list(valid_faces)

                if _photo_bank:
                    exact_faces = self._selected_photo_faces(
                        file_path, _photo_bank, valid_faces, frame.shape,
                    )
                    if exact_faces:
                        faces_to_process = exact_faces
                        print(
                            f"[FOTO] {filename}: {len(exact_faces)} destino(s) "
                            "seleccionado(s) en esta foto",
                            flush=True,
                        )

                # El usuario ha seleccionado una persona de destino para el
                # lote. En una foto con una única cara no hay otra identidad a
                # confundir: una pose que baja ArcFace bajo 0.28 no debe causar
                # un archivo original sin swap. Con varias caras se mantiene el
                # match por referencia o la selección espacial del mismo archivo.
                if not faces_to_process and len(valid_faces) == 1 and getattr(roop.globals, 'TARGET_FACES', None):
                    faces_to_process = [valid_faces[0]]
                    print(f"[FOTO] {filename}: única cara detectada; destino del lote seleccionado (sin veto por similitud)", flush=True)

                # v5.87: Sin fallback — si no hay match fiable, hold o frame original

            if not faces_to_process:
                if not is_real_video:
                    print(f"[FOTO] SKIP {os.path.basename(file_path or '')}: ninguna cara destino inequívoca tras los reintentos", flush=True)
                if is_real_video and video_path:
                    video_basename = os.path.basename(video_path)
                    if self._should_hold_previous_swap(video_basename) or (
                        getattr(self, f'_closeup_active_{video_basename}', False)
                        and self._has_swap_history(video_basename)
                    ):
                        prev_full = getattr(self, '_prev_frame_result', {}).get(self._current_video_key)
                        last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                        if prev_full is not None and last_bbox is not None:
                            # v5.1054c: fade-out — sin match (cambio de plano/persona ausente)
                            # la elipse del swap anterior se desvanece en ~6 frames
                            base_blend = 0.95 if getattr(self, f'_closeup_active_{video_basename}', False) else 0.88
                            fade_blend = self._faded_hold_blend(video_basename, base_blend)
                            if fade_blend < 0.10:
                                return frame
                            prev_mask = getattr(self, '_last_swap_mask', {}).get(video_basename)
                            return self._hold_previous_face_region(
                                frame, prev_full, last_bbox, blend=fade_blend, prev_mask=prev_mask,
                            )
                return frame
                
            # Siempre key frame para máxima calidad
            self._is_key_frame = True
            result_frame = frame.copy()
            
            # En vídeo, un track de destino por fotograma. En fotos, procesar
            # todas las caras de destino seleccionadas en esta imagen.
            _limit_one = is_real_video and face_swap_mode in ['selected_faces_frame', 'selected_faces', 'selected']
            faces_to_process_limited = faces_to_process[:1] if _limit_one else faces_to_process
            
            for i, target_face in enumerate(faces_to_process_limited):
                try:
                    all_faces = []
                    for faceset in self.input_facesets:
                        if hasattr(faceset, "faces") and faceset.faces:
                            all_faces.extend(faceset.faces)
                      
                    if not all_faces:
                        continue

                    source_face = None
                    is_target_profile = ProcessMgr._is_profile_face(
                        getattr(target_face, 'kps', None), getattr(target_face, 'det_score', None)
                    )
                    if is_real_video and video_path:
                        lock_key, _ = self._video_source_lock_keys(video_path)
                        source_face = getattr(self, lock_key, None)

                    if source_face is None:
                        source_face = self._select_source_face(
                            target_face, all_faces, face_swap_mode, video_path,
                            is_profile=is_target_profile,
                        )
                        if source_face is not None and is_real_video and video_path:
                            lock_key, _ = self._video_source_lock_keys(video_path)
                            sim_score = self._calculate_similarity(
                                getattr(source_face, 'embedding', None),
                                getattr(target_face, 'embedding', None),
                            ) if hasattr(source_face, 'embedding') else 0
                            setattr(self, lock_key, source_face)
                            print(
                                f"[MATCH] Source automática fijada para el vídeo "
                                f"(sim={sim_score:.3f})",
                                flush=True,
                            )

                    if source_face is not None and is_target_profile:
                        self._last_profile_match_sim = self._calculate_similarity(
                            getattr(source_face, 'embedding', None),
                            getattr(target_face, 'embedding', None),
                        )

                    if source_face is None:
                        if is_real_video and video_path:
                            video_basename = os.path.basename(video_path)
                            if self._should_hold_previous_swap(video_basename):
                                prev_full = getattr(self, '_prev_frame_result', {}).get(self._current_video_key)
                                last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                                if prev_full is not None and last_bbox is not None:
                                    # v5.1054c: fade-out — sin source (pérdida de identidad)
                                    fade_blend = self._faded_hold_blend(video_basename, 0.88)
                                    if fade_blend < 0.10:
                                        continue
                                    prev_mask = getattr(self, '_last_swap_mask', {}).get(video_basename)
                                    return self._hold_previous_face_region(
                                        frame, prev_full, last_bbox, blend=fade_blend, prev_mask=prev_mask,
                                    )
                        continue

                    # PROCESAR SWAP CON CALIDAD FORZADA
                    result_frame = self._process_face_swap_v21(
                        source_face, target_face, result_frame, frame, enable_temporal_smoothing,
                        person_idx=i, photo_path=file_path if not is_real_video else None,
                    )
                except Exception as e:
                    print(f"Error procesando cara {i+1}: {e}")
                    continue

            return result_frame
        except Exception as e:
            _file_label = os.path.basename(file_path) if file_path else "frame desconocido"
            print(f"Frame processing error [{_file_label}]: {e}", flush=True)
            # print_exc writes to stderr, which the batch logger does not retain.
            # Emit the formatted traceback through stdout so a later audit has
            # the exact source line and does not need the interactive console.
            print(traceback.format_exc(), flush=True)
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

                if face_w < 40 or face_h < 40:
                    score -= 2.5
                else:
                    score += min(2.0, np.log1p(face_area) / 7.0)

                aspect = face_w / face_h if face_h > 0 else 0.0
                if 0.65 <= aspect <= 1.45:
                    score += 0.4
                else:
                    score -= 0.5

                full_ref_img = getattr(face, 'face_img_ref', None)
                ref_img = full_ref_img if full_ref_img is not None else getattr(face, 'face_img', None)
                if isinstance(ref_img, np.ndarray) and ref_img.size > 0:
                    img_h, img_w = ref_img.shape[:2]
                    original_size = getattr(face, 'source_image_size', None)
                    if original_size is not None and len(original_size) == 2:
                        original_w, original_h = original_size
                    elif full_ref_img is not None:
                        original_w, original_h = img_w, img_h
                    else:
                        original_w, original_h = 0, 0
                    if original_w > 0 and original_h > 0:
                        area_ratio = face_area / max(1.0, original_w * original_h)
                        if area_ratio > 0.85:
                            score -= 3.0
                        elif area_ratio < 0.005:
                            score -= 2.0
                        elif 0.03 <= area_ratio <= 0.75:
                            score += 0.6

                    # Detector confidence and face size say little about focus.
                    # Measure the face itself at a fixed size so a blurred
                    # upload with a clean background cannot rank as the best
                    # source for an entire video.
                    crop_origin = getattr(face, 'face_img_origin', None)
                    if crop_origin is not None:
                        ox, oy = crop_origin
                        cx1, cy1 = max(0, int(x1-ox)), max(0, int(y1-oy))
                        cx2, cy2 = min(img_w, int(x2-ox)), min(img_h, int(y2-oy))
                        crop = ref_img[cy1:cy2, cx1:cx2]
                    elif full_ref_img is not None and (
                            original_size is None or tuple(original_size) == (img_w, img_h)):
                        cx1, cy1 = max(0, int(x1)), max(0, int(y1))
                        cx2, cy2 = min(img_w, int(x2)), min(img_h, int(y2))
                        crop = ref_img[cy1:cy2, cx1:cx2]
                    else:
                        crop = ref_img
                    if crop.shape[0] >= 32 and crop.shape[1] >= 32:
                        color_code = cv2.COLOR_RGB2GRAY if getattr(face, 'face_img_is_rgb', False) else cv2.COLOR_BGR2GRAY
                        gray = cv2.cvtColor(crop, color_code) if crop.ndim == 3 else crop
                        gray = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
                        focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                        if focus < 35.0:
                            score -= 2.0
                        elif focus < 80.0:
                            score -= 1.0
                        elif focus >= 180.0:
                            score += 0.35

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
        import hashlib
        normalized = os.path.normcase(os.path.abspath(video_path))
        key = hashlib.sha1(normalized.encode('utf-8')).hexdigest()[:16]
        return (
            f'_video_locked_source_face_{key}',
            f'_video_locked_profile_source_{key}',
        )

    def _clear_video_source_locks(self, video_path: str):
        if not video_path:
            return
        lock_key, profile_lock_key = self._video_source_lock_keys(video_path)
        for key in (lock_key, profile_lock_key):
            if hasattr(self, key):
                delattr(self, key)

    def _get_user_reference_embedding(self, video_basename: str):
        if hasattr(roop.globals, 'selected_face_references'):
            # v5.1117: normalizar via _as_ref_list — la UI guarda un BANCO (lista de
            # dicts, frontal+perfil); tratar la ref como dict singular crasheaba con
            # AttributeError ('list' has no 'get') y mataba _reacquire_face_global
            # entera (frames sin tracking/swap en cada intento de readquisición).
            _bank = self._video_reference_bank(video_basename)
            for _r in _bank:
                emb = np.array(_r.get('embedding'), dtype=np.float32).flatten()
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

    @staticmethod
    def _as_ref_list(ref_data):
        """Normaliza el banco de referencias por archivo: lista de dicts.
        Acepta forma legacy (dict único) y nueva (lista, máx 3)."""
        if ref_data is None:
            return []
        if isinstance(ref_data, dict):
            return [ref_data] if ref_data.get('embedding') is not None else []
        if isinstance(ref_data, (list, tuple)):
            return [r for r in ref_data if isinstance(r, dict) and r.get('embedding') is not None]
        return []

    def _video_reference_bank(self, video_basename, video_path=None):
        """Selected destination references belonging to this exact video."""
        from roop.destination_selection import video_reference_bank

        path = video_path or getattr(getattr(self, 'options', None), 'current_video_path', None)
        if not path:
            refs = getattr(roop.globals, 'selected_face_references', {}) or {}
            return self._as_ref_list(refs.get(f'selected_face_ref_{video_basename}'))
        return video_reference_bank(
            getattr(roop.globals, 'selected_face_references', {}),
            path,
        )

    @staticmethod
    def _photo_refs_for_exact_file(file_path, references):
        """Only UI selections made on this image may use spatial matching."""
        if not file_path:
            return []
        exact_path = os.path.normcase(os.path.abspath(file_path))
        return [
            ref for ref in references
            if ref.get('source_path')
            and os.path.normcase(os.path.abspath(ref['source_path'])) == exact_path
        ]

    @staticmethod
    def _photo_reference_face(ref, frame_shape):
        """Reuse a selected face only while its geometry fits the same image."""
        face = ref.get('face_obj')
        if face is None or getattr(face, 'embedding', None) is None:
            return None
        try:
            box = np.asarray(getattr(face, 'bbox', None), dtype=np.float32).reshape(-1)
            kps = np.asarray(getattr(face, 'kps', None), dtype=np.float32).reshape(-1, 2)
            height, width = frame_shape[:2]
            if box.size != 4 or kps.shape[0] < 5 or not np.isfinite(box).all() or not np.isfinite(kps).all():
                return None
            x1, y1, x2, y2 = box
            if x2 - x1 < 16 or y2 - y1 < 16:
                return None
            if not (0 <= (x1+x2)/2 < width and 0 <= (y1+y2)/2 < height):
                return None
            if not landmarks_fit_bbox(kps, box):
                return None
            return face
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _photo_spatial_match(references, faces):
        """Find a unique detector face at the UI-selected location (same file)."""
        candidates = []
        for ref in references:
            try:
                box = np.asarray(ref.get('bbox'), dtype=np.float32).reshape(-1)
                if box.size != 4 or not np.isfinite(box).all():
                    continue
                ref_area = max(0.0, box[2]-box[0]) * max(0.0, box[3]-box[1])
                if ref_area <= 0:
                    continue
                for face in faces:
                    detected = np.asarray(getattr(face, 'bbox', None), dtype=np.float32).reshape(-1)
                    if detected.size != 4 or not np.isfinite(detected).all():
                        continue
                    detected_area = max(0.0, detected[2]-detected[0]) * max(0.0, detected[3]-detected[1])
                    intersection = max(0.0, min(box[2], detected[2])-max(box[0], detected[0])) * max(
                        0.0, min(box[3], detected[3])-max(box[1], detected[1]),
                    )
                    union = ref_area + detected_area - intersection
                    iou = intersection / union if union > 0 else 0.0
                    candidates.append((iou, face))
            except (TypeError, ValueError):
                continue
        if not candidates:
            return None, 0.0
        candidates.sort(key=lambda item: item[0], reverse=True)
        best_iou, best_face = candidates[0]
        competing = max((iou for iou, face in candidates[1:] if face is not best_face), default=0.0)
        if best_iou >= 0.35 and best_iou - competing >= 0.12:
            return best_face, best_iou
        return None, best_iou

    def _selected_photo_faces(self, file_path, references, detected_faces, frame_shape):
        """Honor every destination face selected on this exact photo."""
        exact_refs = self._photo_refs_for_exact_file(file_path, references)
        if not exact_refs:
            return []
        remaining = list(detected_faces)
        selected = []
        for ref in exact_refs:
            matched, _ = self._photo_spatial_match([ref], remaining)
            if matched is not None:
                remaining = [face for face in remaining if face is not matched]
            else:
                matched = self._photo_reference_face(ref, frame_shape)
            if matched is None:
                continue
            box = np.asarray(matched.bbox, dtype=np.float32).reshape(-1)
            if box.size != 4:
                continue
            if not any(
                np.array_equal(box, np.asarray(face.bbox, dtype=np.float32).reshape(-1))
                for face in selected
            ):
                selected.append(matched)
        return selected

    def _get_user_reference_embeddings(self, video_basename: str):
        """Banco de embeddings destino del vídeo (frontal+perfil+...): lista
        normalizada. Legacy-safe (dict único viejo → lista de 1)."""
        try:
            refs = self._video_reference_bank(video_basename)
            out = []
            for r in refs:
                try:
                    emb = np.array(r.get('embedding'), dtype=np.float32).flatten()
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        out.append(emb / norm)
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def _bank_best_sim(self, face_embedding, video_basename: str):
        """Max-sim contra el banco destino (None si no hay banco/cara).
        Misma persona, varios ángulos: el perfil matchea la ref de perfil
        aunque contra la frontal dé 0.15."""
        try:
            if face_embedding is None:
                return None
            best = None
            for ref_emb in self._get_user_reference_embeddings(video_basename):
                s = float(self._calculate_similarity(ref_emb, face_embedding))
                best = s if best is None else max(best, s)
            return best
        except Exception:
            return None

    def _selected_target_is_unique(self, candidate, faces, video_basename):
        """Confirm a low detector score using the selected destination identity."""
        candidate_sim = self._bank_best_sim(getattr(candidate, 'embedding', None), video_basename)
        if candidate_sim is None or candidate_sim < 0.35:
            return False
        others = [
            self._bank_best_sim(getattr(face, 'embedding', None), video_basename)
            for face in faces if face is not candidate
        ]
        other_best = max((sim for sim in others if sim is not None), default=-1.0)
        return candidate_sim - other_best >= 0.08

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
        self._reset_hold_fade(video_basename)
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

    def _reset_hold_fade(self, video_basename: str):
        """v5.1054c: Resetea el contador de fade de holds (swap real → fade desde cero)."""
        setattr(self, f'_hold_fade_count_{video_basename}', 0)
        setattr(self, f'_ghost_hold_count_{video_basename}', 0)

    def _faded_hold_blend(self, video_basename: str, base_blend: float = 0.88) -> float:
        """v5.1074c: ANTIFARTEFACTO — holds ahora se desvanecen en ~4 frames en
        vez de ~6. Fade más agresivo para evitar "ponia caras cuando no habian".
        """
        attr = f'_hold_fade_count_{video_basename}'
        count = getattr(self, attr, 0) + 1
        setattr(self, attr, count)
        # v5.1074c: Fade más rápido — de 6 pasos a 4 (frame 1=100%, 2=60%, 3=30%, 4=10%)
        factors = (1.0, 0.60, 0.30, 0.10)
        f = factors[min(count - 1, len(factors) - 1)]
        return base_blend * f

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

    def _clear_video_temporal_mouth_and_occlusion(self, video_key):
        """Drop mouth and occlusion memory for one video key.

        A talking-mouth blend near 0.65 pulls the next shot's closed mouth to
        about 0.26, still above the 0.10 apply threshold. Another video's
        keys stay. The instance mouth mask is not touched here.
        """
        if not video_key:
            return
        occ = getattr(self, '_occ_frame_ema', None)
        if isinstance(occ, dict):
            occ.pop(video_key, None)
        prefix = f"{video_key}_p"
        for attr in (
            '_mouth_blend_ema_by_person',
            '_mouth_force_paste_by_person',
            '_mouth_paste_ema_by_person',
            '_mouth_state_history',
            '_mouth_ratio_smooth',
            '_mouth_hold_count',
        ):
            history = getattr(self, attr, None)
            if not isinstance(history, dict):
                continue
            for key in [k for k in history if isinstance(k, str) and k.startswith(prefix)]:
                del history[key]

    def _reset_tracking_on_scene_cut(self, video_path, frame_count, reason="", soft=False):
        video_basename = os.path.basename(video_path) if video_path else ""
        if not video_basename:
            return
        video_key = f"video_{video_basename}"
        closeup_active = getattr(self, f'_closeup_active_{video_basename}', False)
        # The caller distinguishes a cut from continuous zoom. A closeup
        # does not make pixels or mouth history from another shot reusable.

        setattr(self, f'_target_face_assigned_{video_basename}', None)
        setattr(self, f'_tracking_lost_count_{video_basename}', 0)
        setattr(self, f'_position_history_{video_basename}', [])
        setattr(self, f'_consecutive_tracking_success_{video_basename}', 0)
        setattr(self, f'_strict_recovery_tick_{video_basename}', 0)
        if not soft:
            self._mark_unassigned(video_basename, frame_count)
            setattr(self, f'_hard_scene_cut_frame_{video_basename}', frame_count)

        if not soft:
            previous_masks = getattr(self, '_prev_final_mask', None)
            if isinstance(previous_masks, dict):
                previous_masks.pop(video_key, None)
            boost_key = f'_mask_boost_ema_{video_basename}'
            if hasattr(self, boost_key):
                delattr(self, boost_key)
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

        # El target seleccionado sigue siendo la misma identidad tras un corte.
        # Mantener la source del vídeo evita cambiar de cara origen entre planos.

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
            self._clear_video_temporal_mouth_and_occlusion(video_key)
            prof_key = f'_profile_hysteresis_{video_key}'
            if hasattr(self, prof_key):
                delattr(self, prof_key)
        elif closeup_active:
            setattr(self, f'_plan_change_smooth_until_{video_basename}', frame_count + 24)

        if reason:
            tag = "soft, mantiene swap" if soft else "reset tracking, sin hold"
            print(f"[SCENE_CUT] Frame {frame_count}: {reason} — {tag}")

    def _register_mask_skip(self):
        """Acumula caras entregadas sin swap por falta de contorno medido."""
        self._mask_skip_count = getattr(self, '_mask_skip_count', 0) + 1

    def _mask_skip_note(self) -> str:
        count = getattr(self, '_mask_skip_count', 0)
        return f" | {count} cara(s) sin segmentación (sin swap)" if count else ""

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
        frame_shape=None,
    ):
        ref_emb = self._get_user_reference_embedding(video_basename)
        if ref_emb is None or not valid_faces:
            return None
        multi_face = len(valid_faces) >= 2
        if closeup_mode:
            min_sim = min(min_sim, 0.30 if multi_face else 0.24)  # v5.1053: EXQUISITO — closeup multi 0.25→0.30, single 0.22→0.24
        elif after_scene_cut:
            min_sim = min(min_sim, 0.26 if multi_face else 0.18)  # v5.1053: scene_cut multi 0.20→0.26, single 0.16→0.18
        else:
            min_sim = max(min_sim, 0.32 if multi_face else 0.28)

        # v5.1005: Use tracking_embedding (up-to-date) for better recovery
        tracking_emb = getattr(self, f'_tracking_embedding_{video_basename}', None)

        # v5.1005: Progressive threshold lowering after consecutive re-search failures.
        # v5.1022: También para multi-face (evita swap congelado en videos con varias caras).
        # v5.1053: EXQUISITO — floors subidos (multi 0.30, single 0.24-0.26): el floor 0.02-0.05
        # aceptaba cualquier cara (RE-BUSQUEDA real aceptó sim=0.13 en la persona equivocada).
        # NUNCA bajar de ~0.24: ante duda, hold — nunca otra persona.
        re_fail_attr = f'_reacquire_fail_count_{video_basename}'
        re_fails = getattr(self, re_fail_attr, 0)
        if re_fails >= 20:
            min_sim = max(0.24 if not multi_face else 0.30, min_sim - 0.36)
        elif re_fails >= 10:
            min_sim = max(0.24 if not multi_face else 0.30, min_sim - 0.16)
        elif re_fails >= 5:
            min_sim = max(0.26 if not multi_face else 0.30, min_sim - 0.08)

        # v5.1053: EXQUISITO — en multi-face el reacquire debe exigir la MISMA identidad
        # que el loop principal (0.48 recovery / 0.50 normal + margen): si acepta más
        # laxo, el loop lo rechaza al frame siguiente → bucle aceptar→rechazar→ghost
        # = parpadeo (log real: 66 RE-BUSQUEDA aceptadas en 1798 frames).
        # v5.1069e: FIX CONGELACIÓN — este floor ANULABA el umbral progresivo (re_fails
        # 655 con umbral=0.48 constante, log Real_Kinky 222556): la persona salía del
        # plano, volvía con sim 0.30-0.45 y era rechazada PARA SIEMPRE → RE-BUSQUEDA
        # infinita + hold del swap anterior = congelación de 11s (frames 361-693).
        # Ahora el floor solo aplica SIN fallos acumulados; con re_fails>=10 el
        # progresivo gana (floor 0.30 multi / 0.24 single de v5.1005 sigue activo).
        if multi_face and re_fails < 10:
            min_sim = max(min_sim, 0.48 if (closeup_mode or after_scene_cut) else 0.50)

        # Get last known position for spatial proximity check
        position_history = getattr(self, f'_position_history_{video_basename}', [])
        last_center = position_history[-1] if position_history else None

        candidates = []
        for face in valid_faces:
            if not hasattr(face, 'embedding') or face.embedding is None:
                continue
            sim = float(self._calculate_similarity(ref_emb, face.embedding))

            # Banco destino: max-sim (perfil tras corte matchea su ref).
            _bank_sim = self._bank_best_sim(face.embedding, video_basename)
            if _bank_sim is not None:
                sim = max(sim, _bank_sim)

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
                            accept = sim >= max(0.22, min_sim - 0.20)  # v5.1053: floor 0.02→0.22 (exquisito)
                            # v5.1027: Fix last_bbox_val undefined → usar _last_swap_bbox
                            last_bbox_ref = self._last_swap_bbox.get(video_basename) if hasattr(self, '_last_swap_bbox') else None
                            if not accept and last_bbox_ref is not None:
                                lw = max(1, last_bbox_ref[2] - last_bbox_ref[0])
                                lh = max(1, last_bbox_ref[3] - last_bbox_ref[1])
                                last_area = lw * lh
                                fw = face.bbox[2] - face.bbox[0] if (hasattr(face, 'bbox') and face.bbox is not None) else 1
                                fh = face.bbox[3] - face.bbox[1] if (hasattr(face, 'bbox') and face.bbox is not None) else 1
                                area_ratio = (fw * fh) / max(1, last_area)
                                # v5.1066: cara GRANDE (primer plano) sin tope de área —
                                # un closeup legítimo crece 10-30x la última swap y el
                                # tope 6.0 lo mataba (log 095800: tramos sin swap).
                                if 0.18 <= area_ratio <= (100.0 if self._is_large_face(face.bbox, frame_shape) else 6.0):
                                    accept = sim >= max(0.20, min_sim - 0.30)  # v5.1053: floor 0.02→0.20 (exquisito)

            if accept:
                # v5.1051: ANTI-FALLBACK A CARA EQUIVOCADA (póster/foto pequeña).
                # Cuando la cara principal se pierde (perfil, blur, giro) el reacquire
                # aceptaba CUALQUIER cara con sim baja (floor 0.02, closeup 0.22) —
                # incluido un póster/foto de la pared (26x32px, nítido, det alto) que
                # quedaba swappeado el resto del vídeo (log real: RE-BUSQUEDA frame 493
                # sim=0.33 → póster estático → "rectángulo traslúcido" hasta el final).
                # Gates: tamaño mínimo absoluto + consistencia de área con la última
                # cara swappeada (relajado tras scene cut — corte legítimo de planos).
                fw2 = face.bbox[2] - face.bbox[0] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                fh2 = face.bbox[3] - face.bbox[1] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                if fw2 < 32 or fh2 < 32 or fw2 * fh2 < 1024:
                    continue
                # v5.1066: gate cara cortada — permite overflow <35% en caras grandes
                # (primer plano). v5.1060 lo rechazaba todo → mató tramos enteros.
                if frame_shape is not None and face.bbox is not None and self._is_bbox_clipped_by_frame(face.bbox, frame_shape, allow_partial_overflow=True):
                    continue
                if not after_scene_cut:
                    last_bbox_ref = self._last_swap_bbox.get(video_basename) if hasattr(self, '_last_swap_bbox') else None
                    if last_bbox_ref is not None:
                        lw = max(1, last_bbox_ref[2] - last_bbox_ref[0])
                        lh = max(1, last_bbox_ref[3] - last_bbox_ref[1])
                        area_ratio = (fw2 * fh2) / max(1.0, lw * lh)
                        if area_ratio < 0.06 or area_ratio > (100.0 if self._is_large_face(face.bbox, frame_shape) else 6.0):
                            continue
                area = self._largest_face_area(face)
                candidates.append((sim, area, face))

        if not candidates:
            setattr(self, re_fail_attr, re_fails + 1)
            # v5.1152 — RESCATE por embedding de TRACKING en re-búsqueda sin match.
            # Log real (wifefuckedhard 021745): 229 Lost / 490 Success (32% sin swap) con
            # ~130 líneas "RE-BUSQUEDA sin match (mejor_sim=0.06, track_sim=0.59-0.73,
            # umbral=0.48)": el candidato ERA la misma persona contra el embedding de
            # tracking (0.59-0.73) pero se comparaba contra ref_emb (0.06 — perfil
            # extremo/rotación/escena que la ref del banco nunca vio) → descartado
            # SIEMPRE → frame original → Lost en cadena. El tracking_emb es la identidad
            # consolidada del track (solo se actualiza con sim>=0.55 + margin, v5.1053):
            # matchear contra él NO es "fallback a la primera cara", es continuar el
            # track. Umbrales alineados con el loop principal (v5.1053) para que lo
            # aceptado aquí SOBREVIVA su validación (fin del bucle aceptar→rechazar):
            # multi >= 0.48 / single >= 0.38. Solo con >=3 fallos acumulados (el track ya
            # está roto; los 2 primeros no interfieren con la recuperación normal por
            # v5.1155 — RE-aplicar gates de tamaño/corte al rescate. Log real
            # (process_20260916_173055 video 1): mini-cara de fondo de 14x20px con
            # track_sim=0.99 (bbox estático 715,50) → el bucle principal la rechaza
            # SIEMPRE por el gate mini-cara v5.1052 (<32px) pero el rescate la
            # re-inyectaba en candidates cada 3 frames → RE-BUSQUEDA la aceptaba
            # (sim 0.99) → pintaba un swap diminuto → el loop la volvía a rechazar
            # (Lost) → ciclo aceptar-rechazar eterno: 96 frames sin swap estable
            # (59-155) + 64 Lost precedidos de RE-BUSQUEDA. El comentario anterior
            # ("el candidato viene del mismo bucle que ya los aplicó") era FALSO:
            # el bucle los rechaza con `continue` y NUNCA llegan a candidates.
            # Los mismos 3 gates del bucle (mini-cara + cortada + área vs última swap).
            if (re_fails + 1) >= 3 and tracking_emb is not None:
                _rescue_min = 0.48 if multi_face else 0.38
                for face in valid_faces:
                    if not hasattr(face, 'embedding') or face.embedding is None:
                        continue
                    _tsim = float(self._calculate_similarity(tracking_emb, face.embedding))
                    if _tsim >= _rescue_min:
                        # Gate mini-cara v5.1052 (idéntico al bucle y al accept de arriba)
                        _rw = face.bbox[2] - face.bbox[0] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                        _rh = face.bbox[3] - face.bbox[1] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                        if _rw < 32 or _rh < 32 or _rw * _rh < 1024:
                            continue
                        # Gate cara cortada v5.1066 (idéntico al accept de arriba)
                        if frame_shape is not None and face.bbox is not None and self._is_bbox_clipped_by_frame(face.bbox, frame_shape, allow_partial_overflow=True):
                            continue
                        # Gate de área vs última swap (idéntico: solo fuera de scene cut)
                        if not after_scene_cut:
                            _lbr = self._last_swap_bbox.get(video_basename) if hasattr(self, '_last_swap_bbox') else None
                            if _lbr is not None:
                                _lw = max(1, _lbr[2] - _lbr[0])
                                _lh = max(1, _lbr[3] - _lbr[1])
                                _ar = (_rw * _rh) / max(1.0, _lw * _lh)
                                if _ar < 0.06 or _ar > (100.0 if self._is_large_face(face.bbox, frame_shape) else 6.0):
                                    continue
                        area = self._largest_face_area(face)
                        candidates.append((_tsim, area, face))
                if candidates and frame_count % 25 == 1:
                    _rc = candidates[0]
                    print(
                        f"[TRACK] RE-BUSQUEDA frame {frame_count}: rescate por tracking_emb "
                        f"(track_sim={_rc[0]:.2f} >= {_rescue_min:.2f}) — identidad del track "
                        f"confirmada aunque ref_emb no matchee"
                    )
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
                # v5.1156: diagnóstico — si un candidato SUPERABA el umbral de sim
                # pero no llegó a candidates, murió en un gate invisible (mini-cara,
                # cortada o área vs última swap). El log plano "mejor_sim=0.41,
                # umbral=0.30" parece contradictorio y enmascara la causa real.
                _gates = []
                for _fg in valid_faces:
                    if not hasattr(_fg, 'embedding') or _fg.embedding is None:
                        continue
                    _fs = float(self._calculate_similarity(ref_emb, _fg.embedding))
                    _bk = self._bank_best_sim(_fg.embedding, video_basename)
                    if _bk is not None:
                        _fs = max(_fs, _bk)
                    if tracking_emb is not None:
                        _fs = max(_fs, float(self._calculate_similarity(tracking_emb, _fg.embedding)))
                    if _fs < min_sim - 0.005:
                        continue  # rechazo legítimo por sim (ya cubierto por el print)
                    _gw = _fg.bbox[2] - _fg.bbox[0] if (hasattr(_fg, 'bbox') and _fg.bbox is not None) else 0
                    _gh = _fg.bbox[3] - _fg.bbox[1] if (hasattr(_fg, 'bbox') and _fg.bbox is not None) else 0
                    if _gw < 32 or _gh < 32 or _gw * _gh < 1024:
                        _gates.append(f"mini-cara {_gw:.0f}x{_gh:.0f} sim={_fs:.2f}")
                    elif frame_shape is not None and _fg.bbox is not None and self._is_bbox_clipped_by_frame(_fg.bbox, frame_shape, allow_partial_overflow=True):
                        _gates.append(f"cortada sim={_fs:.2f} bbox={tuple(int(v) for v in _fg.bbox)}")
                    elif not after_scene_cut:
                        _lbr2 = self._last_swap_bbox.get(video_basename) if hasattr(self, '_last_swap_bbox') else None
                        if _lbr2 is not None:
                            _lw2 = max(1, _lbr2[2] - _lbr2[0])
                            _lh2 = max(1, _lbr2[3] - _lbr2[1])
                            _ar2 = (_gw * _gh) / max(1.0, _lw2 * _lh2)
                            if _ar2 < 0.06 or _ar2 > (100.0 if self._is_large_face(_fg.bbox, frame_shape) else 6.0):
                                _gates.append(f"area_ratio={_ar2:.1f} sim={_fs:.2f}")
                _gates_str = f", GATES: {'; '.join(_gates[:3])}" if _gates else ""
                print(
                    f"[TRACK] RE-BUSQUEDA sin match frame {frame_count} "
                    f"(mejor_sim={best_any:.2f}{track_sim_str}, "
                    f"umbral={min_sim:.2f}, caras={len(valid_faces)}, "
                    f"re_fails={re_fails}{spatial_info}){_gates_str}"
                )
                # v5.1070: Warning cada 100 frames cuando la similitud es muy baja
                if re_fails >= 100 and re_fails % 100 == 0 and best_any < 0.15:
                    print(
                        f"[WARNING] RE-BUSQUEDA frame {frame_count}: "
                        f"MEJOR SIMILITUD {best_any:.2f} tras {re_fails} intentos — "
                        f"LA CARA DEL SOURCE NO PARECE ESTAR EN ESTE VIDEO. "
                        f"Seleccione la cara correcta del vídeo en la UI."
                    )
            # v5.1027: After scene cut — fallback accept con tracking_emb mínimo 0.10
            # para evitar frames perdidos cuando el position_history se reseteó y
            # ref_emb da baja similitud (ej. frame 861 con ref=0.04 track=0.16).
            if after_scene_cut and tracking_emb is not None:
                for face in valid_faces:
                    if not hasattr(face, 'embedding') or face.embedding is None:
                        continue
                    # v5.1052: gate mini-caras también en el fallback post-corte
                    _fw3 = face.bbox[2] - face.bbox[0] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                    _fh3 = face.bbox[3] - face.bbox[1] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                    if _fw3 < 32 or _fh3 < 32 or _fw3 * _fh3 < 1024:
                        continue
                    # v5.1066: gate cortadas con overflow parcial permitido (primer plano)
                    if frame_shape is not None and face.bbox is not None and self._is_bbox_clipped_by_frame(face.bbox, frame_shape, allow_partial_overflow=True):
                        continue
                    track_sim = float(self._calculate_similarity(tracking_emb, face.embedding))
                    if track_sim >= (0.26 if multi_face else 0.20):  # v5.1053: 0.10→0.20/0.26 (exquisito)
                        area = self._largest_face_area(face)
                        candidates.append((track_sim, area, face))
            if not candidates and re_fails >= 10:
                # Con >=10 fallos ya se intentó la recuperación por la referencia
                # y por el embedding del track. Una similitud 0.22 con la referencia
                # no confirma identidad y este fallback podía saltarse los gates
                # de área del bucle principal para pintar a otra persona.
                if re_fails >= 30 and re_fails % 30 == 0:
                    best_fb_sim = max(
                        (float(self._calculate_similarity(ref_emb, face.embedding))
                         for face in valid_faces
                         if getattr(face, 'embedding', None) is not None),
                        default=-1.0,
                    )
                    print(
                        f"[WARNING] RE-BUSQUEDA frame {frame_count}: "
                        f"MEJOR SIMILITUD {best_fb_sim:.2f} tras {re_fails} intentos — "
                        "SIN IDENTIDAD CONFIRMADA: frame original. "
                        "Seleccione la cara correcta en la UI."
                    )
                return None
            if not candidates:
                return None

        if closeup_mode:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        else:
            candidates.sort(key=lambda item: item[0], reverse=True)
        best_sim, _best_area, best_face = candidates[0]
        if multi_face and len(candidates) >= 2:
            # v5.1053: EXQUISITO — margen alineado con `_validate_locked_identity`
            # (recovery 0.48/margin 0.08, normal 0.50/margin 0.10 del loop principal).
            margin = best_sim - candidates[1][0]
            need_margin = 0.08 if (closeup_mode or after_scene_cut) else 0.10
            if margin < need_margin:
                # v5.1053: el bypass v5.1003 (mayor área) SOLO es válido si ambos
                # candidatos son la MISMA cara física (detección duplicada a otra
                # escala). Dos personas reales con similitudes parecidas → rechazar
                # (hold), nunca quedarse con la más grande por arbitrariedad.
                same_physical = False
                bb_best = best_face.bbox if (hasattr(best_face, 'bbox') and best_face.bbox is not None) else None
                bb_second = candidates[1][2].bbox if (hasattr(candidates[1][2], 'bbox') and candidates[1][2].bbox is not None) else None
                if bb_best is not None and bb_second is not None:
                    try:
                        from roop.face_util_rotation import _same_face_boxes
                        same_physical = _same_face_boxes(bb_best, bb_second)
                    except Exception:
                        same_physical = False
                if same_physical:
                    best_face = max(candidates, key=lambda item: item[1])[2]
                    best_sim = max(c[0] for c in candidates)
                else:
                    if frame_count % 25 == 1:
                        print(
                            f"[TRACK] RE-BUSQUEDA ambigua frame {frame_count} "
                            f"({best_sim:.2f} vs {candidates[1][0]:.2f})"
                        )
                    return None

        # v5.1069b: ANTIPARPADEO en RE-BUSQUEDA — el modo espacial (sim < min_sim) aceptaba
        # candidatos LEJANOS del último swap con identidad débil y los pintaba vía
        # _apply_reacquired_face SIN pasar por el guard del loop principal.
        # Log real (maxmaximin f4207-4209): SCENE_CUT vel=996px → RE-BUSQUEDA global
        # (sim=0.27, mode=spatial, bbox diminuto lejos) → pintado → f4209 Lost = pop.
        # Un retorno legítimo del target tras corte confirma con sim >= min_sim (mode=emb)
        # o está cerca de la última posición conocida; lejos + débil = otra cara.
        # Los casos duros (perfiles con sim 0.15-0.22) siguen recuperándose vía el
        # umbral progresivo de v5.1005 (re_fails baja min_sim → mode pasa a emb).
        re_search_mode = "spatial" if best_sim < min_sim else "emb"
        if re_search_mode == "spatial" and best_sim < 0.30 and video_basename:
            _lb_re = getattr(self, '_last_swap_bbox', {}).get(video_basename)
            if _lb_re is not None:
                try:
                    _nb_re = [float(v) for v in best_face.bbox]
                    _ncx = (_nb_re[0] + _nb_re[2]) * 0.5
                    _ncy = (_nb_re[1] + _nb_re[3]) * 0.5
                    _rcx = (_lb_re[0] + _lb_re[2]) * 0.5
                    _rcy = (_lb_re[1] + _lb_re[3]) * 0.5
                    _jump_re = ((_ncx - _rcx) ** 2 + (_ncy - _rcy) ** 2) ** 0.5
                    _jthr_re = max(120.0, (_nb_re[2] - _nb_re[0]) * 1.30, (_nb_re[3] - _nb_re[1]) * 1.30)
                    if _jump_re > _jthr_re:
                        if frame_count % 25 == 1:
                            print(
                                f"[TRACK] Frame {frame_count}: RE-BUSQUEDA espacial lejana RECHAZADA "
                                f"(salto={_jump_re:.0f}px > {_jthr_re:.0f}, sim={best_sim:.2f} < 0.30) "
                                f"— hold, espera umbral progresivo/mode=emb"
                            )
                        return None
                except Exception:
                    pass

        # v5.1005: Reset re-search fail counter on success
        setattr(self, re_fail_attr, 0)

        tag = "closeup" if closeup_mode else "global"
        try:
            _rb = [int(v) for v in best_face.bbox]
        except Exception:
            _rb = [-1, -1, -1, -1]
        print(f"[TRACK] RE-BUSQUEDA {tag} frame {frame_count} (sim={best_sim:.2f}, mode={re_search_mode}, bbox=({_rb[0]},{_rb[1]},{_rb[2]},{_rb[3]}))")
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
            frame_shape=frame_shape,
        )

    def _resolve_swap_embedding(self, source_face, target_face, is_profile=False, video_basename=None):
        """Embedding para inswapper: siempre usa el embedding de la cara source seleccionada
        (random o match) al 100%, sin diluir con master embedding."""
        src_emb = getattr(source_face, 'embedding', None)
        if src_emb is None:
            return source_face

        src_arr = np.array(src_emb, dtype=np.float32)
        if src_arr.ndim > 1:
            src_arr = src_arr.flatten()
        norm_src = np.linalg.norm(src_arr)
        if norm_src > 0:
            src_arr = src_arr / norm_src

        class _SourceFaceWrap:
            embedding = src_arr.tolist()

        return _SourceFaceWrap()

    # La composición de perfiles usa este umbral para comprobar identidad.
    # No limita la elección automática de la mejor source del banco.
    _SOURCE_IDENTITY_SIM_THRESH = 0.60

    def _source_min_side(self, face):
        bbox = getattr(face, 'bbox', None)
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return 0.0
        return min(max(0.0, x2 - x1), max(0.0, y2 - y1))

    def _cluster_source_candidates(self, candidates):
        """Agrupa caras de origen con coseno ArcFace >= 0.60."""
        if not candidates:
            return []
        if len({c['embedding'].size for c in candidates}) != 1:
            return [list(candidates)]
        embeddings = np.stack([c['embedding'] for c in candidates])
        peers = embeddings @ embeddings.T >= 0.60
        remaining = set(range(len(candidates)))
        groups = []
        while remaining:
            # Start from the strongest consensus, and assign each source once.
            # The old loop could count the same face in several identities.
            seed = max(remaining, key=lambda i: (
                sum(bool(peers[i, j]) for j in remaining),
                candidates[i]['quality'], -candidates[i]['idx'],
            ))
            members = [j for j in sorted(remaining) if peers[seed, j]]
            groups.append([candidates[j] for j in members])
            remaining.difference_update(members)
        return groups

    def _select_source_face_by_similarity(self, target_face, candidate_faces, is_profile=False,
                                          reference_embeddings=None, quality_first=False):
        """
        Selecciona la mejor coincidencia de todo el banco de origen.
        """
        try:
            if not candidate_faces:
                return None
            
            raw_refs = (reference_embeddings if reference_embeddings is not None
                        else [getattr(target_face, 'embedding', None)])
            valid_refs = []
            for ref in raw_refs:
                if ref is None:
                    continue
                arr = np.asarray(ref, dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(arr)
                if arr.size and np.isfinite(arr).all() and norm > 0:
                    valid_refs.append(arr / norm)
            if not valid_refs:
                print("[SELECT_SOURCE] Target sin embedding — sin fallback a otra source")
                return None
            
            candidates = []
            for idx, src_face in enumerate(candidate_faces):
                src_emb = getattr(src_face, 'embedding', None)
                if src_emb is None:
                    continue
                src_arr = np.asarray(src_emb, dtype=np.float32).reshape(-1)
                norm = np.linalg.norm(src_arr)
                if not np.isfinite(src_arr).all() or norm == 0:
                    continue
                unit_src = src_arr / norm
                similarities = [float(np.dot(ref, unit_src)) for ref in valid_refs
                                if ref.size == unit_src.size]
                if not similarities:
                    continue
                sim = max(similarities)
                quality = self._score_source_face_quality(src_face)
                if not np.isfinite(sim) or not np.isfinite(quality):
                    continue
                candidates.append({
                    'face': src_face,
                    'embedding': unit_src,
                    'sim': sim,
                    'quality': quality,
                    'idx': idx,
                    'is_profile': ProcessMgr._is_profile_face(getattr(src_face, 'kps', None)),
                })
            
            if not candidates:
                print("[SELECT_SOURCE] No source con embedding — sin fallback a otra source")
                return None

            if not quality_first:
                # The uploaded bank can contain different source identities.
                # When the destination is the same person, cosine still picks
                # that identity. When every cosine stays in the cross-person
                # noise band, the largest identity in the bank wins and a
                # tiny background detection cannot lock the video.
                best_sim = max(c['sim'] for c in candidates)
                if best_sim < 0.55:
                    groups = self._cluster_source_candidates(candidates)
                    largest_size = max(len(group) for group in groups)
                    largest_groups = [group for group in groups
                                      if len(group) == largest_size]
                    strongest_group_sim = max(
                        max(item['sim'] for item in group)
                        for group in largest_groups
                    )
                    plausible_groups = [
                        group for group in largest_groups
                        if max(item['sim'] for item in group) >= strongest_group_sim - 0.08
                    ]
                    identity_views = max(
                        plausible_groups,
                        key=lambda group: (
                            float(np.mean([item['quality'] for item in group])),
                            max(item['sim'] for item in group),
                        ),
                    )
                    large_views = [c for c in identity_views
                                   if self._source_min_side(c['face']) >= 40.0]
                    if large_views:
                        identity_views = large_views
                    anchor = max(identity_views, key=lambda c: c['quality'])
                    quality_floor = anchor['quality'] - 1.15
                    clean_views = [c for c in identity_views if c['quality'] >= quality_floor]
                    best_candidate = max(
                        clean_views,
                        key=lambda c: (
                            c['is_profile'] == bool(is_profile),
                            c['quality'],
                            c['sim'],
                        ),
                    )
                else:
                    near_best = [c for c in candidates if c['sim'] >= best_sim - 0.08]
                    def same_identity(seed):
                        return [c for c in candidates
                                if c['embedding'].size == seed['embedding'].size
                                and float(np.dot(c['embedding'], seed['embedding'])) >= 0.60]
                    anchors = [(seed, same_identity(seed)) for seed in near_best]
                    anchor, identity_views = max(
                        anchors,
                        key=lambda item: (item[0]['sim']
                                          + min(0.05, 0.012 * np.log1p(len(item[1]))),
                                          item[0]['sim']),
                    )
                    large_views = [c for c in identity_views
                                   if self._source_min_side(c['face']) >= 40.0]
                    if large_views:
                        identity_views = large_views
                    if len(identity_views) > 1:
                        best_quality = max(c['quality'] for c in identity_views)
                        clean_views = [c for c in identity_views
                                       if c['quality'] >= best_quality - 0.85]
                        best_candidate = max(
                            clean_views,
                            key=lambda c: (c['quality']
                                           + (0.35 if c['is_profile'] == bool(is_profile) else 0.0)
                                           + 0.10 * c['sim'], c['sim']),
                        )
                    else:
                        best_candidate = identity_views[0]
                source_name = getattr(self, '_source_face_name_by_id', {}).get(id(best_candidate['face']), '')
                print(
                    f"[SELECT_SOURCE] Mejor de {len(candidates)} source(s) para vídeo: "
                    f"#{best_candidate['idx'] + 1} {source_name} "
                    f"(sim={best_candidate['sim']:.4f}, qual={best_candidate['quality']:.2f}, "
                    f"perfil={best_candidate['is_profile']}, "
                    f"identidad={len(identity_views)}, ancla_sim={anchor['sim']:.4f})",
                    flush=True,
                )
                return best_candidate['face']

            # ArcFace measures identity. Source and destination are different
            # people, so their highest cosine is a poor way to select a clean
            # view of the source identity. Remove isolated faces when the bank
            # has one clear source identity, then favor detection quality.
            bank_key = tuple(id(c['face']) for c in candidates)
            consensus_cache = getattr(self, '_auto_source_consensus_cache', None)
            if consensus_cache is None:
                consensus_cache = self._auto_source_consensus_cache = {}
            trusted_ids = consensus_cache.get(bank_key)
            if trusted_ids is None:
                trusted_ids = set(bank_key)
                if (len(candidates) >= 3 and
                        len({c['embedding'].size for c in candidates}) == 1):
                    groups = self._cluster_source_candidates(candidates)
                    dominant = max(groups, key=lambda group: (
                        len(group),
                        float(np.mean([c['quality'] for c in group])),
                    ))
                    trusted_ids = {id(c['face']) for c in dominant}
                consensus_cache.clear()
                consensus_cache[bank_key] = trusted_ids
            trusted = [c for c in candidates if id(c['face']) in trusted_ids]
            best_quality = max(c['quality'] for c in trusted)
            high_quality = [c for c in trusted if c['quality'] >= best_quality - 1.15]
            pose_matches = [c for c in high_quality if c['is_profile'] == bool(is_profile)]
            pool = pose_matches or high_quality
            def rank(c):
                # Pose of a good view beats raw sharpness. The profile
                # penalty only applies when the destination is frontal, so a
                # matching profile is not erased by a sharper front view.
                pose_match = c['is_profile'] == bool(is_profile)
                penalty = 0.35 if (c['is_profile'] and not is_profile) else 0.0
                return (1 if pose_match else 0, c['quality'] - penalty, 0.20 * c['sim'])
            best_candidate = max(pool, key=lambda c: (rank(c), c['sim']))
            source_name = getattr(self, '_source_face_name_by_id', {}).get(id(best_candidate['face']), '')
            print(
                f"[SELECT_SOURCE] Mejor de {len(candidates)} source(s): "
                f"#{best_candidate['idx'] + 1} {source_name} "
                f"(sim={best_candidate['sim']:.4f}, qual={best_candidate['quality']:.2f}, "
                f"perfil_source={best_candidate['is_profile']}, perfil_destino={is_profile}, "
                f"identidad={len(trusted)}/{len(candidates)}, calidad={len(high_quality)})",
                flush=True,
            )
            return best_candidate['face']
        except Exception as e:
            print(f"[ERROR] _select_source_face_by_similarity: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _select_source_face(self, target_face, candidate_faces, face_swap_mode, video_path=None, is_profile=False):
        """
        Lógica de selección de cara origen:
        - En fotos aleatorias: elige una cara del banco para cada destino.
        - En fotos normales: compara cada destino con todas las caras de origen.
        - En vídeo: compara con la cara de destino seleccionada y conserva la
          misma cara de origen durante todo el vídeo.
        """
        try:
            if not candidate_faces:
                return None

            # Resolver referencia del destino si existe
            target_emb = getattr(target_face, 'embedding', None)
            ref_emb = target_emb
            video_refs = []
            video_basename = os.path.basename(video_path) if video_path else None
            video_key = f"selected_face_ref_{video_basename}" if video_basename else None
            if video_key and hasattr(roop.globals, 'selected_face_references'):
                video_refs = self._video_reference_bank(video_basename, video_path)
                if video_refs:
                    ref_emb = video_refs[0].get('embedding')

            if face_swap_mode in ['selected', 'selected_faces', 'auto', 'selected_faces_frame']:
                if (getattr(roop.globals, 'source_random', False)
                        and not getattr(self.options, 'current_video_path', None)):
                    # SOURCE_RANDOM: elige una cara aleatoria para este target,
                    # si es baja calidad busca parecidas (misma persona, mejor calidad)
                    import random
                    scored = []
                    for idx, src in enumerate(candidate_faces):
                        src_emb = getattr(src, 'embedding', None)
                        sim = 0.0
                        if ref_emb is not None and src_emb is not None:
                            sim = self._cosine_similarity(ref_emb, src_emb)
                        qual = getattr(src, 'det_score', 0.5)
                        face_w = int(getattr(src, 'bbox', [0,0,0,0])[2] - getattr(src, 'bbox', [0,0,0,0])[0])
                        size_bonus = min(1.0, face_w / 200.0)
                        quality = qual * 0.7 + size_bonus * 0.3
                        combined = quality * 0.60 + sim * 0.40
                        scored.append({'face': src, 'sim': sim, 'quality': quality, 'combined': combined, 'idx': idx})

                    scored.sort(key=lambda c: c['combined'], reverse=True)
                    best_qual = scored[0]['quality'] if scored else 0.0
                    min_qual = max(0.35, best_qual * 0.80)
                    pool = [c for c in scored if c['quality'] >= min_qual] or scored[:max(1, min(12, len(scored)))]
                    top_n = max(1, min(5, len(pool)))
                    picked_c = random.choice(pool[:top_n])
                    picked_face = picked_c['face']
                    picked_quality = picked_c['quality']

                    if picked_quality < 0.55:
                        picked_emb = getattr(picked_face, 'embedding', None)
                        if picked_emb is not None:
                            similar = []
                            for c in scored:
                                if c['quality'] <= picked_quality:
                                    continue
                                c_emb = getattr(c['face'], 'embedding', None)
                                if c_emb is not None:
                                    c_sim = self._cosine_similarity(picked_emb, c_emb)
                                    if c_sim > 0.70:
                                        similar.append(c)
                            if similar:
                                similar.sort(key=lambda c: c['quality'], reverse=True)
                                picked_face = similar[0]['face']
                                print(f"[SOURCE_RANDOM] baja calidad ({picked_quality:.2f}) -> reemplazada por similar con qual={similar[0]['quality']:.2f}")

                    print(f"[SOURCE_RANDOM] target -> #{picked_c['idx']+1}/{len(scored)} (qual={picked_c['quality']:.2f}, sim={picked_c['sim']:.4f})")
                    return picked_face

                # Match all selected destination views for this video and lock
                # the best source in process_frame for the duration of the render.
                video_embeddings = ([ref.get('embedding') for ref in video_refs]
                                    if getattr(self.options, 'current_video_path', None) else [])
                return self._select_source_face_by_similarity(
                    target_face, candidate_faces, is_profile=is_profile,
                    reference_embeddings=video_embeddings if video_embeddings else None,
                    quality_first=not bool(getattr(self.options, 'current_video_path', None)),
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
            selected_face = None
            selection_method = "auto"

            face_ref_data = self._video_reference_bank(video_basename, video_path)
            user_has_selected_face = bool(face_ref_data)

            # Guardar flag para saber si el usuario seleccionó una cara
            setattr(self, f'_user_selected_face_{video_basename}', user_has_selected_face)

            if not user_has_selected_face:
                print(
                    f"[SELECTED_FACES_FRAME] Sin cara seleccionada para {video_basename} "
                    "— sin fallback auto (hold hasta selección o match fuerte)"
                )

            if face_ref_data:
                # Banco destino: se prueba cada ref (frontal+perfil) y gana el mejor.
                _bank_refs = self._as_ref_list(face_ref_data)
                _primary = _bank_refs[0] if _bank_refs else {}
                user_selected_face = _primary.get('face_obj')
                _user_embedding = _primary.get('embedding')
                _user_bbox = _primary.get('bbox')

                if user_selected_face is not None:
                    best_match = None
                    best_score = -1

                    for face in valid_faces:
                        if not hasattr(face, 'embedding') or face.embedding is None:
                            continue

                        score = -1.0
                        for _br in _bank_refs:
                            _be = _br.get('embedding')
                            _bb = _br.get('bbox')
                            if _be is not None:
                                score = max(score, self._calculate_similarity(_be, face.embedding))
                            elif _bb is not None:
                                score = max(score, self._bbox_iou(_bb, face.bbox) if face.bbox is not None else 0)
                        if not _bank_refs:
                            score = 0

                        if score > best_score:
                            best_score = score
                            best_match = face

                    if best_match and best_score > 0.35:
                        selected_face = best_match
                        selection_method = f"usuario (score={best_score:.2f})"
                    else:
                        # v5.1139 — SIN fallback en VÍDEO (pedido explícito: "si no se
                        # encuentra la cara NO puede ir a la primera que encuentre").
                        # El fallback-mayor anclaba el tracking a la cara MÁS GRANDE del
                        # primer frame: con varias personas era FRECUENTEMENTE la cara
                        # equivocada y el track entero swappeaba a la persona equivocada.
                        # En single-face la única cara del frame ES el protagonista (sin
                        # ambigüedad); en multi-face se ESPERA al frame donde la cara
                        # seleccionada aparezca (sim >= 0.35) — setup re-intenta cada
                        # frame hasta entonces, frame original sin swap mientras tanto.
                        _non_tiny = [
                            f for f in valid_faces
                            if hasattr(f, 'bbox') and f.bbox is not None
                            and (f.bbox[2] - f.bbox[0]) >= 32 and (f.bbox[3] - f.bbox[1]) >= 32
                        ]
                        if len(_non_tiny) == 1:
                            selected_face = _non_tiny[0]
                            selection_method = "auto-única-cara (sin ambigüedad)"
                            print(
                                f"[SELECTED_FACES_FRAME] Sin match de identidad ({best_score:.2f} < 0.35) "
                                f"pero hay UNA sola cara en el frame — ancla sin ambigüedad"
                            )
                        else:
                            print(
                                f"[SELECTED_FACES_FRAME] Sin match de identidad ({best_score:.2f} < 0.35) "
                                f"con {len(valid_faces)} cara(s) en el frame — NO se ancla ninguna "
                                f"(esperando la cara seleccionada; frame sin swap)"
                            )

            if selected_face is None:
                # Keep looking on later frames when the user-selected person
                # has not appeared yet. The tracker can also acquire them.
                if not user_has_selected_face:
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

    def _validate_locked_identity(self, valid_faces, ref_embedding, candidate_face, multi_face=False, fallback_embedding=None, closeup_recovery=False, is_profile=False, bank_embeddings=None):
        if ref_embedding is None or candidate_face is None:
            return True, 1.0
        # Banco destino: la ref efectiva es la del banco que mejor matchea al
        # candidato (frontal↔frontal, perfil↔perfil). Margen/gates intactos.
        if bank_embeddings:
            try:
                _cand_emb = getattr(candidate_face, 'embedding', None)
                _best_s, _best_r = -1.0, None
                for _br in bank_embeddings:
                    _s = float(self._calculate_similarity(_br, _cand_emb))
                    if _s > _best_s:
                        _best_s, _best_r = _s, _br
                if _best_r is not None and _best_s > float(self._calculate_similarity(ref_embedding, _cand_emb)):
                    ref_embedding = _best_r
            except Exception:
                pass
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
        # v5.1053: EXQUISITO — multi-face: la identidad es la ÚNICA señal fiable (varias personas
        # en frame pueden compartir posición/tamaño). Subidos min_sim/margin: nunca anclar otra persona.
        # v5.1070e: Perfiles — el embedding de perfil es naturalmente más débil (ángulo extremo
        # reduce la similitud cosine 0.10-0.20 vs frontal). Relajar umbrales un 15% para perfiles
        # para no perder el track en vídeos con 80%+ perfiles (log 094937: 21 Lost con score
        # 0.39-0.50, todos rechazados por min_sim=0.50 en multi-face normal).
        recovery_mode = closeup_recovery or (hasattr(self, '_tracking_recovery') and getattr(self, '_tracking_recovery') > 0)
        profile_factor = 0.85 if is_profile else 1.0
        if recovery_mode and not multi_face:
            min_sim = 0.36 * profile_factor
            min_margin = 0.03
        elif recovery_mode and multi_face:
            min_sim = 0.42 * profile_factor
            min_margin = 0.04
        else:
            # v5.1074: Bajado min_sim multi-face 0.50→0.42 para aceptar caras con
            # score decente (0.42-0.49) que antes se rechazaban → ghost/hold → parpadeo.
            # min_margin 0.06→0.03: el margin solo mide separación vs otras caras,
            # no calidad del match. Un margin 0.02-0.05 con score alto = misma persona.
            min_sim = 0.42 * profile_factor
            min_margin = 0.03
        # v5.1074b: Relajado top_sim check de -0.05 a -0.15. El check original
        # requería que cand_sim estuviera dentro de 0.05 de top_sim (la cara con
        # mayor similitud en el frame). Pero el candidato es seleccionado por
        # best_score (sim + distancia + tamaño), no por sim pura. Si hay otra
        # cara en el frame con sim más alta (poster, otra persona parecida), el
        # candidato correcto (con mejor best_score por proximidad) se rechazaba
        # aunque tuviera sim decente (0.66 vs top 0.95) → ghost/hold → parpadeo.
        ok = (
            cand_sim >= min_sim
            and cand_sim >= top_sim - 0.15
            and margin >= min_margin
        )
        # v5.1074b: Retry aún más laxo — aceptar si candidato tiene sim razonable
        if not ok and multi_face:
            ok = (
                cand_sim >= min_sim - 0.08
                and cand_sim >= top_sim - 0.25
                and margin >= 0.04  # v5.1100: 0.02→0.04 — margin 0.02 en multi-face = dos personas indistinguibles → NO aceptar (caer a Lost/hold), era la fuga de "cara donde no toca"
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
        frame_shape=None,
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
            # v5.1052b: Gate mini-caras en re-enganche estricto — misma persona en un
            # póster/foto da sim alta (emb no lo distingue), solo el tamaño lo delata.
            _fw_s = face.bbox[2] - face.bbox[0] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
            _fh_s = face.bbox[3] - face.bbox[1] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
            if _fw_s < 32 or _fh_s < 32 or _fw_s * _fh_s < 1024:
                continue
            # v5.1066: gate cortadas con overflow parcial permitido (primer plano)
            if face.bbox is not None and frame_shape is not None and self._is_bbox_clipped_by_frame(face.bbox, frame_shape, allow_partial_overflow=True):
                continue
            sim = float(self._calculate_similarity(ref_embedding, face.embedding))
            # Banco destino: max-sim por cara (mismo criterio estricto, mejor ref).
            _bank_sim = self._bank_best_sim(face.embedding, video_basename)
            if _bank_sim is not None:
                sim = max(sim, _bank_sim)
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
                # v5.1022b: Multi-face con margen mínimo relajado
                # v5.1100: 0.03→0.05 — margin 0.03 entre dos caras distintas = ambiguo → hold
                if multi_face and margin >= 0.05:
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
        # The detector already located the face in the current frame. Blending
        # its landmarks with the last lost frame shifts the swap off the face,
        # especially after a camera move or a scene cut.
        out = face
        setattr(self, assigned_attr, out)
        setattr(self, f'_reference_face_selected_{video_basename}', True)
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
            _use_fallback = not user_selected

            if not valid_faces:
                return None

            # v5.1057: GATE ANTI CARA MUTADA — rechazar detecciones cuyo bbox esté
            # mayoritariamente FUERA del frame. Log real (VID_20230105, frames 396-401):
            # bbox=(-166,-195,267,325) en 720x1280 = 38% del área visible → el pipeline
            # swapeaba un crop de relleno (fuera de la imagen) → "cara mutada" al
            # re-adquirir el track. Caras legítimas pegadas al borde (p.ej. y1=-36 con
            # 82% visible) siguen pasando el gate.
            # v5.1059: GATE DE ESCALA ANTI ZOOM — log real (Sabrina 4K, frames 647-655):
            # durante el zoom el detector alterna caras GIGANTES cortadas por el borde
            # (bbox=(2082,-477,3392,939) = 22% del frame, 34% fuera) con la cara real
            # de 111px → el swap "explota y se deforma". Regla: si el bbox ocupa >8%
            # del frame, exigir visible ≥75% Y det_score ≥0.70 (closeup legítimo tiene
            # det alto; el bbox inflado por el zoom tiene det bajo o está cortado).
            if current_frame is not None:
                _fh_f, _fw_f = current_frame.shape[:2]
                _frame_area = max(1.0, _fh_f * _fw_f)
                _kept_faces = []
                for _f in valid_faces:
                    try:
                        _bx1, _by1, _bx2, _by2 = [float(v) for v in _f.bbox]
                        _bw = max(1.0, _bx2 - _bx1)
                        _bh = max(1.0, _by2 - _by1)
                        _ix = max(0.0, min(_fw_f, _bx2) - max(0.0, _bx1))
                        _iy = max(0.0, min(_fh_f, _by2) - max(0.0, _by1))
                        _visible_frac = (_ix * _iy) / (_bw * _bh)
                        _ok = _visible_frac >= 0.50
                        _area = _bw * _bh
                        if _ok and _area > 0.08 * _frame_area:
                            _det = float(getattr(_f, 'det_score', 1.0))
                            if _visible_frac < 0.75:
                                _ok = False
                                _why = "gigante cortada por borde del frame"
                            elif _det < 0.70:
                                # v5.1142: RESCATE de la cara única casi-completa en
                                # el frame. El requisito det>=0.70 era para el zoom con
                                # ALTERNANCIA (Sabrina 4K: gigante inflada VS cara real
                                # pequeña en el MISMO frame). En POV/closeup con cámara
                                # mediocra el det legítimo es 0.30-0.69 → el gate mataba
                                # la cara en 76 frames (log Aggressivebbc: Lost 53 vs
                                # Success 88, saltos 20% por el ciclo aceptar→rechazar).
                                # Rescate SOLO si: casi-completa (>=85% visible), det
                                # >=0.30, y es la MISMA cara física (IoU>=0.40 con la
                                # última swap) o la única del frame. El caso zoom-inflado
                                # (fuerte corte + cara real alternativa presente) sigue
                                # rechazado como antes.
                                _rescued = False
                                if _det >= 0.30 and _visible_frac >= 0.85:
                                    _last_bb_gate = self._last_swap_bbox.get(video_basename) if hasattr(self, '_last_swap_bbox') and isinstance(self._last_swap_bbox, dict) else None
                                    if _last_bb_gate is not None:
                                        try:
                                            _lx1, _ly1, _lx2, _ly2 = [float(v) for v in _last_bb_gate]
                                            _iw = max(0.0, min(_bx2, _lx2) - max(_bx1, _lx1))
                                            _ih = max(0.0, min(_by2, _ly2) - max(_by1, _ly1))
                                            _inter = _iw * _ih
                                            _union = (_bw * _bh) + max(1.0, (_lx2 - _lx1) * (_ly2 - _ly1)) - _inter
                                            if _inter / max(1.0, _union) >= 0.40:
                                                _rescued = True
                                        except Exception:
                                            _rescued = False
                                    if not _rescued:
                                        _rescued = self._selected_target_is_unique(
                                            _f, valid_faces, video_basename,
                                        )
                                    if not _rescued and not self._get_user_reference_embeddings(video_basename):
                                        _rescued = len(valid_faces) == 1
                                if not _rescued:
                                    _ok = False
                                    _why = f"gigante con det bajo ({_det:.2f})"
                                else:
                                    _why = None
                            else:
                                _why = None
                        else:
                            _why = None
                        if _ok:
                            _kept_faces.append(_f)
                        elif getattr(self, f'_frame_count_{video_basename}', 0) % 100 == 0:  # v5.1117: frame_count aún no asignado aquí (UnboundLocalError comido por el except = gate anti-mutada desactivado)
                            print(
                                f"[TRACK] Frame {getattr(self, f'_frame_count_{video_basename}', 0)}: bbox "
                                f"({_bx1:.0f},{_by1:.0f},{_bx2:.0f},{_by2:.0f}) "
                                f"solo {_visible_frac*100:.0f}% dentro / área {_area/_frame_area*100:.0f}% "
                                f"-> descartado ({_why or 'evita cara mutada'})"
                            )
                    except Exception:
                        _kept_faces.append(_f)
                valid_faces = _kept_faces
                if not valid_faces:
                    frame_count_attr = f'_frame_count_{video_basename}'
                    setattr(self, frame_count_attr, getattr(self, frame_count_attr, 0) + 1)
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
                    frame_shape=current_frame.shape,
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
                    _bank0 = self._video_reference_bank(video_basename, video_path)
                    user_embedding = _bank0[0].get('embedding') if _bank0 else None
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
                        
                        # v5.1052: Gate mini-caras (póster/foto) en tracking NORMAL —
                        # el gate v5.1051 solo cubría _reacquire_face_global, pero el
                        # póster (26x34px) re-entró por aquí en frame 809 (Success,
                        # margin=0.65: es de la misma persona → embedding no lo
                        # distingue, solo el tamaño lo delata). Mismo umbral que reacquire.
                        _fw_c = face.bbox[2] - face.bbox[0] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                        _fh_c = face.bbox[3] - face.bbox[1] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                        if _fw_c < 32 or _fh_c < 32 or _fw_c * _fh_c < 1024:
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

                            # Banco destino (frontal+perfil): max-sim — el perfil
                            # matchea su ref aunque contra la frontal dé bajo.
                            _bank_sim = self._bank_best_sim(face.embedding, video_basename)
                            if _bank_sim is not None:
                                emb_score = max(emb_score, _bank_sim)

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
                        min_track_score = 0.32 if multi_face else 0.22  # v5.1053: multi 0.25→0.32 (exquisito)
                    elif lost_count > 0:
                        min_track_score = max(0.30 if multi_face else 0.15, min_track_score * 0.50)  # v5.1053: multi floor 0.15→0.30 (exquisito)
                    # v5.1070e: Perfiles — el score del tracker también es más bajo en ángulo extremo.
                    # Relajar 15% para no perder track en vídeos con 80%+ perfiles.
                    if best_match is not None:
                        _bm_kps = getattr(best_match, 'kps', None)
                        if _bm_kps is not None and ProcessMgr._is_profile_face(_bm_kps, getattr(best_match, 'det_score', None)):
                            min_track_score *= 0.85
                    identity_ok = True
                    id_margin = 1.0
                    if best_match and original_embedding is not None:
                        # v5.1005: Pass tracking_embedding as fallback for better match
                        fb_emb = getattr(self, f'_tracking_embedding_{video_basename}', None)
                        closeup_recovery = scene_state.get('is_close_up') and lost_count > 0
                        self._tracking_recovery = lost_count  # v5.1033: pasar a validate para relajar umbrales
                        # v5.1070e: Detectar si el candidato es perfil (embedding más débil)
                        _cand_kps = getattr(best_match, 'kps', None)
                        _cand_is_profile = ProcessMgr._is_profile_face(_cand_kps, getattr(best_match, 'det_score', None)) if _cand_kps is not None else False
                        identity_ok, id_margin = self._validate_locked_identity(
                            valid_faces, original_embedding, best_match,
                            multi_face=multi_face, fallback_embedding=fb_emb,
                            closeup_recovery=closeup_recovery,
                            is_profile=_cand_is_profile,
                            bank_embeddings=self._get_user_reference_embeddings(video_basename),
                        )

                    # v5.1004: Spatial proximity bypass — si la cara está prácticamente
                    # en la misma posición que el ancla, aceptar directamente porque
                    # es la misma persona (indetectable a 60fps). El embedding puede
                    # fallar por iluminación/expresión pero la posición no engaña.
                    spatial_lock = best_match and best_distance < min(150 if not scene_state.get('is_close_up') else 220, base_threshold * (0.35 if not scene_state.get('is_close_up') else 0.35))  # v5.1069d: closeup 0.50→0.35 (menos permisivo)
                    # v5.1053: EXQUISITO — en multi-face el spatial_lock NO puede saltarse la
                    # identidad (dos personas pueden ocupar posiciones adyacentes/superpuestas).
                    # Requiere identity_ok + score sólido.
                    if multi_face:
                        spatial_lock = spatial_lock and identity_ok and best_score > max(0.20, min_track_score * 0.5)
                    # v5.1069: ANTIPARPADEO — el spatial_lock de single-face NO tenía suelo de
                    # score y pintaba candidatos con similitud compuesta NEGATIVA o nula.
                    # Log real (maxmaximin 092753 f6893/f6903): Ghost Activo → Success
                    # (score=-0.11, dist=176-184px, closeup cap 220px) = cara/contenido
                    # equivocado pintado 1 frame → siguiente frame el gate normal lo rechaza
                    # (Lost) → ghost → otro pop → CICLO accept/reject = parpadeo visible
                    # justo cuando el target no está siendo seguido.
                    # Un follow legítimo de la misma persona SIEMPRE suma el término de
                    # distancia (hasta 0.40/0.22 de peso) → score > 0.10. Por debajo de
                    # 0.06 el embedding rechazó al candidato Y además está lejos → basura.
                    if spatial_lock and best_score <= 0.06:
                        spatial_lock = False
                    # v5.1063: rechazar como Success una cara CORTADA por el borde del frame
                    # cuando la identidad es débil (margin<0.12 en multi-face, o score bajo).
                    # Una esquina de frame con bbox parcial (p.ej. y1=-47) + bajo score +
                    # margin débil = cara que SALIÓ del plano / candidato de giro, no el follow.
                    # Aceptarla congelaba el swap en la esquina → "recuadro/congelación" en el
                    # plano siguiente (log ItsAnalGirls 163220: bbox=(589,-47,686,98) score=0.35,
                    # margin=0.08 aceptado como Success 8 frames → nunca se recupera).
                    # Forzamos Lost → ghost/hold con fade (v5.1054c) → escape hatch → re-acquire.
                    # v5.1065: DURANTE la ventana de plan-change/scene-cut (plan_smooth_until,
                    # extendida +24 frames), NINGÚN bbox cortado se acepta como Success, sin
                    # importar margin ni score. Log real (bnwofamily 073534): Frame 1232 Success
                    # (score=0.49, dist=187.5px, bbox=(-48,363,61,533), margin=0.22) — bbox con
                    # x1=-48 fuera del frame aceptado justo tras PLAN_CHANGE (hasta frame 1241)
                    # → el swap anclaba a una media-cara/esquina y perdía el track. La ventana
                    # ya existe como estado (`_plan_change_smooth_until`); el gate solo la usa
                    # para NO dejar pasar cortadas mientras el plano está cambiando.
                    if (
                        current_frame is not None
                        and best_match is not None
                        and self._is_bbox_clipped_by_frame(best_match.bbox, current_frame.shape)
                    ):
                        _recent_plan_change = frame_count <= getattr(self, f'_plan_change_smooth_until_{video_basename}', 0)
                        # v5.1065: durante plan-change, una bbox cortada SOLO se mantiene si es la
                        # cara en su sitio con score sólido (spatial_lock AND score>=0.70). Log real:
                        # frames 571-573 (y1=-55, score 0.78-0.95, dist<54px) = cara legítima en el
                        # borde durante zoom → se MANTIENE; frame 1232 (x1=-48, score 0.49, dist=187px)
                        # = media-cara en esquina tras PLAN_CHANGE → se rechaza y RE-BUSQUEDA.
                        if _recent_plan_change and not (spatial_lock and best_score >= 0.70):
                            if frame_count % 25 == 1:
                                _b = [int(v) for v in best_match.bbox]
                                print(
                                    f"[TRACK] Frame {frame_count}: candidato cortado RECHAZADO tras plan-change "
                                    f"(score={best_score:.2f}, margin={id_margin:.2f}, bbox=({_b[0]},{_b[1]},{_b[2]},{_b[3]})) — RE-BUSQUEDA"
                                )
                            best_match = None
                            identity_ok = False
                            id_margin = 0.0
                        elif (
                            (multi_face and id_margin is not None and id_margin < 0.12)
                            or (best_score < min_track_score + 0.12)
                        ) and not spatial_lock:
                            if frame_count % 25 == 1:
                                _b = [int(v) for v in best_match.bbox]
                                print(
                                    f"[TRACK] Frame {frame_count}: candidato cortado RECHAZADO "
                                    f"(score={best_score:.2f}, margin={id_margin:.2f}, bbox=({_b[0]},{_b[1]},{_b[2]},{_b[3]})) — ghost/hold/fade"
                                )
                            best_match = None
                            identity_ok = False
                            id_margin = 0.0
                    # v5.1069: ANTIPARPADEO — guard de SALTO DE CARA antes del gate de
                    # aceptación. Logs reales (pops cuantificados): edxvidx 222035 = 67
                    # Success débiles aislados (48 con salto entre caras), maxmaximin 42/29,
                    # femdom244 31/14. Mecanismo: en single-face sin gate de identidad, el
                    # loop acepta una cara LEJANA (>1.15x su tamaño) con sim cruda débil
                    # (<0.30) — otra persona/cara diminuta/póster — la pinta 1 frame y al
                    # siguiente la rechaza (Lost) → alterna swap/original = parpadeo.
                    # Un follow legítimo se mueve <1 tamaño de cara por frame, o si hay
                    # movimiento brusco real dispara SCENE_CUT (reset + RE-BUSQUEDA con sus
                    # propios gates). Rechazar aquí cae al camino Lost → ghost → hold-fade
                    # (v5.1054c, continuidad visual) → RE-BUSQUEDA.
                    if (
                        best_match is not None
                        and current_frame is not None
                        and original_embedding is not None
                        and getattr(best_match, 'embedding', None) is not None
                    ):
                        try:
                            _nb = [float(v) for v in best_match.bbox]
                            _prev_af = getattr(self, assigned_attr, None)
                            _lb = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                            if _lb is not None:
                                _rb = [float(v) for v in _lb]
                            elif _prev_af is not None and hasattr(_prev_af, 'bbox') and _prev_af.bbox is not None:
                                _rb = [float(v) for v in _prev_af.bbox]
                            else:
                                _rb = None
                            if _rb is not None:
                                _ncx, _ncy = (_nb[0] + _nb[2]) * 0.5, (_nb[1] + _nb[3]) * 0.5
                                _rcx, _rcy = (_rb[0] + _rb[2]) * 0.5, (_rb[1] + _rb[3]) * 0.5
                                _jump = ((_ncx - _rcx) ** 2 + (_ncy - _rcy) ** 2) ** 0.5
                                _jthr = max(90.0, (_nb[2] - _nb[0]) * 1.15, (_nb[3] - _nb[1]) * 1.15)
                                _raw_sim = self._calculate_similarity(original_embedding, best_match.embedding)
                                if _jump > _jthr and _raw_sim < 0.30:
                                    if frame_count % 20 == 1:
                                        print(
                                            f"[TRACK] Frame {frame_count}: candidato SALTO DE CARA RECHAZADO "
                                            f"(salto={_jump:.0f}px > {_jthr:.0f}, sim_cruda={_raw_sim:.2f} < 0.30) "
                                            f"— ghost/hold-fade, sin pintar"
                                        )
                                    best_match = None
                                    identity_ok = False
                                    id_margin = 0.0
                        except Exception:
                            pass
                    # A detector can return a different face (or a face-like
                    # patch) with a convincing embedding after a camera shake.
                    # The previous guard only rejected large jumps when the
                    # raw similarity was low, so high-similarity false matches
                    # still became Success and moved the swap elsewhere.
                    jump_rejected = False
                    if best_match is not None and lost_count == 0 and assigned_face is not None:
                        try:
                            _pb = [float(v) for v in assigned_face.bbox]
                            _cb = [float(v) for v in best_match.bbox]
                            _pw, _ph = _pb[2] - _pb[0], _pb[3] - _pb[1]
                            _cw, _ch = _cb[2] - _cb[0], _cb[3] - _cb[1]
                            _allowed_step = max(45.0, 1.05 * max(_pw, _ph, _cw, _ch))
                            _area_ratio = (_cw * _ch) / max(1.0, _pw * _ph)
                            if best_distance > _allowed_step or not 0.35 <= _area_ratio <= 2.8:
                                print(
                                    f"[TRACK] Frame {frame_count}: salto/escala rechazado "
                                    f"(dist={best_distance:.1f}px, limite={_allowed_step:.1f}px, "
                                    f"area={_area_ratio:.2f})", flush=True,
                                )
                                best_match = None
                                identity_ok = False
                                spatial_lock = False
                                jump_rejected = True
                        except (TypeError, ValueError, IndexError, AttributeError):
                            best_match = None
                            identity_ok = False
                            spatial_lock = False
                            jump_rejected = True

                    if jump_rejected:
                        # Do not immediately accept the same distant face via
                        # the less restrictive reacquire path in this frame.
                        setattr(self, tracking_lost_attr, lost_count + 1)
                        return None

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
                            # v5.1053: EXQUISITO — el blend es el MECANISMO DE PERMANENCIA del error:
                            # tras aceptar otra persona (score 0.44-0.48), 5 frames después el
                            # embedding LOCKED derivaba hacia ella. Subido a 0.55/0.62 + margin
                            # (id_margin solo es <1.0 en multi-face; single-face queda 1.0).
                            avg_det_emb = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
                            score_threshold = 0.55 if avg_det_emb < 0.30 else 0.62
                            if (consec_success >= 5 and best_score > score_threshold
                                    and id_margin >= 0.10 and lost_count == 0):
                                old_emb = np.array(getattr(self, original_embedding_attr), dtype=np.float32)
                                new_emb = np.array(best_match.embedding, dtype=np.float32)
                                
                                # v5.1074: Perfiles ahora 12% (antes 5%). Con 5% el tracking nunca
                                # se adaptaba a ángulos de perfil — si el embedding inicial era
                                # ligeramente incorrecto, tomaba ~20 frames exitosos para corregir 50%.
                                is_p = ProcessMgr._is_profile_face(getattr(best_match, 'kps', None))
                                blend_alpha = 0.12 if is_p else 0.15
                                
                                blend = blend_alpha * new_emb + (1.0 - blend_alpha) * old_emb
                                norm = np.linalg.norm(blend)
                                if norm > 0:
                                    setattr(self, original_embedding_attr, blend / norm)
                                    setattr(self, consecutive_success_attr, 0)
                                    if frame_count % 30 == 0:
                                        print(f"[TRACK] Frame {frame_count}: Embedding blend updated (profile={is_p}, score={best_score:.2f})")
                        
                        # Save tracking embedding as recovery fallback
                        # v5.1053: EXQUISITO — solo actualizar el embedding de recuperación
                        # con identidad SÓLIDA (score>=0.55). Antes se actualizaba con
                        # CUALQUIER aceptación → el fallback de validación se convertía
                        # en la otra persona y las siguientes validaciones la aceptaban.
                        tracking_emb_attr = f'_tracking_embedding_{video_basename}'
                        if hasattr(best_match, 'embedding') and best_match.embedding is not None and best_score >= 0.55:
                            emb_arr = np.array(best_match.embedding, dtype=np.float32)
                            norm = np.linalg.norm(emb_arr)
                            if norm > 0:
                                setattr(self, tracking_emb_attr, emb_arr / norm)
                        
                        try:
                            _tb = [int(v) for v in best_match.bbox]
                        except Exception:
                            _tb = [-1, -1, -1, -1]
                        print(
                            f"[TRACK] Frame {frame_count}: Success "
                            f"(score={best_score:.2f}, dist={best_distance:.1f}px"
                            f", bbox=({_tb[0]},{_tb[1]},{_tb[2]},{_tb[3]})"
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
                    multi_face = len(valid_faces) >= 2
                    # v5.1069d: strict_min_sim 0.50→0.42 multi-face (menos estricto para re-acquire)
                    strict_min_sim = 0.35 if (scene_state.get('is_close_up') and lost_count > 0) else (0.42 if multi_face else 0.50)
                    strict_match = self._try_strict_same_person_reacquire(
                        valid_faces, rec_emb, assigned_face, video_basename, frame_count,
                        min_sim=strict_min_sim,
                        frame_shape=current_frame.shape,
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
                            # v5.1052b: Gate mini-caras TAMBIÉN aquí — el póster (26x34px,
                            # misma persona → embedding no lo distingue) re-entró por esta
                            # rama en frames 854-877 (closeup recovery 0.10 → swap al póster,
                            # probe vf860 mean=11.11). El gate v5.1052 solo cubría el loop
                            # normal y _reacquire_face_global.
                            _fw_f = face.bbox[2] - face.bbox[0] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                            _fh_f = face.bbox[3] - face.bbox[1] if (hasattr(face, 'bbox') and face.bbox is not None) else 0
                            if _fw_f < 32 or _fh_f < 32 or _fw_f * _fh_f < 1024:
                                continue
                            # v5.1066: gate cortadas con overflow parcial permitido (primer plano)
                            if face.bbox is not None and self._is_bbox_clipped_by_frame(face.bbox, current_frame.shape, allow_partial_overflow=True):
                                continue
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
                                # v5.1053: EXQUISITO — umbrales subidos (0.18→0.30, 0.28→0.34, 0.38→0.42):
                                # el fallback ampliado aceptaba sim 0.18-0.28 en multi-face = otra persona
                                emb_thresh_exp = 0.30 if (scene_state.get('is_close_up') and lost_count > 0) else (0.34 if scene_state.get('is_close_up') else 0.42)
                                if emb_score >= emb_thresh_exp:
                                    score = emb_score * 0.30
                            # v5.1053: EXQUISITO — aceptación solo-espacial (sin identidad) SOLO en
                            # single-face. En multi-face dos personas pueden estar adyacentes.
                            if score <= 0 and scene_state.get('is_close_up') and lost_count > 0 and not multi_face:
                                min_sim_fallback = 0.25
                                if original_embedding is not None and hasattr(face, 'embedding') and face.embedding is not None:
                                    emb_score = float(self._calculate_similarity(original_embedding, face.embedding))
                                    if emb_score >= min_sim_fallback:
                                        score = max(0.05, 0.25 * (1.0 - dist / expanded_threshold))
                            if score > best_score:
                                best_score = score
                                best_match = face
                                best_distance = dist
                        if best_match and best_score > 0:
                            # v5.1069c: Si el tracking es ESTABLE (score alto + distancia baja),
                            # NO hacer closeup recovery — evita RE-BUSQUEDA innecesarias
                            # (logs: sim 0.84-0.99 cada frame en closeup estable)
                            _stable_tracking = (best_score > 0.75 and best_distance < base_threshold * 0.5)
                            if _stable_tracking:
                                if frame_count % 30 == 1:
                                    print(f"[TRACK] Frame {frame_count}: closeup recovery SKIP (tracking estable score={best_score:.2f}, dist={best_distance:.0f}px)")
                            else:
                                closeup_recovery_exp = scene_state.get('is_close_up')
                                min_closeup_score = 0.06  # v5.1034: umbral más bajo para closeup+lost
                            # v5.1053: EXQUISITO — en multi-face el recovery requiere identidad
                            # confirmada del CANDIDATO (margin vs las demás caras), no solo 2
                            # frames consecutivos (dos personas pueden puntuar igual de cerca).
                            identity_ok_exp = True
                            if multi_face and original_embedding is not None:
                                _fb_emb = getattr(self, f'_tracking_embedding_{video_basename}', None)
                                _exp_kps = getattr(best_match, 'kps', None)
                                _exp_is_profile = ProcessMgr._is_profile_face(_exp_kps, getattr(best_match, 'det_score', None)) if _exp_kps is not None else False
                                identity_ok_exp, _ = self._validate_locked_identity(
                                    valid_faces, original_embedding, best_match,
                                    multi_face=True, fallback_embedding=_fb_emb,
                                    closeup_recovery=closeup_recovery_exp,
                                    is_profile=_exp_is_profile,
                                    bank_embeddings=self._get_user_reference_embeddings(video_basename),
                                )
                            if identity_ok_exp:
# v5.1053c: EXQUISITO — recovery a distancia > base_threshold
                                # (324-597px en logs) = scene cut u otra persona: HOLD, nunca
                                # aceptar otro sujeto. La cara sólo se re-adquiere si está cerca
                                # de la posición esperada (misma persona moviéndose) — la
                                # identidad (identity_ok_exp) ya se validó con margen arriba,
                                # pero en multi-face una posición distante + sim bajo casi
                                # siempre es otra persona.
                                _nearby = best_distance < base_threshold
                                if not _nearby:
                                    if frame_count % 10 == 1:
                                        print(f"[TRACK] Frame {frame_count}: closeup recovery RECHAZADO (sim={best_score:.2f}, dist={best_distance:.0f}px > {base_threshold:.0f}px) - hold/ghost")
                                    # cae al siguiente frame: "tracking local perdido → ghost/hold"
                                elif (closeup_recovery_exp and best_score > min_closeup_score) or best_score > 0.15:
                                    if closeup_recovery_exp and best_score < 0.15:
                                        # v5.1027: Low-confidence closeup — require 5 consecutive frames (v5.1069c: 2→5 para estabilidad)
                                        confirm_attr = f'_closeup_confirm_{video_basename}'
                                        confirm_count = getattr(self, confirm_attr, 0)
                                        if confirm_count < 5:
                                            setattr(self, confirm_attr, confirm_count + 1)
                                            setattr(self, tracking_lost_attr, 0)
                                            if frame_count % 10 == 1:
                                                print(f"[TRACK] Frame {frame_count}: closeup pendiente ({best_score:.2f}, confirm {confirm_count+1}/5)")
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
                                            # v5.1157: el accept cae por el `return` de abajo
                                            # SÓLO si la cara supera los gates de tamaño/corte.
                                            # Si NO supera, `return _apply_reacquired_face(...)`
                                            # se salta el gate cortada (allow_partial_overflow=True
                                            # aquí) y el swap se ancla igual... PERO el gate
                                            # anti-mutada del frame SIGUIENTE (visible<50%) la
                                            # mata → Lost. Se marca el resultado real: si la
                                            # cara es cortada (strict), el print "aceptado" es
                                            # engañoso — aplicamos el gate strict ANTES.
                                            if best_match.bbox is not None and current_frame is not None and self._is_bbox_clipped_by_frame(best_match.bbox, current_frame.shape, allow_partial_overflow=False):
                                                _cb = [int(v) for v in best_match.bbox]
                                                print(
                                                    f"[TRACK] Frame {frame_count}: closeup recovery cara CORTADA estricta "
                                                    f"bbox=({_cb[0]},{_cb[1]},{_cb[2]},{_cb[3]}) — hold/ghost (no anclar media-cara)"
                                                )
                                                lost_count += 1
                                                setattr(self, tracking_lost_attr, lost_count)
                                                pfr = getattr(self, '_prev_frame_result', {}).get(getattr(self, '_current_video_key', ''))
                                                bb = best_match.bbox
                                                return self._hold_previous_face_region(
                                                    current_frame, pfr, bb,
                                                    blend=0.90, prev_mask=None,
                                                )
                                return self._apply_reacquired_face(
                                    best_match, assigned_attr, tracking_lost_attr,
                                    position_history_attr, video_basename,
                                    last_bbox=getattr(assigned_face, 'bbox', None),
                                    last_kps=getattr(assigned_face, 'kps', None),
                                )
# v5.1069c: fin del else para tracking estable
                        else:
                            if frame_count % 30 == 1:
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
                            frame_shape=current_frame.shape,
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
                            # v5.1052: mismo gate mini-caras — el fallback espacial
                            # aceptaba el póster sin revisar tamaño
                            _fbw2 = best_match.bbox[2] - best_match.bbox[0] if (hasattr(best_match, 'bbox') and best_match.bbox is not None) else 0
                            _fbh2 = best_match.bbox[3] - best_match.bbox[1] if (hasattr(best_match, 'bbox') and best_match.bbox is not None) else 0
                            if _fbw2 >= 32 and _fbh2 >= 32 and _fbw2 * _fbh2 >= 1024:
                                # v5.1066: gate cortadas con overflow parcial permitido (primer plano)
                                _fb_clipped = best_match.bbox is not None and self._is_bbox_clipped_by_frame(best_match.bbox, current_frame.shape, allow_partial_overflow=True)
                                d = np.sqrt((fc[0] - assigned_center[0])**2 + (fc[1] - assigned_center[1])**2)
                                fbw = max(1.0, best_match.bbox[2] - best_match.bbox[0]) if (hasattr(best_match, 'bbox') and best_match.bbox is not None) else 100
                                max_dist = max(500, fbw * 3.0) if scene_state.get('is_close_up') else max(350, fbw * 2.5)  # v5.1034: closeup más permisivo
                                # v5.1053: EXQUISITO — gate de identidad: el fallback espacial
                                # aceptaba CUALQUIER cara cercana sin validar identidad
                                # (dist 168-333px en multi-face = la otra persona).
                                # Ahora: single-face OK (identity_ok=True default), multi-face
                                # valida la identidad de ESTE candidato (no la del loop principal).
                                fb_identity_ok = True
                                if multi_face and original_embedding is not None:
                                    _fb_emb2 = getattr(self, f'_tracking_embedding_{video_basename}', None)
                                    _fb_kps2 = getattr(best_match, 'kps', None)
                                    _fb_is_profile2 = ProcessMgr._is_profile_face(_fb_kps2, getattr(best_match, 'det_score', None)) if _fb_kps2 is not None else False
                                    fb_identity_ok, _ = self._validate_locked_identity(
                                        valid_faces, original_embedding, best_match,
                                        multi_face=True, fallback_embedding=_fb_emb2,
                                        closeup_recovery=scene_state.get('is_close_up') and lost_count > 0,
                                        is_profile=_fb_is_profile2,
                                        bank_embeddings=self._get_user_reference_embeddings(video_basename),
                                    )
                                elif original_embedding is not None and hasattr(best_match, 'embedding') and best_match.embedding is not None:
                                    fb_sim = float(self._calculate_similarity(original_embedding, best_match.embedding))
                                    _fb_bank = self._bank_best_sim(best_match.embedding, video_basename)
                                    if _fb_bank is not None:
                                        fb_sim = max(fb_sim, _fb_bank)
                                    if fb_sim < 0.20:
                                        fb_identity_ok = False
                                if d < max_dist and fb_identity_ok and not _fb_clipped:
                                    try:
                                        _fb = [int(v) for v in best_match.bbox]
                                    except Exception:
                                        _fb = [-1, -1, -1, -1]
                                    print(f"[TRACK] Frame {frame_count}: fallback espacial (dist={d:.0f}px, closeup={scene_state.get('is_close_up', False)}, bbox=({_fb[0]},{_fb[1]},{_fb[2]},{_fb[3]}))")
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
                    # v5.1069d: ghost limits +3 (11/10/9) — oclusiones breves no deben matar el track
                    ghost_limit = 11 if avg_det_g < 0.30 else (10 if avg_det_g < 0.45 else 9)
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
                            if len(position_history) > 15:
                                position_history.pop(0)
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
                            frame_shape=current_frame.shape,
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
                        # v5.1069c: NO forzar closeup_mode=True — usar estado real de closeup
                        # para evitar RE-BUSQUEDA closeup innecesarias con sim 0.84-0.99
                        global_match = self._reacquire_face_global(
                            valid_faces, video_basename, frame_count,
                            after_scene_cut=True, closeup_mode=scene_state.get('is_close_up', False),
                            frame_shape=current_frame.shape,
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
            if assigned_face is not None and hasattr(assigned_face, 'bbox'):
                # v5.1054: ESCAPE HATCH — colapso del vídeo 20 (log 20260804_191156: 0 swap
                # durante 1859 frames, "congelado" + elipse visible al cambiar el plano).
                # CAUSA: el candidato EXISTE (score 0.42-0.67) pero la identidad multi-cara lo
                # rechaza (margin 0.01-0.06 < 0.08-0.10, v5.1053 EXQUISITO). El bloque
                # "Tracking truly lost" (lost_count++, ghost, reset, RE-BUSQUEDA) vive en el
                # `elif` de la línea 2424 que SOLO corre cuando assigned_face es None → NUNCA
                # se ejecuta aquí → el flujo caía a return None SIN incrementar lost_count ni
                # resetear → _should_hold_previous_swap veía lost_count=0 → HOLD ETERNO del
                # último swap (imagen congelada en la posición vieja = "congelado" + "rectángulo").
                lost_count = getattr(self, tracking_lost_attr, 0) + 1
                setattr(self, tracking_lost_attr, lost_count)
                setattr(self, consecutive_success_attr, 0)
                avg_det_g = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
                # v5.1069d: ghost limits +3 para más persistencia en oclusiones
                ghost_limit_g = 11 if avg_det_g < 0.30 else (10 if avg_det_g < 0.45 else 9)
                if scene_state.get('is_close_up'):
                    ghost_limit_g += 1
                if lost_count >= max(ghost_limit_g, 8):
                    if frame_count % 25 == 1:
                        print(
                            f"[TRACK] Frame {frame_count}: candidato rechazado x{lost_count} "
                            f"— reset + RE-BUSQUEDA"
                        )
                    setattr(self, assigned_attr, None)
                    setattr(self, position_history_attr, [])
                    setattr(self, f'_hold_suppressed_until_{video_basename}', frame_count + 8)
                    self._open_reacquire_window(video_basename, frame_count, frames=90)
                    global_match = self._reacquire_face_global(
                        valid_faces, video_basename, frame_count,
                        after_scene_cut=True,
                        closeup_mode=scene_state.get('is_close_up', False),
                        frame_shape=current_frame.shape,
                    )
                    if global_match is not None:
                        return self._apply_reacquired_face(
                            global_match, assigned_attr, tracking_lost_attr,
                            position_history_attr, video_basename,
                            last_bbox=getattr(assigned_face, 'bbox', None),
                            last_kps=getattr(assigned_face, 'kps', None),
                        )
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
                        frame_shape=current_frame.shape,
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
                        frame_shape=current_frame.shape,
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
            if a is None or b is None:
                return 0.0
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            if a.size != 4 or b.size != 4 or not np.isfinite(a).all() or not np.isfinite(b).all():
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
        except Exception:
            return candidate_faces[0] if candidate_faces else None

    @staticmethod
    def _compensate_face_sharpness(patch, original_frame, M):
        """v5.1203 — H-FS-004: compensa el déficit de nitidez del parche de cara
        en FOTOS midiendo contra el original alineado por M.

        El unsharp fijo (1.10-1.15, sigma 0.4-0.5) dejaba la cara intercambiada
        en 34-41% de la nitidez del original (medido en el lote real:
        f2 484->200, f3 268->91) → "cara fundida". Receta idéntica a la ya
        validada en el Editor (img_editor_manager._integrate_local_generation,
        MEMORY 2026-10-10, test_sharpness_fix: 8->268 de 424):
          gate: parche < 0.80 × original; objetivo = 0.70 × original;
          2 rondas × bandas sigma 1.0/2.0/4.0 con tope 0.6/0.6/0.5;
          strength = min(tope, 0.50 × déficit), salto si < 0.05.
          Tras cada pasada que supere el objetivo se interpola hacia atrás
          para aterrizar cerca del 70% DESDE ABAJO: con déficit grande el
          tope 0.6 sobrepasaba a 93-105% y el grano de la piel ajena quedaba
          áspero sobre la del objetivo (parche visible).

        Mide sobre el crop alineado (warpAffine del original con la MISMA M
        que usa el pegado), así que solo se modifica el contenido que va
        dentro de la ventana ya auditada — fuera de la máscara no se pinta
        nada. Devuelve (parche_out, nota); sin déficit devuelve el parche
        intacto. Cualquier error debe manejarlo el llamador (fallback al
        unsharp fijo).
        """
        ref = cv2.warpAffine(
            original_frame, M,
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
        )

        def _lap(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            return float(cv2.Laplacian(gray, cv2.CV_64F).var())

        src_sharp = _lap(ref)
        patch_sharp = _lap(patch)
        if src_sharp <= 1.0:
            return patch, f"original sin textura (src={src_sharp:.1f}); parche intacto"
        if patch_sharp >= src_sharp * 0.80:
            return patch, (
                f"sin déficit (parche {patch_sharp:.0f} vs original {src_sharp:.0f}, "
                f"{patch_sharp / src_sharp:.0%}); parche intacto"
            )
        initial = patch_sharp
        target = src_sharp * 0.70
        arr = patch.astype(np.float32)
        passes = 0
        for _ in range(2):
            if patch_sharp >= target:
                break
            for sigma, cap in ((1.0, 0.6), (2.0, 0.6), (4.0, 0.5)):
                if patch_sharp >= target:
                    break
                deficit = src_sharp / max(1.0, patch_sharp) - 1.0
                strength = float(min(cap, 0.50 * deficit))
                if strength < 0.05:
                    continue
                pre = arr.copy()
                pre_sharp = patch_sharp
                blur = cv2.GaussianBlur(arr, (0, 0), sigma)
                arr = cv2.addWeighted(arr, 1.0 + strength, blur, -strength, 0)
                passes += 1
                # Medir en gris (igual que _lap): Laplacian sobre 3 canales
                # sobreestima la nitidez y el bucle se detenía antes de tiempo
                # (medido: nota 62 vs 31 real en gris → objetivo no alcanzado).
                patch_sharp = _lap(np.clip(arr, 0, 255).astype(np.uint8))
                if patch_sharp > target and patch_sharp > pre_sharp:
                    # Retroceso: con déficit grande (medido 58 vs 1074) una
                    # pasada de tope 0.6 sobrepasa el objetivo de golpe
                    # (93-105% frente al 70% pretendido) y el grano de la piel
                    # ajena queda áspero sobre la del objetivo (parche
                    # visible). La receta del Editor aterriza en ~63% porque
                    # sus déficits son menores; aquí se mezcla hacia atrás
                    # para aterrizar cerca del objetivo DESDE ABAJO — más
                    # suave que pasararse, nunca más nítido que el original.
                    _t = (target - pre_sharp) / max(1e-6, patch_sharp - pre_sharp)
                    _t = float(min(1.0, max(0.0, _t)))
                    arr = pre + (arr - pre) * _t
                    patch_sharp = _lap(np.clip(arr, 0, 255).astype(np.uint8))
        out = np.clip(arr, 0, 255).astype(np.uint8)
        note = (
            f"parche {initial:.0f} -> {patch_sharp:.0f} (original {src_sharp:.0f}, "
            f"{patch_sharp / max(1.0, src_sharp):.0%} de fuente, {passes} pasadas)"
        )
        return out, note

    @staticmethod
    def _compensate_pasted_sharpness(result_frame, original_frame, mask):
        """v5.1204 — H-FS-004: compensa la nitidez de la cara ya PEGADA, en
        marco, para fotos.

        Medido en la evidencia real (03429): la compensación alineada deja
        el parche en 44% de la textura del original dentro de la máscara,
        pero el warp inverso 512->628 del pegado pierde más y el resultado
        final cae a 18% (parche 10 vs original 55 medido en el núcleo de la
        máscara) — la medición alineada no ve ese round-trip de resample y
        su objetivo full-canvas se detiene antes de tiempo.

        Aplica aquí la misma receta ya validada del Editor
        (img_editor_manager._integrate_local_generation) sobre el resultado
        compuesto, midiendo contra el original EN MARCO (misma escala, sin
        asimetría de resample): gate <0.80×, objetivo 0.70×, bandas sigma
        1.0/2.0/4.0 con topes 0.6/0.6/0.5, 2 rondas, retroceso hacia el
        objetivo desde abajo. La corrección entra multiplicada por la
        máscara de pegado, así que fuera de la región pastada no se escribe
        ni un píxel (los bordes de la ventana quedan intactos). Solo fotos:
        el vídeo no variará nitidez frame a frame sin validación de vídeo
        completo. Devuelve (resultado, nota); sin déficit devuelve intacto.
        """
        region = np.asarray(mask) >= 0.90
        if not region.any():
            return result_frame, "sin región pastada; resultado intacto"

        def _gray(img):
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

        def _lvar(img):
            return float(cv2.Laplacian(_gray(img), cv2.CV_64F)[region].var())

        src_sharp = _lvar(original_frame)
        cur_sharp = _lvar(result_frame)
        if src_sharp <= 1.0:
            return result_frame, (
                f"original sin textura (src={src_sharp:.1f}); resultado intacto"
            )
        if cur_sharp >= src_sharp * 0.80:
            return result_frame, (
                f"sin déficit (pegada {cur_sharp:.0f} vs original {src_sharp:.0f}, "
                f"{cur_sharp / src_sharp:.0%}); resultado intacto"
            )
        initial = cur_sharp
        target = src_sharp * 0.70
        w3 = np.asarray(mask, dtype=np.float32)
        if w3.ndim == 2:
            w3 = w3[:, :, None]
        arr = result_frame.astype(np.float32)
        passes = 0
        for _ in range(2):
            if cur_sharp >= target:
                break
            for sigma, cap in ((1.0, 0.6), (2.0, 0.6), (4.0, 0.5)):
                if cur_sharp >= target:
                    break
                deficit = src_sharp / max(1.0, cur_sharp) - 1.0
                strength = float(min(cap, 0.50 * deficit))
                if strength < 0.05:
                    continue
                pre = arr.copy()
                pre_sharp = cur_sharp
                blur = cv2.GaussianBlur(arr, (0, 0), sigma)
                sharp = arr * (1.0 + strength) - blur * strength
                cand = np.clip(pre * (1.0 - w3) + sharp * w3, 0, 255)
                passes += 1
                cur_sharp = _lvar(np.clip(cand, 0, 255).astype(np.uint8))
                if cur_sharp > target and cur_sharp > pre_sharp:
                    _t = (target - pre_sharp) / max(1e-6, cur_sharp - pre_sharp)
                    _t = float(min(1.0, max(0.0, _t)))
                    cand = pre + (cand - pre) * _t
                    cur_sharp = _lvar(np.clip(cand, 0, 255).astype(np.uint8))
                arr = cand
        out = np.clip(arr, 0, 255).astype(np.uint8)
        note = (
            f"pegada {initial:.0f} -> {cur_sharp:.0f} (original {src_sharp:.0f}, "
            f"{cur_sharp / max(1.0, src_sharp):.0%} de fuente, {passes} pasadas)"
        )
        return out, note

    @staticmethod
    def _is_profile_face(kps, det_score=None):
        """Detect if face is a profile/side view based on 5-point kps landmarks
        v5.1074c: Reducido falsos perfiles — antes 40% del video era "perfil" (4089/10K).
        Causa: eye_ratio<0.25 y (eye_ratio<0.32 AND eye_y_ratio>0.35) eran demasiado
        permisivos — caras con leve inclinación o giro parcial caían en perfil falso →
        M-EMA lento + máscara perfil + swap débil. Ahora: eye_ratio<0.20 (solo perfil
        real ~45°) y (eye_ratio<0.28 AND eye_y_ratio>0.45) (solo inclinación fuerte)."""
        # v5.1074: Umbral subido 0.18→0.12. Con 0.18, frames borrosos/movidos (det 0.15-0.17)
        # se clasificaban como perfil → M-EMA lento + máscara perfil + swap débil en caras
        # que son frontales pero de baja calidad. 0.12 solo forzar perfil en detecciones
        # realmente ambiguas.
        if det_score is not None and det_score < 0.12:
            return True
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
        # En un perfil real RetinaFace aún puede separar ambos ojos, pero la
        # nariz queda claramente fuera de su punto medio. Se observó en una
        # foto de perfil: eye_ratio=0.305, nariz=1.716 separaciones oculares;
        # el gate anterior la clasificaba frontal. Exigir también que los ojos
        # ocupen menos de 55% de la anchura de los 5 puntos evita activar esta
        # ruta en caras frontales ligeramente inclinadas.
        nose_offset = abs(kps[2][0] - (kps[0][0] + kps[1][0]) * 0.5)
        strong_yaw = eye_ratio < 0.55 and nose_offset > eye_dist_x * 0.80
        return strong_yaw or eye_ratio < 0.20 or (eye_ratio < 0.28 and eye_y_ratio > 0.45)

    @staticmethod
    def _source_identity_params(is_profile: bool = False, is_closeup: bool = False):
        """
        Mapea ajustes UI/globals hacia máximo parecido source.
        El control indica la fracción de cara original del swap que se conserva.
        """
        ui_identity = float(getattr(roop.globals, 'enhancer_blend_factor', 0.70))
        gfpgan_mix = float(np.clip(1.0 - ui_identity, 0.0, 1.0))
        base_cm = float(getattr(roop.globals, 'color_match_strength', 0.0))
        color_match = min(0.16, base_cm * 0.90) if is_profile else base_cm  # v5.1074: antes min(0.12, base*0.80) — doble penalización causaba mismatch de color visible en perfiles
        if is_closeup:
            color_match *= 0.80
        brightness = float(getattr(roop.globals, 'brightness_strength', 0.0))
        brightness *= 1.20
        return gfpgan_mix, color_match, brightness

    def _profile_head_window(self, frame_shape, bbox, is_profile=True):
        """Ventana del bbox detector, con margen corto y borde suave.

        En perfil la similitud pega un cuadrado ~1.7–1.9× más ancho que la
        cabeza. La altura ya coincidía; el halo es ese ancho de más. En
        frontal dejamos más margen para frente, barbilla y expresión.
        """
        h, w = int(frame_shape[0]), int(frame_shape[1])
        window = np.zeros((h, w), dtype=np.float32)
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except Exception:
            return window
        fw, fh = x2 - x1, y2 - y1
        if fw < 2 or fh < 2 or h < 1 or w < 1:
            return window
        margin_x = 0.06 if is_profile else 0.12
        margin_y = 0.10 if is_profile else 0.16
        ix1 = max(0, int(np.floor(x1 - margin_x * fw)))
        iy1 = max(0, int(np.floor(y1 - margin_y * fh)))
        ix2 = min(w, int(np.ceil(x2 + margin_x * fw)))
        iy2 = min(h, int(np.ceil(y2 + margin_y * fh)))
        if ix2 <= ix1 or iy2 <= iy1:
            return window
        window[iy1:iy2, ix1:ix2] = 1.0
        feather = int(max(5, min(21, min(fw, fh) * 0.06))) | 1
        window = cv2.GaussianBlur(window, (feather, feather), 0)
        peak = float(window.max())
        if peak > 1e-6:
            window /= peak
        return window

    def _limit_profile_paste(self, mask, is_profile, bbox, frame_shape=None):
        """Recorta el pegado de un perfil al bbox. Una cara frontal no se toca."""
        if not is_profile or mask is None or bbox is None:
            return mask
        if frame_shape is None:
            frame_shape = mask.shape
        window = self._profile_head_window(frame_shape, bbox)
        if window.shape[:2] != mask.shape[:2]:
            return mask
        return np.clip(mask.astype(np.float32) * window, 0.0, 1.0)

    def _limit_face_paste(self, mask, is_profile, bbox, frame_shape=None):
        """Keep the warp and blend inside the detected face with a soft edge."""
        if mask is None or bbox is None:
            return mask
        if frame_shape is None:
            frame_shape = mask.shape
        window = self._profile_head_window(frame_shape, bbox, is_profile=is_profile)
        if window.shape[:2] != mask.shape[:2]:
            return mask
        return np.clip(mask.astype(np.float32) * window, 0.0, 1.0)

    def _apply_profile_mouth_floor(self, m_blend, mouth_open, prof_sim):
        """0.30 solo con boca abierta e identidad confirmada (>=0.60).

        El floor corría después de poner m_blend=0 en boca cerrada, así que
        max() devolvía los labios del destino en un perfil de la misma persona.
        """
        if mouth_open and float(prof_sim) >= self._SOURCE_IDENTITY_SIM_THRESH:
            return max(float(m_blend), 0.30)
        return float(m_blend)

    @staticmethod
    def _profile_occlusion_guides(M, kps, bbox, raw_shape, ref_shape):
        """Puntos y silueta en el crop de detección, antes del x2 de GFPGAN.

        M puede estar a la resolución de reference_region. La oclusión se mide
        en raw_swapped_aligned, así que M_occ = M * (raw_w / ref_w).
        """
        try:
            if M is None or kps is None or bbox is None:
                return None, None
            pts = np.asarray(kps, dtype=np.float32).reshape(-1, 2)
            if pts.shape[0] < 5 or not np.isfinite(pts[:5]).all():
                return None, None
            raw_h, raw_w = int(raw_shape[0]), int(raw_shape[1])
            ref_w = int(ref_shape[1])
            if raw_w < 2 or raw_h < 2 or ref_w < 2:
                return None, None
            scale = float(raw_w) / float(ref_w)
            m_occ = np.asarray(M, dtype=np.float32).reshape(2, 3) * scale
            mapped = cv2.transform(pts[:5].reshape(-1, 1, 2), m_occ).reshape(-1, 2)
            if not np.isfinite(mapped).all():
                return None, None
            x1, y1, x2, y2 = [float(v) for v in bbox]
            fw, fh = x2 - x1, y2 - y1
            if fw < 2 or fh < 2:
                return None, None
            quad = np.array([
                [x1 - 0.08 * fw, y1 - 0.12 * fh],
                [x2 + 0.08 * fw, y1 - 0.12 * fh],
                [x2 + 0.08 * fw, y2 + 0.12 * fh],
                [x1 - 0.08 * fw, y2 + 0.12 * fh],
            ], dtype=np.float32).reshape(-1, 1, 2)
            quad_m = cv2.transform(quad, m_occ).reshape(-1, 2)
            if cv2.contourArea(quad_m.astype(np.float32)) <= 1.0:
                return None, None
            interior_u8 = np.zeros((raw_h, raw_w), dtype=np.uint8)
            cv2.fillConvexPoly(interior_u8, np.round(quad_m).astype(np.int32), 255)
            if int(interior_u8.max()) <= 0:
                return None, None
            return mapped.astype(np.float32), interior_u8.astype(np.float32) / 255.0
        except Exception:
            return None, None

    @staticmethod
    def _mouth_occlusion_evidence(mask, points):
        """Measure an occlusion over the lips and just beyond them."""
        if mask is None or points is None:
            return 0.0, 0.0
        try:
            pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
            occ = np.asarray(mask, dtype=np.float32)
            if pts.shape[0] < 5 or occ.ndim != 2 or not np.isfinite(pts[:5]).all():
                return 0.0, 0.0
            center = tuple(np.rint(pts[3:5].mean(axis=0)).astype(int))
            span = float(np.linalg.norm(pts[3] - pts[4]))
            if span < 4:
                return 0.0, 0.0
            inner = np.zeros(occ.shape, dtype=np.uint8)
            outer = np.zeros(occ.shape, dtype=np.uint8)
            cv2.ellipse(inner, center, (max(4, int(span * 0.8)), max(4, int(span * 0.5))), 0, 0, 360, 1, -1)
            cv2.ellipse(outer, center, (max(6, int(span * 1.4)), max(6, int(span * 1.1))), 0, 0, 360, 1, -1)
            ring = (outer > 0) & (inner == 0)
            if not np.any(inner) or not np.any(ring):
                return 0.0, 0.0
            return float(occ[inner > 0].mean()), float(occ[ring].mean())
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _mouth_occlusion_ring(mask, lips, native_shape):
        """Measure beyond the lips at the native swap crop's physical scale."""
        occ = np.asarray(mask, dtype=np.float32)
        mouth = np.asarray(lips, dtype=np.float32)
        if (occ.ndim != 2 or mouth.shape != occ.shape or not occ.size
                or not np.isfinite(occ).all() or not np.isfinite(mouth).all()
                or occ.min() < 0 or occ.max() > 1
                or mouth.min() < 0 or mouth.max() > 1):
            raise ValueError('Invalid mouth occlusion masks')
        if native_shape is None or len(native_shape) < 2:
            raise ValueError('Missing native mouth crop shape')
        native_h, native_w = native_shape[:2]
        if not np.isfinite([native_h, native_w]).all() or min(native_h, native_w) <= 0:
            raise ValueError('Invalid native mouth crop shape')
        settings = get_faceswap_config()
        outer_size = settings['mouth_occlusion_ring_outer_kernel']
        inner_size = settings['mouth_occlusion_ring_inner_kernel']
        if (any(type(size) is not int or size <= 0 or size % 2 == 0
                for size in (outer_size, inner_size)) or outer_size <= inner_size):
            raise ValueError('Mouth ring kernels must be positive odd integers, outer > inner')
        scale_y, scale_x = occ.shape[0] / native_h, occ.shape[1] / native_w

        def scaled_kernel(size):
            radius = (size - 1) / 2
            return np.ones((2 * round(radius * scale_y) + 1,
                            2 * round(radius * scale_x) + 1), dtype=np.uint8)

        lip_pixels = (mouth > 0.5).astype(np.uint8) * 255
        wide = cv2.dilate(lip_pixels, scaled_kernel(outer_size))
        inner = cv2.dilate(lip_pixels, scaled_kernel(inner_size))
        ring = cv2.bitwise_and(wide, cv2.bitwise_not(inner)).astype(np.float32) / 255.0
        ring_sum = float(ring.sum())
        return float((occ * ring).sum() / ring_sum) if ring_sum > 0 else 0.0

    @staticmethod
    def _mouth_depth_choice(kps, mp_center, center106):
        """Choose the lip center below the nose when only one is plausible."""
        try:
            pts = np.asarray(kps, dtype=np.float32).reshape(5, 2)
            if not np.isfinite(pts).all():
                return None
            axis = pts[2] - pts[:2].mean(axis=0)
            axis_sq = float(np.dot(axis, axis))
            if axis_sq < 64.0:
                return None
            mp_depth = float(np.dot(np.asarray(mp_center) - pts[2], axis) / axis_sq)
            old_depth = float(np.dot(np.asarray(center106) - pts[2], axis) / axis_sq)
            mp_valid = 0.25 <= mp_depth <= 1.60
            old_valid = 0.25 <= old_depth <= 1.60
            if mp_valid != old_valid:
                return bool(old_valid)
        except Exception:
            pass
        return None

    def _is_bbox_clipped_by_frame(self, bbox, frame_shape, allow_partial_overflow=False) -> bool:
        """v5.1060: Rechaza caras CORTADAS por el borde del frame en la readquisición.

        El usuario exige que las caras re-adquiridas sean COMPLETAS (igual que en
        source, v5.1055). RetinaFace detecta caras parciales pegadas al borde del
        vídeo (bbox con coordenadas negativas o y2/x2 más allá del frame) que
        producen swaps de baja calidad / anclajes a otra persona cortada. Margen
        relativo al tamaño de la cara (más tolerante para closeups grandes).

        v5.1066: `allow_partial_overflow` permite overflow <35% del alto/ancho del
        bbox para caras GRANDES (primer plano talking-head, que llena el frame y
        toca los bordes). Sin esto, el gate v5.1060/1063/1065 mató tramos enteros
        de vídeo (log 095800 cuckforever: 426+147 frames sin swap, RE-BUSQUEDA sin
        match con mejor_sim 0.03-0.24: TODA cara grande cortada se descartaba). El
        gate plan-change del loop principal pasa `allow_partial_overflow=False`
        para mantener EXQUISITO el rechazo de casi-cortadas dentro de la ventana.
        """
        try:
            if bbox is None or len(bbox) != 4:
                return True
            if frame_shape is None or len(frame_shape) < 2:
                return False
            x1, y1, x2, y2 = [float(v) for v in bbox]
            fh, fw = frame_shape[:2]
            if fw <= 0 or fh <= 0:
                return True
            face_w = x2 - x1
            face_h = y2 - y1
            if face_w <= 0 or face_h <= 0:
                return True
            margin = max(2.0, 0.03 * min(face_w, face_h))
            # v5.1066: un PRIMER PLANO (talking-head, cara que llena el frame)
            # toca los bordes del frame de forma natural (y1<0, x2>fw). Rechazar
            # por ello mató tramos enteros del vídeo (log 095800 cuckforever:
            # 426+147 frames sin swap, RE-BUSQUEDA sin match con mejor_sim 0.03-0.24
            # porque TODA cara grande cortada se descartaba). Se rechaza solo si la
            # cara sobresale del frame una fracción SIGNIFICATIVA (>35% del alto o
            # ancho del bbox) = media-cara/esquina; un overflow pequeño angular (<35%)
            # es un closeup legítimo. El gate plan-change del loop pasa False
            # (estricto, sin excepción) para mantener v5.1063/1065 EXQUISITO.
            if (x1 < -margin or y1 < -margin or
                    x2 > fw + margin or y2 > fh + margin):
                if allow_partial_overflow:
                    frac_ovl_h = (max(0.0, -y1) + max(0.0, y2 - fh)) / face_h
                    frac_ovl_w = (max(0.0, -x1) + max(0.0, x2 - fw)) / face_w
                    if (frac_ovl_h < 0.35 and frac_ovl_w < 0.35 and
                            self._is_large_face(bbox, frame_shape)):
                        return False
                return True
            return False
        except Exception:
            return False

    @staticmethod
    def _is_large_face(bbox, frame_shape) -> bool:
        """Cara que llena el frame (primer plano talking-head). Heurístico v5.1066:
        `face_h > 35% del alto + face_w > 35% del ancho`. Se usa para permitir
        overflow <35% en el gate de cortadas y para NO topar el área en zoom
        legítimo (log 095800 cuckforever: closeup de 29x la última swap muere en
        `area_ratio <= 6.0` → tramos de 426+147 frames sin swap)."""
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
            fh, fw = frame_shape[:2]
            if fw <= 0 or fh <= 0:
                return False
            face_w = x2 - x1
            face_h = y2 - y1
            return face_h > 0.35 * fh and face_w > 0.35 * fw
        except Exception:
            return False

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
        """Hold only pixels covered by the last real swap mask."""
        if prev_frame is None or bbox is None or current_frame is None or prev_mask is None:
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

            if prev_mask.shape[:2] != (h_f, w_f):
                return current_frame
            applied_mask = np.clip(prev_mask.astype(np.float32), 0, 1)
            blur_sz = max(21, min(fw, fh) // 5) | 1
            soft_mask = cv2.GaussianBlur(applied_mask, (blur_sz, blur_sz), 0)
            soft_mask = np.clip(soft_mask * applied_mask * mix, 0, 1)
            soft_3ch = cv2.merge([soft_mask, soft_mask, soft_mask])

            curr = current_frame.astype(np.float32)
            prev_matched = match_color_histogram(prev_frame, current_frame, blend_factor=0.35).astype(np.float32)
            blended = curr * (1.0 - mix) + prev_matched * mix
            return (curr * (1.0 - soft_3ch) + blended * soft_3ch).astype(np.uint8)
        except Exception:
            return current_frame

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
    def _build_xseg_shaped_safety_mask(
        frame_shape, xseg_mask, bbox, tight=False, floor=0.55,
    ):
        """H-FS-004: fallback con FORMA XSeg, no elipse.

        Cuando la XSeg colapsa (max<0,30) el código la recreaba como
        elipse radial → halo/blur elíptico fuera del rostro. Aquí se
        conserva la forma medida: se normaliza la XSeg a su pico, se
        recorta al bbox y se suaviza con distance-transform. Sin forma
        medible se devuelve None para conservar el fotograma original.
        """
        h_f, w_f = frame_shape[:2]
        if not np.isfinite(floor) or not 0 < floor <= 1:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
        except Exception:
            return None
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w_f - 1, x2), min(h_f - 1, y2)
        if x2c <= x1c or y2c <= y1c:
            return None
        shaped = None
        try:
            if xseg_mask is not None:
                m = np.asarray(xseg_mask, dtype=np.float32)
                if m.shape != (h_f, w_f):
                    return None
                if m.ndim != 2 or not np.isfinite(m).all():
                    return None
                m = np.clip(m, 0, 1)
                peak = float(m[y1c:y2c, x1c:x2c].max()) if m.size else 0.0
                if peak > 1e-3:
                    norm = np.zeros_like(m)
                    norm[y1c:y2c, x1c:x2c] = np.clip(
                        m[y1c:y2c, x1c:x2c] / peak * floor, 0, floor,
                    )
                    shaped = norm
        except Exception:
            shaped = None
        if shaped is None:
            return None
        # Feather solo dentro del bbox+margen: distance transform desde el
        # borde del bbox, sin gradiente radial elíptico.
        try:
            binary = np.zeros((h_f, w_f), dtype=np.uint8)
            mx = max(4, int((x2c - x1c) * 0.06))
            my = max(4, int((y2c - y1c) * 0.06))
            binary[max(0, y1c - my):min(h_f, y2c + my),
                   max(0, x1c - mx):min(w_f, x2c + mx)] = 255
            dist_in = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            feather = max(8.0, float(min(x2c - x1c, y2c - y1c)) / 8.0)
            ramp = np.clip(dist_in / feather, 0, 1).astype(np.float32)
            shaped = np.clip(shaped * ramp, 0, 1.0)
            if tight:
                shaped = np.clip(shaped * 0.92, 0, 1.0)
        except Exception:
            pass
        return np.clip(shaped, 0, 1.0)


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
    def _widen_mask_for_blend(mask, face_w, face_h, frame_shape=None, force=False):
        """Transición suave sin invadir pelo/fondo — un solo blur mantiene el borde en la cara.
        v5.1020: Si frame_shape se proporciona, añade gradiente en bordes de frame para
        evitar que la elipse extendida fuera del frame cree un borde visible al cortarse
        contra el límite del fotograma.
        v5.1070f: Blur mínimo subido de 9→15 para eliminar 'border fino' en caras
        pequeñas (116x163px: blur antiguo=11px, insuficiente). Para caras normales
        (>200px) el cambio es mínimo (15 vs 16)."""
        m = np.clip(mask.astype(np.float32), 0, 1)
        blur = max(15, (min(face_w, face_h) // 8) | 1)  # v5.1070f: 9→15 min, //10→//8 (más ancho)
        blur = min(blur, 61) | 1
        if frame_shape is not None:
            frame_min = min(frame_shape[0], frame_shape[1])
            blur = min(blur, max(15, (frame_min // 6) | 1))  # v5.1070f: floor 9→15
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
        """Gradiente frontal suave: aplica únicamente en la zona de la línea de inserción
        del cabello (top de bbox hasta frente alta) para integrar el pelo sin crear corte
        horizontal sobre la piel de la frente."""
        h_f, w_f = frame_shape[:2]
        try:
            forehead_top = max(0, int(y1 - face_h * 0.10))
            forehead_bottom = min(h_f, int(y1 + face_h * 0.22))
            if hasattr(target_face, 'kps') and target_face.kps is not None and len(target_face.kps) >= 2:
                kps = np.array(target_face.kps)
                eye_y = float(min(kps[0][1], kps[1][1]))
                forehead_bottom = min(forehead_bottom, int(eye_y - face_h * 0.12))
            if forehead_bottom > forehead_top + 3:
                grad_h = forehead_bottom - forehead_top
                t = np.linspace(0.0, 1.0, grad_h, dtype=np.float32)
                smooth_t = t * t * (3.0 - 2.0 * t)
                max_reduction = 0.20 if is_profile else 0.22
                reduction = max_reduction * (1.0 - smooth_t)
                blend_mask[forehead_top:forehead_bottom, :] *= (1.0 - reduction[:, np.newaxis])
        except Exception:
            pass
        return blend_mask

    @staticmethod
    def _prime_warp_edges(warped_face, original_frame, mask, face_w, face_h):
        """Pre-mezcla warped→original en zona de borde — preserva flequillo mezclando
        más original en la transición sin endurecer el borde de máscara.
        v5.1070g: Además reemplaza píxeles OSCUROS del warp (intensidad <15)
        con original, eliminando el 'border fino' causado por borderValue=0 del
        warpAffine que deja píxeles casi-negros en la transición mask>0 pero
        warp≈negro → composite los multiplica = borde oscuro visible."""
        if warped_face is None or original_frame is None or mask is None:
            return warped_face
        m = np.clip(mask.astype(np.float32), 0, 1)
        # Mezcla original en zona de borde según máscara (lógica anterior)
        edge = np.clip((0.72 - m) / 0.70, 0, 1).astype(np.float32)
        blur = max(11, (min(face_w, face_h) // 10) | 1)
        edge = cv2.GaussianBlur(edge, (blur, blur), 0) * 0.80
        # v5.1070g: NUEVO — detectar píxeles OSCUROS del warp (borderValue=0 residual)
        # y reemplazarlos con original. Esto elimina el border fino que la máscara
        # no puede cubrir porque mask>0 donde warp≈negro.
        dark_map = np.max(warped_face.astype(np.float32), axis=2)  # max canal RGB
        dark_threshold = 15.0  # píxeles con intensidad <15 son "negro del warp"
        dark_mask = np.clip((dark_threshold - dark_map) / dark_threshold, 0, 1).astype(np.float32)
        dark_mask = cv2.GaussianBlur(dark_mask, ((blur * 2) | 1, (blur * 2) | 1), 0)
        # Combinar: usar el MAYOR de edge (máscara) y dark_mask (oscuridad real)
        combined = np.maximum(edge, dark_mask)
        if float(combined.max()) < 0.02:
            return warped_face
        c3 = cv2.merge([combined, combined, combined])
        out = warped_face.astype(np.float32) * (1.0 - c3) + original_frame.astype(np.float32) * c3
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
    def _mouth_mask_in_aligned_face(reference_region, mouth_region, original_shape,
                                    transform, bbox, crop_origin=(0, 0), crop_size=None):
        """Map detected lips from their detection crop into the swap alignment."""
        from roop.processors.FaceSwap import (
            create_mouth_preservation_mask, mouth_region_to_frame_points,
        )
        points = mouth_region_to_frame_points(
            mouth_region, original_shape, crop_origin, crop_size,
        )
        if points is None:
            return np.zeros(reference_region.shape[:2], dtype=np.float32)
        if transform is not None:
            aligned_points = cv2.transform(points[None, :, :], transform)[0]
        else:
            x1, y1, x2, y2 = bbox
            aligned_points = points - np.array([x1, y1], dtype=np.float32)
            aligned_points *= np.array([
                reference_region.shape[1] / max(1, x2 - x1),
                reference_region.shape[0] / max(1, y2 - y1),
            ], dtype=np.float32)
        return create_mouth_preservation_mask(
            reference_region, aligned_points, blend_ratio=1.0,
        )

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

    def _swap_from_saved_reference(self, source_face, target_face, result_frame,
                                   original_frame, call_num, photo_path=None):
        """v5.1202: Swap de emergencia cuando los kps del detector son incoherentes
        con el bbox y el warp saldría deformado. Reusa una cara guardada por la UI
        para este archivo exacto solo si su geometría es coherente (eye_dist/bbox_w sano) y
        solapa con el bbox actual; en vídeo no hay referencia espacial fiable, así
        que entrega el frame sin swap. Mejor sin swap que una cara deformada."""
        try:
            if getattr(self, '_align_reject_active', False):
                return result_frame
            self._align_reject_active = True
            try:
                if getattr(self.options, 'current_video_path', None) or not photo_path:
                    return result_frame
                tgt_box = np.asarray(target_face.bbox, dtype=np.float32).reshape(-1)
                if tgt_box.size != 4 or not np.isfinite(tgt_box).all():
                    return result_frame
                refs_bank = getattr(roop.globals, 'selected_face_references', {}) or {}
                for ref_data in refs_bank.values():
                    for ref in self._photo_refs_for_exact_file(
                            photo_path, self._as_ref_list(ref_data)):
                        cand = self._photo_reference_face(ref, original_frame.shape)
                        if cand is None:
                            continue
                        kps_c = np.asarray(getattr(cand, 'kps', None), dtype=np.float32)
                        box_c = np.asarray(getattr(cand, 'bbox', None), dtype=np.float32).reshape(-1)
                        if kps_c.shape != (5, 2) or not np.isfinite(kps_c).all() or box_c.size != 4:
                            continue
                        eye_rel_c = float(np.linalg.norm(kps_c[1] - kps_c[0])) / max(1.0, float(box_c[2] - box_c[0]))
                        if not (0.05 < eye_rel_c < 1.20):
                            continue
                        inter = max(0.0, min(tgt_box[2], box_c[2]) - max(tgt_box[0], box_c[0])) * max(
                            0.0, min(tgt_box[3], box_c[3]) - max(tgt_box[1], box_c[1]),
                        )
                        area_t = max(1.0, float(tgt_box[2] - tgt_box[0]) * float(tgt_box[3] - tgt_box[1]))
                        if inter / area_t < 0.40:
                            continue
                        print(
                            f"[ALIGN_REJECT] Frame {call_num}: recuperación con cara guardada "
                            f"(eye_rel={eye_rel_c:.3f})",
                            flush=True,
                        )
                        return self._process_face_swap_v21(
                            source_face, cand, result_frame, original_frame,
                            person_idx=0, photo_path=photo_path,
                        )
                return result_frame
            finally:
                self._align_reject_active = False
        except Exception as _e_rec:
            print(f"[ALIGN_REJECT] recuperación falló: {_e_rec}", flush=True)
            return result_frame

    def _process_face_swap_v21(self, source_face, target_face, result_frame, original_frame, enable_temporal_smoothing=False, person_idx=0, photo_path=None):
        ProcessMgr._swap_call_count += 1
        call_num = ProcessMgr._swap_call_count
        # A previous face can return before composite and leave its mouth
        # mask on the instance. The next face must not reuse it.
        self._mouth_blend_mask = None
        self._mouth_blend_factor = 0.0

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

            # v5.1202: Gate de coherencia kps↔bbox. La matriz de alineación se
            # estima SOLO con los 5 kps (norm_crop2); si el regresor los devuelve
            # degenerados respecto al bbox, el warp sale sobredimensionado y mal
            # colocado (log real 1994859933.jpg: ojos a 17,6 px en una cara de
            # 632 px → eye_dist/bbox_w=0,028 vs 0,40-0,45 normal → cuadrado de
            # alineación 998 px sobre bbox 632 px = cara 1,6× y desplazada).
            # En una cara real, incluso de perfil, la distancia interocular es una
            # fracción estable del ancho del bbox. Aplica a foto (ambos modos) y
            # vídeo; ante duda no bloquea.
            try:
                _kps_chk = getattr(target_face, 'kps', None)
                if not landmarks_fit_bbox(_kps_chk, target_face.bbox):
                    print(
                        f"[ALIGN_REJECT] Frame {call_num}: landmarks fuera del bbox "
                        "o geometría no válida; se busca una referencia coherente",
                        flush=True,
                    )
                    return self._swap_from_saved_reference(
                        source_face, target_face, result_frame,
                        original_frame, call_num, photo_path=photo_path,
                    )
                if _kps_chk is not None:
                    _kps_arr = np.asarray(_kps_chk, dtype=np.float32)
                    _bb_chk = np.asarray(target_face.bbox, dtype=np.float32)
                    if _kps_arr.shape == (5, 2) and np.isfinite(_kps_arr).all() and np.isfinite(_bb_chk).all():
                        _bw = max(1.0, float(_bb_chk[2] - _bb_chk[0]))
                        _eye_sep = float(np.linalg.norm(_kps_arr[1] - _kps_arr[0]))
                        _mouth_sep = float(np.linalg.norm(_kps_arr[4] - _kps_arr[3]))
                        _eye_rel = _eye_sep / _bw
                        if _eye_rel < 0.05 or _eye_rel > 1.20:
                            print(
                                f"[ALIGN_REJECT] Frame {call_num}: kps incoherentes con bbox "
                                f"(eye_dist/bbox_w={_eye_rel:.3f}, mouth_dist/bbox_w={_mouth_sep/_bw:.3f}, "
                                f"bbox={_bb_chk.tolist()}) — sin swap; usar refs guardadas",
                                flush=True,
                            )
                            return self._swap_from_saved_reference(
                                source_face, target_face, result_frame,
                                original_frame, call_num, photo_path=photo_path,
                            )
            except Exception as _e_align:
                print(f"[ALIGN_REJECT] check falló ({_e_align}); se conserva el fotograma", flush=True)
                return result_frame

            # ============================================
            # SUAVIZADO TEMPORAL AVANZADO (v3.5)
            # ============================================
            video_key = getattr(self, '_current_video_key', 'default')
            video_basename = video_key[6:] if video_key.startswith('video_') else video_key
            use_mouth_temporal_state = bool(
                enable_temporal_smoothing and not getattr(self, '_is_single_photo', False)
            )

            # Detectar perfil con hysteresis para evitar parpadeo
            target_kps = getattr(target_face, 'kps', None)
            raw_is_profile = ProcessMgr._is_profile_face(target_kps, getattr(target_face, 'det_score', None))
            if enable_temporal_smoothing:
                prof_key = f'_profile_hysteresis_{video_key}'
                prof_state = getattr(self, prof_key, 0)
                if raw_is_profile:
                    prof_state = min(prof_state + 2, 6)
                    is_profile = True
                elif prof_state > 0:
                    prof_state = max(0, prof_state - 2)  # v5.1100: -2 — cola 3 frames en vez de 6 al volver a frontal (M-EMA/máscara perfil residual = parpadeo en el giro)
                    is_profile = True
                else:
                    prof_state = 0
                    is_profile = False
                setattr(self, prof_key, prof_state)
            else:
                is_profile = raw_is_profile
            if is_profile:
                print(f"[PROFILE] Frame {call_num}: Perfil detectado (raw={raw_is_profile})")
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

                    # A detector with low confidence jitters more, it does not
                    # justify resetting a stable track more eagerly.  The
                    # absolute threshold protects tiny faces; the relative
                    # displacement below distinguishes a real plan change from
                    # bbox noise on a large/close face.
                    avg_det_scene = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
                    scene_threshold = 160 if avg_det_scene < 0.35 else 160
                    # Also check scene state for face-count-based cut
                    scene_state_obj = getattr(self, f'_scene_state_{video_basename}', {})
                    scene_cut_frame = scene_state_obj.get('last_scene_cut', 0) if isinstance(scene_state_obj, dict) else 0
                    frame_count_scene = getattr(self, f'_frame_count_{video_basename}', 0)
                    is_near_scene_cut = abs(frame_count_scene - scene_cut_frame) < 5 if scene_cut_frame > 0 else False

                    plan_smooth = frame_count_scene <= getattr(
                        self, f'_plan_change_smooth_until_{video_basename}', 0,
                    )
                    prev_extent = max(
                        1.0,
                        float(np.max(prev_bbox_arr[2:] - prev_bbox_arr[:2])),
                    )
                    curr_extent = max(
                        1.0,
                        float(np.max(curr_bbox[2:] - curr_bbox[:2])),
                    )
                    motion_scale = max(24.0, (prev_extent + curr_extent) * 0.5)
                    motion_ratio = float(velocity / motion_scale)
                    hard_scene_cut = (
                        velocity > scene_threshold
                        and motion_ratio > 1.10
                        and not plan_smooth
                    )
                    # v5.1074c: soft_scene_cut subido de vel>35 a vel>80 para evitar
                    # cascada de scene cuts falsos. Con 35px cualquier temblor de
                    # tracking post-scene-cut re-disparaba → 55 scene cuts en 10K frames.
                    # Ahora solo cascade si hay movimiento real (>80px).
                    last_reset = getattr(self, '_last_reset_call', {}).get(video_key, 0)
                    cooldown_ok = (call_num - last_reset) > 8 if last_reset > 0 else True
                    soft_scene_cut = is_near_scene_cut and velocity > 80 and not plan_smooth and cooldown_ok
                    if (hard_scene_cut or soft_scene_cut) and not is_lockout:
                        print(f"[SCENE_CUT] Frame {call_num}: Salto detectado (vel={velocity:.0f}px, ratio={motion_ratio:.2f}, thresh={scene_threshold}, face_cut={is_near_scene_cut}). Reseteando historial temporal.")
                        if not hasattr(self, '_last_reset_call'):
                            self._last_reset_call = {}
                        self._last_reset_call[video_key] = call_num
                        if video_path and hard_scene_cut:
                            self._reset_tracking_on_scene_cut(
                                video_path, frame_count_scene,
                                reason=f"salto bbox vel={velocity:.0f}px",
                            )

                        for attr in ['_color_ema', '_brightness_ema', '_enhancer_ema']:  # v5.1100: + ratio bucal — el EMA de la persona anterior no contamina el post-corte
                            if hasattr(self, attr):
                                history = getattr(self, attr)
                                if isinstance(history, dict) and video_key in history:
                                    del history[video_key]
                        # This local motion reset also drops mouth history;
                        # the tracking reset above clears it on a hard cut.
                        self._clear_video_temporal_mouth_and_occlusion(video_key)
                        
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
                    if not hasattr(self, '_prev_face_bbox'):
                        self._prev_face_bbox = {}
                    if not hasattr(self, '_prev_face_kps'):
                        self._prev_face_kps = {}
                    self._prev_face_bbox[video_key] = target_face.bbox
                    if hasattr(target_face, 'kps'):
                        self._prev_face_kps[video_key] = target_face.kps

            # ============================================
            # NUEVO: DETECTAR BOCA ABIERTA (Con Histeresis)
            # ============================================
            preserve_mouth = getattr(roop.globals, 'preserve_mouth_expression', True)
            mouth_open = False
            mouth_region = None
            mouth_open_ratio = 0.0
            mouth_open_ratio_smooth = 0.0
            mouth_crop_origin = (0, 0)
            mouth_crop_size = None
            mouth_grounding_disagreed = False
            mouth_geometry_preserve = False
            mouth_grounding_axis_corrected = False

            bx1, by1, bx2, by2 = target_face.bbox
            is_closeup = self._is_closeup_face(target_face.bbox, original_frame.shape)

            if preserve_mouth:
                try:
                    from roop.processors.FaceSwap import detect_mouth_open
                except ImportError:
                    mouth_open = False
                else:
                    landmarks_106 = getattr(target_face, 'landmark_106', None)
                    if landmarks_106 is None:
                        # Los frames de vídeo usan objetos InsightFace sin el
                        # alias que crea face_util para la UI de fotos.
                        landmarks_106 = getattr(target_face, 'landmark_2d_106', None)
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
                    # v5.1130: face_bbox ahora SÍ se pasa — la normalización perfil de
                    # v5.1074c (h_mouth/face_width) era CÓDIGO MUERTO: el parámetro
                    # existía en la firma pero ningún call site lo enviaba → el fallback
                    # 106-landmarks dividía por el ancho de boca PROYECTADO (colapsado en
                    # perfil) → ratio 3-5× inflado con boca cerrada → preservación falsa
                    # constante en perfiles.
                    mouth_open, mouth_region, mouth_open_ratio = detect_mouth_open(
                        target_face, landmarks_106, mouth_probe,
                        face_bbox=getattr(target_face, 'bbox', None),
                    )
                    if isinstance(mouth_region, dict) and landmarks_106 is not None:
                        try:
                            mp_left = np.asarray(mouth_region['mouth_left'], dtype=np.float32)
                            mp_right = np.asarray(mouth_region['mouth_right'], dtype=np.float32)
                            mp_center = (mp_left + mp_right) * 0.5 + np.asarray(mouth_crop_origin, dtype=np.float32)
                            pts106 = np.asarray(landmarks_106, dtype=np.float32)[52:72]
                            center106 = pts106.mean(axis=0)
                            mouth_offset = float(np.linalg.norm(mp_center - center106))
                            if mouth_offset > max(12.0, float(bx2 - bx1) * 0.10):
                                # La regresión de 106 puntos también puede desplazar
                                # los labios ante un perfil u objeto. Contrastar ambos
                                # centros con las comisuras del detector de cinco puntos
                                # antes de decidir cuál de los dos está en otra zona.
                                kps = np.asarray(getattr(target_face, 'kps', None), dtype=np.float32)
                                use_106 = True
                                if kps.shape == (5, 2) and np.isfinite(kps).all():
                                    mouth_anchor = kps[3:5].mean(axis=0)
                                    use_106 = (
                                        np.linalg.norm(center106 - mouth_anchor)
                                        < np.linalg.norm(mp_center - mouth_anchor)
                                    )
                                    depth_choice = self._mouth_depth_choice(
                                        kps, mp_center, center106,
                                    )
                                    if depth_choice is not None:
                                        use_106 = depth_choice
                                        mouth_grounding_axis_corrected = bool(depth_choice)
                                if use_106:
                                    mouth_open, mouth_region, mouth_open_ratio = detect_mouth_open(
                                        target_face, landmarks_106, None,
                                        face_bbox=getattr(target_face, 'bbox', None),
                                    )
                                    mouth_crop_origin = (0, 0)
                                    mouth_crop_size = None
                                elif (getattr(self, '_is_single_photo', False)
                                      and mouth_offset > float(bx2 - bx1) * 0.45
                                      and np.linalg.norm(center106 - mouth_anchor)
                                          > np.linalg.norm(mp_center - mouth_anchor) + float(bx2 - bx1) * 0.25):
                                    # En una foto, un fallo grande del regresor 106
                                    # puede colocar el blend sobre un ojo y dejar
                                    # dientes sintéticos junto a la boca real. Una
                                    # región entre los labios de MediaPipe y las
                                    # comisuras de 5 puntos conserva los píxeles
                                    # originales solo donde la geometría es dudosa.
                                    from roop.processors.FaceSwap import mouth_region_to_frame_points
                                    mp_points = mouth_region_to_frame_points(
                                        mouth_region, original_frame.shape,
                                        mouth_crop_origin, mouth_crop_size,
                                    )
                                    if mp_points is not None:
                                        bounds = np.vstack((mp_points, kps[3:5]))
                                        low, high = bounds.min(axis=0), bounds.max(axis=0)
                                        center = np.rint((low + high) * 0.5).astype(int)
                                        radius = np.maximum(
                                            np.rint((high - low) * 0.5 + np.array([
                                                (bx2 - bx1) * 0.10, (by2 - by1) * 0.08,
                                            ])).astype(int),
                                            [8, 8],
                                        )
                                        mouth_region = cv2.ellipse2Poly(
                                            tuple(center), tuple(radius), 0, 0, 360, 12,
                                        ).astype(np.float32)
                                        mouth_crop_origin = (0, 0)
                                        mouth_crop_size = None
                                        mouth_geometry_preserve = True
                                mouth_grounding_disagreed = True
                                if call_num % 50 == 1:
                                    chosen = '106 landmarks' if use_106 else 'MediaPipe'
                                    print(f"[MOUTH_GROUNDING] Detectores separados {mouth_offset:.0f}px; usando {chosen}", flush=True)
                        except (KeyError, TypeError, ValueError, IndexError):
                            pass
                    # v5.1070d: Cap universal — la boca no puede ser más alta que ancha.
                    # Cubre TODOS los caminos (MediaPipe + fallback 106 landmarks).
                    # En closeup con papada, el fallback puede dar ratio 1.08-1.58
                    # (h_mouth/fw inflado por proyección de perfil).
                    mouth_open_ratio = min(mouth_open_ratio, 1.0)
                    
                    # A photograph or an unsmoothed frame uses only its own
                    # measurements. Video memory belongs to the same person.
                    _person_key = f"{video_key}_p{person_idx}"
                    mouth_open_ratio_smooth = mouth_open_ratio
                    prev_state = False
                    if use_mouth_temporal_state:
                        if not hasattr(self, '_mouth_ratio_smooth'):
                            self._mouth_ratio_smooth = {}
                        if not hasattr(self, '_mouth_state_history'):
                            self._mouth_state_history = {}
                        prev_ratio = self._mouth_ratio_smooth.get(_person_key, mouth_open_ratio)
                        mouth_open_ratio_smooth = 0.75 * mouth_open_ratio + 0.25 * prev_ratio
                        self._mouth_ratio_smooth[_person_key] = mouth_open_ratio_smooth
                        prev_state = self._mouth_state_history.get(_person_key, False)

                    # v5.1041: Umbrales de activación subidos — preservación solo con apertura real
                    # (boca ligeramente abierta = swap puro = máxima identidad source)
                    # v5.1074: Perfiles ahora usan MISMOS umbrales que frontales.
                    # Antes perfiles tenían 0.22/0.24 (vs frontal 0.16/0.20) — necesitaban
                    # boca MÁS abierta para activar preservación. La inflación por foreshortening
                    # ya está en la detección misma (detect_mouth_open normaliza por face_w),
                    # no necesita compensación extra en el threshold.
                    if is_closeup:
                        if is_profile:
                            thresh = 0.16 if prev_state else 0.20
                        else:
                            thresh = 0.16 if prev_state else 0.20
                    else:
                        if is_profile:
                            thresh = 0.17 if prev_state else 0.22
                        else:
                            thresh = 0.17 if prev_state else 0.22
                    mouth_open = mouth_open_ratio_smooth > thresh
                    if use_mouth_temporal_state:
                        # v5.1024: Hold sostenido — evita parpadeo al cerrar boca
                        if not hasattr(self, '_mouth_hold_count'):
                            self._mouth_hold_count = {}
                        if mouth_open:
                            self._mouth_hold_count[_person_key] = 4  # v5.1030: 10→4 — se desactiva tras ~0.13s de silencio
                        else:
                            hold = self._mouth_hold_count.get(_person_key, 0)
                            if hold > 0:
                                self._mouth_hold_count[_person_key] = hold - 1
                                mouth_open = True  # mantener activa durante el hold
                        self._mouth_state_history[_person_key] = mouth_open
                    
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
            is_very_small_face = min(face_w, face_h) < 60  # v5.1049: caras lejanas (usado por occ y boca)

            # v5.59: Skip-swap solo para tracking extremadamente malo — det_score < 0.05 + velocity > 100px
            det_score = getattr(target_face, 'det_score', 1.0)
            velocity = getattr(self, '_last_velocity', 0)
            is_ghost = getattr(target_face, 'is_ghost', False)
            lost_count = getattr(self, f'_tracking_lost_count_{video_basename}', 0)

            # Ghost tracking: último swap por máximo 3 frames, luego fade rápido
            if is_ghost:
                closeup_active = getattr(self, f'_closeup_active_{video_basename}', False)
                prev_frame_result = getattr(self, '_prev_frame_result', {}).get(video_key)
                last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                if prev_frame_result is not None and last_bbox is not None:
                    # v5.1074c: Ghost hold máximo 3 frames con fade agresivo.
                    # Antes: hold_blend 0.96/0.90 + fade en 6 frames → hasta 8 frames
                    # de swap congelado en posición vieja ("ponia caras cuando no habian").
                    # Ahora: hold_blend 0.80/0.70 + fade en 4 frames → máximo 4 frames
                    # visibles, luego fade a 0.10 → original.
                    _gh_count = getattr(self, f'_ghost_hold_count_{video_basename}', 0)
                    if _gh_count >= 3:
                        hold_blend = 0.10
                    else:
                        hold_blend = 0.80 if closeup_active else 0.70
                    setattr(self, f'_ghost_hold_count_{video_basename}', _gh_count + 1)
                    if call_num % 25 == 1:
                        print(
                            f"[TRACK] Frame {call_num}: ghost lost={lost_count} "
                            f"→ hold swap {_gh_count+1}/3 (closeup={closeup_active})"
                        )
                    # Fade-out del hold: se reduce gradualmente la región medida anterior.
                    hold_blend = self._faded_hold_blend(video_basename, hold_blend)
                    if hold_blend < 0.10:
                        if call_num % 25 == 1:
                            print(f"[TRACK] Frame {call_num}: ghost fade completo → original")
                        return original_frame
                    # v5.1001: usar bbox predicha de ghost_face (desplazada con dx,dy) en vez de last_bbox
                    if not hasattr(self, '_last_swap_bbox'):
                        self._last_swap_bbox = {}
                    self._last_swap_bbox[video_basename] = (x1, y1, x2, y2)
                    # Sin máscara trasladada válida, el hold conserva el fotograma original.
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
                        # Sin máscara actual válida, el hold conserva el fotograma original.
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

            # The alignment matrix belongs to this frame's landmarks. Mixing
            # it with the previous frame shifts the pasted face during turns
            # and after tracking recovery; smooth colors, never geometry.
            if M is not None and enable_temporal_smoothing:
                m_attr = f'_m_ema_{video_basename}'
                prev_m = getattr(self, m_attr, None)
                if prev_m is not None:
                    m_diff = np.max(np.abs(M - prev_m))
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
            active_enhancer = next((name for name, key in enh_map.items() if key == enhancer_key), selected_enhancer)
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
                            def __init__(self, face): self.faces = [face]
                            self.ref_images = []
                        
                        enhanced = self.processors[enhancer_key].Run(FaceSetMock(source_face), None, swapped_face_aligned)
                        if enhanced is not None:
                            if isinstance(enhanced, tuple):
                                enhanced = enhanced[0]
                            raw_for_blend = swapped_face_aligned
                            
                            # v5.1073: CARAS GRANDES — usar GFPGAN 512px como upscale nativo
                            # GFPGAN genera 512x512 desde 256x256. Para caras grandes (>200px en frame),
                            # mantener 512px en vez de degradar a 256 — 4x más píxeles antes del warp.
                            _large_face = face_h > 200 or (is_closeup and face_h > 150)
                            if enhanced.shape[:2] != swapped_face_aligned.shape[:2]:
                                if _large_face and M is not None:
                                    # Mantener GFPGAN a 512px — escalar M para que el warp llena la cara
                                    scale_up = enhanced.shape[1] / swapped_face_aligned.shape[1]  # 512/256 = 2.0
                                    M = M * scale_up
                                    raw_for_blend = cv2.resize(swapped_face_aligned, (enhanced.shape[1], enhanced.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                                    if call_num % 50 == 1:
                                        print(f"[QUALITY] v5.1073: {active_enhancer} {enhanced.shape[1]}px upscale (M×{scale_up:.1f}, face={face_h}px)")
                                else:
                                    enhanced = cv2.resize(enhanced, (swapped_face_aligned.shape[1], swapped_face_aligned.shape[0]), interpolation=cv2.INTER_AREA)
                            swapped_face_aligned = cv2.addWeighted(
                                raw_for_blend, 1.0 - gfpgan_mix, enhanced, gfpgan_mix, 0.0,
                            )
                            
                            # Estabilización temporal EMA del enhancer
                            if enable_temporal_smoothing and video_key:
                                if not hasattr(self, '_enhancer_ema'):
                                    self._enhancer_ema = {}
                                prev_enh = self._enhancer_ema.get(video_key)
                                if prev_enh is not None and prev_enh.shape == swapped_face_aligned.shape:
                                    enh_ema_alpha = 0.50 if is_profile else 0.55
                                    # v5.1024: EMA condicional — movimiento rápido reduce inercia
                                    cur_m_diff = getattr(self, f'_m_diff_{video_basename}', 0)
                                    if cur_m_diff > 10.0:
                                        enh_ema_alpha = min(0.85, enh_ema_alpha + 0.25)
                                    swapped_face_aligned = cv2.addWeighted(swapped_face_aligned, enh_ema_alpha, prev_enh, 1.0 - enh_ema_alpha, 0)
                                self._enhancer_ema[video_key] = swapped_face_aligned.copy()
                            
                            if call_num <= 3:
                                cv2.imwrite(os.path.join(debug_dir, f'02_after_enhancer_f{call_num}.png'), swapped_face_aligned)
                            
                            # v5.1073: Unsharp reducido — el swap a 128/256px ya tiene artefactos de interpolación,
                            # el sharpening los amplifica visiblemente en caras grandes.
                            face_h_us = max(1, y2 - y1)
                            # v5.1203 — H-FS-004: en FOTOS el unsharp fijo no llega: el lote
                            # real quedó en 34-41% de la nitidez del original dentro de la
                            # cara (f2 484->200, f3 268->91 → "cara fundida/mal de calidad").
                            # La ruta de foto usa ahora la compensación medida por bandas
                            # ya validada en el Editor (img_editor_manager: gate <0.80 del
                            # original, objetivo 70%, 2 rondas, topes 0.6/0.6/0.5). Mide el
                            # propio crop alineado contra el original warpAffine(M), así que
                            # no pinta nada fuera de la ventana de pegado ya auditada. El
                            # vídeo conserva el unsharp fijo (no cambiar nitidez frame a
                            # frame sin validación de vídeo completo).
                            _photo_compensated = False
                            if getattr(self, '_is_single_photo', False) and M is not None:
                                try:
                                    _comp_patch, _sharp_note = self._compensate_face_sharpness(
                                        swapped_face_aligned, original_frame, M,
                                    )
                                    swapped_face_aligned = _comp_patch
                                    _photo_compensated = True
                                    # Siempre en fotos: una línea por foto es el
                                    # registro directo del déficit en el próximo lote.
                                    print(f"[QUALITY] nitidez foto: {_sharp_note}", flush=True)
                                except Exception as sharp_err:
                                    _photo_compensated = False
                                    print(f"[QUALITY] Compensación de nitidez omitida: {sharp_err}", flush=True)
                            if not _photo_compensated:
                                if _large_face:
                                    # Caras grandes: unsharp suave — la resolución real viene de GFPGAN 512px
                                    unsharp_amt = 1.12
                                    unsharp_sigma = 0.4
                                elif is_closeup:
                                    unsharp_amt = 1.15
                                    unsharp_sigma = 0.5
                                else:
                                    unsharp_amt = 1.10
                                    unsharp_sigma = 0.4
                                blurred = cv2.GaussianBlur(swapped_face_aligned, (0, 0), unsharp_sigma)
                                swapped_face_aligned = cv2.addWeighted(swapped_face_aligned, unsharp_amt, blurred, -(unsharp_amt - 1.0), 0)
                            if call_num % 50 == 1:
                                _unsharp_note = "multi-band" if _photo_compensated else f"{unsharp_amt:.1f}"
                                print(f"[QUALITY] {active_enhancer} mix={gfpgan_mix:.2f} color={color_match_strength:.2f} unsharp={_unsharp_note} large={_large_face}")
                    except Exception as e:
                        print(f"[AUTO_PILOT_ERR] {e}")

            # ============================================
            # 3. COLOR Y BRILLO (ESTABILIZADO EMA v2.7)
            # ============================================
            # Obtener región de referencia para color matching
            h_align, w_align = swapped_face_aligned.shape[:2]
            warp_face_support = None
            if M is not None:
                # Extraer región del original que corresponde a la alineación
                reference_region = cv2.warpAffine(original_frame, M, (w_align, h_align), borderMode=cv2.BORDER_REPLICATE)
            else:
                reference_region = original_frame[y1:y2, x1:x2].copy()
                reference_region = cv2.resize(reference_region, (w_align, h_align))

            # Estabilización de Brillo EMA (baja = más tono del swap/source)
            if enable_temporal_smoothing and video_key:
                if not hasattr(self, '_brightness_ema'):
                    self._brightness_ema = {}
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
                if not hasattr(self, '_color_ema'):
                    self._color_ema = {}
                prev_color = self._color_ema.get(video_key)
                if prev_color is not None and prev_color.shape == swapped_face_aligned.shape:
                    # v5.95: Color EMA más responsivo — 42% nuevo (antes 28%) para reducir efecto congelado
                    ui_identity = float(getattr(roop.globals, 'enhancer_blend_factor', 0.70))
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

                # A. Warp de la cara y de su soporte geométrico.
                #
                # Nunca usar BORDER_REFLECT aquí: fuera del crop alineado replica
                # píxeles de la cara hasta formar el cuadrilátero del warp. Si una
                # máscara XSeg falla y entra el fallback de seguridad, ese contenido
                # replicado se mezcla como un marco oscuro alrededor de la cara.
                # El soporte solo indica dónde hay píxeles del crop alineado.
                # La segmentación medida determina el contorno que se pega.
                warp_border = cv2.BORDER_CONSTANT
                support_aligned = np.ones((h_align, w_align), dtype=np.float32)
                warp_face_support = cv2.warpAffine(
                    support_aligned, M_inv, (w_f, h_f), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
                # El soporte warp-eado conserva parte del cuadrado de similitud.
                # Limitarlo a la caja facial evita pintar pelo y fondo alejados.
                warp_face_support = self._limit_face_paste(
                    warp_face_support, is_profile, (x1, y1, x2, y2), (h_f, w_f),
                )
                # v5.1073: INTER_LANCZOS4 para upscaling de mayor calidad (evita bilinear borroso)
                warp_interp = cv2.INTER_LANCZOS4 if swapped_face_aligned.shape[0] >= 400 else cv2.INTER_LINEAR
                warped_face = cv2.warpAffine(
                    swapped_face_aligned, M_inv, (w_f, h_f),
                    flags=warp_interp, borderMode=warp_border, borderValue=0,
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
                        print("[ALIGN_DEBUG]   warped_face: ALL ZERO (warp falló)")
                    emb_log = list(swap_source_face.embedding[:3]) if isinstance(swap_source_face.embedding, list) else swap_source_face.embedding[:3].tolist()
                    print(f"[ALIGN_DEBUG]   swap_source_face embedding[:3] = {emb_log}")
                    print(f"[ALIGN_DEBUG]   Frame size: {w_f}x{h_f}")
                
                # B. NUEVO: Generar máscara XSeg de alta precisión
                final_mask = None
                # XSeg operates on the aligned crop, including small faces.
                # Missing segmentation never generates an invented contour.
                if "mask_xseg" in self.processors:
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

                if final_mask is None:
                    # H-FS-004: sin forma medida no se inventa contorno. En vídeo,
                    # si hay historial de swap y no acaba de haber un corte de
                    # escena, se reutiliza el mismo hold que la pérdida de
                    # detección; sin historial se conserva el fotograma sin swap.
                    prev_full = getattr(self, '_prev_frame_result', {}).get(video_key)
                    last_bbox = getattr(self, '_last_swap_bbox', {}).get(video_basename)
                    if (
                        video_path
                        and prev_full is not None
                        and last_bbox is not None
                        and self._should_hold_previous_swap(video_basename)
                    ):
                        fade_blend = self._faded_hold_blend(video_basename, 0.88)
                        if fade_blend >= 0.10:
                            prev_mask = getattr(self, '_last_swap_mask', {}).get(video_basename)
                            if call_num % 20 == 1:
                                print("[MASK_HOLD] Sin segmentación medida; hold del frame anterior", flush=True)
                            return self._hold_previous_face_region(
                                original_frame, prev_full, last_bbox,
                                blend=fade_blend, prev_mask=prev_mask,
                            )
                    print("[MASK_SKIP] Sin segmentación medida; se conserva el fotograma", flush=True)
                    self._register_mask_skip()
                    return result_frame

                # D. Aplicar oclusión (v5.2: Independiente del blend ratio para proteger objetos siempre)
                # La detección por textura puede cubrir gran parte de una cara sin
                # localizar un objeto. Una cobertura >40% se rechaza para evitar
                # debilitar el swap y para no activar el paste bucal por ruido.
                # v5.1049: En caras lejanas (<60px) o baja calidad (det_score<0.30) el Sobel/
                # Laplaciano mide ruido de compresión/blur, no objetos reales → occ SKIP
                # (swap intacto). Un objeto delante de una cara de 50px es indistinguible del
                # ruido a esa escala; en caras grandes el gate 0.01-0.40 sigue protegiendo.
                occ_low_quality = is_very_small_face or float(getattr(target_face, 'det_score', 1.0) or 1.0) < 0.30
                _occ_pts, _occ_interior = (None, None)
                if raw_swapped_aligned is not None:
                    _ref_for_occ = reference_region if reference_region is not None else raw_swapped_aligned
                    _occ_pts, _occ_interior = self._profile_occlusion_guides(
                        M, getattr(target_face, 'kps', None), (x1, y1, x2, y2),
                        raw_swapped_aligned.shape, _ref_for_occ.shape,
                    )
                occ_mask_aligned = detect_foreground_occlusion(
                    raw_swapped_aligned, reference_region,
                    feature_points=_occ_pts, interior_mask=_occ_interior,
                )
                # A broad frontal mask can be hair/background plus a real
                # object over the mouth. Its global >40% gate would discard
                # the object too. Recheck only those broad masks using the
                # five-point face location, and require evidence both on and
                # beyond the lips before accepting the localized mask.
                if (not is_profile and not occ_low_quality
                        and (mouth_open or mouth_geometry_preserve)
                        and occ_mask_aligned is not None
                        and float(np.mean(occ_mask_aligned > 0.1)) > 0.40):
                    _front_pts, _front_interior = self._profile_occlusion_guides(
                        M, getattr(target_face, 'kps', None), (x1, y1, x2, y2),
                        raw_swapped_aligned.shape, reference_region.shape,
                    )
                    if _front_pts is not None:
                        _front_occ = detect_foreground_occlusion(
                            raw_swapped_aligned, reference_region,
                            feature_points=_front_pts, interior_mask=_front_interior,
                        )
                        _front_frac = float(np.mean(_front_occ > 0.1))
                        _oral, _oral_ring = self._mouth_occlusion_evidence(
                            _front_occ, _front_pts,
                        )
                        if 0.01 <= _front_frac <= 0.40 and _oral > 0.18 and _oral_ring > 0.12:
                            occ_mask_aligned = _front_occ
                            if call_num % 50 == 1:
                                print(
                                    f"[OCCLUSION_RESCUE] Frontal boca objeto "
                                    f"(frac={_front_frac:.2f}, lips={_oral:.2f}, ring={_oral_ring:.2f})",
                                    flush=True,
                                )
                # v5.1073: Si GFPGAN 512px upscale está activo, reference_region es 512x512
                # pero occ_mask_aligned es 256x256 (de raw_swapped_aligned). Redimensionar
                # para que coincidan en mouth bitwise_and.
                if occ_mask_aligned is not None and reference_region is not None:
                    ref_h, ref_w = reference_region.shape[:2]
                    occ_h, occ_w = occ_mask_aligned.shape[:2]
                    if occ_h != ref_h or occ_w != ref_w:
                        occ_mask_aligned = cv2.resize(occ_mask_aligned, (ref_w, ref_h), interpolation=cv2.INTER_LINEAR)
                occ_frac = float(np.mean(occ_mask_aligned > 0.1)) if occ_mask_aligned is not None and occ_mask_aligned.size > 1 else 0.0
                occ_strength = (0.65 if is_small_face else (0.65 if is_profile else 0.80))  # v5.1100: small 0.50→0.65 — con 0.50 una mano deja 50% de swap visible a través (oclusión fallida)
                occ_applied = False  # v5.1052: el refuerzo de máscara no debe pisar una oclusión real

                if occ_low_quality:
                    if call_num % 100 == 1:
                        print(f"[QUALITY] Oclusión SKIP (lejana/baja calidad: {min(face_w, face_h)}px, det={getattr(target_face, 'det_score', 1.0):.2f} — ruido, no objeto)")
                elif occ_frac >= 0.01 and occ_frac <= 0.40 and np.max(occ_mask_aligned) > 0.1:
                    # Una máscara que ocupa casi media cara no localiza un objeto
                    # con suficiente precisión. En los logs se aplicaban máscaras
                    # de 41-55%, debilitando tanto el swap como la boca.
                    occ_mask_frame = cv2.warpAffine(occ_mask_aligned, M_inv, (w_f, h_f), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    # v5.1130: blur adaptativo — 31px fijo sobre una cara de 80px diluye
                    # la máscara de oclusión (el objeto pierde fuerza justo donde debe
                    # cubrir) y en closeup deja swap visible a través del objeto.
                    _occ_blur = int(max(11, min(41, (min(face_w, face_h) // 6) | 1))) | 1
                    occ_mask_frame = cv2.GaussianBlur(occ_mask_frame, (_occ_blur, _occ_blur), 0)
                    if enable_temporal_smoothing and video_key:
                        prev_occ_frame = getattr(self, '_occ_frame_ema', {}).get(video_key)
                        if prev_occ_frame is not None and prev_occ_frame.shape == occ_mask_frame.shape:
                            occ_mask_frame = cv2.addWeighted(occ_mask_frame, 0.80, prev_occ_frame, 0.20, 0)
                        self._occ_frame_ema[video_key] = occ_mask_frame.copy()
                    # v5.1130: cap 0.75→0.88 — con cap 0.75 un objeto real dejaba ≥25%
                    # del swap visible a través ("la mano no tapa la cara"). El gate
                    # anti-falso-positivo (0.01-0.40 + SKIP en baja calidad) protege
                    # contra sombras/pelo, así que el cap puede ser más alto.
                    reduction = occ_mask_frame * occ_strength
                    reduction = np.minimum(reduction, 0.88)
                    final_mask = np.clip(final_mask - reduction, 0, 1.0)
                    occ_applied = True
                    if call_num % 100 == 1:
                        print(f"[QUALITY] Oclusión aplicada (fuerza={occ_strength:.2f}, cap=0.88, frac={occ_frac*100:.0f}%)")
                elif call_num % 100 == 1 and occ_frac > 0.40:
                    print(f"[QUALITY] Oclusión SKIP (frac={occ_frac*100:.0f}% > 40% — máscara no localizada)")

                # La misma máscara se consulta después para decidir si se pegan
                # píxeles originales sobre la boca. Una máscara global rechazada
                # tampoco constituye prueba de que haya un objeto en los labios.
                if occ_low_quality or occ_frac > 0.40:
                    occ_mask_aligned = np.zeros_like(occ_mask_aligned)

                # v5.78: Erosión adaptativa — más suave en perfil / boca abierta / primer plano
                ekernel = max(3, min(face_w, face_h) // 32) | 1
                if is_profile:
                    ekernel = max(3, ekernel - 1)
                if is_closeup and min(face_w, face_h) >= 180:
                    ekernel = max(3, ekernel - 2)
                if mouth_open and mouth_open_ratio > 0.45:  # v5.1072: invertido — boca MUY abierta necesita MÁS área preservada
                    ekernel = max(3, ekernel - 1)
                blur_sz = int(max(7 if not is_closeup else 5, min(face_w, face_h) // (85 if is_closeup else 75))) | 1
                if is_profile or is_closeup or is_small_face:
                    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    final_mask = cv2.dilate(final_mask, dilate_k, iterations=1)
                    blur_sz = max(blur_sz, (min(face_w, face_h) // (7 if is_profile else 8)) | 1)
                    blur_sz = min(blur_sz, 41)
                    final_mask = cv2.GaussianBlur(final_mask, (blur_sz, blur_sz), 0)
                else:
                    kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ekernel, ekernel))
                    final_mask = cv2.erode(final_mask, kernel_e, iterations=1)
                    final_mask = cv2.GaussianBlur(final_mask, (blur_sz, blur_sz), 0)

                # v5.83/94/993/1071/1072b: Content-feather para caras 60-120px, perfiles Y closeup profiles.
                # v5.1152: Closeup profiles ahora también lo necesitan — antes se saltaban y tenían borde más duro.
                # Solo se salta en is_very_small_face (<60px) donde el ruido domina.
                # v5.1216: En fotos individuales se omite — final_mask con GaussianBlur ya realiza el blend
                # continuo; emborronar warped_face a 13-25px en la transición destruía la textura (440->355)
                # y producía el halo/blur elíptico percibido alrededor del rostro.
                if not is_very_small_face and not getattr(self, '_is_single_photo', False):
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
                print("[MASK_SKIP] Sin alineación válida; se conserva el fotograma", flush=True)
                return result_frame

            mask_before_mouth = final_mask.copy() if final_mask is not None else None
            mouth_force_paste = False
            mouth_paste_strength = 0.0

            # ============================================
            # 5c. Preservación oral: expresión actual y oclusión medida.
            # El factor de blend y el paste son etapas distintas; el factor
            # por sí solo no indica cuánto original queda tras ambas.
            # ============================================
            is_single_photo = getattr(self, '_is_single_photo', False)
            # v5.1049: Caras lejanas (<60px) o baja calidad (det_score<0.30) → boca SIEMPRE
            # swap puro. En una cara de 50px la boca mide ~10-15px: detect_mouth_open da
            # ratios poco fiables (jitter frame-a-frame) y preservar la boca del target solo
            # resta swap en una zona diminuta → parpadeo sin expresión útil. Misma lógica
            # que el gate de oclusión: en baja calidad la señal bucal es ruido.
            # v5.1062: umbral de tamaño 60px→45px para la BOCA (la oclusión conserva 60px).
            # Los logs grok (153338/153520/153713) mostraban caras de 40-57px con det
            # 0.76-0.90 (ALTA confianza) que perdían TODA la preservación oral → boca 100%
            # swap → "mal preservación de boca". Con el gate nuevo, caras ≥45px y det≥0.30
            # recuperan la boca preservada (el multiplicador small_face ×0.80 ya acota el
            # riesgo de jitter en caras pequeñas).
            mouth_low_quality = min(face_w, face_h) < 45 or float(getattr(target_face, 'det_score', 1.0) or 1.0) < 0.30
            if mouth_open and mouth_low_quality:
                mouth_open = False
                if call_num % 50 == 1:
                    print(f"[MOUTH_SKIP] Boca lejana/baja calidad ({min(face_w, face_h)}px, det={getattr(target_face, 'det_score', 1.0):.2f}) — swap puro")
            # En vídeo también se evalúa la boca cerrada: así un objeto que la
            # tape (dedo, vaso, comida) se conserva aunque los landmarks no
            # marquen apertura. El blend normal será 0 y solo quedará el objeto.
            # Además permite que el blend oral se desvanezca de forma temporal
            # en vez de encenderse/apagarse de golpe entre frames.
            if (mouth_open or is_single_photo or enable_temporal_smoothing) and mouth_region is not None and mask_before_mouth is not None:
                mouth_occ_score = 0.0
                mouth_occ_rel = 0.0
                mouth_occ_ring = 0.0
                # Foto: costura mínima con boca cerrada; base de expresión
                # con apertura actual. Los objetos se evalúan después.
                if is_single_photo:
                    if mouth_open_ratio <= 0.22:
                        # Boca CERRADA en foto: nada que preservar (sin expresión
                        # ni objeto) — conservar 18% seguía dejando visibles los
                        # labios/dientes del destino en una FOTO estática. Solo un
                        # residual 6% suaviza la costura sin cambiar la identidad.
                        m_blend = 0.06
                    else:
                        m_blend = 0.32
                elif is_profile:
                    # v5.1157b: base perfil 0.32→0.24. Log real (223907/231021): perfiles
                    # con restore 38-62% constantes = boca del target asomando. En perfil
                    # la boca de canto ya oculta dientes/interior — el paste de expresión
                    # aporta poco y lanza identidad. 0.24 mantiene la costura suave pero
                    # el swap domina; los floors de occ (abajo) siguen cubriendo objetos.
                    m_blend = 0.24
                elif is_closeup:
                    m_blend = 0.38
                else:
                    m_blend = 0.32
                # En vídeo una boca cerrada no tiene expresión que preservar.
                # Debe ser swap puro; el objeto oral, si se detecta abajo,
                # vuelve a elevar el blend de forma explícita.
                if not mouth_open and not is_single_photo:
                    m_blend = 0.0
                if mouth_open_ratio > 0.22:
                    oral_strength = min(1.0, (mouth_open_ratio - 0.22) / 0.35)
                    m_blend = min(0.65, m_blend * (1.0 + oral_strength * 0.80))
                if is_small_face:
                    m_blend *= 0.92
                if mask_before_mouth.mean() < 0.04:
                    m_blend *= 0.90
                if is_profile:
                    prof_sim = getattr(self, '_last_profile_match_sim', 0.0)
                    # 0.30 solo con boca abierta e identidad confirmada. Antes
                    # el floor corría después de m_blend=0 y devolvía los labios
                    # del destino en un perfil de la misma persona con boca cerrada.
                    m_blend = self._apply_profile_mouth_floor(
                        m_blend, mouth_open, prof_sim,
                    )
                    if mouth_open and mouth_grounding_axis_corrected:
                        # MediaPipe placed its mouth on the nose/cheek while
                        # the 106-point mouth lies below the nose. The mouth
                        # is uncertain; retain more of the current expression.
                        m_blend = max(m_blend, 0.55)
                if mouth_geometry_preserve:
                    m_blend = max(m_blend, 0.88)
                    mouth_force_paste = True

                try:
                    # Both the 106 points and MediaPipe lips must be transformed
                    # from full-frame pixels into the same aligned coordinates as
                    # reference_region. MediaPipe starts in mouth_probe's crop.
                    mouth_mask_aligned = self._mouth_mask_in_aligned_face(
                        reference_region, mouth_region, original_frame.shape,
                        M, (x1, y1, x2, y2), mouth_crop_origin, mouth_crop_size,
                    )
                    # Solape ARITMÉTICO (no bitwise: ambas son float 0-1 y el AND
                    # de bits cuantiza/tergiversa los valores intermedios del blur).
                    _occ_f = np.asarray(occ_mask_aligned, dtype=np.float32)
                    _mm_f = np.asarray(mouth_mask_aligned, dtype=np.float32)
                    if _occ_f.shape != _mm_f.shape:
                        _occ_f = cv2.resize(_occ_f, (_mm_f.shape[1], _mm_f.shape[0]), interpolation=cv2.INTER_LINEAR)
                    _overlap = _occ_f * _mm_f
                    mouth_occ_score = float(np.mean(_overlap))
                    # Fracción DE LA BOCA cubierta por oclusión: un palo fino da
                    # crop-mean diminuto pero cubre % real de los labios.
                    _mm_sum = float(_mm_f.sum())
                    mouth_occ_rel = float(_overlap.sum() / _mm_sum) if _mm_sum > 1e-6 else 0.0
                    # Anillo peri-bucal: el objeto (polo) CRUZA los labios y deja
                    # oclusión fuera de ellos (palo en barbilla); los dientes no
                    # salen del polígono. Separa objeto real de dientes.
                    mouth_occ_ring = 0.0
                    try:
                        mouth_occ_ring = self._mouth_occlusion_ring(
                            _occ_f, _mm_f, raw_swapped_aligned.shape,
                        )
                    except Exception as error:
                        print(f'[MOUTH_ERR] anillo: {error}', flush=True)
                        mouth_occ_ring = 0.0
                    # v5.1045: Señal oral SOLO de oclusión real (occ target-only).
                    # El content-diff (swap vs original) era el MISMO bug que la oclusión:
                    # con bocas distintas el diff es SIEMPRE alto (0.085-0.167) → boost
                    # falso a 0.77-0.85 → la boca quedaba 85% original cada vez que el
                    # source y el target tenían bocas distintas. La preservación de
                    # expresión oral se maneja con oral_strength (solo apertura), y el
                    # paste (objeto real) con occ.
                    mouth_signal = mouth_occ_score
                    # A strong profile can put MediaPipe's mouth on the cheek.
                    # Once the nose/eye axis confirms the 106-point mouth,
                    # a thin foreground object may have a small crop mean even
                    # though it crosses both the lips and their outer ring.
                    _axis_object = (
                        is_profile and mouth_open and mouth_grounding_axis_corrected
                        and mouth_occ_score > 0.002
                        and mouth_occ_rel > 0.15
                        and mouth_occ_ring > 0.10
                    )
                    if mouth_signal > 0.004 or _axis_object:
                        occ_strength = min(1.0, mouth_signal * 12.0)
                        m_blend = min(0.85, m_blend * (1.0 + occ_strength * 0.65))
                        # v5.1042: Paste SOLO con oclusión real (objeto delante de la boca).
                        # El content-diff mide swap vs original: SIEMPRE es alto cuando las bocas
                        # difieren (source ≠ target), no distingue "objeto en boca" de "boca distinta".
                        # Antes (v5.1039) umbral 0.008 disparaba paste 0.82-0.96 en casi todas las
                        # fotos → boca quedaba 90% original → poco parecido a source.
                        # Criterio relativo: un objeto fino (palo de polo) da crop-mean
                        # <0.010 pero cubre % medible de los labios → también es real.
                        # El anillo exige que la oclusión CRUCE los labios (palo en
                        # barbilla): los dientes quedan dentro y no disparan.
                        # crop>0.035 (masivo) dispara solo — objeto inequívoco.
                        # v5.1157b: los 3 signals deben ser FUERTES a la vez.
                        # Log real (223907/231021/232209): rel>0.08 solo disparaba el
                        # paste CON RUIDO — en perfiles la textura de labios/mandíbula
                        # cubre 30-100% de la boca (rel=1.000 ×decenas) con occ total
                        # 0.002-0.011 (ruido documentado v5.998) → restore 61% constante
                        # = boca del target asomando en perfiles. Objetos REALES del
                        # mismo log: occ 0.012-0.018 + rel 0.53-1.0 + ring 0.66-1.0.
                        # Nuevo gate: occ>0.010 Y rel>0.20 Y ring>0.04 (o crop masivo).
                        # v5.1202: la banda 0.010-0.025 seguía pegando la boca del target
                        # en FOTOS sin objeto (log real 64876795_014_3fe6.jpg: ratio 0.23
                        # → restore 90% con boca «mal»). Los objetos reales de esa banda
                        # documentados cruzan los labios con fuerza: rel 0.53-1.0 y
                        # ring 0.66-1.0. Dentro de la banda se exige rel>0.35 y ring>0.30;
                        # por encima de 0.025 mandan los umbrales generales.
                        _mouth_has_object = _axis_object or (
                            (mouth_occ_score > 0.010)
                            and (mouth_occ_rel > 0.20)
                            and (mouth_occ_ring > 0.04)
                            and (
                                (mouth_occ_score > 0.025)
                                or (mouth_occ_rel > 0.35 and mouth_occ_ring > 0.30)
                            )
                        ) or (mouth_occ_score > 0.035)
                        if _mouth_has_object:
                            mouth_force_paste = True
                            # Con objeto confirmado, dejar 38% de swap encima
                            # aún dibujaba dientes/labios sobre él en un frame
                            # real. Restaurar el objeto casi opaco y suavizar
                            # solo el borde de la región bucal.
                            mouth_paste_strength_raw = min(0.96, 0.88 + mouth_occ_score * 1.2)
                            mouth_paste_strength = mouth_paste_strength_raw
                            if use_mouth_temporal_state:
                                _paste_key = f"{video_key}_p{person_idx}"
                                if not hasattr(self, '_mouth_paste_ema_by_person'):
                                    self._mouth_paste_ema_by_person = {}
                                prev_paste = self._mouth_paste_ema_by_person.get(_paste_key, mouth_paste_strength_raw)
                                mouth_paste_strength = 0.60 * mouth_paste_strength_raw + 0.40 * prev_paste
                                self._mouth_paste_ema_by_person[_paste_key] = mouth_paste_strength
                            m_blend = max(m_blend, mouth_paste_strength)
                        if _mouth_has_object and call_num % 50 == 1:
                            print(
                                f"[MOUTH_OBJECT] Oclusión oral (occ={mouth_occ_score:.3f}) → restore={m_blend:.2f}, "
                                f"paste={mouth_paste_strength:.2f}"
                            )
                except Exception as e:
                    print(f"[MOUTH_ERR] preservación: {e}", flush=True)

                # Detector disagreement locates the correct lips with the 106
                # points; by itself it does not prove an object occludes them.
                # Only measured occlusion above may paste original mouth pixels.
                if mouth_grounding_disagreed and call_num % 50 == 1:
                    print(
                        "[MOUTH_GROUNDING] Boca localizada con 106 landmarks; "
                        + ("paste por oclusión medida" if mouth_paste_strength > 0.01 else "sin paste por discrepancia"),
                        flush=True,
                    )

                m_blend = float(np.clip(m_blend, 0.0, 0.92))
                # El ratio de apertura de MediaPipe oscila uno o dos frames al
                # hablar o girar. Sin memoria el composite alternaba entre swap
                # puro y boca destino, produciendo parpadeo. Suavizado por
                # vídeo+persona; los objetos ya tienen su propio EMA y conservan
                # prioridad mediante mouth_force_paste.
                if use_mouth_temporal_state and video_key:
                    _mouth_ema_key = f"{video_key}_p{person_idx}"
                    if not hasattr(self, '_mouth_blend_ema_by_person'):
                        self._mouth_blend_ema_by_person = {}
                    if not hasattr(self, '_mouth_force_paste_by_person'):
                        self._mouth_force_paste_by_person = {}
                    _prev_m_blend = self._mouth_blend_ema_by_person.get(_mouth_ema_key, m_blend)
                    # Cuando el objeto sale de la boca, no arrastrar durante
                    # varios frames el 90% de restauración del frame anterior.
                    if self._mouth_force_paste_by_person.get(_mouth_ema_key, False) and not mouth_force_paste:
                        _prev_m_blend = m_blend
                    _blend_current = 0.72 if mouth_force_paste else 0.60
                    m_blend = float(np.clip(
                        _blend_current * m_blend + (1.0 - _blend_current) * _prev_m_blend,
                        0.0, 0.92,
                    ))
                    self._mouth_blend_ema_by_person[_mouth_ema_key] = m_blend
                    self._mouth_force_paste_by_person[_mouth_ema_key] = mouth_force_paste
                apply_mouth_preserve = m_blend >= 0.10
                # Always define the diagnostic measurement.  It is reported on
                # a periodic cadence even when mouth preservation is inactive;
                # leaving it unbound made the diagnostic itself abort the swap
                # and returned the original frame.
                m_impact = 0.0

                # Every photograph reports the evidence for its mouth decision;
                # sparse video logging preserves the existing cadence.
                if is_single_photo or call_num % 50 == 1:
                    mouth_reason = (
                        "object" if mouth_paste_strength > 0.0 else
                        "geometry" if mouth_geometry_preserve else
                        "expression" if mouth_open else "seam"
                    )
                    print(
                        f"[MOUTH_DECISION] file={os.path.basename(photo_path or video_basename)!r} "
                        f"person={person_idx} reason={mouth_reason} temporal={use_mouth_temporal_state} "
                        f"ratio={mouth_open_ratio:.3f} smooth={mouth_open_ratio_smooth:.3f} "
                        f"open={mouth_open} occ={mouth_occ_score:.5f} rel={mouth_occ_rel:.3f} "
                        f"ring={mouth_occ_ring:.3f} occ_frac={occ_frac:.3f} occ_applied={occ_applied} "
                        f"geometry={mouth_geometry_preserve} axis={mouth_grounding_axis_corrected} "
                        f"force_paste={mouth_force_paste} paste_strength={mouth_paste_strength:.3f} "
                        f"blend_factor={m_blend:.3f}", flush=True,
                    )

                if apply_mouth_preserve:
                    mouth_mask = self._mouth_mask_full_frame(
                        original_frame, mouth_region, mouth_crop_origin, mouth_crop_size,
                    )
                    mouth_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
                    _mouth_dilate_iters = 2 if min(face_w, face_h) >= 120 else 1  # v5.1100: margen extra solo en caras grandes (el polígono sólido sin margen = línea de costura en labios)
                    if mouth_force_paste:
                        # Objeto confirmado EN boca (dedo, botella, comida): suele
                        # EXTENDERSE más allá de los labios y el paste solo cubría
                        # el polígono bucal → objeto medio-swapeado ("cuando meten
                        # cosas en la boca no lo hace del todo bien"). +1 iteración
                        # peri-bucal solo con objeto confirmado (2 señales).
                        _mouth_dilate_iters += 1
                    mouth_mask = cv2.dilate(mouth_mask, mouth_kernel, iterations=_mouth_dilate_iters)
                    _mouth_blur = min(41, max(15, (min(face_w, face_h) // 6) | 1))  # v5.1100: blur adaptativo — 15px fijo = transición visible en primer plano
                    mouth_mask = cv2.GaussianBlur(mouth_mask, (_mouth_blur, _mouth_blur), 0)

                    m_impact = (mouth_mask * m_blend).mean()
                    # v5.1024: TWO-PASS COMPOSITING — NO restamos mouth_mask de final_mask
                    # para no colapsar la máscara en caras pequeñas. En vez de eso,
                    # guardamos mouth_mask y m_blend para aplicarlos como blend en el
                    # paso de composición final (sección 7), evitando el safety recovery.
                    self._mouth_blend_mask = mouth_mask.copy()
                    self._mouth_blend_factor = m_blend
                    # v5.1074: min_mouth_residual FUERTE — impide que safety recovery
                    # sobreescriba la boca. Antes 0.10 era insuficiente (max +0.06 al mask,
                    # safety threshold ~0.25 → safety siempre se activaba → boca perdida).
                    min_mouth_residual = 0.25 * mouth_mask
                    final_mask = np.maximum(final_mask, min_mouth_residual)

                if call_num % 50 == 1 and apply_mouth_preserve:
                    print(
                        f"[QUALITY] Boca (ratio={mouth_open_ratio:.2f}, occ={mouth_occ_score:.3f}, "
                        f"rel={mouth_occ_rel:.3f}, ring={mouth_occ_ring:.3f}, impacto={m_impact:.3f}, restore={m_blend*100:.0f}%)"
                    )
                elif call_num % 50 == 1:
                    mode = "Foto swap puro" if is_single_photo else "expresión oral fuerte"
                    print(
                        f"[MOUTH_SKIP] Frame {call_num}: {mode} "
                        f"(ratio={mouth_open_ratio:.2f}) — swap visible, oclusión activa"
                    )

            else:
                if is_single_photo or call_num % 50 == 1:
                    mouth_reason = (
                        "disabled" if not preserve_mouth else
                        "unlocalized" if mouth_region is None else "closed"
                    )
                    print(
                        f"[MOUTH_DECISION] file={os.path.basename(photo_path or video_basename)!r} "
                        f"person={person_idx} reason={mouth_reason} temporal={use_mouth_temporal_state} "
                        f"ratio={mouth_open_ratio:.3f} smooth={mouth_open_ratio_smooth:.3f} "
                        f"open={mouth_open} occ=not_measured paste_strength=inactive "
                        "blend_factor=inactive", flush=True,
                    )

            # Refuerzo de intensidad del contorno medido sin oclusión.
            if final_mask is not None and not occ_applied:
                _mmax = float(final_mask.max())
                # El contorno colapsado se revisa después; no se amplifica aquí.
                if _mmax >= 0.30 and _mmax < 0.92:
                    # v5.1100: EMA del factor de boost — escalar a 0.92 crudo cambia el factor
                    # ×1.1↔×1.5 frame a frame con XSeg estable → flicker de intensidad del swap.
                    _raw_factor = 0.92 / _mmax
                    _boost_key = f'_mask_boost_ema_{video_basename}'
                    _prev_factor = getattr(self, _boost_key, _raw_factor)
                    _boost_factor = 0.70 * _raw_factor + 0.30 * _prev_factor
                    setattr(self, _boost_key, _boost_factor)
                    final_mask = np.clip(final_mask * _boost_factor, 0, 1.0)
                    if call_num % 50 == 1:
                        print(f"[MASK_BOOST] Frame {call_num}: máscara débil max={_mmax:.3f} → ×{_boost_factor:.2f} (swap visible)")

            # ============================================
            # 5b. Recover only the current measured shape; no synthetic support.
            if self._mask_needs_safety_recovery(final_mask, face_w, face_h):
                mask_settings = get_faceswap_config()
                recovered_mask = None
                if mask_settings["shaped_fallback"]:
                    recovered_mask = self._build_xseg_shaped_safety_mask(
                        original_frame.shape[:2], final_mask, (x1, y1, x2, y2),
                        tight=is_closeup or is_profile,
                        floor=float(mask_settings["shaped_fallback_floor"]),
                    )
                if self._mask_needs_safety_recovery(recovered_mask, face_w, face_h):
                    print("[MASK_SKIP] Contorno medido insuficiente; se conserva el fotograma", flush=True)
                    self._register_mask_skip()
                    return result_frame
                final_mask = recovered_mask
                if call_num % 20 == 1:
                    print("[MASK_SAFETY] Recuperación del contorno medido", flush=True)

            # El suavizado temporal guarda esta máscara. Si queda demasiado
            # ancha aquí, el frame siguiente reintroduce el halo.
            final_mask = self._limit_face_paste(
                final_mask, is_profile, (x1, y1, x2, y2), original_frame.shape[:2],
            )
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
                self._mouth_blend_mask = None
                self._mouth_blend_factor = 0.0
                return original_frame
            blend_mask = self._widen_mask_for_blend(final_mask, face_w, face_h, original_frame.shape)
            # v5.1024: Gradiente frontal suave sobre blend_mask (post-widening)
            blend_mask = self._apply_forehead_gradient(
                blend_mask, original_frame.shape[:2], target_face, y1, face_h, is_profile,
            )
            # Limitar el pegado a los píxeles disponibles del crop alineado
            # evita oscurecer el fondo fuera del warp.
            if 'warp_face_support' in locals() and warp_face_support is not None:
                blend_mask = np.clip(blend_mask * warp_face_support, 0, 1.0)
            else:
                # Mantener el rango si no se dispone del soporte adicional.
                blend_mask = np.clip(blend_mask, 0, 1.0)
            # Widen puede volver a sacar la máscara del bbox. El segundo límite
            # es el que se ve: va después del ensanchado y del soporte.
            blend_mask = self._limit_face_paste(
                blend_mask, is_profile, (x1, y1, x2, y2), original_frame.shape[:2],
            )
            mask_3ch = cv2.merge([blend_mask, blend_mask, blend_mask])
            
            # Atenuación por perfil, velocidad o "ghosting"
            velocity = getattr(self, '_last_velocity', 0)
            
            # v5.1.4: Bloquear swap_weight a 1.0 si el usuario lo pide (máxima potencia)
            swap_weight = 1.0 if user_blend >= 0.98 else user_blend
            
            # v5.81: Atenuación extra por baja calidad general del video
            avg_det_w = getattr(self, f'_avg_det_score_{video_basename}', 0.5)
            if avg_det_w < 0.30:
                swap_weight *= 0.90  # v5.1100: 0.85→0.90 — la penalización es permanente en un vídeo entero de baja det (taxa de parecido)
            elif avg_det_w < 0.40:
                swap_weight *= 0.92
            
            if is_profile:
                # Perfil: peso completo excepto cuando tracking es inestable (bajo det_score o alta velocidad)
                # v5.1100: floors 0.5→0.72 / 0.6→0.78 — atenuar al 50% en movimiento = cara target asoma (flicker + poco parecido)
                if velocity > 80:
                    swap_weight *= max(0.72, 1.0 - (velocity - 80) / 200)
            
            if not is_profile and velocity > 100:
                swap_weight *= max(0.78, 1.0 - (velocity - 100) / 200)

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
                        result_frame, prev_frame_result, composite_mask[:, :, 0], alpha_prev,
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

            # v5.1204 — H-FS-004: en FOTOS, medido en marco, la cara pegada
            # queda en 18-44% de la nitidez del original dentro de la máscara
            # (el warp inverso 512->628 no lo ve la compensación alineada).
            # Compensación medida sobre el resultado FINAL ya compuesto, con
            # la misma receta (gate 0.80, objetivo 0.70, retroceso desde
            # abajo) y aplicada solo dentro de la máscara de pegado. El vídeo
            # conserva su ruta actual: nada de nitidez variable frame a frame
            # sin validación de vídeo completo.
            if getattr(self, '_is_single_photo', False) and blend_mask is not None:
                try:
                    result_frame, _paste_sharp_note = self._compensate_pasted_sharpness(
                        result_frame, original_frame, blend_mask,
                    )
                    print(f"[QUALITY] nitidez pegada: {_paste_sharp_note}", flush=True)
                except Exception as paste_sharp_err:
                    print(f"[QUALITY] Compensación de nitidez pegada omitida: {paste_sharp_err}", flush=True)

            if call_num <= 3:
                cv2.imwrite(os.path.join(debug_dir, f'04_final_result_f{call_num}.png'), result_frame)

            mask_mean = final_mask.mean() if final_mask is not None else -1
            mask_max = final_mask.max() if final_mask is not None else -1
            if self._mask_needs_safety_recovery(final_mask, face_w, face_h) and call_num % 20 == 1:
                cov = self._mask_face_coverage(final_mask)
                print(f"[MASK_WARN] Frame {call_num}: cobertura crítica (mean={mask_mean:.4f}, cov={cov}, max={mask_max:.3f})")
            with open(os.path.join(os.path.dirname(__file__), '..', 'debug_swap.log'), 'a') as lf:
                lf.write(f"[SWAP_END] Frame {call_num}: resultado devuelto. Mask mean={mask_mean:.3f} max={mask_max:.3f} bbox=({x1},{y1},{x2},{y2})\n")

            # Guardar estado para el siguiente frame
            if enable_temporal_smoothing and video_key:
                if not hasattr(self, '_prev_frame_result'):
                    self._prev_frame_result = {}
                self._prev_frame_result[video_key] = result_frame.copy()
            if not hasattr(self, '_last_swap_bbox'):
                self._last_swap_bbox = {}
            self._last_swap_bbox[video_basename] = (x1, y1, x2, y2)
            self._reset_hold_fade(video_basename)  # v5.1054c: swap real → fade desde cero
            if final_mask is not None:
                if not hasattr(self, '_last_swap_mask'):
                    self._last_swap_mask = {}
                # Ghost recovery must reuse the area actually composited. The
                # raw mask can extend beyond the aligned face support.
                self._last_swap_mask[video_basename] = composite_mask[:, :, 0].copy()
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
            
        except Exception:
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
        self._session_log_file.write("=== AUTO-AUTO v5.1202 — FaceSwap source quality and current-frame alignment ===\n")
        self._session_log_file.write(f"Start: {datetime.now()}\n")
        self._session_log_file.write(f"PID: {_os.getpid()}\n")
        self._session_log_file.write(f"Video: {video_path}\n")
        self._session_log_file.write(f"Output: {output_path}\n\n")
        self._session_log_file.flush()
        self._mask_skip_count = 0
        # Limpiar logs antiguos (mantener solo últimos 10)
        try:
            log_dir = r'D:\PROJECTS\AUTOAUTO\logs'
            all_logs = sorted([
                _os.path.join(log_dir, f) for f in _os.listdir(log_dir)
                if f.startswith('process_') and f.endswith('.log')
            ], key=_os.path.getmtime)
            while len(all_logs) > 10:
                _os.remove(all_logs.pop(0))
        except BaseException:
            pass
        class _Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    try:
                        f.write(obj)
                        f.flush()
                    except BaseException:
                        pass
            def flush(self):
                for f in self.files:
                    try:
                        f.flush()
                    except BaseException:
                        pass
        self._old_stdout = sys.stdout
        sys.stdout = _Tee(sys.stdout, self._session_log_file)
        # Defaults para que el reintento del except inferior nunca falle con
        # NameError/TypeError si la excepción ocurre antes de definir estas
        # variables (p. ej. en la apertura del lector o del escritor).
        cap = None
        out = None
        temp_output = None
        fourcc = None
        fps_video = fps if fps else 24.0
        width = 0
        height = 0
        total_frames_to_process = 0
        try:
            if not self.is_initialized:
                print("ProcessMgr not initialized")
                return

            import cv2
            import os
            import time

            if self.options:
                self.options.current_video_path = video_path
            # A new render of the same path may use a different source bank or
            # newly selected destination. The source remains fixed only within
            # this render.
            self._clear_video_source_locks(video_path)

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
            
            # OpenCV's mp4v working file was lossy, and the later audio mux
            # copied those already-degraded pixels into the delivered video.
            # Keep the working frames lossless; encode only once at delivery.
            fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            temp_output = output_path + ".temp.avi"
            out = cv2.VideoWriter(temp_output, fourcc, fps_video, (width, height))

            if not out.isOpened():
                print(f"Could not create lossless working video: {temp_output}")
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
            print("[BATCH] Leyendo frames para procesamiento en paralelo...")
            
            # Inicializar contador de frames
            current_frame = start_frame
            
            print("[BATCH] Iniciando procesamiento secuencial con streaming...")
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
                    except BaseException:
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
            _finalizing_video = True
            if not os.path.exists(temp_output) or os.path.getsize(temp_output) == 0:
                raise RuntimeError("El vídeo temporal sin pérdidas está vacío")
            
            elapsed_time = time.time() - start_time
            fps_processed = processed_count / elapsed_time if elapsed_time > 0 else 0

            print(f"[BATCH] Procesamiento completado en {elapsed_time:.2f}s ({fps_processed:.2f} fps)")
            
            import gc
            gc.collect()
            
            print("[BATCH] Memoria liberada")

            from roop.util_ffmpeg import encode_face_swap_video
            encode_face_swap_video(temp_output, video_path, output_path, skip_audio=skip_audio,
                                   start_frame=start_frame, end_frame=end_frame, fps=fps_video)
            print("[BATCH] Vídeo codificado con libx264 CRF 14; audio opcional restaurado")

            print(f"[BATCH] Video guardado en: {output_path}")

            # Limpiar archivo temporal
            try:
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                    print(f"[CLEANUP] Temporal eliminado: {temp_output}")
            except Exception as e:
                print(f"[CLEANUP] Error eliminando temporal: {e}")

            # Yield de completado
            yield (100, f"Video completado: {os.path.basename(output_path)}{self._mask_skip_note()}")

        except Exception as e:
            print(f"[BATCH] Error en run_batch_inmem: {e}")
            import traceback
            traceback.print_exc()
            if locals().get('_finalizing_video', False):
                yield (100, f"Error finalizando vídeo: {e}")
                return
            
            # Si la excepción ocurrió antes de preparar el lector/escritor, el
            # reintento no puede funcionar: salir con mensaje claro en vez de
            # morir con NameError/TypeError dentro del generador.
            if cap is None or fourcc is None or temp_output is None or end_frame is None:
                yield (100, f"Error preparando reintento de vídeo: lector/escritor no inicializados ({e})")
                return

            # Si hay error, intentar procesar sin batch
            yield (0, f"Error: {e}. Reintentando...")
            
            # Reiniciar variables para el fallback
            current_frame = start_frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            processed_count = 0
            try:
                if out is not None:
                    out.release()
                if os.path.exists(temp_output):
                    os.remove(temp_output)
                out = cv2.VideoWriter(temp_output, fourcc, fps_video, (width, height))
                if not out.isOpened():
                    raise RuntimeError("No se pudo reabrir el vídeo temporal sin pérdidas")
            except Exception as writer_error:
                yield (100, f"Error preparando reintento de vídeo: {writer_error}")
                return
            
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    processed_frame = self.process_frame(frame, enable_temporal_smoothing=True, file_path=video_path)
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
                        try:
                            self.progress_callback(progress_percent, desc=msg)
                        except BaseException:
                            pass

            cap.release()
            out.release()
            
            yield (100, "Frames procesados, finalizando video...")

            if os.path.exists(temp_output):
                yield (100, "Fusionando audio...")
                try:
                    from roop.util_ffmpeg import encode_face_swap_video
                    encode_face_swap_video(temp_output, video_path, output_path, skip_audio=skip_audio,
                                           start_frame=start_frame, end_frame=end_frame, fps=fps_video)
                    if os.path.exists(temp_output):
                        os.remove(temp_output)
                    print(f"Vídeo libx264 CRF 14 finalizado: {output_path}")
                except Exception as e:
                    print(f"Video encode failed: {e}")
                    yield (100, f"Error finalizando vídeo: {e}")
                    return

            yield (100, f"Video completado con éxito{self._mask_skip_note()}")

        finally:
            if hasattr(self, '_old_stdout') and self._old_stdout:
                sys.stdout = self._old_stdout
            if hasattr(self, '_session_log_file') and self._session_log_file:
                try:
                    # Sin esta marca el finally no corrió: el proceso murió
                    # por fuera (incidente 2026-10-13). El PID correlaciona
                    # con ui_start_*.log y los eventos de Windows.
                    self._session_log_file.write(f"\n=== END (PID {_os.getpid()}) ===\n")
                    self._session_log_file.close()
                except BaseException:
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
        except BaseException:
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
                    except BaseException:
                        pass
        except BaseException:
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
