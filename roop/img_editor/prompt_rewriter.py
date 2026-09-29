#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Rewriter - Inteligencia Semántica Pura (Sin Hardcoding)
"""

import os, sys, json, re
from typing import Optional, Dict, List, Tuple

# Añadir CUDA al PATH si es Windows
if sys.platform == "win32":
    cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
    if cuda_path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = cuda_path + os.pathsep + os.environ.get("PATH", "")

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MOONDREAM_TEXT_PATH = os.path.join(get_project_root(), "models", "moondream", "moondream2-text-model-f16.gguf")
QWEN_LLM_PATH = os.path.join(get_project_root(), "models", "llm", "qwen2.5-0.5b-instruct-q4_k_m.gguf")

def ensure_llm_model():
    """Baixa modelo de lenguaje si no existe"""
    if os.path.exists(QWEN_LLM_PATH):
        return True
    
    llm_dir = os.path.dirname(QWEN_LLM_PATH)
    if not os.path.exists(llm_dir):
        os.makedirs(llm_dir, exist_ok=True)
    
    print("[PromptRewriter] Intentando baixar qwen2.5-0.5b-instruct...")
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            filename="qwen2.5-0.5b-instruct-q4_k_m.gguf",
            local_dir=llm_dir
        )
        print(f"[PromptRewriter] Modelo baixado: {model_path}")
        return True
    except Exception as e:
        print(f"[PromptRewriter] No se pudo baixar modelo: {e}")
        return False

class PromptRewriter:
    """Analiza la instrucción y extrae todos los parámetros de edición mediante razonamiento LLM"""

    def __init__(self):
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        if self._llm is not None and self._llm != "heuristic":
            return
        try:
            from llama_cpp import Llama
            
            root = get_project_root()
            
            llm_paths = [
                QWEN_LLM_PATH,
                os.path.join(root, "models", "llm", "llama-3.2-1b-instruct-q4_k_m.gguf"),
            ]
            
            model_path = None
            for p in llm_paths:
                if os.path.exists(p):
                    model_path = p
                    print(f"[PromptRewriter] Usando modelo de lenguaje dedicado: {os.path.basename(p)}")
                    break
            
            if not model_path:
                moondream_paths = [
                    MOONDREAM_TEXT_PATH,
                ]
                for p in moondream_paths:
                    if os.path.exists(p):
                        model_path = p
                        print(f"[PromptRewriter] ADVERTENCIA: Usando moondream (menos preciso)")
                        break
                
            if not model_path:
                print("[PromptRewriter] ERROR: No se encontró ningún modelo LLM.")
                return

            # Detectar si podemos usar GPU (aceleración GGUF)
            n_gpu_layers = 0
            try:
                import torch
                if torch.cuda.is_available():
                    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    # En una 3060 Ti 8GB, Qwen 0.5B cabe de sobra (ocupa <500MB en VRAM)
                    n_gpu_layers = 32 # Forzar capas a GPU para velocidad
            except: pass

            print(f"[PromptRewriter] Cargando rewriter: {os.path.basename(model_path)} (GPU layers={n_gpu_layers})")
            self._llm = Llama(
                model_path=model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=8192,
                verbose=False
            )
            print("[PromptRewriter] Rewriter listo")
        except ImportError:
            print("[PromptRewriter] llama-cpp-python no disponible, usando modo heurístico")
            self._llm = "heuristic"
        except Exception as e:
            print(f"[PromptRewriter] LLM no disponible: {e}, usando modo heurístico")
            self._llm = "heuristic"

    def unload(self):
        """Libera la memoria del LLM inmediatamente"""
        if self._llm is not None and self._llm != "heuristic":
            print("[PromptRewriter] Liberando memoria del rewriter...")
            try:
                # La destrucción del objeto Llama libera la memoria C++ asociada
                del self._llm
                self._llm = None
                import gc, torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def fix_typos(self, text: str) -> str:
        """Corrige SOLO ortografía del raw (típicamente español con typos).
        El LLM propone y el ALGORITMO dispone: se acepta únicamente si conserva
        las mismas palabras con distancia de edición pequeña (typos, nunca
        reescritura/paráfrasis/traducción). Cero vocabulario: la puerta es
        Damerau-Levenshtein por palabra. Si el LLM alucina, se devuelve el
        original intacto (fail-safe: nunca peor que antes)."""
        if not text or not text.strip():
            return text
        if self._llm is None or self._llm == "heuristic":
            return text
        try:
            sys_msg = (
                "Corrige UNICAMENTE la ortografía del texto en español. "
                "No traduzcas a otro idioma, no añadas ni quites palabras, "
                "no cambies el orden ni el significado, no expliques nada. "
                "Devuelve SOLO el texto corregido en una línea."
            )
            resp = self._llm.create_completion(
                f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
                f"<|im_start|>user\n{text.strip()}<|im_end|>\n"
                "<|im_start|>assistant\n",
                max_tokens=120, temperature=0.0,
                echo=False, stop=["<|im_end|>"],
                repeat_penalty=1.1,
            )
            fixed = resp["choices"][0]["text"].strip().strip("\"'").split("\n")[0].strip()
            if not fixed:
                return text
            from roop.img_editor.prompt_translator import _damerau_leven
            w0, w1 = text.split(), fixed.split()
            if len(w0) != len(w1) or not w1:
                return text
            total = 0
            for a, b in zip(w0, w1):
                d = _damerau_leven(a.lower(), b.lower())
                if d > 2:
                    return text
                total += d
            if total == 0 or total > 8:
                return text
            print(f"[PromptRewriter] Typos corregidos: '{text[:60]}' -> '{fixed[:60]}'", flush=True)
            return fixed
        except Exception as e:
            print(f"[PromptRewriter] fix_typos omitido: {e}")
            return text

    def rewrite(self, prompt: str, image_context: str = "", mode: str = "img2img") -> Dict:
        """Devuelve un diccionario con el análisis completo del LLM.
        NOTA: image_context debe ser metadata estructurada (no descripción generada por VLM) para evitar alucinaciones.
        Por defecto en Imagine usamos solo texto del prompt + semantic ligero.
        """
        if self._llm == "heuristic":
            return self._rewrite_heuristic(prompt)
        
        if self._llm is not None:
            try:
                return self._rewrite_with_llm(prompt, self._llm, image_context=image_context, mode=mode)
            except Exception as e:
                print(f"[PromptRewriter] Error con LLM: {e}")
        
        return {
            "prompt": prompt,
            "magnitude": 0.5,
            "mask_target": "subject",
            "affected_regions": [],
            "requires_scene_regeneration": False,
            "reasoning": "Fallback mode"
        }
    
    def _rewrite_heuristic(self, prompt: str) -> Dict:
        """Fallback neutro cuando no hay LLM disponible."""
        print("[PromptRewriter] Heuristic disabled: usando prompt original sin inferencia semántica")
        return {
            "prompt": prompt,
            "magnitude": 0.5,
            "mask_target": "subject",
            "affected_regions": [],
            "requires_scene_regeneration": False,
            "reasoning": "LLM unavailable"
        }

    @staticmethod
    def _classify_affected_regions(prompt: str, llm) -> List[str]:
        """Ask a narrow, independent scope question when the main plan omits it.

        The main rewriter is optimized for translation and often emits its old,
        compact JSON schema even after new optional fields are added.  This
        classifier only describes spatial scope; it does not inspect verbs or
        alter the user instruction.
        """
        if not prompt or llm is None:
            return []
        try:
            system_msg = (
                "You classify the spatial scope of an image editing request. "
                "Reply ONLY with valid JSON in this exact form: "
                '{"affected_regions":["region one","region two"]}. '
                "List every distinct visual region explicitly requested to change. "
                "Use concise region names and do not infer hidden changes."
            )
            response = llm.create_completion(
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\nRequest: {prompt.strip()}<|im_end|>\n"
                "<|im_start|>assistant\n",
                max_tokens=80, temperature=0.0, echo=False,
                stop=["<|im_end|>", "\n\n"], repeat_penalty=1.05,
            )
            text = response["choices"][0]["text"].strip()
            data = _extract_first_json(text)
            if isinstance(data, dict):
                raw_regions = data.get("affected_regions", [])
            else:
                # Qwen 0.5B suele emitir directamente el array aunque se le
                # pida un objeto envoltorio; ambos formatos expresan el mismo
                # plan estructural y no requieren interpretar lenguaje libre.
                try:
                    raw_regions = json.loads(text)
                except Exception:
                    raw_regions = []
            if not isinstance(raw_regions, list):
                return []
            regions = []
            for raw_region in raw_regions:
                region = str(raw_region or "").strip().lower()
                if region and region not in regions:
                    regions.append(region)
            print(f"[PromptRewriter] Plan de regiones: {regions}", flush=True)
            return regions
        except Exception as exc:
            print(f"[PromptRewriter] Plan de regiones omitido: {exc}", flush=True)
            return []

    @staticmethod
    def _classify_subject_query(prompt: str, llm) -> str:
        """Return the visual entity named by a local edit, without action rules.

        Spatial regions such as "head" say *where* a change lands but not
        *whose* region it is when several faces are present.  This small query
        extracts only the visible entity description so the vision segmenter
        can ground it in the actual image.
        """
        if not prompt or llm is None:
            return ""
        try:
            system_msg = (
                "Identify the visible subject explicitly named to be changed in an image editing request. "
                "Reply ONLY valid JSON in this exact form: {\"subject_query\":\"short visual noun phrase\"}. "
                "Return an empty string only when the request names no particular visible subject. "
                "Do not describe the requested change."
            )
            response = llm.create_completion(
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"<|im_start|>user\nRequest: {prompt.strip()}<|im_end|>\n"
                "<|im_start|>assistant\n",
                max_tokens=40, temperature=0.0, echo=False,
                stop=["<|im_end|>", "\n\n"], repeat_penalty=1.05,
            )
            data = _extract_first_json(response["choices"][0]["text"].strip())
            query = str(data.get("subject_query", "") or "").strip().lower() if isinstance(data, dict) else ""
            if len(query) > 80:
                query = ""
            print(f"[PromptRewriter] Consulta de sujeto: {query or '(sin sujeto único)'}", flush=True)
            return query
        except Exception as exc:
            print(f"[PromptRewriter] Consulta de sujeto omitida: {exc}", flush=True)
            return ""

    def _rewrite_with_llm(self, prompt: str, llm, image_context: str = "", mode: str = "img2img") -> Dict:
        ctx = image_context[:200] if len(image_context) > 200 else image_context
        
        if mode == "txt2img":
            system_msg = (
                "You are a professional prompt engineer. Translate the user's request to English and enrich it with technical quality keywords.\n"
                "CRITICAL: Translate the subject description EXACTLY. Do NOT change, add, or remove any details about the subject (age, hair, skin, body, clothing, setting, action).\n"
                "Keep the user's exact description, only translate it to English. Then append quality tags at the end.\n"
                "Quality tags: lighting (cinematic, soft, raytracing), camera (8k, macro, depth of field), texture (highly detailed skin, photorealistic).\n"
                "Structure: \"<exact translated description>, <quality tags>\"\n"
                "Example:\n"
                "User: una mujer morena en la playa\n"
                "Assistant: {\"prompt\": \"a brunette woman on the beach, cinematic lighting, golden hour, highly detailed skin textures, photorealistic, 8k resolution, sharp focus, depth of field\", \"magnitude\": 0.6, \"mask_target\": \"subject\", \"preserve_face\": true, \"is_global\": true}\n"
            )
        else:
            # Prompt para EDICIÓN (img2img)
            # v5.1108 — los prompts traen MÁS DE UNA condición ("desnuda Y
            # descalza") y los ejemplos eran todos de condición única: el 0.5B
            # aprendía el formato "una petición → una descripción" y OMITÍA la
            # segunda mitad (log 11:57). Se añade instrucción + ejemplo multi-
            # condición. Esto enseña el FORMATO (no ramifica por palabras del
            # usuario: vale para cualquier combinación).
            system_msg = (
                "You analyze image editing requests. Translate Spanish to English accurately. "
                "Make the English prompt vivid and richly descriptive — it directly controls a diffusion model.\n"
                "CRITICAL: requests usually contain SEVERAL conditions at once. Translate EVERY single one of them into the prompt. "
                "Dropping any part of the request is a failure — if the user asks for two or three changes, the prompt must describe all of them.\n"
                "For body edits (naked, undress, masturbating, sex, etc.) describe the action clearly and anatomically.\n"
                "Classify geometry semantically: requires_scene_regeneration is true only when the requested final image needs a new spatial arrangement, body configuration, interaction, or viewpoint. It is false when the requested result can be made by changing appearance while preserving the existing geometry.\n"
                "Output ONLY valid JSON with these fields:\n"
                '- "prompt": English translation, vivid and descriptive\n'
                '- "magnitude": number 0.0-1.0 estimating how much change is requested (0.1=tiny, 0.5=moderate, 0.9=extreme)\n'
                 '- "mask_target": "subject" (person/people), "background", or "clothes"\n'
                 '- "affected_regions": array of every distinct visual region independently changed by the request; use one item for one region and include all regions for a compound request\n'
                '- "preserve_face": true (keep faces unchanged) or false\n'
                '- "is_global": true (edit whole image) or false (edit only target area)\n'
                '- "requires_scene_regeneration": true or false, using the geometric criterion above\n'
                'Examples:\n'
                'User: ella debe ir desnuda sin ropa\n'
                'Assistant: {"prompt": "she must go completely naked, bare skin, no clothes, full body nudity, visible vagina, exposed pubic area, bare breasts, completely nude", "magnitude": 0.9, "mask_target": "subject", "preserve_face": true, "is_global": true}\n'
                'User: debe ir desnuda y descalza\n'
                'Assistant: {"prompt": "she must go completely naked, bare skin, no clothes, and barefoot, bare feet, no shoes, no footwear", "magnitude": 0.9, "mask_target": "subject", "preserve_face": true, "is_global": true}\n'
                'User: se esta masturbando\n'
                'Assistant: {"prompt": "she is masturbating, touching her genitals, sexual act", "magnitude": 0.8, "mask_target": "subject", "preserve_face": true, "is_global": false}\n'
                'User: ponle un sombrero azul\n'
                'Assistant: {"prompt": "put a blue hat on her head", "magnitude": 0.4, "mask_target": "subject", "preserve_face": true, "is_global": false}\n'
                'User: mejora la calidad y nitidez\n'
                'Assistant: {"prompt": "improve image quality, sharpness, detail and clarity", "magnitude": 0.3, "mask_target": "subject", "preserve_face": true, "is_global": true}\n'
                'User: follando a cuatro patas\n'
                'Assistant: {"prompt": "having sex from behind, doggy style position, penetrating", "magnitude": 0.9, "mask_target": "subject", "preserve_face": true, "is_global": false}\n'
                'User: hazla ver como de 80 años\n'
                'Assistant: {"prompt": "make her appear 80 years old, deeply wrinkled skin, age spots, sagging skin, thinning hair, aged elderly appearance", "magnitude": 0.7, "mask_target": "subject", "preserve_face": true, "is_global": true}\n'
                'User: que tenga piel vieja con arrugas\n'
                'Assistant: {"prompt": "old wrinkled skin texture, deep wrinkles and fine lines, aged weathered skin, realistic skin aging details", "magnitude": 0.6, "mask_target": "subject", "preserve_face": true, "is_global": true}\n'
            )

        full_prompt = (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\nContext: {ctx}\nRequest: {prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        response = llm.create_completion(
            full_prompt, max_tokens=250, temperature=0.0,
            echo=False, stop=["<|im_end|>", "\n\n"],
            repeat_penalty=1.15, min_p=0.05, typical_p=0.15
        )
        
        try:
            text = response['choices'][0]['text'].strip()
            print(f"[PromptRewriter] Raw LLM Response: {text}")
            
            data = _extract_first_json(text)
            if data is None:
                raise ValueError("No valid JSON found in response")

            if not isinstance(data, dict):
                return {"prompt": prompt, "magnitude": 0.5, "mask_target": "subject", "preserve_face": True, "is_global": True}

            # Funciones auxiliares para extraer valores de forma robusta
            def extract_text(val):
                if isinstance(val, dict):
                    for k in ['TARGET', 'target', 'value', 'Translation', 'TRANSLATION', 'instruction', 'text', 'prompt']:
                        if k in val: return str(val[k]).strip()
                    for v in val.values():
                        if isinstance(v, str): return v.strip()
                    if val.values(): return str(list(val.values())[0]).strip()
                return str(val).strip() if val is not None else ""

            def extract_float(val, default=0.5):
                if isinstance(val, dict):
                    for k in ['TARGET', 'target', 'value', 'Magnitude', 'MAGNITUDE', 'magnitude']:
                        if k in val: 
                            try: return float(val[k])
                            except: pass
                    for v in val.values():
                        try: return float(v)
                        except: pass
                try: return float(val)
                except: return default

            def extract_bool(val, default=False):
                if isinstance(val, bool):
                    return val
                if isinstance(val, (int, float)):
                    return bool(val)
                if isinstance(val, str):
                    normalized = val.strip().lower()
                    if normalized in ("true", "yes", "1"): return True
                    if normalized in ("false", "no", "0", ""): return False
                return default

            def get_flexible(d, keys):
                """Busca una clave en un dict ignorando mayúsculas, espacios y guiones bajos."""
                if not isinstance(d, dict): return None
                # Crear mapa normalizado: "mandatorytasks" -> "MANDATORY TASKS"
                norm_map = {k.lower().replace(" ", "").replace("_", ""): k for k in d.keys()}
                for k in keys:
                    norm_k = k.lower().replace(" ", "").replace("_", "")
                    if norm_k in norm_map:
                        return d[norm_map[norm_k]]
                return None

            # 1. EXTRAER MAGNITUDE
            magnitude = 0.5
            m_val = get_flexible(data, ["magnitude"])
            if m_val is None:
                mt = get_flexible(data, ["mandatory tasks", "tasks"])
                if mt: m_val = get_flexible(mt, ["magnitude"])
            magnitude = extract_float(m_val, 0.5)
            magnitude = max(0.0, min(1.0, magnitude))

            # 2. EXTRAER MASK_TARGET
            mask_target = "subject"
            t_val = get_flexible(data, ["mask_target", "targeting"])
            if t_val is None:
                mt = get_flexible(data, ["mandatory tasks", "tasks"])
                if mt: t_val = get_flexible(mt, ["mask_target", "targeting"])
            if t_val:
                if isinstance(t_val, list) and t_val:
                    mask_target = str(t_val[0]).strip().lower()
                else:
                    mask_target = extract_text(t_val).lower()

            # 3. BUSCAR TRADUCCIÓN (PROMPT)
            translated_prompt = prompt
            p_val = get_flexible(data, ["prompt", "instruction", "output", "translation"])
            if p_val is None:
                mt = get_flexible(data, ["mandatory tasks", "tasks"])
                if mt: p_val = get_flexible(mt, ["translation", "prompt", "instruction"])
            
            if p_val:
                translated_prompt = extract_text(p_val)

            # Regiones independientes: una propiedad estructural del plan, no
            # una tabla de acciones. Se acepta solo un array para que una frase
            # libre del modelo no convierta accidentalmente un edit local en
            # recomposición global.
            affected_regions = []
            regions_val = get_flexible(data, ["affected_regions", "edit_regions", "regions"])
            if regions_val is None:
                mt = get_flexible(data, ["mandatory tasks", "tasks"])
                if mt:
                    regions_val = get_flexible(mt, ["affected_regions", "edit_regions", "regions"])
            if isinstance(regions_val, list):
                for region in regions_val:
                    value = extract_text(region).strip().lower()
                    if value and value not in affected_regions:
                        affected_regions.append(value)
            if not affected_regions:
                affected_regions = self._classify_affected_regions(prompt, llm)

            subject_query = get_flexible(data, ["subject_query", "edit_subject", "visual_subject"])
            subject_query = extract_text(subject_query).strip().lower() if subject_query is not None else ""
            if not subject_query:
                subject_query = self._classify_subject_query(prompt, llm)
            
            # VALIDATION: rechazar prompts con repetición excesiva de palabras
            _words = translated_prompt.lower().split()
            if len(_words) > 5:
                from collections import Counter
                _counts = Counter(_words)
                _most_common, _most_freq = _counts.most_common(1)[0]
                if _most_freq > 3 and _most_freq / len(_words) > 0.3:
                    print(f"[PromptRewriter] ⚠️ Repetición detectada ('{_most_common}' x{_most_freq}/{len(_words)}) → usando prompt original", flush=True)
                    translated_prompt = prompt
                    magnitude = extract_float(None, 0.5)
            
            # 4. PRESERVE FACE
            preserve_face = True
            id_val = get_flexible(data, ["preserve_face", "identity", "preserve face"])
            if id_val is None:
                mt = get_flexible(data, ["mandatory tasks", "tasks"])
                if mt: id_val = get_flexible(mt, ["identity", "preserve_face"])
            if id_val is not None:
                preserve_face = bool(id_val)

            # 5. IS GLOBAL
            is_global = False
            g_val = get_flexible(data, ["is_global", "scope", "is global"])
            if g_val is None:
                mt = get_flexible(data, ["mandatory tasks", "tasks"])
                if mt: g_val = get_flexible(mt, ["scope", "is_global"])
            if g_val is not None:
                is_global = extract_bool(g_val)

            # El router consume una propiedad semántica abstracta, no un listado
            # de verbos. Así una petición de acción, pose, interacción o nueva
            # composición llega al régimen de regeneración aunque use vocabulario
            # nunca visto por el código.
            regen_val = get_flexible(data, ["requires_scene_regeneration", "scene_regeneration"])
            if regen_val is None:
                mt = get_flexible(data, ["mandatory tasks", "tasks"])
                if mt:
                    regen_val = get_flexible(mt, ["requires_scene_regeneration", "scene_regeneration"])
            requires_scene_regeneration = extract_bool(regen_val, False)

            # If LLM says no change but user typed a real request, it likely missed it
            if magnitude < 0.1 and len(prompt.strip()) > 2:
                magnitude = 0.5

            result = {
                "prompt": translated_prompt,
                "magnitude": magnitude,
                "mask_target": mask_target,
                "affected_regions": affected_regions,
                "subject_query": subject_query,
                "preserve_face": preserve_face,
                "is_global": is_global,
                "requires_scene_regeneration": requires_scene_regeneration,
                "reasoning": data.get("reasoning", "Semantic analysis completed")
            }

            print(f"[PromptRewriter] Analysis: '{result['prompt'][:50]}...' | Mag: {result['magnitude']} | Regions: {result['affected_regions']} | Global: {result['is_global']} | Regen: {result['requires_scene_regeneration']} | Preserve: {result['preserve_face']}")
            
            return result
        except Exception as e:
            print(f"[PromptRewriter] Error parsing LLM JSON: {e}")
            return {"prompt": prompt, "magnitude": 0.5, "mask_target": "subject", "preserve_face": True, "is_global": True, "requires_scene_regeneration": False}

def _fix_malformed_json(s: str) -> str:
    """Preprocessa JSON malformado donde el LLM usa { 'valor' } en vez de 'valor' o valor."""
    def fix_bare_object(m):
        inner = m.group(1).strip()
        if re.match(r'^-?\d+(\.\d+)?$', inner):
            return inner
        if inner in ('true', 'false', 'null'):
            return inner
        return json.dumps(inner)
    s = re.sub(r'\{\s*"([^"]+)"\s*\}', fix_bare_object, s)
    s = re.sub(r"\{\s*'([^']+)'\s*\}", lambda m: json.dumps(m.group(1)), s)
    return s

def _extract_first_json(s: str) -> Optional[Dict]:
    """Extract the first complete JSON object from a string using brace-depth matching."""
    s = s.strip()
    s = re.sub(r'^```(?:json)?\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*```$', '', s)
    start = s.find('{')
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                if '"' not in candidate and "'" in candidate:
                    candidate = candidate.replace("'", '"')
                candidate = _fix_malformed_json(candidate)
                for attempt in [candidate, re.sub(r'[\x00-\x1f\x7f]', '', candidate)]:
                    try:
                        return json.loads(attempt)
                    except json.JSONDecodeError:
                        pass
                return None
    return None

_rewriter = None
def get_prompt_rewriter() -> PromptRewriter:
    global _rewriter
    if _rewriter is None:
        _rewriter = PromptRewriter()
    return _rewriter
