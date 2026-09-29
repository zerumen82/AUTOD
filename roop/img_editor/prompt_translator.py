import re
import os
import json
from difflib import SequenceMatcher

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_mappings():
    path = os.path.join(get_project_root(), "config", "translation_mappings.json")
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[Translate] Error loading mappings: {e}")
    return {"mappings": [], "quality_fixups": []}

try:
    from deep_translator import GoogleTranslator, MyMemoryTranslator
except Exception as e:
    GoogleTranslator = None
    MyMemoryTranslator = None
    print(f"[Translate] deep_translator no disponible: {e}")

_google_translator = None
_mymemory_translator = None
_offline_translator = None
_offline_translation_failed = False


def _offline_model_dir() -> str:
    return os.path.join(get_project_root(), "models", "translation", "opus-mt-es-en")


def _normalize_prompt_instruction(text: str) -> str:
    """Corrige una ambigüedad gramatical ES→EN frecuente sin cambiar contenido.

    En prompts, el imperativo español sin sujeto ("conserva sus caras") puede
    salir como "he retains their faces". Cuando el patrón es inequívocamente
    una orden de preservar, se restaura el imperativo que entiende el modelo de
    imagen. No se añaden atributos ni se tocan sujetos/objetos.
    """
    return re.sub(
        r"\b(?:he|she|it)\s+(?:retains?|preserves?|keeps?)\s+(?=(?:their|the)\b)",
        "preserve ",
        text,
        flags=re.IGNORECASE,
    )


def _normalize_unambiguous_spanish_spelling(text: str) -> str:
    """Corrige únicamente una duplicación ortográfica imposible en español.

    ``hh`` no forma parte de la ortografía española; normalmente es una pulsación
    repetida (``chhico``). Se corrige antes de Marian para que no se convierta en
    un token desconocido inglés. No toca vocabulario ni intención del prompt.
    """
    def _single_h(match):
        return "H" if match.group(0).isupper() else "h"
    return re.sub(r"h{2,}", _single_h, text or "", flags=re.IGNORECASE)


def _normalize_stray_word_punctuation(text: str) -> str:
    """Remove a stray dot immediately before a word, such as 'sin .medias'."""
    return re.sub(r"(?<=\s)\.(?=\w)", "", text or "")


def _offline_translate(prompt: str):
    """Traducción ES→EN local y determinista.

    El modelo Marian se carga perezosamente en CPU: evita tanto los límites de
    Google/MyMemory como el rewriter generativo, que puede inventar detalles en
    prompts compuestos. Si el modelo no está completo, el flujo anterior queda
    disponible como fallback sin bloquear una edición.
    """
    global _offline_translator, _offline_translation_failed
    if not prompt:
        return None
    model_dir = _offline_model_dir()
    required = ("config.json", "pytorch_model.bin", "source.spm", "target.spm")
    if not all(os.path.isfile(os.path.join(model_dir, name)) for name in required):
        return None
    try:
        if _offline_translator is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, local_files_only=True)
            model.eval()
            _offline_translator = (tokenizer, model, torch)
            print(f"[Translate:Local] Modelo ES→EN cargado desde {model_dir}", flush=True)
        tokenizer, model, torch = _offline_translator
        # SentencePiece translation models were trained on normal prose.  An
        # Mostly uppercase messages are an orthographic presentation choice,
        # not an edit instruction. A single lowercase letter ("eL ...") must
        # not disable normalization: Marian then misreads ordinary Spanish
        # words. This changes casing only, without an edit vocabulary.
        source = prompt
        cased = [c for c in prompt if c.isalpha() and c.lower() != c.upper()]
        uppercase_share = sum(c.isupper() for c in cased) / max(1, len(cased))
        if len(cased) >= 8 and uppercase_share >= 0.80:
            source = prompt.lower()
            print("[Translate:Local] Normalizando entrada predominantemente en MAYÚSCULAS", flush=True)
        normalized_punctuation = _normalize_stray_word_punctuation(source)
        if normalized_punctuation != source:
            print("[Translate:Local] Eliminado punto suelto antes de una palabra", flush=True)
        corrected_source = _normalize_unambiguous_spanish_spelling(normalized_punctuation)
        if corrected_source != normalized_punctuation:
            print("[Translate:Local] Corregida duplicación ortográfica inequívoca antes de traducir", flush=True)
        source = corrected_source
        words = max(1, len(re.findall(r"\w+", source)))
        encoded = tokenizer(source, return_tensors="pt", truncation=True, max_length=256)
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                num_beams=4,
                max_new_tokens=min(256, max(32, words * 5 + 24)),
                early_stopping=True,
            )
        translated = _normalize_prompt_instruction(
            clean_text(tokenizer.decode(output_ids[0], skip_special_tokens=True))
        )
        if not translated or translated == clean_text(prompt) or is_error(translated):
            return None
        # La puntuación española se traduce a menudo a punto y coma o a una
        # coordinación inglesa. Exigimos una pérdida clara, no una diferencia
        # de un separador, para no descartar una traducción completa y fiel.
        raw_conditions = _count_conditions(prompt)
        out_conditions = _count_conditions(translated)
        if raw_conditions >= 3 and out_conditions < max(2, raw_conditions - 1):
            print(
                f"[Translate:Local] Descartada: perdió condiciones ({raw_conditions}→{out_conditions})",
                flush=True,
            )
            return None
        print(f"[Translate:Local] '{prompt[:50]}...' -> '{translated[:70]}...'", flush=True)
        return translated
    except Exception as e:
        # Una importación/transitorio fallido no debe inutilizar el traductor
        # local durante toda la sesión. Esta función se invoca por edición, no
        # por fotograma: reintentar en la siguiente petición es barato y evita
        # caer permanentemente a una API rate-limited.
        _offline_translator = None
        _offline_translation_failed = False
        print(f"[Translate:Local] No disponible en esta petición: {e}; se reintentará", flush=True)
        return None

def clean_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[<>]', '', text)
    text = " ".join(text.split())
    return text

def _damerau_leven(a: str, b: str) -> int:
    """Distancia de edición con transposición adyacente (typos tipo 'desclaza'
    vs 'descalza' = 1). Puro Python, sin dependencias."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return 99
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _apply_mappings(p: str, data: dict) -> str:
    p_lower = p.lower()
    for m in data.get("mappings", []):
        if isinstance(m, (list, tuple)):
            trigger, replacement = m[0], m[1]
        else:
            trigger, replacement = m["trigger"], m["replacement"]
        if trigger in p_lower:
            p = re.sub(re.escape(trigger), replacement, p, flags=re.IGNORECASE)
        p_lower = p.lower()
    return p


def _local_translate(prompt: str):
    """Devuelve (texto, typo_reparado). El fuzzy es genérico: acerca cualquier
    palabra a cualquier trigger del JSON (no hay listas nuevas en código).
    Umbral estricto para no corromper palabras correctas."""
    if not prompt:
        return prompt, False
    p = clean_text(prompt)
    data = load_mappings()
    p = _apply_mappings(p, data)

    # Vocabulario: palabras largas de los triggers (len>=6)
    vocab = set()
    for m in data.get("mappings", []):
        trig = m[0] if isinstance(m, (list, tuple)) else m["trigger"]
        for w in re.findall(r"[a-zA-Záéíóúñü]+", trig.lower()):
            if len(w) >= 6:
                vocab.add(w)

    typo_fixed = False
    for w in set(re.findall(r"[a-zA-Záéíóúñü]+", p)):
        wl = w.lower()
        if len(wl) < 5 or wl in vocab:
            continue
        best, best_d = None, 99
        for v in vocab:
            dd = _damerau_leven(wl, v)
            if dd < best_d:
                best, best_d = v, dd
        limit = 2 if (len(wl) >= 9 and len(best or "") >= 9) else 1
        if best and best_d <= limit:
            p = re.sub(r"\b" + re.escape(w) + r"\b", best, p)
            typo_fixed = True

    if typo_fixed:
        p = _apply_mappings(p, data)
    return clean_text(p), typo_fixed

def is_error(text: str) -> bool:
    if not text:
        return True
    indicators = ["error", "server error", "that's an error", "please try again",
                  "500", "html", "doctype", " Internal Server Error"]
    return any(x in text.lower() for x in indicators)

# v5.1140 — Gate del backup MyMemory. El fallo característico de su tier
# gratuito son FRASES CANNED del corpus: la misma salida para inputs distintos
# ('una mujer rubia en la playa' → 'Look in the Auction'). Se detecta por
# estructura, sin lista de palabras: (a) palabras ES largas del original que
# sobreviven intactas (passthrough, no traducción), (b) cache de respuestas
# vistas — la misma salida ya devuelta para UN INPUT DISTINTO es basura
# garantizada. Un falso positivo solo cuesta caer al fallback local (que es
# idéntico/mappings); un falso negativo mandaba basura al modelo.
_MM_SEEN_RESPONSES = {}


def _is_english_word(w: str) -> bool:
    """Cognados EN legítimos que un traductor ES→EN deja intactos."""
    return w in {
        "hotel", "restaurant", "tennis", "supermarket", "hospital", "animal",
        "camera", "video", "terminal", "director", "editor", "original",
        "general", "natural", "central", "mental", "final", "total",
        "important", "elegant", "transparent", "drama", "era", "area",
        "idea", "extra", "artist", "digital", "tropical", "metal",
    }


def _looks_like_real_translation(src: str, out: str) -> bool:
    try:
        out_norm = " ".join(re.findall(r"[a-zA-Z]+", (out or "").lower()))
        if not out_norm:
            return False
        # (a) Survivor: ≥2 palabras ES largas (≥5 letras) del original intactas
        # en la salida, o ≥1 survivor cuando la salida es MÁS CORTA que el
        # original (una traducción real nunca acorta: 'mujer' intacta en una
        # salida de 2 palabras = passthrough). (el ES→EN real cambia esas
        # palabras SIEMPRE; el vocabulario ES no colisiona con EN salvo
        # cognados ya excluidos).
        out_words = set(out_norm.split())
        es_survivors = [
            w for w in re.findall(r"[a-zA-Z]{5,}", (src or "").lower())
            if w in out_words and not _is_english_word(w)
        ]
        _src_words = len(re.findall(r"[a-zA-Z]+", (src or "")))
        _out_words_n = len(out_norm.split())
        if len(es_survivors) >= 2 or (es_survivors and _out_words_n < _src_words):
            return False
        # (b) Cache canned: misma salida ya vista para UN INPUT DISTINTO →
        # respuesta del corpus, no traducción del pedido.
        _norm_src = " ".join(re.findall(r"[a-zA-Z]+", (src or "").lower()))
        prev_src = _MM_SEEN_RESPONSES.get(out_norm)
        if prev_src is not None and prev_src != _norm_src:
            return False
        _MM_SEEN_RESPONSES[out_norm] = _norm_src
        # Cap del cache (dict pequeñísimo, sesiones largas)
        if len(_MM_SEEN_RESPONSES) > 256:
            _MM_SEEN_RESPONSES.pop(next(iter(_MM_SEEN_RESPONSES)))
        return True
    except Exception:
        return True  # ante duda no bloquear (cae al fallback local igualmente)


def _api_translate(prompt: str) -> str:
    if GoogleTranslator is None and MyMemoryTranslator is None:
        return None

    data = load_mappings()

    global _google_translator, _mymemory_translator

    try:
        if _google_translator is None:
            _google_translator = GoogleTranslator(source='auto', target='en')
        try:
            translated = _google_translator.translate(prompt)
        except Exception as ge:
            # v5.1140: 'too many requests' es TRANSITORIO (5 req/s de Google).
            # Antes caía directo al backup MyMemory (flaky: devuelve frases de
            # corpus sin relación con el pedido). 1 reintento tras pausa corta
            # resuelve el caso común sin tocar el backup.
            if "too many request" in str(ge).lower() or "429" in str(ge):
                import time as _t
                _t.sleep(1.5)
                try:
                    _google_translator = GoogleTranslator(source='auto', target='en')
                    translated = _google_translator.translate(prompt)
                    print("[Translate:Google] Rate-limit → reintento tras 1.5s: OK")
                except Exception as ge2:
                    raise ge2
            else:
                raise
        if translated and translated != prompt and not is_error(translated):
            translated = clean_text(translated)
            for f in data.get("quality_fixups", []):
                if isinstance(f, dict):
                    if f["find"].lower() in translated.lower():
                        if "condition" in f:
                            cond = f["condition"]
                            if "not in prompt" in cond:
                                word = cond.split(" ")[0]
                                if word not in prompt.lower():
                                    translated = translated.replace(f["find"], f["replace"])
                        else:
                            translated = translated.replace(f["find"], f["replace"])
            print(f"[Translate:Google] '{prompt[:50]}...' -> '{translated[:50]}...'")
            return translated
    except Exception as e:
        print(f"[Translate:Google] Error: {e}. Intentando backup...")

    try:
        # v5.1107b/c: source explícito con código locale COMPLETO. Con 'auto'
        # fallaba ("No support...", log 11:40) y con 'es' corto TAMBIÉN
        # (log 11:57: la API solo acepta códigos como 'es-ES', ver su propia
        # lista de error). Si no es inglés, es español del usuario.
        _src = 'en-GB' if _is_english(prompt) else 'es-ES'
        if _mymemory_translator is None or getattr(_mymemory_translator, 'source', None) != _src:
            _mymemory_translator = MyMemoryTranslator(source=_src, target='en-GB')
        translated = _mymemory_translator.translate(prompt)
        # v5.1140: MyMemory (email-less free tier) a veces responde BASURA —
        # frases canned de su corpus sin relación con el pedido ('una mujer
        # rubia en la playa' → 'Look in the Auction'). Gate ESTRUCTURAL:
        # (a) survivor check — ≥2 palabras ES largas del original intactas en
        # la salida = passthrough, no traducción; (b) cache de respuestas
        # canned — la MISMA salida vista antes para UN INPUT DISTINTO es
        # basura garantizada (el fallo característico de MyMemory: devuelve
        # la misma frase del corpus para pedidos no relacionados).
        if translated and not _is_english(prompt) and not _looks_like_real_translation(prompt, translated):
            print(
                f"[Translate:MyMemory] Descartado (parece basura, no una traducción): "
                f"'{str(translated)[:50]}...'",
            )
            translated = None
        if translated and translated != prompt and not is_error(translated):
            translated = clean_text(translated)
            print(f"[Translate:MyMemory] '{prompt[:50]}...' -> '{translated[:50]}...'")
            return translated
    except Exception as e:
        print(f"[Translate:MyMemory] Falló backup: {e}")

    return None

# Palabras función ES (detección de idioma, no de intención de edit)
_ES_FUNC = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "al",
    "que", "con", "por", "para", "en", "es", "son", "se", "su", "sus", "mi",
    "tu", "y", "o", "pero", "como", "mas", "más", "muy", "sin", "sobre",
    "este", "esta", "estos", "estas", "ese", "esa", "aqui", "aquí", "alli",
    "colorea", "colorear", "cambia", "cambiar", "quita", "quitar", "añade",
    "anade", "añadir", "haz", "hacer", "pon", "poner", "mejora", "mejorar",
    "personajes", "persona", "fondo", "imagen", "foto", "comic", "cómic",
    "desnuda", "desnudo", "ropa", "pelo", "cara", "cuerpo",
})


def _is_english(text: str) -> bool:
    """
    Heurística de idioma para decidir si traducir.
    No es hardcode de intención de edit: solo evita mandar español al modelo EN.
    """
    if not text:
        return True
    clean = text.strip()
    if not clean:
        return True
    # Acentos / ñ → no inglés
    if any(ord(c) > 127 for c in clean):
        return False
    words = re.findall(r"[a-zA-Z]+", clean.lower())
    if not words:
        return True
    # Morfología: imperativos con pronombre enclítico (haz+lo, quita+la,
    # pon+le...). Se pela UN sufijo pronominal y se mira la raíz. Regla
    # gramatical, cero vocabulario nuevo.
    _ENCLITICS = ("los", "las", "les", "nos", "lo", "la", "le", "me", "te", "se", "os")
    def _roots(w):
        yield w
        for s in _ENCLITICS:
            if w.endswith(s) and len(w) - len(s) >= 3:
                yield w[: -len(s)]
                break
    es_hits = sum(1 for w in words if any(r in _ES_FUNC for r in _roots(w)))
    # ≥1 función ES en prompts cortos, o ≥20% en largos → español
    if len(words) <= 8 and es_hits >= 1:
        return False
    if es_hits / max(len(words), 1) >= 0.18:
        return False
    return True


def _count_conditions(text: str) -> int:
    """v5.1110 — número aproximado de CONDICIONES de un prompt compuesto.

    Los pedidos del usuario encadenan condiciones con conjunciones ("y", "e",
    "y también", commas enumerativas). Cuenta segmentos: un compuesto de 2
    ("desnuda Y descalza") = 2. Es detección de ESTRUCTURA sintáctica, no de
    vocabulario: no mira QUÉ se pide, solo CUÁNTAS partes tiene el pedido.
    Sirve para verificar que la traducción no se comió una mitad.
    """
    if not text:
        return 0
    t = clean_text(text).lower()
    t = re.sub(r"[\"'\.]", " ", t)
    # Conjunciones ES/EN separadoras de condiciones + comas enumerativas.
    # Cada segmento (aunque sea 1 palabra: "desnuda y descalza") cuenta como
    # una condición — la mitad corta también se pierde si la traducción la
    # omite (log 11:57).
    parts = re.split(r"\by\b|\be\b|\band\b|\bplus\b|,", t)
    return len([p for p in parts if p.split()])


def _translation_lost_conditions(raw: str, translated: str) -> bool:
    """True si el prompt crudo es compuesto (≥2 condiciones) pero la traducción
    tiene MENOS condiciones que el original → la API/local se comió una mitad
    (log 12:12: 'desclaza' → 'unclad', sin pies). Solo detecta PÉRDIDA, nunca
    modifica contenido: si dispara, el caller usa el fallback local completo.
    """
    if not raw or not translated:
        return False
    raw_n = _count_conditions(raw)
    tr_n = _count_conditions(translated)
    return raw_n >= 2 and tr_n < raw_n


def translation_suspect_tokens(raw: str, out: str):
    """Tokens que parecen residuos no traducidos en una salida inglesa.

    La detección es 100% estructural: longitud, capitalización y similitud de
    caracteres; no contiene vocabulario de instrucciones. Además de un token
    idéntico, cubre un typo que el traductor translitera dentro de una frase por
    lo demás correcta.
    - TODO MAYÚSCULAS 3-8 letras ('LZA', 'PENE', 'CHICA') → no traducido:
      typo o palabra que el traductor no tocó; el rewriter 0.5B alucina con
      eso (log 13:33: inventó 'naked and barefoot' desde typos y el Body
      SAFETY secuestró el run).
    - MAYÚSCULAS Iniciales 2-8 letras ('El', 'La', 'Le') en un texto cuya
      traducción NO es inglés (el traductor devuelve ES si no pudo): lo
      normal ES→EN es minúsculas ('the', 'a'); una capitalizada intacta =
      palabra que la API no reconoció/omitó.
    Devuelve la lista (vacía = limpio)."""
    try:
        # La salida puede ser inglés *y* conservar un token roto del original
        # ("DFBE" -> "the boy dfbe be bald"). Precisamente ése es el caso que
        # queremos señalar: que el resto de la frase ya parezca inglés no vuelve
        # válido al token que el traductor no entendió.
        if not raw or not out or _is_english(raw):
            return []
        # 100% estructural: si la salida es IDÉNTICA a la entrada y la entrada
        # no es inglés, el traductor local no hizo NADA (log 15:23: 'LA CHICA
        # LE suck EL PENE' pasó crudo porque el typo-fixer reparó algo, se
        # saltó la API, y el local no tenía mappings). Un traductor real
        # SIEMPRE cambia el texto de idioma. Igualdad de strings = cero
        # vocabulario.
        if raw.strip() == out.strip():
            return list(dict.fromkeys(re.findall(r"[A-Za-z]{2,}", raw)))[:8]
        raw_forms = {}
        for t in re.findall(r"[A-Za-z]+", raw):
            raw_forms.setdefault(t.lower(), t)
        out_toks = set(t.lower() for t in re.findall(r"[A-Za-z]+", out))
        sus = []
        for low in (set(raw_forms) & out_toks):
            form = raw_forms[low]
            # POV is already the standard English camera term; preserving it
            # through ES→EN translation is expected, not a failed translation.
            if low == "pov":
                continue
            if form.isupper() and 3 <= len(low) <= 8:
                sus.append(form)
            elif form.istitle() and 2 <= len(low) <= 8:
                sus.append(form)
        # Un typo puede cambiar lo justo para no sobrevivir de forma idéntica:
        # el traductor resuelve el resto de la frase, pero deja un residuo muy
        # parecido al token original. Marcamos ambos extremos para que el gate
        # de fidelidad pueda aceptar exclusivamente una reparación corta.
        for raw_low, raw_form in raw_forms.items():
            if not (raw_form.isupper() and 4 <= len(raw_low) <= 16):
                continue
            for out_low in out_toks:
                if out_low == raw_low or len(out_low) < 4:
                    continue
                if SequenceMatcher(None, raw_low, out_low).ratio() >= 0.68:
                    sus.extend((raw_form, out_low))
        return sorted(dict.fromkeys(sus))
    except Exception:
        return []


def _warn_passthrough(raw: str, out: str):
    """Aviso (solo log) si la traducción dejó pasar tokens intactos."""
    try:
        for t in translation_suspect_tokens(raw, out):
            print(f"[ImgEditor] AVISO traducción: '{t}' pasó sin traducir — ¿typo? (revisa el prompt)", flush=True)
    except Exception:
        pass


def translate_prompt(prompt: str) -> str:
    out = _translate_prompt_impl(prompt)
    _warn_passthrough(prompt, out)
    return out


def _translate_prompt_impl(prompt: str) -> str:
    if not prompt or not any(c.isalpha() for c in prompt):
        return prompt

    if len(prompt.strip()) < 4:
        return clean_text(prompt)

    local_result, typo_fixed = _local_translate(prompt)

    # La ruta principal para español es local: conserva todas las condiciones y
    # sigue funcionando sin red. Las APIs quedan como respaldo si el archivo del
    # modelo no existe o falla al cargar.
    if not _is_english(prompt):
        # Primero traducimos el español original. Los mappings locales son
        # útiles para typos como respaldo, pero mezclar media frase ES/EN antes
        # de Marian degrada su gramática ("una"→"a", por ejemplo).
        translated_local = _offline_translate(prompt)
        if not translated_local and local_result != clean_text(prompt):
            translated_local = _offline_translate(local_result)
        if translated_local:
            return translated_local
        if typo_fixed:
            # v5.1109: typo reparado → se confía en local y se OMITE la API.
            # La traducción automática con typos ALUCINA (log 12:12: "desclaza"
            # → "unclad", se perdió la mitad de pies). Lo determinista gana.
            # v5.1114: PERO solo si el local realmente TRADUJO. El fuzzy solo
            # repara palabras que están en el vocabulario de triggers — un
            # prompt cuyo vocabulario no tiene mapping (log 15:23: 'LA CHICA
            # LE suck EL PENE') sale idéntico y sin traducir, y con el skip
            # de la API el español crudo llegaba al T5 (el rewriter alucinó
            # encima). Si quedan tokens sin traducir → API sobre el texto YA
            # reparado (el raw con typo sí alucina, log 12:12).
            if not translation_suspect_tokens(prompt, local_result):
                print(f"[Translate] Typo reparado, se omite API: '{local_result[:60]}...'")
                return local_result
            if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("AUTOAUTO_OFFLINE") == "1":
                print(f"[Translate] Typo reparado, offline → local parcial: '{local_result[:60]}...'")
                return local_result
            print(f"[Translate] Typo reparado pero quedan tokens sin traducir → API sobre texto reparado: '{local_result[:60]}...'")
            translated = _api_translate(local_result)
            if translated and translated != local_result and not is_error(translated):
                if _translation_lost_conditions(local_result, translated):
                    print(f"[Translate] API perdió una condición del compuesto → local completo: '{local_result[:60]}...'")
                    return local_result
                return clean_text(translated)
            return local_result
        if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("AUTOAUTO_OFFLINE") == "1":
            print(f"[Translate] ES detectado, offline → local: '{local_result[:60]}...'")
            return local_result
        translated = _api_translate(prompt)
        if translated:
            # v5.1110: verificación de compuestos — si la API se comió una
            # condición (traduce peor con typos/frases raras), gana el local.
            if _translation_lost_conditions(prompt, translated):
                print(f"[Translate] API perdió una condición del compuesto ({_count_conditions(prompt)}→{_count_conditions(translated)}) → local completo: '{local_result[:60]}...'")
                return local_result
            return translated
        # Fallback: local + si no cambió, devolver local igual
        if local_result and local_result != clean_text(prompt):
            return local_result
        print(f"[Translate] ES sin API; usando local: '{local_result[:60]}...'")
        return local_result

    already_english = (
        _is_english(prompt)
        and local_result == clean_text(prompt)
    )
    if already_english:
        return local_result

    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("AUTOAUTO_OFFLINE") == "1":
        return local_result

    translated = _api_translate(prompt)
    if translated:
        # v5.1110: misma verificación para la rama no-ES (prompt mixto).
        if _translation_lost_conditions(prompt, translated):
            print(f"[Translate] API perdió una condición del compuesto ({_count_conditions(prompt)}→{_count_conditions(translated)}) → local completo")
            return local_result
        return translated

    return local_result
