import re
import os
import json

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

def clean_text(text: str) -> str:
    """Elimina caracteres que rompen el prompt y limpia el formato"""
    if not text: return ""
    # Eliminar etiquetas HTML accidentales o simbolos de escape
    text = re.sub(r'[<>]', '', text)
    # Normalizar espacios
    text = " ".join(text.split())
    return text

def _local_translate(prompt: str) -> str:
    """Traducción 100% local usando mappings + reglas básicas. Sin internet."""
    if not prompt:
        return prompt
    p = clean_text(prompt)
    data = load_mappings()
    p_lower = p.lower()

    # Aplicar mappings del config
    for m in data.get("mappings", []):
        if m["trigger"] in p_lower:
            p = p.replace(m["trigger"], m["replacement"])

    # Reglas básicas ES -> EN comunes para edición de imagen (sin hardcode agresivo)
    replacements = [
        ("desnudala", "completely undress her, make her fully naked"),
        ("desnúdala", "completely undress her, make her fully naked"),
        ("desnuda", "completely naked, no clothes, bare skin"),
        ("descalza", "barefoot"),
        ("debe ir descalza y desnuda", "must be completely barefoot and fully naked"),
        ("descalza y desnuda", "barefoot and fully naked"),
        ("ponle", "put on her"),
        ("quítale", "remove from her"),
        ("cámbiale", "change her"),
        ("haz que", "make them"),
        ("que estén", "have them"),
        ("bailando", "dancing"),
        ("corriendo", "running"),
        ("caminando", "walking"),
        ("sentados", "sitting"),
        ("de rodillas", "kneeling"),
        ("el fondo", "the background"),
        ("la ropa", "the clothes"),
        ("la cara", "the face"),
        ("más realista", "more realistic"),
        ("mejor calidad", "higher quality"),
    ]
    for es, en in replacements:
        if es in p_lower:
            p = p.replace(es, en) if es in p else p.replace(es.capitalize(), en)

    return clean_text(p)


def translate_prompt(prompt: str) -> str:
    """Traducción. Prioriza modo local/offline. Solo usa Google si está disponible y NO en modo offline."""
    if not prompt or not any(c.isalpha() for c in prompt):
        return prompt
    
    if len(prompt.strip()) < 4:
        return clean_text(prompt)

    # Siempre preferir local primero (sin internet)
    local = _local_translate(prompt)
    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("AUTOAUTO_OFFLINE") == "1":
        return local

    if GoogleTranslator is None or MyMemoryTranslator is None:
        return local

    # Intentar externo solo si no estamos en offline y las libs existen (raro)
    try:
        data = load_mappings()
        if _google_translator is None:
            _google_translator = GoogleTranslator(source='auto', target='en')
        translated = _google_translator.translate(prompt)
        if translated and translated != prompt and not is_error(translated):
            translated = clean_text(translated)
            print(f"[Translate:Google] '{prompt[:40]}...' -> '{translated[:40]}...'")
            return translated
    except Exception:
        pass

    # Fallback final a local (sin internet)
    return local

    def is_error(text: str) -> bool:
        """Detecta si el traductor devolvió un mensaje de error en vez de traducción"""
        if not text:
            return True
        indicators = ["error", "server error", "that's an error", "please try again",
                      "500", "html", "doctype", " Internal Server Error"]
        return any(x in text.lower() for x in indicators)

    # 0. Pre-correcciones Dinámicas (Desde JSON)
    data = load_mappings()
    p_lower = prompt.lower()
    for m in data.get("mappings", []):
        if m["trigger"] in p_lower:
            prompt = prompt.replace(m["trigger"], m["replacement"])

    # 1. Intentar con Google (Principal)
    try:
        if _google_translator is None:
            _google_translator = GoogleTranslator(source='auto', target='en')
        translated = _google_translator.translate(prompt)
        if translated and translated != prompt and not is_error(translated):
            translated = clean_text(translated)
            
            # Fixups Dinámicos
            for f in data.get("quality_fixups", []):
                if f["find"].lower() in translated.lower():
                    # Evaluar condición simple
                    if "condition" in f:
                        cond = f["condition"]
                        if "not in prompt" in cond:
                            word = cond.split(" ")[0]
                            if word not in prompt.lower():
                                translated = translated.replace(f["find"], f["replace"])
                    else:
                        translated = translated.replace(f["find"], f["replace"])

            print(f"[Translate:Google] '{prompt[:40]}...' -> '{translated[:40]}...'")
            return translated
    except Exception as e:
        print(f"[Translate:Google] Error: {e}. Intentando backup...")

    # 2. Intentar con MyMemory (Backup)
    try:
        if _mymemory_translator is None:
            _mymemory_translator = MyMemoryTranslator(source='auto', target='en')
        translated = _mymemory_translator.translate(prompt)
        if translated and translated != prompt and not is_error(translated):
            translated = clean_text(translated)
            print(f"[Translate:MyMemory] '{prompt[:40]}...' -> '{translated[:40]}...'")
            return translated
    except Exception as e:
        print(f"[Translate:MyMemory] Falló backup: {e}")

    # 3. Fallback: devolver el prompt original si todo falla
    print(f"[Translate] Fallback: usando prompt original")
    return clean_text(prompt)
