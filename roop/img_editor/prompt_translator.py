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

def translate_prompt(prompt: str) -> str:
    global _google_translator, _mymemory_translator
    if not prompt or not any(c.isalpha() for c in prompt):
        return prompt
    
    # Si el prompt es muy corto, no intentamos traducir con Google para evitar alucinaciones
    if len(prompt.strip()) < 4:
        return clean_text(prompt)

    if GoogleTranslator is None or MyMemoryTranslator is None:
        return prompt

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
