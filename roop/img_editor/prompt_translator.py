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
    if not text: return ""
    text = re.sub(r'[<>]', '', text)
    text = " ".join(text.split())
    return text

def _local_translate(prompt: str) -> str:
    if not prompt:
        return prompt
    p = clean_text(prompt)
    data = load_mappings()
    p_lower = p.lower()

    for m in data.get("mappings", []):
        if isinstance(m, (list, tuple)):
            trigger, replacement = m[0], m[1]
        else:
            trigger, replacement = m["trigger"], m["replacement"]
        if trigger in p_lower:
            p = re.sub(re.escape(trigger), replacement, p, flags=re.IGNORECASE)
        p_lower = p.lower()

    return clean_text(p)

def is_error(text: str) -> bool:
    if not text:
        return True
    indicators = ["error", "server error", "that's an error", "please try again",
                  "500", "html", "doctype", " Internal Server Error"]
    return any(x in text.lower() for x in indicators)

def _api_translate(prompt: str) -> str:
    if GoogleTranslator is None and MyMemoryTranslator is None:
        return None

    data = load_mappings()

    global _google_translator, _mymemory_translator

    try:
        if _google_translator is None:
            _google_translator = GoogleTranslator(source='auto', target='en')
        translated = _google_translator.translate(prompt)
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
        if _mymemory_translator is None:
            _mymemory_translator = MyMemoryTranslator(source='auto', target='en')
        translated = _mymemory_translator.translate(prompt)
        if translated and translated != prompt and not is_error(translated):
            translated = clean_text(translated)
            print(f"[Translate:MyMemory] '{prompt[:50]}...' -> '{translated[:50]}...'")
            return translated
    except Exception as e:
        print(f"[Translate:MyMemory] Falló backup: {e}")

    return None

def translate_prompt(prompt: str) -> str:
    if not prompt or not any(c.isalpha() for c in prompt):
        return prompt

    if len(prompt.strip()) < 4:
        return clean_text(prompt)

    if os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("AUTOAUTO_OFFLINE") == "1":
        return _local_translate(prompt)

    translated = _api_translate(prompt)
    if translated:
        return translated

    return _local_translate(prompt)
