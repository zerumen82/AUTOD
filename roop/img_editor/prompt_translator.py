from deep_translator import GoogleTranslator, MyMemoryTranslator

_google_translator = None
_mymemory_translator = None

def translate_prompt(prompt: str) -> str:
    global _google_translator, _mymemory_translator
    if not prompt or not any(c.isalpha() for c in prompt):
        return prompt
        
    # 1. Intentar con Google (Principal)
    try:
        if _google_translator is None:
            _google_translator = GoogleTranslator(source='auto', target='en')
        translated = _google_translator.translate(prompt)
        if translated and translated != prompt:
            print(f"[Translate:Google] '{prompt[:40]}...' -> '{translated[:40]}...'")
            return translated
    except Exception as e:
        print(f"[Translate:Google] Error: {e}. Intentando backup...")

    # 2. Intentar con MyMemory (Backup)
    try:
        if _mymemory_translator is None:
            _mymemory_translator = MyMemoryTranslator(source='auto', target='en')
        translated = _mymemory_translator.translate(prompt)
        if translated and translated != prompt:
            print(f"[Translate:MyMemory] '{prompt[:40]}...' -> '{translated[:40]}...'")
            return translated
    except Exception as e:
        print(f"[Translate:MyMemory] Falló backup: {e}")

    return prompt
