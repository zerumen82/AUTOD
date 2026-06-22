import re
import os
import json

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

_google_translator = None
_local_mappings = None


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'[<>]', '', text)
    return " ".join(text.split())


def is_error(text: str) -> bool:
    if not text:
        return True
    indicators = ["error", "server error", "that's an error", "please try again",
                  "500", "html", "doctype"]
    return any(x in text.lower() for x in indicators)


def _load_local_mappings():
    global _local_mappings
    if _local_mappings is not None:
        return _local_mappings
    _local_mappings = []
    try:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, "config", "translation_mappings.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _local_mappings = list(data.get("mappings") or [])
    except Exception:
        _local_mappings = []
    return _local_mappings


def _local_translate(text: str) -> str:
    """Fallback offline: frases largas primero, luego palabras sueltas."""
    out = text
    for src, dst in sorted(_load_local_mappings(), key=lambda x: len(x[0]), reverse=True):
        if src and dst:
            out = re.sub(re.escape(src), dst, out, flags=re.IGNORECASE)
    return clean_text(out)


def translate_prompt(prompt: str) -> str:
    """Traduce ES→EN: Google primero, fallback local si offline o falla."""
    global _google_translator

    if not prompt or not any(c.isalpha() for c in prompt) or len(prompt.strip()) < 4:
        return clean_text(prompt)

    original = clean_text(prompt)

    if GoogleTranslator is not None and \
       os.environ.get("HF_HUB_OFFLINE") != "1" and \
       os.environ.get("AUTOAUTO_OFFLINE") != "1":
        try:
            if _google_translator is None:
                _google_translator = GoogleTranslator(source="es", target="en")
            translated = _google_translator.translate(original)
            if translated and not is_error(translated):
                return clean_text(translated)
        except Exception as e:
            print(f"[Translate:Google] Error: {e}")

    local = _local_translate(original)
    if local and local != original:
        print("[Translate] Usando mappings locales (Google no disponible)")
        return local
    return original