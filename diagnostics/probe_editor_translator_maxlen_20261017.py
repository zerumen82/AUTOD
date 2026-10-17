"""Contrato v5.1223 (cosmético): sin el aviso «Both max_new_tokens and max_length».

Cada llamada al Marian local (mismo modelo del Editor y de Generate) imprimía
en consola el warning de transformers: la config de Marian trae
max_length=512 y la llamada pasa max_new_tokens. La condición está en
transformers/generation/utils.py:2339 — el aviso solo se emite si
`self.generation_config.max_length` NO es None; justo después
`generate()` recalcula `max_length = max_new_tokens + input_ids_length`.
Poner ese atributo en None al cargar el modelo cambia exactamente una cosa:
deja de avisar. El resto del comportamiento (traducción) es idéntico.

Grupo 1 (unidad): tras la carga, `model.generation_config.max_length` es None.
Grupo 2 (flujo): traducir con captura del logger de transformers → cero avisos
y traducción no vacía. Sin modelo local → SKIP legítimo (exit 0).
"""
import logging

from roop.img_editor import prompt_translator as pt


_WARNING = "Both `max_new_tokens`"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


def _ensure_model():
    out = pt._offline_translate("una casa blanca")
    model = None
    if pt._offline_translator is not None:
        model = pt._offline_translator[1]
    return out, model


def group_config():
    out, model = _ensure_model()
    if model is None:
        print("SKIP: modelo Marian local ausente (mismo patrón que los SKIP de fixture)")
        return
    assert model.generation_config.max_length is None, (
        "max_length sigue en %r: transformers seguirá avisando en cada "
        "generate() con max_new_tokens" % (model.generation_config.max_length,)
    )
    print("OK: generation_config.max_length=%r tras la carga" % (
        model.generation_config.max_length,))


def group_no_warning():
    out, model = _ensure_model()
    if model is None:
        print("SKIP: modelo Marian local ausente")
        return
    capture = _Capture()
    logger = logging.getLogger("transformers")
    logger.addHandler(capture)
    try:
        translated = pt._offline_translate("el sofá es de color gris")
    finally:
        logger.removeHandler(capture)
    warnings = [m for m in capture.records if _WARNING in m]
    assert not warnings, warnings[0]
    assert translated and translated.strip(), "traducción vacía"
    print("OK: 0 avisos de max_new_tokens/%s y traducción: %s" % (
        "max_length", translated))


def main():
    failures = []
    for group in (group_config, group_no_warning):
        try:
            group()
        except AssertionError as error:
            failures.append("%s: %s" % (group.__name__, error))
            print("ROJO: %s: %s" % (group.__name__, error))
    if failures:
        raise SystemExit("%d grupo(s) en rojo" % len(failures))
    print("TODO VERDE")


if __name__ == "__main__":
    main()
