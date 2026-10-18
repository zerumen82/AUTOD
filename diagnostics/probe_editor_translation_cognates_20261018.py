"""Cognate translations are not evidence of untranslated typo residue (CPU)."""

from unittest.mock import patch

from roop.img_editor import prompt_translator
from roop.img_editor.prompt_translator import translation_suspect_tokens


def main():
    clean = [
        ("HAZ LA PENETRACION", "make the penetration"),
        ("HAZ LA ANIMACION", "make the animation"),
        ("HAZ LA TRANSFORMACION", "make the transformation"),
        ("HAZ LA CELEBRACION", "make the celebration"),
        ("HAZ LA CIRCULACION", "make the circulation"),
        ("CAMBIA EL POV", "change the pov"),
        ("change the viewpoint", "change the viewpoint"),
    ]
    failures = []
    for raw, translated in clean:
        actual = translation_suspect_tokens(raw, translated)
        print(f"{'PASS' if not actual else 'FAIL'} clean {raw!r}: {actual}")
        if actual:
            failures.append((raw, actual))

    suspect = [
        ("EL CHICO DFBE SER CALVO", "the boy dfbe be bald", {"DFBE"}),
        ("EL CHICO DFBE SER CALVO", "the boy dfbee be bald", {"DFBE", "dfbee"}),
        ("LA CHICA DE PIE", "the CHICA standing", {"CHICA"}),
        ("El chico calvo", "El bald boy", {"El"}),
        ("LA CHICA DE PIE", "LA CHICA DE PIE", {"LA", "CHICA", "DE", "PIE"}),
    ]
    for raw, translated, expected in suspect:
        actual = set(translation_suspect_tokens(raw, translated))
        ok = expected.issubset(actual)
        print(f"{'PASS' if ok else 'FAIL'} residue {raw!r}: {sorted(actual)}")
        if not ok:
            failures.append((raw, expected, actual))
    assert not failures, failures
    print(f"OK: {len(clean) + len(suspect)} translation cognate/residue checks")

    # Without local lexical evidence, retain the conservative previous gate.
    # No downloads or translation weights may be needed to inspect tokens.
    with patch.object(prompt_translator, "_local_translation_target_tokenizer", return_value=None):
        assert set(translation_suspect_tokens(*clean[0])) == {"PENETRACION", "penetration"}
        assert "DFBE" in translation_suspect_tokens(*suspect[0][:2])
    assert prompt_translator._offline_translator is None
    print("OK: missing tokenizer stays conservative; no translation weights loaded")


if __name__ == "__main__":
    main()
