"""English instructions bypass Spanish mappings; Spanish keeps local routing.

Language detection, mappings and prompt cleanup are production code. Translation
I/O alone is mocked, so this probe never loads weights or accesses the network.
"""
import os
from unittest.mock import patch

from roop.img_editor import prompt_translator as translator


def main():
    prompts = (
        "The person wearing the purple tie must have bright blue hair. Leave the other person unchanged.",
        "The astronaut must keep the spacesuit unchanged.",
        "The quiet room has unusual lighting and uniform shadows.",
        "Show the underwater ruins under the moonlight.",
        "Return the uncut version and preserve the background.",
        "  Keep the bright green jacket.\nLeave all other details unchanged.  ",
    )
    with (
        patch.dict(os.environ, {"AUTOAUTO_OFFLINE": "1"}),
        patch.object(translator, "_offline_translate") as offline,
        patch.object(translator, "_api_translate") as api,
        patch.object(translator, "_local_translate", wraps=translator._local_translate) as mappings,
    ):
        for prompt in prompts:
            assert translator._is_english(prompt), prompt
            actual = translator.translate_prompt(prompt)
            expected = translator.clean_text(prompt)
            assert actual == expected, (prompt, actual, expected)
        mappings.assert_not_called()
        offline.assert_not_called()
        api.assert_not_called()
    print(f"PASS: {len(prompts)} English instructions preserve their words before mappings")

    spanish = (
        ("La mujer debe conservar su abrigo rojo", "The woman must keep her red coat"),
        ("el hombre con la purple tie debe tener blue hair", "The man with the purple tie must have blue hair"),
        ("El sofá es gris y la pared es azul", "The sofa is grey and the wall is blue"),
    )
    for prompt, translated in spanish:
        assert not translator._is_english(prompt), prompt
        with (
            patch.object(translator, "_offline_translate", return_value=translated) as offline,
            patch.object(translator, "_api_translate") as api,
            patch.object(translator, "_local_translate", wraps=translator._local_translate) as mappings,
        ):
            assert translator.translate_prompt(prompt) == translated
            mappings.assert_called_once_with(prompt)
            offline.assert_called_once_with(prompt)
            api.assert_not_called()
    print(f"PASS: {len(spanish)} Spanish/mixed instructions retain original local translation route")


if __name__ == "__main__":
    main()
