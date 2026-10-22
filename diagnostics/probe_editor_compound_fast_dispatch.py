"""Exercise ordinary LongCat dispatch without running an image generator.

Los motores legacy (longcat/longcat_full/imagine) se aliasan a kontext al
entrar a generate_intelligent, y la ruta Turbo (pases locales medidos) solo
sigue viva tras el fallback de qwen_camera cuando la petición no pide un
cambio de vista. Las secciones de despacho usan ese engine para ejercitar la
ruta viva con el mismo contrato de siempre.
"""
import re
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from roop.img_editor.flux_edit_comfy_client import (
    APPEARANCE_SAMPLE_DENOISE,
    APPEARANCE_SAMPLE_STEPS,
    REGION_SAMPLE_STEPS,
    SURFACE_CONTINUATION_DENOISE,
    longcat_turbo_denoise,
    measured_appearance_sample,
)
from roop.img_editor.img_editor_manager import ImgEditorManager


class FakeClient:
    def __init__(self):
        self.calls = []

    def load(self, version=None, **params):
        # The manager loads by keyword (flux_version=… / qwen_version=…).
        self.calls.append(("load", version or params.get("flux_version")
                           or params.get("qwen_version")))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (32, 20), "red")), "OK (stub)"


class RetryClient(FakeClient):
    def __init__(self, always_white=False):
        super().__init__()
        self.always_white = always_white

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        calls = sum(name == "generate" for name, _ in self.calls)
        color = "white" if calls == 1 or self.always_white else "red"
        return SimpleNamespace(image=Image.new("RGB", (48, 32), color)), f"OK attempt {calls}"


def _check_subject_growth(manager):
    import time

    import numpy as np

    width, height = 2847, 4263
    region = np.zeros((height, width), np.uint8)
    region[900:2800, 700:1400] = 255
    left = SimpleNamespace(bbox=(900, 1000, 1300, 1500))
    right = SimpleNamespace(bbox=(2000, 1000, 2400, 1500))
    started = time.perf_counter()
    grown = ImgEditorManager._grow_measured_subject(region, [left, right], width, height)
    elapsed = time.perf_counter() - started
    assert grown is not None and elapsed < 2.0, elapsed
    assert int(grown[1200, 1000]) > 96
    assert int(grown[20, 1000]) == 0
    assert int(grown[4200, 1000]) == 0
    assert int(grown[1200:1400, 2100:2300].max()) == 0
    assert ImgEditorManager._grow_measured_subject(region, [right], width, height) is None
    queries = manager._subject_queries(
        "el hombre debe ser calvo, el hombre debe estar desnudo",
        "Man must be bald, man must be naked",
    )
    assert queries == ["Man", "el hombre"], queries
    man = np.zeros((80, 120), np.uint8)
    man[10:70, 10:50] = 255
    woman = np.zeros_like(man)
    woman[10:70, 70:110] = 255
    groups = manager._group_measured_subjects(
        [
            {"query": "the man", "region": man},
            {"query": "el hombre", "region": man.copy()},
            {"query": "the woman", "region": woman},
            {"query": "la mujer", "region": woman.copy()},
        ],
        "el hombre debe ir desnudo, el hombre es calvo, la mujer va descalza y sin medias",
        "the man must go naked, the man is bald, the woman goes barefoot and without stockings",
    )
    texts = [group["text"] for group in groups]
    assert len(groups) == 2, texts
    assert any("naked" in text and "bald" in text and "barefoot" not in text for text in texts)
    assert any("barefoot" in text and "naked" not in text for text in texts)
    print(f"OK: máscara de sujeto en {width}x{height} crece en {elapsed:.3f}s")


def main():
    cases = (
        (
            "la mujer va descalza y sin medias, el hombre blanco que está de rodillas es calvo",
            "the woman goes barefoot and without stockings, "
            "the white man who is on his knees is bald",
        ),
        # Postura medida sin keyword (2026-10-16): sin el concepto medido en
        # facts esta cláusula caía a la ruta anclada de cuerpo y el gato no
        # bailaba; con postura el pase compuesto es sin ancla y "postura nueva".
        (
            "el gato esta bailando, vestido de lo que sea",
            "the cat is dancing, dressed in whatever",
        ),
        (
            "la mujer debe ir sin medias, descalza, vestida solo con un tanga",
            "the woman must go without stockings, barefoot, dressed only in a thong",
        ),
        ("Add a person near the doorway", "Add a person near the doorway"),
        ("Remove the person on the left", "Remove the person on the left"),
        ("Add a lamp to the table", "Add a lamp to the table"),
        ("Remove the chair", "Remove the chair"),
        (
            "el hombre de barba esta de pie y su pene es grande",
            "The bearded man is standing and his penis is big",
        ),
        (
            "EL HOMBRE ES CALVO, EL HOMBRE VA DESNUDO",
            "The man is bald, the man is naked",
        ),
        ("el hombre debe ser calvo", "Man must be bald"),
        # The conjunction strip leaves "go naked" without the person; the
        # global pass must still name the person the user asked about.
        ("EL HOMBRE DEBE SER CALVO E IR DESNUDO", "Man must be bald and go naked"),
        # Sufijo corporal con sujeto equivocado (log real 19:49): el sufijo
        # de desnudez nombraba al SOFÁ (primer sujeto del compuesto) en vez
        # de la mujer que pedía la desnudez → sofá sin pintar y zona dañada.
        (
            "el sofa debe ser gris, las paredes deben ser grises, la mujer debe ir desnuda",
            "the sofa must be grey, the walls must be grey, the woman must go naked",
        ),
    )
    source = Image.new("RGB", (48, 32), "white")
    manager = ImgEditorManager()

    class FailingTurboClient(FakeClient):
        def __init__(self, fail_on):
            super().__init__()
            self.fail_on = fail_on

        def load(self, version=None, **params):
            self.calls.append(("load", version or params.get("flux_version")
                               or params.get("qwen_version")))
            if self.fail_on == "load":
                return False, "Turbo load error"
            return True, "OK"

        def generate(self, **kwargs):
            self.calls.append(("generate", kwargs))
            return None, "HTTP 500 uploading Turbo image"

    for failure in ("load", "generate"):
        failing = FailingTurboClient(failure)
        with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
              patch("roop.face_util.get_face_analyser", return_value=None),
              patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
              patch.object(manager, "_release_local_gpu_before_comfy"),
              patch.object(manager, "_select_longcat_version") as legacy_selector,
              patch("roop.img_editor.prompt_translator.translate_prompt",
                    return_value="Change the lighting in the room"),
              patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                    return_value=failing)):
            failed_output, failed_message, failed_mask = manager.generate_intelligent(
                image=source, prompt="Change the lighting in the room", engine="qwen_camera",
            )
        assert failed_output is None and failed_mask is None, failed_message
        assert "Turbo no pudo aplicar la edición" in failed_message, failed_message
        assert ("Turbo load error" if failure == "load" else "HTTP 500") in failed_message
        assert [name for name, _ in failing.calls] == (
            ["load"] if failure == "load" else ["load", "generate"]
        ), failing.calls
        assert failing.calls[0][1] == "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
        legacy_selector.assert_not_called()
    print("OK: error de carga o subida Turbo se devuelve sin iniciar Full")

    _check_subject_growth(manager)

    # Postura sin keyword: el hecho de postura sale del concepto MEDIDO
    # (CONCEPTS["pose"] describe "dance/bailando"), no solo de la lista de
    # palabras del analizador. Medido 2026-10-16: "el gato esta bailando"
    # ya medía pose 0,289 (0,79 con el alcance "bailando") pero facts
    # seguía en falso → la ruta anclada "exact same framing" sujetaba al
    # gato en el sitio y solo se le aplicaba la ropa.
    dance_facts = manager._prompt_authority_facts(
        "el gato esta bailando", "the cat is dancing",
    )
    assert dance_facts["pose"] is True, dance_facts
    en_dance = manager._prompt_authority_facts(
        "the cat is dancing", "the cat is dancing",
    )
    assert en_dance["pose"] is True, en_dance
    dressed_only = manager._prompt_authority_facts(
        "vestido de lo que sea", "dressed in whatever",
    )
    assert dressed_only["pose"] is False, dressed_only
    print("OK: postura medida enciende el hecho sin keyword de lista")

    assert manager._edit_is_nearly_unchanged(source, source.copy())
    changed_small = source.copy()
    changed_small.paste("blue", (8, 8, 22, 22))
    assert not manager._edit_is_nearly_unchanged(source, changed_small)
    assert manager._explicit_viewpoint_request("cambia el pov", "change the viewpoint")
    assert manager._explicit_viewpoint_request("cambia el ángulo a vista cenital", "show a top view")
    assert manager._explicit_viewpoint_request("cambia la vista de la habitación", "change the room view")
    assert manager._explicit_viewpoint_request("Show the room from the opposite angle", "Show the room from the opposite angle")
    assert not manager._explicit_viewpoint_request(
        "el hombre de perfil debe llevar sombrero", "the man in profile must wear a hat"
    )
    assert not manager._explicit_viewpoint_request(
        "cambia el pelo de la mujer de perfil", "change the hair of the woman in profile"
    )
    assert not manager._explicit_viewpoint_request(
        "añade una ventana con vista al mar", "add a window with a sea view"
    )
    assert not manager._explicit_viewpoint_request("Pon la pared azul", "Make the wall blue")
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=None),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy")):
        for prompt, translated in cases:
            client = FakeClient()
            with (patch("roop.img_editor.prompt_translator.translate_prompt", return_value=translated),
                  patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
                  patch.object(manager, "_get_clipseg_masker", return_value=None)):
                output, message, mask = manager.generate_intelligent(
                    image=source, prompt=prompt, engine="qwen_camera", use_rewriter=True,
                    use_semantic=True,
                )
            assert output is not None and output.size == source.size and mask is None
            facts = manager._prompt_authority_facts(prompt, translated)
            if manager._split_compound_instruction(translated):
                clauses = manager._split_compound_instruction(translated)
                generations = [args for name, args in client.calls if name == "generate"]
                assert len(generations) == 1, (len(generations), clauses)
                prompts = [item["prompt"] for item in generations]
                original_clauses = manager._split_compound_instruction(prompt)
                if len(original_clauses) != len(clauses):
                    original_clauses = [""] * len(clauses)
                # One instruction carries every clause: each one must appear
                # in the single composed prompt.
                for clause, original in zip(clauses, original_clauses):
                    present = any(clause.lower() in item.lower() for item in prompts)
                    assert present, (clause, prompts)
                assert "un solo pase Turbo" in message, (prompt, message)
                assert "cada cambio por separado" not in message, (prompt, message)
                assert generations[0]["mask_image"] is None, (prompt, message)
                assert generations[0]["keep_source_pixels"] is False
                if "naked" not in translated.lower() and "nude" not in translated.lower():
                    assert all("fully nude" not in item.lower() for item in prompts), prompts
                else:
                    assert any("fully nude" in item.lower() for item in prompts), prompts
                    # El sufijo de desnudez nombra al sujeto de LA CLÁUSULA
                    # CORPORAL (log real 19:49: «The sofa is fully nude» cuando
                    # la desnudez la pedía la mujer → sofá intacto, zona
                    # dañada). La cláusula sin sujeto no exige nombre aquí
                    # («calvo e ir desnudo» se cubre con el sujeto global).
                    joined = " ".join(prompts).lower()
                    assert "sofa is fully nude" not in joined, joined
                    for clause in clauses:
                        clause_f = manager._prompt_authority_facts(clause, clause)
                        if (clause_f.get("body")
                                and clause_f.get("body_direction") == "uncover"):
                            nude_subject = (
                                manager._visual_subject_description(clause) or ""
                            ).strip(" ,.;")
                            if nude_subject:
                                assert f"{nude_subject} is fully nude" in joined, (
                                    nude_subject, joined,
                                )
                # A subject-less body clause must reach the model with the
                # person named, or the global pass edits whoever it picks.
                stripped_body = [
                    clause for clause in clauses
                    if not manager._visual_subject_description(clause)
                    and manager._prompt_authority_facts(clause, clause).get("body")
                ]
                if stripped_body:
                    subject = manager._visual_subject_description(translated) or ""
                    assert subject, (stripped_body, translated)
                    assert any(subject in item for item in prompts), (subject, prompts)
                    print(f"OK: {prompt} -> sujeto propagado a {stripped_body}")
                # The compound pass keeps the photo anchored the same way the
                # single-instruction path does: no_kontext=False adds the
                # index_timestep_zero reference so the room survives. A
                # no_kontext=True pass regenerated the scene from noise and
                # left a white blob (edit_1791843357). The accepted
                # exception is a compound whose clause changes the POSE:
                # geometry has to come from the model, under the same
                # contract as the single-clause pose path ("sin ancla de
                # latente", preserve_frame=False), and its measured zones
                # are not pasted over the new geometry.
                clause_flags = [
                    manager._prompt_authority_facts(clause, clause)
                    for clause in clauses
                ]
                if any(flags.get("pose") for flags in clause_flags):
                    assert all(item["no_kontext"] is True for item in generations), (prompt, generations)
                    assert all(item["preserve_frame"] is False for item in generations), (prompt, generations)
                    assert "postura nueva" in message, (prompt, message)
                else:
                    assert all(item["no_kontext"] is False for item in generations), (prompt, generations)
                print(f"OK: {prompt} -> un solo pase Turbo anclado con las cláusulas juntas")
                continue
            if facts.get("local_appearance") or facts.get("local_region") or facts.get("region_ambiguous"):
                assert output is not None and list(output.getdata()) == list(source.getdata())
                assert "no se sustituye" in message, (prompt, message)
                assert not any(name == "generate" for name, _ in client.calls), client.calls
                print(f"OK: {prompt} -> {message}")
                continue
            assert [name for name, _ in client.calls] == ["load", "generate"]
            generation = client.calls[-1][1]
            authority = facts["authority"]
            if authority:
                if facts["pose"]:
                    assert "sin ancla de latente" in message, (prompt, message)
                    assert generation["preserve_frame"] is False
                else:
                    assert "edición corporal global" in message, (prompt, message)
                    assert generation["preserve_frame"] is True
                assert generation["mask_image"] is None
            else:
                assert "cumplimiento visual no verificado" in message, (prompt, message)
            if manager._explicit_viewpoint_request(prompt, translated):
                assert "POV no verificado" in message, (prompt, message)
            assert translated in generation["prompt"]
            if authority:
                if facts["pose"]:
                    assert "pose, framing and appearance may change" in generation["prompt"]
                else:
                    assert "exact same framing" in generation["prompt"]
                if (facts["body"] and facts["body_direction"] == "uncover"
                        and facts["body_intensity"] >= 0.30):
                    assert "fully nude" in generation["prompt"]
                assert generation.get("keep_source_pixels") is not True
                assert generation["prompt_authority"] is True
            elif not facts.get("local_appearance") and not facts.get("local_region"):
                assert "Keep people anatomically coherent" in generation["prompt"]
                assert generation.get("prompt_authority") is not True
                assert generation["preserve_frame"] is False
                assert generation["mask_image"] is None
            assert generation["num_inference_steps"] == 12
            assert generation["no_kontext"] is True
            print(f"OK: {prompt} -> {message}")

        client = FakeClient()
        finish_calls = []
        def fake_finish(image, original, **kwargs):
            finish_calls.append((image.size, original.size, kwargs))
            return Image.new("RGB", (64, 42), "red"), "acabado simulado", {"scores": {}}
        with (patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Make the wall blue"),
              patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=client),
              patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit", side_effect=fake_finish)):
            output, message, mask = manager.generate_intelligent(
                image=source, prompt="Pon la pared azul", engine="qwen_camera",
                quality_mode=True, enhance_tier="4k", quality_preserve_faces=False,
            )
        assert output is not None and output.size == (64, 42) and mask is None
        assert [name for name, _ in client.calls] == ["load", "generate"]
        assert "Keep every person, object, room detail" in client.calls[-1][1]["prompt"]
        assert "Keep people anatomically coherent" in client.calls[-1][1]["prompt"]
        assert finish_calls[0][0:2] == ((32, 20), source.size)
        assert finish_calls[0][2]["enhance_tier"] == "4k"
        assert finish_calls[0][2]["preserve_faces"] is False
        assert finish_calls[0][2]["use_generative"] is False
        assert "Turbo" in message
        print("OK: Calidad con prompt -> Turbo único + acabado clásico, sin restauración facial")

        retry_client = RetryClient()
        import contextlib
        import io
        retry_log = io.StringIO()
        with (patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Make the wall blue"),
              patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=retry_client),
              patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit", side_effect=fake_finish),
              contextlib.redirect_stdout(retry_log)):
            retried, retry_message, _ = manager.generate_intelligent(
                image=source, prompt="Pon la pared azul", engine="qwen_camera",
                quality_mode=True, quality_preserve_faces=False,
            )
        assert retried is not None
        attempts = [call for name, call in retry_client.calls if name == "generate"]
        assert len(attempts) == 2
        assert "not yet been changed" in attempts[1]["prompt"]
        assert "segundo intento" in retry_message
        # La consola debe anunciar el segundo render ANTES de generarlo: la
        # noche del 2026-10-16 el usuario vio dos renders seguidos sin línea
        # alguna que explicara el segundo (log 23:55, run 15).
        console = retry_log.getvalue()
        assert "Cambio casi nulo sobre la foto; segundo intento" in console, console
        assert "Segundo intento aplicado:" in console, console
        print("OK: Calidad reintenta una salida idéntica con la petición visible")

        unchanged_client = RetryClient(always_white=True)
        with (patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Make the wall blue"),
              patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=unchanged_client),
              patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit", side_effect=fake_finish)):
            unchanged, warning, _ = manager.generate_intelligent(
                image=source, prompt="Pon la pared azul", engine="qwen_camera",
                quality_mode=True, quality_preserve_faces=False,
            )
        assert unchanged is not None
        assert "cambio solicitado apenas visible" in warning
        assert manager._last_instruction_compliance["compliant"] is False
        print("OK: Calidad avisa si ambos intentos son casi idénticos")

    import numpy as np

    bald = manager._prompt_authority_facts("el hombre debe ser calvo", "Man must be bald")
    assert bald["local_appearance"] and not bald["body"] and not bald["pose"], bald
    wall = manager._prompt_authority_facts("Make the wall blue", "Make the wall blue")
    assert not wall["local_appearance"], wall
    nude = manager._prompt_authority_facts(
        "EL HOMBRE ES CALVO, EL HOMBRE VA DESNUDO",
        "The man is bald, the man is naked",
    )
    assert nude["body"] and not nude["local_appearance"], nude
    barefoot = manager._prompt_authority_facts(
        "la mujer debe ir descalza", "the woman must go barefoot"
    )
    assert barefoot["body"] and barefoot["body_intensity"] >= 0.30, barefoot
    barefoot_prompt = manager._compose_generation_prompt(
        "the woman must go barefoot", engine="longcat", magnitude=0.85,
        body_transform=True, body_direction=barefoot["body_direction"],
        body_intensity=barefoot["body_intensity"],
    )
    assert "fully nude" not in barefoot_prompt.lower(), barefoot_prompt
    naked_prompt = manager._compose_generation_prompt(
        "the man must go naked", engine="longcat", magnitude=0.85,
        body_transform=True, body_direction="uncover", body_intensity=0.70,
    )
    assert "fully nude" in naked_prompt.lower(), naked_prompt
    # v5.1217 — typo en la grafía del usuario (log real 23:40 «ir ddesnuda»):
    # la cláusula cruda midió direction="change" con intensity=0.70 y ganaba
    # por intensidad, así que el sufijo de desnudez se caía aunque la
    # traducción de la MISMA petición ya midiera uncover. La dirección de la
    # petición no depende de una sola grafía.
    typo = manager._prompt_authority_facts(
        "la mujer de en medio debe ir ddesnuda",
        "The woman in the middle must go naked",
    )
    assert typo["body"] and typo["body_direction"] == "uncover", typo
    assert typo["body_intensity"] >= 0.60, typo
    typo_prompt = manager._compose_generation_prompt(
        "The woman in the middle must go naked", engine="longcat", magnitude=0.85,
        body_transform=True, body_direction=typo["body_direction"],
        body_intensity=typo["body_intensity"],
    )
    assert "fully nude" in typo_prompt.lower(), typo_prompt
    print("OK: typo «ddesnuda» conserva la dirección uncover y el sufijo desnudez")


    class _Face:
        def __init__(self, bbox):
            self.bbox = bbox

    class _Masker:
        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").lower()
            if text == "hair":
                arr[30:52, 145:185] = 255
                arr[78:100, 60:95] = 255
                arr[8:28, 8:40] = 255
            elif re.search(r"\bman\b|\bhombre\b", text):
                arr[40:110, 40:200] = 255
            return Image.fromarray(arr, "L")

    photo = Image.new("RGB", (240, 140), "white")
    man = _Face((150, 48, 176, 74))
    woman = _Face((16, 18, 36, 42))
    restored_client = FakeClient()
    translated_compound = "the woman must go barefoot, the man must be bald"
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[man, woman]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_Masker()),
          patch("roop.img_editor.prompt_translator.translate_prompt", return_value=translated_compound),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=restored_client)):
        restored_output, restored_message, _ = manager.generate_intelligent(
            image=photo,
            prompt="la mujer debe ir descalza, el hombre debe ser calvo",
            engine="qwen_camera",
        )
    assert restored_output is not None and restored_output.size == photo.size
    restored_generations = [args for name, args in restored_client.calls if name == "generate"]
    assert len(restored_generations) == 1, restored_client.calls
    sole_restored = restored_generations[0]
    sole_restored_prompt = sole_restored["prompt"].lower()
    assert "barefoot" in sole_restored_prompt and "bald" in sole_restored_prompt, sole_restored_prompt
    # The measured zone is the sampler hole and the paste, and the photo
    # stays anchored (no_kontext=False -> index_timestep_zero reference).
    assert sole_restored["mask_image"] is not None
    assert sole_restored["keep_source_pixels"] is True
    assert sole_restored["denoise"] == 1.0
    assert sole_restored["num_inference_steps"] == 12
    assert sole_restored["no_kontext"] is False
    assert "un solo pase Turbo" in restored_message, restored_message
    assert "cada cambio por separado" not in restored_message, restored_message
    # "bald" lights nothing on this masker; the man measured is the only
    # paste. The woman is never measured, so her zone keeps the source.
    restored_src = np.asarray(photo.convert("RGB"))
    restored_out = np.asarray(restored_output.convert("RGB"))
    assert (restored_out[5, 5] == restored_src[5, 5]).all(), restored_out[5, 5]
    restored_face = restored_out[61, 163]
    assert int(restored_face[0]) > 180, restored_face
    print("OK: descalza y calvo comparten un pase; solo la persona medida se pega")

    # 2026-10-08 night: sending the measured person as the sampler hole
    # flattened skin and left the request undone. With no distinct part,
    # Turbo sees the photo and the instruction. The person is only the paste.
    body_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[man, woman]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_Masker()),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="the man must not wear stockings"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=body_client)):
        body_output, body_message, _ = manager.generate_intelligent(
            image=photo, prompt="el hombre debe ir sin medias", engine="qwen_camera",
        )
    assert body_output is not None and body_output.size == photo.size, body_message
    body_generations = [args for name, args in body_client.calls if name == "generate"]
    assert len(body_generations) == 1, body_client.calls
    body_call = body_generations[0]
    # No distinct body part is measurable, so the measured person is the
    # zone: a local anchored pass, not the whole photo regen.
    assert body_call["mask_image"] is not None, body_message
    assert body_call["keep_source_pixels"] is True, body_message
    assert body_call["prompt_authority"] is True and body_call["preserve_frame"] is True
    assert body_call["denoise"] == 1.0, body_call["denoise"]
    assert body_call["num_inference_steps"] == 12
    assert body_call["no_kontext"] is False
    assert "cuerpo medido" in body_message, body_message
    assert "edición corporal global" not in body_message, body_message
    assert "fully nude" not in body_call["prompt"].lower(), body_call["prompt"]
    out_arr = np.asarray(body_output.convert("RGB"))
    src_arr = np.asarray(photo.convert("RGB"))
    assert (out_arr[5, 5] == src_arr[5, 5]).all()
    face = out_arr[61, 163]
    assert int(face[0]) > 200 and int(face[1]) < 40 and int(face[2]) < 40, face
    torso = out_arr[90, 80]
    assert int(torso[0]) > 200 and int(torso[1]) < 40 and int(torso[2]) < 40, torso
    print("OK: sin parte distinta la persona medida es la zona; la foto queda anclada")
    print("OK: sin zona de predicado se pega la persona medida en pase local")

    # 2026-10-08 log: "The woman must go without stockings" alone lost body
    # when the rescore kept only the English sentence. The Spanish clause
    # stays in the score, so the measured woman is still the paste.
    class _WomanMasker:
        def __init__(self):
            self.queries = []

        def generate_mask(self, image, prompt, threshold=0.4):
            self.queries.append(prompt)
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").lower()
            part = bool(re.search(r"\b(?:hair|pelo|cabello)\b", text)) and not re.search(
                r"\b(?:woman|mujer|man|hombre)\b", text,
            )
            if part:
                arr[14:32, 12:38] = 255
            elif re.search(r"\b(?:woman|mujer)\b", text):
                arr[10:130, 4:90] = 255
            return Image.fromarray(arr, "L")

    woman_masker = _WomanMasker()
    woman_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[man, woman]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=woman_masker),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="The woman must go without stockings"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=woman_client)):
        woman_output, woman_message, _ = manager.generate_intelligent(
            image=photo, prompt="la mujer debe ir sin medias", engine="qwen_camera",
        )
    assert woman_output is not None and woman_output.size == photo.size, woman_message
    woman_generations = [args for name, args in woman_client.calls if name == "generate"]
    assert len(woman_generations) == 1, woman_client.calls
    woman_call = woman_generations[0]
    # No distinct part is measurable, so the measured woman is the zone:
    # a local anchored pass, not the whole photo regen.
    assert woman_call["mask_image"] is not None, woman_message
    assert woman_call["keep_source_pixels"] is True, woman_message
    assert woman_call["num_inference_steps"] == 12
    assert woman_call["no_kontext"] is False
    assert "cuerpo medido" in woman_message, woman_message
    assert "edición corporal global" not in woman_message, woman_message
    woman_out = np.asarray(woman_output.convert("RGB"))
    woman_src = np.asarray(photo.convert("RGB"))
    assert (woman_out[5, 200] == woman_src[5, 200]).all()
    assert int(woman_out[40, 40][0]) > 200
    print("OK: sin medias en inglés pega a la mujer medida en pase local anclado")

    # 2026-10-08 saved photo: the body clause measured a part (10.3%) but
    # Turbo was given the whole person at denoise 1. The sampler redrew the
    # face and the paste kept 7% of the crop, with none of the lower third.
    # When the segmenter finds that part, the part is the mask that is sent
    # and the part that is written back. The rest of the person stays.
    class _PartMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").lower()
            if re.search(r"\b(?:woman|mujer|man|hombre)\b", text):
                arr[10:130, 4:90] = 255
            elif text.strip():
                arr[100:128, 8:80] = 255
            return Image.fromarray(arr, "L")

    part_masker = _PartMasker()
    part_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[man, woman]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=part_masker),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="The woman must go without stockings"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=part_client)):
        part_output, part_message, _ = manager.generate_intelligent(
            image=photo, prompt="la mujer debe ir sin medias", engine="qwen_camera",
        )
    assert part_output is not None and part_output.size == photo.size, part_message
    part_generations = [args for name, args in part_client.calls if name == "generate"]
    assert part_generations, part_client.calls
    part_call = part_generations[0]
    assert part_call["mask_image"] is not None, part_message
    assert part_call["keep_source_pixels"] is True, part_message
    assert part_call["denoise"] == 1.0 and part_call["num_inference_steps"] == 12
    assert "cuerpo medido" in part_message, part_message
    assert "edición corporal global" not in part_message, part_message
    person_px = int((np.asarray(part_masker.generate_mask(photo, "la mujer")) >= 160).sum())
    sent_px = int((np.asarray(part_call["mask_image"].convert("L")) >= 160).sum())
    assert 0 < sent_px < person_px * 0.55, (sent_px, person_px, part_call["image"].size)
    assert part_call["image"].size[0] * part_call["image"].size[1] < photo.size[0] * photo.size[1]
    src_px = np.asarray(photo.convert("RGB"))
    out_px = np.asarray(part_output.convert("RGB"))
    assert (out_px[30, 26] == src_px[30, 26]).all(), out_px[30, 26]
    assert (out_px[5, 230] == src_px[5, 230]).all(), out_px[5, 230]
    leg = out_px[114, 44]
    assert int(leg[0]) > 200 and int(leg[1]) < 40 and int(leg[2]) < 40, leg
    print("OK: la zona pedida del cuerpo se genera y se pega; la cara no entra")

    # 2026-10-08 20:57: "sin zapatos" lit a 0.53% blob that was 62% skin.
    # That speck is not a part. The measured person is the edit. A dark
    # region on the same person still is a part.
    skin_arr = np.full((160, 220, 3), (30, 40, 70), np.uint8)
    skin_arr[20:150, 40:120] = (190, 140, 110)
    skin_arr[110:145, 48:92] = (22, 22, 24)
    skin_photo = Image.fromarray(skin_arr, "RGB")
    skin_face = _Face((55, 28, 100, 72))

    class _SkinSpeckleMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").lower()
            if re.search(r"\b(?:woman|mujer)\b", text):
                arr[20:150, 40:120] = 255
            elif "zapato" in text or "shoe" in text:
                arr[48:68, 70:90] = 255
            return Image.fromarray(arr, "L")

    class _DarkPartMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").lower()
            if "zapato" in text or "shoe" in text:
                arr[110:145, 48:92] = 255
            return Image.fromarray(arr, "L")

    woman_subject = _SkinSpeckleMasker().generate_mask(skin_photo, "mujer")
    with patch.object(manager, "_get_clipseg_masker", return_value=_DarkPartMasker()):
        dark_part = manager._request_region_on_subject(
            skin_photo, woman_subject,
            ["la mujer debe ir sin zapatos", "the woman must go without shoes"],
        )
    assert dark_part is not None
    dark_px = np.asarray(dark_part.convert("L"))
    assert int(dark_px[120, 60]) >= 160
    assert int(dark_px[40, 70]) == 0
    print("OK: una zona oscura dentro de la persona sigue siendo la parte")

    skin_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[skin_face]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_SkinSpeckleMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="the woman must go without shoes"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=skin_client)):
        skin_output, skin_message, _ = manager.generate_intelligent(
            image=skin_photo,
            prompt="la mujer debe ir sin zapatos",
            engine="qwen_camera",
        )
    assert skin_output is not None and skin_output.size == skin_photo.size, skin_message
    skin_calls = [args for name, args in skin_client.calls if name == "generate"]
    assert skin_calls, skin_message
    # The skin speck is not the part and the shoes are not measurable, so
    # the measured woman is the zone: a local anchored pass.
    assert skin_calls[0]["mask_image"] is not None, skin_message
    assert skin_calls[0]["keep_source_pixels"] is True, skin_message
    assert skin_calls[0]["num_inference_steps"] == 12
    assert skin_calls[0]["no_kontext"] is False
    skin_src = np.asarray(skin_photo.convert("RGB"))
    skin_out = np.asarray(skin_output.convert("RGB"))
    assert (skin_out[5, 5] == skin_src[5, 5]).all()
    assert not (skin_out[40, 60] == skin_src[40, 60]).all()
    assert not (skin_out[130, 60] == skin_src[130, 60]).all()
    print("OK: una mancha de piel no es la parte; se pega la persona medida en pase local")

    # The shoe sits mostly outside the body silhouette and only touches the
    # leg. Clipping to the silhouette kept the skin. The object that touches
    # the person is the part, including the pixels outside the silhouette.
    shoe_arr = np.full((160, 220, 3), (30, 40, 70), np.uint8)
    shoe_arr[40:140, 80:150] = (190, 140, 110)
    shoe_arr[100:150, 40:100] = (24, 22, 20)
    shoe_photo = Image.fromarray(shoe_arr, "RGB")
    shoe_face = _Face((100, 48, 140, 92))

    class _OutsideShoeMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").lower()
            if re.search(r"\b(?:woman|mujer)\b", text):
                arr[40:140, 80:150] = 255
            elif "zapato" in text or "shoe" in text:
                arr[100:150, 40:100] = 255
            return Image.fromarray(arr, "L")

    shoe_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[shoe_face]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_OutsideShoeMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="the woman must go without shoes"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=shoe_client)):
        shoe_output, shoe_message, _ = manager.generate_intelligent(
            image=shoe_photo,
            prompt="la mujer debe ir sin zapatos",
            engine="qwen_camera",
        )
    assert shoe_output is not None and shoe_output.size == shoe_photo.size, shoe_message
    shoe_src = np.asarray(shoe_photo.convert("RGB"))
    shoe_out = np.asarray(shoe_output.convert("RGB"))
    assert (shoe_out[8, 8] == shoe_src[8, 8]).all()
    assert (shoe_out[50, 110] == shoe_src[50, 110]).all(), shoe_out[50, 110]
    assert not (shoe_out[120, 50] == shoe_src[120, 50]).all(), shoe_out[120, 50]
    print("OK: el objeto que toca a la persona entra entero, también fuera de la silueta")

    # The segmenter lit only the top of a long surface. The rest is the same
    # color, still inside the person. Hair is not added. A continuation that
    # would cover the person is rejected and the measured part stays.
    stocking_photo = np.full((180, 200, 3), (230, 228, 224), np.uint8)
    stocking_photo[8:168, 40:120] = (200, 170, 150)
    stocking_photo[110:160, 52:108] = (35, 32, 48)
    stocking_image = Image.fromarray(stocking_photo, "RGB")
    stocking_face = _Face((55, 12, 105, 58))

    class _StockingMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            text = (prompt or "").lower()
            if text.strip() in {"chair", "plant"}:
                return None
            arr = np.zeros((image.height, image.width), np.uint8)
            if re.search(r"\b(?:woman|mujer|man|hombre)\b", text):
                arr[8:168, 40:120] = 255
            elif text.strip():
                arr[110:124, 56:104] = 255
            return Image.fromarray(arr, "L")

    stocking_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[stocking_face]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_StockingMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="The woman must go without stockings"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=stocking_client)):
        stocking_output, stocking_message, _ = manager.generate_intelligent(
            image=stocking_image,
            prompt="la mujer debe ir sin medias",
            engine="qwen_camera",
        )
    assert stocking_output is not None and stocking_output.size == stocking_image.size, stocking_message
    stocking_generations = [args for name, args in stocking_client.calls if name == "generate"]
    assert stocking_generations, stocking_client.calls
    stocking_call = stocking_generations[0]
    assert stocking_call["mask_image"] is not None
    assert stocking_call.get("reference_image") is None
    # "ancla para cuerpo medido" (cambio local pendiente): el pase local de
    # cuerpo va anclado (no_kontext=False -> nodo 11 index_timestep_zero);
    # solo la foto entera se genera sin ancla. Antes de este cambio el pase
    # local de cuerpo usaba no_kontext=True (contrato obsoleto).
    assert stocking_call["no_kontext"] is False
    assert stocking_call["denoise"] == 1.0 and stocking_call["num_inference_steps"] == 12
    stocking_person = int((np.asarray(
        _StockingMasker().generate_mask(stocking_image, "la mujer")
    ) >= 160).sum())
    stocking_sent = int((np.asarray(stocking_call["mask_image"].convert("L")) >= 160).sum())
    assert 1500 < stocking_sent < stocking_person * 0.55, (stocking_sent, stocking_person)
    stocking_src = np.asarray(stocking_image)
    stocking_out = np.asarray(stocking_output)
    assert (stocking_out[30, 80] == stocking_src[30, 80]).all(), stocking_out[30, 80]
    assert (stocking_out[4, 10] == stocking_src[4, 10]).all(), stocking_out[4, 10]
    lower = stocking_out[145, 80]
    assert int(lower[0]) > 200 and int(lower[1]) < 40 and int(lower[2]) < 40, lower
    print("OK: la superficie del cuerpo sigue el color; la cara y el fondo quedan")

    # Same log: "sin medias, tiene el pelo negro" applied only the first
    # clause. "tiene"/"has" were not predicates, so the hair clause was
    # asked to CLIPSeg as a whole sentence and the photo was not replaced.
    chain_masker = _WomanMasker()
    chain_client = FakeClient()
    chain_raw = "la mujer debe ir sin medias, la mujer tiene el pelo negro"
    chain_en = "The woman must go without stockings, the woman has the black hair"
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[man, woman]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=chain_masker),
          patch("roop.img_editor.prompt_translator.translate_prompt", return_value=chain_en),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=chain_client)):
        chain_output, chain_message, _ = manager.generate_intelligent(
            image=photo, prompt=chain_raw, engine="qwen_camera",
        )
    assert chain_output is not None, chain_message
    assert "no se sustituye" not in chain_message, chain_message
    assert "un solo pase Turbo" in chain_message, chain_message
    assert "cada cambio por separado" not in chain_message, chain_message
    chain_generations = [args for name, args in chain_client.calls if name == "generate"]
    # Stockings and hair used to be two chained edits; both clauses now
    # share one Turbo pass of the full instruction, and the woman measured
    # for either clause is the only zone pasted.
    assert len(chain_generations) == 1, chain_generations
    sole_chain = chain_generations[0]
    sole_chain_prompt = sole_chain["prompt"].lower()
    assert "stocking" in sole_chain_prompt, sole_chain_prompt
    assert "hair" in sole_chain_prompt or "pelo" in sole_chain_prompt, sole_chain_prompt
    assert sole_chain["mask_image"] is not None, chain_message
    assert sole_chain["keep_source_pixels"] is True, chain_message
    assert sole_chain["num_inference_steps"] == 12
    assert sole_chain["no_kontext"] is False
    chain_src = np.asarray(photo.convert("RGB"))
    chain_out = np.asarray(chain_output.convert("RGB"))
    woman_zone = np.asarray(
        _WomanMasker().generate_mask(photo, "la mujer").convert("L")
    ) >= 96
    assert (chain_out[5, 200] == chain_src[5, 200]).all(), chain_out[5, 200]
    assert woman_zone[70, 50], "la mujer mide su cuerpo entero"
    chain_inside = chain_out[70, 50]
    assert int(chain_inside[0]) > 180, chain_inside
    # Zone measurement never asks the segmenter the full predicate
    # sentence; a recolor may still ask for the requested part alone.
    assert not any(
        query.casefold() == "the woman has the black hair" for query in chain_masker.queries
    ), chain_masker.queries
    print("OK: medias y pelo negro comparten un pase; la mujer medida es la única zona pegada")
    compound_client = FakeClient()
    woman_mask = Image.new("L", photo.size, 0)
    woman_mask.paste(255, (8, 8, 115, 135))
    man_mask = Image.new("L", photo.size, 0)
    man_mask.paste(255, (120, 12, 220, 130))
    with patch.object(manager, "_get_clipseg_masker", return_value=_Masker()):
        manager._apply_subject_passes(
            compound_client, photo,
            [
                {"text": "the woman must go barefoot", "mask": woman_mask},
                {"text": "the man must be bald", "mask": man_mask},
            ],
            seed=1, negative_prompt="", lora_name="", lora_strength=0,
            progress_callback=None, cancel_check=None, faces=[man, woman],
        )
    compound_calls = [args for name, args in compound_client.calls if name == "generate"]
    appearance_calls = [
        call for call in compound_calls
        if call.get("step_cap") == APPEARANCE_SAMPLE_STEPS
    ]
    assert appearance_calls
    assert all(call.get("denoise") != SURFACE_CONTINUATION_DENOISE for call in compound_calls)
    bald_call = appearance_calls[0]
    assert bald_call["keep_source_pixels"] is True
    assert bald_call["denoise"] == APPEARANCE_SAMPLE_DENOISE
    assert bald_call["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    bald_px = int((np.asarray(bald_call["mask_image"].convert("L")) >= 160).sum())
    man_px = int((np.asarray(man_mask.convert("L")) >= 160).sum())
    assert man_px * 0.8 <= bald_px <= man_px * 1.05, (bald_px, man_px)
    print("OK: el pase de apariencia compuesto pega la persona ya medida")
    body = _Masker().generate_mask(photo, "man")
    local = manager._local_appearance_mask(photo, body, [man, woman])
    assert local is not None
    local_arr = np.asarray(local)
    assert int(local_arr[40:50, 155:170].max()) > 96
    assert int(local_arr[50:65, 185:200].max()) > 96
    assert int(local_arr[82:96, 65:90].max()) == 0
    assert int(local_arr[10:24, 10:30].max()) == 0
    assert float((local_arr >= 96).mean()) < float((np.asarray(body) >= 96).mean()) * 0.5
    pasted = manager._integrate_local_generation(
        photo, Image.new("RGB", photo.size, "red"), local,
        (0, 0, photo.size[0], photo.size[1]), limit_mask=body,
    )
    assert pasted.getpixel((163, 30)) == photo.getpixel((163, 30))
    head_px = pasted.getpixel((163, 54))
    assert head_px[0] > 200 and head_px[1] < 40 and head_px[2] < 40, head_px
    print("OK: la máscara de apariencia sigue a la región medida y deja fuera el resto del sujeto")

    thing = np.zeros((140, 240), np.uint8)
    thing[20:110, 40:90] = 255
    thing[10:28, 180:220] = 255
    seed_arr = np.zeros((140, 240), np.uint8)
    seed_arr[20:50, 40:90] = 255
    seed = Image.fromarray(seed_arr, "L")
    continued = manager._continue_measured_region(seed, thing)
    continued_arr = np.asarray(continued)
    assert int(continued_arr[55, 60]) > 96
    assert int(continued_arr[105, 60]) == 0
    assert int(continued_arr[18, 200]) == 0
    print("OK: la misma continuación vale para cualquier región medida, no solo una cabeza")

    import time
    large_region = np.zeros((3000, 2000), np.uint8)
    large_region[300:2700, 600:1500] = 255
    large_focus = np.zeros((3000, 2000), np.uint8)
    large_focus[400:1600, 700:1600] = 255
    started = time.perf_counter()
    large_continued = manager._continue_measured_region(
        Image.fromarray(large_focus, "L"), large_region, report=False,
    )
    large_elapsed = time.perf_counter() - started
    assert large_continued is not None and large_continued.size == (2000, 3000)
    assert large_elapsed < 2.0, large_elapsed
    print(f"OK: continuación en 2000x3000 termina en {large_elapsed:.3f}s")

    portrait = Image.new("RGB", (400, 600), "white")
    portrait_region = np.zeros((600, 400), np.uint8)
    portrait_region[30:530, 120:380] = 255
    portrait_man = _Face((220, 50, 320, 170))
    portrait_woman = _Face((20, 80, 70, 180))
    portrait_mask = manager._local_appearance_mask(
        portrait, Image.fromarray(portrait_region, "L"), [portrait_man, portrait_woman],
    )
    assert portrait_mask is not None
    portrait_arr = np.asarray(portrait_mask)
    assert float((portrait_arr >= 160).mean()) <= 0.35
    assert int(portrait_arr[110, 270]) > 96
    assert int(portrait_arr[450, 300]) == 0
    assert int(portrait_arr[120, 40]) == 0
    print("OK: una continuación que no cabe se queda en el foco y conserva el resto")

    class _PortraitMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").lower()
            if "man" in text or "hombre" in text:
                arr[30:530, 120:380] = 255
            return Image.fromarray(arr, "L")

    portrait_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[portrait_man, portrait_woman]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_PortraitMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Man must be bald"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=portrait_client)):
        portrait_output, portrait_message, _ = manager.generate_intelligent(
            image=portrait, prompt="el hombre debe ser calvo", engine="qwen_camera",
        )
    assert portrait_output is not None and portrait_output.size == portrait.size
    # El refino focal añade una segunda llamada generate (máscara de zoom);
    # el contrato de esta aserción es la PRIMERA generate (la ventana medida).
    portrait_sent = next(
        c[1] for c in portrait_client.calls if c[0] == "generate"
    )
    # "bald" lights nothing on this masker, so the head of the measured
    # man is the zone: a local pass with the spatial anchor, not a
    # full-photo regen (which painted the scalp glossy).
    assert portrait_sent["mask_image"] is not None
    assert portrait_sent["keep_source_pixels"] is True
    assert portrait_sent["denoise"] == 1.0
    assert portrait_sent["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    assert portrait_sent["no_kontext"] is False
    assert portrait_sent.get("reference_image") is None
    assert portrait_output.getpixel((40, 120)) == portrait.getpixel((40, 120))
    assert portrait_output.getpixel((300, 450)) == portrait.getpixel((300, 450))
    assert portrait_output.getpixel((270, 110)) != portrait.getpixel((270, 110))
    print("OK: calva sin zona medida pega la cabeza; el cuerpo y el fondo quedan")

    head_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[man, woman]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_Masker()),
          patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Man must be bald"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=head_client)):
        output, message, _ = manager.generate_intelligent(
            image=photo, prompt="el hombre debe ser calvo", engine="qwen_camera",
        )
    assert output is not None and output.size == photo.size
    assert "solo la zona medida" in message, message
    # Primera generate: el refino focal añade una segunda (zoom de la zona).
    sent = next(c[1] for c in head_client.calls if c[0] == "generate")
    assert sent["preserve_frame"] is True
    assert sent["prompt_authority"] is True
    assert "mask_denoise" not in sent
    assert "fully nude" not in sent["prompt"].lower()
    # "bald" lights nothing here either; the head of the measured man is
    # the zone (local anchored pass), not the whole person on the photo.
    assert sent["mask_image"] is not None
    assert sent["keep_source_pixels"] is True
    assert sent["denoise"] == 1.0
    assert sent["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    assert sent["no_kontext"] is False
    assert sent.get("reference_image") is None
    assert output.getpixel((100, 120)) == photo.getpixel((100, 120))
    assert output.getpixel((90, 80)) == photo.getpixel((90, 80))
    assert output.getpixel((163, 54)) != photo.getpixel((163, 54))
    print("OK: calva sin zona medida pega la cabeza medida; el cuerpo queda")

    class _RequestMasker:
        """The segmenter answers the predicate, whatever it is, inside the subject."""

        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (prompt or "").strip(" .,!?").lower()
            if "man" in text or "hombre" in text:
                arr[40:110, 40:200] = 255
            elif text == "bald":
                arr[42:54, 148:186] = 255
            elif text == "grow a beard":
                arr[88:108, 140:190] = 255
            return Image.fromarray(arr, "L")

    import numpy as np
    subject = Image.new("L", photo.size, 0)
    subject.paste(255, (40, 40, 200, 110))
    with patch.object(manager, "_get_clipseg_masker", return_value=_RequestMasker()):
        bald_region = manager._request_region_on_subject(
            photo, subject, ["el hombre debe ser calvo", "Man must be bald"],
        )
        beard_region = manager._request_region_on_subject(
            photo, subject, ["the man must grow a beard"],
        )
    assert bald_region is not None and beard_region is not None
    bald_arr = np.asarray(bald_region)
    beard_arr = np.asarray(beard_region)
    assert int(bald_arr[46, 160]) > 96
    assert int(bald_arr[96, 160]) == 0
    assert int(beard_arr[96, 160]) > 96
    assert int(beard_arr[46, 160]) == 0
    assert manager._visual_request_queries(
        ["Man must be bald....", "el hombre debe ser calvo"]
    ) == ["bald", "calvo"]
    print("OK: la zona pedida sigue al predicado, no a una forma fija")

    request_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[man, woman]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_RequestMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Man must be bald"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=request_client)):
        request_output, request_message, _ = manager.generate_intelligent(
            image=photo, prompt="el hombre debe ser calvo", engine="qwen_camera",
        )
    assert request_output is not None
    request_generations = [c[1] for c in request_client.calls if c[0] == "generate"]
    # El contrato es "sin reintentos ciegos": ninguna pasada repite la misma
    # entrada. El refino focal (si dispara) entra con su propia ventana zoom,
    # así que sus entradas son distintas y la unicidad se mantiene.
    assert request_generations, request_client.calls
    assert len({
        (g["image"].size, g["mask_image"].size, g["prompt"])
        for g in request_generations
    }) == len(request_generations), [g["image"].size for g in request_generations]
    request_sent = request_generations[0]
    assert request_sent["keep_source_pixels"] is True
    assert request_sent["denoise"] == APPEARANCE_SAMPLE_DENOISE
    assert request_sent["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    assert request_sent["step_cap"] == APPEARANCE_SAMPLE_STEPS
    assert request_sent["mask_image"].size == request_sent["image"].size
    assert request_output.getpixel((160, 46)) != photo.getpixel((160, 46))
    # v5.1211 (Fix F): el agujero de «calvo» es la cabeza de la persona
    # editada y el pegado es el cambio medido dentro de esa cabeza. Con este
    # stub que devuelve rojo en toda la ventana, toda la cabeza cambia —
    # incluida la bbox de la cara de la persona editada, que el propio prompt
    # dice que puede cambiar; lo que se conserva es lo que queda FUERA de la
    # cabeza: fondo, cuerpo y las otras personas.
    assert request_output.getpixel((20, 20)) == photo.getpixel((20, 20))
    assert request_output.getpixel((60, 120)) == photo.getpixel((60, 120))
    print("OK: lo pedido se edita en la cabeza medida; el resto de la foto se queda")

    wide = Image.new("RGB", (80, 60), (180, 180, 180))
    sample = Image.new("RGB", (80, 60), (140, 140, 140))
    sample.paste((0, 0, 0), (10, 10, 22, 22))
    wide_mask = Image.new("L", (80, 60), 0)
    wide_mask.paste(255, (8, 8, 70, 50))
    kept = manager._integrate_local_generation(
        wide, sample, wide_mask, (0, 0, 80, 60), keep_measured_mask=True,
    )
    assert kept.getpixel((40, 30)) == (140, 140, 140)
    assert kept.getpixel((2, 2)) == (180, 180, 180)
    print("OK: la zona pedida se pega entera, no solo el pico mas oscuro")

    class _WeakMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            return None

        def probability(self, image, prompt):
            heat = np.full((image.height, image.width), 0.008, np.float32)
            heat[244:302, 463:509] = 0.083
            heat[32:97, 146:187] = 0.004
            return heat

    lying = _Face((463, 244, 509, 302))
    standing = _Face((146, 32, 187, 97))
    wide = Image.new("RGB", (800, 450), "white")
    flat = np.full((450, 800), 0.05, np.float32)
    assert manager._leading_face(flat, [lying, standing])[0] is None
    weak_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[lying, standing]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_WeakMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Man must be bald"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=weak_client)):
        output, message, _ = manager.generate_intelligent(
            image=wide, prompt="el hombre debe ser calvo", engine="qwen_camera",
        )
    assert output is not None and output.size == wide.size
    assert "solo la zona medida" in message, message
    assert "zona no medida" not in message
    weak_sent = weak_client.calls[-1][1]
    # CLIPSeg cannot measure the man, so the face with the most response
    # is recovered and its head is the zone: a local anchored pass, not a
    # full-photo regen.
    assert weak_sent["mask_image"] is not None
    assert weak_sent["keep_source_pixels"] is True
    assert weak_sent["denoise"] == 1.0
    assert weak_sent["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    assert weak_sent["no_kontext"] is False
    assert weak_sent.get("reference_image") is None
    assert output.getpixel((20, 20)) == wide.getpixel((20, 20))
    assert output.getpixel((166, 64)) == wide.getpixel((166, 64))
    assert output.getpixel((486, 257)) != wide.getpixel((486, 257))
    assert weak_sent["preserve_frame"] is True
    print("OK: máscara binaria vacía; la cara con más respuesta se muestrea de cerca")

    class _LampMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            arr[40:90, 30:80] = 255
            return Image.fromarray(arr, "L")

    lamp = Image.new("RGB", (240, 140), "white")
    lamp_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_LampMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt", return_value="Add a lamp to the table"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client", return_value=lamp_client)):
        lamp_output, lamp_message, _ = manager.generate_intelligent(
            image=lamp, prompt="añade una lámpara a la mesa", engine="qwen_camera",
        )
    assert lamp_output is not None and lamp_output.size == lamp.size
    assert "región medida" in lamp_message, lamp_message
    assert "región no medida" not in lamp_message
    lamp_sent = lamp_client.calls[-1][1]
    assert lamp_sent["mask_image"].size == lamp_sent["image"].size
    assert float((np.asarray(lamp_sent["mask_image"]) >= 160).mean()) <= 0.35
    assert lamp_sent["keep_source_pixels"] is True
    assert lamp_sent.get("reference_image") is None
    assert lamp_sent["preserve_frame"] is True
    assert lamp_sent["denoise"] == 1.0
    assert lamp_sent["num_inference_steps"] == REGION_SAMPLE_STEPS
    assert lamp_sent["step_cap"] == REGION_SAMPLE_STEPS
    assert lamp_sent["no_kontext"] is False
    assert lamp_output.getpixel((200, 120)) == lamp.getpixel((200, 120))
    assert lamp_output.getpixel((55, 65)) != lamp.getpixel((55, 65))
    viewpoint = manager._prompt_authority_facts("cambia el pov", "change the viewpoint")
    assert not viewpoint["local_region"] and not viewpoint["local_appearance"], viewpoint
    assert manager._explicit_viewpoint_request("cambia el pov", "change the viewpoint")
    print("OK: añadir mide la zona y deja el resto; el pov no entra en ese recorte")

    wall_facts = manager._prompt_authority_facts(
        "las paredes deben ser grises", "the walls must be grey",
    )
    assert wall_facts["local_region"] and wall_facts["surface_recolor"], wall_facts
    assert not wall_facts["local_appearance"], wall_facts
    bald_facts = manager._prompt_authority_facts(
        "el hombre debe ser calvo", "the man must be bald",
    )
    assert bald_facts["local_appearance"] and not bald_facts["local_region"], bald_facts

    class _WallMasker:
        def generate_mask(self, image, query, threshold=0.4):
            words = set(__import__("re").findall(r"[a-záéíóúñ]+", (query or "").lower()))
            arr = np.zeros((image.height, image.width), np.uint8)
            if words & {"wall", "walls", "pared", "paredes"}:
                arr[:, :70] = 255
            elif words & {"bald", "calvo", "calva"}:
                arr[18:40, 72:96] = 255
            elif words & {"man", "hombre", "person", "human"}:
                arr[10:70, 60:110] = 255
            else:
                return None
            return Image.fromarray(arr, "L")

    def _faces_for_shape(image, _analyser, _thresh):
        shape = getattr(image, "shape", None)
        if shape is not None and len(shape) >= 2 and shape[0] == 80 and shape[1] == 120:
            return [SimpleNamespace(bbox=(70, 20, 100, 55), age=34, det_score=0.9)]
        return []

    # v5.1225: the wall surface is one flat color and a different-colored
    # curtain closes it on the right. A whole-surface recolor follows its
    # color, so the distant pixel stays byte-identical exactly where the
    # color changes, not where the segmenter fragment happened to end.
    photo_arr = np.zeros((80, 120, 3), np.float32)
    photo_arr[:] = np.array([180, 170, 160], np.float32)
    photo_arr[:, 98:] = np.array([90, 110, 150], np.float32)
    photo = Image.fromarray(photo_arr.astype(np.uint8))
    order_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", side_effect=_faces_for_shape),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_WallMasker()),
          patch(
              "roop.img_editor.prompt_translator.translate_prompt",
              return_value="the man must be bald, the walls must be grey",
          ),
          patch(
              "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=order_client,
          )):
        ordered, order_message, _ = manager.generate_intelligent(
            image=photo,
            prompt="EL HOMBRE DEBE SER CALVO, LAS PAREDES DEBEN SER GRISES",
            engine="qwen_camera",
        )
    assert ordered is not None and ordered.size == photo.size
    assert "un solo pase Turbo" in order_message, order_message
    assert "cada cambio por separado" not in order_message, order_message
    order_calls = [call for name, call in order_client.calls if name == "generate"]
    assert len(order_calls) == 1, order_calls
    order_call = order_calls[0]
    order_prompt = order_call["prompt"].lower()
    assert ("grey" in order_prompt or "wall" in order_prompt) and "bald" in order_prompt, order_prompt
    assert order_call["mask_image"] is not None
    assert order_call["keep_source_pixels"] is True
    assert order_call["preserve_frame"] is True, order_call
    assert order_call.get("reference_image") is None
    assert order_call["denoise"] == 1.0
    assert order_call["num_inference_steps"] == 12
    assert order_call["no_kontext"] is False
    # Both clauses measured a zone (man and wall); the union is the only
    # paste, so a pixel far from either stays byte-identical, while a
    # pixel inside the wall zone takes the generated color.
    order_src = np.asarray(photo.convert("RGB"))
    order_out = np.asarray(ordered.convert("RGB"))
    assert (order_out[2, 116] == order_src[2, 116]).all(), order_out[2, 116]
    assert (order_out[78, 116] == order_src[78, 116]).all(), order_out[78, 116]
    order_wall = order_out[40, 10]
    assert int(order_wall[0]) > 180, order_wall
    print("OK: calvo y paredes comparten un pase; el fondo lejano queda byte a byte")

    class _SplitPeople:
        def probability(self, image, query):
            heat = np.full((image.height, image.width), 0.05, np.float32)
            text = (query or "").lower()
            if "rodillas" in text:
                heat[40:58, 50:70] = 0.90
            elif "knees" in text:
                heat[8:28, 8:28] = 0.30
            return heat

    room = Image.new("RGB", (80, 60), (210, 200, 190))
    standing = SimpleNamespace(bbox=(8, 8, 28, 28))
    kneeling = SimpleNamespace(bbox=(50, 40, 70, 58))
    with patch.object(manager, "_get_clipseg_masker", return_value=_SplitPeople()):
        chosen = manager._appearance_head_pass(
            room,
            "el hombre blanco que esta de rodillas debe ser calvo",
            "The white man on his knees must be bald",
            [standing, kneeling],
        )
    assert chosen is not None, "la frase en español tiene que señalar al de rodillas"
    mask = np.asarray(chosen["mask"])
    assert float(mask[48:56, 54:66].mean()) > float(mask[10:24, 10:24].mean())
    class _Tie:
        def probability(self, image, query):
            heat = np.full((image.height, image.width), 0.05, np.float32)
            text = (query or "").lower()
            if "hombre" in text:
                heat[40:58, 50:70] = 0.80
            elif "man" in text:
                heat[8:28, 8:28] = 0.80
            return heat

    with patch.object(manager, "_get_clipseg_masker", return_value=_Tie()):
        ambiguous = manager._appearance_head_pass(
            room,
            "el hombre debe ser calvo",
            "the man must be bald",
            [standing, kneeling],
        )
    assert ambiguous is None
    print("OK: con varias personas manda la frase que señala una cara")

    class _SmallPerson:
        """A named person under 4% of a group photo, with the request inside him."""

        def generate_mask(self, image, query, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            text = (query or "").lower()
            if "bald" in text or "calvo" in text:
                arr[36:48, 55:75] = 255
            elif any(word in text for word in ("white", "raza", "hombre", "man")):
                arr[35:95, 45:90] = 255
            else:
                return None
            return Image.fromarray(arr, "L")

    group = Image.new("RGB", (400, 300), (200, 198, 196))
    kneeling = SimpleNamespace(bbox=(50, 40, 80, 70), age=35, det_score=0.9)
    bystander = SimpleNamespace(bbox=(10, 10, 36, 36), age=40, det_score=0.9)
    with patch.object(manager, "_get_clipseg_masker", return_value=_SmallPerson()):
        small_passes = manager._measured_subject_passes(
            group,
            "el hombre de raza blanca es calvo",
            "The white man is bald",
            [kneeling, bystander],
        )
    assert len(small_passes) == 1, small_passes
    assert small_passes[0]["coverage"] < 0.04, small_passes[0]["coverage"]

    class _Speck:
        def generate_mask(self, image, query, threshold=0.4):
            arr = np.zeros((image.height, image.width), np.uint8)
            arr[2:6, 2:6] = 255
            return Image.fromarray(arr, "L")

    with patch.object(manager, "_get_clipseg_masker", return_value=_Speck()):
        speck_passes = manager._measured_subject_passes(
            group, "el hombre es calvo", "the man is bald", [kneeling, bystander],
        )
    assert speck_passes == [], speck_passes

    small_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[kneeling, bystander]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_SmallPerson()),
          patch(
              "roop.img_editor.prompt_translator.translate_prompt",
              return_value="The white man is bald",
          ),
          patch(
              "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=small_client,
          )):
        small_output, small_message, _ = manager.generate_intelligent(
            image=group,
            prompt="el hombre de raza blanca es calvo",
            engine="qwen_camera",
        )
    assert small_output is not None and small_output.size == group.size
    assert "no se sustituye" not in small_message, small_message
    small_generations = [call for name, call in small_client.calls if name == "generate"]
    assert len(small_generations) == 1, small_generations
    small_sent = small_generations[0]
    assert small_sent["keep_source_pixels"] is True
    assert small_sent.get("reference_image") is None
    assert small_sent["denoise"] == APPEARANCE_SAMPLE_DENOISE
    assert small_sent["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    assert small_output.getpixel((65, 42)) != group.getpixel((65, 42))
    assert small_output.getpixel((65, 80)) == group.getpixel((65, 80))
    assert small_output.getpixel((20, 20)) == group.getpixel((20, 20))
    print("OK: una persona por debajo del 4% se mide y se pega la zona pedida")

    class _HeadBald:
        def generate_mask(self, image, query, threshold=0.4):
            text = (query or "").lower()
            if "bald" not in text and "calvo" not in text:
                return None
            arr = np.zeros((image.height, image.width), np.uint8)
            arr[32:46, 55:78] = 255
            return Image.fromarray(arr, "L")

        def probability(self, image, query):
            heat = np.full((image.height, image.width), 0.04, np.float32)
            heat[40:70, 50:80] = 0.70
            return heat

    head_bald_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[kneeling, bystander]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_HeadBald()),
          patch(
              "roop.img_editor.prompt_translator.translate_prompt",
              return_value="The white man is bald",
          ),
          patch(
              "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=head_bald_client,
          )):
        head_output, head_message, _ = manager.generate_intelligent(
            image=group,
            prompt="el hombre de raza blanca es calvo",
            engine="qwen_camera",
        )
    assert head_output is not None and head_output.size == group.size
    assert "no se sustituye" not in head_message, head_message
    assert head_output.getpixel((66, 39)) != group.getpixel((66, 39))
    # v5.1211 (Fix F): el agujero de «calvo» es la cabeza de la persona
    # editada y el pegado es el cambio medido dentro de ella; (65, 62) cae
    # dentro de la cabeza de kneeling (bbox de la cara 50,40–80,70) y con
    # este stub rojo cambia. Byte a byte queda lo que está FUERA de la
    # cabeza: fondo y la otra persona.
    assert head_output.getpixel((20, 20)) == group.getpixel((20, 20))
    head_sent = [call for name, call in head_bald_client.calls if name == "generate"]
    assert len(head_sent) == 1, head_sent
    assert head_sent[0].get("reference_image") is None
    print("OK: sin máscara de persona, la cara elegida pega la zona pedida")

    small = Image.new("RGB", (400, 600), "white")
    small_mask = np.zeros((600, 400), np.uint8)
    small_mask[80:241, 250:351] = 255
    small_work, small_region, small_box = manager._appearance_work_region(
        small, Image.fromarray(small_mask, "L"),
    )
    # 2026-10-01: el suelo de 1 MP hacia que TODA foto por debajo de 1048576 px
    # (400x600, 800x533, 1280x720) abriera la foto entera, y el pegado
    # componia una escena regenerada sobre el original. Ahora la ventana es
    # local tambien en fotos pequenas: la zona medida sigue dentro y la
    # ventana no se come el cuadro.
    small_area = small_work.size[0] * small_work.size[1]
    assert small_box != (0, 0, 400, 600), small_box
    assert small_area <= 400 * 600 * 0.98, small_area
    assert int(np.asarray(small_region)[
        80 - small_box[1]:241 - small_box[1], 250 - small_box[0]:351 - small_box[0]
    ].min()) == 255
    print("OK: foto pequena abre una ventana local en vez de entrar entera")
    wide_photo = Image.new("RGB", (1600, 1200), "white")
    wide_mask = np.zeros((1200, 1600), np.uint8)
    wide_mask[540:660, 760:840] = 255
    wide_work, wide_region, wide_box = manager._appearance_work_region(
        wide_photo, Image.fromarray(wide_mask, "L"),
    )
    wide_area = wide_work.size[0] * wide_work.size[1]
    assert wide_box != (0, 0, 1600, 1200)
    assert 1024 * 1024 * 0.5 < wide_area < 1600 * 1200
    assert float((np.asarray(wide_region) >= 160).mean()) <= 0.35
    assert int(np.asarray(wide_region)[540 - wide_box[1]:660 - wide_box[1], 760 - wide_box[0]:840 - wide_box[0]].min()) == 255
    print("OK: foto pequeña entra entera; foto grande abre una ventana con foto alrededor")
    assert measured_appearance_sample(True) == {
        "num_inference_steps": APPEARANCE_SAMPLE_STEPS,
        "denoise": APPEARANCE_SAMPLE_DENOISE,
        "step_cap": APPEARANCE_SAMPLE_STEPS,
        "no_kontext": False,
    }
    assert measured_appearance_sample(False) == {"num_inference_steps": 12, "denoise": 1.0, "no_kontext": True}
    assert longcat_turbo_denoise(True, True, True, 1.0) == 1.0
    assert longcat_turbo_denoise(
        True, True, True, APPEARANCE_SAMPLE_DENOISE,
    ) == APPEARANCE_SAMPLE_DENOISE
    assert longcat_turbo_denoise(True, True, True, SURFACE_CONTINUATION_DENOISE) == SURFACE_CONTINUATION_DENOISE
    assert SURFACE_CONTINUATION_DENOISE < 1.0
    assert longcat_turbo_denoise(True, True, False, 1.0) == 0.85
    assert longcat_turbo_denoise(True, False, False, 1.0) == 0.55
    assert longcat_turbo_denoise(True, False, False, 1.0, prompt_authority=True) == 1.0
    print("OK: edición corporal sin máscara usa denoise 1.0")
    hole = np.zeros((80, 80), np.uint8)
    hole[10:40, 50:78] = 255
    known = [SimpleNamespace(bbox=[5, 40, 35, 75])]
    inserted = [SimpleNamespace(bbox=[50, 8, 78, 38])]
    same = [SimpleNamespace(bbox=[6, 42, 36, 76])]
    zone = Image.fromarray(hole, "L")
    assert manager._inserted_face_in_mask(known, inserted, zone)
    assert not manager._inserted_face_in_mask(known, same, zone)
    canvas = np.zeros((80, 80, 3), np.uint8)
    canvas[:, :] = (10, 20, 30)
    canvas[30:50, 30:50] = (220, 5, 5)
    continued = manager._continue_measured_surface(
        Image.fromarray(canvas, "RGB"),
        Image.fromarray(
            np.pad(np.full((20, 20), 255, np.uint8), ((30, 30), (30, 30))),
            "L",
        ),
    )
    continued_px = np.asarray(continued)
    assert continued_px.shape == canvas.shape
    assert continued_px[0, 0].tolist() == [10, 20, 30]
    assert int(continued_px[40, 40].max()) < 180
    print("OK: una cara nueva dentro de la zona se rehace desde la superficie de alrededor")

    photo = np.full((80, 80, 3), (180, 170, 160), np.uint8)
    photo[8:36, 40:72] = (30, 28, 26)
    photo[30:58, 36:58] = (190, 160, 140)
    photo[40:46, 42:48] = (20, 18, 16)
    photo[10:16, 44:50] = (220, 190, 170)
    seed = np.zeros((80, 80), np.uint8)
    seed[10:22, 44:68] = 255
    subject = np.zeros((80, 80), np.uint8)
    subject[4:64, 30:76] = 255
    face = SimpleNamespace(bbox=[36, 30, 58, 58])
    completed = manager._complete_request_surface(
        Image.fromarray(photo, "RGB"),
        Image.fromarray(seed, "L"),
        Image.fromarray(subject, "L"),
        [face],
    )
    assert completed is not None
    completed_px = np.asarray(completed) >= 128
    assert completed_px[18, 60]
    assert completed_px[28, 66]
    assert not completed_px[12, 46]
    assert not completed_px[42, 44]
    assert not completed_px[52, 48]
    dark = np.all(photo == np.array([30, 28, 26]), axis=2)
    skin = np.all(photo == np.array([190, 160, 140]), axis=2)
    assert dark.sum() > 0 and (completed_px & dark).sum() / dark.sum() >= 0.80
    assert not (completed_px & skin).any()
    wide = np.zeros((80, 80), np.uint8)
    wide[8:50, 34:70] = 255
    wide_px = wide >= 128
    wide_completed = manager._complete_request_surface(
        Image.fromarray(photo, "RGB"),
        Image.fromarray(wide, "L"),
        Image.fromarray(subject, "L"),
        [face],
    )
    if wide_completed is not None:
        added = (np.asarray(wide_completed) >= 128) & ~wide_px
        eye = np.all(photo == np.array([20, 18, 16]), axis=2)
        assert not (added & skin).any()
        assert not (added & ~dark & ~eye).any()
    print("OK: un pico medido sigue por la misma superficie y no entra en la cara")

    def _follow(photo_rgb, seed_box, subject_box, faces):
        canvas = np.array(photo_rgb, copy=True)
        seed_m = np.zeros(canvas.shape[:2], np.uint8)
        subject_m = np.zeros(canvas.shape[:2], np.uint8)
        sy1, sy2, sx1, sx2 = seed_box
        seed_m[sy1:sy2, sx1:sx2] = 255
        uy1, uy2, ux1, ux2 = subject_box
        subject_m[uy1:uy2, ux1:ux2] = 255
        found = manager._complete_request_surface(
            Image.fromarray(canvas, "RGB"),
            Image.fromarray(seed_m, "L"),
            Image.fromarray(subject_m, "L"),
            faces,
        )
        return found

    def _recall(photo_rgb, seed_box, subject_box, faces, target_color, other_color):
        found = _follow(photo_rgb, seed_box, subject_box, faces)
        assert found is not None
        kept = np.asarray(found) >= 128
        target = np.all(photo_rgb == np.array(target_color), axis=2)
        other = np.all(photo_rgb == np.array(other_color), axis=2)
        subject_on = np.zeros(photo_rgb.shape[:2], bool)
        uy1, uy2, ux1, ux2 = subject_box
        subject_on[uy1:uy2, ux1:ux2] = True
        target &= subject_on
        other &= subject_on
        recall = (kept & target).sum() / max(1, int(target.sum()))
        leak = int((kept & other).sum())
        assert recall >= 0.80, recall
        assert leak == 0, leak
        return kept

    # A long darker surface, a long lighter one, and a same-luminance color.
    # The seed is only the near end. The far end has to come along, and the
    # other color has to stay out. No face is required.
    base = np.full((120, 120, 3), (210, 205, 198), np.uint8)
    base[20:100, 15:40] = (25, 24, 22)
    _recall(base, (20, 36, 18, 34), (5, 115, 5, 100), [], (25, 24, 22), (210, 205, 198))
    light = np.full((120, 120, 3), (30, 28, 26), np.uint8)
    light[20:100, 15:40] = (240, 236, 230)
    _recall(light, (20, 36, 18, 34), (5, 115, 5, 100), [], (240, 236, 230), (30, 28, 26))
    red = np.full((120, 120, 3), (120, 120, 120), np.uint8)
    red[20:100, 15:40] = (200, 80, 80)
    _recall(red, (20, 36, 18, 34), (5, 115, 5, 100), [], (200, 80, 80), (120, 120, 120))
    stripe = np.full((80, 140, 3), (200, 200, 190), np.uint8)
    stripe[48:62, 8:130] = (20, 80, 40)
    stripe_kept = _recall(
        stripe, (50, 60, 10, 24), (40, 75, 5, 135), [], (20, 80, 40), (200, 200, 190),
    )
    assert stripe_kept[54, 120]
    # A beard is the measured color on the lower face. The cheek is not.
    beard = np.full((120, 120, 3), (200, 170, 150), np.uint8)
    beard[60:100, 35:85] = (45, 35, 30)
    beard_face = SimpleNamespace(bbox=[30, 10, 90, 110])
    beard_kept = _recall(
        beard, (62, 78, 45, 70), (8, 112, 20, 100), [beard_face],
        (45, 35, 30), (200, 170, 150),
    )
    assert beard_kept[90, 55]
    assert not beard_kept[30, 55]
    # The other person's hand is a different color lying on the surface.
    handed = np.full((120, 120, 3), (32, 30, 28), np.uint8)
    handed[40:75, 50:85] = (210, 170, 150)
    handed[18:26, 18:28] = (210, 170, 150)
    other_face = SimpleNamespace(bbox=[0, 0, 14, 24])
    hand_kept = _recall(
        handed, (12, 32, 14, 36), (8, 112, 8, 112), [other_face],
        (32, 30, 28), (210, 170, 150),
    )
    assert hand_kept[100, 100]
    assert not hand_kept[22, 22]
    flat = np.full((80, 80, 3), (180, 180, 180), np.uint8)
    assert _follow(flat, (10, 28, 10, 30), (4, 70, 4, 70), []) is None
    print("OK: la misma superficie sigue el color pedido, claro u oscuro, hasta que cambia")

    couch = np.full((70, 80, 3), (240, 240, 240), np.uint8)
    couch[10:60, 20:60] = (190, 150, 120)
    couch[12:28, 28:52] = (30, 20, 15)
    hair = np.zeros((70, 80), np.uint8)
    hair[12:28, 28:52] = 255
    person = np.zeros((70, 80), np.uint8)
    person[10:60, 20:60] = 255
    couch_image = Image.fromarray(couch, "RGB")
    hair_image = Image.fromarray(hair, "L")
    person_image = Image.fromarray(person, "L")
    profile = np.full((80, 100, 3), (240, 240, 240), np.uint8)
    profile[20:70, 15:55] = (190, 150, 120)
    profile[8:40, 25:78] = (30, 20, 15)
    profile_seed = np.zeros((80, 100), np.uint8)
    profile_seed[12:28, 30:48] = 255
    profile_subject = np.zeros((80, 100), np.uint8)
    profile_subject[10:72, 8:52] = 255
    profile_face = SimpleNamespace(bbox=(18, 28, 50, 68))
    profile_grown = manager._complete_request_surface(
        Image.fromarray(profile, "RGB"),
        Image.fromarray(profile_seed, "L"),
        Image.fromarray(profile_subject, "L"),
        [profile_face],
    )
    assert profile_grown is not None
    profile_px = np.asarray(profile_grown) >= 128
    assert profile_px[18, 70]
    assert not profile_px[2, 2]
    assert not profile_px[50, 30]
    print("OK: la misma superficie sigue detras del localizador")

    # The same dark peak must not cross a wooden room just because that
    # brown sits halfway toward the skin. The hair that really matches
    # the peak, including the part past the silhouette, still comes along.
    wood = np.full((100, 140, 3), (90, 70, 55), np.uint8)
    wood[20:90, 40:100] = (210, 170, 150)
    wood[8:40, 55:110] = (26, 44, 45)
    wood_seed = np.zeros((100, 140), np.uint8)
    wood_seed[12:28, 60:80] = 255
    wood_subject = np.zeros((100, 140), np.uint8)
    wood_subject[20:90, 40:100] = 255
    wood_grown = manager._complete_request_surface(
        Image.fromarray(wood, "RGB"),
        Image.fromarray(wood_seed, "L"),
        Image.fromarray(wood_subject, "L"),
        [],
    )
    assert wood_grown is not None
    wood_px = np.asarray(wood_grown) >= 128
    assert wood_px[20, 100]
    assert not wood_px[5, 5]
    assert int(wood_px[:8, :40].sum()) == 0
    print("OK: el color a medias no se lleva el cuarto de fuera")

    cover_client = FakeClient()
    cover_out, cover_msg, cover_ok = manager._apply_subject_passes(
        cover_client,
        couch_image,
        [{
            "text": "the man must be bald",
            "mask": hair_image,
            "person": person_image,
        }],
        1, "", None, 1.0, None, None,
        localize=True, faces=None,
    )
    assert cover_ok, cover_msg
    cover_sent = [call for name, call in cover_client.calls if name == "generate"]
    assert len(cover_sent) == 1, cover_sent
    # The appearance recipe anchors the photo spatially; a flat cover
    # would hide the very surface the model must edit (the bald head came
    # out pale and plasticky). The original photo goes, no cover.
    assert cover_sent[0].get("reference_image") is None
    assert cover_sent[0]["no_kontext"] is False
    assert cover_sent[0]["denoise"] == APPEARANCE_SAMPLE_DENOISE
    assert cover_sent[0]["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    assert APPEARANCE_SAMPLE_DENOISE == 1.0
    print("OK: la pasada de apariencia manda la foto con ancla espacial, sin relleno plano")

    class UnchangedThenEdited(FakeClient):
        def generate(self, **kwargs):
            self.calls.append(("generate", kwargs))
            source = kwargs["image"].convert("RGB")
            if len(self.calls) == 1:
                return SimpleNamespace(image=source.copy()), "sin cambio"
            return SimpleNamespace(image=Image.new("RGB", source.size, "red")), "cambio"

    retry_client = UnchangedThenEdited()
    retry_out, retry_msg, retry_ok = manager._apply_subject_passes(
        retry_client, couch_image,
        [{"text": "the man must be bald", "mask": hair_image, "person": person_image}],
        1, "", None, 1.0, None, None, localize=True, faces=None,
    )
    retry_calls = [call for name, call in retry_client.calls if name == "generate"]
    assert retry_ok and len(retry_calls) == 2, (retry_msg, retry_calls)
    assert "segundo intento: cambio" in retry_msg, retry_msg
    assert not manager._edit_is_nearly_unchanged(couch_image, retry_out)
    for field in ("seed", "num_inference_steps", "denoise", "step_cap", "no_kontext"):
        assert retry_calls[0][field] == retry_calls[1][field], field
    assert retry_calls[0]["no_kontext"] is False
    assert retry_calls[0].get("reference_image") is None
    assert retry_calls[1].get("reference_image") is None
    assert retry_calls[0]["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    print("OK: segunda pasada de apariencia se ejecuta con la misma receta")

    wired = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[kneeling, bystander]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_HeadBald()),
          patch(
              "roop.img_editor.prompt_translator.translate_prompt",
              return_value="The white man is bald",
          ),
          patch(
              "roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=wired,
          )):
        wired_output, wired_message, _ = manager.generate_intelligent(
            image=group,
            prompt="el hombre de raza blanca es calvo",
            engine="qwen_camera",
        )
    assert wired_output is not None, wired_message
    wired_sent = [call for name, call in wired.calls if name == "generate"]
    assert len(wired_sent) == 1, wired_sent
    assert wired_sent[0].get("reference_image") is None
    assert wired_sent[0]["denoise"] == APPEARANCE_SAMPLE_DENOISE
    print("OK: la cara elegida envia a Turbo solo la foto, sin vista cubierta")

    # The predicate lights nothing. Turbo sees the photo. Hair, chest and
    # legs change because the measured person is pasted. The wall stays.
    empty_photo = np.full((200, 160, 3), (240, 240, 240), np.uint8)
    empty_photo[18:185, 40:120] = (190, 150, 120)
    empty_photo[18:52, 50:110] = (40, 30, 25)
    empty_image = Image.fromarray(empty_photo, "RGB")
    empty_face = SimpleNamespace(bbox=(55, 48, 105, 90), age=35, det_score=0.9)

    class _EmptyBald:
        def generate_mask(self, image, query, threshold=0.4):
            text = (query or "").lower()
            names_person = bool(re.search(r"\b(?:man|hombre|person|human)\b", text))
            if not names_person:
                return None
            arr = np.zeros((image.height, image.width), np.uint8)
            arr[18:185, 40:120] = 255
            return Image.fromarray(arr, "L")

    empty_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[empty_face]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_EmptyBald()),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="The man must be bald"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=empty_client)):
        empty_output, empty_message, _ = manager.generate_intelligent(
            image=empty_image,
            prompt="el hombre debe ser calvo",
            engine="qwen_camera",
        )
    assert empty_output is not None and empty_output.size == empty_image.size, empty_message
    assert "no se sustituye" not in empty_message, empty_message
    empty_generations = [call for name, call in empty_client.calls if name == "generate"]
    assert empty_generations, empty_client.calls
    empty_sent = empty_generations[0]
    # The head of the measured person is the zone for the bald fallback:
    # a local anchored pass, not the whole person on the full photo.
    assert empty_sent.get("reference_image") is None
    assert empty_sent["mask_image"] is not None
    assert empty_sent["keep_source_pixels"] is True
    assert empty_sent["no_kontext"] is False
    assert empty_sent["denoise"] == 1.0
    assert empty_sent["num_inference_steps"] == APPEARANCE_SAMPLE_STEPS
    empty_src = np.asarray(empty_image)
    empty_out = np.asarray(empty_output.convert("RGB"))
    assert not np.array_equal(empty_out[30, 80], empty_src[30, 80])
    assert not np.array_equal(empty_out[70, 80], empty_src[70, 80])
    assert np.array_equal(empty_out[160, 80], empty_src[160, 80])
    assert np.array_equal(empty_out[4, 80], empty_src[4, 80])
    print("OK: calva sin zona medida pega la cabeza; el cuerpo y el fondo quedan")

    # "elimina tatoo" was a full-frame LongCat pass: the conjugated verb was
    # only a partial and the English "removes" tied the body score. It has to
    # measure the tattoo and the walls, and leave the rest of the photo.
    tattoo_facts = manager._prompt_authority_facts(
        "elimina tatoo de la mujer", "removes tatoo from woman",
    )
    assert tattoo_facts["local_region"] and not tattoo_facts["body"], tattoo_facts
    assert not tattoo_facts["pose"] and not tattoo_facts["local_appearance"], tattoo_facts
    wall_clause = manager._prompt_authority_facts(
        "las paredes son grises", "walls are gray",
    )
    assert wall_clause["local_region"] and wall_clause["surface_recolor"], wall_clause
    tattoo_photo = np.full((180, 220, 3), (30, 50, 90), np.uint8)
    tattoo_photo[:, 150:220] = (180, 180, 180)
    tattoo_image = Image.fromarray(tattoo_photo, "RGB")

    class _TattooWallMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            assert isinstance(image, Image.Image), type(image)
            text = (prompt or "").lower()
            if text.strip() in {"chair", "plant"}:
                return None
            arr = np.zeros((image.height, image.width), np.uint8)
            if "tatoo" in text or "tattoo" in text or "tatuaje" in text:
                arr[40:80, 30:70] = 255
            elif "pared" in text or "wall" in text or "gris" in text or "gray" in text:
                arr[:, 150:220] = 255
            return Image.fromarray(arr, "L")

    tattoo_client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_TattooWallMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="removes tatoo from woman, walls are gray"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=tattoo_client)):
        tattoo_output, tattoo_message, _ = manager.generate_intelligent(
            image=tattoo_image,
            prompt="elimina tatoo de la mujer, las paredes son grises",
            engine="qwen_camera",
        )
    assert tattoo_output is not None and tattoo_output.size == tattoo_image.size, tattoo_message
    tattoo_generations = [call for name, call in tattoo_client.calls if name == "generate"]
    assert len(tattoo_generations) == 1, (len(tattoo_generations), tattoo_message)
    tattoo_call = tattoo_generations[0]
    assert tattoo_call.get("mask_image") is not None, tattoo_message
    assert tattoo_call.get("preserve_frame") is True, tattoo_message
    assert tattoo_call.get("keep_source_pixels") is True, tattoo_message
    assert tattoo_call.get("no_kontext") is False, tattoo_message
    assert "un solo pase Turbo" in tattoo_message, tattoo_message
    tattoo_out = np.asarray(tattoo_output.convert("RGB"))
    assert np.array_equal(tattoo_out[10, 10], tattoo_photo[10, 10]), tattoo_out[10, 10]
    assert not np.array_equal(tattoo_out[55, 45], tattoo_photo[55, 45]), tattoo_out[55, 45]
    assert not np.array_equal(tattoo_out[20, 180], tattoo_photo[20, 180]), tattoo_out[20, 180]
    print("OK: tatuaje y paredes comparten un pase; la unión medida es lo único pegado")

    # A few skin pixels where the wall meets a cheek are cut off. Covering
    # most of that face drops the continuation and keeps the measured peak.
    edge_photo = np.full((180, 220, 3), (150, 150, 150), np.uint8)
    edge_photo[150:180, :] = (80, 80, 80)
    edge_photo[40:130, 150:210] = (210, 150, 120)
    edge_photo[40:130, 146:154] = (158, 145, 135)
    edge_image = Image.fromarray(edge_photo, "RGB")
    edge_seed = Image.new("L", edge_image.size, 0)
    edge_seed.paste(255, (20, 30, 80, 90))
    edge_subject = Image.new("L", edge_image.size, 255)
    swallow_photo = edge_photo.copy()
    swallow_photo[40:130, 140:210] = (158, 145, 135)
    swallow_image = Image.fromarray(swallow_photo, "RGB")

    class _QuietMasker:
        def generate_mask(self, image, prompt, threshold=0.4):
            assert isinstance(image, Image.Image), type(image)
            return None

    edge_mask = manager._complete_request_surface(
        edge_image, edge_seed, edge_subject,
        [SimpleNamespace(bbox=(146, 40, 210, 130))],
        masker=_QuietMasker(), negative_query="chair", positive_query="plant",
        complete_hair=False,
    )
    assert edge_mask is not None and edge_mask.size == edge_image.size
    edge_arr = np.asarray(edge_mask.convert("L"))
    assert int(edge_arr[40, 100]) >= 128
    assert int(edge_arr[80, 180]) < 128
    swallow_mask = manager._complete_request_surface(
        swallow_image, edge_seed, edge_subject,
        [SimpleNamespace(bbox=(140, 40, 210, 130))],
        masker=_QuietMasker(), negative_query="chair", positive_query="plant",
        complete_hair=False,
    )
    assert swallow_mask is None
    wide = edge_image.resize((900, 180), Image.Resampling.NEAREST)
    wide_seed = edge_seed.resize(wide.size, Image.Resampling.NEAREST)
    wide_subject = edge_subject.resize(wide.size, Image.Resampling.NEAREST)
    wide_mask = manager._complete_request_surface(
        wide, wide_seed, wide_subject,
        [SimpleNamespace(bbox=(146 * 900 / 220, 40, 210 * 900 / 220, 130))],
        masker=_QuietMasker(), negative_query="chair", positive_query="plant",
        complete_hair=False,
    )
    assert wide_mask is not None and wide_mask.size == wide.size
    print("OK: un roce de otra cara se recorta; tragarse la cara no")

    from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient

    workflow_client = FluxEditComfyClient()
    workflow_client._loaded = True
    workflow_client._flux_version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
    workflow_client._is_longcat_turbo = True
    workflow_client._is_qwen_image_edit = False
    workflow_client._is_qwen_q5 = False
    workflow_client._is_kontext_dev = False
    workflow_client._dual_clip = False
    workflow_client._is_gguf_clip = False
    workflow_client._clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
    workflow_client._clip_name2 = None
    workflow_client._clip_type = "longcat_image"
    workflow_client._vae_name = "ae.safetensors"
    workflow_client._is_klein_9b = False
    workflow_client._is_klein_4b = False
    captured = {}

    class _Upload:
        status_code = 200
        text = ""

        def json(self):
            return {}

    def _post(url, **kwargs):
        if str(url).endswith("/prompt"):
            captured["wf"] = kwargs["json"]["prompt"]
        return _Upload()

    def _run_workflow(reference, keep_source, no_kontext=True):
        captured.clear()
        with (patch("roop.img_editor.flux_edit_comfy_client.requests.post", side_effect=_post),
              patch("roop.img_editor.flux_edit_comfy_client.get_comfyui_url", return_value="http://127.0.0.1:8188")):
            workflow_client.generate(
                image=Image.new("RGB", (64, 64), (30, 20, 15)),
                prompt="the man must be bald",
                denoise=APPEARANCE_SAMPLE_DENOISE,
                num_inference_steps=APPEARANCE_SAMPLE_STEPS,
                step_cap=APPEARANCE_SAMPLE_STEPS,
                mask_image=Image.new("L", (64, 64), 255),
                keep_source_pixels=keep_source,
                prompt_authority=True,
                preserve_frame=True,
                no_kontext=no_kontext,
                guidance_scale=1.0,
                reference_image=reference,
            )
        return captured["wf"]

    covered_wf = _run_workflow(Image.new("RGB", (64, 64), (190, 150, 120)), True)
    assert covered_wf["6"]["inputs"]["image1"] == ["1r", 0]
    assert covered_wf["5"]["inputs"]["pixels"] == ["1r", 0]
    assert covered_wf["8"]["inputs"]["latent_image"] == ["16", 0]
    assert covered_wf["8"]["inputs"]["denoise"] == APPEARANCE_SAMPLE_DENOISE
    assert covered_wf["8"]["inputs"]["steps"] == APPEARANCE_SAMPLE_STEPS
    plain_wf = _run_workflow(None, True, measured_appearance_sample(True)["no_kontext"])
    assert plain_wf["6"]["inputs"]["image1"] == ["1", 0]
    assert "1r" not in plain_wf
    assert plain_wf["5"]["inputs"]["pixels"] == ["1", 0]
    assert plain_wf["8"]["inputs"]["denoise"] == APPEARANCE_SAMPLE_DENOISE
    assert plain_wf["6"]["inputs"]["vae"] == ["4", 0]
    assert plain_wf["8"]["inputs"]["positive"] == ["11", 0]
    assert plain_wf["11"]["inputs"]["reference_latents_method"] == "index_timestep_zero"
    body_wf = _run_workflow(Image.new("RGB", (64, 64), (190, 150, 120)), False)
    assert "1r" not in body_wf
    print("OK: Qwen usa la vista cubierta y el latente sigue siendo la foto")


if __name__ == "__main__":
    main()
