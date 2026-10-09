"""Check that descriptive pose and angle do not trigger scene regeneration."""
from roop.img_editor.img_editor_manager import ImgEditorManager
from roop.img_editor.prompt_analyzer import PromptAnalyzer


def main():
    analyzer = PromptAnalyzer()
    cases = (
        ("EL HOMBRE DE RODILLAS DEBE SER CALVO", False, False),
        ("The man on his knees must be bald", False, False),
        ("EL HOMBRE DEBE ESTAR DE RODILLAS", True, False),
        ("The man must be kneeling", True, False),
        ("LA MUJER DE PERFIL DEBE LLEVAR UN SOMBRERO", False, False),
        ("LA MUJER DEBE ESTAR DE PERFIL", False, True),
        ("PON AL HOMBRE DE RODILLAS", True, False),
        ("la mujer de rodillas va descalza y sin medias, el hombre de rodillas debe ser calvo", False, False),
        ("The woman on her knees goes barefoot and without stockings, the man on his knees is bald", False, False),
        ("The woman is kneeling, the man is bald", True, False),
        ("the woman goes barefoot and without stockings, the white man who is on his knees is bald", False, False),
        ("the man who is standing must be kneeling", True, False),
    )
    for instruction, want_pose, want_angle in cases:
        scope = ImgEditorManager._requested_change_scope(instruction)
        pose = bool(analyzer._pose_pattern.findall(scope) or analyzer.subset_hits(scope, analyzer.POSE_KEYWORDS))
        angle = bool(analyzer._angle_pattern.findall(scope) or analyzer.subset_hits(scope, analyzer.ANGLE_KEYWORDS))
        assert (pose, angle) == (want_pose, want_angle), (instruction, scope, pose, angle)
        print(f"{instruction} -> {scope} | pose={pose} angle={angle}")
    prompt = "The woman on her knees goes barefoot and without stockings, the man on his knees is bald"
    aligned = ImgEditorManager._align_compound_regions(
        prompt, ["woman on her knees", "man on his knees"], "woman"
    )
    assert len(aligned) == 2 and [region for _, region in aligned] == ["woman", "man"], aligned
    print(f"compound regions -> {aligned}")
    llm_regions = [
        "woman on her knees goes barefoot and without stockings",
        "man on his knees must be bald",
    ]
    aligned = ImgEditorManager._align_compound_regions(
        "the woman on her knees goes barefoot and without stockings, the man on his knees must be bald",
        llm_regions, "woman",
    )
    assert [region for _, region in aligned] == llm_regions, aligned
    print(f"LLM sentence regions -> {aligned}")
    assert ImgEditorManager._visual_subject_description(
        "the woman on her knees goes barefoot and without stockings"
    ) == "the woman on her knees"
    assert ImgEditorManager._visual_subject_description(
        "the man on his knees must be bald"
    ) == "the man on his knees"
    assert ImgEditorManager._visual_subject_description(
        "the white man who is on his knees is bald"
    ) == "the white man who is on his knees"
    assert ImgEditorManager._visual_subject_description(
        "la mujer tiene el pelo negro"
    ) == "la mujer"
    assert ImgEditorManager._visual_subject_description(
        "the woman has the black hair"
    ) == "the woman"
    assert ImgEditorManager._requested_change_scope(
        "la mujer tiene el pelo negro"
    ) == "el pelo negro"
    assert ImgEditorManager._requested_change_scope(
        "the woman has the black hair"
    ) == "the black hair"
    assert ImgEditorManager._requested_change_scope(
        "la mujer tiene que ir sin medias"
    ) == "ir sin medias"
    assert ImgEditorManager._visual_subject_description(
        "the woman has to leave"
    ) == "the woman"
    assert ImgEditorManager._requested_change_scope(
        "the woman has to leave"
    ) == "leave"


if __name__ == "__main__":
    main()
