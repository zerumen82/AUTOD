#!/usr/bin/env python3
from PIL import Image
from roop.img_editor.image_quality_pipeline import analyze_quality_plan

paths = [
    "img_editor/edit_1782368164.png",
    "img_editor/edit_1782369536.png",
    "debug_test/test_result_same.jpg",
    "debug_test/structural_e2e_result.png",
]

for p in paths:
    try:
        img = Image.open(p)
        print(f"\n=== {p} {img.size} ===")
        for tier in ("hd", "4k", "8k"):
            plan = analyze_quality_plan(img, tier=tier)
            s = plan["scores"]
            pr = plan["profile"]
            print(
                f"  {tier}: art={s['artifact']} nit={s['sharpness']} ruido={s['noise']} "
                f"soft={s['soft_need']} up={s['upscale_need']}"
            )
            print(
                f"       tiles_cap={pr['tile_blend_cap']:.2f} skip_tiles={pr['skip_tiles']} "
                f"skip_dep={pr['skip_depixelize']} denoise={pr['denoise_strength']} "
                f"tex={pr['texture_blend']:.2f} sharpen={pr['sharpen']:.2f} "
                f"detail={pr['detail_boost_blend']:.2f} face_cap={pr['face_enhance_cap']:.2f}"
            )
    except Exception as e:
        print(p, e)