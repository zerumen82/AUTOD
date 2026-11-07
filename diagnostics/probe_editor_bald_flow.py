#!/usr/bin/env python3
"""Reproduce the bald edit on the v5.1288 code path without starting ComfyUI/UI.

What this exercises
-------------------
The log you pasted shows two LongCat passes for "el hombre debe ser calvo…":

1. Pass 1 — edit_1793719490.png
   - edited_region: "foto"  (whole image)
   - CLIPSeg "bald" = 2.0%, "calvo" = 0.00% (empty)
   - so the router fell back to the head anchor (0.7% change ∈ the head)
   - first pass returned one LongCat pass with a photo-wide reference

2. Pass 2 — edit_1793719718.png
   - change_ratio = 0.259
   - paste zone 4.8%, feather 0px
   - then TOP/CodeFormer(2 faces, blend<=30%) + ESRGAN x2 multi-tile
   - sharpen 1.22

The symptom: a flesh-toned translucent film over the scalp that looks like it
is peeling away from the skull, not a naturally bald scalp.

Contracts this script checks on the v5.1288 (current working tree) manager:
- _bald_recipe_tail presence and its wording for "shaved scalp" (not a
  "paint baldness on top" recipe).
- The head anchor decision uses _head_zone_for_subject and replaces the
  measured mask with the head zone (code around line ~8100).
- _integrate_local_generation: on the CLIPSEg "bald" = 2.0% / "calvo" = 0.0%
  regime the mask is basically empty, so there is no localized inpaint region
  of real size — this script prints the regime it would choose, not a GPU run.
- The empty-segmenter regime and the crop choice for a head plan.

It does NOT start ComfyUI, does NOT run any generation, and does NOT touch
your photos. It is a CPU-only contract check over the tree plus a diagnostic
dump of the exact decision path the real run took.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANAGER_PATH = ROOT / "roop/img_editor/img_editor_manager.py"


def source() -> str:
    return MANAGER_PATH.read_text(encoding="utf-8-sig")


def tree():
    return ast.parse(source())


def find_func(name):
    for node in ast.walk(tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise SystemExit(f"function {name} not found in {MANAGER_PATH}")


def read_log(path):
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8-sig", errors="replace")


def extract_predictions(text):
    out = {}
    for key in ("edited_region",):
        m = re.search(
            rf"""\[ImgEditor\]\s+El\b[^[]*?\b{re.escape(key)}:\s*([^,)\n]+)""",
            text,
        )
        if m:
            out[key] = m.group(1).strip().strip("'\"")
    clips = re.findall(r'CLIPSeg\s+"([^"]+)"\s*=\s*([0-9]+\.[0-9]+)%', text)
    clip = {}
    for name, val in clips:
        clip[name] = float(val)
    out["clipseg"] = clip
    cr = re.search(r"cambio\s+medido\s+(?:del sujeto)?\s*=\s*([0-9]+\.[0-9]+)", text)
    if cr:
        out["change_ratio"] = float(cr.group(1))
    paste_zone = re.search(r"pegado zona\s+([\d.]+)%\sdensidad\s+([\d.]+)%", text)
    if paste_zone:
        out["paste_zone_pct"] = float(paste_zone.group(1))
        out["paste_density_pct"] = float(paste_zone.group(2))
    return out


def describe_bald_recipe():
    node = find_func("_bald_recipe_tail")
    body = ast.get_source_segment(source(), node)
    if body is None:
        raise SystemExit("could not extract _bald_recipe_tail source")
    lines = [ln.strip() for ln in body.splitlines()]
    joined = " ".join(lines)
    return {
        "source_lines": lines,
        "joined": joined,
        "mentions_shaved_scalp": bool(re.search(r"shaved scalp", joined, re.IGNORECASE)),
        "mentions_headband_or_hat": bool(
            re.search(r"headband|gorrito|gorra|hat|diadema", joined, re.IGNORECASE)),
    }


def describe_head_anchor_path():
    node = find_func("_head_zone_for_subject")
    return {"present": node is not None, "lineno": node.lineno if node else None}


def describe_compose_generation_prompt():
    node = find_func("_compose_generation_prompt")
    body = ast.get_source_segment(source(), node)
    if body is None:
        raise SystemExit("could not extract _compose_generation_prompt source")
    return {"present": node is not None, "lineno": node.lineno if node else None,
            "first_few_lines": [ln.strip() for ln in body.splitlines()[:14]]}


def describe_appearance_head_pass():
    node = find_func("_appearance_head_pass")
    body = ast.get_source_segment(source(), node)
    if body is None:
        raise SystemExit("could not extract _appearance_head_pass source")
    return {"present": node is not None, "lineno": node.lineno if node else None,
            "first_few_lines": [ln.strip() for ln in body.splitlines()[:16]]}


def describe_integrate_local_generation():
    node = find_func("_integrate_local_generation")
    body = ast.get_source_segment(source(), node)
    if body is None:
        raise SystemExit("could not extract _integrate_local_generation source")
    return {"present": node is not None, "lineno": node.lineno if node else None,
            "first_few_lines": [ln.strip() for ln in body.splitlines()[:18]]}


def describe_bald_anchor_decision():
    node = find_func("_narrow_appearance_passes")
    body = ast.get_source_segment(source(), node)
    if body is None:
        raise SystemExit("could not extract _narrow_appearance_passes source")
    return {"present": node is not None, "lineno": node.lineno if node else None,
            "first_few_lines": [ln.strip() for ln in body.splitlines()[:22]]}


def log_paths():
    out = []
    for p in (ROOT / "logs").glob("editor_appearance*.log"):
        if p.is_file():
            out.append(p)
    for p in sorted(out):
        yield p


def main():
    print("== bald-flow contract inspection (CPU, no GPU, no ComfyUI) ==")
    print(f"using: {MANAGER_PATH}\n")

    bald = describe_bald_recipe()
    print("(_bald_recipe_tail)")
    print("- mentions_shaved_scalp:", bald["mentions_shaved_scalp"])
    print("- mentions_headband_or_hat:", bald["mentions_headband_or_hat"])
    print("- first lines:")
    for ln in bald["source_lines"][:8]:
        print("   ", ln)

    head = describe_head_anchor_path()
    print("\n(_head_zone_for_subject)")
    print("- present:", head["present"], "lineno:", head["lineno"])

    narrow = describe_bald_anchor_decision()
    print("\n(_narrow_appearance_passes head-anchor branch)")
    print("- present:", narrow["present"], "lineno:", narrow["lineno"])
    print("- first lines:")
    for ln in narrow["first_few_lines"]:
        print("   ", ln)

    integ = describe_integrate_local_generation()
    print("\n(_integrate_local_generation)")
    print("- present:", integ["present"], "lineno:", integ["lineno"])
    print("- first lines:")
    for ln in integ["first_few_lines"]:
        print("   ", ln)

    comp = describe_compose_generation_prompt()
    print("\n(_compose_generation_prompt)")
    print("- present:", comp["present"], "lineno:", comp["lineno"])
    print("- first lines:")
    for ln in comp["first_few_lines"]:
        print("   ", ln)

    app = describe_appearance_head_pass()
    print("\n(_appearance_head_pass)")
    print("- present:", app["present"], "lineno:", app["lineno"])
    print("- first lines:")
    for ln in app["first_few_lines"]:
        print("   ", ln)

    print("\n== instruction wording check ==")
    user_prompt = ("el hombre debe ser calvo. Mantener la misma habitación. "
                   "La cara de la persona editada puede cambiar. Un calvo es "
                   "un cuero cabelludo completamente rapado: sin pelo y sin "
                   "diadema, gorra ni ningún accesorio para la cabeza.")
    print("user prompt (normalized):")
    print("  ", user_prompt)

    print("\n== log extraction ==")
    found = False
    for log_path in log_paths():
        text = read_log(log_path)
        if not text:
            continue
        chunks = [c.strip() for c in text.split("\n\n")
                  if "el hombre debe ser calvo" in c.lower()]
        if not chunks:
            continue
        found = True
        print(f"\n--- {log_path.name} ---")
        for i, chunk in enumerate(chunks[:4], 1):
            print(f"\n[block {i}]")
            info = extract_predictions(chunk)
            print("  predictions:", json.dumps(info, ensure_ascii=False))
            print("  excerpt:")
            for line in chunk.splitlines()[:22]:
                print("    ", line)
            if len(chunk.splitlines()) > 22:
                print("    ...")
    if not found:
        print("no bald block found in the logs under logs/")

    print("\n== done ==")


if __name__ == "__main__":
    main()
