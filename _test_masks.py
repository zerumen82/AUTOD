import sys, os
sys.path.insert(0, "D:/PROJECTS/AUTOAUTO")
os.environ["AUTOAUTO_OFFLINE"] = "1"

from PIL import Image
from roop.img_editor.img_editor_manager import ImgEditorManager
from roop.img_editor.nlp.semantic_analyzer import LightLocalIntentAnalyzer

mgr = ImgEditorManager()
an = LightLocalIntentAnalyzer()

img = Image.open("D:/PROJECTS/AUTOAUTO/_test_person.png").convert("RGB")
print(f"Image size: {img.size}")

# 1. Test remove mask
remove_mask = mgr._build_person_removal_mask(img)
if remove_mask:
    arr = list(remove_mask.getdata())
    white = sum(1 for p in arr if p > 128)
    total = len(arr)
    print(f"Remove mask: {white/total*100:.1f}% white ({white}/{total})")
else:
    print("Remove mask: None")

# 2. Test add mask
add_mask = mgr._build_person_addition_mask(img)
if add_mask:
    arr = list(add_mask.getdata())
    white = sum(1 for p in arr if p > 128)
    total = len(arr)
    print(f"Add mask: {white/total*100:.1f}% white ({white}/{total})")
else:
    print("Add mask: None")

# 3. Test semantic analyzer
prompts = [
    "remove the person from this photo",
    "erase the person in the scene",
    "add another person to the picture",
    "add a man next to the woman",
    "make the image brighter",
]
for p in prompts:
    axes = an.get_axis_scores(p)
    bias = an.get_structural_bias(p)
    mag = an.get_magnitude(p)
    dom = an.is_structural_dominant(p)
    print(f"  [{bias:>7}] mag={mag:.2f} dom={dom} struct={axes['structural']:.3f} | {p}")