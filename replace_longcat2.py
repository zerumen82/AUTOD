p = "roop/img_editor/img_editor_manager.py"
s = open(p, encoding="utf-8").read()
old = 'version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"\n'
new = 'version = get_longcat_model_config()["model_file"]\n'
count = s.count(old)
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("replaced", count)
