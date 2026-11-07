p = "roop/img_editor/img_editor_manager.py"
s = open(p, encoding="utf-8").read()
old = 'loaded, load_msg = fast_client.load("LongCat-Image-Edit-Turbo-Q4_K_S.gguf")\n'
new = 'loaded, load_msg = fast_client.load(get_longcat_model_config()["model_file"])\n'
count = s.count(old)
s = s.replace(old, new)
open(p, "w", encoding="utf-8").write(s)
print("replaced", count)
