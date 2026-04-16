import shutil
import os

src = r"C:\Users\INdaHouse\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib"
dst = r"D:\PROJECTS\AUTOAUTO\venv_flux\Lib\site-packages\torch\lib"

for f in os.listdir(src):
    if f.endswith('.dll'):
        src_file = os.path.join(src, f)
        dst_file = os.path.join(dst, f)
        try:
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {f}")
        except Exception as e:
            print(f"Error {f}: {e}")

print("Done!")