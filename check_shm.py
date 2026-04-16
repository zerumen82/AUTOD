import os
import hashlib

# Get shm.dll files
files = [
    r"C:\Users\INdaHouse\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\lib\shm.dll",
    r"D:\PROJECTS\AUTOAUTO\venv\lib\site-packages\torch\lib\shm.dll",
    r"D:\tmp\torch_extracted\torch\lib\shm.dll"
]

for f in files:
    if os.path.exists(f):
        with open(f, "rb") as fp:
            data = fp.read()
            md5 = hashlib.md5(data).hexdigest()
        print(f"{f}: {len(data)} bytes, MD5: {md5}")
