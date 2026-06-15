import sys, os, threading, time, json, glob, urllib.request, requests
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

sys.modules['webview'] = type(sys)('webview')
sys.modules['webview'].create_window = lambda *a,**kw: None
sys.modules['webview'].start = lambda *a,**kw: None

import ui.tabs.comfy_launcher as cl
cl.start = lambda *a,**kw: (False, 'mock', 8188)

def app():
    import run
    run.setup_runtime()
    import roop.core
    roop.core.run()

t = threading.Thread(target=app, daemon=True)
t.start()

for i in range(60):
    time.sleep(2)
    try:
        r = urllib.request.urlopen('http://127.0.0.1:7861', timeout=3)
        print(f'READY on 7861 after {i*2}s', flush=True)
        break
    except:
        pass
else:
    print('TIMEOUT', flush=True); os._exit(1)

time.sleep(3)
if os.path.exists('debug_console.log'):
    print('PRE-LOG:', flush=True)
    with open('debug_console.log') as f: print(f.read(), flush=True)

test_img = None
for ext in ['jpg','jpeg','png']:
    found = glob.glob(f'.autodeep_temp/**/*.{ext}', recursive=True)
    found = [f for f in found if os.path.getsize(f) > 1000]
    if found:
        test_img = found[0]
        ext_used = ext
        break

if test_img:
    print(f'Upload: {test_img}', flush=True)
    r = requests.post('http://127.0.0.1:7861/upload', files={'files': (os.path.basename(test_img), open(test_img,'rb'), f'image/{ext_used}')}, timeout=30)
    print(f'Upload: {r.status_code} {r.text[:200]}', flush=True)
    time.sleep(5)

if os.path.exists('debug_console.log'):
    print('POST-LOG:', flush=True)
    with open('debug_console.log') as f: print(f.read(), flush=True)
else:
    print('NO LOG CREATED', flush=True)

os._exit(0)
