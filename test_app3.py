import sys, os, threading, time, json
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

# Mock pywebview
sys.modules['webview'] = type(sys)('webview')
sys.modules['webview'].create_window = lambda *a,**kw: None
sys.modules['webview'].start = lambda *a,**kw: None

# Mock comfy launcher
import ui.tabs.comfy_launcher as cl
cl.start = lambda *a,**kw: (False, 'mock', 8188)

# Run in thread
def app():
    os.environ.pop('GRADIO_SERVER_PORT', None)
    os.environ.pop('COMFYUI_PORT', None)
    import run
    run.setup_runtime()
    import roop.core
    roop.core.run()

t = threading.Thread(target=app, daemon=True)
t.start()

# Wait for Gradio
import urllib.request
for i in range(120):
    time.sleep(2)
    try:
        r = urllib.request.urlopen('http://127.0.0.1:7861', timeout=3)
        print(f'READY on 7861 after {i*2}s', flush=True)
        break
    except:
        try:
            r = urllib.request.urlopen('http://127.0.0.1:7862', timeout=3)
            print(f'READY on 7862 after {i*2}s', flush=True)
            break
        except:
            pass
else:
    print('TIMEOUT waiting for Gradio', flush=True)

# Check log file
time.sleep(3)
if os.path.exists('debug_console.log'):
    print('PRE-EXISTING LOG:', flush=True)
    with open('debug_console.log') as f: print(f.read(), flush=True)

# Upload file
import requests
test_img = None
for ext in ['jpg','jpeg','png']:
    found = [f for f in glob.glob(f'.autodeep_temp/**/*.{ext}', recursive=True) if os.path.getsize(f) > 1000]
    if found: test_img = found[0]; break
if test_img:
    print(f'Uploading: {test_img}', flush=True)
    with open(test_img, 'rb') as fp:
        r = requests.post('http://127.0.0.1:7861/upload', files={'files': (os.path.basename(test_img), fp, f'image/{ext}')}, timeout=30)
        print(f'Upload status: {r.status_code}', flush=True)

time.sleep(5)

if os.path.exists('debug_console.log'):
    print('AFTER UPLOAD LOG:', flush=True)
    with open('debug_console.log') as f: print(f.read(), flush=True)
else:
    print('NO LOG FILE', flush=True)

os._exit(0)
