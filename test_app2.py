import sys, os, threading, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')
os.environ['GRADIO_SERVER_PORT'] = '17866'
os.environ['COMFYUI_PORT'] = '8190'

# Monkey-patch to skip pywebview and ComfyUI
import roop.core
original_run = roop.core.run
def patched_run():
    print('[SYS] AutoAuto starting (test mode)')
    original_run()
roop.core.run = patched_run

# Patch pywebview
import sys as _sys
class MockWebview:
    def create_window(self, *a, **kw): pass
    def start(self, *a, **kw): pass
_sys.modules['webview'] = MockWebview()

# Also patch comfy launcher
import ui.tabs.comfy_launcher as cl
def mock_start(*a, **kw): return (False, 'mock', 8188)
cl.start = mock_start

# Start in thread
def run_app():
    try:
        import run as _run
        _run.setup_runtime()
        import roop.core as rc
        rc.run()
    except Exception as e:
        print(f'App error: {e}')
        import traceback
        traceback.print_exc()

t = threading.Thread(target=run_app, daemon=True)
t.start()

# Wait for Gradio
import urllib.request
for i in range(90):
    time.sleep(2)
    for port in [17866, 7861]:
        try:
            resp = urllib.request.urlopen(f'http://127.0.0.1:{port}', timeout=3)
            if resp.status == 200:
                print(f'Gradio ready on {port} after {i*2}s')
                break
        except:
            continue
    else:
        continue
    break
else:
    print('Gradio did not start')
    os._exit(1)

time.sleep(3)
if os.path.exists(r'D:\PROJECTS\AUTOAUTO\debug_console.log'):
    print('== LOG FILE (preexisting) ==')
    with open(r'D:\PROJECTS\AUTOAUTO\debug_console.log') as f:
        print(f.read())

# Upload a real image
import requests
upload_url = 'http://127.0.0.1:17866/upload'
import glob
images = glob.glob(r'D:\PROJECTS\AUTOAUTO\.autodeep_temp\*\*.jpg')[:1]
if images:
    test_file = images[0]
    print(f'Uploading: {test_file}')
    with open(test_file, 'rb') as fp:
        r = requests.post(upload_url, files={'files': (os.path.basename(test_file), fp, 'image/jpeg')}, timeout=30)
        print(f'Upload: {r.status_code}')
        if r.status_code == 200:
            uploaded = r.json()
            print(f'Files uploaded: {uploaded}')
else:
    print('No test images found')
    # Use assets/icon.ico
    with open(r'D:\PROJECTS\AUTOAUTO\assets\icon.ico', 'rb') as fp:
        r = requests.post(upload_url, files={'files': ('icon.ico', fp, 'image/x-icon')}, timeout=30)
        print(f'Upload icon: {r.status_code}')

time.sleep(10)

if os.path.exists(r'D:\PROJECTS\AUTOAUTO\debug_console.log'):
    print('== LOG FILE (after upload) ==')
    with open(r'D:\PROJECTS\AUTOAUTO\debug_console.log') as f:
        print(f.read())
else:
    print('== NO LOG FILE ==')

os._exit(0)
