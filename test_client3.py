import sys, os, threading, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

# Patch webview BEFORE importing roop.core
import webview as _real_webview
class MockWindow:
    uid = 'mock_uid_12345'

original_create_window = _real_webview.create_window
def mock_create_window(*a, **kw):
    print('[MOCK] webview.create_window called', flush=True)
    return MockWindow()

original_start = _real_webview.start
def mock_start(*a, **kw):
    print('[MOCK] webview.start would block - returning immediately', flush=True)

_real_webview.create_window = mock_create_window
_real_webview.start = mock_start

import ui.tabs.comfy_launcher as cl
cl.start = lambda *a, **kw: (False, 'mock', 8188)

def app():
    import run
    run.setup_runtime()
    import roop.core
    roop.core.run()

t = threading.Thread(target=app, daemon=True)
t.start()

import urllib.request
for i in range(90):
    time.sleep(2)
    try:
        urllib.request.urlopen('http://127.0.0.1:7861', timeout=3)
        print(f'READY on 7861 after {i*2}s', flush=True)
        break
    except:
        pass
else:
    print('TIMEOUT', flush=True)
    os._exit(1)

time.sleep(3)
print('Gradio is running. Getting client...', flush=True)

try:
    from gradio_client import Client
    client = Client('http://127.0.0.1:7861')
    print('Client created', flush=True)
    
    # Get all API endpoints
    api = client.view_api(all_endpoints=True, print_info=False, return_format='dict')
    print(f'API type: {type(api)}', flush=True)
    if isinstance(api, dict):
        for name, info in list(api.items())[:10]:
            print(f'  Endpoint: {name}', flush=True)
    elif isinstance(api, list):
        for item in api[:10]:
            print(f'  Item: {item}', flush=True)
    else:
        print(f'  API: {str(api)[:500]}', flush=True)
except Exception as e:
    print(f'Client error: {e}', flush=True)
    import traceback
    traceback.print_exc()

# Check debug files
time.sleep(3)
for f in ['debug_console.log', os.path.join(os.path.expanduser('~'), 'Desktop', 'debug_autoauto.txt')]:
    if os.path.exists(f):
        with open(f) as fp: print(f'FILE {f}: {fp.read()[:200]}', flush=True)

print('DONE', flush=True)
os._exit(0)
