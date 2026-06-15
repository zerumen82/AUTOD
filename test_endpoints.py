import sys, os, threading, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import webview as _real_webview
class MockWindow:
    uid = 'mock_uid_12345'
_real_webview.create_window = lambda *a, **kw: MockWindow()
_real_webview.start = lambda *a, **kw: None

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
    print('TIMEOUT', flush=True); os._exit(1)

time.sleep(5)

from gradio_client import Client
client = Client('http://127.0.0.1:7861')

api = client.view_api(all_endpoints=True, print_info=False, return_format='dict')
named = api.get('named_endpoints', {})
unnamed = api.get('unnamed_endpoints', {})
print(f'Named endpoints: {len(named)}', flush=True)
print(f'Unnamed endpoints: {len(unnamed)}', flush=True)

for name, info in list(named.items())[:30]:
    params = info.get('parameters', [])
    param_names = [p.get('label', p.get('name', '?')) for p in params]
    print(f'  {name}: params={param_names}', flush=True)

print('---', flush=True)
os._exit(0)
