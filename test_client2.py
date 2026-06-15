import sys, os, threading, time
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
    original_run = roop.core.run
    roop.core.run = lambda: original_run()
    roop.core.run()

t = threading.Thread(target=app, daemon=True)
t.start()

import urllib.request
for i in range(120):
    time.sleep(2)
    try:
        urllib.request.urlopen('http://127.0.0.1:7861', timeout=3)
        print(f'Gradio on 7861 after {i*2}s', flush=True)
        break
    except:
        pass
else:
    print('TIMEOUT', flush=True)
    import psutil
    for proc in psutil.process_iter(['pid','name','cmdline']):
        try:
            info = proc.info
            if info.get('name') and 'python' in info['name'].lower():
                cl = ' '.join(info['cmdline'][:3]) if info.get('cmdline') else ''
                if 'run.py' in cl or 'ui' in cl.lower():
                    pid = info['pid']
                    print(f'Python proc: {pid} {cl[:200]}', flush=True)
        except:
            pass
    os._exit(1)

time.sleep(2)

if os.path.exists('debug_console.log'):
    print('DEBUG LOG exists pre-upload!', flush=True)
    with open('debug_console.log') as f: print(f.read()[:500], flush=True)
else:
    print('No debug log pre-upload', flush=True)

try:
    from gradio_client import Client
    client = Client('http://127.0.0.1:7861', quiet=True)
    endpoints = client.view_api(all_endpoints=True, print_info=False, return_format='dict')
    keys = list(endpoints.keys()) if isinstance(endpoints, dict) else 'not dict'
    print(f'Endpoints: {keys}', flush=True)
except Exception as e:
    print(f'Client error: {e}', flush=True)

time.sleep(5)
if os.path.exists('debug_console.log'):
    print('FINAL LOG:', flush=True)
    with open('debug_console.log') as f: print(f.read()[:500], flush=True)
else:
    print('FINAL NO LOG', flush=True)

os._exit(0)
