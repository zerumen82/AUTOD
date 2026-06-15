import sys, os, threading, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

# The problem: event not firing. Let's check if the Gradio server's file upload API works.
# Start the app, wait for it, then use gradio_client to simulate file upload

# Mock pywebview to prevent crash
sys.modules['webview'] = type(sys)('webview')
sys.modules['webview'].create_window = lambda *a, **kw: None
sys.modules['webview'].start = lambda *a, **kw: None

import ui.tabs.comfy_launcher as cl
cl.start = lambda *a, **kw: (False, 'mock', 8188)

# Patch the _set_window_icon to not crash
import roop.core as rc
rc._set_window_icon = lambda uid: None

# Run in thread
def app():
    import run
    run.setup_runtime()
    # Also patch core.run to crash less
    import roop.core
    original_run = roop.core.run
    def patched_run():
        original_run()
    roop.core.run = patched_run
    import roop.core
    roop.core.run()

t = threading.Thread(target=app, daemon=True)
t.start()

# Wait for Gradio
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
            if 'python' in proc.info['name'].lower():
                cl = ' '.join(proc.info['cmdline'][:3]) if proc.info['cmdline'] else ''
                if 'run.py' in cl or 'ui' in cl.lower():
                    print(f'Python proc: {proc.info[\"pid\"]} {cl[:200]}', flush=True)
        except: pass
    os._exit(1)

time.sleep(2)

# Check debug file
if os.path.exists('debug_console.log'):
    print('DEBUG LOG exists before upload!', flush=True)
else:
    print('No debug log before upload', flush=True)

# Try using gr.Client to interact
try:
    from gradio_client import Client
    client = Client('http://127.0.0.1:7861', quiet=True)
    print('Client API:', client.view_api(return_format='dict') if hasattr(client, 'view_api') else 'no view_api', flush=True)
    
    # Get all endpoints
    endpoints = client.view_api(all_endpoints=True, print_info=False, return_format='dict')
    print(f'Endpoints: {list(endpoints.keys()) if isinstance(endpoints, dict) else type(endpoints)}', flush=True)
except Exception as e:
    print(f'Client error: {e}', flush=True)

# Check debug log again
time.sleep(5)
if os.path.exists('debug_console.log'):
    print('FINAL DEBUG LOG:', flush=True)
    with open('debug_console.log') as f: print(f.read(), flush=True)
else:
    print('FINAL NO LOG', flush=True)

os._exit(0)
