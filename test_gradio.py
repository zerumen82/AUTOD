import sys, os, threading, time, json
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import gradio as gr

# Test _log directly
with open(r'D:\PROJECTS\AUTOAUTO\debug_console.log', 'w') as f:
    f.write('test started\n')

from ui.globals import _write_console as _log
_log('[TEST1] Before Gradio')

log_file = r'D:\PROJECTS\AUTOAUTO\debug_console.log'

def on_files(files):
    _log(f'[TEST2] on_files called: {len(files)} files')
    with open(log_file, 'a') as f:
        f.write(f'handler called with {len(files)} files\n')
    return 'ok'

demo = gr.Blocks()
with demo:
    fi = gr.Files(label='Test Files', file_count='multiple')
    out = gr.Textbox()
    fi.change(fn=on_files, inputs=fi, outputs=out)

def run():
    demo.launch(server_name='127.0.0.1', server_port=17861, quiet=True)

t = threading.Thread(target=run, daemon=True)
t.start()
print('Server started')
time.sleep(10)

# Try a simple text API call
import urllib.request
url = 'http://127.0.0.1:17861/gradio_api/call/file_input'
data = '{"data": [[]]}'.encode()
try:
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    resp = urllib.request.urlopen(req, timeout=10)
    print(f'API response: {resp.status}')
    print(f'Body: {resp.read().decode()}')
except Exception as e:
    print(f'API error: {e}')

# Also try the upload endpoint
try:
    import requests
    with open(r'D:\PROJECTS\AUTOAUTO\assets\icon.ico', 'rb') as fp:
        r = requests.post('http://127.0.0.1:17861/upload', files={'files': ('icon.ico', fp, 'image/x-icon')}, timeout=10)
        print(f'Upload: {r.status_code}')
except Exception as e:
    print(f'Upload error: {e}')

time.sleep(5)

with open(log_file) as f:
    print('LOG CONTENTS:')
    print(f.read())

os._exit(0)
