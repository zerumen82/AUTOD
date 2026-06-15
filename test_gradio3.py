import sys, os, threading, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import gradio as gr
from ui.globals import _write_console as _log

log_file = r'D:\PROJECTS\AUTOAUTO\debug_console2.log'
with open(log_file, 'w') as f:
    f.write('starting\n')

def on_files(files):
    _log('[HANDLER] on_files called via Gradio')
    with open(log_file, 'a') as f:
        f.write(f'handler executed with {len(files)} files\n')
    return 'ok'

with gr.Blocks() as demo:
    fi = gr.Files(label='Test Files', file_count='multiple')
    out = gr.Textbox()
    fi.change(fn=on_files, inputs=fi, outputs=out)

def run():
    demo.launch(server_name='127.0.0.1', server_port=17863, quiet=True, prevent_thread_lock=True)

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(10)

# Use the correct API format
import json, urllib.request, requests

# First find the API name
api_info = requests.get('http://127.0.0.1:17863/gradio_api/api_info', timeout=10).json()
print('Named endpoints:', list(api_info.get('named_endpoints', {}).keys()))

# Upload a file first
upload_url = 'http://127.0.0.1:17863/upload'
with open(r'D:\PROJECTS\AUTOAUTO\assets\icon.ico', 'rb') as fp:
    upload_resp = requests.post(upload_url, files={'files': ('icon.ico', fp, 'image/x-icon')}, timeout=10)
    print('Upload status:', upload_resp.status_code)
    uploaded = upload_resp.json()
    print('Uploaded files:', uploaded)

time.sleep(3)

with open(log_file) as f:
    print('LOG CONTENTS:')
    print(f.read())

os._exit(0)
