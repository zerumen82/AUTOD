import sys, os, threading, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import gradio as gr
from ui.globals import _write_console as _log

log_file = r'D:\PROJECTS\AUTOAUTO\debug_console2.log'
with open(log_file, 'w') as f:
    f.write('starting\n')

def on_files(files):
    _log('[HANDLER] on_files called in Gradio')
    with open(log_file, 'a') as f:
        f.write(f'handler executed with {len(files)} files\n')
        for fi in files:
            f.write(f'  {fi}\n')
    return 'ok'

with gr.Blocks() as demo:
    fi = gr.Files(label='Test Files', file_count='multiple')
    out = gr.Textbox()
    fi.change(fn=on_files, inputs=fi, outputs=out)

def run():
    demo.launch(server_name='127.0.0.1', server_port=17862, quiet=True, prevent_thread_lock=True)

t = threading.Thread(target=run, daemon=True)
t.start()
time.sleep(8)

# Use gradio_client to trigger the event
from gradio_client import Client, file
client = Client('http://127.0.0.1:17862')
_log('[TEST] Client connected')

# Submit files to the component
result = client.predict(
    [file(r'D:\PROJECTS\AUTOAUTO\assets\icon.ico')],
    api_name='/file_input'
)
_log(f'[TEST] Result: {result}')

time.sleep(3)

with open(log_file) as f:
    print('LOG:')
    print(f.read())

os._exit(0)
