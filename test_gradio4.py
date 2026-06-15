import sys, os, threading, time
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import gradio as gr
from ui.globals import _write_console as _log

log_file = r'D:\PROJECTS\AUTOAUTO\debug_console3.log'
with open(log_file, 'w') as f:
    f.write('starting\n')

def on_files(files):
    _log('[HANDLER] inside Gradio event')
    with open(log_file, 'a') as f:
        f.write('handler ran\n')
    return 'ok'

with gr.Blocks() as demo:
    fi = gr.Files(label='Test Files', file_count='multiple')
    out = gr.Textbox()
    fi.change(fn=on_files, inputs=fi, outputs=out)

# Show API info BEFORE launch
demo.config
demo.api_open = True
demo.show_api = True

demo.launch(server_name='127.0.0.1', server_port=17864, quiet=True, prevent_thread_lock=True)
