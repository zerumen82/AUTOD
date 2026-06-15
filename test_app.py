import sys, os, threading, time, json
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

os.environ['GRADIO_SERVER_PORT'] = '17865'
os.environ['COMFYUI_PORT'] = '8189'  # Different port to avoid conflict

# Import and run the app
import roop.core

# Patch to skip webview
import ui.main as ui_main
original_main = ui_main.main
def patched_main():
    # Same as original but skip webview/pywebview
    from ui.globals import start_capturing_prints
    start_capturing_prints()
    import gradio as gr
    demo = original_main()
    # Skip webview, just run
    demo.launch(server_name='127.0.0.1', server_port=17865, quiet=True, prevent_thread_lock=True)

ui_main.main = patched_main

# Run in thread
def run_app():
    roop.core.run()

t = threading.Thread(target=run_app, daemon=True)
t.start()

# Wait for server
import urllib.request
for i in range(60):
    time.sleep(2)
    try:
        resp = urllib.request.urlopen('http://127.0.0.1:17865', timeout=3)
        if resp.status == 200:
            print(f'Gradio ready after {i*2}s')
            break
    except:
        pass
else:
    print('Gradio did not start')
    os._exit(1)

# Check if debug_console.log was created
time.sleep(2)
if os.path.exists(r'D:\PROJECTS\AUTOAUTO\debug_console.log'):
    with open(r'D:\PROJECTS\AUTOAUTO\debug_console.log') as f:
        print('LOG FOUND at startup:')
        print(f.read())
else:
    print('No log at startup (expected, no source loaded yet)')

# Use Gradio client to upload a file
try:
    from gradio_client import Client, handle_file
    client = Client('http://127.0.0.1:17865')
    print('Client connected')
    
    # Check available APIs
    api_info = client.view_api(all_endpoints=True)
    print(f'API info: {api_info}')
except Exception as e:
    print(f'Client error: {e}')
    import traceback
    traceback.print_exc()

print('DONE')
os._exit(0)
