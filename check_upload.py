import sys, os
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import gradio as gr

# Check the upload route in Gradio
import inspect
from gradio import routes
src = inspect.getsource(routes)
for line in src.split(chr(10)):
    if 'upload' in line.lower() and ('def ' in line or '@' in line or 'route' in line.lower()):
        print(line.strip()[:200])
