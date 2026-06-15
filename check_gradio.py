import sys, os
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import gradio as gr
print('Gradio version:', gr.__version__)

# Check if launch() with quiet=True messes with stdout
from gradio import utils
import inspect
# Check what quiet=True does
if hasattr(gr, '__version__'):
    ver = gr.__version__
    print(f'Version: {ver}')
    
# Check launch() signature
sig = inspect.signature(gr.Blocks.launch)
print(f'launch params: {list(sig.parameters.keys())}')
