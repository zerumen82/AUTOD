import sys, os
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

import gradio as gr

# Check if gr.Files exists
print('gr.Files exists:', hasattr(gr, 'Files'))

# Check gr.File for multiple
if hasattr(gr, 'File'):
    print('gr.File exists')
    import inspect
    sig = inspect.signature(gr.File.__init__)
    print(f'File.__init__ params: {list(sig.parameters.keys())}')
