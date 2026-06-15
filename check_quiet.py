import sys, os
sys.path.insert(0, r'D:\PROJECTS\AUTOAUTO')
os.chdir(r'D:\PROJECTS\AUTOAUTO')

# Check what quiet=True does in Gradio 6.13
import gradio as gr

# Find the Blocks.launch source
import inspect
src = inspect.getsource(gr.Blocks.launch)

# Look for quiet-related code
for i, line in enumerate(src.split(chr(10))):
    if 'quiet' in line.lower():
        # Show context
        lines = src.split(chr(10))
        idx = i
        start = max(0, idx - 2)
        end = min(len(lines), idx + 3)
        for j in range(start, end):
            marker = '>' if j == idx else ' '
            print(f'{marker} {j}: {lines[j][:200]}')
        print('---')
