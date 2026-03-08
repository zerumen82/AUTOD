#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal test script for Gradio
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from PIL import Image
from io import BytesIO

def test_gradio():
    """Test Gradio basic functionality"""
    print("Testing Gradio...")
    
    # Create a simple interface
    def greet(name):
        return f"Hello, {name}!"
    
    with gr.Blocks() as ui:
        name = gr.Textbox(label="Name")
        output = gr.Textbox(label="Output")
        name.change(greet, inputs=[name], outputs=[output])
    
    try:
        print("Launching Gradio interface...")
        ui.launch(
            server_name="127.0.0.1",
            server_port=9002,
            show_error=True,
            quiet=False,
            share=False
        )
    except Exception as e:
        print(f"Error launching Gradio: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gradio()
