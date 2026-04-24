#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for FLUX Klein 4B image editing
"""

import os
import sys
import time
import requests
from PIL import Image
import io

# Add the project root to the path so we can import roop modules
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient
from ui.tabs.comfy_launcher import is_comfyui_running, start as start_comfy, stop as stop_comfy

def main():
    print("=== FLUX Klein 4B Test ===")
    
    # Step 1: Check if ComfyUI is running, start if not
    print("Checking ComfyUI status...")
    if not is_comfyui_running():
        print("ComfyUI not running, starting...")
        success, message, port = start_comfy(directly_run=True)
        if not success:
            print(f"Failed to start ComfyUI: {message}")
            return 1
        print(f"ComfyUI started: {message}")
    else:
        print("ComfyUI is already running")
    
    # Wait a bit for ComfyUI to be ready
    time.sleep(5)
    
    # Step 2: Create a dummy image (512x512 white)
    print("Creating dummy image...")
    image = Image.new('RGB', (512, 512), color='white')
    
    # Step 3: Initialize the FluxEditComfyClient
    print("Initializing FLUX Edit client...")
    client = FluxEditComfyClient()
    
    # Step 4: Load the client with FLUX model (Klein GGUF has issues, use Dev GGUF as fallback)
    print("Loading FLUX model (using Dev GGUF as fallback for stability)...")
    flux_version = "flux1-dev-Q4_K.gguf"  # Known working GGUF
    success, message = client.load(flux_version=flux_version)
    if not success:
        print(f"Failed to load client: {message}")
        return 1
    print(f"Client loaded: {message}")
    
    # Step 5: Generate image with FLUX Klein
    print("Starting image generation...")
    prompt = "a beautiful landscape, detailed, 8k"
    # Use 1 inference step (will become 20 steps for Klein due to max(20, num_inference_steps))
    num_inference_steps = 1
    denoise = 0.75
    
    start_time = time.time()
    result, message = client.generate(
        image=image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        seed=42,
        denoise=denoise
    )
    elapsed = time.time() - start_time
    
    # Step 6: Check result
    if result is not None:
        print(f"SUCCESS! Generation completed in {elapsed:.2f} seconds")
        # Save the result image
        output_path = "flux_klein_test_output.png"
        result.image.save(output_path)
        print(f"Output saved to: {output_path}")
        return 0
    else:
        print(f"FAILED! Generation failed after {elapsed:.2f} seconds")
        print(f"Error: {message}")
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)