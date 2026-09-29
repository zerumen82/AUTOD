#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from PIL import Image

# Add the project root to the path so we can import roop modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roop.img_editor.img_editor_manager import get_img_editor_manager

def main():
    print("=== Testing Image Editor with prompt: 'debe ir desnuda...' ===")
    
    # Load test image
    test_image_path = "testdata/test_person.jpg"
    if not os.path.exists(test_image_path):
        print(f"ERROR: Test image not found at {test_image_path}")
        return 1
    
    print(f"Loading test image: {test_image_path}")
    input_image = Image.open(test_image_path)
    print(f"Image size: {input_image.size}")
    
    # Get the img_editor manager
    print("Initializing Image Editor Manager...")
    manager = get_img_editor_manager()
    
    # Process the image with the specified prompt
    prompt = "debe ir desnuda..."
    print(f"Processing with prompt: '{prompt}'")
    
    try:
        # Call the generate_intelligent method
        result_image, message, _ = manager.generate_intelligent(
            image=input_image,
            prompt=prompt,
            engine="imagine",  # Using the default imagine engine (LongCat Turbo)
            # We'll let the manager auto-detect other parameters
        )
        
        if result_image is None:
            print(f"ERROR: Processing failed - {message}")
            return 1
            
        print(f"SUCCESS: {message}")
        
        # Save the result
        output_path = "test_output_debe_ir_desnuda.png"
        result_image.save(output_path)
        print(f"Result saved to: {output_path}")
        print(f"Result size: {result_image.size}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: Exception occurred during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
