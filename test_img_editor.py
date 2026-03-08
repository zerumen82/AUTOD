#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for the Image Editor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from roop.img_editor.img_editor_manager import get_img_editor_manager

def test_img_editor():
    """Test the Image Editor"""
    print("=== TESTING IMAGE EDITOR ===")
    
    manager = get_img_editor_manager()
    
    print(f"ComfyUI available: {manager.is_comfy_available()}")
    
    if manager.is_comfy_available():
        print("ComfyUI is available")
        print(f"Current model: {manager.get_current_model()}")
    else:
        print("ComfyUI is not available. Please start ComfyUI first.")
    
    print("=== TEST COMPLETED ===")

if __name__ == "__main__":
    test_img_editor()
