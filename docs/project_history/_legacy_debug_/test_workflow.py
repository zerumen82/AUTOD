
import sys
import traceback
import tempfile
import os
sys.path.insert(0, 'd:\\PROJECTS\\AUTOAUTO')

# Monkey patch get_available_models to return fixed values
def mock_get_available_models(category):
    if category == 'clip':
        return ['gemma-3-12b-it-qat-q4_0-unquantized\\model-00001-of-00005.safetensors']
    return []

import roop.comfy_workflows_fixed
roop.comfy_workflows_fixed.get_available_models = mock_get_available_models

from roop.comfy_workflows_fixed import get_ltxvideo2_workflow

# Create a temporary image file
with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
    temp_image = f.name

try:
    print('Testing LTX Video workflow...')
    print(f'Temp image: {temp_image}')
    
    workflow = get_ltxvideo2_workflow(
        image_filename=temp_image,
        prompt='A cat running in a field',
        seed=42,
        width=320,
        height=192,
        frames=25,
        fps=24,
        strength=0.9
    )
    
    print('OK Workflow generation successful')
    print(f'Number of nodes: {len(workflow)}')
    print('Nodes in workflow:', list(workflow.keys()))
    
    # Verify required nodes exist
    required_nodes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17']
    for node in required_nodes:
        if node in workflow:
            print(f'  OK Node {node}: {workflow[node].get("class_type", "Unknown")}')
        else:
            print(f'  ERROR Node {node} missing')
            
except Exception as e:
    print(f'ERROR: {type(e).__name__}: {e}')
    print('Stack trace:')
    print(traceback.format_exc())
finally:
    if os.path.exists(temp_image):
        os.unlink(temp_image)
        print('Temp image deleted')
